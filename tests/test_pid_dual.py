"""Two-axis coordination using fake motors only, never UART or real motion."""
from contextlib import nullcontext, redirect_stdout, redirect_stderr
import copy
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from control.runtime import instance_lock
from pid_tuner.dual import DualTuner
from pid_tuner.server import main
from test_pid_http import CaptureHandler
from test_pid_service import RecordingDevice

ROOT=Path(__file__).resolve().parents[1]


class DualTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.config=json.loads((ROOT/'config/robot.json').read_text())
        self.created={}
        def factory(connection,logger):
            device=RecordingDevice(connection,logger)
            self.created[connection['axis']]=device
            return device
        self.tuner=DualTuner(self.config,{'runtime_dir':self.temp.name},
            mode='real',device_factory=factory,start_worker=False)

    def tearDown(self):
        self.tuner.close()
        self.temp.cleanup()

    def targets(self):
        return {'upper':{'target_deg':170.,'speed_deg_s':2.},
                'lower':{'target_deg':46.,'speed_deg_s':3.}}

    def moves(self):
        return {axis:[c for c in dev.calls if c[0]=='move'] for axis,dev in self.created.items()}

    def test_construction_does_not_open_and_shared_lock_survives_one_disconnect(self):
        self.assertEqual(self.created,{})
        self.tuner.connect_all()
        self.assertEqual({a:d.connection['motor_id'] for a,d in self.created.items()}, {'upper':1,'lower':2})
        with self.assertRaisesRegex(RuntimeError,'runtime_already_running'):
            with instance_lock(self.temp.name):pass
        self.tuner.disconnect_axis('upper')
        self.assertTrue(self.tuner.select('lower').connected)
        with self.assertRaisesRegex(RuntimeError,'runtime_already_running'):
            with instance_lock(self.temp.name):pass
        self.tuner.disconnect_axis('lower')
        with instance_lock(self.temp.name):pass

    def test_commands_overlap_on_two_ports_without_waiting_for_first_ack(self):
        self.tuner.connect_all()
        barrier=threading.Barrier(2)
        for dev in self.created.values():
            original=dev.move
            def move(*args,original=original):
                barrier.wait(timeout=2.)
                original(*args)
            dev.move=move
        self.tuner.move_both(self.targets())
        self.assertEqual(self.moves()['upper'][0][1:3], (170.,2.))
        self.assertEqual(self.moves()['lower'][0][1:3], (46.,3.))
        self.assertEqual(self.tuner.group['result'],'running')
        self.assertTrue(all(t.motion['active'] for t in self.tuner.axes.values()))

    def test_invalid_second_axis_never_sends_either_move(self):
        self.tuner.connect_all()
        targets=self.targets();targets['lower']['target_deg']=91.
        with self.assertRaises(RuntimeError):self.tuner.move_both(targets)
        self.assertTrue(all(not calls for calls in self.moves().values()))
        self.assertFalse(self.tuner.group['active'])
        self.assertTrue(all(any(c[0]=='stop' for c in d.calls) for d in self.created.values()))

    def test_a6_negative_target_is_rejected_before_either_axis_dispatches(self):
        self.tuner.connect_all()
        self.created['lower'].angle=self.created['lower'].target=0.
        targets=self.targets();targets['lower']['target_deg']=-.5
        with self.assertRaisesRegex(RuntimeError,'A6单圈范围'):
            self.tuner.move_both(targets)
        self.assertTrue(all(not calls for calls in self.moves().values()))

    def test_requested_stop_is_not_reclassified_as_group_failure_while_waiting(self):
        self.tuner.connect_all();self.tuner.move_both(self.targets())
        entered=threading.Event();release=threading.Event()
        lower=self.tuner.select('lower');stop=lower.stop
        def paused_stop():
            entered.set();release.wait(3.);stop()
        lower.stop=paused_stop
        worker=threading.Thread(target=lambda:self.tuner.stop_all('手动停止测试'))
        worker.start()
        try:
            self.assertTrue(entered.wait(2.))
            self.assertFalse(self.tuner.group['active'])
            with patch.object(self.tuner,'stop_all') as repeat:
                self.tuner.monitor_once();repeat.assert_not_called()
        finally:
            release.set();worker.join(4.)
        self.assertFalse(worker.is_alive())
        self.assertEqual(self.tuner.group['stop_reason'],'手动停止测试')

    def test_missing_axis_uncertain_pid_and_closed_motor_cannot_start_pair(self):
        self.tuner.connect_all()
        with self.assertRaises(ValueError):self.tuner.move_both({'upper':self.targets()['upper']})
        self.tuner.select('lower').pid_status['speed']='write_unverified'
        with self.assertRaises(RuntimeError):self.tuner.move_both(self.targets())
        self.assertTrue(all(not calls for calls in self.moves().values()))

    def test_one_axis_failed_or_single_stop_stops_the_other_axis(self):
        self.tuner.connect_all();self.tuner.move_both(self.targets())
        self.tuner.select('lower').motion.update(active=False,trial_outcome='failed')
        self.tuner.monitor_once()
        self.assertFalse(self.tuner.group['active'])
        self.assertTrue(all(any(c[0]=='stop' for c in d.calls) for d in self.created.values()))
        self.assertTrue(all(not t.motion['active'] for t in self.tuner.axes.values()))

    def test_selected_axis_stop_during_pair_is_a_dual_stop(self):
        self.tuner.connect_all();self.tuner.move_both(self.targets())
        self.tuner.stop_axis('upper')
        self.assertFalse(self.tuner.group['active'])
        self.assertTrue(all(not t.motion['active'] for t in self.tuner.axes.values()))

    def test_stop_attempts_other_axis_even_when_one_stop_fails(self):
        self.tuner.connect_all()
        self.created['upper'].stop_error=OSError('synthetic stop failure')
        with self.assertRaisesRegex(RuntimeError,'synthetic stop failure'):self.tuner.stop_all()
        self.assertTrue(any(c[0]=='stop' for c in self.created['lower'].calls))
        self.created['upper'].stop_error=None

    def test_stop_cancels_preparation_before_either_wire_move(self):
        self.tuner.connect_all()
        entered=threading.Event();release=threading.Event();errors=[]
        upper=self.tuner.select('upper');persist=upper._persist
        def paused_persist():
            entered.set();release.wait(3.);persist()
        upper._persist=paused_persist
        def move():
            try:self.tuner.move_both(self.targets())
            except Exception as exc:errors.append(exc)
        worker=threading.Thread(target=move);worker.start()
        self.assertTrue(entered.wait(2.))
        stopper=threading.Thread(target=self.tuner.stop_all);stopper.start()
        self.assertTrue(self.tuner._cancel.wait(2.));release.set()
        worker.join(4.);stopper.join(4.)
        self.assertFalse(worker.is_alive());self.assertFalse(stopper.is_alive())
        self.assertTrue(errors)
        self.assertTrue(all(not calls for calls in self.moves().values()))

    def test_connect_failure_rolls_back_read_only_without_stop_or_move(self):
        original=self.tuner.select('lower').factory
        def fail(connection,logger):
            device=original(connection,logger);device.read_error=TimeoutError('PID unavailable');return device
        self.tuner.select('lower').factory=fail
        with self.assertRaisesRegex(RuntimeError,'PID unavailable'):self.tuner.connect_all()
        self.assertFalse(any(t.connected for t in self.tuner.axes.values()))
        self.assertTrue(all(not any(c[0] in ('move','stop','enable') for c in d.calls) for d in self.created.values()))
        with instance_lock(self.temp.name):pass

    def test_dual_heartbeat_and_per_axis_http_routes(self):
        self.tuner.connect_all()
        self.tuner.select('lower')._last_client=0.
        handler=CaptureHandler('/api/dual/state');handler.server.tuner=self.tuner;handler.do_GET()
        self.assertEqual(handler.result[0],200)
        self.assertGreater(self.tuner.select('lower')._last_client,0.)
        self.assertEqual(set(handler.result[1]['axes']),{'upper','lower'})
        handler=CaptureHandler('/api/write?axis=lower',{'loop':'speed','values':{'kp':101,'ki':30,'kd':0}})
        handler.server.tuner=self.tuner;handler.do_POST()
        self.assertEqual(handler.result[0],200)
        self.assertFalse(any(c[0]=='write' for c in self.created['upper'].calls))
        self.assertTrue(any(c[0]=='write' for c in self.created['lower'].calls))
        handler=CaptureHandler('/api/read?axis=invalid',{});handler.server.tuner=self.tuner;handler.do_POST()
        self.assertEqual(handler.result[0],400)
        handler=CaptureHandler('/api/dual/move',{'targets':self.targets()},token='wrong')
        handler.server.tuner=self.tuner;handler.do_POST();self.assertEqual(handler.result[0],401)
        self.assertTrue(all(not calls for calls in self.moves().values()))

    def test_dual_api_exact_schema_and_shell_defaults(self):
        handler=CaptureHandler('/api/dual/move',{'targets':self.targets(),'enable':True})
        handler.server.tuner=self.tuner;handler.do_POST();self.assertEqual(handler.result[0],400)
        with patch('sys.argv',['pid_tuner','--real','--pause-sal']), \
             patch('pid_tuner.server.load_config',return_value=(copy.deepcopy(self.config),{})), \
             patch('pid_tuner.server.DualTuner') as dual, \
             patch('pid_tuner.server.TunerServer') as server, \
             patch('pid_tuner.server.isolate_sal',return_value=nullcontext()) as isolate, \
             redirect_stdout(io.StringIO()):
            server.return_value.serve_forever.side_effect=KeyboardInterrupt
            main()
            isolate.assert_called_once_with(['/dev/ttyS1','/dev/ttyS4'])
            dual.return_value.connect_all.assert_not_called()
            dual.return_value.move_both.assert_not_called()
        with patch('sys.argv',['pid_tuner','--device','/dev/ttyUSB0']), \
             patch('pid_tuner.server.load_config',return_value=(copy.deepcopy(self.config),{})), \
             redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):main()

if __name__=='__main__':unittest.main()
