"""Task-start PID RAM initialization, using local IPC and mocked task services."""
from contextlib import ExitStack, nullcontext, redirect_stdout
import copy
import io
from pathlib import Path
import types
import unittest
from unittest.mock import MagicMock, Mock, patch

from control.config import load_config
from control.motors import HardwareError, MotorPair
from control.protocol import make_frame
from control.task_pid import initialize_task_pid, task_start_pid_settings
from pid_tuner.protocol import decode_parameter
from test_pid_transport import motor_emulator, send_reply
from virtual_drive import config


PID = {"enabled": True,
       "upper": {"angle": {"kp":50,"ki":10,"kd":0}, "speed": {"kp":750,"ki":35,"kd":0}},
       "lower": {"angle": {"kp":65,"ki":10,"kd":0}, "speed": {"kp":600,"ki":35,"kd":0}}}


def hardware():
    return config(task_start_pid=copy.deepcopy(PID))


class TaskPIDSettingsTests(unittest.TestCase):
    def test_production_config_has_requested_values(self):
        cfg,_=load_config(Path(__file__).resolve().parents[1])
        self.assertEqual(task_start_pid_settings(cfg["hardware"]),PID)

    def test_invalid_parameters_rejected_before_motor_creation(self):
        for key,value in (("kp",True),("ki",-1),("kd",65536),("kp",50.0)):
            h=hardware(); h["task_start_pid"]["lower"]["angle"][key]=value
            with self.subTest(key=key,value=value), self.assertRaises(ValueError):
                task_start_pid_settings(h)
        for change in (lambda p:p.pop("upper"),
                       lambda p:p["lower"].pop("speed"),
                       lambda p:p.update(enabled=1),
                       lambda p:p.update(other={})):
            h=hardware();change(h["task_start_pid"])
            with self.assertRaises(ValueError):task_start_pid_settings(h)

    def test_disabled_profile_never_stops_or_creates_device(self):
        pair=Mock();report={}
        with patch('pid_tuner.device.SerialDevice') as device:
            initialize_task_pid(pair,{},report)
        self.assertEqual(report["status"],"disabled")
        pair.halt.assert_not_called();device.assert_not_called()


class TaskPIDTransportTests(unittest.TestCase):
    def run_case(self, alter=None, reversed_axes=False, stop_confirmed=True):
        h=hardware()
        if reversed_axes:h["motors"].reverse()
        pair=MotorPair(h);pair._opened=True
        pair.halt=Mock(return_value={"stopped_confirmed":stop_confirmed})
        report={};received=[]
        with ExitStack() as stack:
            ports=[]
            for motor in pair.motors:
                def respond(req,peer,motor=motor):
                    received.append(req)
                    if alter is not None and alter(req,peer):return
                    send_reply(peer,make_frame(0x42,req.motor_id,bytes((req.payload[0],0))))
                _,port,_,_=stack.enter_context(motor_emulator(respond))
                ports.append(port)
            pair.ports=ports
            error=None
            try:
                with redirect_stdout(io.StringIO()):initialize_task_pid(pair,h,report)
            except BaseException as exc:error=exc
            self.assertFalse(pair._move_lock.locked())
            self.assertTrue(all(port.fd is not None for port in ports))
        return pair,report,received,error

    def test_exact_four_ram_writes_acknowledged_on_existing_ports(self):
        for reverse in (False,True):
            pair,r,frames,error=self.run_case(reversed_axes=reverse)
            self.assertIsNone(error)
            self.assertEqual(r["status"],"acknowledged")
            self.assertFalse(r["runtime_readback_verified"])
            self.assertFalse(r["persistent"])
            self.assertEqual(len(frames),4)
            names={m["id"]:m["name"] for m in pair.motors}
            for f in frames:
                self.assertEqual(f.command,0x42)
                loop={0x96:'angle',0x97:'speed'}[f.payload[0]]
                self.assertEqual(decode_parameter(f.payload[0],f.payload[1:]),PID[names[f.motor_id]][loop])
            self.assertTrue(all(w["ack_ok"] and w["transaction"]["rx_hex"] for w in r["writes"]))
            pair.halt.assert_called_once()

    def test_stop_unconfirmed_writes_nothing(self):
        pair,r,frames,error=self.run_case(stop_confirmed=False)
        self.assertIsInstance(error,HardwareError)
        self.assertEqual(r["status"],"failed")
        self.assertEqual(frames,[])
        self.assertEqual(r["writes"],[])
        self.assertEqual(pair.halt.call_count,2)

    def test_rejected_partial_write_stops_without_remaining_writes_or_rollback(self):
        def reject(req,peer):
            if req.motor_id==1 and req.payload[0]==0x97:
                send_reply(peer,make_frame(0x42,1,b'\x97\x01'));return True
        pair,r,frames,error=self.run_case(alter=reject)
        self.assertIsInstance(error,HardwareError)
        self.assertEqual(error.reason,"task_pid_initialization_failed")
        self.assertEqual(len(frames),2)
        self.assertEqual([w["ack_ok"] for w in r["writes"]],[True,False])
        self.assertEqual(pair.halt.call_count,2)
        self.assertIs(error.details["pid_initialization"],r)

    def test_wrong_motor_parameter_command_and_missing_ack_fail_closed(self):
        for kind in ('motor','parameter','command','timeout'):
            def wrong(req,peer):
                if kind!='timeout':
                    send_reply(peer,make_frame(0x44 if kind=='command' else 0x42,
                        2 if kind=='motor' else req.motor_id,
                        bytes((0x97 if kind=='parameter' else req.payload[0],0))))
                return True
            with self.subTest(kind=kind):
                pair,r,frames,error=self.run_case(alter=wrong)
                self.assertIsInstance(error,HardwareError)
                self.assertEqual(len(frames),1)
                self.assertEqual(r["status"],"failed")
                self.assertEqual(pair.halt.call_count,2)

    def test_keyboard_interrupt_retains_partial_report_and_stops(self):
        pair=MotorPair(hardware());pair._opened=True
        pair.halt=Mock(return_value={"stopped_confirmed":True});report={}
        with patch('pid_tuner.device.SerialDevice') as factory:
            factory.return_value.write_parameter.side_effect=KeyboardInterrupt
            with self.assertRaises(KeyboardInterrupt):initialize_task_pid(pair,hardware(),report)
        self.assertEqual(report["status"],"failed")
        self.assertEqual(len(report["writes"]),1)
        self.assertEqual(pair.halt.call_count,2)
        self.assertFalse(pair._move_lock.locked())


class TaskStartupTests(unittest.TestCase):
    def run_service(self, count, initialization_error=False):
        import main as entry
        events=[];results=[]
        pair=MagicMock();pair.__enter__.return_value=pair
        vision=Mock();vision.close.return_value={"status":"completed"}
        vision.start.side_effect=lambda *a:events.append('camera')
        runner=Mock();runner.run.side_effect=lambda *a:(events.append('run') or {"status":"completed"})
        def initialize(motors,hw,report):
            events.append('pid');report.update(status='failed' if initialization_error else 'acknowledged')
            if initialization_error:raise HardwareError('task_pid_initialization_failed')
        def run():
            execute=runtime.call_args.args[2]
            for i in range(count):results.append(execute({'intent':'push_to_end','color':'蓝'},str(i)))
        cfg={'hardware':hardware(),'vision':{},'poll_interval_s':1.}
        paths={k:'/unused/'+k for k in ('runtime_dir','model_package','video_dir','task_file','result_file')}
        modules={
            'control.language':types.SimpleNamespace(ResidentModel=Mock()),
            'control.planner':types.SimpleNamespace(Planner=Mock()),
            'control.vision':types.SimpleNamespace(Vision=Mock(return_value=vision)),
            'control.motors':types.SimpleNamespace(MotorPair=Mock(return_value=pair)),
            'control.tasks':types.SimpleNamespace(TaskRunner=Mock(return_value=runner)),
            'control.task_pid':types.SimpleNamespace(initialize_task_pid=Mock(side_effect=initialize))}
        with patch.dict('sys.modules',modules), patch.object(entry,'load_config',return_value=(cfg,paths)), \
             patch.object(entry,'instance_lock',return_value=nullcontext()),patch.object(entry,'Inbox'), \
             patch.object(entry,'Runtime') as runtime,patch('sys.argv',['main.py']),redirect_stdout(io.StringIO()):
            runtime.return_value.run.side_effect=run
            self.assertEqual(entry.main(),0)
        return events,results,modules,pair

    def test_every_task_initializes_before_camera_and_runner(self):
        events,results,_,_=self.run_service(2)
        self.assertEqual(events,['pid','camera','run','pid','camera','run'])
        self.assertTrue(all(r['pid_initialization']['status']=='acknowledged' for r in results))

    def test_failed_initialization_never_starts_camera_or_task(self):
        events,results,_,_=self.run_service(1,initialization_error=True)
        self.assertEqual(events,['pid'])
        self.assertEqual(results[0]['status'],'failed')
        self.assertEqual(results[0]['reason'],'task_pid_initialization_failed')
        self.assertEqual(results[0]['pid_initialization']['status'],'failed')

    def test_idle_service_never_opens_ports_or_initializes(self):
        events,results,modules,pair=self.run_service(0)
        self.assertEqual(events,[]);self.assertEqual(results,[])
        modules['control.motors'].MotorPair.assert_not_called()
        modules['control.task_pid'].initialize_task_pid.assert_not_called()


if __name__=='__main__':unittest.main()
