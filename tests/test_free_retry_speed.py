"""Free retry policy, actual A6 speed fields and precontact geometry; no hardware."""
import copy
import json
import math
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from control.config import load_config
from control.motors import HardwareError, MotorPair
from control.planner import Planner
from control.segments import free_retry_phase, phase_profile, segment_settings
from control.tasks import TaskRunner
from virtual_drive import VirtualDrive, config, phase

ROOT = Path(__file__).resolve().parents[1]


def hardware():
    return config(max_speed_deg_s=60., stable_s=.3, goal_hold_s=3.,
                  segmented_motion={'enabled': True, 'free_retry_speed_deg_s': 5.})


def failure(reason='motion_goal_settle_timeout', oscillating=True, name='free'):
    return HardwareError(reason, {'phase': name, 'motion_diagnostics': {
        'execution': {'stage': 'settling', 'endpoint_oscillating_axes': [oscillating, False]}}})


class FreeRetryTests(unittest.TestCase):
    def runner(self, outcomes):
        cfg, _ = load_config(ROOT)
        planner, motors = Mock(), Mock()
        planner.precontact.return_value = [math.radians(168.8), math.radians(20.)]
        planner.free.side_effect = lambda *a, **kw: phase([[169.,20.],[168.8,20.]], times=[0., .02],
            segment_control=phase_profile(segment_settings(cfg['hardware']), 'free'))
        motors.halt.return_value = {'stopped_confirmed': True}
        motors.read_state.return_value = {'q_rad': planner.precontact.return_value}
        motors.move.side_effect = outcomes
        runner = TaskRunner(planner, cfg, motors, Mock())
        runner._snapshot = lambda: {'q_rad': list(map(math.radians,[169.,20.])),
                                    'sliders_x_mm': {'蓝': -40., '橙': 10.}}
        runner._withdraw = Mock()
        runner._push_with_corrections = lambda c,g,d,s,r: s
        return runner

    def test_stall_then_oscillation_changes_only_subsequent_retries_and_resets_next_leg(self):
        r = self.runner([failure('motion_no_progress'), failure(), failure('motion_no_progress'), failure()])
        with patch('builtins.print'):
            result, state = r._run_leg('蓝', 0.)
        self.assertIsNone(state)
        self.assertEqual(len(result['attempts']), 4)
        plans = [c.args[0] for c in r.motors.move.call_args_list]
        self.assertEqual([p['segment_control']['speed_deg_s'] for p in plans], [10.,10.,5.,5.])
        self.assertEqual([p['points'][-1]['time_s'] for p in plans], [.02,.02,.04,.04])
        self.assertEqual(result['attempts'][1]['next_free_retry']['speed_deg_s'], 5.)
        self.assertNotIn('next_free_retry', result['attempts'][3])
        r.motors.move.side_effect = [failure('motion_no_progress')]*4
        r.motors.move.reset_mock()
        with patch('builtins.print'): r._run_leg('蓝', 0.)
        self.assertTrue(all(c.args[0]['segment_control']['speed_deg_s'] == 10.
                            for c in r.motors.move.call_args_list))

    def test_first_free_oscillation_can_trigger_next_retry_but_does_not_change_base_settings(self):
        r = self.runner([failure(), {'status':'completed','phase':'free'}])
        with patch('builtins.print'): result, _ = r._run_leg('蓝', 0.)
        self.assertEqual(result['status'], 'completed')
        plans = [c.args[0] for c in r.motors.move.call_args_list]
        self.assertEqual([p['segment_control']['speed_deg_s'] for p in plans], [10.,5.])
        self.assertEqual(phase_profile(r.segments,'withdraw')['speed_deg_s'], 10.)
        self.assertEqual(phase_profile(r.segments,'free')['speed_deg_s'], 10.)
        self.assertEqual(phase_profile(r.segments,'push')['speed_deg_s'], 60.)

    def test_no_reduction_for_stationary_settle_hold_timeout_or_non_free_errors(self):
        for error in (failure(oscillating=False), failure('motion_goal_hold_timeout'),
                      failure('joint_feedback_stale'), failure(name='withdraw')):
            r = self.runner([error]*4)
            with self.subTest(reason=error.reason,details=error.details), patch('builtins.print'):
                result, _ = r._run_leg('蓝', 0.)
                self.assertTrue(all('next_free_retry' not in a for a in result['attempts']))
                self.assertTrue(all(c.args[0]['segment_control']['speed_deg_s'] == 10.
                                    for c in r.motors.move.call_args_list))

    def test_retiming_preserves_route_and_can_only_select_free_profile(self):
        h=hardware();s=segment_settings(h)
        original=phase([[169.,20.],[168.9,20.]],times=[0.,.01]);saved=copy.deepcopy(original)
        slow=free_retry_phase(original,s)
        self.assertEqual(original,saved)
        self.assertEqual(slow,free_retry_phase(slow,s))
        MotorPair(h)._validate_phase(slow)
        with self.assertRaises(ValueError):
            MotorPair(h)._validate_phase({**original,'free_retry_after_oscillation':True})
        for name in ('push','push_correction','withdraw'):
            with self.assertRaises(ValueError): phase_profile(s,name,retry_after_oscillation=True)
        for speed in (0., 1., 11., float('nan'), True):
            bad=hardware();bad['segmented_motion']['free_retry_speed_deg_s']=speed
            with self.subTest(speed=speed),self.assertRaises(ValueError):segment_settings(bad)

    def test_real_move_emits_five_degree_speed_field_and_unchanged_nominal_targets(self):
        h=hardware();p=phase([[169.,20.],[168.9,20.],[168.8,20.]])
        p=free_retry_phase(p,segment_settings(h));d=VirtualDrive(model_q=[169.,20.],settings=h)
        result=d.run(p)
        self.assertEqual(result['status'],'completed')
        self.assertEqual(result['motion_diagnostics']['execution']['segment_control']['speed_deg_s'],5.)
        self.assertEqual([int.from_bytes(w[9:13],'little') for _,_,_,w in d.a6()], [500,500])
        self.assertEqual([int.from_bytes(w[6:9],'little') for _,_,_,w in d.a6()], [16890,16880])
        self.assertEqual(d.pair._settling_bias_centi,[0,0])

    def test_real_move_records_endpoint_oscillation_without_speed_change(self):
        def oscillate(d):
            if not d.a6(0): return
            angle=2*math.pi*d.elapsed/.55
            d.q[0]=168.9+.55*math.sin(angle)
            d.velocity[0]=.55*2*math.pi/.55*math.cos(angle)
        d=VirtualDrive(model_q=[169.,20.],settings=hardware(),disturbance=oscillate)
        with self.assertRaises(HardwareError) as caught:
            d.run(phase([[169.,20.],[168.9,20.]],times=[0.,.1]))
        self.assertEqual(caught.exception.reason,'motion_goal_settle_timeout')
        e=caught.exception.details['motion_diagnostics']['execution']
        self.assertEqual(e['endpoint_oscillating_axes'],[True,False])
        self.assertEqual(e['settling_bias_deg'],[0,0])
        self.assertTrue(all(int.from_bytes(w[9:13],'little') == 1000 for _,_,_,w in d.a6()))

    def test_real_move_stationary_endpoint_error_is_not_oscillation(self):
        def stuck(d):
            if d.a6(0): d.q[0],d.velocity[0]=169.25,0.
        d=VirtualDrive(model_q=[169.,20.],settings=hardware(),disturbance=stuck)
        with self.assertRaises(HardwareError) as caught:
            d.run(phase([[169.,20.],[168.9,20.]],times=[0.,.1]))
        self.assertEqual(caught.exception.reason,'motion_goal_settle_timeout')
        self.assertEqual(caught.exception.details['motion_diagnostics']['execution']['endpoint_oscillating_axes'],[False,False])

    def test_right_precontact_is_ten_mm_and_left_remains_two(self):
        cfg,_=load_config(ROOT)
        planner=Planner(ROOT,cfg)
        positions={'蓝':-42.568835888799725,'橙':8.950888573262517}
        seed=list(map(math.radians,[171.54,29.26]))
        q=planner.precontact('蓝','+x',positions,seed)
        self.assertAlmostEqual(positions['蓝']-planner.arm.slider_center(q,'+x')[0],10.)
        self.assertEqual(planner.offsets['-x'],2.)
        planned=planner.push(q,'蓝',-5.919949284483067,positions,[-65.88712614953508,-5.919949284483067])
        self.assertAlmostEqual(planned['start_gap_mm'],10.)
        self.assertEqual(planned['segment_control']['speed_deg_s'],60.)
        travelled=0.
        for index,cap in enumerate(planned['segment_step_caps_deg']):
            a,b=planned['points'][index:index+2]
            if travelled < 10.-1e-7:self.assertEqual(cap,.2)
            travelled+=math.dist(planner.arm.slider_center(a['q_rad'],'+x'),planner.arm.slider_center(b['q_rad'],'+x'))


if __name__ == '__main__': unittest.main()
