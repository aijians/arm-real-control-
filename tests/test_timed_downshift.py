"""Exercise real MotorPair A4 writes against a virtual oscillating drive."""
import math
import unittest

from virtual_drive import VirtualDrive, config, phase, m
from control.timed_downshift import TimedSpeedDownshift, timed_downshift_settings


def settings(**extra):
    timed = {"enabled": True, "speed_deg_s": 60., "push_upper_speed_deg_s": 60.,
             "max_step_deg": .2, "period_s": .05,
             "oscillation_downshift": {"enabled": True, "speed_tiers_deg_s": [18., 30., 60.]}}
    return config(free_timed_motion=timed, goal_settle_timeout_s=5., **extra)


class OscillatingDrive(VirtualDrive):
    """A fixed goal rings until its A4 speed is reduced sufficiently."""
    def __init__(self, *args, axes=(0,), stable_at=18., **kwargs):
        self.oscillating_axes = axes
        self.stable_at = stable_at
        self.final = [157., 21.]
        self.final_written_at = [None, None]
        super().__init__(*args, **kwargs)

    def send(self, axis, wire, **kwargs):
        super().send(axis, wire, **kwargs)
        if wire[1] == 0xA4:
            motor = self.pair.motors[axis]
            final = self.pair._motor_deg(motor, self.final[axis])
            if abs(self.target[axis]-final) < .005 and self.final_written_at[axis] is None:
                self.final_written_at[axis] = self.elapsed

    def advance(self, seconds):
        super().advance(seconds)
        for i in self.oscillating_axes:
            since = self.final_written_at[i]
            if since is None or self.command_speed[i] <= self.stable_at:
                continue
            t = self.elapsed - since
            motor = self.pair.motors[i]
            # Include >1 degree error: the old 1-degree-only gate missed c10104e6.
            q = self.final[i] + 1.4*math.sin(2*math.pi*2.5*t)
            self.q[i] = self.pair._motor_deg(motor, q)
            self.velocity[i] = 1.4*2*math.pi*2.5*math.cos(2*math.pi*2.5*t)


def execution(result):
    return result['motion_diagnostics']['execution']


class TimedDownshiftTests(unittest.TestCase):
    def test_two_step_downshift_writes_identical_goal_only_to_oscillating_axis(self):
        for name in ('free', 'push', 'push_correction'):
            with self.subTest(phase=name):
                d=OscillatingDrive(settings=settings())
                result=d.run(phase([[159,20],[157,21]],name=name))
                events=execution(result)['oscillation_downshift']['events']
                self.assertEqual([(e['motor'],e['from_speed_deg_s'],e['to_speed_deg_s']) for e in events],
                                 [('upper',60.,30.),('upper',30.,18.)])
                self.assertGreaterEqual(events[1]['write_started_elapsed_s']-events[0]['write_completed_elapsed_s'],.95)
                actions=result['motion_diagnostics']['actions']
                writes=[a for a in actions if a['purpose']=='oscillation_downshift']
                self.assertEqual([a['target_motor_deg'] for a in writes],[157.,157.])
                self.assertTrue(all(a['bias_motor_deg']==0 for a in actions))
                self.assertEqual(len(d.a4(0)),12)
                self.assertEqual(len(d.a4(1)),10)
                self.assertEqual(d.a6(),[])
                self.assertEqual(execution(result)['waypoints_dispatched'],10)
                self.assertEqual(execution(result)['motion_commands_sent'],22)
                self.assertEqual(result['status'],'completed')

    def test_either_joint_and_both_joints_are_independent(self):
        for axes in [(1,),(0,1)]:
            d=OscillatingDrive(settings=settings(),axes=axes,stable_at=30.)
            result=d.run(phase([[159,20],[157,21]]))
            events=execution(result)['oscillation_downshift']['events']
            self.assertEqual({e['motor'] for e in events},{d.pair.motors[i]['name'] for i in axes})
            self.assertEqual(len(events),len(axes))
            self.assertTrue(all(e['to_speed_deg_s']==30 for e in events))

    def test_caps_survive_retreat_and_retry_but_new_task_starts_high(self):
        d=OscillatingDrive(settings=settings(),stable_at=30.)
        d.run(phase([[159,20],[157,21]]))
        old_count=len(d.frames)
        result=d.run(phase([[157,21],[158,20]],route='observation_retreat'))
        new=[x for x in d.frames[old_count:] if x[2]==0xA4]
        self.assertTrue(all(int.from_bytes(w[13:17],'little')==3000 for _,i,_,w in new if i==0))
        self.assertTrue(all(int.from_bytes(w[13:17],'little')==6000 for _,i,_,w in new if i==1))
        self.assertEqual(execution(result)['oscillation_downshift']['events'],[])
        fresh=VirtualDrive(settings=settings())
        fresh.run(phase([[159,20],[158,21]]))
        self.assertTrue(all(int.from_bytes(w[13:17],'little')==6000 for *_,w in fresh.a4()))

    def test_never_increase_a_lower_configured_base_speed(self):
        cfg=settings();cfg['free_timed_motion']['speed_deg_s']=10.
        d=VirtualDrive(settings=cfg);d.run(phase([[159,20],[158,21]]))
        self.assertTrue(all(int.from_bytes(w[13:17],'little')==1000 for *_,w in d.a4()))

    def test_low_tier_does_not_extend_deadline_or_restart_forever(self):
        d=OscillatingDrive(settings=settings(),stable_at=0.)
        with self.assertRaises(m.HardwareError) as caught:
            d.run(phase([[159,20],[157,21]]))
        self.assertEqual(caught.exception.reason,'motion_goal_settle_timeout')
        diag=caught.exception.details['motion_diagnostics'];ex=diag['execution']
        self.assertEqual([e['to_speed_deg_s'] for e in ex['oscillation_downshift']['events']],[30.,18.])
        elapsed=diag['capture']['captured_monotonic_s']-diag['motion_started_monotonic_s']
        self.assertTrue(5. <= elapsed-ex['dispatch_completed_elapsed_s'] < 5.1)
        self.assertEqual(len(d.halts),1)
        self.assertFalse(d.pair._move_lock.locked())

    def test_disabled_and_stationary_shortfall_do_not_trigger_downshift(self):
        for disabled in (False,True):
            cfg=settings();cfg['free_timed_motion']['oscillation_downshift']['enabled']=not disabled
            d=VirtualDrive(settings=cfg,immobile=True)
            with self.assertRaises(m.HardwareError) as caught:d.run(phase([[159,20],[157,21]]))
            ex=caught.exception.details['motion_diagnostics']['execution']
            self.assertEqual(ex['oscillation_downshift']['events'],[])
            self.assertEqual(len(d.a4()),20)

    def test_write_failure_does_not_commit_new_cap_and_halts(self):
        def fail(axis,wire,drive):
            return wire[1]==0xA4 and int.from_bytes(wire[13:17],'little')==3000
        d=OscillatingDrive(settings=settings(),fail_write=fail)
        with self.assertRaises(m.HardwareError):d.run(phase([[159,20],[157,21]]))
        self.assertEqual(d.pair._timed_downshift.limits,[60.,60.])
        self.assertEqual(len(d.halts),1)

    def test_cancel_health_failure_and_feedback_loss_never_send_after_failure(self):
        for mode in ('cancel','health','feedback'):
            d=OscillatingDrive(settings=settings(),query_loss=(
                lambda axis,cmd,t:t<1.) if mode=='feedback' else None)
            def health():
                if d.elapsed>1.1:
                    if mode=='cancel':d.pair._halt_event.set()
                    elif mode=='health':raise m.HardwareError('camera_unavailable')
            with self.assertRaises(m.HardwareError):
                d.run(phase([[159,20],[157,21]]),health_check=health)
            self.assertEqual(len(d.halts),1)
            self.assertFalse(d.pair._move_lock.locked())

    def test_offset_mapping_is_identical_across_downshift_writes(self):
        cfg=settings();cfg['motors'][0]['offset_deg']=-3.
        d=OscillatingDrive(settings=cfg)
        result=d.run(phase([[159,20],[157,21]]))
        writes=[a for a in result['motion_diagnostics']['actions'] if a['purpose']=='oscillation_downshift']
        self.assertEqual([a['target_motor_deg'] for a in writes],[154.,154.])

    def test_noop_path_never_reasserts_goal(self):
        d=VirtualDrive(settings=settings())
        result=d.run(phase([[159,20]],times=[0]))
        self.assertEqual(d.a4(),[])
        self.assertFalse(execution(result)['oscillation_downshift']['active'])


class PolicyTests(unittest.TestCase):
    def test_duplicate_or_old_speed_stamps_cannot_build_oscillation_window(self):
        policy=TimedSpeedDownshift(timed_downshift_settings(settings()['free_timed_motion']),
                                  config()['motors'])
        for t in range(50):
            state={'q_rad':[math.radians(157+math.sin(t)),math.radians(21)],
                   'axis_stamp_monotonic_s':[t+10,t+10], 'speed_stamp_monotonic_s':[10,10]}
            self.assertEqual(policy.observe(state,[math.radians(157),math.radians(21)],[60,60],[9,9]),{})

    def test_monotonic_convergence_is_not_oscillation(self):
        policy=TimedSpeedDownshift(timed_downshift_settings(settings()['free_timed_motion']),config()['motors'])
        for i in range(50):
            t=10+i*.05
            state={'q_rad':[math.radians(157+2-i*.03),math.radians(21)],
                   'axis_stamp_monotonic_s':[t,t], 'speed_stamp_monotonic_s':[t,t]}
            self.assertEqual(policy.observe(state,[math.radians(157),math.radians(21)],[60,60],[9,9]),{})

    def test_bad_configuration_is_rejected_before_opening_devices(self):
        for raw in [None,{'enabled':1},{'unknown':True}, {'speed_tiers_deg_s':[18,60]},
                    {'speed_tiers_deg_s':[18,30,60,90]}, {'speed_tiers_deg_s':[30,18,60]},
                    {'speed_tiers_deg_s':[18,30,True]}, {'speed_tiers_deg_s':[18,30,float('nan')]},
                    {'oscillation_reversals':1}, {'oscillation_reversal_deg':.2}]:
            cfg=settings();cfg['free_timed_motion']['oscillation_downshift']=raw
            with self.subTest(raw=raw),self.assertRaises(ValueError):m.MotorPair(cfg)
