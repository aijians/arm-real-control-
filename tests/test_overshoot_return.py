"""Controlled nominal return after measured overshoot, using virtual devices."""
from contextlib import ExitStack
import math,unittest
from unittest.mock import patch
from test_recovery90 import VirtualDrive,phase,scene_context,m,MotionRecovery,policy_module,recovery_module
from test_waypoint_retries import three_round_config

def return_config():
    c=three_round_config();c['motion_recovery'].update(overshoot_return_enabled=True,overshoot_return_speed_deg_s=18.);return c

class OvershootDrive(VirtualDrive):
    def __init__(self,*args,axis=0,sign=-1,nominal=157.,stubborn=False,**kw):
        self.axis=axis;self.sign=sign;self.nominal=nominal;self.injected=False;self.return_commanded=False;self.stubborn=stubborn
        super().__init__(*args,**kw)
    def send(self,axis,wire,**kw):
        if axis==self.axis and wire[1]==0xa6:
            target=int.from_bytes(wire[6:9],'little')/100.
            model=(target-self.pair.motors[axis]['offset_deg'])/self.pair.motors[axis]['ratio']
            if abs(model-self.nominal)<.011 and wire[5] == (1 if self.sign>0 else 0):
                self.return_commanded=True
            elif not self.injected:
                self.injected=True
        return super().send(axis,wire,**kw)
    def advance(self,seconds):
        old=self.q[self.axis]
        super().advance(seconds)
        if self.injected and (not self.return_commanded or self.stubborn):
            goal=self.pair._motor_deg(self.pair.motors[self.axis],self.nominal+self.sign*4.)
            self.q[self.axis]=old+self.sign*min(abs(goal-old),8.*max(0.,seconds))
            self.velocity[self.axis]=self.sign*8. if abs(self.q[self.axis]-goal)>.001 else 0.
            self.running[self.axis]=abs(self.q[self.axis]-goal)>.001

class ReturnMoveTests(unittest.TestCase):
    def test_both_axes_directions_and_all_phases_return_to_original_nominal(self):
        for name in ('free','approach','push','push_correction','withdraw'):
            for axis in (0,1):
                for sign in (-1,1):
                    with self.subTest(phase=name,axis=axis,sign=sign):
                        start=[159.,20.];goal=start.copy();goal[axis]+=sign*2.
                        d=OvershootDrive(settings=return_config(),axis=axis,sign=sign,nominal=goal[axis])
                        result=d.run(phase([start,goal],name=name,motion_recovery_geometry=scene_context()))
                        r=result['motion_diagnostics']['execution']['speed_recovery']
                        written=[e for e in r['events'] if e['event']=='overshoot_return_written']
                        self.assertEqual(len(written),1);self.assertEqual(written[0]['speed_deg_s'],18.)
                        self.assertFalse(any(r['overshoot_return_active']))
                        self.assertEqual(d.pair._settling_bias_centi,[0,0]);self.assertEqual(d.a6(1-axis),[])
                        self.assertEqual(d.unsafe_directions,[])
                        self.assertLessEqual(abs(math.degrees(result['feedback']['q_rad'][axis])-goal[axis]),.3)
    def test_failed_physical_return_gets_three_rounds_and_then_exits(self):
        d=OvershootDrive(settings=return_config(),stubborn=True)
        with self.assertRaises(m.HardwareError) as caught:
            d.run(phase([[159.,20.],[157.,20.]],motion_recovery_geometry=scene_context()))
        self.assertEqual(caught.exception.reason,'motion_recovery_exhausted')
        r=caught.exception.details['motion_diagnostics']['execution']['speed_recovery']
        returned=[e for e in r['events'] if e['event']=='overshoot_return_written']
        self.assertEqual([e['recovery_round'] for e in returned],[1,2,3]);self.assertLess(d.elapsed,24.)
        self.assertEqual(d.unsafe_directions,[])
    def test_current_history_of_motion_is_overshoot_not_stuck_despite_zero_status(self):
        c=return_config();r=MotionRecovery(recovery_module.recovery_settings(c),c['motors'],'push',policy_module.motion_policy_settings(c))
        r.select_directions([-1,0],0.)
        goal=list(map(math.radians,[156.5283198677334,50.08699932544579]))
        for i,q in enumerate((155.62,155.19,154.77,154.41,154.04,153.73,153.47,153.22,153.0)):
            state={'q_rad':list(map(math.radians,[q,50.02])),'speed_rad_s':[0.,0.], 'axis_stamp_monotonic_s':[9.+i*.05,9.+i*.05]}
            r.observe_goal(state,goal)
        self.assertEqual(r.classify(0,9.4,state,goal,.604239,8.9),'overshoot')
        self.assertFalse(r._angle_window_still(0,state,.5))

class ReturnWireTests(unittest.TestCase):
    def setUp(self):
        self.d=VirtualDrive(immobile=True,settings=return_config());self.p=self.d.pair
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(m.time,'monotonic',lambda:self.d.now))
        self.stack.enter_context(patch.object(m.time,'sleep',self.d.advance))
        self.refresh();self.send(False)
        self.p._last_target_acked[0]=True
    def refresh(self):
        for i in (0,1):self.d.refresh(i,0x94);self.d.refresh(i,0x9c)
    def send(self,returning=True,validator=None):
        return self.p._send_targets(list(map(math.radians,[157.,20.])),
            planned_directions=[-1,0],command_axes=[True,False],axis_at_endpoint=[True,False],
            adaptive_mode=True,nominal_return_axes=[returning,False],
            recovery_validator=validator or (lambda *a:True),
            axis_min_speed_deg_s=[18.,0.],speed_tiers=[-1 if returning else 0,0],
            command_speed_caps_deg_s=[18.,18.])
    def stopped_beyond(self):
        self.d.q[0]=153.;self.refresh();self.send();self.d.advance(.12);self.refresh()
    def test_same_target_reverse_only_after_sustained_still_feedback(self):
        self.d.q[0]=153.;self.d.velocity[0]=-5.;self.refresh()
        self.assertEqual(self.send(),0)
        self.d.velocity[0]=0.;self.refresh();self.assertEqual(self.send(),0)
        self.d.advance(.12);self.refresh();self.assertEqual(self.send(),1)
        self.assertEqual(self.d.a6(0)[-1][3][5],0)
        self.assertEqual(int.from_bytes(self.d.a6(0)[-1][3][6:9],'little'),15700)
        self.assertEqual(self.p._settling_bias_centi,[0,0]);self.assertEqual(self.d.unsafe_directions,[])
    def test_one_zero_status_with_changing_angles_does_not_allow_reverse(self):
        self.d.q[0]=154.;self.refresh()
        for i in range(6):
            self.d.advance(.05);self.d.q[0]-=.2;self.refresh()
            self.assertEqual(self.send(),0)
        self.assertEqual(len(self.d.a6(0)),1)
    def test_geometry_rejection_blocks_return_even_when_target_is_unchanged(self):
        self.stopped_beyond();self.assertEqual(self.send(validator=lambda *a:False),0)
        self.assertEqual(len(self.d.a6(0)),1)
    def test_failed_return_write_keeps_target_direction_and_bias_state(self):
        self.stopped_beyond()
        previous=self.p._last_targets[0]
        self.d.fail_write=lambda axis,wire,drive:wire[1]==0xa6
        with self.assertRaises(OSError):self.send()
        self.assertEqual(self.p._last_targets[0],previous)
        self.assertEqual(self.p._settling_bias_centi,[0,0])
    def test_stale_feedback_never_allows_return(self):
        self.stopped_beyond();self.d.advance(.6)
        self.assertEqual(self.send(),0)
        self.assertEqual(len(self.d.a6(0)),1)
    def test_without_explicit_return_request_same_target_direction_change_stays_blocked(self):
        self.stopped_beyond();self.assertEqual(self.send(False),0)
        self.assertEqual(len(self.d.a6(0)),1)

class CommandEvidenceTests(unittest.TestCase):
    def test_skipped_decisions_do_not_evict_actual_sent_commands(self):
        d=VirtualDrive(settings=return_config())
        result=d.run(phase([[159.,20.],[157.,20.]],motion_recovery_geometry=scene_context()))
        trace=result['motion_diagnostics']
        self.assertEqual(trace['dropped_sent_actions'],0)
        self.assertEqual(len(trace['sent_actions']),len(d.a6()))
        self.assertTrue(all(e['status']=='sent' and 'frame_hex' in e for e in trace['sent_actions']))

if __name__=='__main__':unittest.main(verbosity=2)
