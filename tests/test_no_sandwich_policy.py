"""No hardware: production angle zones, unchanged-target speed recovery and recall."""
from contextlib import ExitStack
import copy
import math
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from control import motors as m
from control.config import load_config
from control.motion_policy import MotionRecovery, angle_bias_profile, motion_policy_settings
from control.recovery import recovery_settings
from test_overshoot_return import return_config
from test_recovery90 import LastResortDrive, scene_context
from virtual_drive import VirtualDrive, phase


def configured():
    cfg=return_config()
    cfg['motion_recovery'].update(bias_retraction_requires_crossing=True,
        bias_retraction_speed_deg_s=18.,
        axis_angle_bias={'upper': {'below_model_deg':160., 'max_deg':.2}})
    return cfg


class AngleZoneTests(unittest.TestCase):
    def test_actual_and_nominal_must_both_be_strictly_below_160(self):
        cfg=configured();policy=motion_policy_settings(cfg);motor=cfg['motors'][0]
        for actual,goal,allowed in ((169.,168.,False),(160.,159.,False),
                                    (159.,160.,False),(159.99,159.,True)):
            with self.subTest(actual=actual,goal=goal):
                enabled,cap,steps=angle_bias_profile(policy,motor,actual,goal)
                self.assertEqual(enabled,allowed);self.assertEqual(cap,.2);self.assertEqual(steps,[.2])

    def test_other_joint_keeps_its_existing_profile(self):
        cfg=configured();policy=motion_policy_settings(cfg)
        self.assertEqual(angle_bias_profile(policy,cfg['motors'][1],20.,19.),
                         (True,1.2,[.2,.6,1.2]))

    def test_invalid_or_unknown_joint_rules_are_rejected(self):
        for rules in ({'missing':{'below_model_deg':160.,'max_deg':.2}},
                      {'upper':{'below_model_deg':True,'max_deg':.2}},
                      {'upper':{'below_model_deg':160.,'max_deg':2.}},
                      {'upper':{'below_model_deg':160.}}):
            cfg=configured();cfg['motion_recovery']['axis_angle_bias']=rules
            with self.subTest(rules=rules),self.assertRaises(ValueError):motion_policy_settings(cfg)

    def test_production_configuration_uses_segments_without_recovery(self):
        cfg,_=load_config(Path(__file__).resolve().parents[1]);h=cfg['hardware']
        policy=motion_policy_settings(h)
        self.assertTrue(h['segmented_motion']['enabled'])
        self.assertFalse(policy['enabled'])
        self.assertFalse(recovery_settings(h)['enabled'])


class RecoveryIntegrationTests(unittest.TestCase):
    def test_entering_below_160_after_speed_only_90_allows_first_small_bias(self):
        cfg=configured();rec=MotionRecovery(recovery_settings(cfg),cfg['motors'],'free',
            motion_policy_settings(cfg));rec.select_directions([-1,0],0.)
        rec.final_speed_used[0]=True;rec.axis_min_speed_deg_s[0]=90.;rec.speed_tier[0]=4
        state={'q_rad':list(map(math.radians,[159.5,20.])), 'speed_rad_s':[0.,0.],
               'axis_stamp_monotonic_s':[2.,2.]}
        rec.update(2.,state,list(map(math.radians,[158.,20.])),[.3,.3],[0.,0.],
                   [0],[(15800,1,9000,4),None],[0.,0.])
        self.assertEqual(rec.pending[0]['kind'],'angle_up')
        self.assertTrue(rec.wrote(0,4,90.,2.1,bias=-.2))
        state['axis_stamp_monotonic_s']=[3.,3.]
        rec.update(3.,state,list(map(math.radians,[158.,20.])),[.3,.3],[0.,0.],
                   [0],[(15780,1,9000,4),None],[-.2,0.])
        self.assertIsNone(rec.pending[0])

    def test_170_to_160_region_can_reach_90_without_any_bias(self):
        d=LastResortDrive(model_q=[169.,20.],settings=configured(),nominal=168.,release_speed=90.)
        result=d.run(phase([[169.,20.],[168.,20.]],motion_recovery_geometry=scene_context()))
        frames=d.a6(0)
        self.assertTrue(all(int.from_bytes(w[6:9],'little')==16800 for _,_,_,w in frames))
        self.assertEqual([int.from_bytes(w[9:13],'little')/100 for _,_,_,w in frames][-4:],
                         [18.,30.,60.,90.])
        events=result['motion_diagnostics']['execution']['speed_recovery']['events']
        self.assertFalse(any(e['event']=='angle_change_written' for e in events))
        self.assertEqual(d.unsafe_directions,[])

    def test_below_160_only_point2_then_90_on_existing_biased_target(self):
        d=LastResortDrive(model_q=[159.,20.],settings=configured(),nominal=157.,release_speed=90.)
        result=d.run(phase([[159.,20.],[157.,20.]],motion_recovery_geometry=scene_context()))
        frames=d.a6(0);ix=next(i for i,x in enumerate(frames) if int.from_bytes(x[3][9:13],'little')==9000)
        self.assertEqual(int.from_bytes(frames[ix-1][3][6:9],'little'),15680)
        self.assertEqual(frames[ix-1][3][5:9],frames[ix][3][5:9])
        self.assertEqual(int.from_bytes(frames[-1][3][6:9],'little'),15700)
        self.assertEqual(int.from_bytes(frames[-1][3][9:13],'little'),1800)
        events=result['motion_diagnostics']['execution']['speed_recovery']['events']
        biases=[e['bias_motor_deg'] for e in events if e['event']=='angle_change_written']
        self.assertEqual(biases,[-.2]);self.assertEqual(d.pair._settling_bias_centi,[0,0])
        self.assertEqual(d.unsafe_directions,[])

    def test_speed_only_stall_still_has_three_bounded_rounds(self):
        d=VirtualDrive(model_q=[169.,20.],settings=configured(),immobile=True)
        with self.assertRaises(m.HardwareError) as caught:
            d.run(phase([[169.,20.],[168.,20.]],motion_recovery_geometry=scene_context()))
        self.assertEqual(caught.exception.reason,'motion_recovery_exhausted')
        rec=caught.exception.details['motion_diagnostics']['execution']['speed_recovery']
        self.assertEqual(rec['total_rounds_started'],3)
        self.assertFalse(any(e['event']=='angle_change_written' for e in rec['events']))
        self.assertTrue(all(int.from_bytes(w[6:9],'little')==16800 for _,_,_,w in d.a6(0)))
        self.assertLess(d.elapsed,24.)


class RecallWireTests(unittest.TestCase):
    def test_b_and_c_geometry_never_sends_near_side_recall_both_mappings(self):
        for sign,offset in ((1,-1.75),(-1,200.)):
            cfg=configured();cfg['motors'][0].update(sign=sign,offset_deg=offset,limits_deg=[1.75,180.])
            d=VirtualDrive(model_q=[163.41,20.],settings=cfg,immobile=True);p=d.pair
            model_to_motor=lambda q:offset+sign*q
            with ExitStack() as stack:
                stack.enter_context(patch.object(m.time,'monotonic',lambda:d.now))
                stack.enter_context(patch.object(m.time,'sleep',d.advance))
                for i in (0,1):d.refresh(i,0x94);d.refresh(i,0x9c)
                p._last_targets[0]=(round(model_to_motor(161.81)*100),1 if sign==1 else 0,6000,3)
                p._last_target_sent_at[0]=d.now-1.;p._last_target_acked[0]=True
                p._settling_bias_centi[0]=-sign*120
                opts=dict(command_axes=[True,False],axis_at_endpoint=[True,False],adaptive_mode=True,
                    axis_min_speed_deg_s=[60.,0.],speed_tiers=[3,0],command_speed_caps_deg_s=[60.,18.])
                goal=list(map(math.radians,[163.013187,20.]))
                validator=Mock(side_effect=AssertionError('must reject before geometry'))
                p._stationary_probe[0]=(d.now-.12,d.q[0])
                self.assertEqual(p._send_targets(goal,recovery_validator=validator,**opts),0)
                validator.assert_not_called()
                d.q[0]=model_to_motor(162.74);d.velocity[0]=-sign*5.
                d.refresh(0,0x94);d.refresh(0,0x9c)
                self.assertEqual(p._send_targets(goal,recovery_validator=validator,**opts),0)
                d.velocity[0]=0.;d.refresh(0,0x94);d.refresh(0,0x9c)
                p._stationary_probe[0]=(d.now-.12,d.q[0])
                self.assertEqual(p._send_targets(goal,recovery_validator=lambda *a:True,**opts),1)
                wire=d.a6(0)[-1][3]
                self.assertEqual(wire[5],0 if sign==1 else 1)
                self.assertEqual(int.from_bytes(wire[9:13],'little'),1800)
                self.assertEqual(p._settling_bias_centi,[0,0])

    def test_partial_bias_reduction_cannot_create_sandwich(self):
        d=VirtualDrive(model_q=[157.9,20.],settings=configured(),immobile=True);p=d.pair
        with patch.object(m.time,'monotonic',lambda:d.now):
            for i in (0,1):d.refresh(i,0x94);d.refresh(i,0x9c)
            p._last_targets[0]=(15580,1,6000,3);p._last_target_sent_at[0]=d.now-1.
            p._last_target_acked[0]=True;p._settling_bias_centi[0]=-120
            p._stationary_probe[0]=(d.now-.12,d.q[0])
            validator=Mock(side_effect=AssertionError('must reject partial recall'))
            self.assertEqual(p._send_targets(list(map(math.radians,[157.,20.])),
                command_axes=[True,False],axis_at_endpoint=[True,False],adaptive_mode=True,
                recovery_bias_modes=[-1,0],recovery_validator=validator,
                axis_min_speed_deg_s=[60.,0.],speed_tiers=[3,0],command_speed_caps_deg_s=[60.,18.]),0)
            validator.assert_not_called();self.assertEqual(p._settling_bias_centi[0],-120)

    def test_speed_raise_keeps_existing_bias_until_safe_recall(self):
        d=VirtualDrive(model_q=[157.5,20.],settings=configured(),immobile=True);p=d.pair
        with patch.object(m.time,'monotonic',lambda:d.now):
            for i in (0,1):d.refresh(i,0x94);d.refresh(i,0x9c)
            p._last_targets[0]=(15680,1,6000,3);p._last_target_sent_at[0]=d.now-1.
            p._last_target_acked[0]=True;p._settling_bias_centi[0]=-20
            p._stationary_probe[0]=(d.now-.12,d.q[0])
            self.assertEqual(p._send_targets(list(map(math.radians,[157.,20.])),
                command_axes=[True,False],axis_at_endpoint=[True,False],adaptive_mode=True,
                stuck_upshift_axes=[True,False],recovery_validator=lambda *a:True,
                axis_min_speed_deg_s=[90.,0.],speed_tiers=[4,0],command_speed_caps_deg_s=[90.,18.]),1)
            wire=d.a6(0)[-1][3]
            self.assertEqual(int.from_bytes(wire[6:9],'little'),15680)
            self.assertEqual(int.from_bytes(wire[9:13],'little'),9000)
            self.assertEqual(p._settling_bias_centi[0],-20)


if __name__=='__main__':unittest.main()
