"""Current production policy: measured feedback, discrete bias, nominal recall then 90."""
import copy,json,math,os,unittest
from pathlib import Path
from unittest.mock import patch
from virtual_drive import config,VirtualDrive,phase,m
from control.motion_policy import MotionRecovery
from control import motion_policy as policy_module,recovery as recovery_module

def configuration(**extra):
    recovery=extra.pop('recovery',{})
    policy=extra.pop('policy',{})
    return config(speed_recovery={'enabled':True,**recovery},
                  motion_recovery={'enabled':True,**policy},**extra)

def scene_context():
    return {'scene': {'arm':{'joint_limits_rad':[[0,math.pi],[0,math.pi]],
                     'elbow_translation_mm':[0,0],'shoulder_static_yaw_rad':0,'elbow_static_yaw_rad':0},
                     'bodies':[],'workspace_xy_mm':[[-1000,-1000],[1000,1000]],'obstacles':[]},
            'sliders':{},'contact':None}


def fast_config():
    return configuration(recovery={'speed_steps_deg_s':[18.,30.,60.,90.],
        'trigger_s':.5,'observe_s':.5,'max_upshifts':4,'total_recovery_s':24.},
        policy={'angle_steps_deg':[.2,.6,1.2],'angle_max_deg':1.2,'final_speed_deg_s':90.},
        goal_settle_timeout_s=15.)

class LastResortDrive(VirtualDrive):
    def __init__(self,*a,axis=0,release_speed=90.,release_bias=None,nominal=157.,**kw):
        self.axis=axis;self.release_speed=release_speed;self.release_bias=release_bias
        self.nominal=nominal;self.released=False
        super().__init__(*a,**kw)
    def advance(self,seconds):
        old=self.q[self.axis];old_moved=self.has_moved[self.axis]
        motor=self.pair.motors[self.axis]
        target=(self.target[self.axis]-motor['offset_deg'])/(motor['sign']*motor['ratio'])
        if (self.command_speed[self.axis]>=self.release_speed-.005 or
                self.release_bias is not None and target <= self.nominal-self.release_bias+.005):
            self.released=True
        super().advance(seconds)
        if not self.released:
            self.q[self.axis],self.velocity[self.axis],self.running[self.axis],self.has_moved[self.axis]=old,0.,False,old_moved

class FastRecoveryTests(unittest.TestCase):
    def test_all_phases_both_axes_directions_clear_bias_before_90(self):
        for name in ('free','approach','push','push_correction','withdraw'):
            for axis in (0,1):
                for sign in (-1,1):
                    with self.subTest(phase=name,axis=axis,sign=sign):
                        start=[159.,20.];end=start.copy();end[axis]+=sign*2.
                        d=LastResortDrive(settings=fast_config(),axis=axis,nominal=end[axis])
                        result=d.run(phase([start,end],name=name,motion_recovery_geometry=scene_context()))
                        ev=result['motion_diagnostics']['execution']['speed_recovery']['events']
                        bias=[e for e in ev if e['event']=='angle_change_written']
                        self.assertEqual([round(e['bias_motor_deg'],2) for e in bias],[sign*.2,sign*.6,sign*1.2])
                        clear=next(e for e in ev if e['event']=='bias_cleared_before_final_speed')
                        final=next(e for e in ev if e['event']=='final_speed_written')
                        self.assertGreaterEqual(clear['at_monotonic_s']-bias[-1]['at_monotonic_s'],.5-1e-6)
                        self.assertGreater(final['at_monotonic_s'],clear['at_monotonic_s'])
                        wire=d.a6(axis);nominal=round(end[axis]*100)
                        speeds=[int.from_bytes(w[9:13],'little')/100 for _,_,_,w in wire]
                        self.assertNotIn(45.,speeds)
                        pos=[int.from_bytes(w[6:9],'little') for _,_,_,w in wire]
                        ix=next(i for i,s in enumerate(speeds) if s==90.)
                        self.assertEqual(pos[ix-1:ix+1],[nominal,nominal])
                        self.assertEqual(speeds[ix-1],60.)
                        self.assertEqual(d.pair._settling_bias_centi,[0,0]);self.assertEqual(d.a6(1-axis),[])
                        self.assertEqual(d.unsafe_directions,[])
    def test_progress_at_point6_never_requests_point12_or_90(self):
        d=LastResortDrive(settings=fast_config(),release_bias=.6)
        result=d.run(phase([[159.,20.],[157.,20.]],motion_recovery_geometry=scene_context()))
        events=result['motion_diagnostics']['execution']['speed_recovery']['events']
        self.assertEqual([round(e['bias_motor_deg'],2) for e in events if e['event']=='angle_change_written'],[-.2,-.6])
        self.assertFalse(any(e['event']=='final_speed_written' for e in events))
        self.assertEqual(d.pair._settling_bias_centi,[0,0])
    def test_still_blocked_after_90_exhausts_without_new_bias_or_whole_leg_retry(self):
        d=VirtualDrive(immobile=True,settings=fast_config())
        with self.assertRaises(m.HardwareError) as caught:
            d.run(phase([[159.,20.],[157.,20.]],motion_recovery_geometry=scene_context()))
        self.assertEqual(caught.exception.reason,'motion_recovery_exhausted')
        ev=caught.exception.details['motion_diagnostics']['execution']['speed_recovery']['events']
        self.assertEqual(sum(e['event']=='final_speed_written' for e in ev),1)
        self.assertEqual(sum(e['event']=='angle_change_written' for e in ev),3)
        self.assertLess(d.elapsed,12.)
    def test_recall_not_written_does_not_unlock_90(self):
        cfg=fast_config();r=MotionRecovery(recovery_module.recovery_settings(cfg),cfg['motors'],'push',policy_module.motion_policy_settings(cfg))
        r.select_directions([-1,0],0.);r.speed_tier[0]=3;r.axis_min_speed_deg_s[0]=60.;r.angle_changes[0]=3
        state={'q_rad':list(map(math.radians,[159.,20.])),'speed_rad_s':[0,0],'axis_stamp_monotonic_s':[2.,2.]}
        r.update(2.,state,list(map(math.radians,[157.,20.])),[.3,.3],[0,0],[0],[(15580,1,6000,3),None],[-1.2,0])
        self.assertEqual(r.pending[0]['kind'],'bias_clear');self.assertEqual(r.angle_modes(),[-2,0])
        self.assertFalse(r.wrote(0,3,60.,2.1,bias=-1.2))
        r.update(3.,state,list(map(math.radians,[157.,20.])),[.3,.3],[0,0],[0],[(15580,1,6000,3),None],[-1.2,0])
        self.assertEqual(r.axis_min_speed_deg_s[0],60.);self.assertFalse(r.final_speed_ready[0])
    def test_invalid_final_speed_and_nonmonotonic_bias_are_rejected(self):
        for update in ({'final_speed_deg_s':60.},{'angle_steps_deg':[.2,1.2,.6]},{'angle_steps_deg':[.2,.6]}, {'angle_max_deg':2.}):
            cfg=fast_config();cfg['motion_recovery'].update(update)
            with self.subTest(update=update),self.assertRaises(ValueError):policy_module.motion_policy_settings(cfg)


if __name__=='__main__':unittest.main(verbosity=2)
