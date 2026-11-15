"""Return-only threshold; offline qualification and real executor with virtual UART."""
import math, unittest, copy
from unittest.mock import patch
from control.trajectory import GoalQualification, return_goal_lower_bounds
from control.motors import HardwareError
from control.config import validate_config
from virtual_drive import VirtualDrive, phase
from test_free_timed import settings, targets
import test_bounded_push as fixtures

def state(t, upper=164., lower=4., speed=0.):
    return dict(q_rad=list(map(math.radians,[upper,lower])),speed_rad_s=[math.radians(speed),0.],
                stamp_monotonic_s=t,axis_stamp_monotonic_s=[t,t],speed_stamp_monotonic_s=[t,t])

class ReturnAcceptanceTests(unittest.TestCase):
    goal=list(map(math.radians,[180.,4.]))
    def qualifier(self):return GoalQualification(.3,1.,1.,.3,lower_bounds_deg=[160.,None])

    def test_164_accepts_only_after_original_stable_and_hold(self):
        g=self.qualifier()
        self.assertFalse(g.observe(state(0.),self.goal,True))
        self.assertFalse(g.observe(state(.3),self.goal,True));self.assertTrue(g.armed)
        self.assertFalse(g.observe(state(1.29),self.goal,True))
        self.assertTrue(g.observe(state(1.3),self.goal,True))

    def test_strict_boundary_and_lower_and_speed_still_required(self):
        for u,l,v in [(160.,4.,0.),(159.99,4.,0.),(164.,4.31,0.),(164.,4.,5.)]:
            with self.subTest(upper=u,lower=l,speed=v):
                g=self.qualifier()
                for t in [0.,.3,1.3,2.]:self.assertFalse(g.observe(state(t,u,l,v),self.goal,True))
                self.assertFalse(g.armed)

    def test_no_acceptance_before_endpoint_or_using_stale_feedback(self):
        g=self.qualifier()
        for t in [0.,1.,2.]:self.assertFalse(g.observe(state(t),self.goal,False))
        self.assertFalse(g.observe(state(3.),self.goal,True))
        for _ in range(5):self.assertFalse(g.observe(state(3.),self.goal,True))
        self.assertFalse(g.armed)

    def test_boundary_violation_during_hold_fails(self):
        g=self.qualifier();g.observe(state(0.),self.goal,True);g.observe(state(.3),self.goal,True)
        self.assertFalse(g.observe(state(.5,160.),self.goal,True))
        self.assertEqual(g.failure,'motion_goal_hold_hard_limit')

    def test_other_routes_keep_normal_upper_tolerance(self):
        for name,route in [('free',None),('push',None),('free','observation_retreat')]:
            p={'phase':name,'route':route};self.assertIsNone(return_goal_lower_bounds(p))
            p['return_upper_min_deg']=160.
            with self.assertRaises(ValueError):return_goal_lower_bounds(p)
        g=GoalQualification(.3,1.,1.)
        for t in [0.,.3,1.3]:self.assertFalse(g.observe(state(t),self.goal,True))

    def drive(self,bound=164.):
        cfg=settings();cfg['motors'][0].update(offset_deg=-3.,limits_deg=[90.,182.])
        cfg['motors'][1].update(offset_deg=2.25,limits_deg=[-1.,90.])
        def clamp(d):
            if d.q[0]>=bound-3.:d.q[0]=bound-3.;d.velocity[0]=0.
        return VirtualDrive(model_q=(159.,0.),settings=cfg,disturbance=clamp)

    def test_real_executor_sends_complete_180_with_offsets_then_accepts_164(self):
        d=self.drive();r=d.run(phase([[159.,0.],[180.,4.]],times=[0.,3.],route='return_home',return_upper_min_deg=160.))
        self.assertEqual(r['status'],'completed');self.assertEqual(targets(d)[-1],177.)
        self.assertEqual(targets(d,1)[-1],6.25)
        ex=r['motion_diagnostics']['execution'];self.assertEqual(ex['waypoints_dispatched'],105)
        self.assertGreaterEqual(r['elapsed_s']-ex['dispatch_completed_elapsed_s'],.5)

    def test_real_executor_does_not_accept_exactly_160(self):
        d=self.drive(160.)
        with self.assertRaises(HardwareError) as c:d.run(phase([[159.,0.],[180.,4.]],times=[0.,3.],route='return_home',return_upper_min_deg=160.))
        self.assertEqual(str(c.exception),'motion_goal_settle_timeout');self.assertEqual(targets(d)[-1],177.)

    def test_task_marks_only_return_phase(self):
        runner=fixtures.BoundedPushTests().runner([100.]);runner.return_q_deg=[180.,4.];runner.return_upper_min_deg=160.
        with patch('control.tasks.time.sleep'):r=runner.run({'intent':'push_to_end','color':'蓝'})
        self.assertEqual(r['status'],'completed')
        marked=[p for p in runner.motors.moves if 'return_upper_min_deg' in p]
        self.assertEqual(len(marked),1);self.assertEqual(marked[0]['route'],'return_home')
        self.assertEqual(r['return_motion']['upper_min_deg_exclusive'],160.)

    def test_invalid_threshold_configuration(self):
        runner=fixtures.BoundedPushTests().runner([])
        for value in [True,float('nan'),180.,181.]:
            c=copy.deepcopy(runner.config);c.setdefault('planning',{}).update(return_q_deg=[180.,4.],return_upper_min_deg=value)
            with self.assertRaises(ValueError):validate_config(c)
