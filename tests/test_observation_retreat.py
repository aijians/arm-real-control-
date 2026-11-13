"""Clear post-push images and bounded reentry, with no camera or motor IO."""
import copy
import math
import unittest
from unittest.mock import Mock, patch
import test_bounded_push as fixtures
from control.planner import Planner, PlanError
from control.scene import _box


class ObservationTaskTests(unittest.TestCase):
    def runner(self, outcomes):
        return fixtures.BoundedPushTests().runner(outcomes)

    def test_success_and_motor_timeout_clear_view_before_every_image(self):
        for outcomes in ([0.], [(-2., 'motion_goal_settle_timeout'), 0.]):
            runner = self.runner(outcomes)
            with patch('control.tasks.time.sleep'):
                record, _ = runner._run_leg('蓝', 0.)
            self.assertEqual(record['status'], 'completed')
            self.assertEqual(sum(x['phase']=='push' for x in runner.motors.moves),
                             sum(x.get('route')=='observation_retreat' for x in runner.motors.moves))
            self.assertFalse(runner.vision.occluded)
            for read in runner.vision.reads:
                self.assertGreater(read['stamp'], read['after'])

    def test_unsafe_retreat_blocks_images_and_repeated_push(self):
        runner = self.runner([(-2., 'motion_goal_settle_timeout')])
        runner.planner.observation_retreat = Mock(side_effect=PlanError('no_safe_observation_retreat'))
        record, _ = runner._run_leg('蓝', 0.)
        self.assertEqual(record['reason'], 'no_safe_observation_retreat')
        self.assertEqual(len(runner.vision.reads), 1)
        self.assertEqual(sum(p['phase']=='push' for p in runner.motors.moves), 1)

    def test_measured_clearance_failure_blocks_acceptance(self):
        runner = self.runner([0.])
        runner.planner.observation_pose = Mock(side_effect=PlanError('observation_pose_not_clear_of_rail'))
        record, _ = runner._run_leg('蓝', 0.)
        self.assertEqual(record['status'], 'failed')
        self.assertEqual(record['reason'], 'observation_pose_not_clear_of_rail')
        self.assertEqual(len(runner.vision.reads), 1)
        self.assertEqual(sum(p['phase']=='push' for p in runner.motors.moves), 1)

    def test_other_slider_must_also_be_visible_before_images(self):
        runner = self.runner([0.]);check=runner.planner.observation_pose
        def hidden(q,color,*args):
            if color=='橙':raise PlanError('observation_pose_occludes_slider_rail')
            return check(q,color,*args)
        runner.planner.observation_pose=hidden
        record, _ = runner._run_leg('蓝', 0.)
        self.assertEqual(record['reason'],'observation_pose_occludes_slider_rail')
        self.assertEqual(len(runner.vision.reads),1)

    def test_unconfirmed_stop_after_retreat_blocks_images(self):
        runner = self.runner([0.]);move = runner.motors.move
        def lose_stop(phase, **kw):
            result = move(phase, **kw)
            if phase.get('route')=='observation_retreat':runner.motors.stop_confirmed=False
            return result
        runner.motors.move = lose_stop
        record, _ = runner._run_leg('蓝', 0.)
        self.assertEqual(record['reason'], 'stop_not_confirmed')
        self.assertEqual(len(runner.vision.reads), 1)

    def test_corrections_reenter_without_changing_reference_budget(self):
        runner = self.runner([-2.4, -1.4, -.4])
        with patch('control.tasks.time.sleep'):record, _ = runner._run_leg('蓝', 0.)
        self.assertEqual(record['status'], 'completed')
        phases = [p.get('route', p['phase']) for p in runner.motors.moves]
        self.assertEqual(phases, ['free','push','observation_retreat','free','push_correction',
                                 'observation_retreat','free','push_correction','observation_retreat'])
        self.assertEqual([p['reference_target_mm'][0] for p in runner.planner.corrections], [1.,2.])
        self.assertTrue(all(p['extension_mm']==1. and p['task_target_x_mm']==0. for p in runner.planner.corrections))


class ObservationGeometryTests(unittest.TestCase):
    def setUp(self):
        self.p = Planner(fixtures.ROOT)
        self.q = [math.radians(153.04), math.radians(43.93)]
        self.sliders = {'蓝':-9.4404, '橙':9.9818}
        self.bounds = [-51.1138,-6.24559]

    def retreat(self):
        return self.p.observation_retreat(self.q,'蓝','+x',self.sliders,self.bounds)

    def test_reported_collision_escapes_then_reenters_with_checks_enabled(self):
        p=self.p;pre=p.precontact('蓝','+x',self.sliders,self.q)
        with self.assertRaisesRegex(PlanError,'forearm_5_30'):p.free(self.q,pre,self.sliders,direction='+x')
        limits={'蓝':self.bounds,'橙':[-35.6138,9.2544]}
        phase=p.observation_retreat(self.q,'蓝','+x',{'蓝':-46.2273,'橙':9.9818},self.bounds,
                                   all_slider_limits=limits)
        self.assertTrue(phase['initial_stem_overlaps'])
        end=phase['points'][-1]['q_rad'];clear=p.observation_pose(end,'蓝','+x',self.sliders,self.bounds)
        self.assertGreaterEqual(clear['rail_clearance_mm'],20.-1e-6)
        self.assertIsNone(p.scene.check(end,self.sliders))
        for color,bounds in limits.items():
            self.assertGreaterEqual(p.observation_pose(end,color,'+x',self.sliders,bounds)['minimum_projected_margin_mm'],0.)
        entry=p.free(end,pre,self.sliders,direction='+x')
        push=p.push(entry['points'][-1]['q_rad'],'蓝',self.bounds[1],self.sliders,self.bounds)
        self.assertEqual(push['target_x_mm'],self.bounds[1])
        for a,b in zip(phase['points'],phase['points'][1:]):
            self.assertLessEqual(max(abs(math.degrees(v-u)) for u,v in zip(a['q_rad'],b['q_rad'])),.2+1e-9)

    def test_both_directions_move_back_and_out(self):
        for d,c,x,sl,bounds in [('+x','蓝',-10.,{'蓝':-10.,'橙':15.},[-55.,0.]),
                              ('-x','橙',0.,{'蓝':-45.,'橙':0.},[-40.,10.])]:
            for y in (144.,155.):
                q=self.p.arm.inverse([x,y],d);phase=self.p.observation_retreat(q,c,d,sl,bounds)
                a,b=phase['reference_start_mm'],phase['reference_target_mm']
                self.assertLessEqual((1 if d=='+x' else -1)*(b[0]-a[0]),0.)
                self.assertLess(b[1],a[1])
                self.p.observation_pose(phase['points'][-1]['q_rad'],c,d,sl,bounds)

    def test_static_obstacles_remain_protected(self):
        with patch.object(self.p.scene,'check',return_value={'obstacle':'fixture_wall'}):
            with self.assertRaisesRegex(PlanError,'fixture_wall'):self.retreat()

    def test_high_arm_body_that_only_occludes_the_camera_is_rejected(self):
        q=self.retreat()['points'][-1]['q_rad']
        original=self.p.scene.transformed_bodies
        cover={'name':'high_cover','part':'forearm','z_mm':[100.,110.]}
        def bodies(pose):
            return original(pose)+[(cover,_box([-10.,155.],[5.,5.]))]
        with patch.object(self.p.scene,'transformed_bodies',side_effect=bodies):
            with self.assertRaisesRegex(PlanError,'occludes_slider_rail'):
                self.p.observation_pose(q,'蓝','+x',self.sliders,self.bounds)

    def test_other_slider_remains_protected(self):
        original=self.p.scene.check;seen=[]
        def check(q,positions=None,contact=None):
            seen.append(dict(positions or {}));return original(q,positions,contact)
        with patch.object(self.p.scene,'check',side_effect=check):self.retreat()
        self.assertTrue(seen)
        self.assertTrue(all(x=={'橙':self.sliders['橙']} for x in seen))

    def test_reverse_retreat_and_near_rail_measured_pose_are_rejected(self):
        path=[x['q_rad'] for x in self.retreat()['points']][::-1]
        with self.assertRaises(PlanError):self.p._checked_observation_edges(path,'蓝','+x',self.sliders,self.bounds,155.)
        with self.assertRaises(PlanError):self.p.observation_pose(self.q,'蓝','+x',self.sliders,self.bounds)

    def test_deepening_initial_contact_is_rejected(self):
        original=self.p.scene.slider_sweep_margins;calls=0
        def deepens(q,y,limits):
            nonlocal calls
            values=original(q,y,limits);calls+=1
            if calls>1:values[('forearm_5_30','forearm','stem')]=-100.
            return values
        with patch.object(self.p.scene,'slider_sweep_margins',side_effect=deepens):
            with self.assertRaisesRegex(PlanError,'deepens_contact'):self.retreat()

    def test_partial_correction_reentry_preserves_fixed_endpoint(self):
        p=self.p;q=p.arm.inverse([-20.,155.],'+x')
        phase=p.correction(q,'蓝',-20.,'+x',{'蓝':-24.,'橙':15.},[-60.,-20.],1.)
        fixed=copy.deepcopy(phase['points'][-1]['q_rad'])
        for x in [-23.6,-23.]:
            sl={'蓝':x,'橙':15.};pre=p.precontact('蓝','+x',sl,q)
            phase=p.reenter_correction(phase,pre,sl)
            self.assertEqual(phase['points'][-1]['q_rad'],fixed)
            self.assertEqual(phase['extension_mm'],1.)
            self.assertEqual(phase['task_target_x_mm'],-20.)
            self.assertAlmostEqual(phase['reference_target_mm'][0],-19.)


if __name__=='__main__':unittest.main()
