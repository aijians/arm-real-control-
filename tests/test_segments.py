"""Route spacing and real Move/A6 regressions using a virtual serial drive."""
import copy
import math
import unittest
from unittest.mock import patch

from control.motors import HardwareError, MotorPair
from control.segments import phase_profile, resample_route, segment_settings
from virtual_drive import VirtualDrive, config, phase


def settings(**extra):
    return config(max_speed_deg_s=60., segmented_motion={"enabled": True}, **extra)


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.h = settings()
        self.s = segment_settings(self.h)

    def route(self, points, name="free", **kwargs):
        path = [list(map(math.radians, q)) for q in points]
        return resample_route(path, [10.]*(len(path)-1), points, self.s, name,
                              self.h["motors"], kwargs.pop("chord_safe", lambda a, b: True), **kwargs)

    def test_free_bounds_both_model_and_quantized_steps_and_keeps_endpoint(self):
        path, _, sources, caps = self.route([[169.003, 20.],[165.117, 22.3]])
        self.assertEqual(path[-1], list(map(math.radians, [165.117, 22.3])))
        self.assertEqual(sources[-1], 1.)
        self.assertTrue(all(c == [.1, .2] for c in caps))
        self.assert_axis_steps(path, [.1, .2])
        # Upper-dominant route: the upper arm actually uses its 0.1 cap.
        self.assertAlmostEqual(max(abs(math.degrees(b[0]-a[0])) for a, b in zip(path, path[1:])), .1, places=6)

    def assert_axis_steps(self, path, caps):
        for a, b in zip(path, path[1:]):
            for axis, (x, y) in enumerate(zip(a, b)):
                self.assertLessEqual(abs(math.degrees(y-x)), caps[axis]+1e-8)
                self.assertLessEqual(abs(round(math.degrees(y)*100)-round(math.degrees(x)*100)),
                                     round(caps[axis]*100))

    def test_upper_stays_at_or_below_point_one_whichever_axis_dominates(self):
        # c7-like routes: long lower travel (first attempt) and short lower travel (retries).
        for start in ([171.54, 29.26], [170.84, 46.05], [173.11, 52.29]):
            with self.subTest(start=start):
                path, _, _, caps = self.route([start, [163.3, 53.1]])
                self.assert_axis_steps(path, [.1, .2])
                lower_travel = abs(53.1-start[1])
                if lower_travel > 2*abs(163.3-start[0]):
                    # Lower-dominant: the lower arm still reaches its own 0.2 cap.
                    self.assertAlmostEqual(max(abs(math.degrees(b[1]-a[1])) for a, b in zip(path, path[1:])),
                                           .2, places=6)

    def test_withdraw_and_return_use_the_same_axis_caps_as_free(self):
        for name in ('withdraw', 'return'):
            with self.subTest(name=name):
                path, _, _, caps = self.route([[169.,20.],[167.,21.]], name)
                self.assertTrue(all(c == [.1, .2] for c in caps))
                self.assert_axis_steps(path, [.1, .2])

    def test_push_keeps_gap_small_then_grows_every_three_nodes(self):
        path, lengths, _, caps = self.route([[169., 20.],[149., 20.]], "push", gap_mm=1.)
        progress = 0.
        first_grown = None
        for i, (length, cap) in enumerate(zip(lengths, caps)):
            if progress < 1.-1e-9:
                self.assertEqual(cap, .2)
            if cap > .2 and first_grown is None:
                first_grown = i
            progress += length
        self.assertIsNotNone(first_grown)
        self.assertEqual([round(x, 2) for x in caps[first_grown:first_grown+12]],
                         [.4]*3+[.6]*3+[.8]*3+[1.]*3)
        self.assertLessEqual(max(caps), 1.)

    def test_turning_point_and_corner_cannot_be_shortcut(self):
        path, _, _, _ = self.route([[169.,20.],[169.33,20.2],[168.8,20.2]])
        corner = list(map(math.radians,[169.33,20.2]))
        self.assertIn(corner, path)

    def test_geometry_can_shorten_a_scheduled_step(self):
        path, _, _, _ = self.route([[169.,20.],[168.,20.]],
            chord_safe=lambda a,b: abs(math.degrees(b[0]-a[0])) <= .12)
        self.assertTrue(all(abs(math.degrees(b[0]-a[0])) <= .12 for a,b in zip(path,path[1:])))

    def test_invalid_profiles_fail_before_hardware(self):
        for update in ({"free_max_step_deg":.3},{"push_max_step_deg":2.},
                       {"free_upper_max_step_deg":.3},{"free_upper_max_step_deg":.04},
                       {"free_max_step_deg":.08},{"free_upper_max_step_deg":True},
                       {"push_points_per_step":True},{"push_speed_deg_s":61.},
                       {"enabled":1},{"push_initial_step_deg":float('nan')},
                       {"unknown":1}):
            h=copy.deepcopy(self.h);h['segmented_motion'].update(update)
            with self.subTest(update=update),self.assertRaises(ValueError):segment_settings(h)

    def test_withdraw_uses_free_profile_and_correction_uses_push_profile(self):
        self.assertEqual(phase_profile(self.s,'withdraw')['speed_deg_s'],10.)
        self.assertEqual(phase_profile(self.s,'push_correction')['speed_deg_s'],60.)
        for name in ('free','withdraw','return'):
            self.assertEqual(phase_profile(self.s,name)['axis_max_step_deg'],[.1,.2])
        self.assertEqual(phase_profile(self.s,'push')['axis_max_step_deg'],[1.,1.])


class ExecutionTests(unittest.TestCase):
    def assert_wires(self, d, planned, cap, step):
        nominal = [{round(d.pair._motor_deg(m, math.degrees(p['q_rad'][i]))*100)
                    for p in planned['points']} for i,m in enumerate(d.pair.motors)]
        previous=[round(d.pair._motor_deg(m,math.degrees(planned['points'][0]['q_rad'][i]))*100)
                  for i,m in enumerate(d.pair.motors)]
        for _, i, _, w in d.a6():
            target=int.from_bytes(w[6:9],'little')
            self.assertEqual(int.from_bytes(w[9:13],'little'),round(cap*100))
            self.assertIn(target,nominal[i])
            axis_step=step[i] if isinstance(step,list) else step
            self.assertLessEqual(abs(target-previous[i]),math.floor(axis_step*d.pair.motors[i]['ratio']*100+1e-7))
            previous[i]=target
        self.assertEqual(d.pair._settling_bias_centi,[0,0])
        self.assertEqual(d.unsafe_directions,[])

    def test_free_issues_small_nodes_at_ten_without_legacy_coalescing(self):
        pts=[[169.-i*.1,20.+i*.025] for i in range(11)]
        p=phase(pts);d=VirtualDrive(model_q=pts[0],settings=settings())
        result=d.run(p)
        self.assertEqual(result['status'],'completed')
        self.assertEqual(len(d.a6(0)),10)
        self.assert_wires(d,p,10.,[.1,.2])

    def test_push_uses_sixty_with_no_bias_even_when_old_recovery_is_enabled(self):
        pts=[[159.,20.],[158.8,20.],[158.6,20.],[158.2,20.],[157.6,20.],[156.8,20.],[155.8,20.]]
        p=phase(pts,name='push',cartesian_compensation={'legacy':'must be ignored'})
        d=VirtualDrive(model_q=pts[0],settings=settings(speed_recovery={'enabled':True},motion_recovery={'enabled':True}))
        result=d.run(p)
        self.assert_wires(d,p,60.,1.)
        rec=result['motion_diagnostics']['execution']['speed_recovery']
        self.assertEqual(rec['upshifts'],[0,0]);self.assertEqual(rec['speed_tier'],[0,0])
        self.assertIsNone(rec['recovery_started_at'])

    def test_direction_reversal_at_reached_node_does_not_starve(self):
        pts=[[169.,20.],[169.1,20.],[169.2,20.],[169.1,20.],[169.,20.]]
        p=phase(pts);d=VirtualDrive(model_q=pts[0],settings=settings())
        self.assertEqual(d.run(p)['status'],'completed')
        self.assert_wires(d,p,10.,[.1,.2])

    def test_stuck_path_fails_without_speed_or_angle_recovery(self):
        pts=[[169.-i*.1,20.] for i in range(11)]
        p=phase(pts);d=VirtualDrive(model_q=pts[0],settings=settings(),immobile=True)
        with self.assertRaises(HardwareError) as caught:d.run(p)
        self.assertEqual(caught.exception.reason,'motion_no_progress')
        self.assertLess(d.elapsed,6.)
        self.assertTrue(d.halts)
        self.assert_wires(d,p,10.,[.1,.2])

    def test_oversized_or_excessively_fast_free_phase_rejected_before_io(self):
        p=MotorPair(settings())
        for bad in (phase([[169.,20.],[168.7,20.]]),
                    phase([[169.,20.],[168.85,20.]]),      # upper 0.15 > 0.1
                    phase([[169.,20.],[169.,20.25]]),      # lower 0.25 > 0.2
                    phase([[169.,20.],[168.9,20.]],times=[0.,.005])):
            with self.assertRaises(ValueError):p.move(bad)
        # Lower may still step 0.2 while upper steps 0.1.
        p._validate_phase(phase([[169.,20.],[168.9,20.2]]))

    def test_skipped_secondary_axis_cannot_produce_oversized_wire_jump(self):
        d=VirtualDrive(model_q=[168.8,20.],settings=settings());p=d.pair
        p._last_targets[0]=(16900,1,1000,0);p._last_target_sent_at[0]=d.now-1
        for i in (0,1):d.refresh(i,0x94);d.refresh(i,0x9C)
        q=list(map(math.radians,[168.6,20.]))
        with patch('control.motors.time.monotonic',lambda:d.now):
            self.assertEqual(p._send_targets(q,nominal_q_rad=q,command_axes=[True,False],
                command_step_limit_deg=.2,segment_origin_q_rad=list(map(math.radians,[169.,20.])),
                segment_started_at=d.now-2),0)
        self.assertEqual(d.a6(),[])

    def test_new_phase_anchors_first_step_at_measured_start(self):
        pts=[[169.,20.],[168.9,20.],[168.8,20.]]
        p=phase(pts);d=VirtualDrive(model_q=pts[0],settings=settings())
        # Last phase accepted an endpoint 0.15 degrees from its command.
        d.pair._last_targets[0]=(16915,0,6000,0)
        d.pair._last_target_sent_at[0]=d.now-1
        self.assertEqual(d.run(p)['status'],'completed')
        self.assert_wires(d,p,10.,[.1,.2])

    def test_feedback_failure_and_health_failure_still_halt(self):
        p=phase([[169.,20.],[168.9,20.]])
        d=VirtualDrive(model_q=[169.,20.],settings=settings())
        def unhealthy():raise RuntimeError('camera offline')
        with self.assertRaises(HardwareError):d.run(p,health_check=unhealthy)
        self.assertTrue(d.halts);self.assertEqual(d.a6(),[])

    def test_nonunit_mapping_preserves_model_and_wire_step_bounds(self):
        cfg=settings();cfg['motors'][0].update(sign=-1,offset_deg=355.,ratio=2.,limits_deg=[2.,175.])
        pts=[[169.-i*.1,20.] for i in range(6)]
        p=phase(pts);d=VirtualDrive(model_q=pts[0],settings=cfg)
        self.assertEqual(d.run(p)['status'],'completed')
        self.assert_wires(d,p,10.,[.1,.2])


if __name__ == '__main__':unittest.main()
