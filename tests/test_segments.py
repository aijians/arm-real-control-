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
        self.assertTrue(all(c == .2 for c in caps))
        for a, b in zip(path, path[1:]):
            self.assertLessEqual(max(abs(math.degrees(y-x)) for x, y in zip(a, b)), .20000001)
            self.assertTrue(all(abs(round(math.degrees(y)*100)-round(math.degrees(x)*100)) <= 20
                                for x, y in zip(a, b)))

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
                       {"push_points_per_step":True},{"push_speed_deg_s":61.},
                       {"enabled":1},{"push_initial_step_deg":float('nan')},
                       {"unknown":1}):
            h=copy.deepcopy(self.h);h['segmented_motion'].update(update)
            with self.subTest(update=update),self.assertRaises(ValueError):segment_settings(h)

    def test_withdraw_uses_free_profile_and_correction_uses_push_profile(self):
        self.assertEqual(phase_profile(self.s,'withdraw')['speed_deg_s'],10.)
        self.assertEqual(phase_profile(self.s,'push_correction')['speed_deg_s'],60.)


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
            self.assertLessEqual(abs(target-previous[i]),math.floor(step*d.pair.motors[i]['ratio']*100+1e-7))
            previous[i]=target
        self.assertEqual(d.pair._settling_bias_centi,[0,0])
        self.assertEqual(d.unsafe_directions,[])

    def test_free_issues_small_nodes_at_ten_without_legacy_coalescing(self):
        pts=[[169.-i*.2,20.+i*.025] for i in range(11)]
        p=phase(pts);d=VirtualDrive(model_q=pts[0],settings=settings())
        result=d.run(p)
        self.assertEqual(result['status'],'completed')
        self.assertEqual(len(d.a6(0)),10)
        self.assert_wires(d,p,10.,.2)

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
        pts=[[169.,20.],[169.2,20.],[169.4,20.],[169.2,20.],[169.,20.]]
        p=phase(pts);d=VirtualDrive(model_q=pts[0],settings=settings())
        self.assertEqual(d.run(p)['status'],'completed')
        self.assert_wires(d,p,10.,.2)

    def test_stuck_path_fails_without_speed_or_angle_recovery(self):
        pts=[[169.-i*.2,20.] for i in range(11)]
        p=phase(pts);d=VirtualDrive(model_q=pts[0],settings=settings(),immobile=True)
        with self.assertRaises(HardwareError) as caught:d.run(p)
        self.assertEqual(caught.exception.reason,'motion_no_progress')
        self.assertLess(d.elapsed,6.)
        self.assertTrue(d.halts)
        self.assert_wires(d,p,10.,.2)

    def test_oversized_or_excessively_fast_free_phase_rejected_before_io(self):
        p=MotorPair(settings())
        for bad in (phase([[169.,20.],[168.7,20.]]),
                    phase([[169.,20.],[168.8,20.]],times=[0.,.01])):
            with self.assertRaises(ValueError):p.move(bad)

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
        pts=[[169.,20.],[168.8,20.],[168.6,20.]]
        p=phase(pts);d=VirtualDrive(model_q=pts[0],settings=settings())
        # Last phase accepted an endpoint 0.15 degrees from its command.
        d.pair._last_targets[0]=(16915,0,6000,0)
        d.pair._last_target_sent_at[0]=d.now-1
        self.assertEqual(d.run(p)['status'],'completed')
        self.assert_wires(d,p,10.,.2)

    def test_feedback_failure_and_health_failure_still_halt(self):
        p=phase([[169.,20.],[168.8,20.]])
        d=VirtualDrive(model_q=[169.,20.],settings=settings())
        def unhealthy():raise RuntimeError('camera offline')
        with self.assertRaises(HardwareError):d.run(p,health_check=unhealthy)
        self.assertTrue(d.halts);self.assertEqual(d.a6(),[])

    def test_nonunit_mapping_preserves_model_and_wire_step_bounds(self):
        cfg=settings();cfg['motors'][0].update(sign=-1,offset_deg=355.,ratio=2.,limits_deg=[2.,175.])
        pts=[[169.-i*.2,20.] for i in range(6)]
        p=phase(pts);d=VirtualDrive(model_q=pts[0],settings=cfg)
        self.assertEqual(d.run(p)['status'],'completed')
        self.assert_wires(d,p,10.,.2)


if __name__ == '__main__':unittest.main()
