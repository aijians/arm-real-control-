"""Route spacing and real Move/A6 regressions using a virtual serial drive."""
import copy
import math
import unittest
from unittest.mock import patch

from control.motors import HardwareError, MotorPair
from control.segments import phase_profile, resample_route, segment_settings, stream_speed_deg_s
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
        """Targets are nominal nodes, steps bounded, and each A6 speed field is
        the streamed field for that node (never above the phase cap)."""
        pts=planned['points']
        nominal = [{round(d.pair._motor_deg(m, math.degrees(p['q_rad'][i]))*100)
                    for p in pts} for i,m in enumerate(d.pair.motors)]
        profile=phase_profile(
            d.pair.segments, planned['phase'], retry_after_oscillation=planned.get('free_retry_after_oscillation', False))
        fields=[{} for _ in d.pair.motors]
        for k in range(1,len(pts)):
            for i,m in enumerate(d.pair.motors):
                key=round(d.pair._motor_deg(m, math.degrees(pts[k]['q_rad'][i]))*100)
                inc=abs(math.degrees(pts[k]['q_rad'][i]-pts[k-1]['q_rad'][i]))*m['ratio']
                fields[i].setdefault(key,set()).add(round(stream_speed_deg_s(profile,inc,d.pair.config['command_rate_hz'])*100))
        previous=[round(d.pair._motor_deg(m,math.degrees(planned['points'][0]['q_rad'][i]))*100)
                  for i,m in enumerate(d.pair.motors)]
        for _, i, _, w in d.a6():
            target=int.from_bytes(w[6:9],'little')
            field=int.from_bytes(w[9:13],'little')
            self.assertLessEqual(field,round(cap*100))
            self.assertIn(target,nominal[i])
            self.assertIn(field,fields[i].get(target,set()))
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

    def test_tiny_final_residual_does_not_trip_no_progress_while_holding(self):
        # The last node is 0.01 deg past the previous one: below the 0.015 deg
        # direction floor, never sendable, yet the previous write is already
        # in the endpoint band (c7 retry routes end like this).
        pts=[[169.,20.],[168.9,20.],[168.8,20.],[168.79,20.]]
        # Production holds 3 s, longer than the 2 s segment no-progress window.
        p=phase(pts,times=[0.,.1,.2,.3]);d=VirtualDrive(model_q=pts[0],settings=settings(goal_hold_s=3.,no_progress_s=2.))
        result=d.run(p)
        self.assertEqual(result['status'],'completed')
        self.assertEqual(d.unsafe_directions,[])
        self.assertTrue(result['motion_diagnostics']['execution']['active_command_in_acceptance_band'][0])

    def test_same_direction_nodes_stream_without_stopping_at_each_node(self):
        pts=[[169.-i*.1,20.] for i in range(31)]
        p=phase(pts,times=[i*.02 for i in range(31)]);d=VirtualDrive(model_q=pts[0],settings=settings())
        self.assertEqual(d.run(p)['status'],'completed')
        sent=[at for at,i,_,_ in d.a6() if i==0]
        self.assertEqual(len(sent),30)
        intervals=sorted(b-a for a,b in zip(sent,sent[1:]))
        # Previously each node waited for >=0.10 s of confirmed stillness (~0.17 s).
        self.assertLess(intervals[len(intervals)//2],.08)
        self.assertEqual({w[5] for _,i,_,w in d.a6() if i==0},{1})
        self.assert_wires(d,p,10.,[.1,.2])

    def test_same_direction_send_still_respects_reported_speed_and_reversal_margin(self):
        def attempt(speed, target_deg):
            d=VirtualDrive(model_q=[168.95,20.],settings=settings());p=d.pair
            p._last_targets[0]=(16890,1,1000,0);p._last_target_sent_at[0]=d.now-.2
            d.velocity[0]=speed
            for i in (0,1):d.refresh(i,0x94);d.refresh(i,0x9C)
            d.now=round(d.now+.01,9)
            q=list(map(math.radians,[target_deg,20.]))
            with patch('control.motors.time.monotonic',lambda:d.now):
                return p._send_targets(q,nominal_q_rad=q,command_axes=[True,False],
                    axis_min_speed_deg_s=[10.,10.],speed_tiers=[0,0],command_speed_caps_deg_s=[10.,10.],
                    command_step_limit_deg=[.1,.2],segment_origin_q_rad=list(map(math.radians,[169.,20.])),
                    segment_started_at=d.now-2)
        # 0.15 deg ahead, moving at the 10 deg/s field: bounded travel is ~0.13 deg.
        self.assertEqual(attempt(-10.,168.8),1)
        # A reported 40 deg/s exceeds the field and widens the margin: skip.
        self.assertEqual(attempt(-40.,168.8),0)
        # A moving reversal keeps the global 60 deg/s budget: skip.
        self.assertEqual(attempt(-10.,169.1),0)

    def test_streaming_free_never_stops_mid_route(self):
        pts=[[169.-i*.1,20.] for i in range(61)]                    # 6 deg in 0.1 deg nodes
        p=phase(pts,times=[i*.01 for i in range(61)])
        trace=[]
        def watch(d):trace.append((d.elapsed,d.q[0],d.velocity[0],d.target[0]))
        d=VirtualDrive(model_q=pts[0],settings=settings(),disturbance=watch)
        self.assertEqual(d.run(p)['status'],'completed')
        self.assertEqual(d.unsafe_directions,[])
        moving=[(t,q,v) for t,q,v,_ in trace if 168.95>=q>=163.3]
        stops=[]; still_since=None
        for t,q,v in moving:
            if abs(v)<1e-9:
                still_since=t if still_since is None else still_since
            else:
                if still_since is not None:stops.append(t-still_since)
                still_since=None
        # Previously the arm stopped at every node for >= 0.10 s.
        self.assertLess(max(stops,default=0.),.06)
        lead=max(abs(target-q) for _,q,_,target in trace)
        self.assertLessEqual(lead,.5+.1+1e-6)                       # free lead + one node
        self.assert_wires(d,p,10.,[.1,.2])

    def test_push_keeps_advancing_when_load_holds_the_arm_behind_each_target(self):
        pts=[[159.-i*.2,20.] for i in range(11)]
        final=round(pts[-1][0],2)
        def load(d):
            # The contact holds the arm 0.36 deg short of every target except
            # the last (the 5d42f0cd stall: error 0.36 > node tolerance 0.3).
            if d.a6(0) and round(d.target[0],2)!=final and d.q[0]<d.target[0]+.36:
                d.q[0],d.velocity[0]=d.target[0]+.36,0.
        p=phase(pts,name='push');d=VirtualDrive(model_q=pts[0],settings=settings(),disturbance=load)
        self.assertEqual(d.run(p)['status'],'completed')
        self.assertEqual(d.unsafe_directions,[])
        self.assert_wires(d,p,60.,1.)

    def test_isolated_speed_readings_during_hold_do_not_restart_it(self):
        class Blippy(VirtualDrive):
            next_blip=None
            def refresh(self,axis,command):
                super().refresh(axis,command)
                if (command==0x9C and axis==0 and self.a6(0) and abs(self.q[0]-168.8)<1e-6
                        and (self.next_blip is None or self.elapsed>=self.next_blip)):
                    self.next_blip=self.elapsed+.4
                    self.pair._feedback[0].update(speed_deg_s=2.)
        pts=[[169.,20.],[168.9,20.],[168.8,20.]]
        p=phase(pts,times=[0.,.1,.2])
        d=Blippy(model_q=pts[0],settings=settings(goal_hold_s=1.,no_progress_s=2.))
        result=d.run(p)
        self.assertEqual(result['status'],'completed')
        e=result['motion_diagnostics']['execution']
        self.assertGreater(e['goal_speed_blips_tolerated'],0)
        blip=[x for x in e['goal_stability_events'] if x['kind']=='tolerated_speed_blip'][0]
        self.assertEqual(blip['speeds_deg_s'][0],2.)


if __name__ == '__main__':unittest.main()
