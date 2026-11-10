"""Timed Free baseline against the real MotorPair and virtual serial transport."""
import math
import os
import unittest
from unittest.mock import Mock, patch

from virtual_drive import VirtualDrive, config, phase, m
from control.tasks import TaskRunner
from control.protocol import (multi_turn_position_frame, parse_frame, make_frame,
                              state_from_reply)
from control.trajectory import timed_free_points


def settings(**extra):
    return config(free_timed_motion={"enabled": True, "speed_deg_s": 10.,
                                     "max_step_deg": .2, "period_s": .05}, **extra)


def targets(drive, axis=0):
    return [int.from_bytes(wire[5:13], "little", signed=True)/100 for _, _, _, wire in drive.a4(axis)]


class TimedFreeTests(unittest.TestCase):
    def test_exact_schedule_targets_and_fixed_speed_ignore_recovery_and_ack(self):
        d = VirtualDrive(settings=settings(speed_recovery={"enabled": True},
            motion_recovery={"enabled": True}), ack=False)
        with patch.object(d.pair, "_send_targets", side_effect=AssertionError("legacy sender")):
            result = d.run(phase([[159, 20], [158, 21], [157, 22]], times=[0, .3, .9]))
        self.assertEqual(targets(d), [158.8, 158.6, 158.4, 158.2, 158.,
                                      157.8, 157.6, 157.4, 157.2, 157.])
        self.assertEqual(targets(d, 1), [20.2, 20.4, 20.6, 20.8, 21.,
                                         21.2, 21.4, 21.6, 21.8, 22.])
        self.assertTrue(all(int.from_bytes(w[13:17], "little") == 1000 for *_, w in d.a4()))
        self.assertEqual(d.a6(), [])
        execution = result["motion_diagnostics"]["execution"]
        self.assertEqual(execution["controller"], "free_timed")
        self.assertEqual(execution["position_command"], "0xa4")
        self.assertEqual([e["planned_time_s"] for e in execution["schedule"]],
                         [i*.05 for i in range(1, 11)])
        self.assertTrue(all(0 <= e["lateness_s"] < .06 for e in execution["schedule"]))
        self.assertGreater(len(execution["feedback_samples"]), 5)
        self.assertGreater(result["elapsed_s"], .5+.4)
        self.assertEqual(d.unsafe_directions, [])

    def test_immobile_axes_do_not_gate_any_waypoint_or_trigger_midpath_timeout(self):
        d = VirtualDrive(immobile=True, settings=settings(no_progress_s=.1, tracking_abort_deg=3.5))
        path = [[159, 20], [157, 21], [155, 22], [153, 23]]
        with self.assertRaises(m.HardwareError) as caught:
            d.run(phase(path, times=[0, 1, 2, 3]))
        self.assertEqual(len(targets(d)), 30)
        self.assertEqual(len(targets(d, 1)), 30)
        self.assertEqual(targets(d)[-1], 153.)
        self.assertEqual(targets(d, 1)[-1], 23.)
        self.assertEqual(caught.exception.reason, "motion_goal_settle_timeout")
        execution = caught.exception.details["motion_diagnostics"]["execution"]
        self.assertEqual(execution["waypoints_dispatched"], 30)
        self.assertGreaterEqual(execution["dispatch_completed_elapsed_s"], 1.5)
        self.assertEqual(len(d.halts), 1)
        self.assertFalse(d.pair._moving)
        self.assertFalse(d.pair._move_lock.locked())

    def test_feedback_loss_stops_both_axes_and_releases_lock(self):
        d = VirtualDrive(settings=settings(), query_loss=lambda axis, cmd, t: t > .5)
        with self.assertRaises(m.HardwareError) as caught:
            d.run(phase([[159, 20], [156, 23], [153, 26]], times=[0, 1., 2.]))
        self.assertEqual(caught.exception.reason, "joint_feedback_stale")
        self.assertTrue(0 < len(targets(d)) < 30)
        self.assertEqual(len(d.halts), 1)
        self.assertFalse(d.pair._move_lock.locked())

    def test_small_target_sends_a4_without_direction_guard_or_extra_angle_query(self):
        d = VirtualDrive(immobile=True, settings=settings())
        result = d.run(phase([[159, 20], [159.04, 20]], times=[0, .2]))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(targets(d), [159.04])
        self.assertEqual(targets(d, 1), [20.])
        self.assertEqual(d.a6(), [])
        self.assertNotIn("direction", result["motion_diagnostics"]["last_sent"]["upper"])

    def test_reversal_and_nonunit_mapping(self):
        cfg = settings()
        cfg["motors"][0].update(sign=-1, ratio=1.5, offset_deg=300.)
        d = VirtualDrive(settings=cfg)
        d.run(phase([[159, 20], [157, 20], [159, 20]], times=[0, .5, 1.]))
        self.assertEqual(len(targets(d)), 20)
        self.assertEqual(targets(d)[9], 64.5)
        self.assertEqual(targets(d)[-1], 61.5)
        self.assertEqual(targets(d, 1), [20.]*20)
        self.assertEqual(d.a6(), [])
        self.assertEqual(d.unsafe_directions, [])

    def test_send_error_health_failure_and_external_halt_stop(self):
        for kind in ("write", "health", "halt"):
            with self.subTest(kind=kind):
                d = VirtualDrive(settings=settings(), fail_write=(
                    lambda axis, wire, drive: wire[1] == 0xA4) if kind == "write" else None)
                def health():
                    if d.elapsed > .2:
                        if kind == "health":
                            raise RuntimeError("camera failure")
                        if kind == "halt":
                            d.pair._halt_event.set()
                with self.assertRaises(m.HardwareError):
                    d.run(phase([[159, 20], [158, 21]], times=[0, .4]), health_check=health)
                self.assertEqual(len(d.halts), 1)
                self.assertFalse(d.pair._move_lock.locked())

    def test_overdue_nodes_stay_ordered_without_burst_or_drop(self):
        d = VirtualDrive(immobile=True, settings=settings())
        jumped = False
        def health():
            nonlocal jumped
            if d.elapsed > .2 and not jumped:
                jumped = True
                d.advance(.5)
        with self.assertRaises(m.HardwareError) as caught:
            d.run(phase([[159, 20], [158, 21], [157, 22], [156, 23]],
                        times=[0, .3, .4, .5]), health_check=health)
        self.assertEqual(caught.exception.reason, "motion_goal_settle_timeout")
        self.assertEqual(len(targets(d)), 15)
        self.assertEqual(targets(d)[-1], 156.)
        writes = [t for t, *_ in d.a4(0)]
        self.assertTrue(all(b-a >= .05-1e-8 for a, b in zip(writes, writes[1:])))

    def test_model_limit_violation_aborts_before_next_target(self):
        d = VirtualDrive(settings=settings())
        def disturb(drive):
            if drive.elapsed > .3:
                drive.q[0] = 181.
        d.disturbance = disturb
        with self.assertRaises(m.HardwareError) as caught:
            d.run(phase([[159, 20], [158, 21]], times=[0, .5]))
        self.assertEqual(caught.exception.reason, "joint_feedback_out_of_limits")
        self.assertTrue(0 < len(d.a4()) < 20)
        self.assertEqual(len(d.halts), 1)

    def test_other_phase_keeps_existing_executor(self):
        d = VirtualDrive(settings=settings())
        result = d.run(phase([[159, 20], [158, 21]], name="withdraw"))
        self.assertNotEqual(result["motion_diagnostics"]["execution"].get("controller"), "free_timed")

    def test_push_and_correction_send_a4_and_report_their_actual_phase(self):
        for name in ("push", "push_correction"):
            with self.subTest(phase=name):
                d = VirtualDrive(settings=settings())
                with patch.object(d.pair, "_send_targets", side_effect=AssertionError("legacy sender")):
                    result = d.run(phase([[159, 20], [158, 21]], name=name))
                self.assertEqual(result["phase"], name)
                self.assertEqual(result["motion_diagnostics"]["execution"]["controller"], name+"_timed")
                self.assertEqual(len(d.a4()), 10)
                self.assertEqual(d.a6(), [])
                self.assertEqual(targets(d)[-1], 158.)

    def test_push_tracking_error_does_not_gate_waypoints_but_final_goal_still_fails(self):
        for name in ("push", "push_correction"):
            with self.subTest(phase=name):
                d = VirtualDrive(immobile=True, settings=settings(no_progress_s=.1))
                with self.assertRaises(m.HardwareError) as caught:
                    d.run(phase([[159, 20], [153, 23]], name=name))
                self.assertEqual(caught.exception.reason, "motion_goal_settle_timeout")
                self.assertEqual(caught.exception.details["phase"], name)
                self.assertEqual(caught.exception.details["controller"], name+"_timed")
                self.assertEqual(len(d.a4()), 60)
                self.assertEqual(targets(d)[-1], 153.)
                self.assertEqual(targets(d, 1)[-1], 23.)
                self.assertEqual(len(d.halts), 1)

    def test_push_planner_uses_same_small_step_fixed_period_without_recovery_geometry(self):
        from pathlib import Path
        from control.config import load_config
        from control.planner import Planner
        root = Path(__file__).resolve().parents[1]
        cfg, _ = load_config(root)
        planner = Planner(root, cfg)
        path = [list(map(math.radians, q)) for q in [[159, 20], [158, 21]]]
        for name in ("push", "push_correction"):
            with self.subTest(phase=name):
                planned = planner._phase(name, path, [1.], 1., direction="+x")
                self.assertEqual(planned["time_parameterization"], "fixed_period_on_joint_polyline")
                self.assertIsNone(planned["motion_recovery_geometry"])
                self.assertEqual(len(planned["points"]), 6)
                self.assertEqual(planned["points"][-1]["q_rad"], path[-1])
                self.assertEqual(planned["points"][-1]["time_s"], .25)

    def test_noop_path_qualifies_without_motion(self):
        d = VirtualDrive(settings=settings())
        result = d.run(phase([[159, 20]], times=[0]))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(d.a4(), [])
        self.assertEqual(d.a6(), [])

    def test_a4_invalidates_old_a6_target_cache(self):
        d = VirtualDrive(settings=settings())
        d.pair._last_targets = [(15800, 1, 1000, 0), (2100, 0, 1000, 0)]
        d.run(phase([[159, 20], [158, 21]]))
        self.assertEqual(d.pair._last_targets, [None, None])
        self.assertEqual(targets(d)[-1], 158.)

    def test_subdivision_preserves_corners_endpoint_and_has_no_two_degree_speed_cap(self):
        original = [(0., list(map(math.radians, q)))
                    for q in [[159, 20], [158, 21], [158.6, 21]]]
        points = timed_free_points(original, .2, .05)
        self.assertEqual(len(points), 9)
        self.assertEqual(points[5][1], original[1][1])
        self.assertEqual(points[-1][1], original[-1][1])
        self.assertEqual(points, timed_free_points(points, .2, .05))
        for (ta, a), (tb, b) in zip(points, points[1:]):
            self.assertLessEqual(max(abs(math.degrees(y-x)) for x,y in zip(a,b)), .2+1e-9)
            self.assertAlmostEqual(tb-ta, .05)
        self.assertGreater(abs(math.degrees(points[1][1][0]-points[0][1][0]))/.05, 2.)

    def test_invalid_timing_settings(self):
        for key in ("max_step_deg", "period_s"):
            for value in (0, -1, True, float("inf"), float("nan")):
                cfg = settings()
                cfg["free_timed_motion"][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    m.MotorPair(cfg)

    def test_task_does_not_retry_free_baseline_failure(self):
        runner = TaskRunner.__new__(TaskRunner)
        runner.config = {"retry_limit": 3}
        runner.tolerance = .5
        runner.motors = Mock()
        runner.motors.halt.return_value = {"stopped_confirmed": True}
        runner._snapshot = Mock(return_value={"q_rad": [0, 0], "sliders_x_mm": {"蓝": 0}})
        runner._withdraw = Mock()
        runner.planner = Mock()
        runner._move = Mock(side_effect=m.HardwareError("motion_goal_settle_timeout",
                                                       {"controller": "free_timed"}))
        record, state = runner._run_leg("蓝", 10)
        self.assertIsNone(state)
        self.assertEqual(len(record["attempts"]), 1)
        self.assertEqual(runner._move.call_count, 1)


class MultiTurnProtocolTests(unittest.TestCase):
    def test_a4_golden_frame_and_signed_multi_turn_target(self):
        self.assertEqual(multi_turn_position_frame(1, 16000, 1000), bytes.fromhex(
            "3e a4 01 0c ef 80 3e 00 00 00 00 00 00 e8 03 00 00 a9"))
        for value in (-72000, 16000, 72000, -(1 << 63), (1 << 63)-1):
            reply = parse_frame(multi_turn_position_frame(2, value, 1000))
            self.assertEqual((reply.command, len(reply.payload)), (0xA4, 12))
            self.assertEqual(int.from_bytes(reply.payload[:8], "little", signed=True), value)
        for value in (-(1 << 63)-1, (1 << 63), True, 1.):
            with self.assertRaises(ValueError):
                multi_turn_position_frame(1, value, 1000)

    def test_real_receive_loop_accepts_a4_state_and_records_reply(self):
        pair = m.MotorPair(settings())
        pipes = [os.pipe(), os.pipe()]
        try:
            for port, (reader, _) in zip(pair.ports, pipes):
                port.fd = reader
            pair._opened = True
            pair._motion_trace = Mock()
            payload = bytes.fromhex("19 ff ff f8 ff 34 12")
            os.write(pipes[0][1], make_frame(0xA4, 1, payload))
            pair._pump(0)
            self.assertEqual(pair._feedback[0]["speed_deg_s"], -8)
            self.assertEqual(pair._feedback[0]["current_raw"], -1)
            pair._motion_trace.record_reply.assert_called_once()
            self.assertEqual(pair._motion_trace.record_reply.call_args[0][1].command, 0xA4)
            stamp = pair._feedback[0]["speed_stamp"]
            os.write(pipes[0][1], make_frame(0xA4, 1, bytes(6)))
            pair._pump(0)
            self.assertEqual(pair._feedback[0]["speed_stamp"], stamp)
        finally:
            for reader, writer in pipes:
                os.close(reader)
                os.close(writer)

    def test_a4_reply_has_state2_payload_and_rejects_bad_length(self):
        payload = bytes.fromhex("19 ff ff f8 ff 34 12")
        for cmd in (0xA4, 0xA6, 0x9C):
            self.assertEqual(state_from_reply(parse_frame(make_frame(cmd, 1, payload))),
                             {"temperature_c": 25, "current_raw": -1,
                              "speed_deg_s": -8, "encoder_raw": 0x1234})
        with self.assertRaises(ValueError):
            state_from_reply(parse_frame(make_frame(0xA4, 1, bytes(6))))


if __name__ == "__main__":
    unittest.main()
