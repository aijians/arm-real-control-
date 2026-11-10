"""Timed Free baseline against the real MotorPair and virtual serial transport."""
import unittest
from unittest.mock import Mock, patch

from virtual_drive import VirtualDrive, config, phase, m
from control.tasks import TaskRunner


def settings(**extra):
    return config(free_timed_motion={"enabled": True, "speed_deg_s": 10.}, **extra)


def targets(drive, axis=0):
    return [int.from_bytes(wire[6:9], "little")/100 for _, _, _, wire in drive.a6(axis)]


class TimedFreeTests(unittest.TestCase):
    def test_exact_schedule_targets_and_fixed_speed_ignore_recovery_and_ack(self):
        d = VirtualDrive(settings=settings(speed_recovery={"enabled": True},
            motion_recovery={"enabled": True}), ack=False)
        with patch.object(d.pair, "_send_targets", side_effect=AssertionError("legacy sender")):
            result = d.run(phase([[159, 20], [158, 21], [157, 22]], times=[0, .3, .9]))
        self.assertEqual(targets(d), [158, 157])
        self.assertEqual(targets(d, 1), [21, 22])
        self.assertTrue(all(int.from_bytes(w[9:13], "little") == 1000 for *_, w in d.a6()))
        execution = result["motion_diagnostics"]["execution"]
        self.assertEqual(execution["controller"], "free_timed")
        self.assertEqual([e["planned_time_s"] for e in execution["schedule"]], [.3, .9])
        self.assertTrue(all(0 <= e["lateness_s"] < .06 for e in execution["schedule"]))
        self.assertGreater(len(execution["feedback_samples"]), 5)
        self.assertGreater(result["elapsed_s"], .9+.4)
        self.assertEqual(d.unsafe_directions, [])

    def test_immobile_axes_do_not_gate_any_waypoint_or_trigger_midpath_timeout(self):
        d = VirtualDrive(immobile=True, settings=settings(no_progress_s=.1, tracking_abort_deg=3.5))
        path = [[159, 20], [157, 21], [155, 22], [153, 23]]
        with self.assertRaises(m.HardwareError) as caught:
            d.run(phase(path, times=[0, 1, 2, 3]))
        self.assertEqual(targets(d), [157, 155, 153])
        self.assertEqual(targets(d, 1), [21, 22, 23])
        self.assertEqual(caught.exception.reason, "motion_goal_settle_timeout")
        execution = caught.exception.details["motion_diagnostics"]["execution"]
        self.assertEqual(execution["waypoints_dispatched"], 3)
        self.assertGreaterEqual(execution["dispatch_completed_elapsed_s"], 3.)
        self.assertEqual(len(d.halts), 1)
        self.assertFalse(d.pair._moving)
        self.assertFalse(d.pair._move_lock.locked())

    def test_feedback_loss_stops_both_axes_and_releases_lock(self):
        d = VirtualDrive(settings=settings(), query_loss=lambda axis, cmd, t: t > .5)
        with self.assertRaises(m.HardwareError) as caught:
            d.run(phase([[159, 20], [158, 21], [157, 22]], times=[0, .3, 2.]))
        self.assertEqual(caught.exception.reason, "joint_feedback_stale")
        self.assertEqual(targets(d), [158])
        self.assertEqual(len(d.halts), 1)
        self.assertFalse(d.pair._move_lock.locked())

    def test_small_ambiguous_direction_aborts_without_wait_or_unsafe_write(self):
        d = VirtualDrive(immobile=True, settings=settings())
        with self.assertRaises(m.HardwareError) as caught:
            d.run(phase([[159, 20], [159.04, 20]], times=[0, .2]))
        self.assertEqual(caught.exception.reason, "free_timed_direction_ambiguous")
        self.assertEqual(d.a6(), [])
        self.assertLess(d.elapsed, .5)
        self.assertEqual(len(d.halts), 1)

    def test_reversal_and_nonunit_mapping(self):
        cfg = settings()
        cfg["motors"][0].update(sign=-1, ratio=1.5, offset_deg=300.)
        d = VirtualDrive(settings=cfg)
        d.run(phase([[159, 20], [157, 20], [159, 20]], times=[0, .5, 1.]))
        self.assertEqual(targets(d), [64.5, 61.5])
        self.assertEqual([w[5] for *_, w in d.a6(0)], [0, 1])
        self.assertEqual(d.a6(1), [])
        self.assertEqual(d.unsafe_directions, [])

    def test_send_error_health_failure_and_external_halt_stop(self):
        for kind in ("write", "health", "halt"):
            with self.subTest(kind=kind):
                d = VirtualDrive(settings=settings(), fail_write=(
                    lambda axis, wire, drive: wire[1] == 0xA6) if kind == "write" else None)
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
        self.assertEqual(targets(d), [158, 157, 156])
        writes = [t for t, *_ in d.a6(0)]
        self.assertTrue(all(b-a >= .02 for a, b in zip(writes, writes[1:])))

    def test_model_limit_violation_aborts_before_next_target(self):
        d = VirtualDrive(settings=settings())
        def disturb(drive):
            if drive.elapsed > .3:
                drive.q[0] = 181.
        d.disturbance = disturb
        with self.assertRaises(m.HardwareError) as caught:
            d.run(phase([[159, 20], [158, 21]], times=[0, .5]))
        self.assertEqual(caught.exception.reason, "joint_feedback_out_of_limits")
        self.assertEqual(d.a6(), [])
        self.assertEqual(len(d.halts), 1)

    def test_other_phase_keeps_existing_executor(self):
        d = VirtualDrive(settings=settings())
        result = d.run(phase([[159, 20], [158, 21]], name="push"))
        self.assertNotEqual(result["motion_diagnostics"]["execution"].get("controller"), "free_timed")

    def test_noop_path_qualifies_without_a6(self):
        d = VirtualDrive(settings=settings())
        result = d.run(phase([[159, 20]], times=[0]))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(d.a6(), [])

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


if __name__ == "__main__":
    unittest.main()
