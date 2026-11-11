"""Independent stage retries, refreshed planning and fixed contact endpoints."""
import copy
import math
import unittest
from unittest.mock import Mock, patch

from control.motors import HardwareError
from control.planner import PlanError
import test_bounded_push as fixtures
from control.planner import Planner


class StageRetryTests(unittest.TestCase):
    def runner(self, outcomes):
        return fixtures.BoundedPushTests().runner(outcomes)

    def test_free_and_push_each_get_three_retries(self):
        runner = self.runner([(-2., "motion_goal_settle_timeout")] * 3 + [0.])
        original = runner.motors.move
        free_calls = []
        def move(phase, **kwargs):
            if phase["phase"] == "free":
                free_calls.append(phase)
                if len(free_calls) < 4:
                    runner.motors.q = [-10. + len(free_calls), 0.]
                    raise HardwareError("motion_goal_settle_timeout", {"controller": "free_timed"})
            return original(phase, **kwargs)
        runner.motors.move = move
        with patch("control.tasks.time.sleep"):
            record, _ = runner._run_leg("蓝", 0.)
        self.assertEqual(record["status"], "completed")
        stages = record["attempts"][0]["stage_attempts"]
        self.assertEqual([a["attempt"] for a in stages["free"]], [0, 1, 2, 3])
        self.assertEqual([a["attempt"] for a in stages["push"]], [0, 1, 2, 3])
        self.assertEqual(stages["free"][3]["start_state"]["q_rad"], [-7., 0.])
        self.assertEqual(len(free_calls), 7)

    def test_push_retry_reobserves_and_runs_free_before_push(self):
        runner = self.runner([(-2., "motion_goal_settle_timeout"), 0.])
        precontact = runner.planner.precontact = Mock(return_value=[-10., 0.])
        observations = runner._snapshot = Mock(wraps=runner._snapshot)
        original = runner.motors.move
        sequence = []
        def move(phase, **kwargs):
            sequence.append(phase["phase"])
            if phase["phase"] == "push" and sequence.count("push") == 2:
                # Reproduce the reported contact guard: retry is only valid
                # after a new Free, rather than a repeated direct push.
                self.assertEqual(sequence[-2], "free")
                self.assertGreaterEqual(len(runner.vision.reads), 3)
            return original(phase, **kwargs)
        runner.motors.move = move
        with patch("control.tasks.time.sleep"):
            record, _ = runner._run_leg("蓝", 0.)
        self.assertEqual(record["status"], "completed")
        self.assertEqual(sequence, ["free", "push", "free", "push"])
        self.assertEqual(observations.call_count, 2)
        self.assertEqual(precontact.call_args_list[1].args[2]["蓝"], -2.)
        self.assertEqual(precontact.call_args_list[1].args[3], [0., 0.])
        retry = record["attempts"][0]["stage_attempts"]["push"][1]
        self.assertEqual(retry["reentry_free"]["motions"][0]["phase"], "free")

    def test_failed_reentry_consumes_push_retry_and_never_pushes_from_bad_pose(self):
        runner = self.runner([(-2., "motion_goal_settle_timeout"), 0.])
        original = runner.motors.move
        sequence = []
        def move(phase, **kwargs):
            sequence.append(phase["phase"])
            if sequence == ["free", "push", "free"]:
                raise HardwareError("motion_goal_settle_timeout")
            return original(phase, **kwargs)
        runner.motors.move = move
        with patch("control.tasks.time.sleep"):
            record, _ = runner._run_leg("蓝", 0.)
        self.assertEqual(record["status"], "completed")
        self.assertEqual(sequence, ["free", "push", "free", "free", "push"])
        retries = record["attempts"][0]["stage_attempts"]["push"]
        self.assertEqual([a["status"] for a in retries], ["failed", "failed", "completed"])

    def test_free_exhaustion_never_starts_push(self):
        runner = self.runner([])
        runner.motors.move = Mock(side_effect=HardwareError("motion_goal_hold_timeout"))
        record, _ = runner._run_leg("蓝", 0.)
        self.assertEqual(record["reason"], "motion_goal_hold_timeout")
        self.assertEqual(runner.motors.move.call_count, 4)
        self.assertNotIn("push", record["attempts"][0]["stage_attempts"])

    def test_unconfirmed_stop_and_requested_halt_do_not_resume(self):
        for reason in ("motion_halted", "motion_goal_hold_hard_limit", "joint_feedback_out_of_limits"):
            runner = self.runner([])
            runner.motors.move = Mock(side_effect=HardwareError(reason))
            record, _ = runner._run_leg("蓝", 0.)
            self.assertEqual(record["reason"], reason)
            self.assertEqual(runner.motors.move.call_count, 1)
        runner = self.runner([])
        runner.motors.stop_confirmed = False
        runner.motors.move = Mock(side_effect=HardwareError("motion_goal_settle_timeout"))
        runner._run_leg("蓝", 0.)
        self.assertEqual(runner.motors.move.call_count, 1)

    def test_failed_withdraw_does_not_repeat_push(self):
        runner = self.runner([0.])
        # Undo the legacy fixture's withdrawal stub.
        del runner._withdraw
        runner.planner.withdraw = Mock(return_value={"phase": "withdraw"})
        original = runner.motors.move
        count = 0
        def move(phase, **kwargs):
            nonlocal count
            if phase["phase"] == "withdraw":
                count += 1
                if count < 4:
                    raise HardwareError("motion_goal_hold_timeout")
            result = original(phase, **kwargs)
            if phase["phase"] == "withdraw":
                runner.motors.stopped = True
            return result
        runner.motors.move = move
        record, _ = runner._run_leg("蓝", 0.)
        self.assertEqual(record["status"], "completed")
        self.assertEqual(count, 4)
        self.assertEqual(sum(p["phase"] == "push" for p in runner.motors.moves), 1)

    def test_return_has_own_budget_without_repeating_leg(self):
        runner = self.runner([100.])
        original = runner.motors.move
        frees = 0
        def move(phase, **kwargs):
            nonlocal frees
            if phase["phase"] == "free":
                frees += 1
                if 2 <= frees < 5:
                    raise HardwareError("motion_goal_settle_timeout")
            return original(phase, **kwargs)
        runner.motors.move = move
        with patch("control.tasks.time.sleep"):
            result = runner.run({"intent": "push_to_end", "color": "蓝"})
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(result["return_motion"]["stage_attempts"]["return"]), 4)
        self.assertEqual(sum(p["phase"] == "push" for p in runner.motors.moves), 1)

    def test_partial_correction_retries_original_endpoint(self):
        planner = Planner(fixtures.ROOT)
        q = planner.arm.inverse([-20., 155.], "+x")
        phase = planner.correction(q, "蓝", -20., "+x", {"蓝": -24., "橙": 15.}, [-60., -20.], 1.)
        for x in (-19.6, -19.2):
            phase = planner.retry_contact(phase, planner.arm.inverse([x, 155.], "+x"),
                                          {"蓝": -23.6, "橙": 15.})
            self.assertAlmostEqual(phase["reference_target_mm"][0], -19.)
            self.assertEqual(phase["extension_mm"], 1.)
            self.assertLessEqual(max(abs(math.degrees(b-a)) for p, n in
                zip(phase["points"], phase["points"][1:]) for a, b in zip(p["q_rad"], n["q_rad"])), .2 + 1e-9)
        with self.assertRaisesRegex(PlanError, "outside_correction_budget"):
            planner.retry_contact(phase, planner.arm.inverse([-18., 155.], "+x"),
                                  {"蓝": -24., "橙": 15.})

    def test_partial_withdraw_keeps_endpoint_and_rechecks_collision(self):
        planner = Planner(fixtures.ROOT)
        sliders = {"蓝": -21.5, "橙": 15.}
        q = planner.arm.inverse([-20., 155.], "+x")
        with patch.object(planner, "_checked_edges") as checked, patch.object(planner, "_clear"):
            phase = planner.withdraw(q, "蓝", "+x", sliders,
                                     from_reference=True, reference_extension_mm=2.)
            endpoint = copy.deepcopy(phase["points"][-1]["q_rad"])
            before = checked.call_count
            phase = planner.retry_contact(phase, planner.arm.inverse([-25., 155.], "+x"), sliders)
            self.assertEqual(phase["points"][-1]["q_rad"], endpoint)
            self.assertAlmostEqual(phase["reference_target_mm"][0],
                                   -20. - planner.offsets["+x"] - 2.)
            self.assertGreater(checked.call_count, before)


if __name__ == "__main__":
    unittest.main()
