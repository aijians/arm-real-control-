"""End-goal acceptance: fixed model end, arrival window, reliable-only verdicts and force pass.

No camera, serial or arm movement: vision, motors and planner are scripted fakes.
"""
import copy
import json
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from control.tasks import TaskRunner

ROOT = Path(__file__).resolve().parents[1]
LIMITS = {"蓝": [-46.56, -2.06], "橙": [-31.06, 13.44]}
GOAL = -2.06


class Arm:
    def slider_center(self, q, direction):
        return [q[0], 155.]


class Vision:
    """Each read pops the next scripted frame: (blue_x, orange_x, blue_reliable)."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.reads = 0

    def check_health(self):
        pass

    def read(self, after_monotonic_s=None):
        self.reads += 1
        blue, orange, reliable = self.frames.pop(0) if len(self.frames) > 1 else self.frames[0]
        return {"sliders_x_mm": {"蓝": blue, "橙": orange},
                "slider_limits_x_mm": copy.deepcopy(LIMITS),
                "reliability": {"蓝": {"reliable": reliable,
                                      "reasons": [] if reliable else ["rod_width_inconsistent_with_model"]},
                                "橙": {"reliable": True, "reasons": []}},
                "stamp_monotonic_s": max(time.monotonic(), (after_monotonic_s or 0.) + 1e-6)}


class Motors:
    """q[0] is the model push-head X. Each push phase pops (model_x, error_or_None)."""

    def __init__(self, pushes):
        self.pushes = list(pushes)
        self.q = [-50., 0.]
        self.moves = []

    def halt(self):
        return {"stopped_confirmed": True}

    def read_state(self):
        return {"q_rad": list(self.q)}

    def move(self, phase, health_check=None):
        self.moves.append(copy.deepcopy(phase))
        if phase["phase"] in ("push", "push_correction"):
            model_x, error = self.pushes.pop(0)
            self.q[0] = model_x
            if error:
                raise RuntimeError(error)
        return {"status": "completed", "phase": phase["phase"]}


class Planner:
    def __init__(self):
        self.arm = Arm()
        self.corrections = []

    def precontact(self, *args):
        return [-50., 0.]

    def free(self, *args, **kwargs):
        return {"phase": "free"}

    def observation_retreat(self, q, *args, **kwargs):
        return {"phase": "free", "route": "observation_retreat", "points": [{"q_rad": q}]}

    def observation_pose(self, *args):
        return {"rail_clearance_mm": 20.}

    def push(self, q, color, goal, sliders, limits):
        return {"phase": "push", "direction": "+x", "task_target_x_mm": goal}

    def correction(self, q, color, goal, direction, sliders, limits, distance):
        phase = {"phase": "push_correction", "direction": direction, "extension_mm": distance,
                 "reference_start_mm": [q[0], 155.], "reference_target_mm": [q[0] + distance, 155.]}
        self.corrections.append(phase)
        return phase

    def retry_contact(self, phase, q, sliders):
        return copy.deepcopy(phase)

    def reenter_correction(self, phase, q, sliders):
        return copy.deepcopy(phase)


class EndpointAcceptanceTests(unittest.TestCase):
    def runner(self, frames, pushes, force=False):
        config = json.loads((ROOT / "config/robot.json").read_text())
        config["verification"].update(samples=3, interval_s=.001, force_pass_unreliable=force)
        config["verification"]["aggregation"] = "all"
        config["push_correction"] = {"single_recovery": False}
        vision, motors, planner = Vision(frames), Motors(pushes), Planner()
        runner = TaskRunner(planner, config, motors, vision)
        runner._withdraw = lambda *_: None
        runner.legs({"intent": "push_to_end", "color": "蓝"}, {"蓝": -47.9, "橙": 11.75}, LIMITS)
        runner.reference_positions = {"橙": 11.75}
        return runner

    def run_leg(self, runner):
        with patch("control.tasks.time.sleep"):
            return runner._run_leg("蓝", GOAL)

    def pushes(self, runner):
        return [m for m in runner.motors.moves if m["phase"] in ("push", "push_correction")]

    def test_reading_inside_end_window_passes_without_correction(self):
        # Ground run 8ddc48b7: blue read −1.72 after the push; this is now within [−4.56, −1.06].
        runner = self.runner([(-47.9, 11.75, True), (-1.72, 11.45, True)], [(-2.06, None)])
        record, state = self.run_leg(runner)
        self.assertEqual(record["status"], "completed", record.get("reason"))
        self.assertEqual(len(self.pushes(runner)), 1)
        verification = record["attempts"][0]["verification"]
        self.assertEqual(verification["window_mm"], [2.5, 1.0])
        self.assertFalse(verification["forced"])

    def test_reading_beyond_end_is_unreliable_not_overshoot(self):
        runner = self.runner([(-47.9, 11.75, True), (-0.5, 11.75, True)], [(-2.06, None)])
        record, _ = self.run_leg(runner)
        self.assertEqual(record["reason"], "verification_unreliable")
        verification = record["attempts"][0]["stage_attempts"]["push"][0]
        self.assertEqual(len(self.pushes(runner)), 1)          # never pushes again
        self.assertEqual(runner.vision.reads, 1 + 3 + 3)       # observe + 3 samples + 3 extra
        self.assertIn("reading_beyond_target_end_window", verification["details"]["reasons"])

    def test_force_pass_switch_accepts_unreliable_and_continues(self):
        runner = self.runner([(-47.9, 11.75, True), (-0.5, 11.75, True)], [(-2.06, None)], force=True)
        record, state = self.run_leg(runner)
        self.assertEqual(record["status"], "completed", record.get("reason"))
        verification = record["attempts"][0]["verification"]
        self.assertTrue(verification["forced"])
        self.assertEqual(state["sliders_x_mm"]["蓝"], GOAL)
        self.assertEqual(state["forced_from_x_mm"], -0.5)

    def test_unreliable_frames_are_replaced_by_extra_samples(self):
        frames = [(-47.9, 11.75, True), (-2.4, 11.75, False), (-2.4, 11.75, False),
                  (-2.3, 11.75, True), (-2.3, 11.75, True), (-2.3, 11.75, True)]
        runner = self.runner(frames, [(-2.06, None)])
        record, _ = self.run_leg(runner)
        self.assertEqual(record["status"], "completed", record.get("reason"))
        verification = record["attempts"][0]["verification"]
        self.assertEqual(verification["reliable_samples"], 3)
        self.assertEqual(len(verification["samples"]), 5)

    def test_stationary_orange_drift_makes_frame_unreliable(self):
        runner = self.runner([(-47.9, 11.75, True), (-2.3, 13.0, True)], [(-2.06, None)])
        record, _ = self.run_leg(runner)
        self.assertEqual(record["reason"], "verification_unreliable")
        reasons = record["attempts"][0]["stage_attempts"]["push"][0]["details"]["reasons"]
        self.assertTrue(any(r.startswith("stationary_slider_drift:橙") for r in reasons))

    def test_reliable_under_push_corrects_but_model_head_never_passes_end(self):
        frames = [(-47.9, 11.75, True)] + [(-5.2, 11.75, True)] * 3 + [(-3.0, 11.75, True)]
        runner = self.runner(frames, [(-3.0, None), (-2.06, None)])
        record, _ = self.run_leg(runner)
        self.assertEqual(record["status"], "completed", record.get("reason"))
        self.assertEqual(len(runner.planner.corrections), 1)
        self.assertAlmostEqual(runner.planner.corrections[0]["extension_mm"], 0.94)
        self.assertLessEqual(runner.planner.corrections[0]["reference_target_mm"][0], GOAL + 1e-9)

    def test_no_correction_when_model_head_is_already_at_end(self):
        frames = [(-47.9, 11.75, True), (-5.2, 11.75, True)]
        runner = self.runner(frames, [(-2.06, None)])
        record, _ = self.run_leg(runner)
        self.assertEqual(record["reason"], "push_correction_model_target_reached")
        self.assertEqual(runner.planner.corrections, [])

    def test_settle_timeout_inside_arrival_window_is_absorbed(self):
        runner = self.runner([(-47.9, 11.75, True), (-2.6, 11.75, True)],
                             [(-3.9, "motion_goal_settle_timeout")])
        record, _ = self.run_leg(runner)
        self.assertEqual(record["status"], "completed", record.get("reason"))
        arrival = record["attempts"][0]["arrival"]
        self.assertTrue(arrival["within_window"])
        self.assertEqual(arrival["absorbed_motor_error"], "motion_goal_settle_timeout")

    def test_settle_timeout_outside_arrival_window_is_not_absorbed(self):
        runner = self.runner([(-47.9, 11.75, True), (-5.0, 11.75, True)],
                             [(-5.0, "motion_goal_settle_timeout")] * 4)
        record, _ = self.run_leg(runner)
        self.assertEqual(record["reason"], "motion_goal_settle_timeout")
        first = record["attempts"][0]["stage_attempts"]["push"][0]
        self.assertFalse(first["arrival"]["within_window"])

    def test_interior_goal_keeps_symmetric_tolerance_and_reports_overshoot(self):
        runner = self.runner([(-47.9, 11.75, True), (-9.0, 11.75, True)], [(-10.0, None)])
        with patch("control.tasks.time.sleep"):
            record, _ = runner._run_leg("蓝", -10.)
        self.assertEqual(record["reason"], "slider_overshot_target")


if __name__ == "__main__":
    unittest.main()
