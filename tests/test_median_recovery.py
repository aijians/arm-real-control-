"""Median decisions and exactly one certified recovery. No device IO."""
import copy
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import test_endpoint_acceptance as f
from control.config import validate_config
from control.planner import Planner, PlanError
from control.tasks import TaskRunner

ROOT = Path(__file__).resolve().parents[1]


class RecoveryPlanner(f.Planner):
    def recovery_push(self, q, color, goal, direction, sliders, limits):
        sign = 1 if direction == "+x" else -1
        phase = {"phase": "push_correction", "direction": direction,
                 "extension_mm": 1., "reference_start_mm": [q[0], 155.],
                 "reference_target_mm": [q[0] + sign, 155.], "single_recovery": True}
        self.corrections.append(phase)
        return phase


class MedianRecoveryTests(unittest.TestCase):
    def runner(self, frames, pushes=(), samples=5):
        config = json.loads((ROOT / "config/robot.json").read_text())
        config["verification"].update(samples=samples, interval_s=.001)
        runner = TaskRunner(RecoveryPlanner(), config, f.Motors(pushes), f.Vision(frames))
        runner.legs({"intent": "push_to_end", "color": "蓝"},
                    {"蓝": -47.9, "橙": 11.75}, f.LIMITS)
        runner.reference_positions = {"橙": 11.75}
        return runner

    def verify(self, runner):
        with patch("control.tasks.time.sleep"):
            return runner._verify("蓝", f.GOAL, 0.)

    def frames(self, errors):
        return [(f.GOAL - error, 11.75, True) for error in errors]

    def run_leg(self, runner):
        with patch("control.tasks.time.sleep"):
            return runner._run_leg("蓝", f.GOAL)

    def test_recorded_three_frame_boundary_uses_median_and_preserves_raw_last(self):
        r = self.runner(self.frames([2.4741459689, 2.4741459689, 2.5205434107]), samples=3)
        v = self.verify(r)
        self.assertTrue(v["passed"])
        self.assertAlmostEqual(v["median_signed_error_mm"], -2.4741459689)
        self.assertAlmostEqual(v["last_observation"]["sliders_x_mm"]["蓝"], f.GOAL-2.4741459689)
        self.assertAlmostEqual(v["last_observation"]["raw_last_sliders_x_mm"]["蓝"], f.GOAL-2.5205434107)

    def test_five_frames_use_median_even_when_last_is_outside_window(self):
        v = self.verify(self.runner(self.frames([2.47, 2.48, 2.49, 2.51, 2.52])))
        self.assertTrue(v["passed"])
        self.assertEqual(v["reliable_samples"], 5)
        self.assertAlmostEqual(v["median_signed_error_mm"], -2.49)

    def test_large_spread_is_unreliable_even_with_passing_median(self):
        v = self.verify(self.runner(self.frames([2.4, 2.4, 2.4, 2.4, 3.])))
        self.assertTrue(v["unreliable"])
        self.assertFalse(v["passed"])
        self.assertIn("verification_position_unstable", v["unreliable_reasons"])

    def test_bad_detection_and_stationary_drift_do_not_enter_the_median(self):
        frames = [(f.GOAL, 11.75, False), (f.GOAL, 13., True)] + self.frames([.1]*5)
        r = self.runner(frames)
        v = self.verify(r)
        self.assertTrue(v["passed"])
        self.assertEqual(len(v["samples"]), 7)
        self.assertEqual(v["reliable_samples"], 5)

    def test_passing_median_never_plans_recovery(self):
        r = self.runner([(-47.9, 11.75, True)] + self.frames([2.47,2.48,2.49,2.51,2.52]),
                        [(f.GOAL, None)])
        leg, _ = self.run_leg(r)
        self.assertEqual(leg["status"], "completed", leg.get("reason"))
        self.assertEqual(r.planner.corrections, [])

    def test_model_at_goal_still_gets_one_extra_mm_and_reobserves(self):
        r = self.runner([(-47.9, 11.75, True)] + self.frames([3.1]*5 + [2.1]*5),
                        [(f.GOAL, None), (f.GOAL+1., None)])
        leg, _ = self.run_leg(r)
        self.assertEqual(leg["status"], "completed", leg.get("reason"))
        self.assertEqual(len(r.planner.corrections), 1)
        phase = r.planner.corrections[0]
        self.assertAlmostEqual(phase["reference_target_mm"][0]-phase["reference_start_mm"][0], 1.)
        self.assertEqual(sum(p["phase"] == "push_correction" for p in r.motors.moves), 1)
        self.assertEqual(r.vision.reads, 11)

    def test_still_short_after_one_mm_has_no_second_recovery(self):
        r = self.runner([(-47.9, 11.75, True)] + self.frames([4.]*5 + [3.]*5),
                        [(f.GOAL, None), (f.GOAL+1., None)])
        leg, _ = self.run_leg(r)
        self.assertEqual(leg["reason"], "slider_not_at_target_after_single_recovery")
        self.assertEqual(sum(p["phase"] == "push_correction" for p in r.motors.moves), 1)
        self.assertEqual(r.vision.reads, 11)

    def test_recovery_timeout_reobserves_but_never_retries_the_motion(self):
        r = self.runner([(-47.9, 11.75, True)] + self.frames([4.]*5 + [3.]*5),
                        [(f.GOAL, None), (f.GOAL+1., "motion_goal_settle_timeout")])
        leg, _ = self.run_leg(r)
        self.assertEqual(leg["status"], "failed")
        self.assertEqual(sum(p["phase"] == "push_correction" for p in r.motors.moves), 1)
        self.assertEqual(r.vision.reads, 11)

    def test_unsafe_recovery_is_rejected_before_reentry(self):
        r = self.runner([(-47.9, 11.75, True)] + self.frames([3.]*5), [(f.GOAL, None)])
        with patch.object(r.planner, "recovery_push", side_effect=PlanError("recovery_other_slider_clearance")):
            leg, _ = self.run_leg(r)
        self.assertEqual(leg["reason"], "recovery_other_slider_clearance")
        self.assertEqual(sum(p["phase"] == "push_correction" for p in r.motors.moves), 0)
        self.assertEqual(sum(p["phase"] == "free" and not p.get("route") for p in r.motors.moves), 1)

    def test_invalid_aggregation_and_recovery_settings_are_rejected(self):
        for section, key, value in [("verification","aggregation","average"),
                                    ("verification","max_spread_mm",0.),
                                    ("push_correction","max_steps",2),
                                    ("push_correction","step_mm",.5)]:
            c = json.loads((ROOT / "config/robot.json").read_text())
            c[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(c)


class RecoveryGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = Planner(ROOT)

    def test_left_extension_is_exactly_one_mm_and_checks_the_exit(self):
        p = self.p
        q = p.arm.inverse([-20.,155.], "-x")
        phase = p.recovery_push(q,"橙",-20.,"-x",{"橙":-18.,"蓝":-46.56},[-31.06,13.44])
        self.assertAlmostEqual(phase["reference_target_mm"][0], -21.)
        self.assertTrue(phase["collision_checked"])
        self.assertTrue(phase["recovery_exit_checked"])
        self.assertEqual(phase["task_target_x_mm"], -20.)
        for point in phase["points"]:
            self.assertTrue(p.arm.within_limits(point["q_rad"]))

    def test_other_slider_blocks_unsafe_extension(self):
        p = self.p
        q = p.arm.inverse([-31.06,155.], "-x")
        with self.assertRaisesRegex(PlanError,"recovery_other_slider_clearance"):
            p.recovery_push(q,"橙",-31.06,"-x",{"橙":-28.54,"蓝":-45.93},[-31.06,13.44])

    def test_outer_rail_limit_cannot_be_crossed(self):
        p = self.p
        phase = {"points":[{"q_rad":[0.,0.]},{"q_rad":[0.,.01]}]}
        with patch.object(p,"_checked_edges"), patch.object(p.arm,"slider_center",return_value=[-47.,155.]):
            with self.assertRaisesRegex(PlanError,"recovery_outside_physical_rail"):
                p._certify_recovery(phase,{"橙":-28.,"蓝":0.},"橙","-x",[-31.06,13.44])

    def test_static_collision_and_unsafe_departure_block_recovery(self):
        p = self.p
        q = p.arm.inverse([-20.,155.], "-x")
        with patch.object(p.scene,"check",return_value={"obstacle":"fixture"}):
            with self.assertRaisesRegex(PlanError,"fixture"):
                p.recovery_push(q,"橙",-20.,"-x",{"橙":-18.,"蓝":-46.56},[-31.06,13.44])
        with patch.object(p,"observation_retreat",side_effect=PlanError("unsafe_exit")):
            with self.assertRaisesRegex(PlanError,"unsafe_exit"):
                p.recovery_push(q,"橙",-20.,"-x",{"橙":-18.,"蓝":-46.56},[-31.06,13.44])

    def test_complete_task_uses_real_planner_for_reentry_and_both_home_returns(self):
        p = self.p
        class Motors:
            q = [math.radians(180.), math.radians(4.)]
            moves = None
            def __init__(self): self.moves = []
            def read_state(self): return {"q_rad": list(self.q)}
            def halt(self): return {"stopped_confirmed": True}
            def move(self, phase, health_check=None):
                self.moves.append(copy.deepcopy(phase))
                self.q = list(phase["points"][-1]["q_rad"])
                return {"status":"completed", "phase":phase["phase"]}
        class Vision:
            stamp = 0.
            def check_health(self): pass
            def read(self, after_monotonic_s=None):
                self.stamp = max(self.stamp + 1., (after_monotonic_s or 0.) + 1.)
                count = sum(m["phase"] in ("push", "push_correction") for m in motors.moves)
                return {"sliders_x_mm":{"橙": [11.3,1.,.1][count], "蓝":-46.56},
                        "stamp_monotonic_s":self.stamp}
        motors = Motors()
        runner = TaskRunner(p, p.config, motors, Vision())
        runner.legs({"intent":"push_distance","color":"橙","direction":"-x","distance_mm":10},
                    {"橙":11.3,"蓝":-46.56}, f.LIMITS)
        with patch("control.tasks.time.sleep"):
            leg, state = runner._run_leg("橙", 0.)
        self.assertEqual(leg["status"], "completed", leg.get("reason"))
        corrections = [m for m in motors.moves if m["phase"] == "push_correction"]
        self.assertEqual(len(corrections), 1)
        self.assertAlmostEqual(p.arm.slider_center(corrections[0]["points"][-1]["q_rad"],"-x")[0], -1.)
        self.assertEqual(sum(m.get("route") == "return_home" for m in motors.moves), 2)
        self.assertAlmostEqual(state["sliders_x_mm"]["橙"], .1)


if __name__ == "__main__":
    unittest.main()
