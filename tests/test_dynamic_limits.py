"""Offline task checks; no camera, serial port or hardware dependencies."""
import copy
import unittest
from unittest.mock import Mock

from control.config import validate_config
from control.planner import Planner, PlanError
from control.targets import TargetResolver
from control.tasks import TaskError, TaskRunner


def configuration():
    return {"sliders": {"蓝": {"y_mm": 155., "end": "right"},
                        "橙": {"y_mm": 155., "end": "left"}},
            "verification": {"tolerance_mm": .5, "samples": 1, "interval_s": .01},
            "retry_limit": 0, "poll_interval_s": 1.}


POSITIONS = {"蓝": -56.9, "橙": 8.3}
LIMITS = {"蓝": [-56.9, -12.4], "橙": [-36.2, 8.3]}


class DynamicTaskLimitsTests(unittest.TestCase):
    def runner(self):
        return TaskRunner(Mock(), configuration(), Mock(), Mock())

    def test_limits_follow_installation_translation(self):
        for shift in (0., 7.25, -12.):
            runner = self.runner()
            positions = {c: x + shift for c, x in POSITIONS.items()}
            limits = {c: [x + shift for x in bounds] for c, bounds in LIMITS.items()}
            blue, _ = runner.legs({"intent": "push_to_end", "color": "蓝"}, positions, limits)
            orange, _ = runner.legs({"intent": "push_to_end", "color": "橙"}, positions, limits)
            self.assertEqual(blue, [-12.4 + shift])
            self.assertEqual(orange, [-36.2 + shift])

    def test_distance_clips_to_observed_limits(self):
        goals, extra = self.runner().legs({"intent": "push_distance", "color": "蓝",
            "direction": "+x", "distance_mm": 60}, POSITIONS, LIMITS)
        self.assertEqual(goals, [-12.4])
        self.assertTrue(extra["clipped"])
        self.assertAlmostEqual(extra["requested_x_mm"], 3.1)

    def test_round_trip_returns_to_actual_start_in_middle(self):
        positions = {**POSITIONS, "蓝": -34.}
        goals, extra = self.runner().legs({"intent": "push_round_trip", "color": "蓝",
            "round_trips": 2}, positions, LIMITS)
        self.assertEqual(goals, [-12.4, -34., -12.4, -34.])
        self.assertEqual(extra["return_x_mm"], -34.)

    def test_round_trip_selects_other_end_when_already_at_configured_end(self):
        goals, _ = self.runner().legs({"intent": "push_round_trip", "color": "蓝",
            "round_trips": 1}, {**POSITIONS, "蓝": -12.4}, LIMITS)
        self.assertEqual(goals, [-56.9, -12.4])

    def test_limits_are_copied_from_initial_observation(self):
        runner = self.runner()
        limits = copy.deepcopy(LIMITS)
        goals, _ = runner.legs({"intent": "push_to_end", "color": "蓝"}, POSITIONS, limits)
        limits["蓝"][:] = [-100., 100.]
        self.assertEqual(runner.targets.bounds("蓝"), (-56.9, -12.4))
        self.assertEqual(goals, [-12.4])

    def test_missing_limits_fail_before_any_planning_or_move(self):
        runner = self.runner()
        runner.motors.halt.return_value = {"stopped_confirmed": True}
        runner._snapshot = lambda: {"sliders_x_mm": POSITIONS, "q_rad": [3., .5]}
        result = runner.run({"intent": "push_to_end", "color": "蓝"})
        self.assertEqual(result["reason"], "slider_limits_unavailable")
        self.assertEqual(result["segments"], [])
        self.assertEqual(runner.planner.mock_calls, [])
        runner.motors.move.assert_not_called()

    def test_partial_and_nonfinite_limits_are_rejected(self):
        for limits in ({"蓝": LIMITS["蓝"]}, {**LIMITS, "蓝": [float("nan"), 0.]},
                       {**LIMITS, "蓝": [0., -1.]}):
            with self.subTest(limits=limits), self.assertRaisesRegex(TaskError, "invalid_observed_slider_limits"):
                self.runner().legs({"intent": "push_to_end", "color": "蓝"}, POSITIONS, limits)

    def test_leg_gives_planner_the_frozen_task_limits(self):
        runner = self.runner()
        runner.legs({"intent": "push_to_end", "color": "蓝"}, POSITIONS, LIMITS)
        live = {"sliders_x_mm": POSITIONS, "slider_limits_x_mm": {"蓝": [-60., -8.], "橙": [-40., 12.]},
                "q_rad": [3., .5]}
        runner._snapshot = lambda: copy.deepcopy(live)
        runner._withdraw = lambda *_: None
        runner._move = lambda *_: live["q_rad"]
        runner.motors.read_state.return_value = {"q_rad": live["q_rad"]}
        runner._verify = lambda *_: {"passed": True, "last_observation": copy.deepcopy(live)}
        record, _ = runner._run_leg("蓝", -12.4)
        self.assertEqual(record["status"], "completed")
        self.assertEqual(runner.planner.push.call_args.args[-1], (-56.9, -12.4))

    def test_config_requires_direction_and_finite_fixed_y_only(self):
        config = configuration()
        validate_config(config)
        config["sliders"]["蓝"]["end"] = "other"
        with self.assertRaises(ValueError):
            validate_config(config)
        config["sliders"]["蓝"]["end"] = "right"
        config["sliders"]["蓝"]["y_mm"] = float("nan")
        with self.assertRaises(ValueError):
            validate_config(config)


class PlannerObservedLimitsTests(unittest.TestCase):
    def planner(self):
        planner = Planner.__new__(Planner)
        planner.config = configuration()
        planner.settings = {"contact_position_tolerance_mm": 1., "push_speed_mm_s": 5., "linear_step_mm": 1.}
        planner._q = lambda q: q
        planner._positions = lambda p: p
        planner.arm = Mock()
        planner.arm.slider_center.return_value = [POSITIONS["蓝"], 155.]
        planner.arm.world_normal.return_value = [1., 0.]
        planner.arm.faces = {"+x": {"point_canonical_xy_mm": [134., 36.]}}
        planner._line = Mock(return_value=([[0., 0.], [0., .1]], [44.5]))
        planner._phase = lambda name, path, distances, speed, **data: {"phase": name, **data}
        return planner

    def test_push_uses_observed_limits_without_fixed_config_bounds(self):
        planner = self.planner()
        phase = planner.push([0., 0.], "蓝", -12.4, POSITIONS, LIMITS["蓝"])
        self.assertEqual(phase["feasible_x_limits_mm"], [-56.9, -12.4])
        with self.assertRaisesRegex(PlanError, "push_target_outside_observed_limits"):
            planner.push([0., 0.], "蓝", 0., POSITIONS, LIMITS["蓝"])

    def test_push_never_falls_back_if_observed_limits_absent(self):
        planner = self.planner()
        with self.assertRaisesRegex(PlanError, "invalid_observed_slider_limits"):
            planner.push([0., 0.], "蓝", -12.4, POSITIONS, None)
        planner._line.assert_not_called()


if __name__ == "__main__":
    unittest.main()
