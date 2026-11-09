"""Stopped-observation recovery tests; no camera, serial or arm movement."""
import copy
import json
import math
from pathlib import Path
import time
import unittest
from unittest.mock import Mock, patch

from control.config import push_correction_settings, validate_config
from control.planner import Planner, PlanError
from control.scene import SimpleScene
from control.tasks import TaskRunner


ROOT = Path(__file__).resolve().parents[1]


class Vision:
    def __init__(self, x):
        self.x = x
        self.reads = []
        self.motors = None

    def check_health(self):
        pass

    def read(self, after_monotonic_s=None):
        if not self.motors.stopped:
            raise AssertionError("pin detection attempted during motion")
        stamp = max(time.monotonic(), (after_monotonic_s or 0.) + 1e-6)
        self.reads.append({"after": after_monotonic_s, "stamp": stamp,
                           "last_stop": self.motors.last_stop})
        # Deliberately change the live limits: the task must keep its initial ones.
        return {"sliders_x_mm": {"蓝": self.x, "橙": 15.},
                "slider_limits_x_mm": {"蓝": [-100., 100.], "橙": [-100., 100.]},
                "stamp_monotonic_s": stamp}


class Motors:
    def __init__(self, vision, outcomes):
        self.vision, self.outcomes = vision, list(outcomes)
        vision.motors = self
        self.q = [-10., 0.]
        self.moves = []
        self.stopped = True
        self.stop_confirmed = True
        self.last_stop = time.monotonic()
        self.no_feedback_after_push = False

    def halt(self):
        self.stopped = self.stop_confirmed
        self.last_stop = time.monotonic()
        return {"stopped_confirmed": self.stop_confirmed}

    def read_state(self):
        if self.no_feedback_after_push and any(p["phase"] == "push" for p in self.moves):
            raise RuntimeError("joint_feedback_stale")
        return {"q_rad": list(self.q)}

    def move(self, phase, health_check=None):
        self.moves.append(copy.deepcopy(phase))
        self.stopped = False
        if phase["phase"] in ("push", "push_correction"):
            outcome = self.outcomes.pop(0)
            self.vision.x = outcome[0] if isinstance(outcome, tuple) else outcome
            self.q[0] = phase["reference_target_mm"][0]
            if isinstance(outcome, tuple):
                raise RuntimeError(outcome[1])
        return {"status": "completed", "phase": phase["phase"]}


class FakePlanner:
    def __init__(self):
        self.corrections = []

    def precontact(self, *args):
        return [-10., 0.]

    def free(self, *args, **kwargs):
        return {"phase": "free"}

    def approach(self, *args):
        return {"phase": "approach"}

    def push(self, q, color, goal, sliders, limits):
        return {"phase": "push", "reference_target_mm": [goal, 155.],
                "task_target_x_mm": goal, "limits": tuple(limits)}

    def correction(self, q, color, goal, direction, sliders, limits, distance):
        sign = 1 if direction == "+x" else -1
        phase = {"phase": "push_correction", "task_target_x_mm": goal,
                 "reference_start_mm": [q[0], 155.],
                 "reference_target_mm": [q[0] + sign * distance, 155.],
                 "extension_mm": distance, "limits": tuple(limits)}
        self.corrections.append(phase)
        return phase


class BoundedPushTests(unittest.TestCase):
    def runner(self, outcomes, start=-10., **settings):
        config = json.loads((ROOT / "config/robot.json").read_text())
        config["verification"] = {"tolerance_mm": .5, "samples": 2, "interval_s": .001}
        # Production robot.json omits push_correction and relies on the
        # defaults in control.config.push_correction_settings().
        config.setdefault("push_correction", {}).update(settings)
        vision = Vision(start)
        motors = Motors(vision, outcomes)
        planner = FakePlanner()
        runner = TaskRunner(planner, config, motors, vision)
        runner._withdraw = lambda *_: None
        runner.legs({"intent": "push_to_end", "color": "蓝"},
                    {"蓝": start, "橙": 15.}, {"蓝": [-20., 0.], "橙": [-10., 15.]})
        return runner

    def run_leg(self, runner, goal=0.):
        with patch("control.tasks.time.sleep"):
            return runner._run_leg("蓝", goal)

    def test_initial_arrival_needs_no_correction_and_verifies_only_after_stop(self):
        runner = self.runner([0.])
        record, state = self.run_leg(runner)
        self.assertEqual(record["status"], "completed")
        self.assertEqual(runner.planner.corrections, [])
        self.assertEqual(state["sliders_x_mm"]["蓝"], 0.)
        verify = record["attempts"][0]["verification_after_monotonic_s"]
        for read in runner.vision.reads[-2:]:
            self.assertGreaterEqual(read["after"], verify)
            self.assertGreater(read["after"], read["last_stop"])
            self.assertGreater(read["stamp"], read["after"])

    def test_correction_extends_measured_reference_and_keeps_task_goal(self):
        runner = self.runner([-2.4, -1.4, -.4])
        record, state = self.run_leg(runner)
        self.assertEqual(record["status"], "completed")
        self.assertEqual(len(runner.planner.corrections), 2)
        self.assertEqual([p["reference_start_mm"][0] for p in runner.planner.corrections], [0., 1.])
        self.assertEqual([p["reference_target_mm"][0] for p in runner.planner.corrections], [1., 2.])
        self.assertTrue(all(p["task_target_x_mm"] == 0. for p in runner.planner.corrections))
        self.assertTrue(all(p["limits"] == (-20., 0.) for p in runner.planner.corrections))
        self.assertEqual(runner.targets.bounds("蓝"), (-20., 0.))
        self.assertEqual(record["target_x_mm"], 0.)
        self.assertEqual(runner.withdraw_extension_mm, 2.)
        self.assertAlmostEqual(state["sliders_x_mm"]["蓝"], -.4)
        for step in [record["attempts"][0], *record["attempts"][0]["corrections"]]:
            self.assertTrue(step["stop"]["stopped_confirmed"])
            self.assertTrue(all(s["stamp_monotonic_s"] > step["verification_after_monotonic_s"]
                                for s in step["verification"]["samples"]))

    def test_never_more_than_three_corrections_or_three_mm(self):
        runner = self.runner([-5., -4.5, -4., -3.5])
        record, state = self.run_leg(runner)
        self.assertIsNone(state)
        self.assertEqual(record["reason"], "push_correction_budget_exhausted")
        self.assertEqual(len(record["attempts"]), 1)
        self.assertEqual(len(runner.planner.corrections), 3)
        self.assertEqual(sum(p["extension_mm"] for p in runner.planner.corrections), 3.)

    def test_smaller_distance_budget_and_step_are_respected(self):
        runner = self.runner([-5., -4.7, -4.4, -4.1],
                             step_mm=.6, max_distance_mm=1.3)
        record, _ = self.run_leg(runner)
        self.assertEqual(record["reason"], "push_correction_budget_exhausted")
        self.assertEqual(len(runner.planner.corrections), 3)
        self.assertAlmostEqual(sum(p["extension_mm"] for p in runner.planner.corrections), 1.3)
        self.assertTrue(all(p["extension_mm"] <= .6 for p in runner.planner.corrections))

    def test_no_visual_progress_stops_after_first_ineffective_correction(self):
        runner = self.runner([-2., -1.95])
        record, _ = self.run_leg(runner)
        self.assertEqual(record["reason"], "push_no_progress")
        self.assertEqual(len(record["attempts"]), 1)
        self.assertEqual(len(runner.planner.corrections), 1)

    def test_no_progress_in_main_push_never_starts_correction(self):
        runner = self.runner([-9.95])
        record, _ = self.run_leg(runner)
        self.assertEqual(record["reason"], "push_no_progress")
        self.assertEqual(len(record["attempts"]), 1)
        self.assertEqual(runner.planner.corrections, [])

    def test_main_motor_stall_does_not_replay_or_start_correction(self):
        runner = self.runner([(-2., "motion_no_progress")])
        record, _ = self.run_leg(runner)
        self.assertEqual(record["reason"], "motion_no_progress")
        self.assertEqual(len(record["attempts"]), 1)
        self.assertEqual(runner.planner.corrections, [])
        self.assertEqual(sum(p["phase"] == "push" for p in runner.motors.moves), 1)

    def test_correction_motor_stall_stops_without_using_remaining_budget(self):
        runner = self.runner([-2., (-1.5, "motion_no_progress")])
        record, _ = self.run_leg(runner)
        self.assertEqual(record["reason"], "motion_no_progress")
        self.assertEqual(len(record["attempts"]), 1)
        self.assertEqual(len(runner.planner.corrections), 1)

    def test_verified_goal_can_recover_a_stopped_motor_error(self):
        for outcomes in ([(-.2, "motion_no_progress")], [-1.4, (-.4, "motion_no_progress")]):
            with self.subTest(outcomes=outcomes):
                runner = self.runner(outcomes)
                record, _ = self.run_leg(runner)
                self.assertEqual(record["status"], "completed")

    def test_overshoot_is_not_reversed_or_accepted_by_larger_tolerance(self):
        for outcomes in ([.7], [-1.3, .8]):
            with self.subTest(outcomes=outcomes):
                runner = self.runner(outcomes)
                record, _ = self.run_leg(runner)
                self.assertEqual(record["reason"], "slider_overshot_target")
                self.assertEqual(len(record["attempts"]), 1)
                self.assertEqual(len(runner.planner.corrections), len(outcomes) - 1)
                self.assertEqual(runner.tolerance, .5)

    def test_correction_step_is_shortened_to_remaining_error(self):
        runner = self.runner([-.7, 0.])
        record, _ = self.run_leg(runner)
        self.assertEqual(record["status"], "completed")
        self.assertAlmostEqual(runner.planner.corrections[0]["extension_mm"], .7)

    def test_already_at_goal_also_requires_confirmed_stop_and_new_frames(self):
        runner = self.runner([], start=0.)
        record, _ = self.run_leg(runner)
        self.assertEqual(record["status"], "completed")
        self.assertTrue(record["attempts"][0]["no_motion"])
        self.assertEqual(runner.motors.moves, [])
        verify = record["attempts"][0]["verification_after_monotonic_s"]
        self.assertTrue(all(r["after"] >= verify for r in runner.vision.reads[-2:]))

    def test_missing_motor_feedback_never_starts_correction(self):
        runner = self.runner([-2.])
        runner.motors.no_feedback_after_push = True
        record, _ = self.run_leg(runner)
        self.assertEqual(record["reason"], "joint_feedback_stale")
        self.assertEqual(len(record["attempts"]), 1)
        self.assertEqual(runner.planner.corrections, [])

    def test_unconfirmed_stop_prevents_verification_and_correction(self):
        runner = self.runner([-2.])
        runner.motors.stop_confirmed = False
        record, _ = self.run_leg(runner)
        self.assertEqual(record["reason"], "stop_not_confirmed")
        self.assertEqual(len(runner.vision.reads), 1)
        self.assertEqual(runner.planner.corrections, [])

    def test_free_entry_still_has_three_retries_without_replaying_a_push(self):
        runner = self.runner([0.])
        runner.planner.free = Mock(side_effect=RuntimeError("free_path_not_found"))
        record, _ = self.run_leg(runner)
        self.assertEqual(len(record["attempts"]), 4)
        self.assertEqual(record["reason"], "free_path_not_found")
        self.assertEqual(runner.motors.moves, [])

    def test_leftward_correction_extends_left_from_current_reference(self):
        runner = self.runner([-8., -9., -10.], start=0.)
        record, _ = self.run_leg(runner, -10.)
        self.assertEqual(record["status"], "completed")
        self.assertEqual([p["reference_target_mm"][0] for p in runner.planner.corrections], [-11., -12.])
        self.assertTrue(all(p["task_target_x_mm"] == -10. for p in runner.planner.corrections))


class CorrectionGeometryTests(unittest.TestCase):
    def setUp(self):
        self.planner = Planner(ROOT)

    def q(self, x, direction="+x", y=155.):
        return self.planner.arm.inverse([x, y], direction)

    def test_real_ik_extends_fk_instead_of_visually_estimated_pin(self):
        planner = self.planner
        q = self.q(-20.)
        phase = planner.correction(q, "蓝", -20., "+x", {"蓝": -24., "橙": 15.}, [-60., -20.], 1.)
        self.assertEqual(phase["task_target_x_mm"], -20.)
        self.assertEqual(phase["feasible_x_limits_mm"], [-60., -20.])
        self.assertAlmostEqual(phase["reference_start_mm"][0], -20.)
        self.assertAlmostEqual(phase["reference_target_mm"][0], -19.)
        self.assertAlmostEqual(planner.arm.slider_center(phase["points"][-1]["q_rad"], "+x")[0], -19.)
        self.assertGreaterEqual(phase["points"][-1]["time_s"], 1.)
        for point in phase["points"]:
            self.assertTrue(planner.arm.within_limits(point["q_rad"]))
            self.assertGreater(planner.arm.world_normal(point["q_rad"], "+x")[0], 0.)
        # The ordinary push retains its stricter vision/FK agreement.
        with self.assertRaisesRegex(PlanError, "push_start_contact_error"):
            planner.push(q, "蓝", -20., {"蓝": -24., "橙": 15.}, [-60., -20.])

    def test_y_alignment_face_direction_and_task_limits_are_still_enforced(self):
        planner = self.planner
        with self.assertRaisesRegex(PlanError, "push_correction_y_error"):
            planner.correction(self.q(-20., y=153.), "蓝", -20., "+x",
                {"蓝": -24., "橙": 15.}, [-60., -20.], 1.)
        q = self.q(-20.)
        with patch.object(planner.arm, "world_normal", return_value=(-1., 0.)):
            with self.assertRaisesRegex(PlanError, "contact_face_points_away"):
                planner.correction(q, "蓝", -20., "+x",
                    {"蓝": -24., "橙": 15.}, [-60., -20.], 1.)
        with self.assertRaisesRegex(PlanError, "push_target_outside_observed_limits"):
            planner.correction(self.q(-20.), "蓝", 0., "+x",
                {"蓝": -24., "橙": 15.}, [-60., -20.], 1.)

    def test_step_is_limited_and_branch_jump_remains_forbidden(self):
        planner = self.planner
        q = self.q(-20.)
        for value in (0., -1., 1.01, float("nan"), True):
            with self.subTest(value=value), self.assertRaisesRegex(PlanError, "invalid_push_correction_distance"):
                planner.correction(q, "蓝", -20., "+x", {"蓝": -24., "橙": 15.}, [-60., -20.], value)
        with patch.object(planner.arm, "inverse", return_value=(q[0] + math.radians(6), q[1])):
            with self.assertRaisesRegex(PlanError, "inverse_branch_discontinuity"):
                planner.correction(q, "蓝", -20., "+x", {"蓝": -24., "橙": 15.}, [-60., -20.], 1.)

    def test_reference_withdraw_retracts_from_fk_with_only_actual_extension_exemption(self):
        planner = self.planner
        q = self.q(-20.)
        with patch.object(planner, "_checked_edges") as checked:
            phase = planner.withdraw(q, "蓝", "+x", {"蓝": -21.5, "橙": 15.},
                                      from_reference=True, reference_extension_mm=2.)
        self.assertAlmostEqual(phase["reference_start_mm"][0], -20.)
        # Right-side clearance is 10 mm; retract the 2 mm extension as well.
        self.assertAlmostEqual(phase["reference_target_mm"][0], -32.)
        self.assertEqual(phase["withdraw_distance_mm"], 12.)
        self.assertEqual(phase["reference_contact_tolerance_mm"], 2.)
        self.assertEqual(checked.call_args.args[-1]["reference_contact_tolerance_mm"], 2.)
        with self.assertRaisesRegex(PlanError, "contact_start_position_error"):
            planner.withdraw(q, "蓝", "+x", {"蓝": -21.5, "橙": 15.})
        with self.assertRaisesRegex(PlanError, "reference_withdraw_x_error_exceeds_budget"):
            planner.withdraw(q, "蓝", "+x", {"蓝": -23., "橙": 15.},
                             from_reference=True, reference_extension_mm=2.)
        with self.assertRaisesRegex(PlanError, "invalid_reference_withdraw_extension"):
            planner.withdraw(q, "蓝", "+x", {"蓝": -21.5, "橙": 15.},
                             from_reference=True, reference_extension_mm=3.1)

    def test_corrected_withdraw_ends_completely_clear_in_both_directions(self):
        planner = self.planner
        for direction, extension in (("-x", 2.), ("-x", 3.), ("+x", 3.)):
            with self.subTest(direction=direction, extension=extension):
                sign = 1 if direction == "+x" else -1
                goal = .4
                q = self.q(goal + sign * extension, direction)
                positions = {"橙": goal, "蓝": -31.2}
                phase = planner.withdraw(q, "橙", direction, positions,
                    from_reference=True, reference_extension_mm=extension)
                end = phase["points"][-1]["q_rad"]
                self.assertAlmostEqual(planner.arm.slider_center(end, direction)[0],
                                       goal - sign * planner.offsets[direction])
                self.assertIsNone(planner.scene.check(end, positions))

    def test_withdraw_endpoint_is_clear_through_the_half_mm_acceptance_band(self):
        planner = self.planner
        goal = .4
        for direction, extension in (("-x", 2.), ("-x", 3.), ("+x", 3.)):
            sign = 1 if direction == "+x" else -1
            q = self.q(goal + sign * extension, direction)
            # The end must support an ordinary free move, with no exemption.
            end = self.q(goal - sign * planner.offsets[direction], direction)
            for error in (-.5, 0., .5):
                with self.subTest(direction=direction, extension=extension, error=error):
                    positions = {"橙": goal + error, "蓝": -31.2}
                    self.assertIsNone(planner.scene.check(end, positions))
                    if (direction, extension, error) in (("-x", 2., .5), ("+x", 3., -.5)):
                        # The path's initial overlap exceeds the actual
                        # extension exemption. A clear endpoint cannot excuse it.
                        with self.assertRaisesRegex(PlanError, "collision:"):
                            planner.withdraw(q, "橙", direction, positions,
                                from_reference=True, reference_extension_mm=extension)
                    else:
                        phase = planner.withdraw(q, "橙", direction, positions,
                            from_reference=True, reference_extension_mm=extension)
                        self.assertIsNone(planner.scene.check(phase["points"][-1]["q_rad"], positions))

    def test_reference_withdraw_checks_its_last_point_without_any_exemption(self):
        planner = self.planner
        q = self.q(-2.6, "-x")
        with patch.object(planner, "_checked_edges"), patch.object(planner, "_clear",
                side_effect=PlanError('collision:{"obstacle":"mass_block"}')) as clear:
            with self.assertRaisesRegex(PlanError, "mass_block"):
                planner.withdraw(q, "橙", "-x", {"橙": .4, "蓝": -31.2},
                    from_reference=True, reference_extension_mm=3.)
        self.assertEqual(len(clear.call_args.args), 2)

    def test_invalid_correction_budgets_are_rejected_at_config_load(self):
        config = json.loads((ROOT / "config/robot.json").read_text())
        for key, value in (("max_steps", 4), ("step_mm", 1.1), ("max_distance_mm", 3.1),
                           ("speed_mm_s", 1.1), ("min_progress_mm", 0.)):
            with self.subTest(key=key):
                altered = copy.deepcopy(config)
                altered.setdefault("push_correction", {})[key] = value
                with self.assertRaises(ValueError):
                    validate_config(altered)
        self.assertEqual(push_correction_settings(config)["max_steps"], 3)


class ReferenceContactExemptionTests(unittest.TestCase):
    def config(self):
        return {"arm": {"joint_limits_rad": [[-1., 1.], [-1., 1.]],
                        "elbow_translation_mm": [1., 0.],
                        "shoulder_static_yaw_rad": 0., "elbow_static_yaw_rad": 0.},
                "bodies": [{"part": "upper", "name": "face", "contact_face": True,
                            "z_mm": [0., 1.], "polygon_xy_mm": [[0., 0.], [1., 0.], [1., 1.], [0., 1.]]}],
                "workspace_xy_mm": [[-10., -10.], [10., 10.]], "obstacles": [],
                "slider_primitives": [{"name": "stem", "shape": "circle", "radius_mm": 1.5,
                                       "z_mm": [0., 1.]}]}

    def test_only_explicit_reference_field_exempts_the_selected_stem_against_a_face(self):
        scene = SimpleScene(self.config())
        pins = {"蓝": [2., .5]}
        self.assertIsNotNone(scene.check([0., 0.], pins))
        self.assertIsNotNone(scene.check([0., 0.], pins, {"color": "蓝", "tolerance_mm": 3.}))
        self.assertIsNone(scene.check([0., 0.], pins,
            {"color": "蓝", "reference_contact_tolerance_mm": .6}))
        self.assertIsNotNone(scene.check([0., 0.], pins,
            {"color": "橙", "reference_contact_tolerance_mm": .6}))
        self.assertEqual(scene.check([0., 0.], pins,
            {"color": "蓝", "reference_contact_tolerance_mm": 3.1})["obstacle"],
            "invalid_reference_contact_tolerance")

    def test_other_body_slider_block_obstacles_and_workspace_remain_checked(self):
        contact = {"color": "蓝", "reference_contact_tolerance_mm": 3.}
        for kind in ("non_contact_body", "block", "obstacle", "workspace"):
            with self.subTest(kind=kind):
                config = self.config()
                if kind == "non_contact_body":
                    config["bodies"][0]["contact_face"] = False
                elif kind == "block":
                    config["slider_primitives"].append({"name": "block", "shape": "box",
                        "size_xy_mm": [2.5, 1.], "z_mm": [0., 1.]})
                elif kind == "obstacle":
                    config["obstacles"].append({"name": "mass", "center_mm": [.5, .5, .5],
                                                "size_mm": [1., 1., 1.]})
                else:
                    config["workspace_xy_mm"][1] = [.9, .9]
                hit = SimpleScene(config).check([0., 0.], {"蓝": [2., .5]}, contact)
                self.assertIsNotNone(hit)
                if kind == "obstacle":
                    self.assertEqual(hit["obstacle"], "mass")
                elif kind == "workspace":
                    self.assertEqual(hit["obstacle"], "workspace_boundary")


if __name__ == "__main__":
    unittest.main()
