"""Offline checks for registered measurements and physical cylinder tangency."""

import json
from pathlib import Path
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from control.tasks import TaskRunner
from control.vision import VisionFrameRejected
from vision_tools.pixel_to_world import convert_detections
from vision_tools.rail_registration import RailRegistration, RailRegistrationError

ROOT = Path(__file__).resolve().parents[1]
CAL = json.loads((ROOT / "config/camera_world_scheme_e_20261118.json").read_text())


class TestCylinderTangent(unittest.TestCase):
    def test_both_tangent_sides_recover_axis_through_scheme_e_distortion(self):
        camera = np.array(CAL["C_world_mm"])
        rotation = np.array(CAL["R_world_from_camera"])
        radius = 4.
        for x in (54., 69., 113.5):
            centre = np.array([x, CAL["plane_y_mm"]])
            v = centre - camera[:2]
            d = np.linalg.norm(v)
            base = camera[:2] + v * (1 - (radius / d)**2)
            offset = np.array([-v[1], v[0]]) * (radius * np.sqrt(d*d - radius*radius) / (d*d))
            projected = []
            for xy in (base + offset, base - offset):
                points = np.array([[*xy, z] for z in np.linspace(-55., -72., 31)])
                cam = (points - camera) @ rotation
                pixels, _ = cv2.projectPoints(cam, np.zeros(3), np.zeros(3),
                                              np.array(CAL["K"]), np.array(CAL["D"]))
                projected.append(pixels.reshape(-1, 2))
            projected.sort(key=lambda pts: pts[:, 0].mean())
            for side, pixels in zip(("left", "right"), projected):
                data = {"schema_version": 1, "profile": "real",
                        "position_definition": "shaft_axis_intersection_with_image_reference_line",
                        "sliders": {"yellow": {"pixel_xy": None,
                            "reference_method": "rod_band_far_edge_tangent",
                            "measurement": {"edge_pixels": pixels.tolist(), "visible_side": side}}}}
                with self.subTest(x=x, side=side):
                    item = convert_detections(data, CAL, stem_radius_mm=radius)["sliders"]["yellow"]
                    self.assertTrue(item["occluded"])
                    self.assertAlmostEqual(item["world_xyz_mm"][0], x, places=7)
                    self.assertAlmostEqual(item["axis_x_core_spread_mm"], 0., places=7)
                    self.assertEqual(item["world_xyz_mm"][1], 181.5)


class TestRegistration(unittest.TestCase):
    def test_known_shift_and_covered_patch(self):
        rng = np.random.default_rng(56)
        gray = cv2.GaussianBlur(rng.integers(0, 256, (260, 410), dtype=np.uint8), (0, 0), 2)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        reference = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        rois = [(38, 45, 105, 100), (170, 121, 235, 175), (300, 175, 356, 224)]
        tracker = RailRegistration(reference, rois)
        shift = (2.25, -3.125)
        moved = cv2.warpAffine(reference, np.float32([[1, 0, shift[0]], [0, 1, shift[1]]]),
                               (410, 260), flags=cv2.INTER_CUBIC)
        _, result = tracker.align(moved)
        np.testing.assert_allclose(result["shift_from_reference_px"], shift, atol=.13)
        moved[29:114, 22:121] = 128
        with self.assertRaises(RailRegistrationError):
            tracker.align(moved)


class TestRelativeVerification(unittest.TestCase):
    def runner(self):
        config = json.loads((ROOT / "config/robot.json").read_text())
        config["verification"]["interval_s"] = 0
        config["verification"]["force_pass_unreliable"] = True
        vision = Mock(relative_measurement=True)
        runner = TaskRunner(None, config, vision=vision)
        runner.targets = Mock()
        runner.targets.bounds.return_value = (-48.5, -4.0)
        return runner, vision

    def test_round_trip_uses_measured_stop_then_returns_to_session_start(self):
        runner, vision = self.runner()
        vision.stop_reference_x.return_value = -.9
        refs = runner._relative_references(
            {"intent": "push_round_trip", "color": "蓝"}, [-4., -48.5], -48.7)
        self.assertAlmostEqual(refs[0]["target_delta_x_mm"], 47.8)
        self.assertEqual(refs[1]["target_delta_x_mm"], 0)
        self.assertIsNone(refs[1]["stop_direction"])
        runner._relative_leg = refs[0]
        reasons = ["reading_beyond_physical_stop"]
        vision.read.side_effect = [
            {"sliders_x_mm": {"蓝": x}, "stamp_monotonic_s": i + 1,
             "reliability": {"蓝": {"reliable": False, "measurement_reliable": True, "reasons": reasons}}}
            for i, x in enumerate((-.89, -.85, -.90, -.93, -.96))]
        accepted = runner._verify("蓝", -4., 0.)
        self.assertTrue(accepted["passed"])
        self.assertAlmostEqual(accepted["d_stop_mm"], 0.)
        self.assertAlmostEqual(accepted["delta_x_mm"], 47.8)
        runner._relative_leg = refs[1]
        self.assertAlmostEqual(runner._position_error(-48.5, -48.7), 0.0)

    def test_registration_rejection_breaks_the_consecutive_frame_window(self):
        runner, vision = self.runner()
        runner._relative_leg = runner._relative_references(
            {"intent": "push_distance", "color": "蓝"}, [-20.], -35.)[0]
        frames = [{"sliders_x_mm": {"蓝": -20.}, "stamp_monotonic_s": i + 1}
                  for i in range(8)]
        frames[4] = VisionFrameRejected("rail_registration_patch_disagreement")
        vision.read.side_effect = frames
        result = runner._verify("蓝", -20., 0.)
        self.assertFalse(result["passed"])
        self.assertFalse(result["forced"])
        self.assertEqual(result["reliable_samples"], 3)

    def test_stop_reference_is_checked_before_motion(self):
        runner, vision = self.runner()
        runner._initial_environment = Mock(return_value={
            "sliders_x_mm": {"蓝": -21.0, "橙": 10.0},
            "slider_limits_x_mm": {"蓝": [-48.5, -4.], "橙": [-33., 11.5]}})
        runner.motors = Mock()
        runner._run_leg = Mock()
        vision.stop_reference_x.side_effect = ValueError("missing calibration frames")
        result = runner.run({"intent": "push_to_end", "color": "蓝"})
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "missing calibration frames")
        runner._run_leg.assert_not_called()
