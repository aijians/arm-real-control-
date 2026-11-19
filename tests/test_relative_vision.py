"""Offline checks for registered measurements and physical cylinder tangency."""

import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from control.tasks import TaskRunner
from control.vision import Vision, VisionError, VisionFrameRejected
from vision_tools.detect_sliders import _refine_orange_mask_edge, _shift_roi, detect_frame
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
        measured, warp, _ = tracker.estimate_shift(moved)
        measured[:] = 0.
        np.testing.assert_allclose(warp[:, 2], shift, atol=.13)
        _, result = tracker.align(moved)
        np.testing.assert_allclose(result["shift_from_reference_px"], shift, atol=.13)
        moved[29:114, 22:121] = 128
        with self.assertRaises(RailRegistrationError):
            tracker.align(moved)

    def test_registered_crop_keeps_rod_past_fixed_right_edge(self):
        image = np.full((1080, 1920, 3), 35, np.uint8)
        image[150:248, 668:692] = (30, 140, 240)
        moved = cv2.warpAffine(image, np.float32([[1, 0, 10], [0, 1, -4]]), (1920, 1080))
        with patch("vision_tools.detect_sliders._detect_rail_endpoints", return_value=None) as endpoints:
            result = detect_frame(moved, profile="real", already_rotated=True,
                                  reference_rail=(.7, -210.), reference_shift=(10, -4))
        orange = result["sliders"]["yellow"]
        self.assertTrue(orange["reliable"], orange["reasons"])
        self.assertGreater(orange["measurement"]["rows_used"], 20)
        self.assertAlmostEqual(orange["pixel_xy"][0], 689.5)
        self.assertNotIn("rod_touches_roi_boundary", orange["reasons"])
        self.assertEqual(endpoints.call_args.args[-1]["roi"], (430, 76, 710, 361))
        self.assertEqual(_shift_roi((2, 10, 20, 24), -3.2, 2.8, 25, 26), (0, 13, 17, 26))

    def test_orange_edge_uses_first_crossing_and_keeps_morphological_boundary(self):
        low = np.array([[10, 50, 10]])
        high = np.array([[40, 30, 50]])
        tests = ((low, 20, 1), (high, 35, -1))
        self.assertAlmostEqual(_refine_orange_mask_edge(tests, 1, 0, -1), 0.5)
        self.assertAlmostEqual(_refine_orange_mask_edge(tests, 1, 0, 1), 1.25)
        self.assertAlmostEqual(_refine_orange_mask_edge(tests, 0, 0, 1), 0.5)


class TestRegisteredVision(unittest.TestCase):
    def vision(self):
        vision = Vision(ROOT, json.loads((ROOT / "config/robot.json").read_text())["vision"])
        vision.relative_measurement = True
        return vision

    def test_injected_detector_gets_canonical_shift_and_native_points_keep_missing_end(self):
        vision = self.vision()
        vision.registration = Mock(size=(960, 540))
        vision.registration.estimate_shift.return_value = ((2., -3.), None, {"ecc": 1.})
        vision.reference_rail = (.6, -70.)
        vision.radius = 4.
        image = np.zeros((540, 960, 3), np.uint8)
        detector = Mock(return_value={
            "rail_line": None, "rail_endpoints_px": [[320., 135.], None],
            "sliders": {"blue": {"pixel_xy": [250., 113.], "measurement": None},
                        "yellow": {"pixel_xy": None, "measurement": {"edge_pixels": [[331., 111.]]}}}})
        result = {"units": "mm", "coordinate_frame": vision.world_frame,
                  "position_definition": "shaft_axis_reference_point",
                  "sliders": {"blue": {"world_xyz_mm": [105., 181.5, 0.]},
                              "yellow": {"world_xyz_mm": [54., 181.5, 0.], "occluded": True}}}
        with patch("control.vision.convert_detections", return_value=result) as convert:
            observation = vision._detect(image, detector, "test", upright=True)
        detector.assert_called_once_with(image, profile="real", already_rotated=True,
            use_tangent=True, reference_rail=(.6, -70.), reference_shift=(4., -6.))
        payload = convert.call_args.args[0]
        self.assertEqual(payload["sliders"]["blue"]["pixel_xy"], [248., 116.])
        self.assertEqual(payload["sliders"]["yellow"]["measurement"]["edge_pixels"], [[329., 114.]])
        self.assertEqual(observation["rail_observation"]["endpoints_px"], [[318., 138.], None])
        self.assertIsNone(observation["rail_observation"]["rail_line"])
        self.assertIs(observation["reliability"]["橙"]["occluded"], True)
        self.assertIs(observation["reliability"]["蓝"]["occluded"], False)

    def test_stale_frame_is_consumed_and_logs_latency_as_hard_error(self):
        vision = self.vision()
        vision.max_age = 1.
        vision.camera = Mock()
        vision.camera.frame.return_value = {"image": None, "sequence": 23, "received_monotonic_s": 4.}
        for optical_reject in (None, VisionFrameRejected("rail_registration_patch_disagreement")):
            with self.subTest(optical_reject=optical_reject):
                vision._detect = Mock(return_value={}, side_effect=optical_reject)
                with patch("control.vision.time.monotonic", side_effect=[4.0, 4.1, 5.5]), patch("builtins.print") as log:
                    with self.assertRaisesRegex(VisionError, "^camera_frame_stale_after_detection$") as exc:
                        vision.read()
                self.assertNotIsInstance(exc.exception, VisionFrameRejected)
                self.assertEqual(vision.last_sequence, 23)
                self.assertIn("age=1.500s detection=1.400s max_age=1.000s", log.call_args.args[0])


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

    def test_staleness_aborts_verification_without_consuming_more_reads_or_retries(self):
        runner, vision = self.runner()
        runner._relative_leg = runner._relative_references(
            {"intent": "push_distance", "color": "蓝"}, [-20.], -35.)[0]
        runner.motors = Mock()
        runner.motors.halt.return_value = {"stopped_confirmed": True}
        record = {}
        vision.read.side_effect = VisionError("camera_frame_stale_after_detection")
        with self.assertRaisesRegex(VisionError, "^camera_frame_stale_after_detection$"):
            runner._stage("verification", {}, record, lambda _: runner._verify("蓝", -20., 0.))
        vision.read.assert_called_once()
        self.assertEqual(len(record["stage_attempts"]["verification"]), 1)

    def test_non_target_drift_uses_reliable_occluded_readings_and_logs_both_reasons(self):
        runner, vision = self.runner()
        runner.reference_positions = {"橙": 1.0}
        runner.stationary_drift_mm = 0.5
        runner._relative_leg = runner._relative_references(
            {"intent": "push_distance", "color": "蓝"}, [-20.], -35.)[0]
        state = {"sliders_x_mm": {"蓝": -20., "橙": 3.},
                 "stamp_monotonic_s": 1.,
                 "reliability": {"橙": {"reliable": True, "occluded": True}}}
        vision.read.return_value = state
        rejected = runner._verify("蓝", -20., 0.)
        self.assertEqual(rejected["unreliable_reasons"], [
            "stationary_slider_drift:橙:2.000", "stationary_slider_occluded_drift:橙:2.000"])
        self.assertFalse(rejected["passed"])
        state["reliability"]["橙"]["reliable"] = False
        accepted = runner._verify("蓝", -20., 0.)
        self.assertTrue(accepted["passed"])
        self.assertEqual(accepted["samples"][0]["reasons"], ["stationary_slider_unreliable:橙"])

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
