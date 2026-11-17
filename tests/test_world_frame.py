"""Measured mirrored world, unchanged arm frame, and the physical 180° camera."""
import copy
import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from control.vision import Vision, VisionError
from vision_tools.pixel_to_world import (
    DEFAULT_CALIBRATION, convert_detections, pixel_to_world,
)

ROOT = Path(__file__).resolve().parents[1]
CAL = json.loads((ROOT / 'config/camera_world_20261117.json').read_text())


def candidate_config():
    return dict(profile='real', already_rotated=False,
                calibration='config/camera_world_20261117.json',
                shoulder_global_xyz_mm=[65., 26.5, -64.5],
                world_to_model_axis_signs=[-1, 1, 1])


def raw_then_upright(point, size=(1920, 1080)):
    # Independent physical axes: sensor right -> world +Y, sensor down -> +Z,
    # optical forward -> -X. Use the original sensor K/D, then rotate the image.
    dx, dy, dz = np.subtract(point, CAL['C_world_mm'])
    camera = np.array([[dy, dz, -dx]], dtype=float)
    sensor, _ = cv2.projectPoints(camera, np.zeros(3), np.zeros(3),
                                 np.array(DEFAULT_CALIBRATION['K']),
                                 np.array(DEFAULT_CALIBRATION['D']))
    upright = np.array([1919., 1079.]) - sensor.reshape(2)
    return ((upright + .5) * (np.array(size) / [1920., 1080.]) - .5).tolist()


class MeasuredWorldTests(unittest.TestCase):
    def test_new_world_and_camera_basis_are_orthogonal_with_declared_parity(self):
        r = np.array(CAL['R_world_from_camera'])
        np.testing.assert_array_equal(r.T @ r, np.eye(3))
        self.assertAlmostEqual(np.linalg.det(r), -1.)
        # Composing world->arm restores the same right-handed camera->arm basis.
        np.testing.assert_array_equal(np.diag([-1, 1, 1]) @ r,
                                      [[0, 0, 1], [-1, 0, 0], [0, -1, 0]])

    def test_raw_sensor_rotation_recovers_new_measured_world_and_old_arm_axes(self):
        vision = Vision(ROOT, candidate_config())
        points = {'blue': [113.5, 181.5, -65.], 'yellow': [53.5, 181.5, -65.]}
        edges = [[122.5, 172.5, -67.], [44.5, 172.5, -67.]]
        for size in ((1920, 1080), (960, 540)):
            with self.subTest(size=size):
                payload = dict(schema_version=1, profile='real', input_size_px=list(size),
                    position_definition='shaft_axis_intersection_with_image_reference_line',
                    sliders={c: dict(pixel_xy=raw_then_upright(p, size)) for c, p in points.items()},
                    rail_endpoints_px=[raw_then_upright(p, size) for p in edges])
                world = convert_detections(payload, CAL)
                for color, expected in points.items():
                    np.testing.assert_allclose(world['sliders'][color]['world_xyz_mm'],
                                               expected, atol=1e-8, rtol=0)
                obs = vision._detect(None, lambda *a, **kw: payload, 'offline')
                np.testing.assert_allclose([obs['sliders_x_mm']['蓝'], obs['sliders_x_mm']['橙']],
                                           [-48.5, 11.5], atol=1e-8, rtol=0)
                self.assertAlmostEqual(obs['rail_observation']['endpoint_diagnostic']['span_mm'], 78.)
                self.assertEqual(world['coordinate_frame'], CAL['world_coordinate_frame'])

    def test_origin_and_known_points_in_unchanged_arm_local_frame(self):
        vision = Vision(ROOT, candidate_config())
        np.testing.assert_allclose(vision._world_to_model([65., 26.5, -64.5]), [0, 0, 0])
        np.testing.assert_allclose(vision._world_to_model([113.5, 181.5, -64.5]), [-48.5, 155., 0])
        np.testing.assert_allclose(vision._world_to_model([53.5, 181.5, -64.5]), [11.5, 155., 0])

    def test_mirror_requires_explicit_handedness_and_distinct_world_name(self):
        for changes in ({'world_handedness': 'right'}, {'world_handedness': 'unknown'},
                        {'world_coordinate_frame': 'world_equals_global_z_up'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                pixel_to_world([500, 350], calibration={**CAL, **changes})
        legacy = copy.deepcopy(DEFAULT_CALIBRATION)
        legacy['R_world_from_camera'][0][2] = -1.
        with self.assertRaises(ValueError):
            pixel_to_world([500, 350], calibration=legacy)

    def test_world_arm_parity_or_ambiguous_origin_is_rejected(self):
        for changes in ({'world_to_model_axis_signs': [1, 1, 1]},
                        {'world_to_model_axis_signs': [-2, 1, 1]},
                        {'shoulder_global_xy_mm': [65., 26.5]}):
            with self.subTest(changes=changes), self.assertRaises(VisionError):
                Vision(ROOT, {**candidate_config(), **changes})

    def test_explicit_measured_pose_is_not_replaced_by_old_image_slope_branch(self):
        point = [100., 181.5, -65.]
        payload = dict(schema_version=1, profile='real', input_size_px=[1920, 1080],
            position_definition='shaft_axis_intersection_with_image_reference_line',
            sliders={'blue': {'pixel_xy': raw_then_upright(point)}},
            rail_line={'source': 'fitted', 'k': .65, 'm': 100.})
        converted = convert_detections(payload, CAL)
        np.testing.assert_allclose(converted['sliders']['blue']['world_xyz_mm'], point, atol=1e-8)
        self.assertEqual(converted['calibration']['R_world_from_camera'], CAL['R_world_from_camera'])
        self.assertEqual(converted['calibration']['C_world_mm'], [228.99, 97., -104.22])


if __name__ == '__main__':
    unittest.main()
