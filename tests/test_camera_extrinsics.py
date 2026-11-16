"""Offline projection checks for the measured height and inverted real camera."""
import copy
import unittest
from pathlib import Path

import cv2
import numpy as np

from control.vision import Vision
from vision_tools.pixel_to_world import (
    DEFAULT_CALIBRATION, DEFAULT_REAL_CALIBRATION, convert_detections,
    pixel_to_world, with_intrinsics,
)


ROOT = Path(__file__).resolve().parents[1]


def project(point, calibration, rotation=None):
    r = np.asarray(calibration['R_world_from_camera'] if rotation is None else rotation)
    camera = r.T @ (np.asarray(point) - calibration['C_world_mm'])
    pixels, _ = cv2.projectPoints(camera.reshape(1, 3), np.zeros(3), np.zeros(3),
                                 np.asarray(calibration['K']), np.asarray(calibration['D']))
    return pixels.reshape(2)


def real_payload(size=(1920, 1080), calibration=None):
    # Forward-project using the raw sensor K/D and physical inverted orientation;
    # then apply the image operation performed by detect_frame(real).
    raw = copy.deepcopy(DEFAULT_CALIBRATION)
    if calibration is not None:
        raw['C_world_mm'] = calibration['C_world_mm']
    physical_rotation = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    points = {'blue': [148.3, 183.5, -92.], 'yellow': [201.3, 183.5, -80.]}
    edges = [[129.74, 174.5, -90.], [208.74, 174.5, -87.]]

    def upright_pixel(point):
        sensor = project(point, raw, physical_rotation)
        upright = np.array([1919., 1079.]) - sensor
        return ((upright + .5) * (np.array(size) / [1920., 1080.]) - .5).tolist()

    payload = {'schema_version': 1, 'profile': 'real', 'input_size_px': list(size),
               'position_definition': 'shaft_axis_intersection_with_image_reference_line',
               'sliders': {c: {'pixel_xy': upright_pixel(p)} for c, p in points.items()},
               'rail_endpoints_px': [upright_pixel(p) for p in edges]}
    return payload, points, edges


class CameraExtrinsicsTests(unittest.TestCase):
    def test_height_change_moves_only_z_for_the_same_ground_pixel(self):
        before = copy.deepcopy(DEFAULT_CALIBRATION)
        before['C_world_mm'][2] = -102.
        self.assertEqual(DEFAULT_CALIBRATION['C_world_mm'], [24.5, 99., -108.])
        for pixel in ([460., 420.], [540., 500.], [640., 570.]):
            old = pixel_to_world(pixel, calibration=before)['world_xyz_mm']
            new = pixel_to_world(pixel)['world_xyz_mm']
            np.testing.assert_allclose(np.subtract(new, old), [0., 0., -6.], atol=1e-10)

    def test_raw_projection_then_image_rotation_recovers_world_points(self):
        for size in ((1920, 1080), (960, 540)):
            with self.subTest(size=size):
                payload, points, edges = real_payload(size)
                result = convert_detections(payload)
                self.assertEqual(result['calibration']['C_world_mm'], [24.5, 99., -108.])
                self.assertEqual(result['calibration']['pixel_coordinate_frame'], 'scene_upright')
                for color, expected in points.items():
                    np.testing.assert_allclose(result['sliders'][color]['world_xyz_mm'], expected,
                                               atol=1e-8, rtol=0)
                for side, expected in zip(('left', 'right'), edges):
                    np.testing.assert_allclose(result['rail_endpoints'][side]['world_xyz_mm'], expected,
                                               atol=1e-8, rtol=0)

    def test_real_conversion_does_not_mutate_ground_intrinsics(self):
        ground = copy.deepcopy(DEFAULT_CALIBRATION)
        convert_detections(real_payload()[0])
        self.assertEqual(DEFAULT_CALIBRATION, ground)
        self.assertEqual(ground['K'][0][2], 960.8663237262527)
        self.assertEqual(ground['K'][1][2], 567.6531461824138)
        self.assertEqual(ground['R_world_from_camera'], DEFAULT_REAL_CALIBRATION['R_world_from_camera'])
        self.assertTrue(DEFAULT_REAL_CALIBRATION['is_approximate'])

    def test_explicit_upright_calibration_is_used_without_another_rotation(self):
        calibration = copy.deepcopy(DEFAULT_REAL_CALIBRATION)
        calibration['C_world_mm'][2] = -105.
        payload, points, _ = real_payload(calibration=calibration)
        result = convert_detections(payload, calibration)
        for color, expected in points.items():
            np.testing.assert_allclose(result['sliders'][color]['world_xyz_mm'], expected,
                                       atol=1e-8, rtol=0)
        self.assertEqual(result['calibration']['C_world_mm'][2], -105.)

    def test_ground_or_intrinsics_only_override_cannot_bypass_real_frame_contract(self):
        payload = real_payload()[0]
        with self.assertRaisesRegex(ValueError, 'scene_upright'):
            convert_detections(payload, DEFAULT_CALIBRATION)
        intrinsics = {'camera_matrix': DEFAULT_CALIBRATION['K'],
                      'distortion_coefficients': DEFAULT_CALIBRATION['D'],
                      'image_width': 1920, 'image_height': 1080}
        with self.assertRaisesRegex(ValueError, '完整标定'):
            convert_detections(payload, with_intrinsics(intrinsics))

    def test_task_vision_uses_real_default_with_raw_or_already_upright_input(self):
        payload, _, _ = real_payload()
        for already_rotated in (False, True):
            with self.subTest(already_rotated=already_rotated):
                config = {'profile': 'real', 'already_rotated': already_rotated}
                vision = Vision(ROOT, config)

                def detector(image, **kwargs):
                    self.assertEqual(kwargs, config)
                    return payload

                result = vision._detect(None, detector, 'offline')
                np.testing.assert_allclose([result['sliders_x_mm']['蓝'],
                                            result['sliders_x_mm']['橙']], [-41., 12.], atol=1e-8)


if __name__ == '__main__':
    unittest.main()
