"""Active configuration: known world points pass through the physical camera basis."""
import copy
import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from control.config import load_config
from control.vision import Vision
from vision_tools.pixel_to_world import convert_detections, pixel_to_world

ROOT = Path(__file__).resolve().parents[1]


class ActiveCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.config, _ = load_config(ROOT)
        self.vision = Vision(ROOT, self.config['vision'])
        self.raw = json.loads((ROOT / 'config/camera_world_scheme_e_sensor_raw_20261118.json').read_text())
        self.upright = self.vision.calibration

    def project_sensor(self, point):
        camera = np.asarray(self.raw['R_world_from_camera']).T @ (
            np.asarray(point) - self.raw['C_world_mm'])
        self.assertGreater(camera[2], 0)
        return cv2.projectPoints(camera.reshape(1, 3), np.zeros(3), np.zeros(3),
            np.asarray(self.raw['K']), np.asarray(self.raw['D']))[0].reshape(2)

    def payload(self, points, size=(1920, 1080)):
        pixels = {}
        for color, world in points.items():
            upright = np.array([1919., 1079.]) - self.project_sensor(world)
            scaled = (upright + .5) * (np.asarray(size) / [1920., 1080.]) - .5
            pixels[color] = dict(pixel_xy=scaled.tolist(), reliable=True, reasons=[])
        return dict(schema_version=1, profile='real', input_size_px=list(size),
            position_definition='shaft_axis_intersection_with_image_reference_line',
            sliders=pixels, rail_endpoints_px=None,
            rail_line={'source': 'fitted', 'k': .65, 'm': 100.})

    def test_active_geometry_maps_confirmed_origin_and_outer_stops(self):
        np.testing.assert_allclose(self.vision._world_to_model([65., 26.5, -64.8]), [0, 0, 0])
        np.testing.assert_allclose(self.vision._world_to_model([113.5, 181.5, -64.8]), [-48.5, 155., 0])
        np.testing.assert_allclose(self.vision._world_to_model([53.5, 181.5, -64.8]), [11.5, 155., 0])
        self.assertEqual(self.vision.fixed_limits, {'蓝': [-48.5, -4.], '橙': [-33., 11.5]})

    def test_full_projection_recovers_world_and_task_coordinates_at_both_sizes(self):
        points = {'blue': [113.5, 181.5, -65.], 'yellow': [53.5, 181.5, -65.]}
        for size in [(1920, 1080), (960, 540)]:
            with self.subTest(size=size):
                payload = self.payload(points, size)
                world = convert_detections(payload, self.upright)
                for color, point in points.items():
                    np.testing.assert_allclose(world['sliders'][color]['world_xyz_mm'], point, atol=1e-8, rtol=0)
                observed = self.vision._detect(None, lambda *a, **kw: payload, 'offline')
                np.testing.assert_allclose([observed['sliders_x_mm']['蓝'], observed['sliders_x_mm']['橙']],
                                          [-48.5, 11.5], atol=1e-8, rtol=0)
                self.assertEqual(observed['slider_limits_x_mm'], self.vision.fixed_limits)
                self.assertTrue(all(i['reliable'] for i in observed['reliability'].values()))

    def test_raw_and_upright_calibration_intersect_same_world_point(self):
        point = [92., 181.5, -68.]
        sensor = self.project_sensor(point)
        raw_world = pixel_to_world(sensor.tolist(), calibration=self.raw)['world_xyz_mm']
        upright_world = pixel_to_world((np.array([1919., 1079.])-sensor).tolist(), calibration=self.upright)['world_xyz_mm']
        np.testing.assert_allclose(raw_world, point, atol=1e-8, rtol=0)
        np.testing.assert_allclose(upright_world, point, atol=1e-8, rtol=0)

    def test_explicit_active_pose_is_not_changed_by_historical_slope_refinement(self):
        payload = self.payload({'blue': [100., 181.5, -65.], 'yellow': [70., 181.5, -65.]})
        world = convert_detections(payload, self.upright)
        for key in ['C_world_mm', 'R_world_from_camera', 'K', 'D']:
            self.assertEqual(world['calibration'][key], self.upright[key])
        self.assertEqual(world['coordinate_frame'], self.upright['world_coordinate_frame'])

    def test_camera_rotation_does_not_change_physical_centre_or_world_parity(self):
        np.testing.assert_allclose(np.asarray(self.upright['R_world_from_camera']),
            np.asarray(self.raw['R_world_from_camera']) @ np.diag([-1., -1., 1.]), atol=0, rtol=0)
        self.assertEqual(self.raw['C_world_mm'], self.upright['C_world_mm'])
        self.assertAlmostEqual(np.linalg.det(self.upright['R_world_from_camera']), -1.)


if __name__ == '__main__':
    unittest.main()
