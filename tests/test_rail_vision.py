"""Camera-free geometry contract checks, including installation translation."""
import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import cv2

from control.vision import Vision, VisionError
from control.tasks import TaskRunner
from control.config import load_config
from vision_tools.pixel_to_world import DEFAULT_CALIBRATION

ROOT = Path(__file__).resolve().parents[1]


def pixel_for_x(x, edge=False):
    cal = DEFAULT_CALIBRATION
    y = cal['rail_lip_y_mm'] if edge else cal['plane_y_mm']
    camera_point = np.array([[[cal['C_world_mm'][1]-y, 0.,
                              189.3+x-cal['C_world_mm'][0]]]], dtype=float)
    pixels, _ = cv2.projectPoints(camera_point, np.zeros(3), np.zeros(3),
                                 np.array(cal['K']), np.array(cal['D']))
    return pixels.reshape(2).tolist()


def detection(shift=0.):
    return {'schema_version': 1, 'profile': 'ground', 'input_size_px': [1920,1080],
            'position_definition': 'shaft_axis_intersection_with_image_reference_line',
            'rail_endpoints_px': [pixel_for_x(-50.+shift, edge=True), pixel_for_x(28.+shift, edge=True)],
            'sliders': {color: {'pixel_xy': pixel_for_x(x+shift),
                'bbox': [350., 350., 450., 550.],
                'part_boxes': {'rod': [350., 350., 450., 450.]}}
                for color, x in [('blue', -41.), ('orange', 19.)]}}


class RailVisionTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT/'config/robot.json').read_text())
        self.vision = Vision(ROOT, self.config['vision'])
        self.image = np.zeros((1080, 1920, 3), np.uint8)

    def observe(self, d):
        return self.vision._detect(self.image, lambda *a, **kw: d, 'offline_test')

    def test_limits_follow_actual_edges_with_nine_mm_inset(self):
        for shift in (-8., 0., 6.5):
            observed = self.observe(detection(shift))
            self.assertAlmostEqual(observed['sliders_x_mm']['蓝'], -41.+shift, places=8)
            self.assertAlmostEqual(observed['sliders_x_mm']['橙'], 19.+shift, places=8)
            for color, expected in [('蓝', [-41., 3.5]), ('橙', [-25.5, 19.])]:
                limits=observed['slider_limits_x_mm'][color]
                self.assertAlmostEqual(limits[0], expected[0]+shift, places=8)
                self.assertAlmostEqual(limits[1], expected[1]+shift, places=8)

    def test_two_pin_affine_correction_is_not_applied(self):
        # Even a leftover key cannot snap a new installation to old anchors.
        c = copy.deepcopy(self.config['vision'])
        c['rail_x_correction'] = {'scale': 99., 'offset_mm': 500.}
        vision = Vision(ROOT, c)
        result = vision._detect(self.image, lambda *a, **k: detection(), 'offline_test')
        self.assertAlmostEqual(result['sliders_x_mm']['蓝'], -41.)
        self.assertNotIn('rail_x_correction', result)

    def test_no_endpoints_preserves_pin_measurement_but_cannot_plan(self):
        d = detection(); d['rail_endpoints_px'] = None
        observed = self.observe(d)
        self.assertAlmostEqual(observed['sliders_x_mm']['蓝'], -41.)
        self.assertIsNone(observed['slider_limits_x_mm'])
        self.assertEqual(observed['rail_observation']['reason'], 'rail_endpoints_missing')
        with self.assertRaisesRegex(Exception, 'slider_limits_unavailable'):
            TaskRunner(None, self.config).legs({'intent':'push_to_end','color':'蓝'},
                observed['sliders_x_mm'], observed['slider_limits_x_mm'])

    def test_reversed_pixel_endpoints_are_rejected(self):
        d=detection(); d['rail_endpoints_px'].reverse()
        result=self.observe(d)
        self.assertIsNone(result['slider_limits_x_mm'])
        self.assertEqual(result['rail_observation']['reason'], 'rail_span_out_of_range')

    def test_edge_and_shaft_have_distinct_projection_planes(self):
        observed=self.observe(detection())
        self.assertNotEqual(DEFAULT_CALIBRATION['rail_lip_y_mm'],DEFAULT_CALIBRATION['plane_y_mm'])
        self.assertAlmostEqual(observed['rail_observation']['span_mm'],78.)

    def test_impossible_insets_fail(self):
        self.vision.center_inset_mm=50.
        result=self.observe(detection())
        self.assertIsNone(result['slider_limits_x_mm'])
        self.assertEqual(result['rail_observation']['reason'], 'rail_too_short_for_center_insets')

    def test_near_end_pin_measurement_does_not_expand_motion_limits(self):
        d=detection()
        # Orange's nominal upper bound is 19; allow a measured 2.05 mm residual.
        d['sliders']['orange']['pixel_xy']=pixel_for_x(21.05)
        result=self.observe(d)
        self.assertAlmostEqual(result['rail_observation']['outside_by_mm']['橙'],2.05)
        self.assertAlmostEqual(result['slider_limits_x_mm']['蓝'][1],3.5)
        self.assertAlmostEqual(result['slider_limits_x_mm']['橙'][1],19.)

    def test_large_residual_and_truncated_rail_still_fail(self):
        for endpoint, orange, reason in [(28.,21.51,'slider_position_outside_limits'),
                                         (0.,19.,'rail_span_out_of_range')]:
            d=detection()
            d['rail_endpoints_px'][1]=pixel_for_x(endpoint,edge=True)
            d['sliders']['orange']['pixel_xy']=pixel_for_x(orange)
            result=self.observe(d)
            self.assertIsNone(result['slider_limits_x_mm'])
            self.assertEqual(result['rail_observation']['reason'],reason)

    def test_explicit_measurement_tolerance_is_preserved(self):
        config=copy.deepcopy(self.config['vision'])
        config['rail_geometry']['limit_position_tolerance_mm']=1.5
        v=Vision(ROOT,config)
        d=detection();d['sliders']['orange']['pixel_xy']=pixel_for_x(21.)
        self.assertIsNone(v._detect(self.image, lambda *a,**kw:d,'offline_test')['slider_limits_x_mm'])

    def test_live_report_keeps_frame_identity_for_video_diagnosis(self):
        from unittest.mock import Mock
        self.vision.camera=Mock()
        self.vision.camera.frame.return_value={'image':self.image,'sequence':61,
            'received_monotonic_s':100.,'pipeline_pts_ns':2000000000}
        with patch('control.vision.time.monotonic',return_value=100.1), \
                patch('control.vision.detect_frame',return_value=detection()):
            result=self.vision.read(after_monotonic_s=99.9)
        self.assertEqual(result['rail_observation']['frame_sequence'],61)
        self.assertEqual(result['rail_observation']['pipeline_pts_ns'],2000000000)
        self.vision.camera.frame.assert_called_once_with(0,99.9,self.vision.timeout,self.vision.max_age)

    def test_configuration_loads_without_model_global_endpoints(self):
        config,_=load_config(ROOT)
        self.assertEqual(config['vision']['rail_geometry']['center_inset_mm'],9.)
        for slider in config['sliders'].values():
            self.assertNotIn('x_limits_mm',slider)
            self.assertNotIn('end_x_mm',slider)


if __name__=='__main__': unittest.main()
