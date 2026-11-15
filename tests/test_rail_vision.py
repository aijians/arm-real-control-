"""Camera-free geometry contract checks: fixed model limits, diagnostic-only rail endpoints."""
import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import cv2

from control.vision import Vision, VisionError, DEFAULT_FIXED_SLIDER_LIMITS_X_MM
from control.tasks import TaskRunner
from control.config import load_config
from vision_tools.pixel_to_world import DEFAULT_CALIBRATION

ROOT = Path(__file__).resolve().parents[1]
FIXED = {"蓝": [-46.56, -2.06], "橙": [-31.06, 13.44]}


def pixel_for_x(x, edge=False):
    cal = DEFAULT_CALIBRATION
    y = cal['rail_lip_y_mm'] if edge else cal['plane_y_mm']
    camera_point = np.array([[[cal['C_world_mm'][1]-y, 0.,
                              189.3+x-cal['C_world_mm'][0]]]], dtype=float)
    pixels, _ = cv2.projectPoints(camera_point, np.zeros(3), np.zeros(3),
                                 np.array(cal['K']), np.array(cal['D']))
    return pixels.reshape(2).tolist()


def detection(shift=0., blue=-41., orange=12., blue_reliable=None):
    sliders = {color: {'pixel_xy': pixel_for_x(x+shift),
                       'bbox': [350., 350., 450., 550.],
                       'part_boxes': {'rod': [350., 350., 450., 450.]}}
               for color, x in [('blue', blue), ('orange', orange)]}
    if blue_reliable is not None:
        sliders['blue'].update(reliable=blue_reliable,
                               reasons=[] if blue_reliable else ['rod_width_inconsistent_with_model'])
    return {'schema_version': 1, 'profile': 'ground', 'input_size_px': [1920,1080],
            'position_definition': 'shaft_axis_intersection_with_image_reference_line',
            'rail_endpoints_px': [pixel_for_x(-50.+shift, edge=True), pixel_for_x(28.+shift, edge=True)],
            'sliders': sliders}


class RailVisionTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT/'config/robot.json').read_text())
        self.vision = Vision(ROOT, self.config['vision'])
        self.image = np.zeros((1080, 1920, 3), np.uint8)

    def observe(self, d):
        return self.vision._detect(self.image, lambda *a, **kw: d, 'offline_test')

    def test_positions_follow_pixels_and_limits_stay_fixed(self):
        for shift in (-1., 0., 1.):
            observed = self.observe(detection(shift))
            self.assertAlmostEqual(observed['sliders_x_mm']['蓝'], -41.+shift, places=8)
            self.assertAlmostEqual(observed['sliders_x_mm']['橙'], 12.+shift, places=8)
            self.assertEqual(observed['slider_limits_x_mm'], FIXED)
            self.assertEqual(observed['rail_observation']['limits_source'], 'fixed_model')
            self.assertTrue(all(r['reliable'] for r in observed['reliability'].values()))

    def test_config_and_code_default_fixed_limits_agree(self):
        self.assertEqual(self.config['vision']['fixed_slider_limits_x_mm'], FIXED)
        self.assertEqual(DEFAULT_FIXED_SLIDER_LIMITS_X_MM, FIXED)

    def test_two_pin_affine_correction_is_not_applied(self):
        # Even a leftover key cannot snap a new installation to old anchors.
        c = copy.deepcopy(self.config['vision'])
        c['rail_x_correction'] = {'scale': 99., 'offset_mm': 500.}
        vision = Vision(ROOT, c)
        result = vision._detect(self.image, lambda *a, **k: detection(), 'offline_test')
        self.assertAlmostEqual(result['sliders_x_mm']['蓝'], -41.)
        self.assertNotIn('rail_x_correction', result)

    def test_missing_endpoints_no_longer_block_planning(self):
        d = detection(); d['rail_endpoints_px'] = None
        observed = self.observe(d)
        self.assertAlmostEqual(observed['sliders_x_mm']['蓝'], -41.)
        self.assertEqual(observed['slider_limits_x_mm'], FIXED)
        self.assertEqual(observed['rail_observation']['endpoint_diagnostic']['reason'],
                         'rail_endpoints_missing')
        goals, _ = TaskRunner(None, self.config).legs({'intent':'push_to_end','color':'蓝'},
            observed['sliders_x_mm'], observed['slider_limits_x_mm'])
        self.assertEqual(goals, [-2.06])

    def test_bad_endpoints_are_reported_only_as_diagnostics(self):
        d = detection(); d['rail_endpoints_px'].reverse()
        result = self.observe(d)
        self.assertEqual(result['slider_limits_x_mm'], FIXED)
        self.assertEqual(result['rail_observation']['endpoint_diagnostic']['reason'],
                         'rail_span_out_of_range')

    def test_edge_and_shaft_have_distinct_projection_planes(self):
        observed = self.observe(detection())
        self.assertNotEqual(DEFAULT_CALIBRATION['rail_lip_y_mm'], DEFAULT_CALIBRATION['plane_y_mm'])
        self.assertAlmostEqual(observed['rail_observation']['endpoint_diagnostic']['span_mm'], 78.)

    def test_detector_reliability_is_forwarded(self):
        result = self.observe(detection(blue_reliable=False))
        self.assertFalse(result['reliability']['蓝']['reliable'])
        self.assertIn('rod_width_inconsistent_with_model', result['reliability']['蓝']['reasons'])
        self.assertTrue(result['reliability']['橙']['reliable'])

    def test_reading_beyond_physical_stop_is_unreliable(self):
        # Orange's fixed end is 13.44; a 2.4 mm residual is tolerated, 2.6 mm is not.
        ok = self.observe(detection(orange=15.84))
        self.assertTrue(ok['reliability']['橙']['reliable'])
        bad = self.observe(detection(orange=16.04))
        self.assertFalse(bad['reliability']['橙']['reliable'])
        self.assertIn('reading_beyond_physical_stop', bad['reliability']['橙']['reasons'])
        self.assertAlmostEqual(bad['reliability']['橙']['beyond_fixed_limits_mm'], 2.6)

    def test_explicit_beyond_stop_tolerance_is_preserved(self):
        config = copy.deepcopy(self.config['vision'])
        config['beyond_stop_tolerance_mm'] = 1.5
        v = Vision(ROOT, config)
        d = detection(orange=15.44)
        result = v._detect(self.image, lambda *a, **kw: d, 'offline_test')
        self.assertFalse(result['reliability']['橙']['reliable'])

    def test_invalid_fixed_limits_are_rejected(self):
        config = copy.deepcopy(self.config['vision'])
        config['fixed_slider_limits_x_mm'] = {'蓝': [-2., -40.], '橙': [-31., 13.]}
        with self.assertRaises(VisionError):
            Vision(ROOT, config)

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

    def test_force_pass_switch_is_accepted_by_config_validation(self):
        from control.config import validate_config
        config = copy.deepcopy(self.config)
        config['verification']['force_pass_unreliable'] = True
        validate_config(config)
        config['verification']['force_pass_unreliable'] = 'yes'
        with self.assertRaises(ValueError):
            validate_config(config)


if __name__=='__main__': unittest.main()
