"""Camera-free geometry contract checks, including installation translation."""
import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from control.vision import Vision, VisionError
from control.tasks import TaskRunner
from control.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def pixel_for_x(x, edge=False):
    # Inverse of the accepted nominal ground camera's rail-Y projection.
    return [960. - (54191.72413793103 if edge else 59764.13793103448) / (x + 164.6), 480.]


def detection(shift=0.):
    return {'scene': 'ground', 'rail_line': [[260., 480.], [690., 480.]],
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
        self.assertNotIn('rail_x_correction', result['raw'])

    def test_no_endpoints_preserves_pin_measurement_but_cannot_plan(self):
        d = detection(); d['rail_endpoints_px'] = None
        observed = self.observe(d)
        self.assertAlmostEqual(observed['sliders_x_mm']['蓝'], -41.)
        self.assertIsNone(observed['slider_limits_x_mm'])
        with self.assertRaisesRegex(Exception, 'slider_limits_unavailable'):
            TaskRunner(None, self.config).legs({'intent':'push_to_end','color':'蓝'},
                observed['sliders_x_mm'], observed['slider_limits_x_mm'])

    def test_reversed_pixel_endpoints_sort_in_arm_coordinates(self):
        d=detection(); d['rail_endpoints_px'].reverse()
        result=self.observe(d)
        self.assertAlmostEqual(result['slider_limits_x_mm']['蓝'][0],-41.)
        self.assertAlmostEqual(result['slider_limits_x_mm']['蓝'][1],3.5)

    def test_edge_and_shaft_have_distinct_projection_planes(self):
        observed=self.observe(detection())
        self.assertAlmostEqual(observed['raw']['rail']['edge_global_y_mm'],173.3)
        self.assertAlmostEqual(observed['raw']['rail']['measured_length_mm'],78.)
        self.assertAlmostEqual(observed['raw']['rail']['length_error_mm'],-.8)

    def test_impossible_insets_fail(self):
        self.vision.center_inset=50.
        with self.assertRaisesRegex(VisionError, 'rail_too_short_for_center_insets'):
            self.observe(detection())

    def test_configuration_loads_without_model_global_endpoints(self):
        config,_=load_config(ROOT)
        self.assertEqual(config['vision']['rail_geometry']['center_inset_mm'],9.)
        for slider in config['sliders'].values():
            self.assertNotIn('x_limits_mm',slider)
            self.assertNotIn('end_x_mm',slider)


if __name__=='__main__': unittest.main()
