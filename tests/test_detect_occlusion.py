"""Synthetic-image checks of the rod measurement band, occlusion handling and reliability."""
import unittest

import numpy as np

from vision_tools.detect_sliders import SCENES, detect_frame

CFG = SCENES["ground"]
X0, Y0, K = CFG["rail_nominal"]
BLUE_BGR = (200, 60, 20)
ORANGE_BGR = (30, 140, 240)


def rail_y(u):
    return Y0 + K * (u - X0)


def model_width(color, u):
    a, b = CFG["width_model_px"][color]
    return a + b * u


def draw_rod(image, u, width, bgr):
    left, right = int(round(u - (width - 1) / 2.)), int(round(u + (width - 1) / 2.))
    for x in range(left, right + 1):
        top = int(rail_y(x) - 110)
        bottom = int(rail_y(x) - 1)
        image[top:bottom, x] = bgr
    return left, right


def scene():
    image = np.full((1080, 1920, 3), 35, np.uint8)
    for x in range(250, 700):
        y = rail_y(x)
        image[int(y) - 12:int(y), x] = 205      # bright front lip above the edge line
    return image


def detect(image):
    return detect_frame(image, profile="ground")


class OcclusionTests(unittest.TestCase):
    def test_separate_rods_use_both_edges(self):
        image = scene()
        draw_rod(image, 451., model_width("blue", 451.), BLUE_BGR)
        draw_rod(image, 624., model_width("yellow", 624.), ORANGE_BGR)
        result = detect(image)
        for key, u in (("blue", 451.), ("yellow", 624.)):
            item = result["sliders"][key]
            self.assertEqual(item["reference_method"], "rod_band_both_edges")
            self.assertTrue(item["reliable"], item["reasons"])
            self.assertLess(abs(item["pixel_xy"][0] - u), 0.75)

    def test_front_blue_hiding_orange_left_side_uses_orange_far_edge(self):
        image = scene()
        orange_u, blue_u = 624., 596.
        draw_rod(image, orange_u, model_width("yellow", orange_u), ORANGE_BGR)
        draw_rod(image, blue_u, model_width("blue", blue_u), BLUE_BGR)   # drawn last: in front
        result = detect(image)
        orange = result["sliders"]["yellow"]
        self.assertEqual(orange["reference_method"], "rod_band_far_edge_width_model")
        self.assertEqual(orange["measurement"]["occluded_side"], "left")
        self.assertTrue(orange["reliable"], orange["reasons"])
        self.assertLess(abs(orange["pixel_xy"][0] - orange_u), 1.0)
        visible_mid = 0.5 * (blue_u + (model_width("blue", blue_u) - 1) / 2. + 1
                             + orange_u + (model_width("yellow", orange_u) - 1) / 2.)
        self.assertGreater(visible_mid - orange_u, 3.0)
        blue = result["sliders"]["blue"]
        self.assertEqual(blue["reference_method"], "rod_band_both_edges")
        self.assertLess(abs(blue["pixel_xy"][0] - blue_u), 0.75)

    def test_far_edge_also_hidden_is_unreliable(self):
        image = scene()
        orange_u, blue_u = 624., 596.
        _, right = draw_rod(image, orange_u, model_width("yellow", orange_u), ORANGE_BGR)
        draw_rod(image, blue_u, model_width("blue", blue_u), BLUE_BGR)
        image[300:600, right - 10:right + 30] = (90, 90, 90)
        orange = detect(image)["sliders"]["yellow"]
        self.assertFalse(orange["reliable"])

    def test_partly_covered_front_rod_is_unreliable(self):
        image = scene()
        left, right = draw_rod(image, 596., model_width("blue", 596.), BLUE_BGR)
        draw_rod(image, 640., model_width("yellow", 640.), ORANGE_BGR)
        image[300:600, left:left + 12] = (230, 230, 230)
        blue = detect(image)["sliders"]["blue"]
        self.assertFalse(blue["reliable"])
        self.assertIn("rod_width_inconsistent_with_model", blue["reasons"])

    def test_output_keeps_schema_for_pixel_to_world(self):
        image = scene()
        draw_rod(image, 451., model_width("blue", 451.), BLUE_BGR)
        draw_rod(image, 624., model_width("yellow", 624.), ORANGE_BGR)
        result = detect(image)
        self.assertEqual(result["position_definition"], "shaft_axis_intersection_with_image_reference_line")
        self.assertIn(result["rail_line"]["source"], ("fitted", "nominal_fallback"))
        for item in result["sliders"].values():
            self.assertFalse(item["approximate"])

    def test_glare_and_bracket_bottom_distractors_do_not_corrupt_rail_line(self):
        image = scene()
        image[140:195, 580:700] = 250
        image[515:545, 400:530] = 210
        draw_rod(image, 451., model_width("blue", 451.), BLUE_BGR)
        draw_rod(image, 624., model_width("yellow", 624.), ORANGE_BGR)
        result = detect(image)
        self.assertEqual(result["rail_line"]["source"], "fitted")
        self.assertAlmostEqual(result["rail_line"]["k"], K, delta=0.025)
        for key, u in (("blue", 451.), ("yellow", 624.)):
            item = result["sliders"][key]
            self.assertTrue(item["reliable"], item["reasons"])
            self.assertLess(abs(item["pixel_xy"][0] - u), 0.75)


if __name__ == "__main__":
    unittest.main()
