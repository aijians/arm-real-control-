"""Finite back-face coverage and full orange push/escape regression. No IO."""
import copy
import math
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from control.planner import Planner, PlanError
from control.scene import SimpleScene

ROOT = Path(__file__).resolve().parents[1]
POSITIONS = {'蓝': -45.456585375516966, '橙': 11.336144172773373}
LIMITS = {'蓝': [-46.56, -2.06], '橙': [-31.06, 13.44]}


class DeepContactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = Planner(ROOT)
        cls.seed = list(map(math.radians, [170., 4.]))
        cls.pre = cls.p.precontact('橙', '-x', POSITIONS, cls.seed)
        cls.push = cls.p.push(cls.pre, '橙', -31.06, POSITIONS, LIMITS['橙'])

    def test_full_command_chords_cover_stem_and_keep_wall_clear(self):
        p = self.p
        points = self.push['points']
        segment = p.config['kinematics']['faces']['-x']['contact_segment_canonical_xy_mm']
        self.assertTrue(self.push['collision_checked'])
        self.assertLess(self.push['command_chord_cartesian_error_limit_mm'], p.scene.contact_tolerance)
        for a, b in zip(points, points[1:]):
            for t in np.linspace(0, 1, 5):
                # A4 sends .01 degree motor angles; -3/+2.25 offsets are
                # exactly representable at this resolution too.
                q = [math.radians(round(math.degrees(x+(y-x)*t), 2))
                     for x, y in zip(a['q_rad'], b['q_rad'])]
                centre = p.arm.slider_center(q, '-x')
                pin = min(POSITIONS['橙'], centre[0])
                self.assertIsNone(p.scene.check(q, {**POSITIONS, '橙': [pin, 155.]},
                                               contact={'color': '橙', 'direction': '-x'}))
                ends = [p.arm.forward(q, v)[1] for v in segment]
                self.assertGreaterEqual(max(ends), 160.)
                self.assertLessEqual(min(ends), 150.)
                slot = np.concatenate([poly for item, poly in p.scene.transformed_bodies(q)
                                       if item.get('contact_face')])
                self.assertGreater(167.-slot[:, 1].max(), 2.9)
        self.assertAlmostEqual(p.arm.slider_center(points[-1]['q_rad'], '-x')[0], -31.06)

    def test_old_shallow_point_is_rejected_with_depth_guard(self):
        cfg = copy.deepcopy(self.p.config)
        cfg['kinematics']['faces']['-x']['point_canonical_xy_mm'] = [135.94, 40.58]
        p = Planner(ROOT, cfg)
        pre = p.precontact('橙', '-x', POSITIONS, self.seed)
        with self.assertRaisesRegex(PlanError, 'insufficient_finite_face_stem_coverage'):
            p.push(pre, '橙', -31.06, POSITIONS, LIMITS['橙'])

    def test_uncertified_decimation_cannot_bypass_final_collision_check(self):
        def endpoints(name, points, *args):
            return [points[0], points[-1]], [0, len(points)-1]
        with patch.object(self.p, '_command_points', side_effect=endpoints):
            with self.assertRaisesRegex(PlanError, 'collision:'):
                self.p.push(self.pre, '橙', -31.06, POSITIONS, LIMITS['橙'])

    def test_corrected_obstacle_keeps_height_seat_and_plate(self):
        p = self.p
        obstacles = {o['name']: o for o in p.scene.config['obstacles']}
        self.assertAlmostEqual(obstacles['mass_block']['size_mm'][0], 14.776737891089823)
        self.assertEqual(obstacles['mass_block']['center_mm'][2], 27.5)
        self.assertEqual(obstacles['mass_block']['size_mm'][2], 10.)
        self.assertIn('mounting_seat', obstacles)
        self.assertIn('installation_plate', obstacles)
        old = copy.deepcopy(p.scene.config)
        block = next(o for o in old['obstacles'] if o['name'] == 'mass_block')
        block['center_mm'] = [40., 93.86, 27.5]
        block['size_mm'] = [20., 15., 10.]
        scene = SimpleScene(old)
        self.assertTrue(any(scene.check(point['q_rad']) for point in self.push['points']))

    def test_free_push_retreat_and_no_progress_retry_remain_clear(self):
        p = self.p
        p.free(self.seed, self.pre, POSITIONS, direction='-x')
        end = self.push['points'][-1]['q_rad']
        retreat = p.observation_retreat(end, '橙', '-x', POSITIONS, LIMITS['橙'], all_slider_limits=LIMITS)
        clear = retreat['points'][-1]['q_rad']
        for color, bounds in LIMITS.items():
            margins = p.observation_pose(clear, color, '-x', POSITIONS, bounds)
            self.assertGreater(margins['minimum_sweep_margin_mm'], 0.)
            self.assertGreater(margins['minimum_projected_margin_mm'], 0.)
        p.free(clear, self.pre, POSITIONS, direction='-x')
        with self.assertRaises(PlanError):
            p._checked_observation_edges([point['q_rad'] for point in retreat['points']][::-1],
                                        '橙', '-x', POSITIONS, LIMITS['橙'], 155.)

    def test_left_correction_and_fixed_endpoint_retry_keep_checks(self):
        p = self.p
        q = p.arm.inverse([-20., 155.], '-x', seed=self.pre)
        phase = p.correction(q, '橙', -31.06, '-x', POSITIONS, LIMITS['橙'], 1.)
        partial = p.arm.inverse([-20.4, 155.], '-x', seed=q)
        retry = p.retry_contact(phase, partial, POSITIONS)
        self.assertTrue(phase['collision_checked'])
        self.assertTrue(retry['collision_checked'])
        self.assertEqual(retry['points'][-1]['q_rad'], phase['points'][-1]['q_rad'])


if __name__ == '__main__':
    unittest.main()
