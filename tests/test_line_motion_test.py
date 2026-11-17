"""Offline checks of manual Cartesian legs and fail-stop sequencing."""
from contextlib import nullcontext
import math
from pathlib import Path
import tempfile
import unittest

from line_motion_test import LineTestPlan, MOTOR_TARGETS, execute


ROOT = Path(__file__).resolve().parents[1]


class FakePair:
    def __init__(self, plan, fail_at=None):
        self.q = list(plan.home)
        self.moves = []
        self.fail_at = fail_at
        self.exited_with_error = False

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.exited_with_error = kind is not None

    def read_state(self):
        return {'q_rad': list(self.q)}

    def move(self, phase):
        # Every leg must start at fresh feedback, including after home acceptance.
        assert phase['points'][0]['q_rad'] == self.q
        self.moves.append(phase)
        if len(self.moves) == self.fail_at:
            raise RuntimeError('simulated_feedback_failure')
        self.q = list(phase['points'][-1]['q_rad'])
        if phase['route'] == 'return_home':
            self.q[0] -= math.radians(0.1)
        return {'status': 'completed', 'feedback': {'q_rad': list(self.q)}}


class LineMotionTestTests(unittest.TestCase):
    def setUp(self):
        self.plan = LineTestPlan(ROOT)

    def test_exact_motor_targets_and_no_double_offset(self):
        preview = self.plan.preview()
        for index, expected in zip((0, 1, 4, 5), MOTOR_TARGETS):
            for actual, target in zip(preview[index]['target_motor_deg'], expected):
                self.assertAlmostEqual(actual, target, places=9)
        self.assertEqual(preview[3]['target_model_deg'], [180.0, 4.0])

    def test_cartesian_line_and_command_chords(self):
        for start, goal, face in ((self.plan.targets[0], self.plan.targets[1], '+x'),
                                  (self.plan.targets[2], self.plan.targets[3], '-x')):
            a, b = (self.plan.arm.slider_center(q, face) for q in (start, goal))
            phase = self.plan.phase('test', self.plan.line(start, goal, face))
            points = phase['points']
            self.assertEqual(points[-1]['q_rad'], goal)
            for left, right in zip(points, points[1:]):
                self.assertGreater(right['time_s'], left['time_s'])
                self.assertLessEqual(max(abs(math.degrees(x-y)) for x, y in
                                         zip(left['q_rad'], right['q_rad'])), .2+1e-9)
                for t in (0., .5, 1.):
                    q = [x+(y-x)*t for x, y in zip(left['q_rad'], right['q_rad'])]
                    x, y = self.plan.arm.slider_center(q, face)
                    error = abs((b[0]-a[0])*(y-a[1])-(b[1]-a[1])*(x-a[0]))/math.dist(a, b)
                    self.assertLess(error, .01)

    def test_full_sequence_and_stop_after_failure(self):
        for failure in (None, 2):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                pair = FakePair(self.plan, failure)
                record = {'stages': []}
                kwargs = dict(pair_factory=lambda _: pair, lock_factory=lambda _: nullcontext())
                if failure:
                    with self.assertRaisesRegex(RuntimeError, 'simulated_feedback_failure'):
                        execute(self.plan, record, Path(directory)/'log.json', **kwargs)
                    self.assertEqual(len(pair.moves), 2)
                    self.assertTrue(pair.exited_with_error)
                    self.assertEqual(record['stages'][-1]['status'], 'failed')
                else:
                    execute(self.plan, record, Path(directory)/'log.json', **kwargs)
                    self.assertEqual(len(pair.moves), 6)
                    self.assertEqual(pair.moves[3]['route'], 'return_home')
                    self.assertEqual(pair.q, self.plan.targets[-1])

    def test_runtime_lock_failure_prevents_device_construction(self):
        def busy(_):
            raise RuntimeError('runtime_already_running')
        def forbidden(_):
            self.fail('busy lock must prevent serial device construction')
        with self.assertRaisesRegex(RuntimeError, 'runtime_already_running'):
            execute(self.plan, {'stages': []}, Path('/tmp/unused-line-test.json'),
                    pair_factory=forbidden, lock_factory=busy)


if __name__ == '__main__':
    unittest.main()
