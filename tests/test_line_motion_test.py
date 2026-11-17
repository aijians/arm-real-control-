"""Offline checks of manual Cartesian legs and fail-stop sequencing."""
from contextlib import nullcontext
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from line_motion_test import LineTestPlan, TUNER_TARGETS, LineTestMotorPair, execute
from virtual_drive import VirtualDrive


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

    def test_targets_match_pid_tuner_display_and_motor_commands(self):
        preview = self.plan.preview()
        expected_motor = ((162.80, 54.59), (143.84, 52.56),
                          (142.83, 48.95), (156.00, 54.52))
        for index, expected, raw in zip((0, 1, 4, 5), TUNER_TARGETS, expected_motor):
            for actual, target in zip(preview[index]['target_pid_deg'], expected):
                self.assertAlmostEqual(actual, target, places=9)
            for actual, target in zip(preview[index]['target_motor_deg'], raw):
                self.assertAlmostEqual(actual, target, places=9)
        self.assertEqual(preview[3]['target_model_deg'], [180.0, 4.0])
        self.assertEqual([s['phase'] for s in preview], ['free', 'push', 'free', 'free', 'free', 'push'])
        self.assertEqual([s['speed_limit_deg_s'] for s in preview], [18., 60., 18., 18., 18., 60.])

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
                    self.assertEqual([p['phase'] for p in pair.moves],
                                     ['free', 'push', 'free', 'free', 'free', 'push'])
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

    def test_real_executor_sends_18_for_free_60_for_both_push_axes(self):
        import control.motors as motors
        for motion, speed in (('free', 18), ('push', 60)):
            with self.subTest(motion=motion), patch.object(motors, 'MotorPair', LineTestMotorPair):
                drive = VirtualDrive(model_q=(150., 40.), settings=self.plan.hardware)
                path = [list(map(math.radians, q)) for q in ((150., 40.), (151., 41.))]
                phase = self.plan.phase('speed_test', path, motion=motion)
                drive.run(phase)
                for axis in range(2):
                    packets = drive.a4(axis)
                    self.assertTrue(packets)
                    self.assertEqual({int.from_bytes(w[13:17], 'little') for *_, w in packets}, {speed*100})
                self.assertEqual(drive.a6(), [])

    def test_immobile_push_still_dispatches_every_node_before_endpoint_failure(self):
        import control.motors as motors
        with patch.object(motors, 'MotorPair', LineTestMotorPair):
            drive = VirtualDrive(model_q=(150., 40.), settings=self.plan.hardware, immobile=True)
            path = [list(map(math.radians, q)) for q in ((150., 40.), (152., 42.))]
            phase = self.plan.phase('continuous_push', path, motion='push')
            with self.assertRaisesRegex(motors.HardwareError, 'motion_goal_settle_timeout'):
                drive.run(phase)
            for axis in range(2):
                self.assertEqual(len(drive.a4(axis)), len(phase['points'])-1)


if __name__ == '__main__':
    unittest.main()
