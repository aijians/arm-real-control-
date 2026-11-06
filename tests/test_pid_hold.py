"""Controlled feedback proves trial success before releasing position hold.

All operations use the existing synthetic device; no serial port is opened.
"""
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pid_tuner.service import Tuner
from test_pid_service import Clock, RecordingDevice


class HoldLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.clock = Clock()
        self.addCleanup(patch.stopall)
        patch('pid_tuner.service.time.monotonic', side_effect=self.clock.monotonic).start()
        patch('pid_tuner.service.time.sleep', side_effect=self.clock.advance).start()
        config = {'hardware': {
            'baudrate': 115200, 'max_speed_deg_s': 18,
            'motors': [{'name': 'upper', 'device': '/synthetic/upper-only', 'id': 1,
                        'sign': 1, 'ratio': 1., 'offset_deg': 0.,
                        'limits_deg': [0, 180]}]}}

        def factory(connection, logger):
            self.device = RecordingDevice(connection, logger)
            return self.device

        self.tuner = Tuner(config, {'runtime_dir': Path(self.directory.name)},
                           device_factory=factory, start_worker=False)
        self.addCleanup(self.cleanup_tuner)
        self.tuner.connect()
        self.target = 170.

    def cleanup_tuner(self):
        if self.tuner._device:
            self.device.stop_error = self.device.telemetry_error = None
            self.device.overrides = {}
        self.tuner.close()

    def audit(self):
        path = self.tuner.audit_dir / (self.tuner.session_id + '.json')
        return json.loads(path.read_text())

    def stop_count(self):
        return sum(call[0] == 'stop' for call in self.device.calls)

    def move_count(self):
        return sum(call[0] == 'move' for call in self.device.calls)

    def feedback(self, angle=None, speed=0., **extra):
        angle = self.target if angle is None else angle
        self.device.overrides = {'angle_deg': angle, 'motor_angle_deg': angle,
                                 'speed_deg_s': speed, **extra}

    def tick(self, seconds=0., heartbeat=True):
        self.clock.advance(seconds)
        if heartbeat:
            self.tuner.state()
        self.tuner._monitor_once()

    def arrive(self):
        self.tuner.move(self.target, 18)
        self.feedback()
        self.tick()
        self.tick(0.31)
        self.assertEqual(self.tuner.motion['phase'], 'holding')

    def pass_hold(self):
        self.arrive()
        for _ in range(3):
            self.tick(1.)
        self.assertEqual(self.tuner.motion['trial_outcome'], 'passed')

    def test_arrival_does_not_send_stop_or_return_move(self):
        self.tuner.move(self.target, 18)
        self.feedback()
        self.tick()
        self.tick(0.29)
        self.assertEqual(self.tuner.motion['phase'], 'running')
        self.assertEqual(self.tuner.motion['trial_outcome'], 'pending')
        self.tick(0.02)
        state = self.tuner.state()['motion']
        self.assertEqual(state['phase'], 'holding')
        self.assertTrue(state['active'])
        self.assertEqual(state['trial_outcome'], 'pending')
        self.assertEqual(self.stop_count(), 0)
        self.assertEqual(self.move_count(), 1)
        self.assertAlmostEqual(state['start_deg'], 169.11)
        self.assertAlmostEqual(state['hold_elapsed_s'], 0.)
        self.assertAlmostEqual(state['hold_remaining_s'], 30.)
        self.assertEqual(state['verify_hold_s'], 3.)
        self.assertIsNone(state['max_hold_s'])
        self.assertEqual(state['verify_timeout_s'], 30.)
        self.assertEqual(state['tolerance_deg'], 0.2)

    def test_less_than_three_seconds_holding_cannot_mark_success(self):
        self.arrive()
        self.tick(1.5)
        self.tick(1.49)
        state = self.tuner.state()['motion']
        self.assertEqual(state['trial_outcome'], 'pending')
        self.assertNotEqual(state['result'], 'completed')
        self.assertTrue(state['active'])
        self.assertAlmostEqual(state['hold_stable_s'], 2.99)
        self.assertEqual(self.stop_count(), 0)

    def test_unstable_arrival_must_restart_the_point_three_second_window(self):
        self.tuner.move(self.target, 18)
        self.feedback()
        self.tick()
        self.tick(.2)
        self.feedback(angle=self.target + .3)
        self.tick(.01)
        self.feedback()
        self.tick(.01)
        self.tick(.29)
        self.assertEqual(self.tuner.motion['phase'], 'running')
        self.assertEqual(self.tuner.motion['trial_outcome'], 'pending')
        self.tick(.02)
        self.assertEqual(self.tuner.motion['phase'], 'holding')
        self.assertEqual(self.stop_count(), 0)

    def test_feedback_at_hold_tolerance_and_speed_limit_is_not_failure(self):
        self.arrive()
        self.feedback(angle=self.target + .2, speed=1.)
        self.tick(1.)
        self.tick(1.)
        self.tick(1.)
        self.assertEqual(self.tuner.motion['phase'], 'holding')
        self.assertEqual(self.tuner.motion['trial_outcome'], 'passed')
        self.assertEqual(self.stop_count(), 0)

    def test_three_seconds_marks_success_and_continues_position_hold(self):
        self.pass_hold()
        state = self.tuner.state()['motion']
        self.assertEqual(state['result'], 'completed')
        self.assertEqual(state['phase'], 'holding')
        self.assertTrue(state['active'])
        self.assertEqual(self.stop_count(), 0)
        self.assertEqual(self.move_count(), 1)
        self.tick(1.)
        self.assertTrue(self.tuner.motion['active'])
        self.assertEqual(self.tuner.motion['trial_outcome'], 'passed')
        self.assertEqual(self.stop_count(), 0)
        with self.assertRaises(ValueError):
            self.tuner.write('speed', {'kp': 500, 'ki': 10, 'kd': 0})

    def test_successful_hold_continues_beyond_thirty_seconds_without_stop(self):
        self.pass_hold()
        for _ in range(28):
            self.tick(1.)
        motion = self.tuner.state()['motion']
        self.assertTrue(motion['active'])
        self.assertFalse(motion['stop_required'])
        self.assertEqual(motion['phase'], 'holding')
        self.assertEqual(motion['result'], 'completed')
        self.assertEqual(motion['trial_outcome'], 'passed')
        self.assertEqual(motion['target_deg'], self.target)
        self.assertGreater(motion['hold_elapsed_s'], 30.)
        self.assertIsNone(motion['hold_remaining_s'])
        self.assertIsNone(motion['max_hold_s'])
        self.assertEqual(self.stop_count(), 0)
        self.assertEqual(self.move_count(), 1)
        self.assertEqual(self.audit()['motion']['trial_outcome'], 'passed')

    def test_departure_during_hold_fails_even_after_success(self):
        for already_passed in (False, True):
            with self.subTest(already_passed=already_passed):
                if already_passed:
                    self.pass_hold()
                else:
                    self.arrive()
                before_stops = self.stop_count()
                self.feedback(angle=self.target + .35)
                self.tick(.1)
                self.assertTrue(self.tuner.motion['active'])
                self.assertEqual(self.stop_count(), before_stops)
                self.tick(.25)
                self.tick(.26)
                motion = self.tuner.state()['motion']
                self.assertFalse(motion['active'])
                self.assertEqual(motion['phase'], 'stopped')
                self.assertEqual(motion['result'], 'aborted')
                self.assertEqual(motion['trial_outcome'], 'failed')
                self.assertTrue(motion['stop_reason'])
                self.assertIn('角度误差', motion['stop_reason'])
                self.assertNotIn('速度', motion['stop_reason'])
                self.assertAlmostEqual(motion['last_trial_error_deg'], .35)
                self.assertGreaterEqual(motion['hold_peak_error_deg'], .35 - 1e-9)
                self.assertEqual(self.stop_count(), before_stops + 1)
                self.assertEqual(self.audit()['motion']['trial_outcome'], 'failed')
                self.feedback(angle=169.11)

    def test_overspeed_during_hold_fails_and_attempts_stop(self):
        self.arrive()
        self.feedback(speed=1.01)
        original_stop = self.device.stop

        def stop_with_stationary_feedback():
            original_stop()
            self.feedback()

        with patch.object(self.device, 'stop', side_effect=stop_with_stationary_feedback):
            self.tick(.1)
            self.assertTrue(self.tuner.motion['active'])
            self.tick(.25)
            self.tick(.26)
        self.assertEqual(self.tuner.motion['trial_outcome'], 'failed')
        self.assertEqual(self.tuner.motion['result'], 'aborted')
        self.assertFalse(self.tuner.motion['active'])
        self.assertEqual(self.stop_count(), 1)

    def test_manual_stop_preserves_passed_and_cancels_unverified_trial(self):
        self.arrive()
        self.tick(1.)
        self.tuner.stop()
        self.assertEqual(self.tuner.motion['trial_outcome'], 'cancelled')
        self.assertEqual(self.tuner.motion['phase'], 'stopped')
        self.feedback(angle=169.11)
        self.pass_hold()
        self.tuner.stop()
        self.assertEqual(self.tuner.motion['trial_outcome'], 'passed')
        self.assertEqual(self.tuner.motion['phase'], 'stopped')
        self.assertFalse(self.tuner.motion['active'])
        self.assertEqual(self.audit()['motion']['trial_outcome'], 'passed')

    def test_lost_browser_heartbeat_during_hold_stops_and_invalidates_success(self):
        self.pass_hold()
        self.tick(3.01, heartbeat=False)
        self.assertEqual(self.tuner.motion['trial_outcome'], 'failed')
        self.assertEqual(self.tuner.motion['result'], 'aborted')
        self.assertFalse(self.tuner.motion['active'])
        self.assertIn('心跳', self.tuner.motion['stop_reason'])
        self.assertEqual(self.stop_count(), 1)

    def test_driver_fault_during_hold_stops_and_invalidates_success(self):
        self.pass_hold()
        self.feedback(error_state=1, error_flags=['overcurrent'])
        self.tick(.1)
        self.assertEqual(self.tuner.motion['trial_outcome'], 'failed')
        self.assertEqual(self.tuner.motion['result'], 'aborted')
        self.assertFalse(self.tuner.motion['active'])
        self.assertIn('overcurrent', self.tuner.motion['stop_reason'])
        self.assertEqual(self.stop_count(), 1)

    def test_feedback_loss_attempts_stop_and_recovery_keeps_failed_trial(self):
        self.pass_hold()
        self.device.telemetry_error = OSError('feedback disconnected')
        iterations = []

        def next_iteration(_timeout):
            iterations.append(1)
            if len(iterations) == 2:
                self.device.telemetry_error = None
            return len(iterations) > 2

        with patch.object(self.tuner._shutdown, 'wait', side_effect=next_iteration):
            self.tuner._poll_loop()
        self.assertEqual(self.tuner.motion['trial_outcome'], 'failed')
        self.assertFalse(self.tuner.motion['active'])
        self.assertFalse(self.tuner.motion['stop_required'])
        self.assertEqual(self.stop_count(), 2)
        self.assertIn('feedback disconnected', self.tuner.motion['stop_reason'])
        self.assertEqual(self.audit()['motion']['trial_outcome'], 'failed')

    def test_failed_stop_ack_does_not_erase_passed_outcome_on_retry(self):
        self.pass_hold()
        self.device.stop_error = TimeoutError('stop ACK lost')
        with self.assertRaises(TimeoutError):
            self.tuner.stop()
        self.assertTrue(self.tuner.motion['stop_required'])
        self.assertEqual(self.tuner.motion['trial_outcome'], 'passed')
        self.device.stop_error = None
        self.tuner._monitor_once()
        self.assertFalse(self.tuner.motion['stop_required'])
        self.assertEqual(self.tuner.motion['phase'], 'stopped')
        self.assertEqual(self.tuner.motion['trial_outcome'], 'passed')

    def poll_iterations(self, count=2):
        """Run the real worker exception path without a thread or wall time."""
        iterations = []

        def next_iteration(_timeout):
            iterations.append(1)
            return len(iterations) > count

        with patch.object(self.tuner._shutdown, 'wait', side_effect=next_iteration):
            self.tuner._poll_loop()

    def test_manual_stop_ack_failures_keep_passed_conclusion_during_worker_retry(self):
        self.pass_hold()
        self.device.stop_error = TimeoutError('stop ACK still missing')
        with self.assertRaises(TimeoutError):
            self.tuner.stop()
        self.assertEqual(self.tuner.motion['stop_reason'], '手动停止')
        self.poll_iterations()
        motion = self.tuner.state()['motion']
        self.assertTrue(motion['stop_required'])
        self.assertFalse(motion['active'])
        self.assertEqual(motion['trial_outcome'], 'passed')
        self.assertEqual(motion['stop_reason'], '手动停止')
        self.assertIn('stop ACK still missing', motion['stop_error'])
        self.assertEqual(self.audit()['motion']['trial_outcome'], 'passed')
        self.assertEqual(self.audit()['motion']['stop_reason'], '手动停止')
        self.assertGreaterEqual(self.stop_count(), 3)
        self.device.stop_error = None
        self.tuner._monitor_once()
        recovered = self.tuner.state()['motion']
        self.assertFalse(recovered['stop_required'])
        self.assertEqual(recovered['trial_outcome'], 'passed')
        self.assertEqual(recovered['stop_reason'], '手动停止')
        self.assertIsNone(recovered.get('stop_error'))

    def test_worker_keeps_successful_hold_beyond_three_hundred_seconds(self):
        self.pass_hold()
        # Any accidental 0x81 would fail, making an unwanted release visible.
        self.device.stop_error = TimeoutError('no automatic stop is expected')
        self.clock.advance(300.)
        self.tuner.state()
        self.poll_iterations()
        motion = self.tuner.state()['motion']
        self.assertFalse(motion['stop_required'])
        self.assertTrue(motion['active'])
        self.assertEqual(motion['phase'], 'holding')
        self.assertEqual(motion['trial_outcome'], 'passed')
        self.assertGreater(motion['hold_elapsed_s'], 300.)
        self.assertIsNone(motion['hold_remaining_s'])
        self.assertIsNone(motion['max_hold_s'])
        self.assertIsNone(motion.get('stop_error'))
        self.assertEqual(self.stop_count(), 0)
        self.assertEqual(self.audit()['motion']['trial_outcome'], 'passed')

    def test_unverified_hold_still_times_out_after_thirty_seconds(self):
        self.arrive()
        # A worker gap cannot count as a completed, observed 3-second hold.
        self.clock.advance(30.01)
        self.tuner.state()
        self.poll_iterations(count=1)
        motion = self.tuner.state()['motion']
        self.assertEqual(motion['trial_outcome'], 'failed')
        self.assertTrue(motion['stop_reason'])
        self.assertFalse(motion['stop_required'])
        self.assertFalse(motion['active'])
        self.assertEqual(motion['phase'], 'stopped')
        self.assertEqual(self.stop_count(), 1)
        self.assertEqual(self.audit()['motion']['trial_outcome'], 'failed')

    def test_passed_hold_accepts_next_target_in_either_direction_without_stop(self):
        self.pass_hold()
        for delta in (2., -2.):
            with self.subTest(delta=delta):
                previous = self.tuner.state()['motion']
                latest_angle = self.target + .1
                self.feedback(angle=latest_angle)
                new_target = latest_angle + delta
                self.tuner.move(new_target, 18.)
                motion = self.tuner.state()['motion']
                self.assertTrue(motion['active'])
                self.assertEqual(motion['phase'], 'running')
                self.assertEqual(motion['trial_outcome'], 'pending')
                self.assertEqual(motion['result'], 'running')
                self.assertAlmostEqual(motion['start_deg'], latest_angle)
                self.assertAlmostEqual(motion['target_deg'], new_target)
                self.assertNotEqual(motion['trial_id'], previous['trial_id'])
                self.assertEqual(motion['previous_trial']['trial_id'], previous['trial_id'])
                self.assertEqual(motion['previous_trial']['trial_outcome'], 'passed')
                self.assertEqual(motion['previous_trial']['target_deg'], self.target)
                self.assertEqual(motion['previous_trial']['phase'], 'superseded')
                self.assertFalse(motion['previous_trial']['active'])
                self.assertNotIn('previous_trial', motion['previous_trial'])
                self.assertAlmostEqual(motion['previous_trial']['last_trial_error_deg'], .1)
                self.assertEqual(motion['hold_elapsed_s'], 0.)
                self.assertEqual(self.stop_count(), 0)
                self.assertEqual(self.device.calls[-1][0], 'move')
                self.target = new_target
                self.feedback()
                self.tick()
                self.tick(.31)
                for _ in range(3):
                    self.tick(1.)
                self.assertEqual(self.tuner.motion['trial_outcome'], 'passed')
        self.assertEqual(self.move_count(), 3)
        self.assertEqual(self.stop_count(), 0)

    def test_invalid_retarget_input_is_rejected_without_releasing_current_hold(self):
        self.pass_hold()
        self.feedback(angle=self.target + .1)
        original = copy.deepcopy(self.tuner.motion)
        cases = [
            (159.9, 18., {}),  # More than 10 degrees from latest 170.1.
            (180.01, 18., {}), (-.01, 18., {}),
            (172., 60.01, {}), (172., 0., {}),
            (float('nan'), 18., {}), (172., float('inf'), {}),
        ]
        for target, speed, extra in cases:
            with self.subTest(target=target, speed=speed, extra=extra):
                self.feedback(angle=self.target + .1, **extra)
                with self.assertRaises(ValueError):
                    self.tuner.move(target, speed)
                control_fields = ('active', 'phase', 'result', 'trial_outcome', 'trial_id',
                                  'target_deg', 'start_deg', 'speed_deg_s')
                self.assertEqual({k: self.tuner.motion[k] for k in control_fields},
                                 {k: original[k] for k in control_fields})
                self.assertEqual(self.stop_count(), 0)
                self.assertEqual(self.move_count(), 1)
        self.feedback(angle=self.target + .1)
        self.tuner.move(180., 60.)
        self.assertEqual(self.move_count(), 2)
        self.assertEqual(self.stop_count(), 0)
        self.assertEqual(self.tuner.motion['speed_deg_s'], 60.)

    def test_transient_unsafe_feedback_during_retarget_fails_and_stops_old_trial(self):
        cases = [
            {'speed_deg_s': 10.},
            {'angle_deg': self.target + 1.},
            {'angle_deg': 180.1},
            {'error_state': 1, 'error_flags': ['overcurrent']},
            {'motor_state': 0x10},
        ]
        for extra in cases:
            with self.subTest(feedback=extra):
                self.pass_hold()
                original_id = self.tuner.motion['trial_id']
                before_stops, before_moves = self.stop_count(), self.move_count()
                self.feedback(**extra)
                original_stop = self.device.stop

                def stop_then_normal_feedback():
                    original_stop()
                    # Only the retarget's first sample is unsafe; a later
                    # normal stop-confirmation sample must not erase the trip.
                    self.feedback()

                with patch.object(self.device, 'stop', side_effect=stop_then_normal_feedback):
                    with self.assertRaises(ValueError):
                        self.tuner.move(self.target + 2., 18.)
                motion = self.tuner.state()['motion']
                self.assertEqual(motion['trial_id'], original_id)
                self.assertEqual(motion['trial_outcome'], 'failed')
                self.assertFalse(motion['active'])
                self.assertFalse(motion['stop_required'])
                self.assertTrue(motion['stop_reason'])
                self.assertEqual(self.stop_count(), before_stops + 1)
                self.assertEqual(self.move_count(), before_moves)
                self.assertEqual(self.audit()['motion']['trial_outcome'], 'failed')
                if extra.get('angle_deg') == self.target + 1.:
                    self.assertAlmostEqual(motion['last_trial_error_deg'], 1.)
                    self.assertGreaterEqual(motion['hold_peak_error_deg'], 1. - 1e-9)
                self.feedback(angle=169.11)

    def test_transient_retarget_sample_exception_stops_before_second_move(self):
        for error in (OSError('transient feedback lost'), TimeoutError('one feedback timeout')):
            with self.subTest(error=error):
                self.pass_hold()
                before_stops, before_moves = self.stop_count(), self.move_count()
                original_id = self.tuner.motion['trial_id']
                original_telemetry = self.device.telemetry
                seen = []

                def one_bad_read():
                    if not seen:
                        seen.append(True)
                        raise error
                    return original_telemetry()

                with patch.object(self.device, 'telemetry', side_effect=one_bad_read):
                    with self.assertRaises(type(error)):
                        self.tuner.move(self.target + 2., 18.)
                motion = self.tuner.state()['motion']
                self.assertEqual(motion['trial_id'], original_id)
                self.assertEqual(motion['trial_outcome'], 'failed')
                self.assertFalse(motion['active'])
                self.assertFalse(motion['stop_required'])
                self.assertIn(str(error), motion['stop_reason'])
                self.assertEqual(self.stop_count(), before_stops + 1)
                self.assertEqual(self.move_count(), before_moves)
                self.assertEqual(self.audit()['motion']['trial_outcome'], 'failed')
                self.feedback(angle=169.11)

    def test_retarget_request_after_heartbeat_loss_stops_old_trial(self):
        self.pass_hold()
        original_id = self.tuner.motion['trial_id']
        self.clock.advance(3.01)
        with self.assertRaises(ValueError):
            self.tuner.move(self.target + 2., 18.)
        motion = self.tuner.state()['motion']
        self.assertEqual(motion['trial_id'], original_id)
        self.assertEqual(motion['trial_outcome'], 'failed')
        self.assertIn('心跳', motion['stop_reason'])
        self.assertFalse(motion['active'])
        self.assertEqual(self.stop_count(), 1)
        self.assertEqual(self.move_count(), 1)

    def test_disk_full_before_retarget_stops_even_when_failure_audit_also_cannot_save(self):
        self.pass_hold()
        original_id = self.tuner.motion['trial_id']
        with patch('pid_tuner.service.save_json', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                self.tuner.move(self.target + 2., 18.)
        motion = self.tuner.state()['motion']
        self.assertEqual(motion['trial_id'], original_id)
        self.assertEqual(motion['trial_outcome'], 'failed')
        self.assertFalse(motion['active'])
        self.assertFalse(motion['stop_required'])
        self.assertIn('disk full', motion['stop_reason'])
        self.assertEqual(self.stop_count(), 1)
        self.assertEqual(self.move_count(), 1)

    def test_running_and_unverified_hold_do_not_accept_retarget(self):
        self.tuner.move(self.target, 18.)
        original_id = self.tuner.motion['trial_id']
        with self.assertRaises(ValueError):
            self.tuner.move(self.target + 2., 18.)
        self.feedback()
        self.tick()
        self.tick(.31)
        self.assertEqual(self.tuner.motion['phase'], 'holding')
        self.assertEqual(self.tuner.motion['trial_outcome'], 'pending')
        with self.assertRaises(ValueError):
            self.tuner.move(self.target + 2., 18.)
        self.assertEqual(self.tuner.motion['trial_id'], original_id)
        self.assertEqual(self.move_count(), 1)
        self.assertEqual(self.stop_count(), 0)

    def test_retarget_lost_ack_stops_and_marks_new_trial_failed(self):
        self.pass_hold()
        previous_id = self.tuner.motion['trial_id']
        self.device.move_error = TimeoutError('second A6 ACK lost after acceptance')
        with self.assertRaises(TimeoutError):
            self.tuner.move(self.target + 2., 18.)
        motion = self.tuner.state()['motion']
        self.assertNotEqual(motion['trial_id'], previous_id)
        self.assertEqual(motion['previous_trial']['trial_id'], previous_id)
        self.assertEqual(motion['previous_trial']['trial_outcome'], 'passed')
        self.assertEqual(motion['trial_outcome'], 'failed')
        self.assertEqual(motion['result'], 'command_failed')
        self.assertFalse(motion['active'])
        self.assertFalse(motion['stop_required'])
        self.assertEqual(self.move_count(), 2)
        self.assertEqual(self.stop_count(), 1)
        self.assertEqual(self.audit()['motion']['trial_outcome'], 'failed')

    def test_observation_times_freeze_at_stop_without_counting_stop_confirmation(self):
        self.arrive()
        self.tick(1.25)
        before = self.tuner.state()['motion']
        self.assertAlmostEqual(before['hold_elapsed_s'], 1.25)
        self.assertAlmostEqual(before['hold_stable_s'], 1.25)
        self.tuner.stop()
        stopped = self.tuner.state()['motion']
        self.assertEqual(stopped['trial_outcome'], 'cancelled')
        self.assertAlmostEqual(stopped['hold_elapsed_s'], 1.25)
        self.assertAlmostEqual(stopped['hold_stable_s'], 1.25)
        self.clock.advance(10.)
        later = self.tuner.state()['motion']
        self.assertAlmostEqual(later['hold_elapsed_s'], stopped['hold_elapsed_s'])
        self.assertAlmostEqual(later['hold_stable_s'], stopped['hold_stable_s'])
        self.assertEqual(later['hold_remaining_s'], 0.)
        self.assertAlmostEqual(self.audit()['motion']['hold_elapsed_s'], 1.25)
        self.assertAlmostEqual(self.audit()['motion']['hold_stable_s'], 1.25)

    def test_trial_phase_and_outcome_are_in_readings_csv_and_durable_audit(self):
        self.pass_hold()
        self.feedback(angle=self.target - .1)
        self.tick(.1)
        state = self.tuner.state()
        row = state['latest']
        self.assertEqual(row['motion_phase'], 'holding')
        self.assertEqual(row['trial_outcome'], 'passed')
        self.assertTrue(row['trial_id'])
        self.assertAlmostEqual(row['target_error_deg'], -.1)
        self.assertEqual(row['trial_id'], state['motion']['trial_id'])
        self.tuner.stop()
        audit = self.audit()
        self.assertEqual(audit['motion']['trial_outcome'], 'passed')
        self.assertEqual(audit['motion']['phase'], 'stopped')
        self.assertTrue(any(event.get('action') for event in audit['events']))
        self.assertTrue(any(r.get('motion_phase') == 'holding' and
                            r.get('trial_outcome') == 'passed' for r in audit['readings']))
        rows = list(csv.DictReader(io.StringIO(self.tuner.csv().lstrip('\ufeff'))))
        held = [r for r in rows if r['motion_phase'] == 'holding' and
                r['trial_outcome'] == 'passed']
        self.assertTrue(held)
        self.assertEqual(held[-1]['trial_id'], row['trial_id'])
        self.assertAlmostEqual(float(held[-1]['target_error_deg']), -.1)


if __name__ == '__main__':
    unittest.main()
