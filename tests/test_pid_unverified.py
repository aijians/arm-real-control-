"""Endpoint opt-out behavior with synthetic telemetry and a controlled clock."""
import unittest
from unittest.mock import patch
import test_pid_hold as lifecycle

class UnverifiedTests(unittest.TestCase):
    setUp = lifecycle.HoldLifecycleTests.setUp
    cleanup_tuner = lifecycle.HoldLifecycleTests.cleanup_tuner
    feedback = lifecycle.HoldLifecycleTests.feedback
    tick = lifecycle.HoldLifecycleTests.tick
    stop_count = lifecycle.HoldLifecycleTests.stop_count
    move_count = lifecycle.HoldLifecycleTests.move_count
    audit = lifecycle.HoldLifecycleTests.audit

    def test_far_target_never_times_out_or_passes_but_records_telemetry(self):
        self.tuner.move(140., 18., verify_endpoint=False)
        self.feedback(angle=169., speed=4.)
        self.tick(60.)
        self.assertTrue(self.tuner.motion['active'])
        self.assertEqual(self.tuner.motion['phase'], 'unverified')
        self.assertEqual(self.tuner.motion['trial_outcome'], 'skipped')
        self.assertEqual(self.stop_count(), 0)
        self.assertEqual(self.tuner.motion['last_trial_error_deg'], 29.)
        self.feedback(angle=140.)
        self.tick(60.)
        self.assertEqual(self.tuner.motion['trial_outcome'], 'skipped')
        self.assertFalse(self.audit()['motion']['verify_endpoint'])
        self.assertTrue(any(e.get('verify_endpoint') is False for e in self.audit()['events']))

    def test_retarget_moving_unverified_target_without_stop_and_enable_checks_again(self):
        self.tuner.move(140., 18., verify_endpoint=False)
        self.feedback(angle=160., speed=5.)
        self.tuner.move(130., 18., verify_endpoint=False)
        self.assertEqual(self.move_count(), 2)
        self.assertEqual(self.stop_count(), 0)
        self.assertEqual(self.tuner.motion['previous_trial']['trial_outcome'], 'skipped')
        self.tuner.move(150., 18., verify_endpoint=True)
        self.assertTrue(self.tuner.motion['verify_endpoint'])
        self.feedback(angle=160., speed=0.)
        self.tick(60.)
        self.assertEqual(self.tuner.motion['trial_outcome'], 'failed')
        self.assertIn('到位时间', self.tuner.motion['stop_reason'])

    def test_unverified_motion_still_stops_for_heartbeat_loss(self):
        self.tuner.move(140., 18., verify_endpoint=False)
        self.tick(3.01, heartbeat=False)
        self.assertFalse(self.tuner.motion['active'])
        self.assertEqual(self.stop_count(), 1)
        self.assertIn('心跳', self.tuner.motion['stop_reason'])

    def test_manual_stop_and_parameter_interlock_still_apply(self):
        self.tuner.move(140., 18., verify_endpoint=False)
        with self.assertRaisesRegex(ValueError, '先停止'):
            self.tuner.write('angle', {'kp':50,'ki':0,'kd':0})
        self.tuner.stop()
        self.assertFalse(self.tuner.motion['active'])
        self.assertEqual(self.stop_count(), 1)
        self.assertEqual(self.tuner.motion['trial_outcome'], 'skipped')

    def test_non_boolean_option_and_unconfirmed_command_do_not_claim_skipped(self):
        for flag in (0, None, 'false'):
            with self.assertRaises(ValueError):
                self.tuner.move(140., 18., verify_endpoint=flag)
        self.assertEqual(self.move_count(), 0)
        self.device.move_error=TimeoutError('lost ACK')
        with self.assertRaises(TimeoutError):
            self.tuner.move(140., 18., verify_endpoint=False)
        self.assertEqual(self.tuner.motion['trial_outcome'], 'failed')
        self.assertEqual(self.stop_count(), 1)

if __name__ == '__main__':
    unittest.main()
