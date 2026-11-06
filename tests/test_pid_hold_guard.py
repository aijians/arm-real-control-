"""Real glitch regression and persistence checks using only synthetic feedback."""
import unittest
from unittest.mock import patch

import test_pid_hold as lifecycle


class HoldGuardTests(unittest.TestCase):
    setUp = lifecycle.HoldLifecycleTests.setUp
    cleanup_tuner = lifecycle.HoldLifecycleTests.cleanup_tuner
    audit = lifecycle.HoldLifecycleTests.audit
    feedback = lifecycle.HoldLifecycleTests.feedback
    tick = lifecycle.HoldLifecycleTests.tick
    arrive = lifecycle.HoldLifecycleTests.arrive
    pass_hold = lifecycle.HoldLifecycleTests.pass_hold
    stop_count = lifecycle.HoldLifecycleTests.stop_count
    move_count = lifecycle.HoldLifecycleTests.move_count

    def test_zero_error_minus_four_glitch_does_not_stop_and_recovers(self):
        self.pass_hold()
        self.feedback(speed=-4.)
        self.tick(.1128)
        motion = self.tuner.state()['motion']
        self.assertTrue(motion['active'])
        self.assertEqual(motion['trial_outcome'], 'passed')
        self.assertEqual(motion['last_trial_error_deg'], 0.)
        self.assertEqual(motion['last_trial_speed_deg_s'], -4.)
        self.assertEqual(motion['hold_warning']['samples'], 1)
        self.assertEqual(self.stop_count(), 0)
        self.feedback(angle=self.target-.01)
        self.tick(.0206)
        motion = self.tuner.state()['motion']
        self.assertTrue(motion['active'])
        self.assertIsNone(motion['hold_warning'])
        self.assertTrue(motion['last_hold_warning']['recovered'])
        self.assertEqual(motion['hold_peak_speed_deg_s'], 4.)
        self.assertEqual(self.stop_count(), 0)

    def test_sustained_speed_stops_and_records_speed_not_position(self):
        self.pass_hold()
        self.feedback(speed=-4.)
        original_stop = self.device.stop

        def stop_and_settle():
            original_stop()
            self.feedback()

        with patch.object(self.device, 'stop', side_effect=stop_and_settle):
            self.tick(.1)
            self.tick(.25)
            self.assertEqual(self.stop_count(), 0)
            self.tick(.25)
        motion = self.tuner.state()['motion']
        self.assertEqual(self.stop_count(), 1)
        self.assertEqual(motion['trial_outcome'], 'failed')
        self.assertIn('速度-4.00', motion['stop_reason'])
        self.assertNotIn('角度误差', motion['stop_reason'])
        self.assertEqual(motion['last_trial_error_deg'], 0.)
        self.assertEqual(motion['last_trial_speed_deg_s'], -4.)
        trigger = self.audit()['motion']['stop_trigger']
        self.assertEqual(trigger['sample']['speed_deg_s'], -4.)
        self.assertEqual(trigger['hold_warning']['samples'], 3)
        self.assertAlmostEqual(trigger['hold_warning']['elapsed_s'], .5)

    def test_two_sparse_bad_frames_do_not_satisfy_minimum_samples(self):
        self.pass_hold()
        self.feedback(speed=2.)
        self.tick(.1)
        self.tick(.6)
        self.assertTrue(self.tuner.motion['active'])
        self.assertEqual(self.tuner.motion['hold_warning']['samples'], 2)
        self.feedback()
        self.tick(.1)
        self.assertEqual(self.stop_count(), 0)

    def test_three_fast_bad_frames_do_not_satisfy_duration(self):
        self.pass_hold()
        self.feedback(speed=2.)
        for _ in range(3):
            self.tick(.01)
        self.assertEqual(self.tuner.motion['hold_warning']['samples'], 3)
        self.assertEqual(self.stop_count(), 0)
        self.feedback()
        self.tick(.01)
        self.assertEqual(self.stop_count(), 0)

    def test_separate_glitches_do_not_accumulate_to_a_stop(self):
        self.pass_hold()
        for _ in range(5):
            self.feedback(speed=-4.)
            self.tick(.1)
            self.feedback()
            self.tick(.1)
        self.assertTrue(self.tuner.motion['active'])
        self.assertEqual(self.stop_count(), 0)
        self.assertEqual(self.tuner.motion['hold_observation_count'], 5)

    def test_position_excursion_recovers_without_stop(self):
        self.pass_hold()
        self.feedback(angle=self.target+.35)
        self.tick(.1)
        self.assertTrue(self.tuner.motion['active'])
        self.feedback()
        self.tick(.1)
        self.assertEqual(self.stop_count(), 0)
        self.assertEqual(self.tuner.motion['trial_outcome'], 'passed')

    def test_observation_restarts_pending_three_second_verification(self):
        self.arrive()
        self.tick(2.9)
        self.feedback(speed=-4.)
        self.tick(.1)
        self.assertEqual(self.tuner.motion['trial_outcome'], 'pending')
        self.assertEqual(self.tuner.state()['motion']['hold_stable_s'], 0.)
        self.feedback()
        self.tick(.1)
        self.tick(1.5)
        self.tick(1.49)
        self.assertEqual(self.tuner.motion['trial_outcome'], 'pending')
        self.tick(.01)
        self.assertEqual(self.tuner.motion['trial_outcome'], 'passed')
        self.assertEqual(self.stop_count(), 0)

    def test_hard_position_and_speed_limits_stop_on_first_frame(self):
        for position, speed in ((1., 0.), (0., 10.), (0., -10.)):
            with self.subTest(position=position, speed=speed):
                self.feedback(angle=169.11)
                self.pass_hold()
                before = self.stop_count()
                self.feedback(angle=self.target+position, speed=speed)
                original_stop = self.device.stop

                def stop_and_settle():
                    original_stop()
                    self.feedback()

                with patch.object(self.device, 'stop', side_effect=stop_and_settle):
                    self.tick(.01)
                self.assertEqual(self.stop_count(), before+1)
                self.assertIn('立即停止', self.tuner.motion['stop_reason'])
                self.assertEqual(self.tuner.motion['trial_outcome'], 'failed')

    def test_retarget_during_small_glitch_rejects_without_releasing_hold(self):
        self.pass_hold()
        self.feedback(speed=-4.)
        with self.assertRaisesRegex(ValueError, '波动观察'):
            self.tuner.move(self.target+2., 18.)
        self.assertEqual(self.stop_count(), 0)
        self.assertEqual(self.move_count(), 1)
        self.assertEqual(self.tuner.motion['trial_outcome'], 'passed')
        self.feedback()
        self.tick(.1)
        self.tuner.move(self.target+2., 18.)
        self.assertEqual(self.move_count(), 2)
        self.assertEqual(self.stop_count(), 0)

    def test_manual_stop_during_observation_cannot_fail_on_missing_stable_timer(self):
        self.arrive()
        self.feedback(speed=-4.)
        self.tick(.1)
        self.assertIsNone(self.tuner._stable_since)
        self.feedback()
        self.tuner.stop()
        self.assertEqual(self.stop_count(), 1)
        self.assertEqual(self.tuner.motion['trial_outcome'], 'cancelled')
        self.assertFalse(self.tuner.motion['stop_required'])


if __name__ == '__main__':
    unittest.main()
