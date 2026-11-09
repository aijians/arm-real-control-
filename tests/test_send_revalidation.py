"""No hardware: regression cases for stale recovery decisions and cache bounds."""
from contextlib import ExitStack
import math
import unittest
from unittest.mock import Mock, patch

from control.recovery_geometry import RecoveryGeometry
from test_overshoot_return import return_config
from virtual_drive import VirtualDrive, phase
from control import motors as m


class SendRevalidationTests(unittest.TestCase):
    def setUp(self):
        self.d = VirtualDrive(immobile=True, settings=return_config())
        self.p = self.d.pair
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(m.time, 'monotonic', lambda: self.d.now))
        self.stack.enter_context(patch.object(m.time, 'sleep', self.d.advance))
        self.stack.enter_context(patch.object(self.p, '_pump', self.d.pump))
        self.refresh()
        self.p._last_targets[0] = (15580, 1, 6000, 3)
        self.p._last_target_acked[0] = True
        self.p._last_target_sent_at[0] = self.d.now-1.
        self.p._settling_bias_centi[0] = -120
        self.options = dict(planned_directions=[-1, 0], command_axes=[True, False],
                            axis_at_endpoint=[True, False], adaptive_mode=True,
                            axis_min_speed_deg_s=[60., 0.], speed_tiers=[3, 0],
                            command_speed_caps_deg_s=[60., 18.])

    def refresh(self):
        for i in (0, 1):
            self.d.refresh(i, 0x94)
            self.d.refresh(i, 0x9C)

    def send(self, **options):
        return self.p._send_targets(list(map(math.radians, [157., 20.])),
                                    **{**self.options, **options})

    def still(self, angle):
        self.d.q[0] = angle
        self.refresh()
        self.p._stationary_probe[0] = (self.d.now-.12, angle)

    def test_moving_nominal_return_never_enters_geometry(self):
        self.d.q[0], self.d.velocity[0] = 153., -12.
        self.refresh()
        validator = Mock(side_effect=AssertionError('expensive check must not run'))
        self.assertEqual(self.send(nominal_return_axes=[True, False], recovery_validator=validator), 0)
        validator.assert_not_called()

    def test_acked_unchanged_bias_target_never_enters_geometry(self):
        validator = Mock(side_effect=AssertionError('dedup must precede geometry'))
        self.assertEqual(self.send(recovery_validator=validator), 0)
        validator.assert_not_called()

    def test_direction_uncertainty_rejects_before_geometry(self):
        self.p._settling_bias_centi[0] = -20
        self.d.q[0], self.d.velocity[0] = 157.01, -6.
        self.refresh()
        validator = Mock(side_effect=AssertionError('uncertain direction'))
        self.assertEqual(self.send(recovery_validator=validator), 0)
        validator.assert_not_called()

    def test_axis_crossing_during_slow_validation_cannot_send_old_direction(self):
        self.p._opened = True
        self.still(157.3)
        def delayed(axis, target):
            self.d.advance(.25)
            self.d.q[0], self.d.velocity[0] = 156.8, -8.
            self.refresh()
            return True
        self.assertEqual(self.send(recovery_validator=delayed), 0)
        self.assertEqual(self.d.a6(), [])

    def test_late_angle_reply_does_not_disguise_old_query(self):
        self.p._opened = True
        self.still(157.3)
        self.p._feedback[0]['angle_query_stamp'] = self.d.now-.2
        with patch.object(self.p, '_request_feedback'), patch.object(self.p, '_pump'):
            self.assertEqual(self.send(recovery_validator=lambda *a: True), 0)
        self.assertEqual(self.d.a6(), [])

    def test_uncertainty_reject_gets_fresh_feedback_instead_of_starving(self):
        self.p._opened = True
        self.p._settling_bias_centi[0] = 0
        self.p._last_targets[0] = (17238, 0, 3000, 2)
        self.d.q[0] = 172.38
        self.refresh()
        self.p._feedback[0]['angle_query_stamp'] = self.d.now-.04
        self.assertEqual(self.p._send_targets(list(map(math.radians, [171.88, 20.])),
            **self.options), 1)
        self.assertEqual(self.d.a6(0)[-1][3][5], 1)

    def test_angle_query_stamp_excludes_wait_before_query_write(self):
        self.p.ports[0]._next_send_at = self.d.now+.004
        initial = self.d.now
        self.p._request_feedback(axes=[0], include_speed=False)
        self.assertGreaterEqual(self.p._pending_angle_query_at[0], initial+.004-1e-8)

    def test_low_speed_option_changes_only_retraction_frame(self):
        self.still(157.3)
        self.p.motion_policy['bias_retraction_speed_deg_s'] = 18.
        self.assertEqual(self.send(recovery_validator=lambda *a: True), 1)
        wire = self.d.a6(0)[-1][3]
        self.assertEqual((wire[5], int.from_bytes(wire[6:9], 'little'),
                          int.from_bytes(wire[9:13], 'little')), (1, 15700, 1800))
        self.assertEqual(self.p._last_targets[0][3], 3)

    def test_default_retraction_speed_preserves_control_condition(self):
        self.still(157.3)
        self.assertEqual(self.send(recovery_validator=lambda *a: True), 1)
        self.assertEqual(int.from_bytes(self.d.a6(0)[-1][3][9:13], 'little'), 6000)

    def test_crossing_option_waits_for_both_crossing_and_stillness(self):
        self.p.motion_policy['bias_retraction_requires_crossing'] = True
        self.still(157.3)
        validator = Mock(side_effect=AssertionError('not crossed'))
        self.assertEqual(self.send(recovery_validator=validator), 0)
        validator.assert_not_called()
        self.still(156.7)
        self.assertEqual(self.send(recovery_validator=lambda *a: True), 1)
        self.assertEqual(self.d.a6(0)[-1][3][5], 0)


class GeometryEnvelopeTests(unittest.TestCase):
    def setUp(self):
        self.d = VirtualDrive()
        for i in (0, 1):
            self.d.refresh(i, 0x94)
        self.calls = []
        self.check = lambda q: not self.calls.append(q)
        self.g = RecoveryGeometry(self.d.pair.motors, self.d.pair._feedback,
                                  self.d.pair._last_targets, self.check,
                                  nominal_targets=lambda: [15700, 2000])

    def finish(self, target=15580):
        for _ in range(1000):
            before = len(self.calls)
            result = self.g(0, target)
            self.assertLessEqual(len(self.calls)-before, 1)
            if result is not None:
                return result
        self.fail('geometry did not complete')

    def test_cold_work_yields_and_warms_nominal_return(self):
        self.assertIsNone(self.g(0, 15580))
        self.assertTrue(self.finish())
        count = len(self.calls)
        self.assertTrue(self.g.contains(0, 15700))
        self.assertTrue(self.g(0, 15580))
        self.assertEqual(count, len(self.calls))

    def test_new_start_outside_checked_interval_is_not_a_cache_hit(self):
        self.assertTrue(self.finish())
        self.d.pair._feedback[0]['motor_deg'] = 161.
        self.assertFalse(self.g.contains(0, 15580))

    def test_grid_clipping_never_certifies_a_start_outside_joint_limits(self):
        self.d.pair._feedback[1]['motor_deg'] = -.01
        self.assertFalse(self.g(0, 15580))
        self.assertEqual(self.calls, [])

    def test_other_axis_movement_or_command_invalidates_certificate(self):
        self.assertTrue(self.finish())
        self.d.pair._last_targets[1] = (2050, 0, 1800, 0)
        self.assertFalse(self.g.contains(0, 15580))
        self.d.pair._last_targets[1] = None
        self.d.pair._feedback[1]['motor_deg'] = 20.5
        self.assertFalse(self.g.contains(0, 15580))

    def test_obstruction_inside_other_axis_band_is_checked(self):
        self.d.pair._last_targets[1] = (2040, 0, 1800, 0)
        def check(q):
            self.calls.append(q)
            return not 20.15 < math.degrees(q[1]) < 20.25
        self.g.check = check
        self.assertFalse(self.finish())

    def test_different_move_context_never_reuses_previous_certificate(self):
        self.assertTrue(self.finish())
        other = RecoveryGeometry(self.d.pair.motors, self.d.pair._feedback,
                                  self.d.pair._last_targets, lambda q: False)
        self.assertFalse(other(0, 15580))


if __name__ == '__main__':
    unittest.main()
