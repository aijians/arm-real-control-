"""Endpoint qualification; no UART or real hardware."""
import math
import unittest
from unittest.mock import patch
from control.trajectory import GoalQualification
from virtual_drive import VirtualDrive, config, phase


def state(t, speed=0., q=10., speed_stamp=None):
    return {'q_rad': [math.radians(q), math.radians(20.)],
            'speed_rad_s': [math.radians(speed), 0.],
            'stamp_monotonic_s': t, 'axis_stamp_monotonic_s': [t,t],
            'speed_stamp_monotonic_s': [t if speed_stamp is None else speed_stamp,t]}


class GoalHoldTests(unittest.TestCase):
    goal = [math.radians(10.), math.radians(20.)]

    def armed(self):
        g=GoalQualification(.3,1.,1.)
        self.assertFalse(g.observe(state(0.),self.goal,True))
        self.assertFalse(g.observe(state(.3),self.goal,True))
        self.assertTrue(g.armed)
        return g

    def test_one_second_hold_after_arrival(self):
        g=self.armed()
        self.assertFalse(g.observe(state(1.29),self.goal,True))
        self.assertTrue(g.observe(state(1.3),self.goal,True))

    def test_isolated_speed_spike_keeps_timer_and_waits_for_good_sample(self):
        g=self.armed()
        self.assertFalse(g.observe(state(.8,2.),self.goal,True))
        self.assertEqual(g.since,.3)
        self.assertEqual(g.resets,0)
        self.assertEqual(g.blips_tolerated,1)
        self.assertFalse(g.observe(state(1.31,2.,speed_stamp=.8),self.goal,True))
        self.assertEqual(g.since,.3)
        self.assertEqual(g.blips_tolerated,1)
        self.assertTrue(g.observe(state(1.32),self.goal,True))

    def test_second_distinct_overspeed_resets_timer(self):
        g=self.armed()
        g.observe(state(.8,2.),self.goal,True)
        g.observe(state(.85,2.),self.goal,True)
        self.assertIsNone(g.since)
        self.assertEqual(g.resets,1)
        self.assertFalse(g.observe(state(.9),self.goal,True))
        self.assertFalse(g.observe(state(1.31),self.goal,True))
        self.assertTrue(g.observe(state(1.9),self.goal,True))

    def test_out_of_band_angle_resets_even_with_single_speed_spike(self):
        g=self.armed()
        g.observe(state(.8,2.,q=10.31),self.goal,True)
        self.assertIsNone(g.since)
        self.assertEqual(g.resets,1)

    def test_angle_shift_inside_band_does_not_hide_movement(self):
        g=self.armed()
        g.observe(state(.8,2.,q=10.06),self.goal,True)
        self.assertIsNone(g.since)
        self.assertEqual(g.blips_tolerated,0)

    def test_hard_speed_limit_still_fails(self):
        g=self.armed()
        g.observe(state(.8,10.),self.goal,True)
        self.assertEqual(g.failure,'motion_goal_hold_hard_limit')

    def test_reset_after_command_discards_old_evidence(self):
        g=self.armed()
        g.observe(state(.8,2.),self.goal,True)
        g.reset_after_command()
        self.assertFalse(g.armed)
        self.assertIsNone(g.since)
        self.assertEqual(g.blip_speed_stamps,{})

    def test_snapshot_exposes_speed_reply_timestamps(self):
        d=VirtualDrive(settings=config())
        d.run(phase([[159,20]],times=[0]))
        with patch('control.motors.time.monotonic', return_value=d.now):
            snapshot=d.pair._snapshot()
        self.assertEqual(snapshot['speed_stamp_monotonic_s'],
                         [reply['speed_stamp'] for reply in d.pair._feedback])

    def test_free_integration_isolated_speed_spike_preserves_hold(self):
        cfg=config(goal_hold_s=1.,stable_s=.3,
                   free_timed_motion={'enabled':True,'speed_deg_s':30.})
        d=VirtualDrive(settings=cfg)
        disturbed=False
        def health():
            nonlocal disturbed
            if d.elapsed>.8 and not disturbed:
                disturbed=True
                d.pair._feedback[0].update(speed_deg_s=2.,speed_stamp=d.now)
        result=d.run(phase([[159,20]],times=[0]),health_check=health)
        execution=result['motion_diagnostics']['execution']
        self.assertEqual(execution['goal_speed_blips_tolerated'],1)
        self.assertEqual(execution['goal_stability_resets'],0)
        self.assertLess(result['elapsed_s'],1.5)


if __name__=='__main__':
    unittest.main()
