"""Endpoint stability: isolated integer speed readings and reset diagnostics."""
import math
import unittest

from control.trajectory import GoalQualification


GOAL = [math.radians(163.4), math.radians(53.1)]


def sample(t, *, speed=(0., 0.), error=(0.01, 0.01), speed_stamp=None):
    q = [g + math.radians(e) for g, e in zip(GOAL, error)]
    stamp = [t, t]
    return {"q_rad": q, "speed_rad_s": [math.radians(v) for v in speed],
            "axis_stamp_monotonic_s": stamp, "stamp_monotonic_s": t,
            "speed_stamp_monotonic_s": list(speed_stamp or stamp)}


class GoalQualificationBlipTests(unittest.TestCase):
    def qualifier(self, **kw):
        return GoalQualification(0.3, 1.0, 1.0, 0.3, **kw)

    def run_samples(self, g, samples):
        done_at = None
        for s in samples:
            if g.observe(s, GOAL, True) and done_at is None:
                done_at = s["stamp_monotonic_s"]
        return done_at

    def test_isolated_speed_reading_keeps_accumulated_stable_time(self):
        g = self.qualifier()
        samples = [sample(i * .05) for i in range(20)]           # 0.0 .. 0.95 s
        samples.append(sample(1.0, speed=(2., 0.)))               # 0 -> 2 -> 0
        samples += [sample(1.05 + i * .05) for i in range(20)]
        done = self.run_samples(g, samples)
        # Armed at 0.30 s; 1.0 s hold completes at 1.30 s without restarting.
        self.assertAlmostEqual(done, 1.3, places=6)
        self.assertEqual(g.resets, 0)
        self.assertEqual(g.blips_tolerated, 1)
        event = g.events[-1]
        self.assertEqual((event["kind"], event["reason"]), ("tolerated_speed_blip", "isolated_speed_sample"))
        self.assertEqual(event["speeds_deg_s"][0], 2.0)
        self.assertEqual(len(event["errors_deg"]), 2)
        self.assertEqual(event["stamp_monotonic_s"], 1.0)

    def test_two_consecutive_overspeed_samples_reset_and_are_logged(self):
        g = self.qualifier()
        samples = [sample(i * .05) for i in range(12)]
        samples += [sample(.6, speed=(2., 0.)), sample(.65, speed=(2., 0.))]
        self.run_samples(g, samples)
        self.assertEqual(g.resets, 1)
        self.assertEqual(g.stable_s, 0.)
        reset = [e for e in g.events if e["kind"] == "reset"][-1]
        self.assertEqual(reset["reason"], "consecutive_overspeed")
        self.assertEqual(reset["stamp_monotonic_s"], .65)
        self.assertGreater(reset["stable_s_before"], 0.)
        self.assertEqual(reset["speeds_deg_s"], [2.0, 0.0])

    def test_rereading_the_same_speed_sample_is_not_a_second_overspeed(self):
        g = self.qualifier()
        self.run_samples(g, [sample(i * .05) for i in range(12)])
        blip = sample(.6, speed=(2., 0.))
        for _ in range(3):                                         # same sample, same speed stamp
            g.observe(blip, GOAL, True)
        self.assertEqual(g.resets, 0)
        self.assertEqual(g.blips_tolerated, 1)

    def test_overspeed_with_angle_shift_resets(self):
        g = self.qualifier()
        samples = [sample(i * .05) for i in range(12)]
        samples.append(sample(.6, speed=(2., 0.), error=(0.08, 0.01)))   # moved 0.07 deg
        self.run_samples(g, samples)
        reset = [e for e in g.events if e["kind"] == "reset"][-1]
        self.assertEqual(reset["reason"], "angle_shift")
        self.assertAlmostEqual(reset["angle_change_deg"][0], .07, places=6)

    def test_error_outside_tolerance_resets(self):
        g = self.qualifier()
        samples = [sample(i * .05) for i in range(12)]
        samples.append(sample(.6, error=(0.35, 0.01)))
        self.run_samples(g, samples)
        reset = [e for e in g.events if e["kind"] == "reset"][-1]
        self.assertEqual(reset["reason"], "angle_error_out_of_tolerance")
        self.assertEqual(g.resets, 1)

    def test_hard_speed_is_never_tolerated(self):
        g = self.qualifier()
        samples = [sample(i * .05) for i in range(12)]
        samples.append(sample(.6, speed=(12., 0.)))
        self.run_samples(g, samples)
        self.assertEqual(g.failure, "motion_goal_hold_hard_limit")
        self.assertEqual(g.blips_tolerated, 0)

    def test_tolerance_can_be_disabled(self):
        g = self.qualifier(tolerate_speed_blip=False)
        samples = [sample(i * .05) for i in range(12)]
        samples.append(sample(.6, speed=(2., 0.)))
        self.run_samples(g, samples)
        self.assertEqual(g.resets, 1)
        self.assertEqual([e for e in g.events if e["kind"] == "reset"][-1]["reason"], "overspeed")


if __name__ == "__main__":
    unittest.main()
