"""Offline policy and real-executor checks; no physical drive model claims."""
import math
import unittest

from control.goal_refresh import StationaryGoalRefresh, goal_refresh_settings
from virtual_drive import VirtualDrive, phase, m
from test_free_timed import settings, targets


def enabled(**extra):
    cfg = settings(**extra)
    cfg["free_timed_motion"]["goal_refresh"] = {"enabled": True}
    return cfg


class GoalRefreshPolicyTests(unittest.TestCase):
    def state(self, stamp, q=20., speed=0., speed_stamp=None):
        return {"q_rad": [math.radians(q)], "speed_rad_s": [math.radians(speed)],
                "axis_stamp_monotonic_s": [stamp],
                "speed_stamp_monotonic_s": [stamp if speed_stamp is None else speed_stamp]}

    def policy(self):
        return StationaryGoalRefresh(goal_refresh_settings({"goal_refresh": {"enabled": True}}),
                                     .3, 1., 1)

    def test_requires_new_angle_and_speed_after_command(self):
        p = self.policy(); goal = [math.radians(22)]
        for t in (1., 2., 3.):
            self.assertEqual(p.observe(self.state(t, speed_stamp=.9), goal, [1.]), [])
        self.assertEqual(p.observe(self.state(4.), goal, [1.]), [])
        for _ in range(10):
            self.assertEqual(p.observe(self.state(4.), goal, [1.]), [])
        self.assertEqual(p.observe(self.state(4.6), goal, [1.]), [0])
        p.sent([0])
        self.assertEqual(p.observe(self.state(4.6), goal, [4.7]), [])
        self.assertEqual(p.observe(self.state(4.8), goal, [4.7]), [])
        self.assertEqual(p.observe(self.state(5.4), goal, [4.7]), [0])
        p.sent([0])
        self.assertEqual(p.observe(self.state(10.), goal, [5.5]), [])

    def test_progress_speed_and_in_band_reset_stationary_window(self):
        for state in (self.state(1.5, q=20.06), self.state(1.5, speed=2),
                      self.state(1.5, q=22)):
            with self.subTest(state=state):
                p=self.policy(); goal=[math.radians(22)]
                self.assertEqual(p.observe(self.state(1.), goal, [1.]), [])
                self.assertEqual(p.observe(state, goal, [1.]), [])
                self.assertEqual(p.observe(self.state(1.6, q=20.06), goal, [1.]), [])

    def test_settings_reject_invalid_values(self):
        for key, bad in (("enabled", 1), ("stationary_s", True), ("stationary_s", 0),
                         ("progress_deg", float("nan")), ("max_attempts", True),
                         ("max_attempts", 4), ("unknown", 1)):
            with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                goal_refresh_settings({"goal_refresh": {key: bad}})


class GoalRefreshExecutorTests(unittest.TestCase):
    def test_normal_arrival_does_not_refresh_or_change_schedule(self):
        d = VirtualDrive(settings=enabled())
        r = d.run(phase([[159,20],[158,22]]))
        self.assertEqual(len(d.a4()), 20)
        self.assertEqual(r["motion_diagnostics"]["execution"]["goal_refresh"]["attempts"], [0,0])

    def test_stationary_axis_only_twice_and_original_deadline_still_stops(self):
        d = VirtualDrive(settings=enabled(), immobile=True)
        with self.assertRaises(m.HardwareError) as caught:
            d.run(phase([[159,20],[159,22]]))
        e = caught.exception.details["motion_diagnostics"]["execution"]
        self.assertEqual(caught.exception.reason, "motion_goal_settle_timeout")
        self.assertEqual(e["goal_refresh"]["attempts"], [0,2])
        self.assertEqual(e["waypoints_dispatched"], 10)
        self.assertEqual(len(targets(d)), 10)
        self.assertEqual(targets(d,1)[-3:], [22.,22.,22.])
        self.assertEqual(len(targets(d,1)),12)
        self.assertEqual(e["motion_commands_sent"],22)
        self.assertLess(d.halts[0]-e["dispatch_completed_elapsed_s"],3.1)
        self.assertEqual(len(d.halts),1)
        self.assertFalse(d.pair._move_lock.locked())

    def test_repeated_goal_can_recover_synthetic_latched_target(self):
        # A synthetic latch exercises the recovery branch, not actual friction.
        class LatchedDrive(VirtualDrive):
            repeats = 0
            def send(self, axis, wire, **kwargs):
                super().send(axis, wire, **kwargs)
                if axis == 1 and wire[1] == 0xa4 and int.from_bytes(wire[5:13],"little") == 2200:
                    self.repeats += 1
                    if self.repeats == 2:
                        self.immobile = False
        d = LatchedDrive(settings=enabled(), immobile=True)
        r = d.run(phase([[159,20],[159,22]]))
        e = r["motion_diagnostics"]["execution"]
        self.assertEqual(e["goal_refresh"]["attempts"],[0,1])
        self.assertEqual(targets(d,1)[-2:],[22.,22.])
        self.assertEqual(d.a4(1)[-1][3],d.a4(1)[-2][3])
        self.assertEqual(len(d.a4(0)),10)
        self.assertGreater(e["goal_stable_s"],.39)
        self.assertEqual(d.halts,[])

    def test_push_correction_disabled_and_noop_never_refresh(self):
        for name in ("push","push_correction"):
            d=VirtualDrive(settings=enabled(),immobile=True)
            with self.assertRaises(m.HardwareError):
                d.run(phase([[159,20],[159,22]],name=name))
            self.assertEqual(len(d.a4()),20)
        cfg=enabled(); cfg["free_timed_motion"]["goal_refresh"]["enabled"]=False
        d=VirtualDrive(settings=cfg,immobile=True)
        with self.assertRaises(m.HardwareError):
            d.run(phase([[159,20],[159,22]]))
        self.assertEqual(len(d.a4()),20)
        d=VirtualDrive(settings=enabled())
        self.assertEqual(d.run(phase([[159,20]],times=[0]))["status"],"completed")
        self.assertEqual(d.a4(),[])

    def test_feedback_loss_prevents_refresh(self):
        d=VirtualDrive(settings=enabled(),immobile=True,
                       query_loss=lambda axis,cmd,t: axis==1 and t>.6)
        with self.assertRaises(m.HardwareError) as caught:
            d.run(phase([[159,20],[159,22]]))
        self.assertEqual(caught.exception.reason,"joint_feedback_stale")
        self.assertEqual(len(d.a4(1)),10)
        self.assertEqual(len(d.halts),1)

    def test_refresh_send_failure_stops_and_unlocks(self):
        d=VirtualDrive(settings=enabled(),immobile=True,
            fail_write=lambda axis,wire,d: axis==1 and wire[1]==0xa4 and len(d.a4(1))>=10)
        with self.assertRaises(m.HardwareError) as caught:
            d.run(phase([[159,20],[159,22]]))
        self.assertEqual(caught.exception.reason,"motion_failed")
        self.assertEqual(len(d.halts),1)
        self.assertFalse(d.pair._move_lock.locked())

    def test_both_axes_have_separate_budgets_and_preserve_motor_mapping(self):
        cfg=enabled()
        cfg["motors"][0].update(sign=-1,ratio=1.5,offset_deg=300.)
        d=VirtualDrive(settings=cfg,immobile=True)
        with self.assertRaises(m.HardwareError) as caught:
            d.run(phase([[159,20],[157,22]]))
        e=caught.exception.details["motion_diagnostics"]["execution"]
        self.assertEqual(e["goal_refresh"]["attempts"],[2,2])
        self.assertEqual(targets(d)[-3:],[64.5]*3)
        self.assertEqual(targets(d,1)[-3:],[22.]*3)
        for axis in (0,1):
            self.assertEqual(d.a4(axis)[-1][3],d.a4(axis)[-3][3])

    def test_continued_slow_progress_does_not_refresh(self):
        # The integer speed may read zero while position continues changing.
        def creeping(d):
            if d.elapsed > .6:
                d.command_speed[1]=.2
        d=VirtualDrive(settings=enabled(),disturbance=creeping)
        # Keep the lower axis immobile through tracking, then let it creep.
        d.immobile=True
        original=creeping
        def release(d):
            original(d)
            if d.elapsed > .6:
                d.immobile=False
        d.disturbance=release
        with self.assertRaises(m.HardwareError) as caught:
            d.run(phase([[159,20],[159,22]]))
        self.assertEqual(caught.exception.reason,"motion_goal_settle_timeout")
        self.assertEqual(len(d.a4(1)),10)

    def test_cancel_or_health_failure_during_settling_never_refreshes(self):
        for mode in ("cancel","health"):
            d=VirtualDrive(settings=enabled(),immobile=True)
            def health():
                if d.elapsed>1.:
                    if mode=="cancel":
                        d.pair._halt_event.set()
                    else:
                        raise m.HardwareError("camera_unavailable")
            with self.assertRaises(m.HardwareError) as caught:
                d.run(phase([[159,20],[159,22]]),health_check=health)
            self.assertEqual(caught.exception.reason,
                             "motion_halted" if mode=="cancel" else "camera_unavailable")
            self.assertEqual(len(d.a4(1)),10)
            self.assertEqual(len(d.halts),1)
            self.assertFalse(d.pair._move_lock.locked())


if __name__ == "__main__":
    unittest.main()
