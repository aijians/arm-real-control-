"""D19 rail following: pure unit tests and closed-loop runs on real routes."""
import copy
import math
from pathlib import Path
import unittest

from control.config import load_config
from control.follower import FollowerSettings, RailFollower, monotone_segments
from control.motors import HardwareError
from control.planner import Planner
from follower_sim import simulate
from virtual_drive import VirtualDrive

ROOT = Path(__file__).resolve().parents[1]
SLIDERS = {"蓝": -42.94380256916034, "橙": 8.95059312363668}
LIMITS = [-65.80299245109094, -6.0521089463364035]


def feed(f, t, q, written=None):
    return f.update(t, q, written)


class SegmentationTests(unittest.TestCase):
    def test_splits_where_any_axis_reverses_and_keeps_axes_monotone(self):
        path = [[164., 0.], [166., 2.], [168., 4.], [168., 6.], [166., 8.], [164., 10.], [164., 9.]]
        segs = monotone_segments(path)
        self.assertEqual(segs, [(0, 2), (2, 5), (5, 6)])
        for a, b in segs:
            for axis in (0, 1):
                d = [path[k + 1][axis] - path[k][axis] for k in range(a, b)]
                self.assertTrue(all(x >= 0 for x in d) or all(x <= 0 for x in d))


class LimitTests(unittest.TestCase):
    PATH = [[170. - .1 * k, 20.] for k in range(31)]          # 3 deg on the upper arm

    def test_write_step_and_total_lead_are_limited_separately(self):
        f = RailFollower(self.PATH, "free")
        q, written, steps, t = [170., 20.], [170., 20.], [], 0.
        for _ in range(200):                                     # arm never moves: 4 s
            out = feed(f, t, q, written)
            steps.append(abs(out["targets_deg"][0] - written[0]))
            written = out["targets_deg"]
            t += .02
            if out["failure"]:
                break
        self.assertLessEqual(max(steps), .1 + 1e-12)              # write step <= 0.1 deg
        self.assertAlmostEqual(170. - written[0], 1.0, places=9)  # total lead capped at 1.0 deg
        self.assertGreater(sum(1 for s in steps if s > 1e-12), 9)  # reached 1.0 in >= 10 writes
        self.assertEqual(out["failure"]["reason"], "motion_no_progress")
        self.assertEqual(out["failure"]["axis"], 0)

    def test_lead_grows_only_while_stalled_and_decays_while_moving(self):
        f = RailFollower(self.PATH, "free")
        t, q = 0., [170., 20.]
        for _ in range(40):
            out = feed(f, t, q); t += .02
        grown = out["lead_state_deg"][0]
        self.assertGreater(grown, .35 + .3)
        for _ in range(40):                                      # now moving well: 4 deg/s
            q = [q[0] - .08, 20.]
            out = feed(f, t, q); t += .02
        self.assertLess(out["lead_state_deg"][0], grown)

    def test_target_never_moves_backwards_when_arm_jumps_ahead(self):
        f = RailFollower(self.PATH, "free")
        out = feed(f, 0., [170., 20.])
        out = feed(f, .5, [170., 20.])
        before = out["targets_deg"][0]
        out = feed(f, .52, [169.0, 20.])                         # arm lurches past the target
        self.assertLessEqual(out["targets_deg"][0], before + 1e-12)
        out = feed(f, .54, [169.95, 20.])                        # rebound behind again
        self.assertLessEqual(out["targets_deg"][0], before + 1e-12)

    def test_lead_tapers_toward_the_end_and_target_stops_at_the_end(self):
        f = RailFollower(self.PATH, "free")
        t, q, written = 0., [170., 20.], [170., 20.]
        checked = 0
        while t < 20:
            out = feed(f, t, q, written)
            remaining = q[0] - 167.
            if remaining > 1e-9:
                self.assertLessEqual(out["lead_limit_deg"][0], max(.35, .6 * remaining) + 1e-9)
                checked += 1
            self.assertGreaterEqual(out["targets_deg"][0], 167. - 1e-12)   # never past the end
            written = out["targets_deg"]
            q = [written[0], 20.]                                # arm reaches each target
            t += .02
            if out["at_end"]:
                break
        self.assertTrue(out["at_end"])
        self.assertAlmostEqual(written[0], 167., places=9)
        self.assertGreater(checked, 10)

    def test_settings_validation(self):
        with self.assertRaises(ValueError):
            FollowerSettings(lead_min_deg=1.2, lead_max_deg=1.0).validate()
        with self.assertRaises(ValueError):
            FollowerSettings(speed_deg_s=(20., 5.)).validate()   # margin 1.235 > lead_max


class CornerAndSyncTests(unittest.TestCase):
    def test_no_axis_passes_a_route_corner_before_every_axis_reached_it(self):
        path = ([[165. + .1 * k, 10. + .1 * k] for k in range(21)]       # both rise to 167/12
                + [[167. - .1 * k, 12. + .1 * k] for k in range(1, 21)])  # upper reverses
        f = RailFollower(path, "free")
        t, written = 0., list(path[0])
        q = [166.95, 10.5]                                       # upper almost at corner, lower far
        for _ in range(100):
            out = feed(f, t, q, written); written = out["targets_deg"]; t += .02
        self.assertEqual(out["segment"], 0)
        self.assertLessEqual(written[0], 167. + 1e-9)            # upper held at the corner
        q = [167., 11.9]
        out = feed(f, t, q, written)
        self.assertEqual(out["segment"], 1)

    def test_free_axes_advance_independently_within_desync_bound(self):
        path = [[170. - .1 * k, 20. + .2 * k] for k in range(51)]      # 5 deg upper, 10 deg lower
        f = RailFollower(path, "free")
        t, written = 0., list(path[0])
        q = [170., 20.]
        for _ in range(300):
            out = feed(f, t, q, written); written = out["targets_deg"]; t += .02
            q = [170., written[1]]                               # upper stuck, lower follows exactly
            if out["failure"]:
                break
        # Lower advanced on its own, but not beyond 1.0 deg past the rail point at the upper's progress.
        self.assertGreater(written[1], 20.5)
        self.assertLessEqual(written[1], 20. + 1.0 + 1e-9)

    def test_push_slave_follows_master_measured_progress_not_its_command(self):
        path = [[163.4 - .2 * k, 53.15 - .06 * k] for k in range(51)]  # straight, both monotone
        f = RailFollower(path, "push")
        t, written = 0., list(path[0])
        out = None
        for _ in range(60):
            q = [163.4 - .0, written[1]]                         # master held back by the load
            out = feed(f, t, q, written); written = out["targets_deg"]; t += .02
        self.assertEqual(out["master"], 0)
        self.assertGreater(163.4 - written[0], .6)               # master pulled ahead (force)
        # Slave target stays at the rail point of the master's *measured* progress + a small lead.
        self.assertLessEqual(53.15 - written[1], f.slave_lead + 1e-9)


class RouteTests(unittest.TestCase):
    """Real planner routes under a load with a critical point."""

    @classmethod
    def setUpClass(cls):
        cfg, _ = load_config(ROOT)
        cls.cfg, cls.planner = cfg, Planner(ROOT, cfg)

    def free_route(self, start):
        seed = list(map(math.radians, start))
        pre = self.planner.precontact("蓝", "+x", SLIDERS, seed)
        return self.planner.free(seed, pre, SLIDERS, direction="+x")

    def push_route(self):
        seed = list(map(math.radians, [163.40, 53.15]))
        return self.planner.push(seed, "蓝", LIMITS[1], SLIDERS, LIMITS)

    @staticmethod
    def deg(phase):
        return [[math.degrees(v) for v in p["q_rad"]] for p in phase["points"]]

    # 8e4d-like: 0.15 deg lag outside heavy bands; above a critical angle the
    # upper needs 0.75 deg, and the lower 0.65 deg between 25 and 45 deg (the
    # real 8e4d stalls: upper ~170.9-171.9, lower 29.5/34.3, lag 0.52-0.60).
    # The endpoint region stays light, as in 5d42 (endpoint error 0.013 deg).
    @staticmethod
    def free_lag(axis, q):
        return (.15 if q < 170. else .75) if axis == 0 else (.65 if 25. <= q <= 45. else .15)

    # 5d42-like push: the loaded upper arm lags 0.8 deg once in contact and is
    # light again over the last 1.5 deg (slider at its goal); lower is light.
    @staticmethod
    def push_lag(axis, q):
        return (.15 if q > 161. or q < 150.5 else .8) if axis == 0 else .02

    def check_common(self, run, settings=FollowerSettings()):
        self.assertEqual(run["unsafe"], [])
        for axis, step in run["steps"]:
            self.assertLessEqual(abs(step), settings.write_step_deg[axis] + 1e-9)
        for row in run["trace"]:
            self.assertTrue(all(l <= settings.lead_max_deg + 1e-9 for l in row["lead"]))

    def test_free_route_with_critical_load_completes(self):
        path = self.deg(self.free_route([164.70, 0.06]))
        run = simulate(path, "free", self.free_lag)
        self.assertEqual(run["status"], "completed", run.get("reason"))
        self.check_common(run)
        self.assertGreater(max(max(r["lead_state"]) for r in run["trace"]), .65)

    def test_push_route_with_heavy_master_keeps_the_tool_on_the_line(self):
        path = self.deg(self.push_route())
        run = simulate(path, "push", self.push_lag)
        self.assertEqual(run["status"], "completed", run.get("reason"))
        self.check_common(run)
        y = lambda q: self.planner.arm.slider_center(list(map(math.radians, q)), "+x")[1]
        line = y(path[0])
        offsets = sorted(abs(y(r["q"]) - line) for r in run["trace"])
        self.assertLess(offsets[-1], .2, "tool left the Y line by %.3f mm" % offsets[-1])
        self.assertLess(offsets[len(offsets) // 2], .1)

    def test_genuine_stall_is_still_reported(self):
        path = self.deg(self.push_route())
        run = simulate(path, "push", lambda axis, q: (.15 if q > 161. else 5.) if axis == 0 else .02)
        self.assertEqual(run["status"], "failed")
        self.assertEqual(run["reason"], "motion_no_progress")

    def _node_executor(self, phase, lag_model_deg, trace=None):
        hw = copy.deepcopy(self.cfg["hardware"])
        for m in hw["motors"]:
            m["device"] = "/tmp/never-opened-" + m["name"]
        motors = hw["motors"]
        model = lambda i, motor_deg: (motor_deg - motors[i]["offset_deg"]) / (motors[i]["sign"] * motors[i]["ratio"])
        start = [math.degrees(v) for v in phase["points"][0]["q_rad"]]
        record = (lambda d: trace.append([model(i, d.q[i]) for i in range(2)])) if trace is not None else None
        d = VirtualDrive(model_q=start, settings=hw, disturbance=record,
                         load_model=lambda i, m: lag_model_deg(i, model(i, m)))
        return d.run(phase)

    def test_motorpair_completes_free_in_the_heavy_band(self):
        """Wired executor (MotorPair._move_follow) on the 8e4d-like route and load."""
        trace = []
        result = self._node_executor(self.free_route([164.70, 0.06]), self.free_lag, trace)
        self.assertEqual(result["status"], "completed")
        log = result["motion_diagnostics"]["execution"]["follower_log"]
        self.assertTrue(log)
        self.assertLessEqual(max(max(abs(x) for x in s["lead"]) for s in log), 1.0 + 1e-6)

    def test_motorpair_push_with_heavy_master_completes_near_the_line(self):
        phase = self.push_route()
        y = lambda q: self.planner.arm.slider_center(list(map(math.radians, q)), "+x")[1]
        line = y(self.deg(phase)[0])
        trace = []
        result = self._node_executor(phase, self.push_lag, trace)
        self.assertEqual(result["status"], "completed")
        # Real executor timing (20 Hz feedback, A6 gate) adds to the slave's
        # direction-margin offset; the standalone simulation stays below 0.2 mm.
        self.assertLess(max(abs(y(q) - line) for q in trace), .3)


class StallResendAndDriveStatusTests(unittest.TestCase):
    """Fix A: resend a stalled axis's target at standstill; log 0x9A drive status."""

    def run_lower(self, *, drop_final_once=False, error_state=0, force=True):
        from unittest.mock import patch
        import control.motors as mm
        from virtual_drive import config, phase
        hw = config(max_speed_deg_s=60., segmented_motion={"enabled": True})
        pts = [[169., 20. + .2 * k] for k in range(16)]               # lower 20 -> 23 deg
        dropped = set()
        def drop(axis, target):
            # Each target within 1 deg of the end is acknowledged once but not
            # adopted: the arm stops >1 deg short while the follower already
            # holds the (unchanging) end target.
            key = round(target * 100)
            hit = drop_final_once and axis == 1 and target >= 22. and key not in dropped
            if hit:
                dropped.add(key)
            return hit
        d = VirtualDrive(model_q=pts[0], settings=hw, drop_acked=drop)
        d.drive_error_state = error_state
        original = mm.MotorPair._prepare_target
        def maybe_no_force(self, i, q, **options):
            if not force:
                options["force_resend_axes"] = None
            return original(self, i, q, **options)
        with patch.object(mm.MotorPair, "_prepare_target", maybe_no_force):
            try:
                return d, d.run(phase(pts, times=[.02 * k for k in range(16)])), None
            except HardwareError as exc:
                return d, exc.details.get("motion_diagnostics"), exc.reason

    def test_dropped_final_target_is_recovered_by_a_stall_resend(self):
        d, result, reason = self.run_lower(drop_final_once=True)
        self.assertIsNone(reason, reason)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(d.ignored_targets)
        self.assertGreater(result["motion_diagnostics"]["execution"]["stall_resends"][1], 0)
        self.assertEqual(d.unsafe_directions, [])

    def test_without_the_resend_the_same_drop_fails(self):
        d, diagnostics, reason = self.run_lower(drop_final_once=True, force=False)
        self.assertIn(reason, ("motion_goal_settle_timeout", "motion_no_progress"))

    def test_drive_status_is_logged_each_cycle(self):
        d, result, reason = self.run_lower(error_state=64)
        self.assertIsNone(reason, reason)
        ex = result["motion_diagnostics"]["execution"]
        self.assertEqual(ex["drive_status1"][1]["error_flags"], ["stall"])
        logged = [s["drive"][1]["status1"] for s in ex["follower_log"] if s["drive"][1]["status1"]]
        self.assertTrue(logged)
        self.assertIn("current_raw", ex["follower_log"][-1]["drive"][1])


if __name__ == "__main__":
    unittest.main()
