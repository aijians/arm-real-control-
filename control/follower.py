"""Rail following ("rubber band") targets for segmented phases (decision D19).

Pure computation: no I/O and no clock of its own; the caller passes time,
measured joint angles and the targets it actually wrote. Angles are model
degrees, axis 0 = upper arm, axis 1 = lower arm.

The planned joint polyline P[0..N] is treated as a rail, not a list of stops.
Each update projects the measured joints onto the rail as a monotone progress
and places each axis's A6 target a bounded, adaptive lead ahead of its
measured angle along the rail.

Two separately limited quantities:

* ``write_step_deg`` -- change of one axis's written target per update
  (a rate limiter; D16: upper 0.1 deg, lower 0.2 deg);
* ``lead`` -- target minus measured angle along the travel direction,
  between the direction margin and ``lead_max_deg`` (1.0 deg).

Hard constraints:

* a target is never commanded backwards inside a monotone segment; a lead
  shrinks only by holding the target while the arm catches up;
* every lead is at least the A6 direction margin (speed field x worst
  feedback age + 0.015 deg) -- A6 carries a direction byte;
* a target never passes the segment end; while an axis moves freely its
  lead also tapers toward that end (no taper while it is held back by load:
  removing the lead it needs would stall it, and a loaded axis stops short
  of its target anyway);
* the next monotone segment starts only after every axis reached the end of
  the current one -- within the corner tolerance, or held by load at the
  corner target no further than lead_max.

Lead control per adaptive axis, over ``stall_window_s``: stalled (moved
< ``progress_eps_deg`` while actually pulled by at least the minimum lead)
-> grow at ``lead_up_deg_s``; moving slowly (pulled, below
``progress_good_ratio`` x speed field) -> grow at ``slow_lead_gain`` x that
rate; moving well -> shrink; otherwise hold.
An axis waiting at its target (a Push slave, a Free axis held by the
desync bound) is not stalled.

Push (``mode="push"``) is master-slave: the master (the axis with the larger
travel in the segment, i.e. the loaded upper arm) defines the measured
progress; the slave is aligned with that *measured* progress plus a small
forward lead, never with the master's command, so the tool stays on the
planned straight line however far the loaded master lags.

Free (``mode="free"``) advances each axis independently along its monotone
segment. For collision safety no axis's target may run more than
``free_desync_deg`` (in its own angle) ahead of the rail point at the slowest
axis's progress.
"""
import bisect
from collections import deque
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class FollowerSettings:
    write_step_deg: tuple = (0.1, 0.2)
    lead_min_deg: float = 0.35
    lead_max_deg: float = 1.0
    lead_up_deg_s: float = 1.0
    lead_down_deg_s: float = 0.5
    stall_window_s: float = 0.3
    progress_eps_deg: float = 0.02
    progress_good_ratio: float = 0.4
    slow_lead_gain: float = 0.5
    slave_lead_deg: float = 0.05
    slave_catch_up_per_s: float = 4.0
    free_desync_deg: float = 1.0
    taper_gain: float = 0.6
    corner_tolerance_deg: float = 0.3
    speed_deg_s: tuple = (5.0, 5.0)
    feedback_age_s: float = 0.06
    margin_pad_deg: float = 0.02
    no_progress_s: float = 2.0
    min_speed_deg_s: float = 0.5

    def validate(self):
        values = (self.lead_min_deg, self.lead_max_deg, self.lead_up_deg_s, self.lead_down_deg_s,
                  self.stall_window_s, self.progress_eps_deg, self.progress_good_ratio, self.slow_lead_gain, self.slave_lead_deg, self.slave_catch_up_per_s,
                  self.free_desync_deg, self.taper_gain, self.corner_tolerance_deg,
                  self.feedback_age_s, self.no_progress_s, self.min_speed_deg_s)
        if (len(self.write_step_deg) != 2 or len(self.speed_deg_s) != 2
                or any(not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0)
                       for v in (*values, *self.write_step_deg, *self.speed_deg_s))
                or self.margin_pad_deg < 0):
            raise ValueError("invalid follower settings")
        if not self.lead_min_deg <= self.lead_max_deg <= 1.8:
            raise ValueError("follower lead range must satisfy lead_min <= lead_max <= 1.8 deg")
        for speed in self.speed_deg_s:
            if self.margin(speed) > self.lead_max_deg:
                raise ValueError("speed field too high: direction margin exceeds lead_max")
        return self

    def margin(self, speed_deg_s):
        """Smallest lead that keeps an A6 target direction-safe while moving."""
        return speed_deg_s * self.feedback_age_s + 0.015 + self.margin_pad_deg


def _sign(value):
    return 0 if abs(value) <= 1e-12 else (1 if value > 0 else -1)


def monotone_segments(path):
    """Split node indices at every node where any axis reverses direction.

    Within each returned (start, end) pair every axis is monotone
    (non-strictly): a flat stretch followed by motion either way is allowed.
    """
    n = len(path)
    if n < 2:
        return [(0, 0)]
    boundaries = {0, n - 1}
    for axis in range(len(path[0])):
        last_sign, last_end = 0, 0
        for k in range(1, n):
            s = _sign(path[k][axis] - path[k - 1][axis])
            if not s:
                continue
            if last_sign and s != last_sign:
                boundaries.add(last_end)
            last_sign, last_end = s, k
    ordered = sorted(boundaries)
    return list(zip(ordered, ordered[1:]))


class RailFollower:
    """Stateful target generator for one planned phase; see module doc."""

    def __init__(self, path_deg, mode, settings=None, *, start_q_deg=None):
        if mode not in ("free", "push"):
            raise ValueError("mode must be 'free' or 'push'")
        if len(path_deg) < 2 or any(len(p) != 2 for p in path_deg):
            raise ValueError("path needs at least two two-axis points")
        if any(not math.isfinite(v) for p in path_deg for v in p):
            raise ValueError("path must be finite")
        self.settings = (settings or FollowerSettings()).validate()
        self.mode = mode
        self.path = [list(map(float, p)) for p in path_deg]
        self.segments = monotone_segments(self.path)
        self.seg = 0
        self.targets = list(map(float, start_q_deg if start_q_deg is not None else self.path[0]))
        self.lead = [self.settings.lead_min_deg] * 2
        self.history = [deque(), deque()]
        self.stall_since = [None, None]
        self.moving_well = [False, False]
        self.failure = None
        self.last_t = None
        self._enter_segment()

    # ----- rail geometry within the current monotone segment -------------
    def _enter_segment(self):
        a, b = self.segments[self.seg]
        self.dirs = [_sign(self.path[b][i] - self.path[a][i]) for i in range(2)]
        self.values = [[self.dirs[i] * self.path[k][i] for k in range(a, b + 1)] for i in range(2)]
        travel = [abs(self.path[b][i] - self.path[a][i]) for i in range(2)]
        self.master = (0 if travel[0] >= travel[1] else 1) if self.mode == "push" else None
        self.progress = [float(a)] * 2
        s = self.settings
        speeds = list(s.speed_deg_s)
        if self.master is not None:
            slave = 1 - self.master
            ratio = travel[slave] / travel[self.master] if travel[self.master] > 1e-12 else 0.
            # Just fast enough to keep pace with the master: a higher field
            # would only enlarge the slave's direction margin (= its lead).
            speeds[slave] = min(speeds[self.master],
                                max(s.min_speed_deg_s, ratio * speeds[self.master]))
        self.speeds = speeds
        self.lead_floor = [max(s.lead_min_deg, s.margin(v)) for v in speeds]
        self.lead = [min(max(l, f), s.lead_max_deg) for l, f in zip(self.lead, self.lead_floor)]
        if self.master is not None:
            slave = 1 - self.master
            self.slave_lead = max(s.slave_lead_deg, s.margin(speeds[slave]))
        for h in self.history:
            h.clear()
        self.stall_since = [None, None]

    def _angle(self, axis, s):
        a, b = self.segments[self.seg]
        s = min(max(s, a), b)
        k = min(int(s), b - 1) if b > a else a
        f = s - k
        return self.path[k][axis] + (self.path[min(k + 1, b)][axis] - self.path[k][axis]) * f

    def _progress_of(self, axis, q):
        """Furthest progress whose rail angle the measured angle has reached."""
        a, b = self.segments[self.seg]
        if not self.dirs[axis]:
            return float(b)
        v, x = self.values[axis], self.dirs[axis] * q
        j = bisect.bisect_right(v, x) - 1
        if j < 0:
            return float(a)
        if j >= len(v) - 1:
            return float(b)
        return a + j + (x - v[j]) / (v[j + 1] - v[j])

    def _s_for_value(self, axis, x):
        """Earliest progress at which the axis's signed rail angle reaches x."""
        a, b = self.segments[self.seg]
        v = self.values[axis]
        k = bisect.bisect_left(v, x)
        if k <= 0:
            return float(a)
        if k >= len(v):
            return float(b)
        return a + k - 1 + (x - v[k - 1]) / (v[k] - v[k - 1])

    # ----- main update -----------------------------------------------------
    def update(self, now, q_deg, written_deg=None, speed_deg_s=None):
        """Return the next targets.

        ``written_deg``: targets actually sent (the A6 gate may hold some);
        ``speed_deg_s``: measured joint speeds, used to pace a Push slave.
        """
        s = self.settings
        dt = 0. if self.last_t is None else max(0., now - self.last_t)
        self.last_t = now
        if written_deg is not None:
            self.targets = [t if w is None else float(w) for t, w in zip(self.targets, written_deg)]
        # Leave a segment only when every axis reached its end (route corner).
        while self.seg < len(self.segments) - 1:
            b = self.segments[self.seg][1]
            if all(self._corner_reached(i, b, q_deg[i], now) for i in range(2)):
                self.seg += 1
                self._enter_segment()
            else:
                break
        a, b = self.segments[self.seg]
        last = self.seg == len(self.segments) - 1
        for i in range(2):
            self.progress[i] = max(self.progress[i], self._progress_of(i, q_deg[i]))
        stalled = [False, False]
        for i in range(2):
            d = self.dirs[i]
            if not d:
                continue
            h = self.history[i]
            h.append((now, q_deg[i]))
            while len(h) > 2 and now - h[1][0] >= s.stall_window_s:
                h.popleft()
            remaining = d * (self.path[b][i] - q_deg[i])
            spans = now - h[0][0] >= s.stall_window_s - 1e-9
            moved = d * (q_deg[i] - h[0][1])
            pulled = d * (self.targets[i] - q_deg[i]) >= self.lead_floor[i] - 1e-9 if (
                self.master is None or i == self.master) else d * (self.targets[i] - q_deg[i]) >= self.slave_lead - 1e-9
            stalled[i] = bool(spans and pulled and remaining > s.corner_tolerance_deg
                              and moved < s.progress_eps_deg and not self._corner_reached(i, b, q_deg[i], now))
            good = s.progress_good_ratio * self.speeds[i] * s.stall_window_s
            self.moving_well[i] = bool(spans and moved >= good)
            slow = bool(spans and pulled and not stalled[i] and not self.moving_well[i]
                        and remaining > s.corner_tolerance_deg)
            adaptive = self.master is None or i == self.master
            if adaptive:
                rate = (s.lead_up_deg_s if stalled[i] else
                        -s.lead_down_deg_s if self.moving_well[i] else
                        s.slow_lead_gain * s.lead_up_deg_s if slow else 0.)
                self.lead[i] = min(s.lead_max_deg, max(self.lead_floor[i], self.lead[i] + rate * dt))
            at_limit = (not adaptive) or self.lead[i] >= s.lead_max_deg - 1e-9
            if stalled[i] and at_limit:
                self.stall_since[i] = now if self.stall_since[i] is None else self.stall_since[i]
                if now - self.stall_since[i] >= s.no_progress_s - 1e-9 and self.failure is None:
                    self.failure = {"reason": "motion_no_progress", "axis": i,
                                    "lead_deg": self.lead[i], "remaining_deg": remaining}
            elif not stalled[i]:
                self.stall_since[i] = None
        desired = list(self.targets)
        lead_eff = [0., 0.]
        for i in range(2):
            d = self.dirs[i]
            if not d:
                desired[i] = self.path[b][i]
                continue
            remaining = d * (self.path[b][i] - q_deg[i])
            lead_eff[i] = (min(self.lead[i], max(self.lead_floor[i], s.taper_gain * remaining))
                           if self.moving_well[i] else self.lead[i])
        if self.master is not None:
            m, sl = self.master, 1 - self.master
            h = self.history[m]
            span = now - h[0][0] if h else 0.
            # No history yet: assume the master is still (never over-pace the slave).
            master_avg = abs(q_deg[m] - h[0][1]) / span if span > 1e-9 else 0.
            behind = (self.dirs[sl] * (self._angle(sl, self.progress[m]) - q_deg[sl])
                      if self.dirs[sl] else 0.)
            self._update_slave_speed(self.progress[m], master_avg, behind)
            if self.dirs[m]:
                x_cmd = min(self.dirs[m] * q_deg[m] + lead_eff[m], self.values[m][-1])
                s_cmd = self._s_for_value(m, x_cmd)
                desired[m] = self._angle(m, s_cmd)
            else:
                s_cmd = float(b)
            if self.dirs[sl]:
                base = self.dirs[sl] * self._angle(sl, self.progress[m])
                cap = self.dirs[sl] * self._angle(sl, s_cmd)
                x = min(base + self.slave_lead, max(cap, base), self.values[sl][-1])
                desired[sl] = self.dirs[sl] * x
                lead_eff[sl] = x - self.dirs[sl] * q_deg[sl]
        else:
            moving = [i for i in range(2) if self.dirs[i]]
            # An axis held by load at its segment-end target is done for this
            # segment; it must not hold the other axis back from the corner.
            pacing = [i for i in moving if not self._corner_reached(i, b, q_deg[i], now)]
            slowest = min((self.progress[i] for i in pacing), default=float(b))
            for i in moving:
                d = self.dirs[i]
                limit = d * self._angle(i, slowest) + s.free_desync_deg
                x = min(d * q_deg[i] + lead_eff[i], limit, self.values[i][-1])
                desired[i] = d * x
        out = []
        for i in range(2):
            d, prev = self.dirs[i], self.targets[i]
            want = desired[i]
            if d and d * (want - prev) < 0 and self.seg_contains(prev, i):
                want = prev                              # never command backwards
            step = s.write_step_deg[i]
            want = prev + max(-step, min(step, want - prev))
            out.append(want)
        self.targets = out
        at_end = last and all(abs(out[i] - self.path[b][i]) <= 1e-9 for i in range(2))
        return {"targets_deg": list(out), "speeds_deg_s": list(self.speeds),
                "segment": self.seg, "segment_count": len(self.segments),
                "progress": list(self.progress), "master": self.master,
                "lead_deg": [self.dirs[i] * (out[i] - q_deg[i]) for i in range(2)],
                "lead_limit_deg": list(lead_eff), "lead_state_deg": list(self.lead),
                "stalled": stalled, "failure": self.failure, "at_end": at_end,
                "direction_margin_deg": [s.margin(v) for v in self.speeds]}

    def _update_slave_speed(self, s_master, master_speed=None, behind_deg=0.):
        """Slave field from the *local* slave/master rate ratio a few nodes ahead.

        The ratio varies along a straight push (0.06-0.64 on real routes); an
        average-based field lets the slave fall behind where it is high, while
        a needlessly high field enlarges its direction margin (= its lead).
        """
        s = self.settings
        a, b = self.segments[self.seg]
        m, sl = self.master, 1 - self.master
        k0 = min(max(int(s_master), a), b - 1)
        ratio = 0.
        for k in range(k0, min(k0 + 4, b)):
            dm = abs(self.path[k + 1][m] - self.path[k][m])
            if dm > 1e-12:
                ratio = max(ratio, abs(self.path[k + 1][sl] - self.path[k][sl]) / dm)
        # Pace the slave by the master's *average* speed over the stall window
        # (instantaneous readings are mostly 0 under stick-slip) plus a term
        # closing any gap behind the aligned position. A stopped master must
        # not let the slave creep ahead by a large direction margin.
        pace = self.speeds[m] if master_speed is None else min(self.speeds[m], master_speed)
        catch_up = s.slave_catch_up_per_s * max(0., behind_deg)
        self.speeds[sl] = min(self.speeds[m], max(s.min_speed_deg_s, 1.3 * ratio * pace + catch_up))
        self.slave_lead = max(s.slave_lead_deg, s.margin(self.speeds[sl]))

    def _corner_reached(self, axis, b, q, now):
        d = self.dirs[axis]
        short = d * (self.path[b][axis] - q)
        if short <= self.settings.corner_tolerance_deg:
            return True
        h = self.history[axis]
        held = (abs(self.targets[axis] - self.path[b][axis]) <= 1e-9 and h
                and now - h[0][0] >= self.settings.stall_window_s - 1e-9
                and abs(q - h[0][1]) < self.settings.progress_eps_deg)
        return bool(held and short <= self.settings.lead_max_deg)

    def seg_contains(self, angle, axis):
        """Whether a previous target lies inside the current segment's range."""
        a, b = self.segments[self.seg]
        lo, hi = sorted((self.path[a][axis], self.path[b][axis]))
        return lo - 1e-9 <= angle <= hi + 1e-9
