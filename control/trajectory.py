"""Discrete execution and qualification of an already planned joint path."""
import bisect
from collections import deque
import math


class WaypointCursor:
    """Select one planned waypoint at a time using new joint feedback.

    A waypoint is a command target, not a stop: intermediate points require
    proximity only. Their per-axis tolerance is ``K * incoming segment span``,
    floored at the final position tolerance and half the wire resolution. This
    keeps a small secondary-axis segment from demanding stricter accuracy than
    the final goal. The caller selects suitably spaced shared command nodes;
    this floor is not a substitute for that selection.

    ``advance`` selects at most one successor for each feedback sample. The
    first successor can be selected immediately after validating the start;
    later successors must wait for the current segment's planned duration.
    Qualification at the final waypoint is the caller's responsibility.
    """

    def __init__(self, points, tolerance_ratio=0.5, command_interval_s=0.02, *,
                 zero_span_tolerance_deg=0.3, quantization_tolerance_deg=0.005,
                 lead_tolerance_deg=None, require_fresh_feedback=True,
                 follow_planned_time=True):
        if not points:
            raise ValueError("waypoint path is empty")
        self.num_axes = len(points[0][1])
        if self.num_axes == 0:
            raise ValueError("waypoint path has no joints")
        self.points = [(float(t), tuple(float(q) for q in values))
                       for t, values in points]
        if any(len(values) != self.num_axes for _, values in self.points):
            raise ValueError("waypoint joint counts differ")
        if any(not math.isfinite(t) or any(not math.isfinite(q) for q in values)
               for t, values in self.points):
            raise ValueError("waypoints must be finite")
        self.times = [t for t, _ in self.points]
        if any(b <= a for a, b in zip(self.times, self.times[1:])):
            raise ValueError("waypoint times must strictly increase")
        self.tolerance_ratio = float(tolerance_ratio)
        self.interval = float(command_interval_s)
        self.zero_span_tolerance_deg = float(zero_span_tolerance_deg)
        self.quantization_tolerance_deg = float(quantization_tolerance_deg)
        if not 0 < self.tolerance_ratio < 1:
            raise ValueError("waypoint tolerance ratio must be between zero and one")
        if not math.isfinite(self.interval) or self.interval <= 0:
            raise ValueError("command interval must be positive")
        if (not math.isfinite(self.zero_span_tolerance_deg)
                or self.zero_span_tolerance_deg < 0):
            raise ValueError("zero-span tolerance must be nonnegative")
        if (not math.isfinite(self.quantization_tolerance_deg)
                or self.quantization_tolerance_deg < 0):
            raise ValueError("quantization tolerance must be nonnegative")
        self.active_axes = tuple(
            max(q[axis] for _, q in self.points)
            - min(q[axis] for _, q in self.points) > 1e-12
            for axis in range(self.num_axes))
        self.index = 0
        self.selected_at = None
        self.last_feedback_stamp = None
        # Streaming: an intermediate node may be released while the arm is
        # still up to lead_tolerance_deg from the current node, except where
        # that axis reverses (a route corner must actually be reached).
        self.lead_tolerance_deg = (None if lead_tolerance_deg is None
                                   else float(lead_tolerance_deg))
        if self.lead_tolerance_deg is not None and (
                not math.isfinite(self.lead_tolerance_deg) or self.lead_tolerance_deg <= 0):
            raise ValueError("lead tolerance must be positive")
        self.require_fresh_feedback = bool(require_fresh_feedback)
        # When streaming, the A6 speed fields pace the motion; the planned
        # clock would otherwise make the arm reach each node and wait.
        self.follow_planned_time = bool(follow_planned_time)
        self.reversal_nodes = [[False]*self.num_axes for _ in self.points]
        for axis in range(self.num_axes):
            signs = [0 if abs(b[1][axis]-a[1][axis]) <= 1e-12 else (1 if b[1][axis] > a[1][axis] else -1)
                     for a, b in zip(self.points, self.points[1:])]
            for node in range(1, len(self.points)-1):
                before = next((s for s in reversed(signs[:node]) if s), 0)
                after = next((s for s in signs[node:] if s), 0)
                self.reversal_nodes[node][axis] = before * after < 0

    @property
    def elapsed(self):
        """The selected node's planned time, not a continuously running clock."""
        return self.times[self.index]

    @property
    def axis_elapsed(self):
        return [self.elapsed] * self.num_axes

    @property
    def at_endpoint(self):
        return self.index == len(self.points) - 1

    @property
    def minimum_interval_s(self):
        if self.index == 0:
            return 0.0
        if not self.follow_planned_time:
            return self.interval
        return max(self.interval, self.times[self.index] - self.times[self.index - 1])

    @property
    def tolerances_deg(self):
        if self.index == 0:
            return [self.zero_span_tolerance_deg] * self.num_axes
        previous = self.points[self.index - 1][1]
        current = self.points[self.index][1]
        return [max(self.zero_span_tolerance_deg, self.quantization_tolerance_deg,
                    self.tolerance_ratio * abs(math.degrees(b - a)))
                if abs(b - a) > 1e-12 else self.zero_span_tolerance_deg
                for a, b in zip(previous, current)]

    @property
    def advance_tolerances_deg(self):
        """Proximity required before releasing the next node."""
        base = self.tolerances_deg
        if self.lead_tolerance_deg is None or self.index == 0:
            return base
        return [tol if self.reversal_nodes[self.index][axis]
                else max(tol, self.lead_tolerance_deg)
                for axis, tol in enumerate(base)]

    def target(self):
        return list(self.points[self.index][1])

    def planned_velocities(self):
        if len(self.points) == 1:
            return [0.0] * self.num_axes
        segment = max(0, self.index - 1)
        a, b = self.points[segment], self.points[segment + 1]
        return [(qb - qa) / (b[0] - a[0]) for qa, qb in zip(a[1], b[1])]

    def planned_directions(self):
        return [1 if v > 1e-9 else (-1 if v < -1e-9 else 0)
                for v in self.planned_velocities()]

    def _feedback_stamp(self, feedback_stamps):
        if isinstance(feedback_stamps, (tuple, list)):
            if len(feedback_stamps) != self.num_axes:
                raise ValueError("feedback joint counts differ")
            selected = [float(stamp) for stamp, active in
                        zip(feedback_stamps, self.active_axes) if active]
            return min(selected or [float(stamp) for stamp in feedback_stamps])
        return float(feedback_stamps)

    def advance(self, now, actual, feedback_stamps):
        """Select the next node if timing and a new nearby sample allow it.

        The caller rejects stale/invalid feedback separately. Timestamps must
        identify actual samples, rather than the host loop's current time.
        Globally stationary axes are excluded here and monitored by the caller.
        A moving axis that pauses in this segment still has to remain nearby.
        """
        now = float(now)
        if len(actual) != self.num_axes:
            raise ValueError("feedback joint counts differ")
        if not math.isfinite(now) or any(not math.isfinite(q) for q in actual):
            raise ValueError("feedback must be finite")
        stamp = self._feedback_stamp(feedback_stamps)
        if not math.isfinite(stamp) or stamp > now + 1e-9:
            raise ValueError("feedback timestamp is invalid")
        if self.selected_at is None:
            self.selected_at = now
        if self.last_feedback_stamp is not None and stamp <= self.last_feedback_stamp:
            if self.require_fresh_feedback:
                return False
        else:
            self.last_feedback_stamp = stamp
        if self.at_endpoint:
            return False
        if now - self.selected_at < self.minimum_interval_s - 1e-9:
            return False
        if any(active and abs(math.degrees(q - a)) > tolerance + 1e-9
               for q, a, tolerance, active in
               zip(self.target(), actual, self.advance_tolerances_deg, self.active_axes)):
            return False
        self.index += 1
        self.selected_at = now
        return True


class FeedbackClock:
    """Legacy compatibility API; new motor execution uses WaypointCursor."""
    def __init__(self, points, max_lead_deg, command_interval_s, *,
                 per_axis=False):
        self.points = points
        self.times = [p[0] for p in points]
        self.num_axes = len(points[0][1])
        self.elapsed = 0.0
        self.axis_elapsed = [0.0] * self.num_axes
        self.max_lead_deg = max_lead_deg
        self.interval = command_interval_s
        self.per_axis = bool(per_axis)
        self.limited_s = 0.0
        self.limited_steps = 0

    def _resolve_times(self, elapsed=None):
        if elapsed is None:
            return self.axis_elapsed if self.per_axis else [self.elapsed] * self.num_axes
        if isinstance(elapsed, (list, tuple)):
            return list(elapsed)
        return [elapsed] * self.num_axes

    def _segment_index(self, t):
        return max(0, bisect.bisect_right(self.times, t) - 1)

    def _axis_target(self, axis, elapsed):
        if len(self.points) == 1 or elapsed >= self.times[-1]:
            return self.points[-1][1][axis]
        i = self._segment_index(elapsed)
        a, b = self.points[i], self.points[i + 1]
        fraction = max(0.0, (elapsed - a[0]) / (b[0] - a[0]))
        return a[1][axis] + fraction * (b[1][axis] - a[1][axis])

    def target(self, elapsed=None):
        times = self._resolve_times(elapsed)
        return [self._axis_target(axis, t) for axis, t in enumerate(times)]

    def planned_velocities(self, elapsed=None):
        """Return per-joint planned signed velocity (rad/s) on the active segment."""
        if len(self.points) <= 1:
            return [0.0] * self.num_axes
        times = self._resolve_times(elapsed)
        velocities = []
        for axis, t in enumerate(times):
            if t >= self.times[-1]:
                velocities.append(0.0)
                continue
            i = self._segment_index(t)
            a, b = self.points[i], self.points[i + 1]
            dt = b[0] - a[0]
            velocities.append((b[1][axis] - a[1][axis]) / dt if dt > 1e-9 else 0.0)
        return velocities

    def planned_directions(self, elapsed=None):
        """Return per-joint planned motion sign (-1, 0, +1) on the active segment."""
        return [1 if v > 1e-9 else (-1 if v < -1e-9 else 0)
                for v in self.planned_velocities(elapsed)]

    def _search_advance(self, start, end, inside_fn):
        if not inside_fn(start):
            return start
        lo = start
        boundaries = self.times[bisect.bisect_right(self.times, start):
                                bisect.bisect_left(self.times, end)] + [end]
        for hi in boundaries:
            if inside_fn(hi):
                lo = hi
                continue
            for _ in range(28):
                mid = (lo + hi) / 2
                if inside_fn(mid):
                    lo = mid
                else:
                    hi = mid
            break
        return lo

    def _advance_axis(self, axis, start, dt, actual_q):
        end = min(self.times[-1], start + dt)
        return self._search_advance(
            start, end,
            lambda t: abs(math.degrees(self._axis_target(axis, t) - actual_q)) <= self.max_lead_deg
        )

    def advance(self, dt, actual):
        # Never catch up a delayed host loop in a single large reference jump.
        dt = min(max(0.0, dt), self.interval)
        if self.per_axis:
            prev_axes = list(self.axis_elapsed)
            for axis, a in enumerate(actual):
                self.axis_elapsed[axis] = self._advance_axis(
                    axis, self.axis_elapsed[axis], dt, a)
            self.elapsed = min(self.axis_elapsed)
            shortfalls = [
                dt - (new_t - old_t)
                for old_t, new_t in zip(prev_axes, self.axis_elapsed)
                if old_t < self.times[-1] and new_t - old_t < dt - 1e-8
            ]
            if shortfalls:
                self.limited_steps += 1
                self.limited_s += max(0.0, min(dt, max(shortfalls)))
            return self.target()

        start = self.elapsed
        end = min(self.times[-1], start + dt)
        # Check each knot crossed, including reversals. A later in-range
        # endpoint must not allow crossing an out-of-range intermediate knot.
        end = self._search_advance(
            start, end,
            lambda t: all(abs(math.degrees(q - a)) <= self.max_lead_deg
                          for q, a in zip(self.target(t), actual))
        )
        self.elapsed = end
        self.axis_elapsed = [end] * self.num_axes
        if end - start < dt - 1e-8 and start < self.times[-1]:
            self.limited_steps += 1
            self.limited_s += dt - (end - start)
        return self.target()


class GoalQualification:
    """Require a continuous interval of new, strictly qualifying feedback.

    An isolated bad sample resets qualification; it does not itself stop a
    motor. Stale feedback, limits, tracking guards and deadlines are separate.
    """
    def __init__(self, tolerance_deg, still_deg_s, stable_s, arrival_s=0.3,
                 *, hard_error_deg=None, hard_speed_deg_s=None,
                 unstable_window_s=0.5, unstable_samples=3, active_axes=None,
                 steady_angle_deg=0.05, tolerate_speed_blip=True):
        self.tolerance = tolerance_deg
        self.still = still_deg_s
        self.required = stable_s
        self.arrival_s = arrival_s
        self.hard_error_deg = (max(1.0, 2.0 * tolerance_deg)
                               if hard_error_deg is None else hard_error_deg)
        self.hard_speed_deg_s = (max(10.0, 5.0 * still_deg_s)
                                 if hard_speed_deg_s is None else hard_speed_deg_s)
        self.unstable_window_s = unstable_window_s
        self.unstable_samples = int(unstable_samples)
        self.armed = False
        self.since = None
        self.last_stamp = None
        self.stable_s = 0.0
        self.resets = 0
        self.violation_since = None
        self.violation_samples = 0
        self.failure = None
        self.active_axes = None if active_axes is None else tuple(bool(a) for a in active_axes)
        # An isolated integer speed reading (e.g. 0 -> 2 -> 0 deg/s) while
        # the angles stay in tolerance and steady does not discard the
        # accumulated stable time; a second distinct over-speed speed sample,
        # an angle shift or an error outside tolerance still resets it.
        self.steady_angle_deg = float(steady_angle_deg)
        self.tolerate_speed_blip = bool(tolerate_speed_blip)
        self.reference_q = None          # angles of the last qualifying sample
        self.blip_speed_stamps = {}      # axis -> speed stamp of a tolerated over-speed sample
        self.blips_tolerated = 0
        self.events = deque(maxlen=128)  # every reset / tolerated blip, with its sample

    def _event(self, kind, reason, stamp, errors, speeds, changes):
        self.events.append({"kind": kind, "reason": reason, "stamp_monotonic_s": stamp,
                            "errors_deg": errors, "speeds_deg_s": speeds,
                            "angle_change_deg": changes, "armed": self.armed,
                            "stable_s_before": self.stable_s})

    def reset_after_command(self):
        """A new actual speed command invalidates earlier stability evidence."""
        if self.since is not None or self.armed:
            self._event("reset", "new_command", self.last_stamp, None, None, None)
        self.reference_q = None
        self.blip_speed_stamps = {}
        self.armed = False
        self.since = self.last_stamp = None
        self.stable_s = 0.
        self.violation_since = None
        self.violation_samples = 0
        self.failure = None
        self.resets += 1

    def observe(self, state, goal, at_endpoint):
        stamp = state['stamp_monotonic_s']
        active = (tuple(True for _ in goal) if self.active_axes is None
                  else self.active_axes)
        if (len(active) != len(goal) or len(state['q_rad']) != len(goal)
                or len(state['speed_rad_s']) != len(goal)):
            raise ValueError("goal qualification joint counts differ")
        axis_stamps = state.get('axis_stamp_monotonic_s')
        if axis_stamps is not None:
            if len(axis_stamps) != len(goal):
                raise ValueError("feedback timestamp joint counts differ")
            stamp = min((s for s, enabled in zip(axis_stamps, active) if enabled),
                        default=stamp)
        fresh = self.last_stamp is None or stamp > self.last_stamp
        errors = [abs(math.degrees(a - b))
                  for a, b, enabled in zip(state['q_rad'], goal, active) if enabled]
        speeds = [abs(math.degrees(v))
                  for v, enabled in zip(state['speed_rad_s'], active) if enabled]
        errors_ok = at_endpoint and all(e <= self.tolerance + 1e-9 for e in errors)
        qualifies = errors_ok and all(v <= self.still + 1e-9 for v in speeds)
        indices = [i for i, enabled in enumerate(active) if enabled]
        q_deg = [math.degrees(state['q_rad'][i]) for i in indices]
        changes = (None if self.reference_q is None else
                   [abs(a - b) for a, b in zip(q_deg, self.reference_q)])
        speed_stamps = state.get('speed_stamp_monotonic_s')
        if not qualifies and self.since is not None and self.tolerate_speed_blip:
            over = [i for i, v in zip(indices, speeds) if v > self.still + 1e-9]
            steady = changes is not None and all(c < self.steady_angle_deg for c in changes)
            blip = (errors_ok and steady and over and speed_stamps is not None
                    and all(v < self.hard_speed_deg_s - 1e-9 for v in speeds))
            if blip:
                repeated = [i for i in over if i in self.blip_speed_stamps
                            and speed_stamps[i] != self.blip_speed_stamps[i]]
                if not repeated:
                    new = [i for i in over if i not in self.blip_speed_stamps]
                    for i in new:
                        self.blip_speed_stamps[i] = speed_stamps[i]
                    if new:
                        self.blips_tolerated += 1
                        self._event("tolerated_speed_blip", "isolated_speed_sample", stamp,
                                    errors, speeds, changes)
                    if fresh:
                        self.last_stamp = stamp
                    return False
                reason = "consecutive_overspeed"
            elif not errors_ok:
                reason = "angle_error_out_of_tolerance" if at_endpoint else "not_at_endpoint"
            elif not steady:
                reason = "angle_shift" if changes is not None else "overspeed_without_reference"
            else:
                reason = "overspeed"
            self._event("reset", reason, stamp, errors, speeds, changes)
        elif not qualifies and self.since is not None:
            self._event("reset", "angle_error_out_of_tolerance" if not errors_ok else "overspeed",
                        stamp, errors, speeds, changes)
        if qualifies:
            # Speeds are back at rest: a later over-speed sample is isolated again.
            self.blip_speed_stamps = {}
            self.reference_q = q_deg
        # Evaluate even when only one of four feedback stamps has changed.
        # This prevents a fast speed reply from being hidden by an older angle.
        if not qualifies:
            self.reference_q = None
            self.blip_speed_stamps = {}
            self.resets += self.since is not None
            self.since = None
            self.stable_s = 0.0
            if self.armed:
                if (max(errors, default=0.0) >= self.hard_error_deg - 1e-9
                        or max(speeds, default=0.0) >= self.hard_speed_deg_s - 1e-9):
                    self.failure = 'motion_goal_hold_hard_limit'
                if fresh:
                    if self.violation_since is None:
                        self.violation_since = stamp
                    self.violation_samples += 1
                    if (stamp - self.violation_since >= self.unstable_window_s - 1e-9
                            and self.violation_samples >= self.unstable_samples):
                        self.failure = 'motion_goal_hold_unstable'
        else:
            self.violation_since = None
            self.violation_samples = 0
        if not fresh:
            return False
        self.last_stamp = stamp
        if qualifies:
            if self.since is None:
                self.since = stamp
            self.stable_s = stamp - self.since
            if not self.armed and self.stable_s >= self.arrival_s - 1e-9:
                self.armed = True
                self.since = stamp
                self.stable_s = 0.0
        return self.armed and qualifies and self.stable_s >= self.required - 1e-9
