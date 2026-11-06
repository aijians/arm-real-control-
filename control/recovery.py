"""Bounded discrete speed recovery. No serial, process or filesystem access."""
from collections import deque
import math


DEFAULTS = {
    "enabled": False,
    "axes": ["upper"],
    "phases": ["free"],
    "directions": [-1],
    "speed_steps_deg_s": [18.0, 30.0],
    "trigger_s": 2.0,
    "observe_s": 2.0,
    "total_recovery_s": 15.0,
    "max_upshifts": 2,
    "max_no_gain_windows": 2,
    "settling_enabled": True,
    "settling_speed_deg_s": 18.0,
    "settling_safe_distance_deg": 0.10,
    "oscillation_window_s": 1.2,
    "oscillation_amplitude_deg": 0.12,
    "oscillation_reversal_deg": 0.04,
    "oscillation_reversals": 3,
    "near_goal_limit_deg": 1.0,
}


def recovery_settings(hardware):
    raw = hardware.get("speed_recovery", {})
    if not isinstance(raw, dict) or set(raw) - set(DEFAULTS):
        raise ValueError("invalid speed_recovery settings")
    settings = {**DEFAULTS, **raw, "progress_deg": hardware.get("progress_deg", 0.05)}
    for key in ("enabled", "settling_enabled"):
        if type(settings[key]) is not bool:
            raise ValueError("speed_recovery." + key + " must be bool")
    for key in ("axes", "phases", "directions"):
        values = settings[key]
        valid = {"upper", "lower"} if key == "axes" else ({"free"} if key == "phases" else {-1, 1})
        if (not isinstance(values, (list, tuple)) or not values
                or any(type(x) is not (int if key == "directions" else str)
                       or x not in valid for x in values)
                or len(set(values)) != len(values)):
            raise ValueError("invalid speed_recovery." + key)
    for key in ("trigger_s", "observe_s", "total_recovery_s", "progress_deg",
                "settling_speed_deg_s", "settling_safe_distance_deg", "oscillation_window_s",
                "oscillation_amplitude_deg", "oscillation_reversal_deg", "near_goal_limit_deg"):
        value = settings[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("invalid speed_recovery." + key)
    for key, lo, hi in (("max_upshifts", 1, 4), ("max_no_gain_windows", 1, 3),
                        ("oscillation_reversals", 2, 10)):
        value = settings[key]
        if type(value) is not int or not lo <= value <= hi:
            raise ValueError("invalid speed_recovery." + key)
    steps = settings["speed_steps_deg_s"]
    if (not isinstance(steps, (list, tuple)) or not 2 <= len(steps) <= 4
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in steps)
            or not 0 < steps[0] or steps[-1] > 60
            or any(a >= b for a, b in zip(steps, steps[1:]))):
        raise ValueError("speed_recovery requires 2..4 increasing speed steps at most 60 deg/s")
    if (settings["max_upshifts"] > len(steps)
            or not max(settings["trigger_s"], settings["observe_s"]) <= settings["total_recovery_s"] <= 30
            or not hardware.get("min_speed_deg_s", 2) <= settings["settling_speed_deg_s"] <= steps[-1]
            or settings["oscillation_reversal_deg"] >= settings["oscillation_amplitude_deg"]):
        raise ValueError("inconsistent speed_recovery limits")
    return settings


class SpeedRecovery:
    """Floors survive nodes, reset at a true per-axis direction reversal.

    Tier meanings: 0 formula; positive tiers index configured speed floors;
    -1 terminal settling. Pending changes commit only after a real write.
    The shared absolute recovery deadline and action counters never reset
    when a node advances, an axis reverses, or a small movement is observed.
    """
    def __init__(self, settings, motors, phase):
        self.settings, self.motors, self.phase = settings, motors, phase
        n = len(motors)
        self.axis_min_speed_deg_s = [0.0] * n
        self.speed_tier = [0] * n
        self.directions = [0] * n
        self.pending = [None] * n
        self.upshifts = [0] * n
        self.downshifted = [False] * n
        self.recovering = [False] * n
        self.evaluation_at = [-math.inf] * n
        self.no_gain_windows = [0] * n
        self.started_at = self.deadline = None
        self.events = deque(maxlen=128)
        self.history = [deque() for _ in motors]
        self.deferred = [None] * n

    def event(self, now, event, axis=None, **values):
        self.events.append({"at_monotonic_s": now, "event": event,
                            "motor": None if axis is None else self.motors[axis]["name"], **values})

    def eligible(self, axis):
        s = self.settings
        return (s["enabled"] and self.phase in s["phases"]
                and self.motors[axis]["name"] in s["axes"]
                and self.directions[axis] in s["directions"])

    def cancel_pending(self, axis, now, reason):
        request = self.pending[axis]
        if request is not None:
            self.speed_tier[axis], self.axis_min_speed_deg_s[axis] = request["previous"]
            self.pending[axis] = None
            self.event(now, "speed_change_cancelled", axis, reason=reason)

    def node_changed(self, now):
        for axis in range(len(self.motors)):
            self.cancel_pending(axis, now, "node_advanced_before_write")
            self.recovering[axis] = False
            self.no_gain_windows[axis] = 0
            self.history[axis].clear()

    def select_directions(self, directions, now):
        for axis, sign in enumerate(directions):
            if not sign:
                continue  # A plateau does not start a new monotonic segment.
            if self.directions[axis] and sign != self.directions[axis]:
                self.cancel_pending(axis, now, "planned_direction_reversed")
                self.axis_min_speed_deg_s[axis] = 0.0
                self.speed_tier[axis] = 0
                self.recovering[axis] = False
                self.history[axis].clear()
                self.event(now, "monotonic_segment_reversed", axis, direction=sign)
            self.directions[axis] = sign

    def _begin(self, now):
        if self.started_at is None:
            self.started_at = now
            self.deadline = now + self.settings["total_recovery_s"]

    def _request(self, axis, now, tier, floor, kind):
        self._begin(now)
        self.pending[axis] = {"tier": tier, "at": now, "kind": kind,
                              "previous": (self.speed_tier[axis], self.axis_min_speed_deg_s[axis])}
        self.speed_tier[axis] = tier
        self.axis_min_speed_deg_s[axis] = floor
        self.evaluation_at[axis] = now
        self.recovering[axis] = True
        self.event(now, "speed_change_requested", axis, speed_tier=tier,
                   axis_min_speed_deg_s=floor, kind=kind)

    def wrote(self, axis, tier, speed, at):
        request = self.pending[axis]
        if request is None or tier != request["tier"] or at < request["at"]:
            return False
        if request["kind"] == "up":
            self.upshifts[axis] += 1
        else:
            self.downshifted[axis] = True
        self.pending[axis] = None
        self.evaluation_at[axis] = at
        self.event(at, "speed_change_written", axis, speed_tier=tier, speed_deg_s=speed)
        return True

    def progress(self, axis, in_band):
        self.no_gain_windows[axis] = 0
        if in_band:
            self.recovering[axis] = False

    def stall(self, axis, now, last_speed):
        """Return legacy/wait/requested/exhausted; no serial side effects."""
        if not self.eligible(axis) or self.speed_tier[axis] == -1:
            return "legacy"
        if self.deadline is not None and now >= self.deadline:
            return "exhausted"
        if self.pending[axis] is not None:
            return "wait"
        if now - self.evaluation_at[axis] < self.settings["observe_s"]:
            return "wait"
        if self.upshifts[axis] < self.settings["max_upshifts"]:
            for tier, floor in enumerate(self.settings["speed_steps_deg_s"], 1):
                if floor > max(last_speed, self.axis_min_speed_deg_s[axis]) + .005:
                    self._request(axis, now, tier, floor, "up")
                    return "requested"
        self._begin(now)
        self.recovering[axis] = True
        self.evaluation_at[axis] = now
        self.no_gain_windows[axis] += 1
        self.event(now, "speed_ceiling_no_gain", axis, windows=self.no_gain_windows[axis])
        return ("exhausted" if self.no_gain_windows[axis] >= self.settings["max_no_gain_windows"]
                else "wait")

    def observe_goal(self, state, goal):
        for axis, history in enumerate(self.history):
            stamp = state["axis_stamp_monotonic_s"][axis]
            if history and stamp <= history[-1][0]:
                continue
            history.append((stamp, math.degrees(state["q_rad"][axis] - goal[axis])))
            while history and stamp - history[0][0] > self.settings["oscillation_window_s"]:
                history.popleft()

    def oscillating(self, axis):
        points = list(self.history[axis])
        s = self.settings
        if len(points) < 6 or points[-1][0] - points[0][0] < .8 * s["oscillation_window_s"]:
            return False
        errors = [value for _, value in points]
        amplitude = max(errors) - min(errors)
        mean_time = sum(t for t, _ in points) / len(points)
        mean_error = sum(errors) / len(errors)
        variance = sum((t - mean_time) ** 2 for t, _ in points)
        slope = (sum((t - mean_time) * (error - mean_error)
                     for t, error in points) / variance) if variance else 0.
        trend = slope * (points[-1][0] - points[0][0])
        if (max(abs(value) for value in errors) > s["near_goal_limit_deg"]
                or amplitude < s["oscillation_amplitude_deg"]
                or abs(trend) > .6 * amplitude):
            return False
        pivot, direction, reversals = errors[0], 0, 0
        for value in errors[1:]:
            delta = value - pivot
            if abs(delta) < s["oscillation_reversal_deg"]:
                continue
            sign = 1 if delta > 0 else -1
            reversals += bool(direction and sign != direction)
            direction, pivot = sign, value
        return reversals >= s["oscillation_reversals"]

    def request_settling(self, axis, now, last_speed):
        if (not self.eligible(axis) or not self.settings["settling_enabled"]
                or self.downshifted[axis] or self.pending[axis] is not None
                or not self.oscillating(axis)):
            return False
        speed = self.settings["settling_speed_deg_s"]
        if last_speed <= speed + .005:
            if self.deferred[axis] != "already_below_settling_speed":
                self.deferred[axis] = "already_below_settling_speed"
                self.event(now, "settling_change_deferred", axis, reason=self.deferred[axis])
            return False
        if self.deadline is not None and now >= self.deadline:
            return False
        self._request(axis, now, -1, speed, "settle")
        return True

    def profiles(self, base_cap):
        caps = [max(base_cap, floor) if tier > 0 else
                (self.settings["settling_speed_deg_s"] if tier == -1 else base_cap)
                for floor, tier in zip(self.axis_min_speed_deg_s, self.speed_tier)]
        return list(self.axis_min_speed_deg_s), list(self.speed_tier), caps

    def expired(self, now):
        return (self.deadline is not None and now >= self.deadline
                and any(self.recovering[axis] or self.pending[axis] is not None
                        for axis in range(len(self.motors))))

    def snapshot(self):
        return {"axis_min_speed_deg_s": list(self.axis_min_speed_deg_s),
                "speed_tier": list(self.speed_tier), "monotonic_directions": list(self.directions),
                "upshifts": list(self.upshifts), "downshifted": list(self.downshifted),
                "recovery_started_at": self.started_at, "recovery_deadline": self.deadline,
                "pending_speed_change": [None if r is None else dict(r) for r in self.pending],
                "events": list(self.events)}
