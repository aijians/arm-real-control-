"""Three speed caps for endpoint oscillation; no I/O or target changes."""
import math

from .recovery import SpeedRecovery


DEFAULTS = {
    "enabled": False,
    "speed_tiers_deg_s": [18.0, 30.0, 60.0],
    "oscillation_window_s": 1.2,
    "oscillation_amplitude_deg": 0.12,
    "oscillation_reversal_deg": 0.04,
    "oscillation_reversals": 3,
    "near_goal_limit_deg": 3.5,
}


def timed_downshift_settings(timed):
    if not isinstance(timed, dict):
        raise ValueError("invalid free_timed_motion")
    raw = timed.get("oscillation_downshift", {})
    if not isinstance(raw, dict) or set(raw) - set(DEFAULTS):
        raise ValueError("invalid free_timed_motion.oscillation_downshift")
    cfg = {**DEFAULTS, **raw}
    if type(cfg["enabled"]) is not bool:
        raise ValueError("oscillation_downshift.enabled must be bool")
    tiers = cfg["speed_tiers_deg_s"]
    if (not isinstance(tiers, (list, tuple)) or len(tiers) != 3
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in tiers)
            or not 0 < tiers[0] or tiers[-1] > 360
            or any(a >= b for a, b in zip(tiers, tiers[1:]))):
        raise ValueError("oscillation_downshift requires three increasing positive speed tiers")
    cfg["speed_tiers_deg_s"] = list(tiers)
    for key in ("oscillation_window_s", "oscillation_amplitude_deg",
                "oscillation_reversal_deg", "near_goal_limit_deg"):
        v = cfg[key]
        if type(v) not in (int, float) or not math.isfinite(v) or v <= 0:
            raise ValueError("invalid oscillation_downshift." + key)
    if (type(cfg["oscillation_reversals"]) is not int
            or not 2 <= cfg["oscillation_reversals"] <= 10
            or cfg["oscillation_reversal_deg"] >= cfg["oscillation_amplitude_deg"]
            or cfg["near_goal_limit_deg"] * 2 < cfg["oscillation_amplitude_deg"]):
        raise ValueError("inconsistent oscillation_downshift thresholds")
    return cfg


class TimedSpeedDownshift:
    """Reuse the existing reversal/amplitude/trend detector at a fixed endpoint.

    Per-axis caps survive move failures, observation retreats and task retries.
    A new MotorPair (created for each task) starts from the high tier. Pending
    changes commit only after the caller successfully writes the same goal.
    Each downshift clears history, requiring a fresh observation window before
    the next one. The policy never increases a speed or extends a deadline.
    """
    def __init__(self, settings, motors):
        self.settings = settings
        self.names = [m["name"] for m in motors]
        self.limits = [settings["speed_tiers_deg_s"][-1]] * len(motors)
        # Only the history-based detector is reused, not legacy upshift policy.
        self.detector = SpeedRecovery(settings, motors, "timed")
        self.begin_move()

    def begin_move(self):
        self.last_stamps = [-math.inf] * len(self.names)
        for history in self.detector.history:
            history.clear()

    def speed(self, name, base):
        if not self.settings["enabled"]:
            return base
        return min(base, self.limits[self.names.index(name)])

    def observe(self, state, goal, current_speeds, after):
        requests = {}
        if not self.settings["enabled"]:
            return requests
        for i, history in enumerate(self.detector.history):
            stamp = min(state["axis_stamp_monotonic_s"][i],
                        state["speed_stamp_monotonic_s"][i])
            if stamp <= max(self.last_stamps[i], after[i]):
                continue
            self.last_stamps[i] = stamp
            history.append((stamp, math.degrees(state["q_rad"][i] - goal[i])))
            while history and stamp - history[0][0] > self.settings["oscillation_window_s"]:
                history.popleft()
            lower = [v for v in self.settings["speed_tiers_deg_s"]
                     if v < min(current_speeds[i], self.limits[i]) - .005]
            if lower and self.detector.oscillating(i):
                requests[i] = lower[-1]
        return requests

    def wrote(self, axis, speed):
        if speed not in self.settings["speed_tiers_deg_s"] or speed >= self.limits[axis]:
            raise ValueError("invalid oscillation downshift commit")
        self.limits[axis] = speed
        self.detector.history[axis].clear()
