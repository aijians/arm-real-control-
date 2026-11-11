"""Bounded reassertion of an unchanged goal after fresh, stationary feedback.

This policy cannot compensate friction or prove that a drive accepted a goal.
It only selects axes; the caller retains the original deadline and goal checks.
"""
import math


def goal_refresh_settings(timed):
    raw = timed.get("goal_refresh", {})
    if not isinstance(raw, dict):
        raise ValueError("invalid free_timed_motion.goal_refresh")
    cfg = dict(enabled=False, stationary_s=.6, progress_deg=.05, max_attempts=2)
    if set(raw) - set(cfg):
        raise ValueError("unknown free_timed_motion.goal_refresh setting")
    cfg.update(raw)
    if type(cfg["enabled"]) is not bool:
        raise ValueError("invalid goal_refresh.enabled")
    for key in ("stationary_s", "progress_deg"):
        value = cfg[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError("invalid goal_refresh." + key)
    if type(cfg["max_attempts"]) is not int or not 1 <= cfg["max_attempts"] <= 3:
        raise ValueError("invalid goal_refresh.max_attempts")
    return cfg


class StationaryGoalRefresh:
    def __init__(self, cfg, tolerance_deg, still_deg_s, count):
        self.cfg = cfg
        self.tolerance = tolerance_deg
        self.still = still_deg_s
        self.attempts = [0] * count
        self.windows = [None] * count
        self.last_stamps = [None] * count

    def observe(self, state, goal, after):
        axes = []
        for i, target in enumerate(goal):
            stamp = min(state["axis_stamp_monotonic_s"][i],
                        state["speed_stamp_monotonic_s"][i])
            q = math.degrees(state["q_rad"][i])
            error = abs(math.degrees(target - state["q_rad"][i]))
            speed = abs(math.degrees(state["speed_rad_s"][i]))
            if (not self.cfg["enabled"] or stamp < after[i]
                    or error <= self.tolerance + 1e-9 or speed > self.still + 1e-9
                    or self.attempts[i] >= self.cfg["max_attempts"]):
                self.windows[i] = None
                continue
            if self.last_stamps[i] is not None and stamp <= self.last_stamps[i]:
                continue
            self.last_stamps[i] = stamp
            window = self.windows[i]
            if window is None:
                window = [stamp, q, q]
            else:
                window[1], window[2] = min(window[1], q), max(window[2], q)
                if window[2] - window[1] >= self.cfg["progress_deg"] - 1e-9:
                    window = [stamp, q, q]
            self.windows[i] = window
            if stamp - window[0] >= self.cfg["stationary_s"] - 1e-9:
                axes.append(i)
        return axes

    def sent(self, axes):
        for i in axes:
            self.attempts[i] += 1
            self.windows[i] = None
