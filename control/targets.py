"""Task parameters and observed slider travel limits (shoulder-frame mm)."""
from dataclasses import dataclass
import math

COLORS = ("橙", "蓝")


def finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def normalize_task(task):
    if not isinstance(task, dict):
        raise ValueError("task must be a JSON object")
    intent, color = task.get("intent"), task.get("color")
    if intent not in ("push_distance", "push_to_end", "push_round_trip"):
        raise ValueError("unknown intent")
    if color not in COLORS:
        raise ValueError("color must be 橙 or 蓝")
    result = {"intent": intent, "color": color}
    if intent == "push_distance":
        direction, distance = task.get("direction"), task.get("distance_mm")
        if direction not in ("+x", "-x"):
            raise ValueError("push_distance requires direction +x or -x")
        if type(distance) is not int or not 10 <= distance <= 60:
            raise ValueError("distance_mm must be an integer within 10..60")
        result.update(direction=direction, distance_mm=distance)
    elif intent == "push_round_trip":
        count = task.get("round_trips")
        if type(count) is not int or not 1 <= count <= 8:
            raise ValueError("round_trips must be an integer within 1..8")
        result["round_trips"] = count
    return result


@dataclass(frozen=True)
class Target:
    color: str
    start_x_mm: float
    requested_x_mm: float
    target_x_mm: float
    y_mm: float
    feasible_x_limits_mm: tuple
    direction: str | None
    clipped: bool
    no_motion: bool


class TargetResolver:
    def __init__(self, config, slider_limits_x_mm):
        self.sliders = config["sliders"]
        self.tolerance = finite_number(config["feedback"]["slider_position_tolerance_mm"], "tolerance")
        if self.tolerance <= 0:
            raise ValueError("position tolerance must be positive")
        self.limits = {}
        for color in COLORS:
            lo, hi = slider_limits_x_mm[color]
            lo, hi = finite_number(lo, "lower limit"), finite_number(hi, "upper limit")
            if lo >= hi:
                raise ValueError("invalid slider limits")
            self.limits[color] = (lo, hi)

    def bounds(self, color):
        if color not in COLORS:
            raise ValueError("invalid color")
        return self.limits[color]

    def resolve(self, color, requested_x_mm, positions):
        lo, hi = self.bounds(color)
        start = finite_number(positions[color], "slider position")
        requested = finite_number(requested_x_mm, "requested position")
        target = min(hi, max(lo, requested))
        delta = target - start
        no_motion = abs(delta) <= self.tolerance
        if (requested - start) * delta < 0:
            target, delta, no_motion = start, 0.0, True
        return Target(color, start, requested, target, self.sliders[color]["y_mm"],
                      (lo, hi), None if no_motion else ("+x" if delta > 0 else "-x"),
                      not math.isclose(target, requested, abs_tol=1e-9), no_motion)

    def distance(self, color, direction, distance_mm, positions):
        if direction not in ("+x", "-x"):
            raise ValueError("invalid direction")
        start = finite_number(positions[color], "slider position")
        return self.resolve(color, start + (1 if direction == "+x" else -1) * distance_mm, positions)

    def endpoint(self, color, positions, direction=None):
        if direction is None:
            direction = "+x" if self.sliders[color]["end"] == "right" else "-x"
        if direction not in ("+x", "-x"):
            raise ValueError("invalid direction")
        lo, hi = self.bounds(color)
        return self.resolve(color, hi if direction == "+x" else lo, positions)
