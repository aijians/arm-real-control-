"""Route-based command spacing. Pure planning/configuration, no hardware I/O."""
import bisect
import math


DEFAULTS = {
    "enabled": False,
    "free_speed_deg_s": 10.0,
    "free_retry_speed_deg_s": 5.0,
    "push_speed_deg_s": 60.0,
    "free_max_step_deg": 0.2,
    # Upper arm (axis 0) only, for every non-push phase (Free/withdraw/return).
    # The lower arm (axis 1) keeps free_max_step_deg.
    "free_upper_max_step_deg": 0.1,
    "push_initial_step_deg": 0.2,
    "push_step_increment_deg": 0.2,
    "push_points_per_step": 3,
    "push_max_step_deg": 1.0,
}


def segment_settings(hardware):
    raw = hardware.get("segmented_motion", {})
    if not isinstance(raw, dict) or set(raw) - set(DEFAULTS):
        raise ValueError("invalid segmented_motion settings")
    result = {**DEFAULTS, **raw}
    if type(result["enabled"]) is not bool:
        raise ValueError("segmented_motion.enabled must be bool")
    for key in DEFAULTS.keys() - {"enabled", "push_points_per_step"}:
        v = result[key]
        if type(v) not in (int, float) or not math.isfinite(v) or v <= 0:
            raise ValueError("invalid segmented_motion." + key)
    count = result["push_points_per_step"]
    if type(count) is not int or not 1 <= count <= 20:
        raise ValueError("invalid segmented_motion.push_points_per_step")
    if not (0.05 <= result["free_upper_max_step_deg"] <= result["free_max_step_deg"] <= 0.2
            and 0.05 <= result["push_initial_step_deg"] <= result["push_max_step_deg"] <= 1.2
            and result["push_step_increment_deg"] <= result["push_max_step_deg"]):
        raise ValueError("inconsistent segmented_motion angle steps")
    if (max(result["free_speed_deg_s"], result["push_speed_deg_s"]) > 360
            or (result["enabled"] and max(result["free_speed_deg_s"], result["push_speed_deg_s"])
                > hardware.get("max_speed_deg_s", 360))):
        raise ValueError("segmented_motion exceeds motor speed cap")
    if result["enabled"] and not hardware.get("min_speed_deg_s", 2.) <= result["free_retry_speed_deg_s"] <= result["free_speed_deg_s"]:
        raise ValueError("invalid segmented_motion.free_retry_speed_deg_s")
    return result


def phase_profile(settings, name, *, retry_after_oscillation=False):
    if type(retry_after_oscillation) is not bool or (retry_after_oscillation and name != "free"):
        raise ValueError("oscillation retry profile requires a free phase")
    push = name in ("push", "push_correction")
    speed = settings["push_speed_deg_s" if push else "free_speed_deg_s"]
    if retry_after_oscillation:
        speed = settings["free_retry_speed_deg_s"]
    axis_caps = ([settings["push_max_step_deg"]]*2 if push else
                 [settings["free_upper_max_step_deg"], settings["free_max_step_deg"]])
    return {"version": 1, "speed_deg_s": speed,
            "max_step_deg": max(axis_caps), "axis_max_step_deg": axis_caps,
            "mode": "progressive_push" if push else "small_step_free",
            "bias_enabled": False, "speed_shifts_enabled": False}


def axis_cap(cap, axis):
    """Per-axis step cap: Free-family caps are [upper, lower], push caps scalar."""
    return cap[axis] if isinstance(cap, (list, tuple)) else cap


def free_retry_phase(phase, settings):
    """Apply a slower, fixed Free speed between attempts, never during a move.

    Retiming preserves the planned route and the executor's velocity checks.
    This returns a new phase and is idempotent; settings remain task-independent.
    """
    if not settings["enabled"] or phase.get("phase") != "free":
        raise ValueError("slow Free retry requires segmented free motion")
    profile = phase_profile(settings, "free", retry_after_oscillation=True)
    scale = (1. if phase.get("free_retry_after_oscillation") else
             settings["free_speed_deg_s"] / profile["speed_deg_s"])
    return {**phase, "free_retry_after_oscillation": True,
            "segment_control": profile,
            "points": [{**p, "time_s": p["time_s"] * scale,
                        "q_rad": list(p["q_rad"])} for p in phase["points"]]}


def resample_route(path, distances, centres, settings, name, motors, chord_safe, gap_mm=0.0):
    """Keep the dense route's corners/reversals, bound model and wire steps.

    A longer chord must pass the planner's existing geometry checks. Geometry
    and the last remainder may shorten a scheduled step; no minimum precision
    is imposed on a secondary axis or a protected route corner.
    """
    if len(path) == 1:
        return [list(path[0])], [], [0.0], []
    mandatory = {len(path)-1}
    for axis in range(2):
        sign, last_end = 0, 0
        for index in range(1, len(path)):
            delta = path[index][axis] - path[index-1][axis]
            if abs(delta) < 1e-10:
                continue
            nxt = 1 if delta > 0 else -1
            if sign and nxt != sign:
                mandatory.update((last_end, index-1))
            sign, last_end = nxt, index
    for i in range(1, len(path)-1):
        left = [b-a for a, b in zip(centres[i-1], centres[i])]
        right = [b-a for a, b in zip(centres[i], centres[i+1])]
        scale = math.hypot(*left)*math.hypot(*right)
        if scale > 1e-12 and (abs(left[0]*right[1]-left[1]*right[0]) > 1e-5*scale
                               or sum(a*b for a, b in zip(left, right)) < 0):
            mandatory.add(i)
    boundaries = sorted(mandatory)
    cumulative = [0.0]
    for distance in distances:
        cumulative.append(cumulative[-1] + distance)

    def at(position, values):
        i = min(len(path)-2, int(position))
        f = position-i
        return [a+(b-a)*f for a, b in zip(values[i], values[i+1])]

    def distance_at(position):
        i = min(len(path)-2, int(position))
        return cumulative[i] + (cumulative[i+1]-cumulative[i])*(position-i)

    def wire(q):
        return [round((m["offset_deg"]+m["sign"]*m["ratio"]*math.degrees(v))*100)
                for m, v in zip(motors, q)]

    push = name in ("push", "push_correction")
    selected, sources, caps, lengths = [list(path[0])], [0.0], [], []
    position, postcontact_nodes = 0.0, 0
    while position < len(path)-1-1e-9:
        progress = distance_at(position)
        cap = [settings["free_upper_max_step_deg"], settings["free_max_step_deg"]]
        if push:
            level = 0 if progress < max(0.0, gap_mm)-1e-9 else postcontact_nodes//settings["push_points_per_step"]
            cap = min(settings["push_max_step_deg"], settings["push_initial_step_deg"]
                      + level*settings["push_step_increment_deg"])
        start, start_wire = selected[-1], wire(selected[-1])

        def fits(end):
            q = at(end, path)
            return (all(abs(math.degrees(b-a)) <= axis_cap(cap, i)+1e-10
                        for i, (a, b) in enumerate(zip(start, q)))
                    and all(abs(b-a) <= math.floor(axis_cap(cap, i)*m["ratio"]*100+1e-7)
                            for i, (a, b, m) in enumerate(zip(start_wire, wire(q), motors))))

        boundary = boundaries[bisect.bisect_right(boundaries, position+1e-9)]
        low, high = position, float(boundary)
        # Each axis is monotonic between protected boundaries.
        if not fits(high):
            for _ in range(45):
                mid = (low+high)/2
                if fits(mid):
                    low = mid
                else:
                    high = mid
            end = low
        else:
            end = high
        for _ in range(30):
            q = at(end, path)
            if chord_safe(start, q):
                break
            end = (end+position)/2
        else:
            raise ValueError("segmented_route_chord_not_certified")
        if end-position < 1e-8:
            raise ValueError("segmented_route_cannot_advance")
        selected.append(q)
        sources.append(end)
        caps.append(cap)
        lengths.append(distance_at(end)-progress)
        if push and progress >= max(0.0, gap_mm)-1e-9:
            postcontact_nodes += 1
        position = end
    return selected, lengths, sources, caps
