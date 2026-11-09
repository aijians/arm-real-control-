"""One fixed configuration file; paths are relative to the application root."""
import json
import math
import os
from pathlib import Path
import tempfile

from .recovery import recovery_settings
from .motion_policy import motion_policy_settings
from .segments import segment_settings


def push_correction_settings(config):
    settings = {"max_steps": 3, "step_mm": 1.0, "max_distance_mm": 3.0,
                "speed_mm_s": 1.0, "min_progress_mm": 0.2,
                **config.get("push_correction", {})}
    if type(settings["max_steps"]) is not int or not 0 <= settings["max_steps"] <= 3:
        raise ValueError("push_correction.max_steps must be an integer in 0..3")
    for name, upper in (("step_mm", 1.), ("max_distance_mm", 3.),
                        ("speed_mm_s", 1.), ("min_progress_mm", 1.)):
        value = settings[name]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= upper:
            raise ValueError("invalid push_correction." + name)
    return settings


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate_json_key: " + key)
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError("nonfinite_json: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                      object_pairs_hook=pairs, parse_constant=nonfinite)


def resolve(root, value):
    path = Path(value)
    return (path if path.is_absolute() else Path(root) / path).resolve()


def sync_directory(directory):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save_json(result, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        sync_directory(path.parent)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def validate_config(config):
    segment_settings(config.get("hardware", {}))
    recovery_settings(config.get("hardware", {}))
    motion_policy_settings(config.get("hardware", {}))
    push_correction_settings(config)
    if type(config["retry_limit"]) is not int or not 0 <= config["retry_limit"] <= 3:
        raise ValueError("retry_limit must be an integer in 0..3")
    verify = config["verification"]
    if type(verify["samples"]) is not int or not 1 <= verify["samples"] <= 10:
        raise ValueError("verification.samples must be an integer in 1..10")
    for name, value in [("poll_interval_s", config["poll_interval_s"]),
                        ("tolerance_mm", verify["tolerance_mm"]), ("interval_s", verify["interval_s"])]:
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("invalid " + name)
    for color in ("橙", "蓝"):
        slider = config["sliders"][color]
        if type(slider["y_mm"]) not in (int, float) or not math.isfinite(slider["y_mm"]):
            raise ValueError("invalid slider configuration")
        if slider.get("end") not in ("left", "right"):
            raise ValueError("slider.end must be left or right")


def _apply_effective_motor_limits(config):
    """Derive motor command limits in memory from freshly loaded model limits.

    motor_deg = offset_deg + sign * ratio * model_deg. Installation offsets
    calibrate this mapping only; geometry and scene limits stay unchanged.
    robot.json keeps the physical model limits, so reloading a new offset
    always starts from the same limits rather than accumulating corrections.
    """
    for motor in config["hardware"]["motors"]:
        limits = motor["limits_deg"]
        if not isinstance(limits, (list, tuple)) or len(limits) != 2:
            raise ValueError("invalid motor limits_deg")
        offset, sign, ratio = motor["offset_deg"], motor["sign"], motor["ratio"]
        if any(type(v) not in (int, float) or not math.isfinite(v)
               for v in (offset, sign, ratio, *limits)):
            raise ValueError("nonfinite or nonnumeric motor angle mapping")
        if sign not in (-1, 1) or ratio <= 0 or limits[0] >= limits[1]:
            raise ValueError("invalid motor angle mapping or limits")
        scale = sign * ratio
        protocol = sorted(((0.0 - offset) / scale,
                           (359.99 - offset) / scale))
        lower, upper = max(limits[0], protocol[0]), min(limits[1], protocol[1])
        if lower >= upper:
            raise ValueError("no A6-representable joint range: " + motor["name"])
        # Inverse mapping can round one ulp outside A6's closed interval.
        # Move only such endpoints inward, never enlarge physical limits.
        effective = [lower, upper]
        for i, toward in ((0, upper), (1, lower)):
            for _ in range(8):
                mapped = offset + scale * effective[i]
                if 0.0 <= mapped <= 359.99:
                    break
                effective[i] = math.nextafter(effective[i], toward)
            else:
                raise ValueError("unrepresentable motor limit: " + motor["name"])
        if effective[0] >= effective[1]:
            raise ValueError("empty effective motor limits: " + motor["name"])
        motor["limits_deg"] = effective


def load_config(root):
    root = Path(root).resolve()
    config_file = root / "config/robot.json"
    config = read_json(config_file)
    validate_config(config)
    names = ("task_file", "result_file", "runtime_dir", "model_package", "scene", "video_dir", "python")
    for name in names:
        if not isinstance(config["paths"].get(name), str) or not config["paths"][name].strip():
            raise ValueError("missing path: " + name)
    paths = {name: resolve(root, config["paths"][name]) for name in names}
    task, result = paths["task_file"], paths["result_file"]
    if (task.is_relative_to(result) or result.is_relative_to(task)
            or any(paths[d].is_relative_to(p) for d in ("runtime_dir", "video_dir") for p in (task, result))
            or task.name == "task.lock" or result.name == "task.lock"):
        raise ValueError("task/result paths must be distinct from one another and the lock")
    if (result.parent == task.parent and result.name.startswith("." + task.name + ".")
            and result.name.endswith(".processing")):
        raise ValueError("result_file overlaps reserved processing filenames")
    protected_files = {config_file, root / "main.py", root / "run.sh", paths["scene"], paths["python"]}
    for output in (task, result):
        resource_directories = [paths["model_package"], paths["video_dir"]] + [root / p for p in ("control", "vision_tools", "config")]
        if output in protected_files or any(output.is_relative_to(p) for p in resource_directories):
            raise ValueError("task/result path overlaps application resources")
        if output.exists() and output.is_dir():
            raise ValueError("task/result path must be a file")
    _apply_effective_motor_limits(config)
    return config, paths
