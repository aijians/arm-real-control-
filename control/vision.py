"""Continuous task video, latest-frame detection and robot-shoulder geometric coordinates."""
import json
import math
from pathlib import Path
import time

try:
    from vision_tools.detect_sliders import detect_sliders, detect_frame
    from .camera import CameraSession, CameraError
    from vision_tools.pixel_to_world import convert_detections, DEFAULT_CALIBRATION
except ImportError:
    from detect_sliders import detect_sliders, detect_frame
    from camera import CameraSession, CameraError
    from pixel_to_world import convert_detections, DEFAULT_CALIBRATION


class VisionError(RuntimeError):
    pass


def _finite(value, name):
    if isinstance(value, bool):
        raise VisionError(name + " must be finite")
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise VisionError(name + " must be finite") from exc
    if not math.isfinite(number):
        raise VisionError(name + " must be finite")
    return number


class Vision:
    """A task owns one video stream. Detection reads its latest new frame.

    Project Coordinate System (Robot Shoulder Origin):
      - Origin (0, 0, 0): Robot arm base joint center (`shoulder_global_xy_mm = [189.30, 28.50]`)
      - +X axis: Horizontal right along the slider rail (`X_model = X_global - 189.30 mm`)
      - +Y axis: Horizontal forward towards the slider rail (`Y_model = Y_global - 28.50 = 155.00 mm`)
      - +Z axis: Vertically upward perpendicular to XY plane (right-hand rule)
    """

    def __init__(self, root, config):
        self.root = Path(root).resolve()
        self.config = dict(config)
        self.video_dir = Path(config.get("video_dir", self.root / "videos"))
        if not self.video_dir.is_absolute():
            self.video_dir = self.root / self.video_dir
        self.camera = None
        self.last_sequence = 0
        self.last_video = {"status": "not_started", "file": None}

        shoulder = config.get("shoulder_global_xy_mm", [189.30, 28.50])
        if not isinstance(shoulder, (list, tuple)) or len(shoulder) != 2:
            raise VisionError("shoulder_global_xy_mm must contain X,Y")
        self.shoulder = [_finite(x, "shoulder_global_xy_mm") for x in shoulder]
        self.rail_y = _finite(config.get("rail_model_y_mm", 155.0), "rail_model_y_mm")
        self.y_tolerance = _finite(config.get("rail_y_tolerance_mm", 0.01), "rail_y_tolerance_mm")

        rail_geom = config.get("rail_geometry") or {}
        if not isinstance(rail_geom, dict):
            raise VisionError("rail_geometry must be a mapping")
        self.center_inset_mm = _finite(rail_geom.get("center_inset_mm", 9.0), "center_inset_mm")
        self.occupied_end_clearance_mm = _finite(
            rail_geom.get("occupied_end_clearance_mm", 15.5), "occupied_end_clearance_mm"
        )
        # 导轨物理总跨度与滑杆位置一致性校验参数（80 mm 导轨支架在 174.5 mm 平面实测跨度约 79~82 mm）
        self.min_rail_span_mm = _finite(rail_geom.get("min_rail_span_mm", 72.0), "min_rail_span_mm")
        self.max_rail_span_mm = _finite(rail_geom.get("max_rail_span_mm", 90.0), "max_rail_span_mm")
        self.limit_position_tolerance_mm = _finite(
            rail_geom.get("limit_position_tolerance_mm", 1.5), "limit_position_tolerance_mm"
        )
        if (self.center_inset_mm <= 0 or self.occupied_end_clearance_mm <= 0
                or self.min_rail_span_mm <= 0 or self.min_rail_span_mm >= self.max_rail_span_mm
                or self.limit_position_tolerance_mm < 0):
            raise VisionError("invalid rail_geometry parameters")

        self.timeout = _finite(config.get("capture_timeout_s", 20.0), "capture_timeout_s")
        self.max_age = _finite(config.get("max_frame_age_s", 3.0), "max_frame_age_s")
        if min(self.timeout, self.max_age) <= 0 or self.y_tolerance < 0:
            raise VisionError("invalid capture timing or Y tolerance")

        self.calibration = config.get("calibration")
        if isinstance(self.calibration, str):
            path = Path(self.calibration)
            if not path.is_absolute():
                path = self.root / path
            self.calibration = json.loads(path.read_text(encoding="utf-8"))

        cal_plane_y = _finite(
            (self.calibration or DEFAULT_CALIBRATION).get("plane_y_mm", 183.5), "plane_y_mm"
        )
        if abs((cal_plane_y - self.shoulder[1]) - self.rail_y) > self.y_tolerance:
            raise VisionError(
                f"vision_model_rail_y_mismatch: plane_y({cal_plane_y}) - shoulder_y({self.shoulder[1]}) "
                f"!= rail_model_y_mm({self.rail_y})"
            )

    def start(self, task_id):
        if self.camera is not None:
            raise VisionError("camera_task_already_started")
        if (not isinstance(task_id, str) or not task_id
                or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in task_id)):
            raise VisionError("invalid_video_task_id")
        self.last_sequence = 0
        self.camera = CameraSession(self.config, self.video_dir / (task_id + ".mp4"))
        try:
            self.camera.start()
        except Exception as exc:
            raise VisionError("camera_start_failed: " + str(exc)) from exc

    def close(self):
        if self.camera is not None:
            self.last_video = self.camera.close()
            self.camera = None
        return self.last_video

    def check_health(self):
        if self.camera is None:
            raise VisionError("camera_task_not_started")
        try:
            self.camera.check_health(self.max_age)
        except CameraError as exc:
            raise VisionError(str(exc)) from exc

    def detect_image(self, path):
        """Saved-image analysis grants no live-observation freshness."""
        return self._detect(path, detect_sliders, "offline_image")

    def _compute_slider_limits(self, world, positions):
        """由实测导轨左右端点计算按颜色区分的动态行程，并校验导轨总跨度及与两个滑杆当前位置的一致性。

        任一校验不通过时返回 None（拒绝将截断/偏斜端点交给下游生成错误 target）：
        1. 左、右端点均存在且 xl_mod < xr_mod；
        2. 导轨实测总跨度 span = xr_mod - xl_mod 在 [min_rail_span_mm, max_rail_span_mm] (72~90 mm) 范围内；
        3. 蓝、橙两滑杆动态行程满足 lo < hi；
        4. 当前帧观测到的蓝杆与橙杆位置均落在各自动态行程 [lo - tol, hi + tol] 内（防止右端点落在橙杆附近截断导轨）。
        """
        rail_ep = world.get("rail_endpoints")
        if not isinstance(rail_ep, dict) or not rail_ep.get("left") or not rail_ep.get("right"):
            return None
        try:
            xl_glob = _finite(rail_ep["left"]["world_xyz_mm"][0], "rail_left_x")
            xr_glob = _finite(rail_ep["right"]["world_xyz_mm"][0], "rail_right_x")
        except (KeyError, IndexError, TypeError, VisionError):
            return None

        xl_mod = xl_glob - self.shoulder[0]
        xr_mod = xr_glob - self.shoulder[0]
        span_mm = xr_mod - xl_mod
        if not (self.min_rail_span_mm <= span_mm <= self.max_rail_span_mm):
            return None

        inset = self.center_inset_mm
        occupied = self.occupied_end_clearance_mm
        blue_lo = xl_mod + inset
        blue_hi = xr_mod - inset - occupied
        orange_lo = xl_mod + inset + occupied
        orange_hi = xr_mod - inset
        if blue_lo >= blue_hi or orange_lo >= orange_hi:
            return None

        tol = self.limit_position_tolerance_mm
        x_blue = positions["蓝"]
        x_orange = positions["橙"]
        if x_blue >= x_orange:
            return None
        if not (blue_lo - tol <= x_blue <= blue_hi + tol):
            return None
        if not (orange_lo - tol <= x_orange <= orange_hi + tol):
            return None

        return {
            "蓝": [blue_lo, blue_hi],
            "橙": [orange_lo, orange_hi],
        }

    def _detect(self, image, detector, source):
        del source
        profile = self.config.get("profile", self.config.get("scene", "ground"))
        already_rotated = bool(self.config.get("already_rotated", False))
        try:
            pixels = detector(image, profile=profile, already_rotated=already_rotated)
            world = convert_detections(pixels, calibration=self.calibration,
                                       rail_z_mm=self.config.get("rail_z_mm"))
        except Exception as exc:
            raise VisionError("slider_detection_failed: " + str(exc)) from exc
        if (world.get("units") != "mm"
                or world.get("coordinate_frame") != "world_equals_global_z_up"
                or world.get("position_definition") not in (
                    "shaft_axis_reference_point", "vertical_shaft_axis_at_known_rail_height")):
            raise VisionError("unknown_vision_coordinate_contract")
        positions = {}
        converted = world.get("sliders", {})
        for external, color in (("yellow", "橙"), ("blue", "蓝")):
            item = converted.get(external)
            if not isinstance(item, dict):
                raise VisionError("slider_missing: " + color)
            if item.get("approximate") is True:
                raise VisionError("shaft_axis_not_observed: " + color)
            xyz = item.get("world_xyz_mm")
            if not isinstance(xyz, (list, tuple)) or len(xyz) != 3:
                raise VisionError("invalid_world_xyz_mm: " + color)
            x, y, _ = [_finite(v, "world_xyz_mm") for v in xyz]
            model_y = y - self.shoulder[1]
            if abs(model_y - self.rail_y) > self.y_tolerance:
                raise VisionError(
                    f"vision_model_rail_y_mismatch: got {model_y:.3f} mm, expected {self.rail_y:.3f} mm"
                )
            positions[color] = x - self.shoulder[0]

        if positions["蓝"] >= positions["橙"]:
            raise VisionError("slider_order_invalid: 蓝 must be left of 橙")

        slider_limits_x_mm = self._compute_slider_limits(world, positions)

        return {
            "sliders_x_mm": positions,
            "slider_limits_x_mm": slider_limits_x_mm,
            "stamp_monotonic_s": None,
            "approximate": bool(world.get("is_approximate", True)),
        }

    def read(self, after_monotonic_s=None):
        started = time.monotonic()
        after = None if after_monotonic_s is None else _finite(after_monotonic_s, "after_monotonic_s")
        if after is not None and after > started:
            raise VisionError("freshness_boundary_is_in_the_future")
        if self.camera is None:
            raise VisionError("camera_task_not_started")
        try:
            frame = self.camera.frame(self.last_sequence, after, self.timeout, self.max_age)
        except CameraError as exc:
            raise VisionError(str(exc)) from exc
        self.last_sequence = frame["sequence"]
        result = self._detect(frame["image"], detect_frame, "latest_video_frame")
        received = frame["received_monotonic_s"]
        if time.monotonic() - received > self.max_age:
            raise VisionError("camera_frame_stale_after_detection")
        result.update(stamp_monotonic_s=received)
        return result