"""Latest video frame: slider positions and limits from observed rail ends."""
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from vision_tools.detect_sliders import detect_frame
from .camera import CameraSession, CameraError
from vision_tools.pixel_to_world import convert_detections, pixel_to_world, DEFAULT_CALIBRATION


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

    Receipt times are monotonic arrival times on this boot, not calibrated
    exposure timestamps. Starting and stopping are explicit task boundaries.
    """

    def __init__(self, root, config):
        self.root = Path(root).resolve()
        self.config = dict(config)
        self.video_dir = Path(config.get("video_dir", self.root / "videos"))
        if not self.video_dir.is_absolute():
            self.video_dir = self.root / self.video_dir
        self.camera, self.last_sequence = None, 0
        self.last_video = {"status": "not_started", "file": None}
        self.scene = config.get("scene", config.get("profile", "ground"))
        if self.scene not in ("ground", "real"):
            raise VisionError("scene_must_be_ground_or_real")
        self.already_rotated = config.get("already_rotated", False)
        if not isinstance(self.already_rotated, bool):
            raise VisionError("already_rotated_must_be_bool")
        self.calibration = config.get("calibration")
        if isinstance(self.calibration, str):
            path = Path(self.calibration)
            if not path.is_absolute():
                path = self.root / path
            self.calibration = json.loads(path.read_text(encoding="utf-8"))
        base = DEFAULT_CALIBRATION if self.calibration is None else self.calibration
        if not isinstance(base, dict):
            raise VisionError("invalid_calibration")
        shoulder = config.get("shoulder_global_xy_mm", base.get("shoulder_global_xyz_mm", [189.3, 28.5, -67.])[:2])
        if not isinstance(shoulder, (list, tuple)) or len(shoulder) != 2:
            raise VisionError("shoulder_global_xy_mm must contain X,Y")
        self.shoulder = [_finite(x, "shoulder_global_xy_mm") for x in shoulder]
        geometry = config.get("rail_geometry", {})
        self.center_inset = _finite(geometry.get("center_inset_mm", 9.), "center_inset_mm")
        self.occupied_end_clearance = _finite(geometry.get("occupied_end_clearance_mm", 15.5), "occupied_end_clearance_mm")
        if min(self.center_inset, self.occupied_end_clearance) < 0:
            raise VisionError("invalid_rail_geometry")
        self.rail_y = _finite(config.get("rail_model_y_mm", 155.), "rail_model_y_mm")
        self.y_tolerance = _finite(config.get("rail_y_tolerance_mm", .01), "rail_y_tolerance_mm")
        self.timeout = _finite(config.get("capture_timeout_s", 20.), "capture_timeout_s")
        self.max_age = _finite(config.get("max_frame_age_s", 3.), "max_frame_age_s")
        if min(self.timeout, self.max_age) <= 0 or self.y_tolerance < 0:
            raise VisionError("invalid capture timing or Y tolerance")
        plane_y = _finite(base.get("plane_y_mm"), "plane_y_mm")
        if abs(plane_y - self.shoulder[1] - self.rail_y) > self.y_tolerance:
            raise VisionError("vision_model_rail_y_mismatch")

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
        try:
            encoded = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
            image = (cv2.imdecode(encoded, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                     if encoded.size else None)
            if image is None or image.size == 0:
                raise ValueError("invalid_or_empty_image")
        except Exception as exc:
            raise VisionError("slider_detection_failed: " + str(exc)) from exc
        return self._detect(image, detect_frame, "offline_image")

    def _detect(self, image, detector, source):
        try:
            pixels = detector(image, profile=self.scene, already_rotated=self.already_rotated)
            if not isinstance(pixels, dict) or pixels.get("profile") != self.scene:
                raise ValueError("detector_scene_mismatch")
            world = convert_detections(pixels, calibration=self.calibration,
                                       rail_z_mm=self.config.get("rail_z_mm"))
            endpoints = pixels.get("rail_endpoints_px")
            rail_world = None
            if endpoints is not None:
                if not isinstance(endpoints, (list, tuple)) or len(endpoints) != 2:
                    raise ValueError("invalid_rail_endpoints")
                # Both observed lip endpoints belong to the same measured front plane.
                c = world["calibration"]
                edge_calibration = {**c, "plane_y_mm": c["rail_lip_y_mm"]}
                rail_world = [pixel_to_world(p, pixels["input_size_px"], edge_calibration)
                              for p in endpoints]
        except Exception as exc:
            raise VisionError("slider_detection_failed: " + str(exc)) from exc
        if (world.get("units") != "mm"
                or world.get("coordinate_frame") != "world_equals_global_z_up"
                or world.get("position_definition") not in (
                    "shaft_axis_reference_point", "vertical_shaft_axis_at_known_rail_height")):
            raise VisionError("unknown_vision_coordinate_contract")
        positions = {}
        for external, color in (("yellow", "橙"), ("blue", "蓝")):
            item = world.get("sliders", {}).get(external)
            if not isinstance(item, dict):
                raise VisionError("slider_missing: " + color)
            if item.get("approximate") is not False:
                raise VisionError("shaft_axis_not_observed: " + color)
            xyz = item.get("world_xyz_mm")
            if not isinstance(xyz, (list, tuple)) or len(xyz) != 3:
                raise VisionError("invalid_world_xyz_mm: " + color)
            x, y, _ = [_finite(v, "world_xyz_mm") for v in xyz]
            if abs(y - self.shoulder[1] - self.rail_y) > self.y_tolerance:
                raise VisionError("vision_model_rail_y_mismatch")
            positions[color] = x - self.shoulder[0]
        limits = None
        if rail_world is not None:
            left, right = sorted(_finite(p["world_xyz_mm"][0], "rail_endpoint_x_mm") - self.shoulder[0]
                                 for p in rail_world)
            left, right = left + self.center_inset, right - self.center_inset
            # Physical carriage dimensions; no fitted position correction or endpoint snapping.
            limits = {"蓝": [left, right - self.occupied_end_clearance],
                      "橙": [left + self.occupied_end_clearance, right]}
            if any(lo >= hi for lo, hi in limits.values()):
                raise VisionError("rail_too_short_for_center_insets")
        return {"sliders_x_mm": positions, "slider_limits_x_mm": limits,
                "stamp_monotonic_s": None, "approximate": bool(world.get("is_approximate", True))}

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
