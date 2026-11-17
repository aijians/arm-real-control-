#!/usr/bin/env python3
"""将滑杆轴线像素与导轨真实端点像素转换到世界坐标系，单位为毫米。

坐标系说明：
1. 历史默认箱体坐标系 (world_equals_global_z_up)：
   - 原点：箱体左下内角；+X 水平向右，+Y 水平指向滑杆导轨墙面，+Z 竖直向上（右手法则）。
   - 历史默认标定值（新测世界系通过完整 calibration 显式提供）：
     * ground 与 real 相机名义光心位置 C_world_mm = [24.5, 99.0, -108.0] mm
     * real（及重装后倒装 ground）原图物理倒装 180°；检测器先转正，坐标转换使用转正视图的 K、D、R
     * 导轨滑杆轴心平面 plane_y_mm = 183.5 mm（底座 Y=28.5 mm + 155.0 mm）
     * 导轨前挡边表面平面 rail_lip_y_mm = 174.5 mm（左、右导轨端点统一按同一前挡边平面求交）
     * 当检测结果包含实测拟合导轨线 rail_line (source='fitted') 时，自动根据导轨透视斜率与高度区间
       按现有分支补偿安装姿态角（Roll/Yaw）；光心保持用户指定的 [24.5, 99.0, -108.0] mm。
2. 完整显式标定可声明 world_handedness='left' 及独立的 world_coordinate_frame。
   此时 R_world_from_camera 是 det=-1 的正交坐标基变换，不能称为物理旋转。
   显式标定不叠加历史导轨姿态补偿；180°倒装由检测器统一转正一次。
"""

import argparse
import copy
import json
import math
from pathlib import Path
import sys

import numpy as np


DEFAULT_CALIBRATION = {
    "name": "imx415_calibrated_intrinsics_and_measured_box_geometry_1080p",
    "image_size_px": [1920, 1080],
    "K": [[698.5891449464962, 0.0, 960.8663237262527],
          [0.0, 698.2399519207177, 567.6531461824138],
          [0.0, 0.0, 1.0]],
    "D": [0.05322918059115373, -0.06375883808099647,
          0.00008002273224807077, 0.0011399378118325647,
          0.015484807643491312],
    "distortion_model": "plumb_bob",
    "pixels_are_undistorted": False,
    "pixel_coordinate_frame": "sensor_raw",
    "intrinsics_origin_device": "/dev/video22",
    "intrinsics_target_device": "/dev/video31",
    "intrinsics_transfer_unverified": True,
    "intrinsics_rms_px": 0.5077611957595818,
    "intrinsics_sample_count": 41,
    "C_world_mm": [24.5, 99.0, -108.0],
    "R_world_from_camera": [[0.0, 0.0, 1.0],
                            [-1.0, 0.0, 0.0],
                            [0.0, -1.0, 0.0]],
    "plane_y_mm": 183.5,
    "rail_lip_y_mm": 174.5,
    "is_approximate": True,
    "source": (
        "41张8x6(20mm)棋盘格实测内参(RMS=0.508px, 授权迁移至video31) + "
        "2026-11-16用户确认两场景相机位置C=[24.5,99.0,-108.0]mm、"
        "导轨轴心平面Y=183.5mm、前挡边平面Y=174.5mm"
    ),
    "assumptions": [
        "相机旋转仍按安装约束名义正交姿态，跨相机内参复用与安装角存在近似",
        "滑杆轴线按 Y=183.5 mm 平面求交，导轨左、右物理端点统一按前挡边 Y=174.5 mm 平面求交",
    ],
}


def _real_upright_default():
    """物理倒装相机的原始内参随图像转正；虚拟相机方向恢复为 ground 的方向。"""
    config = copy.deepcopy(DEFAULT_CALIBRATION)
    width, height = config["image_size_px"]
    config["K"][0][2] = width - 1.0 - config["K"][0][2]
    config["K"][1][2] = height - 1.0 - config["K"][1][2]
    config["D"][2] *= -1.0
    config["D"][3] *= -1.0
    config["name"] += "_real_upright"
    config["pixel_coordinate_frame"] = "scene_upright"
    config["source"] += "；real物理倒装180°，K/D按全帧180°转正变换，R使用转正后的名义方向"
    config["assumptions"].append(
        "real与ground仅绕光轴相差180°；原始传感器K/D复用，转正后cx'=W-1-cx、cy'=H-1-cy、p1/p2反号"
    )
    return config


DEFAULT_REAL_CALIBRATION = _real_upright_default()

_CALIBRATION_FIELDS = ("name", "image_size_px", "K", "C_world_mm",
                       "R_world_from_camera", "plane_y_mm")
_PIXEL_DEFINITION = "shaft_axis_intersection_with_image_reference_line"
_SUPPORTED_PROFILES = ("ground", "real")


def _rot_x(deg):
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def _rot_z(deg):
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _refine_pose_for_rail_line(config, rail_line):
    """根据实测导轨线补偿安装姿态角，保持标定中的光心坐标。"""
    if not isinstance(rail_line, dict) or rail_line.get("source") != "fitted":
        return config
    try:
        k = float(rail_line.get("k", 0.0))
        m = float(rail_line.get("m", 0.0))
    except (TypeError, ValueError):
        return config
    if not (0.12 <= k <= 0.80):
        return config

    y_mid = k * 550.0 + m
    # 若为转正倒装视图但传入了默认 sensor_raw 标定，先切換至转正内参
    base = (copy.deepcopy(DEFAULT_REAL_CALIBRATION)
            if (y_mid <= 420.0 or k >= 0.30) and config.get("pixel_coordinate_frame") == "sensor_raw"
            else copy.deepcopy(config))
    R0 = np.asarray(base["R_world_from_camera"], dtype=float)

    if k >= 0.50 or y_mid < 275.0:
        # 在轨 (real) 相机安装姿态补偿：绕世界 X 轴滚转 -22.55°、绕世界 Z 轴偏航 -2.545°
        R_corr = _rot_z(-2.545) @ _rot_x(-22.55)
        base["R_world_from_camera"] = (R_corr @ R0).tolist()
    return base


def _finite_array(value, shape, label):
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(label + " 必须是有限数值") from error
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError("%s 必须是形状 %s 的有限数值" % (label, shape))
    return array


def _finite_number(value, label):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(label + " 必须是有限数值")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(label + " 必须是有限数值") from error
    if not math.isfinite(number):
        raise ValueError(label + " 必须是有限数值")
    return number


def _image_size(value, label="image_size_px"):
    array = _finite_array(value, (2,), label)
    if np.any(array <= 0) or np.any(array != np.floor(array)):
        raise ValueError(label + " 必须为正整数 [宽, 高]")
    return tuple(int(item) for item in array)


def _calibration(calibration):
    explicit = calibration is not None
    if explicit and not isinstance(calibration, dict):
        raise ValueError("calibration 必须是完整的标定字典")
    config = copy.deepcopy(calibration if explicit else DEFAULT_CALIBRATION)
    missing = [key for key in _CALIBRATION_FIELDS if key not in config]
    if missing:
        raise ValueError("完整标定缺少字段：" + ", ".join(missing))
    if not isinstance(config["name"], str) or not config["name"].strip():
        raise ValueError("标定 name 必须是非空字符串")
    size = _image_size(config["image_size_px"], "标定 image_size_px")
    intrinsic = _finite_array(config["K"], (3, 3), "K")
    if (not np.allclose(intrinsic[2], [0, 0, 1], atol=1e-9, rtol=0)
            or abs(intrinsic[1, 0]) > 1e-9
            or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0):
        raise ValueError("K 必须是焦距为正、末行为 [0,0,1] 的相机内参矩阵")
    centre = _finite_array(config["C_world_mm"], (3,), "C_world_mm")
    rotation = _finite_array(config["R_world_from_camera"], (3, 3),
                             "R_world_from_camera")
    # A camera basis is right handed. A left-handed reporting world therefore
    # requires an orthogonal basis change with det=-1, not a physical rotation.
    handedness = config.get("world_handedness", "right")
    if handedness not in ("right", "left"):
        raise ValueError("world_handedness 必须为 right 或 left")
    expected_det = 1.0 if handedness == "right" else -1.0
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(rotation), expected_det, atol=1e-6, rtol=0)):
        raise ValueError("R_world_from_camera 必须正交且行列式与 world_handedness 一致")
    coordinate_frame = config.get("world_coordinate_frame", "world_equals_global_z_up")
    if not isinstance(coordinate_frame, str) or not coordinate_frame.strip():
        raise ValueError("world_coordinate_frame 必须是非空字符串")
    if handedness == "left" and coordinate_frame == "world_equals_global_z_up":
        raise ValueError("左手世界系必须显式指定区别于旧世界系的 world_coordinate_frame")
    plane_y = _finite_number(config["plane_y_mm"], "plane_y_mm")
    rail_lip_y = _finite_number(config.get("rail_lip_y_mm", plane_y - 9.0), "rail_lip_y_mm")
    approximate = config.get("is_approximate", True)
    if not isinstance(approximate, bool):
        raise ValueError("is_approximate 必须是布尔值")
    assumptions = config.get("assumptions", [])
    if not isinstance(assumptions, list) or any(not isinstance(x, str) for x in assumptions):
        raise ValueError("assumptions 必须为字符串列表")
    distortion = _finite_array(config.get("D", [0.] * 5), (5,), "D")
    if config.get("distortion_model", "plumb_bob") != "plumb_bob":
        raise ValueError("仅支持plumb_bob五参数畸变，不支持鱼眼模型")
    undistorted = config.get("pixels_are_undistorted", False)
    if not isinstance(undistorted, bool):
        raise ValueError("pixels_are_undistorted 必须是布尔值")
    config.update({"image_size_px": list(size), "K": intrinsic.tolist(),
                   "D": distortion.tolist(), "distortion_model": "plumb_bob",
                   "pixels_are_undistorted": undistorted,
                   "C_world_mm": centre.tolist(),
                   "R_world_from_camera": rotation.tolist(),
                   "world_handedness": handedness,
                   "world_coordinate_frame": coordinate_frame,
                   "plane_y_mm": plane_y,
                   "rail_lip_y_mm": rail_lip_y,
                   "is_approximate": approximate,
                   "assumptions": assumptions,
                   "source": str(config.get("source", "显式提供的完整视图标定"))})
    return config, explicit


def load_intrinsics(path):
    """读取网页导出的内参JSON或OpenCV FileStorage YAML，不包含世界外参。"""
    path = Path(path)
    if path.suffix.lower() in (".yaml", ".yml"):
        import cv2
        fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise ValueError("无法读取内参YAML")
        try:
            K = fs.getNode("camera_matrix").mat()
            D = fs.getNode("distortion_coefficients").mat()
            if K is None or D is None:
                raise ValueError("内参YAML缺少camera_matrix或distortion_coefficients")
            return dict(camera_matrix=K.tolist(), distortion_coefficients=D.ravel().tolist(),
                        image_width=fs.getNode("image_width").real(),
                        image_height=fs.getNode("image_height").real(),
                        distortion_model=fs.getNode("distortion_model").string(),
                        device=fs.getNode("camera_device").string(),
                        rms_px=fs.getNode("rms_px").real())
        finally:
            fs.release()
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def with_intrinsics(intrinsics, calibration=None):
    """以实测内参替换视图内参，保留视图原有C、R与导轨平面。"""
    if not isinstance(intrinsics, dict):
        raise ValueError("内参文件必须是JSON对象")
    required = ("camera_matrix", "distortion_coefficients", "image_width", "image_height")
    missing = [k for k in required if k not in intrinsics]
    if missing:
        raise ValueError("内参文件缺少字段：" + ", ".join(missing))
    base, _ = _calibration(calibration)
    base["intrinsics_only_default_view"] = calibration is None
    base.update(K=intrinsics["camera_matrix"], D=intrinsics["distortion_coefficients"],
                image_size_px=[intrinsics["image_width"], intrinsics["image_height"]],
                distortion_model=intrinsics.get("distortion_model", "plumb_bob"),
                pixels_are_undistorted=False,
                intrinsics_origin_device=intrinsics.get("device", "unspecified"))
    if "rms_px" in intrinsics:
        base["intrinsics_rms_px"] = _finite_number(intrinsics["rms_px"], "rms_px")
    if "sample_count" in intrinsics:
        base["intrinsics_sample_count"] = intrinsics["sample_count"]
    base["name"] = "loaded_intrinsics_" + base["name"]
    base["source"] += "；K、D和图像尺寸由--intrinsics文件覆盖，外参未改变"
    return _calibration(base)[0]


def _distortion_handling(config):
    if config["pixels_are_undistorted"]:
        return "input_already_undistorted_using_this_K"
    return "opencv_undistortPointsIter_plumb_bob" if np.any(config["D"]) else "zero_distortion_coefficients"


def _camera_ray(pixel, intrinsic, config):
    distorted = np.linalg.solve(intrinsic, [pixel[0], pixel[1], 1.0])
    if config["pixels_are_undistorted"] or not np.any(config["D"]):
        return distorted
    try:
        import cv2
    except ImportError as error:
        raise ValueError("非零畸变点校正需要OpenCV，请使用安装了cv2的Python") from error
    observed = distorted[:2].reshape(1, 1, 2)
    D = np.asarray(config["D"], dtype=float)
    corrected = cv2.undistortPointsIter(observed, np.eye(3), D, None, None,
        (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12)).reshape(2)
    ray = np.r_[corrected, 1.]
    if not np.all(np.isfinite(ray)):
        raise ValueError("畸变反解得到非有限射线")
    projected, _ = cv2.projectPoints(ray.reshape(1, 3), np.zeros(3), np.zeros(3), np.eye(3), D)
    normalized_error = projected.reshape(2) - distorted[:2]
    if np.linalg.norm(intrinsic[:2, :2] @ normalized_error) > 1e-5:
        raise ValueError("畸变反解未收敛，请检查畸变系数和图像模式")
    return ray


def _scaled_intrinsics(config, size):
    width, height = size
    calibration_width, calibration_height = config["image_size_px"]
    sx, sy = width / calibration_width, height / calibration_height
    if abs(width * calibration_height - height * calibration_width) > max(
            calibration_width, calibration_height):
        raise ValueError("输入图像宽高比与标定不同；裁剪或拉伸后必须提供对应标定")
    transform = np.asarray([[sx, 0., (sx-1.)/2.],
                            [0., sy, (sy-1.)/2.], [0., 0., 1.]])
    return transform @ np.asarray(config["K"], dtype=float)


def pixel_to_world(pixel_xy, image_size_px=(1920, 1080), calibration=None,
                   rail_z_mm=None, plane_y_override_mm=None):
    """返回一个像素对应的世界点。"""
    config, _ = _calibration(calibration)
    width, height = _image_size(image_size_px)
    pixel = _finite_array(pixel_xy, (2,), "pixel_xy")
    if not (0 <= pixel[0] < width and 0 <= pixel[1] < height):
        raise ValueError("pixel_xy 超出图像范围，坐标应满足 0≤u<宽、0≤v<高")
    intrinsic = _scaled_intrinsics(config, (width, height))
    rotation = np.asarray(config["R_world_from_camera"], dtype=float)
    centre = np.asarray(config["C_world_mm"], dtype=float)
    target_plane_y = (config["plane_y_mm"] if plane_y_override_mm is None
                      else _finite_number(plane_y_override_mm, "plane_y_override_mm"))
    camera_ray = _camera_ray(pixel, intrinsic, config)
    world_ray = rotation @ camera_ray
    norm = np.linalg.norm(world_ray)
    if abs(world_ray[1]) <= 1e-10 * norm:
        raise ValueError("像素射线与 Y 导轨平面平行，无法确定深度")
    distance_parameter = (target_plane_y - centre[1]) / world_ray[1]
    if distance_parameter <= 0:
        raise ValueError("交点位于相机后方或光心处，请检查像素及相机朝向/位置")
    reference_point = centre + distance_parameter * world_ray
    if rail_z_mm is None:
        point = reference_point
        definition = "shaft_axis_reference_point"
        method = "pixel_ray_intersection_with_known_world_y_plane"
    else:
        rail_z = _finite_number(rail_z_mm, "rail_z_mm")
        observation_normal = np.cross(world_ray, [0.0, 0.0, 1.0])
        system = np.asarray([observation_normal, [0., 1., 0.], [0., 0., 1.]])
        rhs = [float(observation_normal @ centre), target_plane_y, rail_z]
        try:
            point = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError as error:
            raise ValueError("竖直杆轴观测平面退化，无法求出轨道中心") from error
        camera_point = rotation.T @ (point - centre)
        if camera_point[2] <= 0:
            raise ValueError("指定轨道高度的杆轴点位于相机后方")
        definition = "vertical_shaft_axis_at_known_rail_height"
        method = "vertical_shaft_observation_plane_intersection_with_known_y_and_z"
    if not np.all(np.isfinite(point)):
        raise ValueError("求交结果不是有限数值，请检查标定和像素")

    result = {
        "pixel_xy": pixel.tolist(),
        "world_xyz_mm": point.tolist(),
        "position_definition": definition,
        "method": method,
        "units": "mm",
        "coordinate_frame": config["world_coordinate_frame"],
        "world_handedness": config["world_handedness"],
        "is_approximate": config["is_approximate"],
    }
    if rail_z_mm is not None:
        result["rail_z_mm"] = float(rail_z_mm)
        result["reference_ray_world_xyz_mm"] = reference_point.tolist()
    return result


def convert_detections(payload, calibration=None, rail_z_mm=None):
    """接收 detect_sliders.py 的 JSON 字典；转换滑杆坐标与诊断用导轨端点坐标，并透传可靠性字段。"""
    if not isinstance(payload, dict):
        raise ValueError("检测输入必须是 JSON 对象")
    if payload.get("schema_version") != 1:
        raise ValueError("检测输入 schema_version 必须为 1")
    if payload.get("position_definition") != _PIXEL_DEFINITION:
        raise ValueError("检测像素必须是杆轴与图像参考线交点，不能使用部件框中心替代")
    profile = payload.get("profile") or payload.get("scene") or "ground"
    if profile not in _SUPPORTED_PROFILES:
        raise ValueError(f"不支持的视图 profile={profile!r}；仅支持 {_SUPPORTED_PROFILES}")
    explicit_cal = calibration is not None
    selected_calibration = (DEFAULT_REAL_CALIBRATION
                            if profile == "real" and calibration is None else calibration)
    config, _ = _calibration(selected_calibration)
    explicit = explicit_cal
    if profile == "real":
        if config.get("intrinsics_only_default_view", False):
            raise ValueError("real 视图不能直接使用仅替换内参的地面标定；需提供转正画面的完整标定")
        if config.get("pixel_coordinate_frame") != "scene_upright":
            raise ValueError("real 视图标定必须声明 pixel_coordinate_frame='scene_upright'")

    size = _image_size(payload.get("input_size_px", config["image_size_px"]), "检测 input_size_px")
    _scaled_intrinsics(config, size)
    if rail_z_mm is not None:
        _finite_number(rail_z_mm, "rail_z_mm")

    active_config = (config if explicit
                     else _refine_pose_for_rail_line(config, payload.get("rail_line")))
    active_config, _ = _calibration(active_config)

    sliders = payload.get("sliders")
    if not isinstance(sliders, dict):
        raise ValueError("检测 sliders 必须为含 blue/yellow 的对象")
    converted = {"blue": None, "yellow": None}
    for color in ("blue", "yellow"):
        item = sliders.get(color)
        if color == "yellow" and item is None:
            item = sliders.get("orange")
        if item is None:
            continue
        if not isinstance(item, dict) or "pixel_xy" not in item:
            raise ValueError(color + " 检测项必须提供 pixel_xy")
        point = pixel_to_world(item["pixel_xy"], size, active_config, rail_z_mm)
        for key in (
            "method",
            "reference_method",
            "representative_part",
            "approximate",
            "occluded_side",
            "rows_used",
            "width_px",
            "width_model_px",
            "reliable",
            "reasons",
        ):
            if key in item:
                point[key] = item[key]
        reference_method = item.get("reference_method", "")
        block_fallback = ("approximate" in str(reference_method)
                          or str(reference_method).startswith("block"))
        if item.get("approximate") is True or block_fallback:
            point["is_approximate"] = True
        converted[color] = point

    rail_endpoints = None
    eps_px = payload.get("rail_endpoints_px")
    if eps_px is None:
        eps_px = payload.get("rail_line")
    if (isinstance(eps_px, (list, tuple)) and len(eps_px) == 2
            and eps_px[0] is not None and eps_px[1] is not None):
        try:
            lip_y = active_config.get("rail_lip_y_mm", active_config["plane_y_mm"] - 9.0)
            ep_left = pixel_to_world(eps_px[0], size, active_config, rail_z_mm=None, plane_y_override_mm=lip_y)
            ep_right = pixel_to_world(eps_px[1], size, active_config, rail_z_mm=None, plane_y_override_mm=lip_y)
            rail_endpoints = {
                "left": ep_left,
                "right": ep_right,
            }
        except ValueError:
            rail_endpoints = None

    found = sum(converted[c] is not None for c in ("blue", "yellow"))
    return {
        "schema_version": 1,
        "profile": profile,
        "input_size_px": list(size),
        "status": "complete" if found == 2 else ("partial" if found else "missing"),
        "units": "mm",
        "coordinate_frame": active_config["world_coordinate_frame"],
        "world_handedness": active_config["world_handedness"],
        "position_definition": ("shaft_axis_reference_point" if rail_z_mm is None
                                else "vertical_shaft_axis_at_known_rail_height"),
        "is_approximate": config["is_approximate"] or any(
            converted[c] is not None and converted[c]["is_approximate"] for c in ("blue", "yellow")),
        "calibration": active_config,
        "distortion_handling": _distortion_handling(active_config),
        "rail_line_kb": payload.get("rail_line_kb"),
        "rail_line_fallback": bool(payload.get("rail_line_fallback", False)),
        "rail_endpoints_px": eps_px,
        "rail_endpoints": rail_endpoints,
        "sliders": converted,
    }


def _load_json(path):
    if path == "-":
        return json.load(sys.stdin)
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", help="检测 JSON；- 表示标准输入")
    parser.add_argument("--blue", nargs=2, type=float, metavar=("U", "V"))
    parser.add_argument("--yellow", "--orange", dest="yellow", nargs=2, type=float, metavar=("U", "V"))
    parser.add_argument("--image-size", nargs=2, type=int, default=[1920, 1080],
                        metavar=("W", "H"), help="手动像素所在图像尺寸，默认 1920 1080")
    parser.add_argument("--calibration", help="当前视图的完整标定 JSON")
    parser.add_argument("--intrinsics", help="棋盘实测内参JSON/OpenCV YAML，仅覆盖K、D和图像尺寸")
    parser.add_argument("--rail-z-mm", type=float, help="已知轨道中心 Z；未提供时返回杆轴参考点")
    parser.add_argument("--output", help="输出 JSON，默认标准输出；- 也表示标准输出")
    args = parser.parse_args(argv)
    if args.input and (args.blue is not None or args.yellow is not None):
        parser.error("--input 与 --blue/--yellow 不能同时使用")
    if not args.input and args.blue is None and args.yellow is None:
        parser.error("请提供 --input，或至少一组 --blue/--yellow 像素")
    if args.input == "-" and args.calibration == "-":
        parser.error("检测和标定不能同时从标准输入读取")
    try:
        if args.output and args.output != "-":
            destination = Path(args.output).resolve()
            for source in (args.input, args.calibration, args.intrinsics):
                if source and source != "-":
                    source_path = Path(source).resolve()
                    same_file = (destination.exists() and source_path.exists()
                                 and destination.samefile(source_path))
                    if destination == source_path or same_file:
                        raise ValueError("输出路径不能覆盖输入或标定文件")
        calibration = _load_json(args.calibration) if args.calibration else None
        if args.intrinsics:
            calibration = with_intrinsics(load_intrinsics(args.intrinsics), calibration)
        if args.input:
            payload = _load_json(args.input)
        else:
            payload = {
                "schema_version": 1, "profile": "ground",
                "input_size_px": args.image_size,
                "position_definition": _PIXEL_DEFINITION,
                "sliders": {
                    "blue": {"pixel_xy": args.blue, "reference_method": "manual_shaft_axis_pixel"}
                            if args.blue is not None else None,
                    "yellow": {"pixel_xy": args.yellow, "reference_method": "manual_shaft_axis_pixel"}
                              if args.yellow is not None else None,
                },
            }
        result = convert_detections(payload, calibration, args.rail_z_mm)
        output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output and args.output != "-":
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(output, encoding="utf-8")
        else:
            sys.stdout.write(output)
    except (ValueError, OSError, TypeError, np.linalg.LinAlgError) as error:
        parser.exit(2, "坐标转换失败：%s\n" % error)
    return 0 if result["status"] == "complete" or not args.input else 1


if __name__ == "__main__":
    raise SystemExit(main())
