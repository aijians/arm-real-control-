#!/usr/bin/env python3
"""将滑杆轴线像素转换到世界坐标系，单位为毫米。

依赖 Python、NumPy；非零畸变校正另外依赖 OpenCV。世界坐标系与全局
坐标系相同，Z 向上。默认使用2026-11-05提供的video22棋盘实测内参、
五参数畸变，并按用户要求复用到video31的1920x1080原始图像。
内参来源相机与使用相机不同；尚未验证跨相机复用的毫米级精度。
相机位置为 [24.5, 99.0, -102.0] mm，光轴 +X、画面上方 +Z、零滚转；
位置已重新量测，旋转仍按安装约束；导轨轴线Y=183.5，前挡边Y=174.5 mm。

默认输出：像素射线与 Y 平面的交点，对应检测脚本的杆轴参考点，
并不自动代表杆脚、导轨中心或质量中心。--rail-z-mm 给出已知轨道中心
高度时，将像素解释为世界竖直杆轴上的一点或其延长线，建立包含相机、
该射线及世界 Z 方向的观测平面，与指定 Y、Z 联立求交，输出轨道中心。
方块回退只提供近似杆轴位置，转换结果会保留此标记。

API:
    pixel_to_world([420.4, 544.6])
    convert_detections(detection_payload)

CLI:
    python3 pixel_to_world.py --input detection.json --output world.json
    python3 pixel_to_world.py --input - < detection.json
    python3 pixel_to_world.py --blue 420.4 544.6 --yellow 604 552
    python3 pixel_to_world.py --blue 420.4 544.6 --rail-z-mm -80
    python3 pixel_to_world.py --blue 420.4 544.6 --intrinsics calibration.json

--calibration 可接收独立 JSON 对象，必须完整提供：name、image_size_px、
K（3x3）、C_world_mm（3 元素）、R_world_from_camera（3x3 旋转矩阵）、
plane_y_mm；可选 D（k1,k2,p1,p2,k3）、pixels_are_undistorted、source、
assumptions、is_approximate。D未提供时按零畸变处理，不借用默认D。
pixels_are_undistorted=true仅用于已经按该K去畸变、未额外裁剪的像素，
避免二次去畸变。--intrinsics接受网页导出的JSON或OpenCV YAML，只替换
K、D和标定分辨率，不从内参文件推断或覆盖外参。R 的列分别是
相机 x(右)、y(下)、z(前)轴在世界系中的方向。图像尺寸可以按相同比例
缩放（允许尺寸取整造成的一像素差异），禁止不同比例拉伸；不会自动旋转
图像或坐标。检测 JSON 中有目标缺失时，CLI 返回 1；手动输入单色坐标
可正常返回 0，输入错误返回 2。ground与real均支持默认参数；real必须
先在检测器中转正，复用默认安装几何会保留近似标记，不作额外旋转。
"""

import argparse
import copy
import json
import math
from pathlib import Path
import sys

import numpy as np


DEFAULT_CALIBRATION = {
    "name": "imx415_measured_position_1080p",
    "image_size_px": [1920, 1080],
    "K": [[698.5891449464962, 0.0, 960.8663237262527],
          [0.0, 698.2399519207177, 567.6531461824138],
          [0.0, 0.0, 1.0]],
    "D": [0.05322918059115373, -0.06375883808099647,
          0.00008002273224807077, 0.0011399378118325647,
          0.015484807643491312],
    "distortion_model": "plumb_bob",
    "pixels_are_undistorted": False,
    "intrinsics_origin_device": "/dev/video22",
    "intrinsics_target_device": "/dev/video31",
    "intrinsics_transfer_unverified": True,
    "intrinsics_rms_px": 0.5077611957595818,
    "intrinsics_sample_count": 41,
    "extrinsics_is_measured": False,
    "position_is_measured": True,
    "C_world_mm": [24.5, 99.0, -102.0],
    "R_world_from_camera": [[0.0, 0.0, 1.0],
                            [-1.0, 0.0, 0.0],
                            [0.0, -1.0, 0.0]],
    "shoulder_global_xyz_mm": [189.3, 28.5, -67.0],
    "plane_y_mm": 183.5,
    "rail_lip_y_mm": 174.5,
    "is_approximate": True,
    "source": "2026-11-05用户提供calibration.json：video22，41张8x6内角点、"
              "20mm方格棋盘图，RMS=0.5077611957595818px；按用户要求复用到video31；"
              "用户重新量测相机C=[24.5,99.0,-102.0]、肩轴=[189.3,28.5,-67.0]、"
              "轴心面Y=183.5、前挡边Y=174.5mm；旋转按安装约束",
    "assumptions": [
        "世界坐标系与全局坐标系相同，毫米，Z 向上",
        "光轴沿世界 +X，画面上方沿世界 +Z，零滚转；未实测外参",
        "K及五参数D来自video22棋盘实测；按用户决定迁移到video31，个体差异尚未验证",
        "完整 3840x2160 有效画面宽高各缩小至 1/2，无额外裁剪或旋转",
        "输入为原始畸变图像上的轴线像素，求射线前按D进行点去畸变",
        "重投影误差单位为像素，不能直接当作世界坐标毫米误差",
        "世界坐标按相机几何直接求交，不应用端点拟合比例、偏移等经验补偿",
        "real按用户约定复用转正后的默认内参及安装几何；该场景的外参/毫米精度未经独立标定",
    ],
}

_CALIBRATION_FIELDS = ("name", "image_size_px", "K", "C_world_mm",
                       "R_world_from_camera", "plane_y_mm")
_PIXEL_DEFINITION = "shaft_axis_intersection_with_image_reference_line"


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
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6, rtol=0)):
        raise ValueError("R_world_from_camera 必须是正交且行列式为 +1 的旋转矩阵")
    plane_y = _finite_number(config["plane_y_mm"], "plane_y_mm")
    lip_y = _finite_number(config.get("rail_lip_y_mm", plane_y), "rail_lip_y_mm")
    shoulder_xyz = _finite_array(config.get("shoulder_global_xyz_mm", [189.3, plane_y - 155., -67.]),
                                 (3,), "shoulder_global_xyz_mm")
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
                   "plane_y_mm": plane_y, "rail_lip_y_mm": lip_y,
                   "shoulder_global_xyz_mm": shoulder_xyz.tolist(), "is_approximate": approximate,
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
    base["intrinsics_transfer_unverified"] = (
        base["intrinsics_origin_device"] != base.get("intrinsics_target_device", "unspecified"))
    if "rms_px" in intrinsics:
        base["intrinsics_rms_px"] = _finite_number(intrinsics["rms_px"], "rms_px")
    if "sample_count" in intrinsics:
        base["intrinsics_sample_count"] = intrinsics["sample_count"]
    base["name"] = "loaded_intrinsics_" + base["name"]
    base["source"] += "；K、D和图像尺寸由--intrinsics文件覆盖，外参未改变"
    if base["intrinsics_transfer_unverified"]:
        base["is_approximate"] = True
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
    # Work in normalized coordinates so even a valid nonzero K skew is handled.
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
    # Use the same one-pixel resize-rounding tolerance as the detector.
    if abs(width * calibration_height - height * calibration_width) > max(
            calibration_width, calibration_height):
        raise ValueError("输入图像宽高比与标定不同；裁剪或拉伸后必须提供对应标定")
    # Match the detector/OpenCV resize convention for pixel-centre coordinates:
    # u'=(u+0.5)*sx-0.5, v'=(v+0.5)*sy-0.5.
    transform = np.asarray([[sx, 0., (sx-1.)/2.],
                            [0., sy, (sy-1.)/2.], [0., 0., 1.]])
    return transform @ np.asarray(config["K"], dtype=float)


def pixel_to_world(pixel_xy, image_size_px=(1920, 1080), calibration=None,
                   rail_z_mm=None):
    """返回一个像素对应的世界点及其语义；不能确定有效交点时抛 ValueError。

    默认点是射线与导轨 Y 平面的交点。指定 rail_z_mm 时，要求 pixel_xy
    位于世界竖直滑杆的轴线投影上，输出同一轴线在已知轨道高度的点。
    标定只随完整同视野图像进行等比例缩放，不处理额外裁剪或图像旋转。
    """
    config, _ = _calibration(calibration)
    width, height = _image_size(image_size_px)
    pixel = _finite_array(pixel_xy, (2,), "pixel_xy")
    if not (0 <= pixel[0] < width and 0 <= pixel[1] < height):
        raise ValueError("pixel_xy 超出图像范围，坐标应满足 0≤u<宽、0≤v<高")
    intrinsic = _scaled_intrinsics(config, (width, height))
    rotation = np.asarray(config["R_world_from_camera"], dtype=float)
    centre = np.asarray(config["C_world_mm"], dtype=float)
    camera_ray = _camera_ray(pixel, intrinsic, config)
    world_ray = rotation @ camera_ray
    norm = np.linalg.norm(world_ray)
    if abs(world_ray[1]) <= 1e-10 * norm:
        raise ValueError("像素射线与 Y 导轨平面平行，无法确定深度")
    distance_parameter = (config["plane_y_mm"] - centre[1]) / world_ray[1]
    if distance_parameter <= 0:
        raise ValueError("交点位于相机后方或光心处，请检查像素及相机朝向/位置")
    reference_point = centre + distance_parameter * world_ray
    if rail_z_mm is None:
        point = reference_point
        definition = "shaft_axis_reference_point"
    else:
        rail_z = _finite_number(rail_z_mm, "rail_z_mm")
        # The known vertical shaft and the observed camera ray span a plane.
        # Its intersection with Y=rail_y and Z=rail_z is the requested axis point.
        # Solving the constraints makes the vertical-shaft assumption explicit;
        # this is not an arbitrary change of a surface point's depth or height.
        observation_normal = np.cross(world_ray, [0.0, 0.0, 1.0])
        system = np.asarray([observation_normal, [0., 1., 0.], [0., 0., 1.]])
        rhs = [float(observation_normal @ centre), config["plane_y_mm"], rail_z]
        try:
            point = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError as error:
            raise ValueError("竖直杆轴观测平面退化，无法求出轨道中心") from error
        camera_point = rotation.T @ (point - centre)
        if camera_point[2] <= 0:
            raise ValueError("指定轨道高度的杆轴点位于相机后方")
        definition = "vertical_shaft_axis_at_known_rail_height"
    if not np.all(np.isfinite(point)):
        raise ValueError("求交结果不是有限数值，请检查标定和像素")
    result = {
        "pixel_xy": pixel.tolist(),
        "world_xyz_mm": point.tolist(),
        "position_definition": definition,
        "is_approximate": config["is_approximate"],
        "distortion_handling": _distortion_handling(config),
    }
    if rail_z_mm is not None:
        result["rail_z_mm"] = float(rail_z_mm)
    return result


def convert_detections(payload, calibration=None, rail_z_mm=None):
    """接收 detect_sliders.py 的 JSON 字典；返回固定 blue/yellow 两色字典。

    未检出目标保持 null。ground和real均可使用默认参数；real像素须为
    检测器转正后的坐标。复用安装几何作为近似假设，不再次旋转像素或K。
    """
    if not isinstance(payload, dict):
        raise ValueError("检测输入必须是 JSON 对象")
    if payload.get("schema_version") != 1:
        raise ValueError("检测输入 schema_version 必须为 1")
    if payload.get("position_definition") != _PIXEL_DEFINITION:
        raise ValueError("检测像素必须是杆轴与图像参考线交点，不能使用部件框中心替代")
    config, explicit = _calibration(calibration)
    profile = payload.get("profile")
    if profile not in ("ground", "real"):
        raise ValueError("profile必须是ground或real")
    if profile == "real":
        frame = payload.get("pixel_coordinate_frame", "scene_upright")
        if frame != "scene_upright" or config.get("pixel_coordinate_frame", frame) != frame:
            raise ValueError("real输入像素和标定必须对应scene_upright转正画面")
        config["pixel_coordinate_frame"] = frame
        reused = not explicit or config.get("intrinsics_only_default_view", False)
        config["real_geometry_reused"] = reused
        if reused:
            config["is_approximate"] = True
            config["assumptions"].append(
                "real按用户约定复用默认K、D、C、R及平面距离，未独立标定该安装外参")
    size = _image_size(payload.get("input_size_px"), "检测 input_size_px")
    _scaled_intrinsics(config, size)
    if rail_z_mm is not None:
        _finite_number(rail_z_mm, "rail_z_mm")
    sliders = payload.get("sliders")
    if not isinstance(sliders, dict):
        raise ValueError("检测 sliders 必须为含 blue/yellow 的对象")
    converted = {"blue": None, "yellow": None}
    for color in converted:
        item = sliders.get(color)
        if color == "yellow" and "yellow" not in sliders:
            item = sliders.get("orange")
        if item is None:
            continue
        if not isinstance(item, dict) or "pixel_xy" not in item:
            raise ValueError(color + " 检测项必须提供 pixel_xy")
        point = pixel_to_world(item["pixel_xy"], size, config, rail_z_mm)
        for key in ("reference_method", "representative_part", "confidence", "approximate"):
            if key in item:
                point[key] = item[key]
        reference_method = item.get("reference_method", "")
        block_fallback = ("approximate" in str(reference_method)
                          or str(reference_method).startswith("block"))
        if item.get("approximate") is True or block_fallback:
            point["is_approximate"] = True
        converted[color] = point
    found = sum(item is not None for item in converted.values())
    return {
        "schema_version": 1,
        "profile": profile,
        "input_size_px": list(size),
        "source_image": payload.get("source_image"),
        "status": "complete" if found == 2 else ("partial" if found else "missing"),
        "units": "mm",
        "coordinate_frame": "world_equals_global_z_up",
        "position_definition": ("shaft_axis_reference_point" if rail_z_mm is None
                                else "vertical_shaft_axis_at_known_rail_height"),
        "is_approximate": config["is_approximate"] or any(
            item is not None and item["is_approximate"] for item in converted.values()),
        "calibration": config,
        "distortion_handling": _distortion_handling(config),
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
    parser.add_argument("--yellow", nargs=2, type=float, metavar=("U", "V"))
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
