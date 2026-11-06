#!/usr/bin/env python3
"""蓝色滑杆与橙色滑杆像素位置及导轨真实端点识别脚本（支持地面仿真 ground 与真实目标 real 场景）。

核心算法与工程接口设计：
1. 自动白色导轨直线拟合与真实两端点检测（方案 C + 真实槽端点定位）：
   - 在导轨走廊内通过竖直梯度峰值 + 抛物线亚像素插值 + Huber IRLS 自动拟合导轨槽线 y = k * x + m；
   - 沿拟合出的导轨直线提取一维法向边缘强度剖面 E(x)，通过补零一维形态学闭运算（跨越滑杆遮挡区）
     与线性亚像素阈值插值，直接定位导轨真实左端点与右端点作为输出 `rail_endpoints_px`。
2. 四重抗高光分裂与稳健轴线提取机制：
   - 曲面高光物理补偿掩膜：识别蓝色圆柱表面因 B 通道饱和 (B=255) 导致色相略偏青 (H∈[90,99))
     但红通道极低 (S>=165, B-R>=150) 的中央高光竖条，避免在掩膜阶段将圆柱一分为二；
   - 横向各向异性闭运算：对上端圆柱杆区域执行 (15, 3) 水平形态学闭运算，横向弥合极端纯白反光缝隙；
   - 上下刚体耦合与碎片归并：以大面积下端滑块 (block) 锁定水平走廊，自动合并同一高度层内被高光
     切开的左右半圆柱连通块，杜绝单取最大连通域时“丢掉半边圆柱”引起的跳变；
   - 逐行外轮廓对称中点中位数：在圆柱主体段（15%~70%高度）逐行取左、右外轮廓边界中点
     0.5 * (x_left + x_right)，过滤残缺窄行后取中位数作为轴线 X 坐标，完全不受内部空洞影响。
3. 统一正向像素坐标系与全流程接口兼容：
   - real 场景在输入时自动旋转 180° 转正，输出使用转正后的像素坐标系（不表示可共享ground的相机外参）；
   - 同时支持 `profile` 与 `scene` 参数名，输出结构完整兼容 `pixel_to_world.convert_detections` 与 `vision.Vision`。
"""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


SCENES = {
    "ground": {
        "canonical_size": (1920, 1080),
        "rotation_deg": 0,
        "roi": (260, 310, 690, 620),
        "rail_x_range": (455, 660),
        "rail_nominal": (465.0, 468.0, 0.225),
        "rail_band_px": 30,
        "lip_split_half_width": 6,
        "endpoint_edge_thresh": 28.0,
    },
    "real": {
        "canonical_size": (1920, 1080),
        "rotation_deg": 180,
        "roi": (420, 140, 700, 365),
        "rail_x_range": (465, 680),
        "rail_nominal": (480.0, 168.0, 0.650),
        "rail_band_px": 30,
        "lip_split_half_width": 6,
        "endpoint_edge_thresh": 28.0,
    },
}

PROFILES = SCENES


def _read_image(image_path):
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    if not encoded.size:
        raise ValueError(f"Image file is empty: {path}")
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None or image.size == 0:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def _detect_rail_line(work_img, cfg):
    """在导轨走廊内自动提取白色导轨挡边/滑槽下沿直线 y = k * x + m（亚像素精度）。"""
    gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_smooth = cv2.GaussianBlur(gray, (9, 5), 2.0, 1.0)
    dy = cv2.Sobel(gray_smooth, cv2.CV_32F, 0, 1, ksize=3)

    x_start, x_end = cfg["rail_x_range"]
    x0, y0, k_nom = cfg["rail_nominal"]
    half_band = cfg["rail_band_px"]
    h = work_img.shape[0]

    xs, ys, weights = [], [], []
    for x in range(x_start, x_end):
        yn = y0 + k_nom * (x - x0)
        y1 = max(0, int(round(yn - half_band)))
        y2 = min(h, int(round(yn + half_band)))
        col_edge = -dy[y1:y2, x]
        idx = int(np.argmax(col_edge))
        strength = float(col_edge[idx])
        if strength > 25.0:
            if 0 < idx < len(col_edge) - 1:
                ym, yc, yp = col_edge[idx - 1], col_edge[idx], col_edge[idx + 1]
                denom = ym - 2.0 * yc + yp
                offset = float(np.clip(0.5 * (ym - yp) / denom, -0.5, 0.5)) if abs(denom) > 1e-5 else 0.0
            else:
                offset = 0.0
            xs.append(float(x))
            ys.append(float(y1 + idx + offset))
            weights.append(strength)

    if len(xs) < 15:
        raise ValueError("rail_line_not_detected")

    xs_arr = np.asarray(xs, dtype=np.float64)
    ys_arr = np.asarray(ys, dtype=np.float64)
    w_arr = np.asarray(weights, dtype=np.float64)

    k, m = np.polyfit(xs_arr, ys_arr, 1, w=w_arr)
    for _ in range(5):
        residual = np.abs(ys_arr - (k * xs_arr + m))
        w_robust = w_arr / np.maximum(1.0, residual / 2.0)
        k, m = np.polyfit(xs_arr, ys_arr, 1, w=w_robust)
    return float(k), float(m)


def _detect_rail_endpoints(work_img, k, m, cfg):
    """沿已拟合的导轨直线 y = k * x + m 扫描一维梯度剖面，亚像素定位真实导轨左、右端点。"""
    gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_smooth = cv2.GaussianBlur(gray, (9, 5), 2.0, 1.0)
    neg_dy = -cv2.Sobel(gray_smooth, cv2.CV_32F, 0, 1, ksize=3)

    x1, _, x2, _ = cfg["roi"]
    h = work_img.shape[0]
    xs = np.arange(x1, x2, dtype=np.int32)
    e_prof = np.zeros(len(xs), dtype=np.float32)
    for i, x in enumerate(xs):
        yr = int(round(k * x + m))
        by1 = max(0, yr - 6)
        by2 = min(h, yr + 7)
        if by2 > by1:
            e_prof[i] = float(np.max(neg_dy[by1:by2, x]))

    e_smooth = cv2.GaussianBlur(e_prof.reshape(1, -1), (7, 1), 1.5).ravel()
    thresh = float(cfg.get("endpoint_edge_thresh", 28.0))
    active = (e_smooth >= thresh).astype(np.uint8)

    pad = 25
    active_pad = np.pad(active, (pad, pad), mode="constant", constant_values=0)
    closed_pad = cv2.morphologyEx(
        active_pad.reshape(1, -1),
        cv2.MORPH_CLOSE,
        np.ones((1, 35), np.uint8),
    ).ravel()
    active_closed = closed_pad[pad:-pad]

    nz = np.flatnonzero(active_closed)
    if len(nz) < 20:
        return None

    splits = np.where(np.diff(nz) > 1)[0] + 1
    best_seg = max(np.split(nz, splits), key=len)
    idx_l, idx_r = int(best_seg[0]), int(best_seg[-1])
    if len(best_seg) < 20 or idx_l == 0 or idx_r == len(xs) - 1:
        return None

    if idx_l > 0 and e_smooth[idx_l] > e_smooth[idx_l - 1]:
        frac_l = (thresh - e_smooth[idx_l - 1]) / (e_smooth[idx_l] - e_smooth[idx_l - 1])
        x_left = float(xs[idx_l - 1] + np.clip(frac_l, 0.0, 1.0))
    else:
        x_left = float(xs[idx_l])

    if idx_r < len(xs) - 1 and e_smooth[idx_r] > e_smooth[idx_r + 1]:
        frac_r = (e_smooth[idx_r] - thresh) / (e_smooth[idx_r] - e_smooth[idx_r + 1])
        x_right = float(xs[idx_r] + np.clip(frac_r, 0.0, 1.0))
    else:
        x_right = float(xs[idx_r])

    return (x_left, float(k * x_left + m)), (x_right, float(k * x_right + m))


def _build_color_masks(patch):
    """结合 HSV、CIE LAB 与 RGB 通道差提取蓝色与橙色滑杆掩膜（含圆柱表面高光条补偿）。"""
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    b, g, r = cv2.split(patch.astype(np.int16))

    is_core_blue = (
        (hsv[:, :, 0] >= 99) & (hsv[:, :, 0] <= 132) &
        (hsv[:, :, 1] >= 128) & (hsv[:, :, 2] >= 70) &
        ((b - g) >= 45) & ((b - r) >= 90) &
        (lab[:, :, 2] <= 94)
    )
    is_specular_blue = (
        (hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 132) &
        (hsv[:, :, 1] >= 165) & (hsv[:, :, 2] >= 180) &
        ((b - r) >= 150) & (lab[:, :, 2] <= 116)
    )
    blue_mask = (is_core_blue | is_specular_blue).astype(np.uint8) * 255

    orange_mask = (
        (hsv[:, :, 0] >= 3) & (hsv[:, :, 0] <= 38) &
        (hsv[:, :, 1] >= 38) & (hsv[:, :, 2] >= 85) &
        ((r - b) >= 35) & ((g - b) >= 12) &
        (lab[:, :, 2] >= 136)
    ).astype(np.uint8) * 255

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    blue_mask = cv2.morphologyEx(cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, k_open), cv2.MORPH_CLOSE, k_close)
    orange_mask = cv2.morphologyEx(cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, k_open), cv2.MORPH_CLOSE, k_close)
    return {"blue": blue_mask, "orange": orange_mask}


def _extract_merged_component(mask_part, offset_xy, min_area, ref_x_range=None, max_merge_gap=20):
    """提取连通域并自动合并同一滑杆走廊内因极端高光切开的左右相邻碎片。"""
    x1, y1 = offset_xy
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_part)
    comps = []
    for i in range(1, num):
        bx, by, bw, bh, area = stats[i]
        if area < min_area or min(bw, bh) < 5:
            continue
        comps.append({
            "id": i,
            "bbox": (float(bx + x1), float(by + y1), float(bx + x1 + bw), float(by + y1 + bh)),
            "center": (float(centroids[i][0] + x1), float(centroids[i][1] + y1)),
            "area": int(area),
        })
    if not comps:
        return None

    if ref_x_range is not None:
        rx1, rx2 = ref_x_range
        aligned = [c for c in comps if c["bbox"][2] >= rx1 - 12.0 and c["bbox"][0] <= rx2 + 12.0]
        if aligned:
            comps = aligned

    seed = max(comps, key=lambda c: c["area"])
    sx1, sy1, sx2, sy2 = seed["bbox"]
    merged_ids = [seed["id"]]
    for c in comps:
        if c["id"] == seed["id"]:
            continue
        cx1, cy1, cx2, cy2 = c["bbox"]
        v_overlap = min(sy2, cy2) - max(sy1, cy1)
        h_gap = max(0.0, max(sx1, cx1) - min(sx2, cx2))
        if v_overlap > 0.4 * min(sy2 - sy1, cy2 - cy1) and h_gap <= max_merge_gap:
            merged_ids.append(c["id"])
            sx1, sy1 = min(sx1, cx1), min(sy1, cy1)
            sx2, sy2 = max(sx2, cx2), max(sy2, cy2)

    merged_mask = np.isin(labels, merged_ids)
    ys, xs = np.nonzero(merged_mask)
    mbx1, mby1 = int(xs.min()), int(ys.min())
    mbx2, mby2 = int(xs.max()) + 1, int(ys.max()) + 1
    return {
        "bbox": (float(mbx1 + x1), float(mby1 + y1), float(mbx2 + x1), float(mby2 + y1)),
        "local_bbox": (mbx1, mby1, mbx2 - mbx1, mby2 - mby1),
        "center": (float(xs.mean() + x1), float(ys.mean() + y1)),
        "mask": merged_mask,
        "area": int(len(xs)),
    }


def _compute_rod_axis_x(rod, offset_x):
    """在圆柱杆 15%~70% 高度段内，逐行计算左、右外轮廓边界中点并取中位数。"""
    bx, by, bw, bh = rod["local_bbox"]
    r_top = by + int(round(bh * 0.15))
    r_bot = by + int(round(bh * 0.70))
    sub_mask = rod["mask"][r_top:r_bot, bx : bx + bw]
    row_midpoints = []
    row_widths = []
    for row in sub_mask:
        nz = np.flatnonzero(row)
        if len(nz) >= 4:
            row_midpoints.append(0.5 * float(nz[0] + nz[-1]) + offset_x + bx)
            row_widths.append(float(nz[-1] - nz[0] + 1))
    if len(row_midpoints) >= 5:
        mid_arr = np.asarray(row_midpoints, dtype=np.float64)
        w_arr = np.asarray(row_widths, dtype=np.float64)
        valid = w_arr >= 0.75 * np.median(w_arr)
        return float(np.median(mid_arr[valid] if np.any(valid) else mid_arr))
    return rod["center"][0]


def _map_point(point, canonical_size, original_size):
    """将标准尺度坐标按比例映射回输入图分辨率（保持转正后的正向像素坐标系，不做 180° 逆旋转）。"""
    x, y = point
    cw, ch = canonical_size
    ow, oh = original_size
    return [
        round(float((x + 0.5) * ow / cw - 0.5), 2),
        round(float((y + 0.5) * oh / ch - 0.5), 2),
    ]





def detect_frame(image, profile="ground", scene=None, already_rotated=False):
    """识别单帧 BGR 图像中的导轨真实两端点及蓝色、橙色滑杆坐标。

    同时兼容 `profile`（供 vision.py 调用）与 `scene` 参数名。
    """
    active_scene = scene if scene is not None else profile
    if active_scene not in SCENES:
        raise ValueError(f"Unknown scene/profile {active_scene!r}; choose from {tuple(SCENES)}")
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3 or image.size == 0):
        raise ValueError("Input must be a non-empty uint8 3-channel BGR image")
    if not isinstance(already_rotated, bool):
        raise ValueError("already_rotated must be a bool")
    if scene is not None and profile != "ground" and scene != profile:
        raise ValueError("scene and profile disagree")

    cfg = SCENES[active_scene]
    cw, ch = cfg["canonical_size"]
    oh, ow = image.shape[:2]
    if abs(ow * ch - oh * cw) > max(cw, ch):
        raise ValueError(
            f"Image aspect ratio {ow}x{oh} does not match {active_scene} ({cw}x{ch}); "
            "use a full-frame 16:9 image."
        )
    rotation = 0 if already_rotated else cfg["rotation_deg"]

    work = cv2.rotate(image, cv2.ROTATE_180) if rotation == 180 else image
    if (ow, oh) != (cw, ch):
        work = cv2.resize(work, (cw, ch), interpolation=cv2.INTER_LINEAR)

    # 1. 自动拟合白色导轨槽线 y = k * x + m，并检测导轨真实左、右物理端点
    k, m = _detect_rail_line(work, cfg)
    rail_endpoints = _detect_rail_endpoints(work, k, m, cfg)

    # 2. 截取 ROI 并分割蓝、橙滑杆上端圆柱杆 (rod) 与下端滑块 (block)
    x1, y1, x2, y2 = cfg["roi"]
    patch = work[y1:y2, x1:x2]
    masks = _build_color_masks(patch)

    py, px = np.indices((y2 - y1, x2 - x1))
    gx, gy = px + x1, py + y1
    rail_y_grid = k * gx + m
    half_lip = cfg["lip_split_half_width"]
    k_rod_horiz_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    k_block_open = np.ones((7, 7), np.uint8)

    sliders = {"blue": None, "yellow": None}
    for color, mask in masks.items():
        upper_raw = np.where(gy < rail_y_grid - half_lip, mask, 0).astype(np.uint8)
        upper_mask = cv2.morphologyEx(upper_raw, cv2.MORPH_CLOSE, k_rod_horiz_close)
        lower_mask = cv2.morphologyEx(
            np.where(gy > rail_y_grid + half_lip, mask, 0).astype(np.uint8),
            cv2.MORPH_OPEN,
            k_block_open,
        )

        block = _extract_merged_component(lower_mask, (x1, y1), min_area=200)
        ref_x = (block["bbox"][0], block["bbox"][2]) if block is not None else None
        rod = _extract_merged_component(upper_mask, (x1, y1), min_area=120, ref_x_range=ref_x)
        if rod is None and block is None:
            continue

        # 3. 逐行外轮廓对称中点中位数提取 x_axis
        if rod is not None:
            x_axis = _compute_rod_axis_x(rod, x1)
            method = "rod_outer_midpoint_median_intersection"
            is_approx = False
        else:
            x_axis = block["center"][0]
            method = "block_center_projection_approximate"
            is_approx = True

        y_axis = float(k * x_axis + m)

        sliders["yellow" if color == "orange" else color] = {
            "pixel_xy": _map_point((x_axis, y_axis), (cw, ch), (ow, oh)),
            "reference_method": method,
            "approximate": is_approx,
        }

    complete = sliders["blue"] is not None and sliders["yellow"] is not None

    return {
        "schema_version": 1,
        "profile": active_scene,
        "input_size_px": [ow, oh],
        "status": "complete" if complete else "partial_or_missing",
        "position_definition": "shaft_axis_intersection_with_image_reference_line",
        "pixel_coordinate_frame": "scene_upright",
        "rail_endpoints_px": ([_map_point(p, (cw, ch), (ow, oh)) for p in rail_endpoints]
                              if rail_endpoints is not None else None),
        "sliders": sliders,
    }


def detect_sliders(image_path, profile="ground", scene=None, already_rotated=False):
    """读取图片文件并返回识别结果字典。"""
    res = detect_frame(_read_image(image_path), profile=profile, scene=scene, already_rotated=already_rotated)
    res["source_image"] = str(image_path)
    return res


def _write_preview(image_path, output_path, result, already_rotated=False):
    image = _read_image(image_path)
    cfg = SCENES[result["profile"]]
    rotation = 0 if already_rotated else cfg["rotation_deg"]
    if rotation == 180:
        image = cv2.rotate(image, cv2.ROTATE_180)

    endpoints = result.get("rail_endpoints_px")
    if endpoints:
        cv2.line(image, tuple(round(x) for x in endpoints[0]), tuple(round(x) for x in endpoints[1]), (0, 255, 0), 2)
    for ep, label in zip(endpoints or [], ("L", "R")):
        ep = tuple(round(x) for x in ep)
        cv2.circle(image, ep, 6, (0, 255, 255), -1)
        cv2.circle(image, ep, 6, (0, 140, 0), 2)
        cv2.putText(
            image,
            label,
            (ep[0] - 6, ep[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
        )

    for color in ("blue", "yellow"):
        item = result["sliders"].get(color)
        if item is None:
            continue
        ink = (255, 120, 0) if color == "blue" else (0, 180, 255)
        u, v = (round(n) for n in item["pixel_xy"])
        cv2.drawMarker(image, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(
            image,
            f"{color} ({item['pixel_xy'][0]:.1f}, {item['pixel_xy'][1]:.1f})",
            (u + 8, max(20, v - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            ink,
            2,
        )
    suffix = Path(output_path).suffix.lower()
    if not suffix:
        raise ValueError("Preview path requires an image extension, e.g. .jpg or .png")
    success, encoded = cv2.imencode(suffix, image)
    if not success:
        raise ValueError(f"Cannot encode preview: {output_path}")
    Path(output_path).write_bytes(encoded.tobytes())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", help="Input image path")
    parser.add_argument(
        "--scene",
        "--profile",
        dest="scene",
        choices=tuple(SCENES),
        default="ground",
        help="Scene type: 'ground' (ground simulation) or 'real' (real target, rotated 180 deg internally)",
    )
    parser.add_argument(
        "--already-rotated",
        action="store_true",
        help="Set if the input real-scene image has already been rotated 180 degrees beforehand",
    )
    parser.add_argument("--output", help="JSON output file path (default: stdout)")
    parser.add_argument("--preview", help="Optional annotated preview image path")
    args = parser.parse_args(argv)

    try:
        input_path = Path(args.image).resolve()
        output_paths = [Path(p).resolve() for p in (args.output, args.preview) if p]
        if any(path == input_path or (path.exists() and path.samefile(input_path)) for path in output_paths):
            raise ValueError("Output path must not overwrite the input image")
        if len(output_paths) != len(set(output_paths)):
            raise ValueError("JSON output and preview must use different paths")

        result = detect_sliders(args.image, profile=args.scene, already_rotated=args.already_rotated)
        if args.preview:
            _write_preview(args.image, args.preview, result, already_rotated=args.already_rotated)

        payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output:
            Path(args.output).write_text(payload, encoding="utf-8")
        else:
            sys.stdout.write(payload)

        return 0 if result["status"] == "complete" else 1
    except (OSError, ValueError, TypeError, cv2.error) as error:
        print(f"detect_sliders: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
