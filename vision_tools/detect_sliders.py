#!/usr/bin/env python3
"""蓝色滑杆与橙色(黄色)滑杆像素位置及导轨真实端点识别脚本（支持 ground 与 real 场景）。

核心算法与工程修复说明：
1. 先掩膜屏蔽滑杆、再拟合外侧前挡边下沿直线（修复初始姿态斜率 0.333 偏斜问题）：
   - 先提取蓝、橙滑杆颜色掩膜并做形态学膨胀，在拟合导轨直线时彻底剔除滑杆遮挡列，
     防止左端蓝杆轴环与右端黄杆方块把导轨直线斜率拉偏；
   - 在未遮挡列中自上而下寻找第一个（最上层）“无色差灰白前挡边 -> 暗槽”的垂直梯度峰值（Peak 1），
     严格锁定贯通全长的外侧前挡边下沿，避免误锁低 13~15 px 的槽内后壁反光台阶（Peak 2）；
   - 引入基于相机-导轨刚性安装约束的斜率一致性护栏（|k - k_nom| <= max_slope_deviation），
     当遮挡导致有效跨度较短或斜率异常时自动约束斜率并稳健重估截距 m；若有效点不足则返回 None。
2. 一维剖面水平平滑与物理端点校验（_detect_rail_endpoints）：
   - 使用 (7, 1) 高斯核沿一维剖面水平长度方向平滑；
   - 校验边缘下方确实进入暗槽/滑杆区（排除左侧 x < 446 实心灰色三角斜撑水平面）；
   - 恢复 ROI 截断边界检查（idx_l == 0 或 idx_r == len - 1 时返回 None）与最小有效段长度检查，
     绝不以先验扫描边界冒充实测物理端点。
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
        "rail_x_range": (445, 672),
        "rail_nominal": (465.0, 458.0, 0.225),
        "rail_band_px": 26,
        "max_slope_deviation": 0.025,
        "lip_split_half_width": 6,
        "endpoint_edge_thresh": 28.0,
        "min_endpoint_span_px": 180,
        # 杆轴测量带：导轨前挡边线以上 d1..d2 px（避开轴环/挡边，及机械臂从上方遮挡的部分）
        "rod_band_px": (20, 70),
        # 两杆在导轨上的先后顺序不变，相机侧前后遮挡关系也固定：蓝杆在前
        "front_color": "blue",
        "occlusion_gap_px": 8,
        "min_rod_rows": 8,
        # 杆宽模型 w(u)=a+b*u（像素，u 为杆轴像素 x）。2026-11-15 由 8ddc48b7/01f43e83/a63369c4
        # 三段地面录像经 eval_slider_detection.py calibrate-width（位置区间上分位数+RANSAC）拟合。
        "width_model_px": {"blue": (119.438, -0.14111), "yellow": (144.583, -0.17082)},
        "width_tolerance_ratio": 0.2,
        "min_visible_width_ratio": 0.5,
    },
    "real": {
        "canonical_size": (1920, 1080),
        "rotation_deg": 180,
        "roi": (420, 80, 700, 365),
        "rail_x_range": (465, 680),
        "rail_nominal": (480.0, 168.0, 0.650),
        "rail_band_px": 30,
        "max_slope_deviation": 0.045,
        "lip_split_half_width": 6,
        "endpoint_edge_thresh": 28.0,
        "min_endpoint_span_px": 150,
        "rod_band_px": (40, 80),
        "front_color": "blue",
        "occlusion_gap_px": 8,
        "min_rod_rows": 8,
        # 2026-11-15 由在轨 init.mp4/pushed_to_right.mp4 拟合（同上方法）。
        "width_model_px": {"blue": (124.743, -0.14575), "yellow": (134.459, -0.1571)},
        "width_tolerance_ratio": 0.2,
        "min_visible_width_ratio": 0.65,
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


def _detect_rail_line(work_img, cfg, masks=None):
    """在剔除滑杆遮挡后的导轨走廊内提取外侧前挡边下沿直线 y = k * x + m；失败返回 (None, None)。"""
    gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_smooth = cv2.GaussianBlur(gray, (9, 5), 2.0, 1.0)
    dy = cv2.Sobel(gray_smooth, cv2.CV_32F, 0, 1, ksize=3)
    chroma = work_img.max(axis=2).astype(np.float32) - work_img.min(axis=2).astype(np.float32)

    x1, y1, x2, y2 = cfg["roi"]
    if masks is None:
        masks = _build_color_masks(work_img[y1:y2, x1:x2])
    slider_occ = cv2.dilate(masks["blue"] | masks["orange"], np.ones((19, 19), np.uint8))

    x_start, x_end = cfg["rail_x_range"]
    x0, y0, k_nom = cfg["rail_nominal"]
    half_band = cfg["rail_band_px"]
    h = work_img.shape[0]
    is_ground = (cfg.get("rotation_deg", 0) == 0 and y0 > 350)

    xs, ys, weights = [], [], []
    for x in range(x_start, x_end):
        yn = y0 + k_nom * (x - x0)
        by1 = max(8, int(round(yn - half_band)))
        by2 = min(h - 8, int(round(yn + half_band)))
        col_edge = -dy[by1:by2, x]
        if len(col_edge) < 3:
            continue

        chosen_idx = None
        chosen_strength = 0.0
        if is_ground:
            # 自上而下寻找第一个“无色差明亮前挡边 -> 暗槽”梯度峰值（外侧前挡边下沿 Peak 1），
            # 避免跳到下方 13~15 px 的槽内后壁反光台阶（Peak 2）或滑杆边缘。
            for idx in range(1, len(col_edge) - 1):
                strength = float(col_edge[idx])
                if strength < 28.0 or strength < col_edge[idx - 1] or strength < col_edge[idx + 1]:
                    continue
                yg = by1 + idx
                if y1 <= yg < y2 and x1 <= x < x2 and slider_occ[yg - y1, x - x1] > 0:
                    continue
                above_g = gray_smooth[yg - 7 : yg - 1, x]
                below_g = gray_smooth[yg + 1 : yg + 6, x]
                above_c = chroma[yg - 7 : yg, x]
                if (len(above_g) > 0 and len(below_g) > 0
                        and float(np.mean(above_g)) >= 135.0
                        and float(np.min(above_g)) >= 110.0
                        and float(np.mean(above_g)) - float(np.min(below_g)) >= 30.0
                        and float(np.mean(above_c)) <= 45.0):
                    chosen_idx = idx
                    chosen_strength = strength
                    break
        else:
            for idx in np.argsort(col_edge)[::-1]:
                idx = int(idx)
                strength = float(col_edge[idx])
                if strength <= 25.0:
                    break
                yg = by1 + idx
                if y1 <= yg < y2 and x1 <= x < x2 and slider_occ[yg - y1, x - x1] > 0:
                    continue
                chosen_idx = idx
                chosen_strength = strength
                break

        if chosen_idx is not None:
            if 0 < chosen_idx < len(col_edge) - 1:
                ym, yc, yp = col_edge[chosen_idx - 1], col_edge[chosen_idx], col_edge[chosen_idx + 1]
                denom = ym - 2.0 * yc + yp
                offset = float(np.clip(0.5 * (ym - yp) / denom, -0.5, 0.5)) if abs(denom) > 1e-5 else 0.0
            else:
                offset = 0.0
            xs.append(float(x))
            ys.append(float(by1 + chosen_idx + offset))
            weights.append(chosen_strength)

    if len(xs) < 15:
        return None, None

    xs_arr = np.asarray(xs, dtype=np.float64)
    ys_arr = np.asarray(ys, dtype=np.float64)
    w_arr = np.asarray(weights, dtype=np.float64)

    k, m = np.polyfit(xs_arr, ys_arr, 1, w=w_arr)
    for _ in range(5):
        residual = np.abs(ys_arr - (k * xs_arr + m))
        w_robust = w_arr / np.maximum(1.0, residual / 1.5)
        k, m = np.polyfit(xs_arr, ys_arr, 1, w=w_robust)

    max_k_dev = float(cfg.get("max_slope_deviation", 0.025))
    if abs(k - k_nom) > max_k_dev or (xs_arr.max() - xs_arr.min()) < 70.0:
        k = float(np.clip(k, k_nom - max_k_dev, k_nom + max_k_dev))
        m = float(np.median(ys_arr - k * xs_arr))

    return float(k), float(m)


def _detect_rail_endpoints(work_img, k, m, cfg):
    """沿已拟合的导轨直线 y = k * x + m 扫描一维梯度剖面，亚像素定位真实左、右物理端点；不可靠时返回 None。"""
    if k is None or m is None:
        return None

    gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_smooth = cv2.GaussianBlur(gray, (9, 5), 2.0, 1.0)
    neg_dy = -cv2.Sobel(gray_smooth, cv2.CV_32F, 0, 1, ksize=3)

    x1, _, x2, _ = cfg["roi"]
    h = work_img.shape[0]
    xs = np.arange(x1, x2, dtype=np.int32)
    e_prof = np.zeros(len(xs), dtype=np.float32)
    for i, x in enumerate(xs):
        yr = int(round(k * x + m))
        by1 = max(8, yr - 6)
        by2 = min(h - 12, yr + 7)
        if by2 > by1:
            col = neg_dy[by1:by2, x]
            idx = int(np.argmax(col))
            ypeak = by1 + idx
            # 要求边沿下方确实比上方前挡边暗（过滤左侧实心灰色三角斜撑台面）
            above_med = float(np.median(gray_smooth[ypeak - 6 : ypeak - 1, x]))
            below_med = float(np.median(gray_smooth[ypeak + 3 : ypeak + 10, x]))
            if above_med - below_med >= 15.0:
                e_prof[i] = float(col[idx])

    # OpenCV GaussianBlur 的 ksize=(width, height)：(7, 1) 表示沿 1xN 剖面的水平长度方向平滑
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
    min_span = int(cfg.get("min_endpoint_span_px", 150))
    if len(nz) < min_span:
        return None

    splits = np.where(np.diff(nz) > 1)[0] + 1
    best_seg = max(np.split(nz, splits), key=len)
    if len(best_seg) < min_span:
        return None

    idx_l, idx_r = int(best_seg[0]), int(best_seg[-1])
    # 防止 ROI 截断边界被当成物理端点
    if idx_l <= 0 or idx_r >= len(xs) - 1:
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


def _rod_band_rows(rod, offset_xy, k, m, band, roi_width):
    """返回测量带内每行杆轮廓 (y, 左边缘, 右边缘, 是否触及ROI左/右边界)，像素为全图坐标。

    测量带按“导轨前挡边线以上 d1..d2 像素”取固定行，不随部件框高度变化，
    因此机械臂从上方遮挡杆顶、或导轨支架遮挡杆底时，所用行区间保持一致。
    """
    x1, y1 = offset_xy
    bx, by, bw, bh = rod["local_bbox"]
    d1, d2 = band
    rows = []
    for yy in range(by, by + bh):
        nz = np.flatnonzero(rod["mask"][yy])
        if len(nz) < 4:
            continue
        left, right = int(nz[0]), int(nz[-1])
        u = 0.5 * (left + right) + x1
        dy = k * u + m - (yy + y1)
        if d1 <= dy <= d2:
            rows.append((float(yy + y1), float(left + x1), float(right + x1),
                         left <= 0, right >= roi_width - 1))
    return rows


def _trimmed_mean(values, keep=0.5):
    """去掉两端各 (1-keep)/2 后求均值；逐行整像素边缘经平均后得到亚像素估计。"""
    arr = np.sort(np.asarray(values, dtype=np.float64))
    if arr.size == 0:
        return None
    cut = int(arr.size * (1.0 - keep) / 2.0)
    core = arr[cut: arr.size - cut] if arr.size - 2 * cut > 0 else arr
    return float(core.mean())


def _width_model(cfg, out_key, u):
    model = (cfg.get("width_model_px") or {}).get(out_key)
    if model is None:
        return None
    a, b = model
    width = float(a + b * u)
    return width if width > 4.0 else None


def _measure_rod(rows, out_key, cfg, near_side_occluded=None):
    """由测量带内逐行轮廓求杆轴像素 x，并给出可靠性判断。

    - 两侧边缘都可见：各行中点的截尾均值（与原逐行中点法一致，改为固定测量带）。
    - 前杆紧邻后杆一侧（near_side_occluded='left'/'right'）：该侧边缘被前杆遮挡，
      改用未被遮挡的远侧边缘减去半个模型杆宽。
    """
    min_rows = int(cfg.get("min_rod_rows", 8))
    tol = float(cfg.get("width_tolerance_ratio", 0.2))
    reasons = []
    usable = [r for r in rows if not (r[3] or r[4])]
    if len(usable) < len(rows):
        reasons.append("rod_touches_roi_boundary")
    info = {"rows_in_band": len(rows), "rows_used": 0, "occluded_side": near_side_occluded,
            "width_px": None, "width_model_px": None}
    if len(usable) < min_rows:
        reasons.append("too_few_rod_rows_in_band")
        return None, False, reasons, info

    lefts = np.asarray([r[1] for r in usable])
    rights = np.asarray([r[2] for r in usable])
    widths = rights - lefts + 1.0
    mid = _trimmed_mean(0.5 * (lefts + rights))
    width = _trimmed_mean(widths)
    info.update(rows_used=len(usable), width_px=round(width, 2))

    if near_side_occluded is None:
        model = _width_model(cfg, out_key, mid)
        info["width_model_px"] = None if model is None else round(model, 2)
        info["method"] = "rod_band_both_edges"
        reliable = True
        if model is not None and abs(width / model - 1.0) > tol:
            reliable = False
            reasons.append("rod_width_inconsistent_with_model")
        return mid, reliable, reasons, info

    # 近侧被前杆遮挡：只用远侧边缘。宽度模型依赖杆轴位置，迭代即可收敛。
    far = _trimmed_mean(rights if near_side_occluded == "left" else lefts)
    sign = -1.0 if near_side_occluded == "left" else 1.0
    axis = far + sign * 0.5 * (width - 1.0)
    model = None
    for _ in range(3):
        model = _width_model(cfg, out_key, axis)
        if model is None:
            break
        axis = far + sign * 0.5 * (model - 1.0)
    info["method"] = "rod_band_far_edge_width_model"
    if model is None:
        reasons.append("width_model_missing_for_occluded_rod")
        return axis, False, reasons, info
    info["width_model_px"] = round(model, 2)
    reliable = True
    # 可见宽度不可能明显大于整杆宽度；若大于则说明遮挡判断或分割有误。
    if width > model * (1.0 + tol):
        reliable = False
        reasons.append("rod_width_inconsistent_with_model")
    # 仅被前杆遮挡时可见比例基本恒定（地面约 0.63、在轨约 0.74）；明显更窄说明远侧边缘
    # 也被其它物体（如机械臂）遮挡，远侧边缘不可信。
    min_ratio = float(cfg.get("min_visible_width_ratio", 0.5))
    if width < model * min_ratio:
        reliable = False
        reasons.append("visible_width_too_small_for_far_edge")
    return axis, reliable, reasons, info


def _near_side_contact(back_rows, front_rows, gap_px):
    """判断前杆是否紧贴后杆一侧：取两杆在测量带重叠行的水平间隙中位数。"""
    if not back_rows or not front_rows:
        return None
    front = {r[0]: r for r in front_rows}
    gaps_left, gaps_right = [], []
    for r in back_rows:
        f = front.get(r[0])
        if f is None:
            continue
        if f[1] < r[1]:          # 前杆在左：前杆右边缘到后杆左边缘
            gaps_left.append(r[1] - f[2])
        else:                    # 前杆在右：后杆右边缘到前杆左边缘
            gaps_right.append(f[1] - r[2])
    if len(gaps_left) >= 3 and np.median(gaps_left) <= gap_px:
        return "left"
    if len(gaps_right) >= 3 and np.median(gaps_right) <= gap_px:
        return "right"
    return None


def _map_point(point, canonical_size, original_size):
    """将标准尺度坐标按比例映射回输入图分辨率（保持转正后的正向像素坐标系）。"""
    x, y = point
    cw, ch = canonical_size
    ow, oh = original_size
    return [
        round(float((x + 0.5) * ow / cw - 0.5), 2),
        round(float((y + 0.5) * oh / ch - 0.5), 2),
    ]


def detect_frame(image, profile="ground", scene=None, already_rotated=False):
    """识别单帧 BGR 图像中的导轨真实两端点及蓝色、橙(黄)色滑杆坐标。"""
    active_profile = scene if scene is not None else profile
    if active_profile not in SCENES:
        raise ValueError(f"Unknown profile {active_profile!r}; choose from {tuple(SCENES)}")
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3 or image.size == 0):
        raise ValueError("Frame must be a non-empty uint8 3-channel BGR image")

    cfg = SCENES[active_profile]
    cw, ch = cfg["canonical_size"]
    oh, ow = image.shape[:2]
    if abs(ow * ch - oh * cw) > max(cw, ch):
        raise ValueError(
            f"Image aspect ratio {ow}x{oh} does not match profile {active_profile} ({cw}x{ch}); "
            "use a full-frame proportional resize."
        )
    rotation = 0 if already_rotated else cfg["rotation_deg"]

    work = cv2.rotate(image, cv2.ROTATE_180) if rotation == 180 else image
    if (ow, oh) != (cw, ch):
        work = cv2.resize(work, (cw, ch), interpolation=cv2.INTER_LINEAR)

    # 1. 先提取 ROI 内的蓝、橙颜色掩膜，用于屏蔽滑杆遮挡后再拟合导轨直线
    x1, y1, x2, y2 = cfg["roi"]
    patch = work[y1:y2, x1:x2]
    masks = _build_color_masks(patch)

    # 2. 剔除滑杆遮挡列，拟合外侧前挡边下沿直线 y = k * x + m。
    #    该直线是杆轴取交点的图像参考线；拟合失败时回退到场景名义线（安装位置固定）。
    k, m = _detect_rail_line(work, cfg, masks=masks)
    rail_line_source = "fitted"
    if k is None or m is None:
        x0, y0, k_nom = cfg["rail_nominal"]
        k, m = float(k_nom), float(y0 - k_nom * x0)
        rail_line_source = "nominal_fallback"
    # 导轨端点仅作诊断输出，不参与行程范围或验收判定（行程范围改用模型固定值）。
    endpoints = _detect_rail_endpoints(work, k, m, cfg)
    rail_endpoints_px = (
        [_map_point(endpoints[0], (cw, ch), (ow, oh)),
         _map_point(endpoints[1], (cw, ch), (ow, oh))]
        if endpoints is not None else None
    )

    sliders = {"blue": None, "yellow": None}
    py, px = np.indices((y2 - y1, x2 - x1))
    gx, gy = px + x1, py + y1
    rail_y_grid = k * gx + m
    half_lip = cfg["lip_split_half_width"]
    k_rod_horiz_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    k_block_open = np.ones((7, 7), np.uint8)
    band = cfg.get("rod_band_px", (20, 70))

    parts = {}
    for mask_key, out_key in (("blue", "blue"), ("orange", "yellow")):
        mask = masks[mask_key]
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
        rows = _rod_band_rows(rod, (x1, y1), k, m, band, x2 - x1) if rod is not None else []
        parts[out_key] = (rod, block, rows)

    front = cfg.get("front_color", "blue")
    gap = float(cfg.get("occlusion_gap_px", 8))
    for out_key in ("blue", "yellow"):
        rod, block, rows = parts[out_key]
        if rod is None and block is None:
            continue
        other_rows = parts["yellow" if out_key == "blue" else "blue"][2]
        near = None if out_key == front else _near_side_contact(rows, other_rows, gap)

        reasons, info, reliable = [], {}, False
        x_axis = None
        if rod is not None:
            x_axis, reliable, reasons, info = _measure_rod(rows, out_key, cfg, near)
            if (x_axis is not None and near is None and out_key != front
                    and "rod_width_inconsistent_with_model" in reasons):
                # 前杆阴影侧常不在颜色掩膜内，间隙看似大于阈值；若后杆可见宽度偏窄且前杆
                # 就在附近（3 倍间隙内），按近侧被遮挡重新测量。
                near_wide = _near_side_contact(rows, other_rows, 3.0 * gap)
                if near_wide is not None and info.get("width_px") and info.get("width_model_px") \
                        and info["width_px"] < info["width_model_px"]:
                    x_axis, reliable, reasons, info = _measure_rod(rows, out_key, cfg, near_wide)
                    reasons = reasons + ["occlusion_inferred_from_narrow_width"]
        if x_axis is not None:
            method = info.get("method", "rod_band_both_edges")
            rep_part, is_approx = "rod", False
        else:
            source = block if block is not None else rod
            x_axis = source["center"][0]
            method = "block_center_projection_approximate"
            rep_part, is_approx, reliable = "block" if block is not None else "rod", True, False
            reasons = reasons + ["shaft_axis_not_measured"]

        y_axis = float(k * x_axis + m)
        sliders[out_key] = {
            "pixel_xy": _map_point((x_axis, y_axis), (cw, ch), (ow, oh)),
            "reference_method": method,
            "representative_part": rep_part,
            "approximate": is_approx,
            "reliable": bool(reliable),
            "reasons": reasons,
            "measurement": info,
        }

    complete = sliders["blue"] is not None and sliders["yellow"] is not None
    return {
        "schema_version": 1,
        "profile": active_profile,
        "input_size_px": [ow, oh],
        "status": "complete" if complete else "partial_or_missing",
        "position_definition": "shaft_axis_intersection_with_image_reference_line",
        "rail_line": {"k": round(float(k), 6), "m": round(float(m), 3), "source": rail_line_source},
        "rail_endpoints_px": rail_endpoints_px,
        "sliders": sliders,
    }


def detect_sliders(image_path, profile="ground", scene=None, already_rotated=False):
    """读取图片文件并返回识别结果字典。"""
    return detect_frame(_read_image(image_path), profile=profile, scene=scene, already_rotated=already_rotated)


def _write_preview(image_path, output_path, result, already_rotated=False):
    image = _read_image(image_path)
    cfg = SCENES[result["profile"]]
    rotation = 0 if already_rotated else cfg["rotation_deg"]
    if rotation == 180:
        image = cv2.rotate(image, cv2.ROTATE_180)

    eps = result.get("rail_endpoints_px")
    if isinstance(eps, (list, tuple)) and len(eps) == 2 and eps[0] is not None and eps[1] is not None:
        pt0 = (round(eps[0][0]), round(eps[0][1]))
        pt1 = (round(eps[1][0]), round(eps[1][1]))
        cv2.line(image, pt0, pt1, (0, 255, 0), 2)
        for ep, label in [(pt0, "L"), (pt1, "R")]:
            cv2.circle(image, ep, 6, (0, 255, 255), -1)
            cv2.circle(image, ep, 6, (0, 140, 0), 2)
            cv2.putText(image, label, (ep[0] - 6, ep[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

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
        "--profile",
        "--scene",
        dest="profile",
        choices=tuple(SCENES),
        default="ground",
        help="Profile type: 'ground' or 'real'",
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

        result = detect_sliders(args.image, profile=args.profile, already_rotated=args.already_rotated)
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