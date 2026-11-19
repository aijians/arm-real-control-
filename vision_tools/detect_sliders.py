#!/usr/bin/env python3
"""蓝色滑杆与橙色(黄色)滑杆像素位置及导轨真实端点识别脚本（支持 ground 与 real 场景）。

核心算法与工程修复说明：
1. 滑块上下分界锚点引导 + Peak 1 / 最强峰搜索 + 共线 RANSAC 拟合导轨前挡边线（修复新地面与在轨多结构混入问题）：
   - 先提取蓝、橙滑杆颜色掩膜，并以蓝色滑杆位置作为空间参考过滤背景黄胶带干扰；
   - 利用每个滑杆“上方圆柱杆 + 下方方形滑块”被白色导轨前挡边自然切开（或缩颈）的几何结构，
     提取导轨前挡边穿过滑杆处的高度锚点 (x_anchor, y_lip_anchor)；
   - 在锚点约束的合法斜率走廊 [0.15, 0.72] 内：对地面视角（y_anchor >= 295 px）自上而下寻找第一个
     “无色差明亮前挡边 -> 暗槽”垂直梯度峰值（Peak 1），彻底排除左下方支架底边（v≈450）、槽内后壁台阶
     与右上方背景强反光（v≈140..200）；对在轨仰视视角（y_anchor < 295 px）提取走廊内最强前挡边梯度峰；
   - 对候选边缘点执行共线 RANSAC（内点阈值 2.5 px）与 Huber 加权最小二乘精化，杜绝孤立异物边缘拉偏斜率。
2. 一维剖面水平平滑与物理端点校验（_detect_rail_endpoints）：
   - 在一维边缘剖面中融合“明亮无色差挡边 -> 暗槽”与“明亮无色差挡边 -> 高饱和度彩色滑块”两类过渡，
     防止亮橙色滑块位于导轨右端时截断导轨端点剖面；
   - 保留 ROI 截断边界检查与最小有效段长度检查。
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
        "roi": (260, 65, 700, 620),
        "rail_x_range": (445, 680),
        "rail_nominal": (465.0, 458.0, 0.225),
        "rail_band_px": 26,
        "max_slope_deviation": 0.025,
        "lip_split_half_width": 6,
        "endpoint_edge_thresh": 28.0,
        "min_endpoint_span_px": 150,
        # 杆轴测量带：导轨前挡边线以上 d1..d2 px（避开轴环/挡边，及机械臂从上方遮挡的部分）
        "rod_band_px": (20, 70),
        # 两杆在导轨上的先后顺序不变，相机侧前后遮挡关系也固定：蓝杆在前
        "front_color": "blue",
        "occlusion_gap_px": 8,
        "min_rod_rows": 8,
        # 杆宽模型 w(u)=a+b*u（像素，u 为杆轴像素 x）
        "width_model_px": {"blue": (119.438, -0.14111), "yellow": (144.583, -0.17082)},
        "width_tolerance_ratio": 0.2,
        "min_visible_width_ratio": 0.5,
    },
    "real": {
        "canonical_size": (1920, 1080),
        "rotation_deg": 180,
        "roi": (380, 65, 700, 520),
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


def _build_color_masks(patch, *, recover_shaded_blue=False):
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
    if recover_shaded_blue:
        # The registered ROI constrains the rail view. Keep blue silhouette
        # pixels at the shaded flank that the historical high-contrast mask
        # drops while retaining the hue and chroma tests at the background.
        is_core_blue |= (
            (hsv[:, :, 0] >= 99) & (hsv[:, :, 0] <= 132) &
            (hsv[:, :, 1] >= 63) & (hsv[:, :, 2] >= 65) &
            (b - g >= 24) & (b - r >= 30) & (lab[:, :, 2] <= 113)
        )
    blue_mask = (is_core_blue | is_specular_blue).astype(np.uint8) * 255

    orange_mask = (
        (hsv[:, :, 0] >= 3) & (hsv[:, :, 0] <= 38) &
        (hsv[:, :, 1] >= 38) & (hsv[:, :, 2] >= 125) &
        ((r - b) >= 35) & ((g - b) >= 12) &
        (lab[:, :, 2] >= 136)
    ).astype(np.uint8) * 255

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    blue_mask = cv2.morphologyEx(cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, k_open), cv2.MORPH_CLOSE, k_close)
    orange_mask = cv2.morphologyEx(cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, k_open), cv2.MORPH_CLOSE, k_close)
    return {"blue": blue_mask, "orange": orange_mask}


def _estimate_slider_split_anchors(masks, x1, y1):
    """提取蓝、橙滑杆被导轨前挡边切分（或缩颈）处的高度锚点 (x_center, y_lip_est)。"""
    anchors = []
    blue_ref_y = None
    blue_ref_x = None
    for c_key in ("blue", "orange"):
        mask = masks[c_key]
        num, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        comps = []
        for i in range(1, num):
            bx, by, bw, bh, area = stats[i]
            if area < 110 or min(bw, bh) < 5:
                continue
            gx1, gy1, gx2, gy2 = bx + x1, by + y1, bx + x1 + bw, by + y1 + bh
            if gx2 < 380 or gx1 > 700:
                continue
            if c_key == "orange" and blue_ref_y is not None:
                yc = 0.5 * (gy1 + gy2)
                xc = 0.5 * (gx1 + gx2)
                if xc < blue_ref_x - 20.0 or not (blue_ref_y - 120.0 <= yc <= blue_ref_y + 180.0):
                    continue
            comps.append((area, gx1, gy1, gx2, gy2, i))
        if not comps:
            continue
        comps.sort(reverse=True)
        seed = comps[0]
        sx1, sy1, sx2, sy2 = seed[1], seed[2], seed[3], seed[4]
        cluster = [seed]
        for c in comps[1:]:
            cx1, cy1, cx2, cy2 = c[1], c[2], c[3], c[4]
            h_ov = min(sx2, cx2) - max(sx1, cx1)
            if h_ov >= 0.25 * min(sx2 - sx1, cx2 - cx1) and abs(0.5 * (cy1 + cy2) - 0.5 * (sy1 + sy2)) <= 165.0:
                cluster.append(c)
        total_top = min(c[2] for c in cluster)
        total_bot = max(c[4] for c in cluster)
        xc = 0.5 * (seed[1] + seed[3])
        if len(cluster) >= 2:
            cluster.sort(key=lambda c: c[2])
            upper = cluster[0]
            lower = cluster[-1]
            if lower[2] >= upper[4] - 12 and (lower[4] - upper[2]) >= 75:
                xc = 0.5 * (max(upper[1], lower[1]) + min(upper[3], lower[3]))
                y_lip = float(lower[2] - 2.0)
                anchors.append((float(xc), y_lip))
                if c_key == "blue":
                    blue_ref_y, blue_ref_x = y_lip, xc
                continue
        if total_bot - total_top >= 135:
            sub_m = mask[int(total_top - y1) : int(total_bot - y1), int(sx1 - x1) : int(sx2 - x1)]
            h_sub = sub_m.shape[0]
            r_start, r_end = int(0.35 * h_sub), int(0.75 * h_sub)
            row_counts = (sub_m[r_start:r_end] > 0).sum(axis=1)
            pinch_row = r_start + int(np.argmin(row_counts))
            y_lip = float(total_top + pinch_row + 4.0)
        else:
            y_lip = float(total_bot + 1.0)
        anchors.append((float(xc), y_lip))
        if c_key == "blue":
            blue_ref_y, blue_ref_x = y_lip, xc
    return anchors


def _detect_rail_line(work_img, cfg, masks=None):
    """在滑块分界锚点约束的导轨走廊内提取外侧前挡边下沿直线 y = k * x + m；失败返回 (None, None)。"""
    gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_smooth = cv2.GaussianBlur(gray, (9, 5), 2.0, 1.0)
    dy = cv2.Sobel(gray_smooth, cv2.CV_32F, 0, 1, ksize=3)
    chroma = work_img.max(axis=2).astype(np.float32) - work_img.min(axis=2).astype(np.float32)

    x1, y1, x2, y2 = cfg["roi"]
    if masks is None:
        masks = _build_color_masks(work_img[y1:y2, x1:x2])
    slider_occ = cv2.dilate(masks["blue"] | masks["orange"], np.ones((19, 19), np.uint8))

    anchors = _estimate_slider_split_anchors(masks, x1, y1)
    x0, y0, k_nom = cfg["rail_nominal"]
    half_band = cfg["rail_band_px"]
    h = work_img.shape[0]
    k_min, k_max = 0.15, 0.72

    # 判断是否为地面俯视槽视角（y_anchor >= 295 px）或在轨仰视底面视角（y_anchor < 295 px）
    mean_anchor_y = float(np.mean([ya for _, ya in anchors])) if anchors else y0
    is_ground_view = mean_anchor_y >= 295.0
    x_start, x_end = (445, 680) if is_ground_view else (465, 680)

    xs, ys, weights = [], [], []
    for x in range(x_start, x_end):
        if is_ground_view and anchors:
            lows, highs = [], []
            for xa, ya in anchors:
                dx = x - xa
                if dx >= 0:
                    lows.append(ya + k_min * dx - 22.0)
                    highs.append(ya + k_max * dx + 22.0)
                else:
                    lows.append(ya + k_max * dx - 22.0)
                    highs.append(ya + k_min * dx + 22.0)
            by1 = max(12, int(round(max(lows))))
            by2 = min(h - 12, int(round(min(highs))))
        else:
            # 在轨视角使用 real 名义走廊 (480, 168, 0.650) ± 30 px
            rx0, ry0, rkn = (480.0, 168.0, 0.650) if not is_ground_view else (x0, y0, k_nom)
            rband = 30 if not is_ground_view else half_band
            yn = ry0 + rkn * (x - rx0)
            by1 = max(8, int(round(yn - rband)))
            by2 = min(h - 8, int(round(yn + rband)))

        if by2 - by1 < 3:
            continue

        col_edge = -dy[by1:by2, x]
        chosen_idx = None
        chosen_strength = 0.0
        if is_ground_view:
            # 地面视角：自上而下寻找走廊内第一个“无色差明亮前挡边 -> 暗槽”梯度峰值（Peak 1）
            for idx in range(1, len(col_edge) - 1):
                strength = float(col_edge[idx])
                if strength < 25.0 or strength < col_edge[idx - 1] or strength < col_edge[idx + 1]:
                    continue
                yg = by1 + idx
                if y1 <= yg < y2 and x1 <= x < x2 and slider_occ[yg - y1, x - x1] > 0:
                    continue
                above_g = gray_smooth[yg - 7 : yg - 1, x]
                below_g = gray_smooth[yg + 1 : yg + 6, x]
                above_c = chroma[yg - 7 : yg, x]
                if (len(above_g) > 0 and len(below_g) > 0
                        and float(np.mean(above_g)) >= 125.0
                        and float(np.min(above_g)) >= 100.0
                        and float(np.mean(above_g)) - float(np.min(below_g)) >= 25.0
                        and float(np.min(below_g)) <= 145.0
                        and float(np.mean(above_c)) <= 48.0):
                    chosen_idx = idx
                    chosen_strength = strength
                    break
        else:
            # 在轨视角：选取未遮挡列中的最强前挡边下沿梯度峰
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

    if is_ground_view:
        # 共线 RANSAC 筛选同一条实体前挡边上的内点，彻底剔除残余异物边缘点
        best_inliers = None
        best_score = -1.0
        n = len(xs_arr)
        for i in range(n):
            for j in range(i + 1, n):
                dx = xs_arr[j] - xs_arr[i]
                if abs(dx) < 25.0:
                    continue
                k_c = (ys_arr[j] - ys_arr[i]) / dx
                if not (k_min <= k_c <= k_max):
                    continue
                m_c = ys_arr[i] - k_c * xs_arr[i]
                resid = np.abs(ys_arr - (k_c * xs_arr + m_c))
                inliers = resid <= 2.5
                if np.count_nonzero(inliers) < 12:
                    continue
                score = float(w_arr[inliers].sum())
                if score > best_score:
                    best_score = score
                    best_inliers = inliers

        if best_inliers is None:
            return None, None
        xs_in = xs_arr[best_inliers]
        ys_in = ys_arr[best_inliers]
        w_in = w_arr[best_inliers]
    else:
        xs_in, ys_in, w_in = xs_arr, ys_arr, w_arr

    k, m = np.polyfit(xs_in, ys_in, 1, w=w_in)
    for _ in range(5):
        residual = np.abs(ys_in - (k * xs_in + m))
        w_robust = w_in / np.maximum(1.0, residual / 1.5)
        k, m = np.polyfit(xs_in, ys_in, 1, w=w_robust)

    if not is_ground_view:
        k_real_nom = 0.650
        max_k_dev = 0.045
        if abs(k - k_real_nom) > max_k_dev or (xs_in.max() - xs_in.min()) < 70.0:
            k = float(np.clip(k, k_real_nom - max_k_dev, k_real_nom + max_k_dev))
            m = float(np.median(ys_in - k * xs_in))
    elif mean_anchor_y >= 420.0:
        max_k_dev = float(cfg.get("max_slope_deviation", 0.025))
        if abs(k - k_nom) > max_k_dev or (xs_in.max() - xs_in.min()) < 70.0:
            k = float(np.clip(k, k_nom - max_k_dev, k_nom + max_k_dev))
            m = float(np.median(ys_in - k * xs_in))

    if not (k_min <= k <= k_max):
        return None, None

    return float(k), float(m)


def _detect_rail_endpoints(work_img, k, m, cfg):
    """沿已拟合的导轨直线 y = k * x + m 扫描一维梯度剖面，亚像素定位真实左、右物理端点；不可靠时返回 None。"""
    if k is None or m is None:
        return None

    gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_smooth = cv2.GaussianBlur(gray, (9, 5), 2.0, 1.0)
    neg_dy = -cv2.Sobel(gray_smooth, cv2.CV_32F, 0, 1, ksize=3)
    chroma = work_img.max(axis=2).astype(np.float32) - work_img.min(axis=2).astype(np.float32)

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
            above_med = float(np.median(gray_smooth[ypeak - 6 : ypeak - 1, x]))
            below_med = float(np.median(gray_smooth[ypeak + 3 : ypeak + 10, x]))
            below_chr = float(np.median(chroma[ypeak + 3 : ypeak + 12, x]))
            # 要求边沿下方进入暗槽，或进入高色度滑块区（防止亮橙色滑块截断右端点）
            if above_med - below_med >= 15.0 or (above_med >= 130.0 and below_chr >= 45.0):
                e_prof[i] = max(float(col[idx]), 32.0 if below_chr >= 45.0 else 0.0)

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
    """返回测量带内每行杆轮廓 (y, 左边缘, 右边缘, 是否触及ROI左/右边界)，像素为全图坐标。"""
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


def _measure_rod(rows, out_key, cfg, near_side_occluded=None, *, use_tangent=False):
    """由测量带内逐行轮廓求杆轴像素 x，并给出可靠性判断。"""
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

    visible = rights if near_side_occluded == "left" else lefts
    if use_tangent:
        # Mask positions are pixel centres. The silhouette is at the outer
        # boundary of the last covered pixel, on the side away from the blocker.
        side = "right" if near_side_occluded == "left" else "left"
        offset = 0.5 if side == "right" else -0.5
        info.update(method="rod_band_far_edge_tangent", visible_side=side,
                    edge_pixels=[[float(x + offset), row[0]] for x, row in zip(visible, usable)])
        return mid, True, reasons, info

    far = _trimmed_mean(visible)
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
    if width > model * (1.0 + tol):
        reliable = False
        reasons.append("rod_width_inconsistent_with_model")
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
        if f[1] < r[1]:
            gaps_left.append(r[1] - f[2])
        else:
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


def _needs_180_rotation(image_1080, roi, already_rotated):
    """若输入未声明已转正，检查正向 ROI 与倒装对应区域中的蓝色滑杆掩膜，自动兼容倒装原图与正向图。"""
    if already_rotated:
        return False
    x1, y1, x2, y2 = roi
    up_blue = int(np.count_nonzero(_build_color_masks(image_1080[y1:y2, x1:x2])["blue"]))
    rot_img = cv2.rotate(image_1080, cv2.ROTATE_180)
    rot_blue = int(np.count_nonzero(_build_color_masks(rot_img[y1:y2, x1:x2])["blue"]))
    return rot_blue > up_blue and rot_blue >= 400


def detect_frame(image, profile="ground", scene=None, already_rotated=False, *,
                 use_tangent=False, reference_rail=None):
    """识别单帧 BGR 图像中的导轨真实两端点及蓝色、橙(黄)色滑杆坐标。"""
    active_profile = scene if scene is not None else profile
    if active_profile not in SCENES:
        raise ValueError(f"Unknown profile {active_profile!r}; choose from {tuple(SCENES)}")
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3 or image.size == 0):
        raise ValueError("Frame must be a non-empty uint8 3-channel BGR image")

    base_cfg = SCENES[active_profile]
    cw, ch = base_cfg["canonical_size"]
    oh, ow = image.shape[:2]
    if abs(ow * ch - oh * cw) > max(cw, ch):
        raise ValueError(
            f"Image aspect ratio {ow}x{oh} does not match profile {active_profile} ({cw}x{ch}); "
            "use a full-frame proportional resize."
        )

    work = cv2.resize(image, (cw, ch), interpolation=cv2.INTER_LINEAR) if (ow, oh) != (cw, ch) else image.copy()
    if not already_rotated and (base_cfg["rotation_deg"] == 180 or _needs_180_rotation(work, base_cfg["roi"], False)):
        work = cv2.rotate(work, cv2.ROTATE_180)

    # 1. 提取 ROI 内的蓝、橙颜色掩膜
    x1, y1, x2, y2 = base_cfg["roi"]
    patch = work[y1:y2, x1:x2]
    masks = _build_color_masks(patch)

    # 2. 结合滑块分界锚点、Peak 1 / 最强峰与共线 RANSAC 拟合导轨前挡边下沿直线 y = k * x + m
    if reference_rail is not None:
        k, m = reference_rail
        rail_line_source = "registered_reference"
    else:
        k, m = _detect_rail_line(work, base_cfg, masks=masks)
        rail_line_source = "fitted"
        if k is None or m is None:
            x0, y0, k_nom = base_cfg["rail_nominal"]
            k, m = float(k_nom), float(y0 - k_nom * x0)
            rail_line_source = "nominal_fallback"

    # 根据实测导轨在转正画面中的垂直高度自适应选择高斜率/重装参数或传统正向地面参数
    y_mid = k * 550.0 + m
    if y_mid <= 420.0 or k >= 0.30:
        cfg = dict(SCENES["real"])
        if y_mid < 275.0:
            cfg["roi"] = (420, 80, 700, 365)
    else:
        cfg = dict(SCENES["ground"])

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
    band = cfg.get("rod_band_px", (40, 80))

    parts = {}
    blue_x_center = None
    for mask_key, out_key in (("blue", "blue"), ("orange", "yellow")):
        mask = masks[mask_key]
        upper_raw = np.where(gy < rail_y_grid - half_lip, mask, 0).astype(np.uint8)
        upper_raw = np.where(gy >= rail_y_grid - 175.0, upper_raw, 0).astype(np.uint8)
        if out_key == "yellow" and blue_x_center is not None:
            upper_raw = np.where(gx >= blue_x_center - 20.0, upper_raw, 0).astype(np.uint8)
        upper_mask = cv2.morphologyEx(upper_raw, cv2.MORPH_CLOSE, k_rod_horiz_close)
        lower_raw = np.where((gy > rail_y_grid + half_lip) & (gy <= rail_y_grid + 135.0), mask, 0).astype(np.uint8)
        if out_key == "yellow" and blue_x_center is not None:
            lower_raw = np.where(gx >= blue_x_center - 20.0, lower_raw, 0).astype(np.uint8)
        lower_mask = cv2.morphologyEx(lower_raw, cv2.MORPH_OPEN, k_block_open)
        block = _extract_merged_component(lower_mask, (x1, y1), min_area=200)
        ref_x = (block["bbox"][0], block["bbox"][2]) if block is not None else None
        rod = _extract_merged_component(upper_mask, (x1, y1), min_area=120, ref_x_range=ref_x)
        if reference_rail is not None and out_key == "blue" and rod is not None:
            # Expand only near an existing shaft component. The full image can
            # contain blue wiring or reflection outside the selected rod.
            center = rod["center"][0]
            width = _width_model(cfg, out_key, center)
            if width is not None:
                shaded = _build_color_masks(patch, recover_shaded_blue=True)["blue"]
                region = (upper_raw > 0) | ((np.abs(gx - center) <= .65 * width)
                    & (gy < rail_y_grid - half_lip) & (gy >= rail_y_grid - 175.0))
                expanded = np.where(region, shaded, 0).astype(np.uint8)
                expanded = cv2.morphologyEx(expanded, cv2.MORPH_CLOSE, k_rod_horiz_close)
                candidate = _extract_merged_component(expanded, (x1, y1), min_area=120, ref_x_range=ref_x)
                if candidate is not None and abs(candidate["center"][0] - center) <= .3 * width:
                    candidate_rows = _rod_band_rows(candidate, (x1, y1), k, m, band, x2 - x1)
                    if _measure_rod(candidate_rows, out_key, cfg)[1]:
                        rod = candidate
        if out_key == "blue" and (rod is not None or block is not None):
            src = rod if rod is not None else block
            blue_x_center = src["center"][0]
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
            x_axis, reliable, reasons, info = _measure_rod(rows, out_key, cfg, near, use_tangent=use_tangent)
            if (x_axis is not None and near is None and out_key != front
                    and "rod_width_inconsistent_with_model" in reasons):
                near_wide = _near_side_contact(rows, other_rows, 3.0 * gap)
                if near_wide is not None and info.get("width_px") and info.get("width_model_px") \
                        and info["width_px"] < info["width_model_px"]:
                    x_axis, reliable, reasons, info = _measure_rod(
                        rows, out_key, cfg, near_wide, use_tangent=use_tangent)
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
        tangent = method == "rod_band_far_edge_tangent"
        if tangent:
            info["edge_pixels"] = [_map_point(point, (cw, ch), (ow, oh))
                                   for point in info["edge_pixels"]]
        sliders[out_key] = {
            "pixel_xy": None if tangent else _map_point((x_axis, y_axis), (cw, ch), (ow, oh)),
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
    cw, ch = cfg["canonical_size"]
    oh, ow = image.shape[:2]
    work_check = cv2.resize(image, (cw, ch), interpolation=cv2.INTER_LINEAR) if (ow, oh) != (cw, ch) else image
    if not already_rotated and (cfg["rotation_deg"] == 180 or _needs_180_rotation(work_check, cfg["roi"], False)):
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