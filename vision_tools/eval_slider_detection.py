#!/usr/bin/env python3
"""离线评估滑杆识别（不访问相机/串口/运动），并可从无遮挡帧拟合杆宽模型。

输入源写成 `路径:场景[:rotated]`，路径可以是图片或录像；场景为 ground/real，
`rotated` 表示该源已转正（例如 *_rot180.mp4）。

  evaluate        逐帧识别，输出每帧 CSV 及汇总 JSON（橙/蓝像素与地面毫米统计、可靠帧数）。
  calibrate-width 从两侧边缘都可见的帧拟合 w(u)=a+b*u（蓝杆随推动覆盖多个位置），
                  橙杆按与蓝杆模型的比例缩放（同一相机下杆宽均与 u 线性相关）。

可用 --baseline DIR 指向旧版代码目录（含 vision_tools/），对同一帧给出旧算法结果对照。
"""
import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from vision_tools.detect_sliders import detect_frame, SCENES  # noqa: E402

SHOULDER_X_MM = 189.3
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _load_baseline(directory):
    path = Path(directory) / "vision_tools" / "detect_sliders.py"
    spec = importlib.util.spec_from_file_location("baseline_detect_sliders", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.detect_frame


def _parse_source(text):
    parts = text.rsplit(":", 2)
    if len(parts) >= 2 and parts[-1] == "rotated":
        path, profile, rotated = ":".join(parts[:-2]), parts[-2], True
    else:
        path, profile = text.rsplit(":", 1)
        rotated = False
    if profile not in SCENES:
        raise ValueError(f"unknown profile in source {text!r}")
    return Path(path), profile, rotated


def _frames(path, step, limit):
    if path.suffix.lower() in IMAGE_SUFFIXES:
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot decode {path}")
        yield 0, image
        return
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open {path}")
    index = 0
    emitted = 0
    try:
        while True:
            ok = capture.grab()
            if not ok:
                break
            if index % step == 0:
                ok, image = capture.retrieve()
                if ok:
                    yield index, image
                    emitted += 1
                    if limit and emitted >= limit:
                        break
            index += 1
    finally:
        capture.release()


def _ground_mm(pixel_xy):
    from vision_tools.pixel_to_world import pixel_to_world
    try:
        return pixel_to_world(pixel_xy)["world_xyz_mm"][0] - SHOULDER_X_MM
    except ValueError:
        return None


def _row(profile, result, prefix=""):
    row = {}
    for key in ("blue", "yellow"):
        item = result["sliders"].get(key)
        name = prefix + key
        if item is None:
            row.update({name + "_u": None, name + "_mm": None, name + "_reliable": None,
                        name + "_method": None, name + "_width": None, name + "_reasons": None})
            continue
        u = item["pixel_xy"][0]
        measurement = item.get("measurement", {})
        mm = _ground_mm(item["pixel_xy"]) if profile == "ground" else None
        row.update({
            name + "_u": round(u, 3),
            name + "_mm": None if mm is None else round(mm, 4),
            name + "_reliable": item.get("reliable", not item.get("approximate", False)),
            name + "_method": item.get("reference_method"),
            name + "_width": measurement.get("width_px"),
            name + "_reasons": ";".join(item.get("reasons", [])),
        })
    return row


def _stats(values):
    arr = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    if not arr.size:
        return None
    return {"n": int(arr.size), "min": round(float(arr.min()), 3), "max": round(float(arr.max()), 3),
            "range": round(float(arr.max() - arr.min()), 3), "std": round(float(arr.std()), 3),
            "p05": round(float(np.percentile(arr, 5)), 3), "p95": round(float(np.percentile(arr, 95)), 3)}


def evaluate(args):
    baseline = _load_baseline(args.baseline) if args.baseline else None
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for text in args.sources:
        path, profile, rotated = _parse_source(text)
        rows = []
        for frame_index, image in _frames(path, args.step, args.limit):
            row = {"source": path.name, "profile": profile, "frame": frame_index}
            row.update(_row(profile, detect_frame(image, profile=profile, already_rotated=rotated)))
            if baseline is not None:
                row.update(_row(profile, baseline(image, profile=profile, already_rotated=rotated), "old_"))
            rows.append(row)
        if not rows:
            continue
        with (out_dir / (path.stem + ".csv")).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        unit = "mm" if profile == "ground" else "u"
        entry = {"profile": profile, "frames": len(rows), "unit": unit}
        for key in ("blue", "yellow"):
            reliable = [r for r in rows if r.get(key + "_reliable")]
            entry[key] = {
                "detected": sum(r.get(key + "_u") is not None for r in rows),
                "reliable": len(reliable),
                "methods": {m: sum(r.get(key + "_method") == m for r in rows)
                            for m in sorted({r.get(key + "_method") for r in rows} - {None})},
                "reliable_" + unit: _stats([r.get(key + "_" + unit) for r in reliable]),
                "all_" + unit: _stats([r.get(key + "_" + unit) for r in rows]),
            }
            if baseline is not None:
                entry[key]["old_all_" + unit] = _stats([r.get("old_" + key + "_" + unit) for r in rows])
        summary[path.name] = entry
        print(json.dumps({path.name: entry}, ensure_ascii=False), flush=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")


def _upper_envelope(u, w, bin_px=5.0, quantile=85.0, min_count=3):
    """遮挡（机械臂、相邻杆）只会让可见杆宽变窄，故每个位置区间取上分位数代表真实杆宽。"""
    u, w = np.asarray(u, float), np.asarray(w, float)
    bins = np.floor(u / bin_px).astype(int)
    points = []
    for value in np.unique(bins):
        sel = bins == value
        if sel.sum() >= min_count:
            points.append((float(np.median(u[sel])), float(np.percentile(w[sel], quantile)),
                           int(sel.sum())))
    return np.asarray(points, float).reshape(-1, 3)


def _robust_line(u, w, weight=None, inlier_px=2.0):
    """对位置区间做穷举两点 RANSAC（按样本数加权计内点），再用内点加权最小二乘精化。

    运动中机械臂部分遮挡会产生偏窄、偏移的区间，迭代最小二乘会被其拉偏。
    """
    u, w = np.asarray(u, float), np.asarray(w, float)
    weight = np.ones(u.size) if weight is None else np.asarray(weight, float)
    best, best_score = None, -1.0
    for i in range(u.size):
        for j in range(i + 1, u.size):
            if abs(u[j] - u[i]) < 20.0:
                continue
            b = (w[j] - w[i]) / (u[j] - u[i])
            a = w[i] - b * u[i]
            inliers = np.abs(w - (a + b * u)) <= inlier_px
            score = float(weight[inliers].sum())
            if score > best_score:
                best, best_score = inliers, score
    keep = best if best is not None and best.sum() >= 2 else np.ones(u.size, bool)
    b, a = np.polyfit(u[keep], w[keep], 1, w=np.sqrt(weight[keep]))
    return float(a), float(b), int(keep.sum())


def calibrate_width(args):
    samples = {}
    for text in args.sources:
        path, profile, rotated = _parse_source(text)
        bucket = samples.setdefault(profile, {"blue": [], "yellow": []})
        for _, image in _frames(path, args.step, args.limit):
            result = detect_frame(image, profile=profile, already_rotated=rotated)
            for key in ("blue", "yellow"):
                item = result["sliders"].get(key)
                if not item or item.get("reference_method") != "rod_band_both_edges":
                    continue
                measurement = item["measurement"]
                if measurement.get("width_px"):
                    bucket[key].append((item["pixel_xy"][0], measurement["width_px"]))
    report = {}
    for profile, bucket in samples.items():
        blue = np.asarray(bucket["blue"], float).reshape(-1, 2)
        yellow = np.asarray(bucket["yellow"], float).reshape(-1, 2)
        envelope = _upper_envelope(blue[:, 0], blue[:, 1]) if len(blue) else np.zeros((0, 3))
        if len(envelope) < 3 or np.ptp(envelope[:, 0]) < 40:
            report[profile] = {"error": "blue samples do not span enough rail positions",
                               "blue_samples": int(len(blue))}
            continue
        a, b, kept = _robust_line(envelope[:, 0], envelope[:, 1], envelope[:, 2])
        entry = {"blue": [round(a, 3), round(b, 5)], "blue_samples": int(len(blue)),
                 "blue_bins": int(len(envelope)), "blue_bin_inliers": kept,
                 "blue_envelope": [[round(x, 1), round(y, 2), int(n)] for x, y, n in envelope],
                 "blue_u_range": [round(float(blue[:, 0].min()), 1), round(float(blue[:, 0].max()), 1)]}
        if len(yellow) >= 5:
            ratio = yellow[:, 1] / (a + b * yellow[:, 0])
            c = float(np.percentile(ratio, 85.0))
            entry.update(yellow=[round(c * a, 3), round(c * b, 5)], yellow_ratio=round(c, 4),
                         yellow_samples=int(len(yellow)),
                         yellow_u_range=[round(float(yellow[:, 0].min()), 1),
                                         round(float(yellow[:, 0].max()), 1)])
        report[profile] = entry
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("evaluate", "calibrate-width"):
        p = sub.add_parser(name)
        p.add_argument("sources", nargs="+", help="路径:场景[:rotated]")
        p.add_argument("--step", type=int, default=10, help="录像每隔多少帧取一帧")
        p.add_argument("--limit", type=int, default=0, help="每个录像最多取多少帧（0 为不限）")
    sub.choices["evaluate"].add_argument("--output", required=True, help="CSV/JSON 输出目录")
    sub.choices["evaluate"].add_argument("--baseline", help="旧版代码目录，用于逐帧对照")
    sub.choices["calibrate-width"].add_argument("--output", help="拟合结果 JSON 路径")
    args = parser.parse_args(argv)
    if args.step < 1:
        parser.error("--step must be >= 1")
    (evaluate if args.command == "evaluate" else calibrate_width)(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
