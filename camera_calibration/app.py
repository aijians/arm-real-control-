#!/usr/bin/env python3
"""Loopback-only IMX415 calibration UI. A single explicitly selected camera."""
import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import secrets
import signal
import stat
import subprocess
import threading
import time
from urllib.parse import urlsplit

import cv2
import numpy as np

from extrinsics import load_intrinsics, solve_world_pose

CAMERAS = {
    "/dev/video22": "m00_b_imx415 3-001a / rkisp0-vir0",
    "/dev/video31": "m01_f_imx415 4-001a / rkisp1-vir0",
}

WIDTH, HEIGHT = 1920, 1080
MIN_SAMPLES, TARGET_SAMPLES = 15, 30


def atomic_json(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)


def board_config(data):
    cols, rows = int(data['cols']), int(data['rows'])
    square = float(data['square_mm'])
    if not (3 <= cols <= 20 and 3 <= rows <= 20 and math.isfinite(square)
            and 0.1 <= square <= 200):
        raise ValueError('内角点数需为3～20；方格边长需为0.1～200毫米')
    return dict(cols=cols, rows=rows, square_mm=square)


def object_points(config):
    p = np.zeros((config['rows'] * config['cols'], 3), np.float32)
    p[:, :2] = np.mgrid[0:config['cols'], 0:config['rows']].T.reshape(-1, 2)
    p *= config['square_mm']
    return p


def solve(config, corners, metadata):
    if len(corners) < MIN_SAMPLES:
        raise ValueError('至少需要15张不同姿态的有效照片，建议25～40张')
    obj = object_points(config)
    rms, K, D, rotations, translations = cv2.calibrateCamera(
        [obj.copy() for _ in corners], corners, (WIDTH, HEIGHT), None, None)
    errors = []
    for points, r, t in zip(corners, rotations, translations):
        projected, _ = cv2.projectPoints(obj, r, t, K, D)
        errors.append(float(np.sqrt(np.mean(np.sum((projected - points) ** 2, axis=2)))))
    if not all(np.isfinite(x).all() for x in (K, D)) or not math.isfinite(rms):
        raise ValueError('求解结果非有限数，请重新采集')
    warnings = []
    if rms > 1.0:
        warnings.append('重投影RMS大于1像素，建议检查高误差照片并补拍；此阈值只是初筛')
    if len(corners) < 25:
        warnings.append('有效照片少于25张，建议补拍更多位置和倾斜角度')
    cells = sorted({m['cell'] for m in metadata})
    if len(cells) < 5:
        warnings.append('棋盘中心覆盖少于5个九宫格区域，建议补拍画面四周')
    if not (0 < K[0, 2] < WIDTH and 0 < K[1, 2] < HEIGHT and K[0, 0] > 0 and K[1, 1] > 0):
        warnings.append('内参明显异常，检查棋盘规格、图像和姿态分布')
    if max(errors) > 2.0:
        warnings.append('存在单张RMS超过2像素的照片，请按文件名检查报告')
    return dict(camera_matrix=K.tolist(), distortion_coefficients=D.ravel().tolist(),
                distortion_model='plumb_bob', rms_px=float(rms),
                per_image_rms_px=errors, coverage_cells=cells, warnings=warnings,
                sample_count=len(corners), image_width=WIDTH, image_height=HEIGHT,
                quality='needs_review' if warnings else 'review_undistortion',
                pixel_coordinate_frame='sensor_raw', pixels_are_undistorted=False,
                board=config, extrinsics_note='每张棋盘位姿相对于相机；不是全局坐标系外参')


def camera_owners(device):
    identity = Path(device).stat().st_rdev
    owners = []
    for proc in Path('/proc').glob('[0-9]*'):
        if int(proc.name) == os.getpid():
            continue
        for fd in (proc / 'fd').glob('*'):
            try:
                st = fd.stat()
                if stat.S_ISCHR(st.st_mode) and st.st_rdev == identity:
                    owners.append(int(proc.name))
                    break
            except OSError:
                pass
    return owners


class Calibration:
    def __init__(self, output, device, identity):
        self.output, self.device, self.identity = Path(output), device, identity
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.operation = threading.RLock()
        self.halt = threading.Event()
        self.worker = None
        self.state = 'idle'
        self.message = '填写棋盘规格，点击启动相机。未打开相机。'
        self.config = None
        self.session = None
        self.corners, self.samples = [], []
        self.preview = None
        self.frame_time = 0.0
        self.auto = False
        self.take = False
        self.result = None
        self.maps = None
        self.corrected = False
        self.detected = False
        self.last_saved = 0.0
        self.thread_error = None
        self.intrinsics = None
        self.latest = None
        self.frozen = None
        self.extrinsic_result = None
        self.extrinsic_directory = None
        self.exported_candidate = None

    def status(self):
        with self.lock:
            age = None if not self.frame_time else round(time.monotonic() - self.frame_time, 2)
            return dict(state=self.state, message=self.message, device=self.device,
                        sensor=self.identity, resolution=[WIDTH, HEIGHT],
                        samples=len(self.samples), target=TARGET_SAMPLES,
                        coverage=sorted({m['cell'] for m in self.samples}),
                        detected=self.detected, auto=self.auto, config=self.config,
                        frame_age_s=age, session=None if self.session is None else str(self.session),
                        can_calibrate=len(self.samples) >= MIN_SAMPLES and self.state in ('capturing', 'stopped', 'complete'),
                        result=self.result, corrected=self.corrected,
                        cameras=CAMERAS, intrinsics=self.intrinsics,
                        frozen=None if self.frozen is None else dict(
                            token=self.frozen['token'], corners=self.frozen['corners'].reshape(-1, 2).tolist(),
                            board=self.frozen['board']),
                        extrinsics=self.extrinsic_result, exported_candidate=self.exported_candidate)

    def start(self, data):
        config = board_config(data)
        with self.operation, self.lock:
            if self.state not in ('idle', 'stopped', 'error', 'complete'):
                raise ValueError('请先停止当前采集')
            if self.worker and self.worker.is_alive():
                raise ValueError('相机尚未释放，请稍后重试')
            if self.samples and config != self.config:
                raise ValueError('本组样本已绑定棋盘规格；请点击新建一组后修改')
            if not stat.S_ISCHR(Path(self.device).stat().st_mode):
                raise ValueError('设备不是相机字符节点')
            owners = camera_owners(self.device)
            if owners:
                raise ValueError('相机已被占用，PID：' + ','.join(map(str, owners)))
            if self.session is None:
                self.session = self.output / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + secrets.token_hex(3))
                self.session.mkdir()
            self.config = config
            self.latest = None
            self.halt.clear()
            self.state, self.message = 'starting', '正在启动相机并预热…'
            self.auto = False
            self.thread_error = None
            self.worker = threading.Thread(target=self.capture, daemon=True)
            self.worker.start()
            self.manifest()

    def manifest(self):
        if self.session:
            atomic_json(self.session / 'samples.json', dict(
                device=self.device, sensor=self.identity, board=self.config,
                resolution=[WIDTH, HEIGHT], samples=self.samples,
                timestamp_note='文件名时间为板端系统UTC；不保证时钟准确',
                pipeline='NV12 1920x1080 -> BGR; no application crop/flip/resize'))

    def stop(self):
        with self.operation:
            with self.lock:
                if self.state == 'calibrating':
                    raise ValueError('正在计算，请等待完成')
                self.halt.set()
                self.auto, self.take = False, False
                worker = self.worker
            if worker:
                worker.join(20)
                if worker.is_alive():
                    raise ValueError('等待相机释放超时，请查看服务日志')
            with self.lock:
                if self.state != 'error':
                    self.state, self.message = 'stopped', '已停止取流，样本保留；可继续采集或计算'

    def new_session(self):
        with self.operation:
            self.stop()
            with self.lock:
                self.session, self.config = None, None
                self.corners, self.samples = [], []
                self.result, self.maps, self.preview = None, None, None
                self.corrected = False
                self.frame_time = 0
                self.detected = False
                self.latest = self.frozen = self.extrinsic_result = None
                self.extrinsic_directory = self.exported_candidate = None
                self.state, self.message = 'idle', '新组已就绪，旧组文件保留在磁盘'

    def capture(self):
        pipeline = None
        try:
            import gi
            gi.require_version('Gst', '1.0')
            from gi.repository import Gst
            Gst.init(None)
            pipeline = Gst.parse_launch(
                'v4l2src name=camera io-mode=2 ! '
                'video/x-raw,format=NV12,width=1920,height=1080 ! '
                'videoconvert ! video/x-raw,format=BGR ! '
                'appsink name=frames emit-signals=false max-buffers=1 drop=true sync=false wait-on-eos=false')
            pipeline.get_by_name('camera').set_property('device', self.device)
            sink, bus = pipeline.get_by_name('frames'), pipeline.get_bus()
            if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError('相机管线启动失败')
            began = time.monotonic()
            last_sample, last_processed, frames = began, 0., 0
            previous, previous_time = None, 0.
            stable = 0
            while not self.halt.is_set():
                msg = bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
                if msg:
                    if msg.type == Gst.MessageType.ERROR:
                        err, debug = msg.parse_error()
                        raise RuntimeError(err.message + '; ' + str(debug))
                    raise RuntimeError('相机取流结束')
                sample = sink.emit('try-pull-sample', 200 * Gst.MSECOND)
                if sample is None:
                    if time.monotonic() - last_sample > 10:
                        raise RuntimeError('连续10秒未收到相机帧')
                    continue
                now = time.monotonic()
                last_sample = now
                frames += 1
                if now - began < 2 or frames <= 30 or now - last_processed < .35:
                    continue
                last_processed = now
                cap = sample.get_caps().get_structure(0)
                if (cap.get_string('format'), cap.get_value('width'), cap.get_value('height')) != ('BGR', WIDTH, HEIGHT):
                    raise RuntimeError('实际图像尺寸/格式不符合1920×1080 BGR')
                buf = sample.get_buffer()
                ok, info = buf.map(Gst.MapFlags.READ)
                if not ok:
                    raise RuntimeError('映射图像缓冲失败')
                try:
                    stride = (WIDTH * 3 + 3) // 4 * 4
                    if len(info.data) != stride * HEIGHT:
                        raise RuntimeError('不支持的BGR步长；拒绝猜测布局')
                    image = np.ndarray((HEIGHT, WIDTH, 3), np.uint8, buffer=info.data,
                                       strides=(stride, 3, 1)).copy()
                finally:
                    buf.unmap(info)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                small = cv2.resize(gray, (960, 540), interpolation=cv2.INTER_AREA)
                pattern = (self.config['cols'], self.config['rows'])
                found, pts = cv2.findChessboardCorners(small, pattern,
                    cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK)
                if not found:
                    found, pts = cv2.findChessboardCornersSB(small, pattern, cv2.CALIB_CB_NORMALIZE_IMAGE)
                full_resolution = False
                if not found:
                    found, pts = cv2.findChessboardCornersSB(gray, pattern, cv2.CALIB_CB_NORMALIZE_IMAGE)
                    full_resolution = bool(found)
                sharpness, cell = 0., None
                if found:
                    pts = cv2.cornerSubPix(gray, np.float32(pts if full_resolution else pts * 2), (5, 5), (-1, -1),
                                          (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, .001))
                    low = np.maximum(pts.min(axis=0).ravel().astype(int), 0)
                    high = np.minimum(pts.max(axis=0).ravel().astype(int), (WIDTH - 1, HEIGHT - 1))
                    roi = gray[low[1]:high[1]+1, low[0]:high[0]+1]
                    sharpness = float(cv2.Laplacian(roi, cv2.CV_64F).var()) if roi.size else 0.
                    center = pts.mean(axis=0).ravel() / [WIDTH, HEIGHT]
                    cell = int(min(2, center[1] * 3)) * 3 + int(min(2, center[0] * 3))
                    drift = float(np.sqrt(np.mean((pts - previous) ** 2))) if previous is not None else float('inf')
                    stable = stable + 1 if drift < 2 and now - previous_time < 2 else 0
                    previous, previous_time = pts.copy(), now
                else:
                    previous, stable = None, 0
                with self.lock:
                    if self.state == 'starting':
                        self.state = 'capturing'
                    self.detected = bool(found)
                    self.frame_time = now
                    self.latest = dict(image=image.copy(), corners=pts.copy() if found else None,
                                       time=now, stable=stable, sharpness=sharpness,
                                       board=dict(self.config))
                    if self.state == 'capturing':
                        eligible = found and sharpness >= 35 and stable >= 1 and time.monotonic() - now < 3
                        duplicate = found and any(float(np.sqrt(np.mean(((pts - p) / [WIDTH, HEIGHT]) ** 2))) < .025 for p in self.corners)
                        if self.take or (self.auto and eligible and now - self.last_saved > 2):
                            self.take = False
                            if not eligible:
                                self.message = '未保存：请让完整棋盘清晰可见，并停稳约1秒'
                            elif duplicate:
                                self.message = '未保存：姿态重复，请换位置、距离或倾斜角度'
                            else:
                                filename = 'sample_%03d.png' % (len(self.samples) + 1)
                                if not cv2.imwrite(str(self.session / filename), image):
                                    raise RuntimeError('保存原始分辨率图像失败')
                                self.corners.append(pts.copy())
                                self.samples.append(dict(file=filename, cell=cell, sharpness=sharpness,
                                                         corners=pts.reshape(-1, 2).tolist()))
                                self.last_saved = now
                                self.result, self.maps, self.corrected = None, None, False
                                self.manifest()
                                self.message = '已保存%d张；请继续改变棋盘姿态' % len(self.samples)
                                if len(self.samples) >= 40:
                                    self.auto = False
                                    self.message = '已保存40张，自动采集已暂停，可开始标定'
                        elif not self.auto and self.message.startswith('正在启动'):
                            self.message = '画面已就绪；点击自动采集或手动采一张'
                    view = image
                    if self.corrected and self.maps is not None:
                        view = cv2.remap(image, *self.maps, interpolation=cv2.INTER_LINEAR)
                    else:
                        view = image.copy()
                        if found:
                            cv2.drawChessboardCorners(view, pattern, pts, True)
                    view = cv2.resize(view, (960, 540), interpolation=cv2.INTER_AREA)
                    ok, jpg = cv2.imencode('.jpg', view, [cv2.IMWRITE_JPEG_QUALITY, 82])
                    if ok:
                        self.preview = jpg.tobytes()
        except Exception as exc:
            with self.lock:
                self.thread_error = str(exc)
                self.state, self.message = 'error', '相机错误：' + str(exc)
                self.auto = False
        finally:
            if pipeline is not None:
                pipeline.set_state(Gst.State.NULL)

    def calibrate(self):
        with self.operation, self.lock:
            if self.state not in ('capturing', 'stopped', 'complete') or len(self.corners) < MIN_SAMPLES:
                raise ValueError('至少15张有效照片，且当前不能有其他标定任务')
            self.auto, self.take = False, False
            self.state, self.message = 'calibrating', '正在计算；请等待结果，不需要继续移动棋盘'
            snapshot = [p.copy() for p in self.corners]
            metadata, config, directory = list(self.samples), dict(self.config), self.session
            threading.Thread(target=self._solve, args=(snapshot, metadata, config, directory), daemon=True).start()

    def _solve(self, corners, metadata, config, directory):
        try:
            result = solve(config, corners, metadata)
            result.update(device=self.device, sensor=self.identity,
                          created_utc=datetime.now(timezone.utc).isoformat(),
                          files=[m['file'] for m in metadata],
                          coordinate_units='intrinsic pixels; board translations mm',
                          applicability='This physical camera and capture mode only')
            K, D = np.array(result['camera_matrix']), np.array(result['distortion_coefficients'])
            maps = cv2.initUndistortRectifyMap(K, D, None, K, (WIDTH, HEIGHT), cv2.CV_32FC1)
            atomic_json(directory / 'calibration.json', result)
            temp = directory / 'calibration.tmp.yaml'
            fs = cv2.FileStorage(str(temp), cv2.FILE_STORAGE_WRITE)
            if not fs.isOpened():
                raise RuntimeError('无法写入YAML文件')
            try:
                for key, value in [('image_width', WIDTH), ('image_height', HEIGHT),
                                   ('camera_matrix', K), ('distortion_coefficients', D.reshape(1, -1)),
                                   ('distortion_model', 'plumb_bob'), ('rms_px', result['rms_px']),
                                   ('camera_device', self.device), ('sensor', self.identity)]:
                    fs.write(key, value)
            finally:
                fs.release()
            temp.replace(directory / 'calibration.yaml')
            with self.lock:
                self.result, self.maps = result, maps
                if self.thread_error:
                    self.state = 'error'
                    self.message = '结果已保存，但相机预览异常：' + self.thread_error
                else:
                    self.state = 'complete'
                    self.message = '计算完成并已保存；请检查误差和去畸变预览。' + ('有质量提示。' if result['warnings'] else '')
        except Exception as exc:
            with self.lock:
                self.state = 'stopped' if not self.worker or not self.worker.is_alive() else 'capturing'
                self.message = '计算失败：' + str(exc)

    def set_device(self, data):
        device = data.get('device')
        if device not in CAMERAS:
            raise ValueError('仅支持video22/video31')
        with self.operation:
            if self.worker and self.worker.is_alive():
                raise ValueError('请先停止取流再切换相机')
            if self.state == 'calibrating':
                raise ValueError('请等待内参计算完成')
            if device != self.device:
                self.new_session()
                with self.lock:
                    self.device, self.identity = device, CAMERAS[device]
                    self.intrinsics = None
                    self.message = '已选择'+device+'；历史组仍保留，请加载本机内参或重新标定'

    def import_intrinsics(self, data):
        with self.operation, self.lock:
            if self.state == 'calibrating':
                raise ValueError('请等待内参计算完成')
            source = self.result if data.get('use_current') else data.get('intrinsics')
            self.intrinsics = load_intrinsics(source, self.device, self.identity)
            self.extrinsic_result = self.exported_candidate = None
            self.message = '已加载本机原始图像内参；可拍摄已知位置棋盘求外参'

    def freeze_extrinsics(self):
        with self.operation, self.lock:
            if self.state not in ('capturing', 'complete') or self.latest is None:
                raise ValueError('请先启动相机，摆好已知世界位置的棋盘')
            if self.intrinsics is None:
                raise ValueError('请先导入本机内参，或使用本组计算的内参')
            frame = self.latest
            if time.monotonic()-frame['time'] > 2 or frame['corners'] is None or frame['stable'] < 1 or frame['sharpness'] < 35:
                raise ValueError('需要新鲜、完整、清晰且停稳的棋盘；请稍后重新拍摄')
            self.auto, self.take = False, False
            token = secrets.token_hex(8)
            directory = self.session / ('world_pose_'+token)
            directory.mkdir()
            image = frame['image'].copy()
            corners = frame['corners'].copy()
            if not cv2.imwrite(str(directory/'sensor_raw.png'), image):
                raise OSError('保存外参原图失败')
            for i, point in enumerate(corners.reshape(-1, 2)):
                xy = tuple(np.rint(point).astype(int))
                cv2.circle(image, xy, 5, (0, 255, 255), -1)
                cv2.putText(image, str(i), (xy[0]+6, xy[1]-6), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 255), 2)
            ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise OSError('生成外参预览失败')
            (directory/'corners.jpg').write_bytes(encoded.tobytes())
            self.frozen = dict(token=token, corners=corners, board=dict(frame['board']), preview=encoded.tobytes())
            self.extrinsic_directory = directory
            self.extrinsic_result = self.exported_candidate = None
            atomic_json(directory/'capture.json', dict(board=frame['board'], device=self.device,
                sensor=self.identity, intrinsics=self.intrinsics, pixel_coordinate_frame='sensor_raw',
                image_sha256=__import__('hashlib').sha256((directory/'sensor_raw.png').read_bytes()).hexdigest(),
                corners=corners.reshape(-1, 2).tolist(), captured_utc=datetime.now(timezone.utc).isoformat()))
            self.message = '外参图已固定；依次点击打印板O、U、V三个内角点，再填写其世界坐标'

    def compute_extrinsics(self, data):
        with self.operation, self.lock:
            if self.frozen is None or data.get('token') != self.frozen['token']:
                raise ValueError('外参图已变化，请重新点选')
            if self.intrinsics is None:
                raise ValueError('请先加载本机内参')
            if self.state == 'calibrating':
                raise ValueError('请等待内参计算完成')
            # Changing any inputs invalidates downloadable candidates even if the
            # new request is rejected; never keep an old pose under new inputs.
            self.extrinsic_result = self.exported_candidate = None
            result = solve_world_pose(self.frozen['board'], self.frozen['corners'],
                                      data.get('corner_ids', []), data, self.intrinsics)
            result['capture_token'] = self.frozen['token']
            result['created_utc'] = datetime.now(timezone.utc).isoformat()
            atomic_json(self.extrinsic_directory/'extrinsics.json', result)
            self.extrinsic_result = result
            self.message = '外参已计算并保存，请检查重投影、光心和光轴；选定候选后导出，不自动应用到机械臂'

    def export_extrinsics(self, data):
        with self.operation, self.lock:
            if self.extrinsic_result is None or data.get('token') != self.frozen['token']:
                raise ValueError('请先求解当前固定图像')
            index = data.get('candidate')
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(self.extrinsic_result['candidates']):
                raise ValueError('请选择有效候选')
            candidate = self.extrinsic_result['candidates'][index]
            for frame in ('sensor_raw', 'scene_upright'):
                atomic_json(self.extrinsic_directory/('camera_world_'+frame+'.json'), candidate['calibration_'+frame])
            atomic_json(self.extrinsic_directory/'selection.json', dict(candidate=index,
                capture_token=self.frozen['token'], independent_world_validation=False, applied_to_robot=False))
            self.exported_candidate = index
            self.message = '候选已导出，尚未写入机械臂配置；请做独立已知点验证'

    def action(self, name, data):
        if name == 'device':
            return self.set_device(data)
        if name == 'intrinsics':
            return self.import_intrinsics(data)
        if name == 'freeze_extrinsics':
            return self.freeze_extrinsics()
        if name == 'solve_extrinsics':
            return self.compute_extrinsics(data)
        if name == 'export_extrinsics':
            return self.export_extrinsics(data)
        if name == 'start':
            self.start(data)
        elif name == 'stop':
            self.stop()
        elif name == 'new':
            self.new_session()
        elif name == 'calibrate':
            self.calibrate()
        else:
            with self.lock:
                if name in ('auto', 'take'):
                    if self.state != 'capturing':
                        raise ValueError('请先启动相机，计算完成后如需补拍请先停止再启动')
                    if name == 'auto':
                        self.auto = not self.auto
                        self.message = '自动采集中：摆好完整棋盘并停稳，改变姿态后再采下一张' if self.auto else '自动采集已暂停'
                    else:
                        self.take = True
                elif name == 'corrected' and self.result:
                    self.corrected = not self.corrected
                else:
                    raise ValueError('此操作当前不可用')


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if urlsplit(self.path).path not in ('/api/status', '/preview.jpg'):
            super().log_message(fmt, *args)

    def send(self, code, content, mime):
        self.send_response(code)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self'")
        self.end_headers()
        try:
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        app = self.server.app
        path = urlsplit(self.path).path
        if path == '/':
            return self.send(200, Path(__file__).with_name('index.html').read_bytes(), 'text/html; charset=utf-8')
        if path == '/download/checkerboard_6mm_8x6_A4.pdf':
            return self.send(200, Path(__file__).with_name('patterns').joinpath('checkerboard_6mm_8x6_A4.pdf').read_bytes(), 'application/pdf')
        if path == '/api/status':
            return self.send(200, json.dumps(app.status(), ensure_ascii=False).encode(), 'application/json')
        if path == '/world-preview.jpg':
            with app.lock:
                jpg = app.frozen['preview'] if app.frozen else None
            return self.send(200 if jpg else 404, jpg or b'No pose image', 'image/jpeg' if jpg else 'text/plain')
        if path == '/preview.jpg':
            with app.lock:
                jpg = app.preview
            return self.send(200 if jpg else 503, jpg or b'No camera frame yet', 'image/jpeg' if jpg else 'text/plain')
        if path == '/extrinsics.jpg':
            with app.lock:
                jpg = app.frozen['preview'] if app.frozen else None
            return self.send(200 if jpg else 404, jpg or b'No frozen frame', 'image/jpeg' if jpg else 'text/plain')
        if path.startswith('/world-download/'):
            name = path.rsplit('/', 1)[-1]
            with app.lock:
                allowed = name in ('extrinsics.json', 'sensor_raw.png', 'corners.jpg') and app.extrinsic_result is not None
                allowed |= name in ('camera_world_sensor_raw.json', 'camera_world_scene_upright.json', 'selection.json') and app.exported_candidate is not None
                target = app.extrinsic_directory / name if allowed and app.extrinsic_directory else None
                content = target.read_bytes() if target and target.exists() else None
            return self.send(200 if content else 404, content or b'Not generated yet', 'application/octet-stream')
        if path in ('/download/calibration.yaml' , '/download/calibration.json', '/download/samples.json'):
            with app.lock:
                target = app.session / path.rsplit('/', 1)[-1] if app.session else None
                current = path.endswith('/samples.json') or app.result is not None
                content = target.read_bytes() if current and target and target.exists() else None
            return self.send(200 if content else 404, content or b'Not generated yet', 'application/octet-stream')
        self.send(404, b'Not found', 'text/plain')

    def do_POST(self):
        # JSON custom header + no CORS: another website cannot drive this loopback service.
        if self.headers.get('X-Calibration-UI') != '1' or not self.headers.get('Content-Type', '').startswith('application/json'):
            return self.send(403, b'UI request required', 'text/plain')
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length < 0 or length > 65536:
                raise ValueError('请求过大')
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError('参数格式错误')
            path = urlsplit(self.path).path
            if not path.startswith('/api/'):
                raise ValueError('未知操作')
            self.server.app.action(path[5:], payload)
            self.send(200, b'{"ok":true}', 'application/json')
        except (ValueError, KeyError, TypeError, OSError) as exc:
            self.send(400, json.dumps(dict(error=str(exc)), ensure_ascii=False).encode(), 'application/json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='/dev/video22', choices=list(CAMERAS))
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--output', default='/var/lib/camera-calibration-video22')
    args = parser.parse_args()
    cv2.setNumThreads(2)
    app = Calibration(args.output, args.device, CAMERAS[args.device])
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.daemon_threads = True
    server.app = app
    def terminate(signum, frame):
        app.halt.set()
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    print('Calibration web listening on 127.0.0.1:%d; camera stays closed until Start' % args.port, flush=True)
    try:
        server.serve_forever()
    finally:
        app.halt.set()
        if app.worker:
            app.worker.join(20)
        server.server_close()


if __name__ == '__main__':
    main()
