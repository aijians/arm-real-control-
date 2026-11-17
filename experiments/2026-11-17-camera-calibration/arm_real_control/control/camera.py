"""One task, one GStreamer stream: MP4 recording plus a one-frame cache."""
from pathlib import Path
import threading
import time

import numpy as np


class CameraError(RuntimeError):
    pass


def _load_gst():
    # Import only when a task starts. Idle startup never opens the camera.
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst
    Gst.init(None)
    return Gst


class CameraSession:
    def __init__(self, config, video_path):
        self.config = config
        self.path = Path(video_path)
        self.condition = threading.Condition()
        self.abort = threading.Event()
        self.done = threading.Event()
        self.latest = None
        self.sequence = 0
        self.error = None
        self.eos = False
        self.pipeline = None
        self.thread = None
        self.report = None

    def start(self):
        self.Gst = _load_gst()
        width, height, fps = (int(self.config.get(k, v)) for k, v in
                              (("width", 1920), ("height", 1080), ("fps", 30)))
        if min(width, height, fps) <= 0:
            raise CameraError("invalid_camera_size_or_rate")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Task IDs are unique; never silently truncate an earlier task's video.
        self.path.open("xb").close()
        spec = (
            "v4l2src name=camera io-mode=2 ! "
            f"video/x-raw,format=NV12,width={width},height={height},framerate={fps}/1 ! "
            "tee name=split "
            "split. ! queue ! mpph264enc name=encoder ! h264parse ! mp4mux ! "
            "filesink name=recording "
            "split. ! queue leaky=downstream max-size-buffers=1 max-size-bytes=0 "
            "max-size-time=0 ! videoconvert ! video/x-raw,format=BGR ! "
            "appsink name=frames emit-signals=false max-buffers=1 drop=true "
            "sync=false wait-on-eos=false"
        )
        try:
            self.pipeline = self.Gst.parse_launch(spec)
            self.pipeline.get_by_name("camera").set_property(
                "device", self.config.get("camera_device", "/dev/video31"))
            self.pipeline.get_by_name("recording").set_property("location", str(self.path))
            self.pipeline.get_by_name("encoder").set_property(
                "bps", int(self.config.get("bitrate", 8000000)))
            self.sink = self.pipeline.get_by_name("frames")
            self.bus = self.pipeline.get_bus()
            self.started = time.monotonic()
            self.warmup_until = self.started + float(self.config.get("warmup_seconds", 2.0))
            if self.pipeline.set_state(self.Gst.State.PLAYING) == self.Gst.StateChangeReturn.FAILURE:
                raise CameraError("camera_pipeline_could_not_start")
            self.thread = threading.Thread(target=self._capture, name="arm-camera", daemon=True)
            self.thread.start()
            self.frame(0, None, float(self.config.get("capture_timeout_s", 20.0)),
                       float(self.config.get("max_frame_age_s", 3.0)))
        except BaseException:
            self.close()
            raise

    def _copy_sample(self, sample):
        caps = sample.get_caps().get_structure(0)
        if caps.get_string("format") != "BGR":
            raise CameraError("camera_video_caps_not_bgr")
        width, height = int(caps.get_value("width")), int(caps.get_value("height"))
        if min(width, height) <= 0:
            raise CameraError("camera_video_caps_invalid")
        # Our videoconvert -> BGR branch uses standard four-byte row alignment.
        # Reject other layouts instead of guessing padding; GstVideo's optional
        # Python typelib is not installed on the board.
        stride = (width * 3 + 3) // 4 * 4
        buf = sample.get_buffer()
        mapped, info = buf.map(self.Gst.MapFlags.READ)
        if not mapped:
            raise CameraError("camera_buffer_map_failed")
        try:
            if len(info.data) != stride * height:
                raise CameraError("camera_bgr_buffer_size_mismatch")
            image = np.ndarray((height, width, 3), dtype=np.uint8,
                               buffer=info.data, strides=(stride, 3, 1)).copy()
        finally:
            buf.unmap(info)
        pts = None if buf.pts == self.Gst.CLOCK_TIME_NONE else int(buf.pts)
        return image, pts

    def _publish(self, image, received, pts):
        with self.condition:
            self.sequence += 1
            if (received >= self.warmup_until
                    and self.sequence > int(self.config.get("warmup_frames", 30))):
                self.latest = {"image": image, "sequence": self.sequence,
                               "received_monotonic_s": received, "pipeline_pts_ns": pts}
                self.condition.notify_all()

    def _capture(self):
        Gst = self.Gst
        try:
            while not self.abort.is_set():
                message = self.bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
                if message is not None:
                    if message.type == Gst.MessageType.ERROR:
                        error, _ = message.parse_error()
                        raise CameraError("camera_pipeline_error: " + error.message)
                    self.eos = True
                    break
                sample = self.sink.emit("try-pull-sample", 100 * Gst.MSECOND)
                if sample is not None:
                    received = time.monotonic()
                    image, pts = self._copy_sample(sample)
                    self._publish(image, received, pts)
        except Exception as exc:
            self.error = str(exc)
        finally:
            self.done.set()
            with self.condition:
                self.condition.notify_all()

    def frame(self, after_sequence, after_monotonic_s, timeout, max_age):
        """Wait for a fresh, unconsumed latest frame; no queue of old frames."""
        deadline = time.monotonic() + timeout
        with self.condition:
            while True:
                if self.error:
                    raise CameraError(self.error)
                if self.done.is_set():
                    raise CameraError("camera_stream_ended")
                item = self.latest
                if (item is not None and item["sequence"] > after_sequence
                        and (after_monotonic_s is None
                             or item["received_monotonic_s"] > after_monotonic_s)
                        and time.monotonic() - item["received_monotonic_s"] <= max_age):
                    return item
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CameraError("camera_fresh_frame_timeout")
                self.condition.wait(remaining)

    def check_health(self, max_age):
        with self.condition:
            if self.error:
                raise CameraError(self.error)
            if self.done.is_set():
                raise CameraError("camera_stream_ended")
            if (self.latest is None
                    or time.monotonic() - self.latest["received_monotonic_s"] > max_age):
                raise CameraError("camera_frame_stale")

    def close(self):
        """EOS finalizes the MP4 before releasing the device; safe to call twice."""
        if self.report is not None:
            return self.report
        released = True
        if self.pipeline is not None:
            try:
                if self.thread is not None and not self.done.is_set():
                    if not self.pipeline.send_event(self.Gst.Event.new_eos()):
                        self.error = self.error or "camera_eos_rejected"
                    elif not self.done.wait(float(self.config.get("eos_timeout_s", 8.0))):
                        self.error = self.error or "camera_eos_timeout"
            except Exception as exc:
                self.error = self.error or str(exc)
            finally:
                self.abort.set()
                try:
                    result = self.pipeline.set_state(self.Gst.State.NULL)
                    _, state, _ = self.pipeline.get_state(3 * self.Gst.SECOND)
                    released = (result != self.Gst.StateChangeReturn.FAILURE
                                and state == self.Gst.State.NULL)
                except Exception as exc:
                    released = False
                    self.error = self.error or str(exc)
                if self.thread is not None:
                    self.thread.join(1.0)
                with self.condition:
                    self.condition.notify_all()
        try:
            has_video = self.path.is_file() and self.path.stat().st_size > 0
        except OSError as exc:
            has_video = False
            self.error = self.error or str(exc)
        completed = self.eos and released and not self.error and has_video
        self.report = {"file": str(self.path), "status": "completed" if completed else "failed",
                       "frames_received": self.sequence, "eos_received": self.eos,
                       "pipeline_released": released}
        if not completed:
            self.report["reason"] = self.error or "video_not_finalized"
        return self.report
