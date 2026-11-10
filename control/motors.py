"""Two independent RS485 motors, with continuous trajectory streaming.

Only A4/A6 (position), 94/9C (feedback), and 81 (stop) are emitted. Opening a
port never enables, homes, or writes an encoder zero. Linux standard library
only; the protocol is the one already used by the board's existing program.
"""

from collections import deque
import fcntl
import math
import os
from pathlib import Path
import select
import termios
import threading
import time

from .trajectory import WaypointCursor, GoalQualification, timed_free_points
from .kinematics import PlanarArm
from .recovery import SpeedRecovery, recovery_settings
from .motion_policy import MotionRecovery, motion_policy_settings

from .protocol import (angle_from_reply, extract_frame, make_frame,
                       position_frame, multi_turn_position_frame, state_from_reply)


class HardwareError(RuntimeError):
    def __init__(self, reason, details=None, halt_result=None):
        super().__init__(reason)
        self.reason = reason
        self.details = details or {}
        self.halt_result = halt_result


class _MotionTrace:
    """Bounded in-memory observations; never writes files in the control loop."""

    MAX_ACTIONS = 256

    def __init__(self, phase, points, config, motors):
        self.phase = phase.get("phase")
        self.duration = points[-1][0]
        self.started = None
        self.motors = motors
        self.settings = {k: config[k] for k in (
            "command_rate_hz", "feedback_rate_hz", "max_speed_deg_s",
            "feedback_timeout_s", "angle_tolerance_deg", "still_speed_deg_s",
            "stable_s", "no_progress_s", "progress_deg", "tracking_lead_deg",
            "tracking_abort_deg", "goal_hold_s", "goal_settle_timeout_s",
            "waypoint_tolerance_ratio", "inactive_feedback_hz", "inactive_drift_deg",
            "coalesce_step_deg", "coalesce_wait_s", "settling_bias_wait_s",
            "settling_bias_max_deg", "settling_bias_step_deg")}
        self.settings["speed_recovery"] = recovery_settings(config)
        self.settings["motion_recovery"] = motion_policy_settings(config)
        self.plan = {"point_count": len(points), "duration_s": self.duration,
                     "start_q_deg": [math.degrees(q) for q in points[0][1]],
                     "end_q_deg": [math.degrees(q) for q in points[-1][1]]}
        self.plan.update(waypoint_min_axis_step_deg=phase.get("waypoint_min_axis_step_deg"),
                         protected_micro_steps=phase.get("protected_micro_steps", []))
        self.actions = deque(maxlen=self.MAX_ACTIONS)
        self.action_count = 0
        self.counts = {m["name"]: {"sent": 0, "skipped": 0, "send_failed": 0}
                       for m in motors}
        self.first_actions = {}
        self.last_sent = {}
        self.last_queries = {}
        self.query_counts = {}
        self.reply_counts = {}
        self.last_replies = {}
        self.recording_error = None
        self.execution = {}
        self.stage_transitions = []

    def record_stage(self, stage, entered, feedback, state, target, elapsed,
                     progress_at, progress_q):
        try:
            context = self.context(feedback, state, target, elapsed, progress_at, progress_q)
            self.stage_transitions.append({
                "stage": stage, "entered_monotonic_s": entered,
                "wall_elapsed_s": entered - self.started,
                "reference_elapsed_s": elapsed, "axes": context["axes"],
                "snapshot_stamp_monotonic_s": context["snapshot_stamp_monotonic_s"],
                "query_counts": dict(self.query_counts),
                "reply_counts": dict(self.reply_counts),
            })
        except Exception as exc:
            self.recording_error = type(exc).__name__

    def record_action(self, event):
        try:
            self.action_count += 1
            event["sequence"] = self.action_count
            event["elapsed_s"] = (None if self.started is None else
                                  event["decision_monotonic_s"] - self.started)
            name = event["motor"]
            self.counts[name][event["status"]] += 1
            self.first_actions.setdefault(name, event)
            if event["status"] == "sent":
                self.last_sent[name] = event
            self.actions.append(event)
        except Exception as exc:
            self.recording_error = type(exc).__name__

    def record_query(self, motor, command, completed):
        try:
            key = motor["name"] + ":0x%02x" % command
            self.query_counts[key] = self.query_counts.get(key, 0) + 1
            self.last_queries[key] = {"motor": motor["name"], "command": "0x%02x" % command,
                                      "write_completed_monotonic_s": completed}
        except Exception as exc:
            self.recording_error = type(exc).__name__

    def context(self, feedback, state, target, elapsed, progress_at, progress_q):
        captured = time.monotonic()
        axes = []
        for i, (motor, reply) in enumerate(zip(self.motors, feedback)):
            actual = None if state is None else math.degrees(state["q_rad"][i])
            reference = None if target is None else math.degrees(target[i])
            angle_stamp, speed_stamp = reply.get("angle_stamp"), reply.get("speed_stamp")
            axes.append({
                "motor": motor["name"], "device": motor["device"],
                "actual_q_deg": actual, "reference_q_deg": reference,
                "tracking_error_deg": (None if actual is None or reference is None
                                       else abs(actual - reference)),
                "snapshot_speed_deg_s": (None if state is None else
                                         math.degrees(state["speed_rad_s"][i])),
                "cached_motor_deg": reply.get("motor_deg"),
                "angle_feedback_age_s": None if angle_stamp is None else captured - angle_stamp,
                "speed_feedback_age_s": None if speed_stamp is None else captured - speed_stamp,
                "since_progress_s": None if progress_at is None else captured - progress_at[i],
                "progress_anchor_q_deg": None if progress_q is None else math.degrees(progress_q[i]),
                "movement_from_anchor_deg": (None if actual is None or progress_q is None
                                             else abs(actual - math.degrees(progress_q[i]))),
            })
        return {"captured_monotonic_s": captured, "reference_elapsed_s": elapsed,
                "phase": self.phase, "axes": axes,
                "snapshot_stamp_monotonic_s": None if state is None else state["stamp_monotonic_s"]}

    def record_reply(self, motor, frame, received):
        try:
            key = motor["name"] + ":0x%02x" % frame.command
            self.reply_counts[key] = self.reply_counts.get(key, 0) + 1
            self.last_replies[key] = {"received_monotonic_s": received,
                                      "payload_hex": frame.payload.hex(" ")}
        except Exception as exc:
            self.recording_error = type(exc).__name__

    def export(self, status, context):
        return {"schema_version": 2, "status": status, "phase": self.phase,
                "plan": self.plan, "settings": self.settings,
                "motion_started_monotonic_s": self.started,
                "capture": context, "execution": {**self.execution,
                                                     "stage_transitions": list(self.stage_transitions)},
                "recording_error": self.recording_error,
                "action_count": self.action_count,
                "retained_action_limit": self.MAX_ACTIONS,
                "dropped_actions": self.action_count - len(self.actions),
                "counts": {k: dict(v) for k, v in self.counts.items()},
                "first_actions": dict(self.first_actions), "last_sent": dict(self.last_sent),
                "query_counts": dict(self.query_counts), "last_queries": dict(self.last_queries),
                "reply_counts": dict(self.reply_counts), "last_replies": dict(self.last_replies),
                "actions": list(self.actions),
                "timestamp_definition": "host_monotonic_not_motor_sample_time",
                "sent_definition": "serial_write_completed_not_motor_execution_ack"}


def _capture_motion_trace(trace, status, feedback, state, target, elapsed, progress_at, progress_q):
    # A diagnostic failure must never prevent the existing halt/error path.
    try:
        return trace.export(status, trace.context(
            feedback, state, target, elapsed, progress_at, progress_q))
    except Exception as exc:
        return {"schema_version": 2, "status": status, "phase": trace.phase,
                "recording_error": type(exc).__name__}


def _poll_port_fault(device, events):
    """Only explicit poll faults indicate a hangup or unusable descriptor."""
    faults = events & (select.POLLHUP | select.POLLERR | select.POLLNVAL)
    if not faults:
        return
    names = [name for name in ("POLLHUP", "POLLERR", "POLLNVAL")
             if faults & getattr(select, name)]
    reason = "serial_disconnected" if faults & select.POLLHUP else "serial_read_failed"
    raise HardwareError(reason, {"device": device, "poll_events": events,
                                 "poll_faults": names})


def _finite(value, name):
    if isinstance(value, bool):
        raise ValueError(name + " must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(name + " must be finite")
    return result


def _device_users(device, ignore_fd=None):
    """Inspect current device users, in addition to kernel/advisory locks."""
    identity = os.stat(device)
    users = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            descriptors = list((proc / "fd").iterdir())
        except OSError:
            continue
        for descriptor in descriptors:
            if int(proc.name) == os.getpid() and descriptor.name == str(ignore_fd):
                continue
            try:
                info = descriptor.stat()
            except OSError:
                continue
            if (info.st_dev, info.st_ino) == (identity.st_dev, identity.st_ino):
                users.append({"pid": int(proc.name), "fd": descriptor.name})
    return users


class _Port:
    # Board RS485 is half duplex: leave time for the motor's reply before
    # writing another frame. Immediate 94/9C writes corrupt both replies.
    REPLY_GAP_S = 0.005

    def __init__(self, device, baudrate):
        self.device, self.baudrate = device, baudrate
        self.fd = None
        self.buffer = bytearray()
        self._next_send_at = 0.0

    def open(self):
        try:
            users = _device_users(self.device)
            if users:
                raise HardwareError("serial_busy", {"device": self.device, "users": users})
            self.fd = os.open(self.device, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.ioctl(self.fd, termios.TIOCEXCL)
            users = _device_users(self.device, self.fd)
            if users:
                raise HardwareError("serial_busy", {"device": self.device, "users": users})
            attrs = termios.tcgetattr(self.fd)
            attrs[0] = attrs[1] = attrs[3] = 0
            attrs[2] = termios.CS8 | termios.CLOCAL | termios.CREAD
            speed = getattr(termios, "B" + str(self.baudrate))
            attrs[4] = attrs[5] = speed
            # With NONBLOCK, an empty queue should normally report EAGAIN.
            # Empty reads are still handled separately from explicit hangups.
            attrs[6][termios.VMIN] = 1
            attrs[6][termios.VTIME] = 0
            termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
            termios.tcflush(self.fd, termios.TCIOFLUSH)
        except (OSError, termios.error, HardwareError) as exc:
            self.close()
            if isinstance(exc, HardwareError):
                raise
            raise HardwareError("serial_open_failed", {"device": self.device,
                                                       "error": str(exc)}) from exc

    def close(self):
        if self.fd is not None:
            try:
                fcntl.ioctl(self.fd, termios.TIOCNXCL)
            except OSError:
                pass
            os.close(self.fd)
            self.fd = None
        self.buffer.clear()
        self._next_send_at = 0.0

    def send(self, data, timeout_s=0.05, cancel_event=None):
        if self.fd is None:
            raise HardwareError("serial_not_open", {"device": self.device})
        delay = self._next_send_at - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        if cancel_event is not None and cancel_event.is_set():
            raise HardwareError("motion_halted")
        deadline, count = time.monotonic() + timeout_s, 0
        try:
            while count < len(data):
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([], [self.fd], [], remaining)[1]:
                    raise HardwareError("serial_write_timeout", {"device": self.device})
                try:
                    wrote = os.write(self.fd, data[count:])
                except BlockingIOError:
                    continue
                if wrote <= 0:
                    raise OSError("Zero-byte serial write")
                count += wrote
            self._next_send_at = time.monotonic() + self.REPLY_GAP_S
        except OSError as exc:
            raise HardwareError("serial_write_failed", {"device": self.device,
                                                        "error": str(exc)}) from exc


class MotorPair:
    """Angles are model radians. Mapping: motor_deg=offset+sign*ratio*model_deg.

    ``limits_deg`` are MODEL joint limits. ``max_speed_deg_s`` is the motor
    shaft speed limit; a ratio other than one is accounted for in validation.
    A6 is single-turn: mapped targets must also lie in [0, 359.99] degrees.
    This class does not change the board's existing electrical RS485 setup.
    """

    DEFAULTS = {'baudrate': 115200,
     'command_rate_hz': 50,
     'feedback_rate_hz': 20,
     'max_speed_deg_s': 18,
     'feedback_timeout_s': 0.5,
     'move_timeout_margin_s': 5,
     'angle_tolerance_deg': 0.3,
     'still_speed_deg_s': 1,
     'stable_s': 0.3,
     'no_progress_s': 2,
     'progress_deg': 0.05,
     'tracking_lead_deg': 1.8,
     'tracking_abort_deg': 3.5,
     'goal_hold_s': 3.0,
     'goal_settle_timeout_s': 5.0,
     'waypoint_tolerance_ratio': 0.5,
     'inactive_feedback_hz': 2.0,
     'inactive_feedback_timeout_s': 1.5,
     'inactive_drift_deg': 1.0,
     'min_speed_deg_s': 2.0,
     'speed_feedforward_gain': 1.15,
     'speed_error_gain_s': 2.5,
     'coalesce_step_deg': 0.45,
     'coalesce_wait_s': 0.45,
     'settling_bias_wait_s': 0.45,
     'settling_bias_max_deg': 0.6,
     'settling_bias_step_deg': 0.2,
     'settling_bias_trigger_margin_deg': 0.01}

    def __init__(self, hardware_config):
        self.config = dict(self.DEFAULTS, **hardware_config)
        self.recovery_config = recovery_settings(self.config)
        self.motion_policy = motion_policy_settings(self.config)
        timed = self.config.get("free_timed_motion", {})
        if not isinstance(timed, dict) or type(timed.get("enabled", False)) is not bool:
            raise ValueError("invalid free_timed_motion.enabled")
        self.free_timed = timed.get("enabled", False)
        self.free_timed_speed = _finite(timed.get("speed_deg_s", 10.),
                                        "free_timed_motion.speed_deg_s")
        # Free has its own fixed A4 speed, independent of the legacy executor cap.
        if not 0 < self.free_timed_speed <= 360:
            raise ValueError("invalid free_timed_motion.speed_deg_s")
        self.free_timed_step = _finite(timed.get("max_step_deg", .2),
                                       "free_timed_motion.max_step_deg")
        self.free_timed_period = _finite(timed.get("period_s", .05),
                                         "free_timed_motion.period_s")
        if self.free_timed_step <= 0 or self.free_timed_period <= 0:
            raise ValueError("free_timed_motion step and period must be positive")
        self.motors = [dict(m) for m in self.config["motors"]]
        if len(self.motors) != 2:
            raise ValueError("Exactly two motors are required")
        if self.config["baudrate"] not in (9600, 19200, 38400, 57600, 115200):
            raise ValueError("Unsupported serial baudrate")
        for key in self.DEFAULTS:
            self.config[key] = _finite(self.config[key], key)
            if self.config[key] <= 0:
                raise ValueError(key + " must be positive")
        if not 0 < self.config["waypoint_tolerance_ratio"] < 1:
            raise ValueError("waypoint_tolerance_ratio must be between zero and one")
        if self.config["min_speed_deg_s"] > self.config["max_speed_deg_s"]:
            raise ValueError("Minimum speed exceeds maximum speed")
        if not (self.config["angle_tolerance_deg"] < self.config["tracking_lead_deg"]
                < self.config["tracking_abort_deg"]):
            raise ValueError("Require angle tolerance < tracking lead < tracking abort")
        # Promote legacy 0.6 / 2.0 tracking thresholds from robot.json so the
        # reference lead can overcome the ~0.56 deg steady-state load droop.
        if self.config["tracking_lead_deg"] < 1.8 and self.config["angle_tolerance_deg"] >= 0.25:
            self.config["tracking_lead_deg"] = 1.8
            self.config["tracking_abort_deg"] = max(self.config["tracking_abort_deg"], 3.5)
        if self.config["max_speed_deg_s"] > 360:
            raise ValueError("Protocol maximum motor speed is 360 deg/s")
        for motor in self.motors:
            if not isinstance(motor["id"], int) or not 1 <= motor["id"] <= 32:
                raise ValueError("Invalid motor ID")
            if motor["sign"] not in (-1, 1):
                raise ValueError("Motor sign must be -1 or +1")
            motor["ratio"] = _finite(motor["ratio"], "ratio")
            motor["offset_deg"] = _finite(motor["offset_deg"], "offset_deg")
            limits = [_finite(v, "limits_deg") for v in motor["limits_deg"]]
            if motor["ratio"] <= 0 or len(limits) != 2 or limits[0] >= limits[1]:
                raise ValueError("Invalid ratio or joint limits")
            motor["limits_deg"] = limits
            for value in limits:
                if not 0 <= self._motor_deg(motor, value) <= 359.99:
                    raise ValueError("Configured mapping exceeds A6 single-turn range")
        if os.path.realpath(self.motors[0]["device"]) == os.path.realpath(self.motors[1]["device"]):
            raise ValueError("The two independent motors require distinct serial devices")
        self.ports = [_Port(m["device"], int(self.config["baudrate"])) for m in self.motors]
        self._feedback = [{}, {}]
        self._pending_angle_query_at = [None, None]
        self._io_lock = threading.RLock()
        self._move_lock = threading.Lock()
        self._halt_event = threading.Event()
        self._moving = False
        self._opened = False
        self._motion_trace = None
        self._last_targets = [None, None]
        self._last_target_sent_at = [0.0, 0.0]
        self._last_target_acked = [False, False]
        self._settling_bias_centi = [0, 0]
        self._settling_bias_retracted = [False, False]
        self._stationary_probe = [None, None]
        self._active_axes = [True, True]

    @staticmethod
    def _motor_deg(motor, model_deg):
        return motor["offset_deg"] + motor["sign"] * motor["ratio"] * model_deg

    def open(self):
        with self._io_lock:
            if self._opened:
                return self
            try:
                for port in self.ports:
                    port.open()
            except Exception:
                for port in self.ports:
                    port.close()
                raise
            self._feedback = [{}, {}]
            self._pending_angle_query_at = [None, None]
            self._last_targets = [None, None]
            self._last_target_sent_at = [0.0, 0.0]
            self._last_target_acked = [False, False]
            self._settling_bias_centi = [0, 0]
            self._settling_bias_retracted = [False, False]
            self._stationary_probe = [None, None]
            self._active_axes = [True, True]
            self._opened = True
        return self

    def close(self):
        if self._moving:
            self.halt()
        with self._io_lock:
            for port in self.ports:
                port.close()
            self._opened = False

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is not None and self._opened:
            self.halt()
        self.close()

    def _request_feedback(self, *, axes=None, include_speed=True):
        """Query selected axes; passive axes are scheduled separately by move."""
        selected = range(len(self.motors)) if axes is None else axes
        with self._io_lock:
            for command in ((0x94, 0x9C) if include_speed else (0x94,)):
                for i in selected:
                    motor, port = self.motors[i], self.ports[i]
                    if command == 0x94:
                        self._pending_angle_query_at[i] = time.monotonic()
                    port.send(make_frame(command, motor["id"]))
                    if self._motion_trace is not None:
                        self._motion_trace.record_query(motor, command, time.monotonic())

    def _pump(self, timeout_s=0):
        with self._io_lock:
            if not self._opened:
                raise HardwareError("serial_not_open")
            timeout_s = max(0, timeout_s)
            started = time.monotonic()
            try:
                poller = select.poll()
                for port in self.ports:
                    poller.register(port.fd, select.POLLIN)
                ready = dict(poller.poll(math.ceil(timeout_s * 1000)))
                # A readable fd may also have HUP/ERR. Do not accept residual
                # bytes as proof that a failed connection is healthy.
                for port in self.ports:
                    _poll_port_fault(port.device, ready.get(port.fd, 0))
                empty_read = False
                for i, port in enumerate(self.ports):
                    if ready.get(port.fd, 0) & select.POLLIN:
                        try:
                            data = os.read(port.fd, 4096)
                        except BlockingIOError:
                            data = None
                        if data:
                            port.buffer.extend(data)
                        else:
                            # Another reader or a driver readiness race can
                            # leave no bytes. Keep partial frames and do not
                            # refresh feedback; the original timeout applies.
                            empty_read = True
                    while True:
                        frame = extract_frame(port.buffer)
                        if frame is None:
                            break
                        if frame.motor_id != self.motors[i]["id"]:
                            continue
                        now = time.monotonic()
                        try:
                            if frame.command == 0x94:
                                angle = angle_from_reply(frame)
                                query_at = self._pending_angle_query_at[i]
                                self._feedback[i].update(
                                    motor_deg=angle, angle_stamp=now,
                                    angle_query_stamp=(now if query_at is None or query_at > now
                                                       else query_at))
                            elif frame.command in (0x9C, 0xA4, 0xA6):
                                state = state_from_reply(frame)
                                self._feedback[i].update(state, speed_stamp=now)
                                if frame.command == 0xA6:
                                    self._last_target_acked[i] = True
                        except ValueError:
                            # A well-checksummed frame with the wrong payload is
                            # still not valid feedback. It cannot refresh age.
                            continue
                        if self._motion_trace is not None and frame.command in (0x94, 0x9C, 0xA4, 0xA6):
                            self._motion_trace.record_reply(self.motors[i], frame, now)
                if empty_read:
                    # Check once for a fault that arrived after the first
                    # poll. Never spin here waiting for a complete reply.
                    later = dict(poller.poll(0))
                    for port in self.ports:
                        _poll_port_fault(port.device, later.get(port.fd, 0))
                    remaining = timeout_s - (time.monotonic() - started)
                    if remaining > 0:
                        time.sleep(min(0.001, remaining))
            except OSError as exc:
                raise HardwareError("serial_read_failed", {"error": str(exc)}) from exc

    def _snapshot(self, since=None):
        now = time.monotonic()
        angles, speeds, motor_angles, axis_stamps, speed_stamps = [], [], [], [], []
        for i, (motor, reply) in enumerate(zip(self.motors, self._feedback)):
            timeout = (self.config["feedback_timeout_s"] if since is not None or self._active_axes[i]
                       else max(self.config["inactive_feedback_timeout_s"],
                                2 / self.config["inactive_feedback_hz"] + self.config["feedback_timeout_s"]))
            for key in ("angle_stamp", "speed_stamp"):
                stamp = reply.get(key)
                if (stamp is None or now - stamp > timeout
                        or (since is not None and stamp < since)):
                    raise HardwareError("joint_feedback_stale", {
                        "motor": motor["name"], "device": motor["device"],
                        "feedback": "angle" if key == "angle_stamp" else "speed",
                        "allowed_age_s": timeout})
            scale = motor["sign"] * motor["ratio"]
            angle = (reply["motor_deg"] - motor["offset_deg"]) / scale
            if not motor["limits_deg"][0] - .02 <= angle <= motor["limits_deg"][1] + .02:
                raise HardwareError("joint_feedback_out_of_limits", {"motor": motor["name"], "model_deg": angle})
            motor_angles.append(reply["motor_deg"])
            angles.append(math.radians(angle))
            speeds.append(math.radians(reply["speed_deg_s"] / scale))
            axis_stamps.append(min(reply["angle_stamp"], reply["speed_stamp"]))
            speed_stamps.append(reply["speed_stamp"])
        selected = [stamp for stamp, active in zip(axis_stamps, self._active_axes) if active]
        return {"q_rad": angles, "speed_rad_s": speeds, "motor_deg": motor_angles,
                "axis_stamp_monotonic_s": axis_stamps,
                "speed_stamp_monotonic_s": speed_stamps,
                "stamp_monotonic_s": min(selected or axis_stamps)}

    def _read_state(self, timeout_s=None):
        # Drain old replies before starting the fresh query interval.
        self._pump(0)
        since = time.monotonic()
        self._request_feedback()
        next_query = time.monotonic() + 1 / self.config["feedback_rate_hz"]
        deadline = since + (self.config["feedback_timeout_s"] if timeout_s is None else timeout_s)
        while True:
            self._pump(min(0.01, max(0, deadline - time.monotonic())))
            try:
                return self._snapshot(since)
            except HardwareError as exc:
                if exc.reason != "joint_feedback_stale":
                    raise
                if time.monotonic() >= deadline:
                    raise
                if time.monotonic() >= next_query:
                    self._request_feedback()
                    next_query = time.monotonic() + 1 / self.config["feedback_rate_hz"]

    def read_state(self):
        try:
            return self._read_state()
        except HardwareError as exc:
            exc.halt_result = self.halt()
            raise

    def halt(self):
        """Send stop on BOTH ports, then attempt fresh, stable speed feedback.

        ``stop_sent`` alone never means stopped. Confirmation requires both
        fresh angles and speeds, speed below the configured threshold, and
        stable observations for ``stable_s``. A timeout is explicitly reported.
        """
        self._halt_event.set()
        result = {"stop_sent": [False, False], "stopped_confirmed": False,
                  "errors": [], "feedback": None}
        with self._io_lock:
            self._last_targets = [None, None]
            self._last_target_sent_at = [0.0, 0.0]
            self._last_target_acked = [False, False]
            self._settling_bias_centi = [0, 0]
            self._settling_bias_retracted = [False, False]
            self._stationary_probe = [None, None]
            self._active_axes = [True, True]
            for i, (motor, port) in enumerate(zip(self.motors, self.ports)):
                try:
                    port.send(make_frame(0x81, motor["id"]))
                    result["stop_sent"][i] = True
                except Exception as exc:
                    result["errors"].append({"motor": motor["name"], "error": str(exc)})
        if not self._opened:
            return result
        # Allow enough time for the upper arm's passive spring-back (~0.8-1.2s)
        # after 0x81 releases holding torque, plus stable_s of 0/0 confirmation.
        deadline = (time.monotonic()
                    + max(2.0, 3 * self.config["feedback_timeout_s"])
                    + self.config["stable_s"])
        still_since = None
        anchor_angles = None
        while time.monotonic() < deadline:
            try:
                state = self._read_state(timeout_s=min(self.config["feedback_timeout_s"],
                                                       deadline - time.monotonic()))
                result["feedback"] = state
                stationary = all(abs(math.degrees(v)) <= self.config["still_speed_deg_s"]
                                 for v in state["speed_rad_s"])
                if anchor_angles is not None:
                    stationary &= all(abs(math.degrees(a - b)) <= self.config["angle_tolerance_deg"]
                                      for a, b in zip(state["q_rad"], anchor_angles))
                if stationary:
                    if still_since is None:
                        still_since = state["stamp_monotonic_s"]
                        anchor_angles = state["q_rad"]
                    if state["stamp_monotonic_s"] - still_since >= self.config["stable_s"]:
                        result["stopped_confirmed"] = all(result["stop_sent"])
                        break
                else:
                    still_since = None
                    anchor_angles = None
                    # Re-emit 0x81 if an in-flight reply caused the initial
                    # stop frame to be missed and the motor is still moving.
                    with self._io_lock:
                        for i, (motor, port) in enumerate(zip(self.motors, self.ports)):
                            try:
                                port.send(make_frame(0x81, motor["id"]))
                                result["stop_sent"][i] = True
                            except Exception as exc:
                                result["errors"].append({"motor": motor["name"], "error": str(exc)})
                time.sleep(min(1 / self.config["feedback_rate_hz"], max(0, deadline - time.monotonic())))
            except Exception as exc:
                error = {"error": str(exc)}
                if isinstance(exc, HardwareError):
                    error["details"] = exc.details
                result["errors"].append(error)
                still_since = anchor_angles = None
                if (isinstance(exc, HardwareError) and exc.reason == "joint_feedback_stale"
                        and time.monotonic() < deadline):
                    time.sleep(min(1 / self.config["feedback_rate_hz"], max(0, deadline-time.monotonic())))
                    continue
                break
        if not result["stopped_confirmed"]:
            result["reason"] = "stop_not_confirmed_by_feedback"
        return result

    def _validate_phase(self, phase):
        points = phase.get("points")
        if not isinstance(points, list) or not points:
            raise ValueError("Phase requires a non-empty points list")
        result = []
        for point in points:
            stamp = _finite(point["time_s"], "time_s")
            if len(point["q_rad"]) != 2:
                raise ValueError("Each point must contain two joint angles")
            angles = [_finite(v, "q_rad") for v in point["q_rad"]]
            if (not result and abs(stamp) > 1e-9) or (result and stamp <= result[-1][0]):
                raise ValueError("Phase times must start at zero and strictly increase")
            for motor, angle in zip(self.motors, angles):
                deg = math.degrees(angle)
                if not motor["limits_deg"][0] - 1e-9 <= deg <= motor["limits_deg"][1] + 1e-9:
                    raise ValueError("Target outside model joint limits: " + motor["name"])
                # Validate centidegree quantization as well as the float input.
                centi = round(self._motor_deg(motor, deg) * 100)
                if not 0 <= centi <= 35999:
                    raise ValueError("Mapped target outside A6 range")
            if result:
                dt = stamp - result[-1][0]
                for motor, a, b in zip(self.motors, result[-1][1], angles):
                    velocity = abs(math.degrees(b - a)) * motor["ratio"] / dt
                    if velocity > self.config["max_speed_deg_s"] + 1e-6:
                        raise ValueError("Trajectory exceeds motor speed limit: " + motor["name"])
            result.append((stamp, angles))
        return result

    def _clamp_motor_centi(self, motor, centi):
        bounds = [round(self._motor_deg(motor, lim) * 100) for lim in motor["limits_deg"]]
        lo, hi = max(0, min(bounds)), min(35999, max(bounds))
        return max(lo, min(hi, int(round(centi))))

    def _active_command_in_acceptance_band(self, q_rad):
        # This is the last host write, not a drive target-register readback.
        return [key is not None and abs(key[0] / 100 - self._motor_deg(m, math.degrees(q)))
                <= 0.85 * self.config["angle_tolerance_deg"] * m["ratio"] + 1e-9
                for m, key, q in zip(self.motors, self._last_targets, q_rad)]

    def _send_targets(self, q_rad, planned_directions=None, at_endpoint=False,
                      axis_at_endpoint=None, coalesce_steps=False,
                      tolerances_deg=None, allow_bias_axes=None, command_axes=None,
                      planned_velocities=None, nominal_q_rad=None, external_force=False,
                      external_retract=False, axis_min_speed_deg_s=None,
                      speed_tiers=None, command_speed_caps_deg_s=None,
                      recovery_bias_modes=None, recovery_validator=None, adaptive_mode=False,
                      stuck_upshift_axes=None, stuck_angle_boost_axes=None):
        sent = 0
        with self._io_lock:
            if self._halt_event.is_set():
                raise HardwareError("motion_halted")
            for i, (motor, port, q) in enumerate(zip(self.motors, self.ports, q_rad)):
                if command_axes is not None and not command_axes[i]:
                    continue
                current = self._feedback[i]["motor_deg"]
                nominal_q = q if nominal_q_rad is None else nominal_q_rad[i]
                nominal_motor_deg = self._motor_deg(motor, math.degrees(nominal_q))
                command_motor_deg = self._motor_deg(motor, math.degrees(q))
                nominal_target = round(nominal_motor_deg * 100)
                nominal_delta = nominal_motor_deg - current
                axis_end = bool(at_endpoint or (axis_at_endpoint is not None
                                                and i < len(axis_at_endpoint)
                                                and axis_at_endpoint[i]))
                if not axis_end:
                    self._settling_bias_centi[i] = 0
                    self._settling_bias_retracted[i] = False
                decision_at = time.monotonic()
                age = max(0, decision_at - self._feedback[i]["angle_stamp"])
                query_stamp = self._feedback[i].get("angle_query_stamp",
                                                    self._feedback[i]["angle_stamp"])
                query_age = max(age, max(0.0, decision_at - query_stamp))
                send_delay = max(0.0, getattr(port, "_next_send_at", 0.0) - decision_at)
                measured_speed = abs(self._feedback[i].get("speed_deg_s") or 0.0)
                still_motor_speed = self.config["still_speed_deg_s"] * motor["ratio"]
                tolerance_motor_deg = (self.config["angle_tolerance_deg"] if tolerances_deg is None
                                       else tolerances_deg[i]) * motor["ratio"]
                lead_motor_deg = self.config["tracking_lead_deg"] * motor["ratio"]
                abort_motor_deg = self.config["tracking_abort_deg"] * motor["ratio"]
                last_key = self._last_targets[i]
                speed_tier = 0 if speed_tiers is None else speed_tiers[i]
                if type(speed_tier) is not int or not -4 <= speed_tier <= 4:
                    raise ValueError("invalid discrete speed_tier")
                speed_floor = 0. if axis_min_speed_deg_s is None else axis_min_speed_deg_s[i]
                command_cap = (self.config["max_speed_deg_s"] if command_speed_caps_deg_s is None
                               else command_speed_caps_deg_s[i])
                if not 0 <= speed_floor <= command_cap <= 360:
                    raise ValueError("invalid A6 recovery speed limits")
                previous_tier = 0 if last_key is None or len(last_key) < 4 else last_key[3]
                tier_changed = last_key is not None and previous_tier != speed_tier

                # A single zero-speed sample (especially one preceding the last
                # A6) is not proof that a newly commanded joint has stopped.
                speed_stamp = self._feedback[i].get("speed_stamp", 0.0)
                sample_stamp = min(self._feedback[i]["angle_stamp"], speed_stamp)
                fresh_stationary = (
                    sample_stamp >= self._last_target_sent_at[i]
                    and max(age, decision_at - speed_stamp) <= self.config["feedback_timeout_s"]
                    and measured_speed <= still_motor_speed)
                probe = self._stationary_probe[i]
                if not fresh_stationary:
                    self._stationary_probe[i] = None
                elif (probe is None or sample_stamp < probe[0]
                      or abs(current-probe[1]) > still_motor_speed * max(0., sample_stamp-probe[0]) + 0.015):
                    self._stationary_probe[i] = (sample_stamp, current)
                probe = self._stationary_probe[i]
                stationary_confirmed = (fresh_stationary and probe is not None
                                        and sample_stamp-probe[0] >= 0.10 - 1e-9)
                wire_time = 14 * 10 / self.config["baudrate"]
                stationary_uncertainty = (
                    still_motor_speed * (query_age + send_delay + wire_time) + 0.015
                )
                moving_uncertainty = (
                    max(self.config["max_speed_deg_s"], command_cap,
                        0. if last_key is None else last_key[2]/100., measured_speed)
                    * (query_age + send_delay + wire_time) + 0.015
                )
                same_dir_uncertainty = (
                    stationary_uncertainty if stationary_confirmed
                    else moving_uncertainty
                )

                retract_latch_centi = None
                pending_bias = self._settling_bias_centi[i]
                pending_retracted = self._settling_bias_retracted[i]
                active_in_band = (last_key is not None and
                    abs(last_key[0] / 100 - nominal_motor_deg) <= .85 * tolerance_motor_deg + 1e-9)
                latch_locked = axis_end and pending_retracted and active_in_band
                if (not adaptive_mode and axis_end and nominal_q_rad is None
                        and last_key is not None and not latch_locked):
                    max_hold_offset_centi = max(1, round(0.45 * tolerance_motor_deg * 100))
                    active_offset_deg = abs(last_key[0] / 100 - nominal_motor_deg)
                    approaching_band = (
                        pending_bias != 0 and nominal_delta * pending_bias > 0
                        and ((measured_speed > still_motor_speed
                              and abs(nominal_delta) <= tolerance_motor_deg + .25 * motor["ratio"])
                             or abs(nominal_delta) <= tolerance_motor_deg + 1e-9))
                    in_band_with_overdrive = (
                        abs(nominal_delta) <= tolerance_motor_deg + 1e-9
                        and active_offset_deg > .45 * tolerance_motor_deg + 1e-9)
                    if approaching_band or in_band_with_overdrive:
                        step_sign = 1 if last_key[1] == 0 else -1
                        min_ahead_centi = round(current * 100) + step_sign * max(
                            3, math.ceil((same_dir_uncertainty + .02) * 100))
                        ideal = nominal_target + step_sign * min(
                            max_hold_offset_centi, round(.12 * motor["ratio"] * 100))
                        low = math.ceil((nominal_motor_deg - .85 * tolerance_motor_deg) * 100 - 1e-9)
                        high = math.floor((nominal_motor_deg + .85 * tolerance_motor_deg) * 100 + 1e-9)
                        if low <= high:
                            candidate = (max(ideal, min_ahead_centi) if step_sign > 0
                                         else min(ideal, min_ahead_centi))
                            candidate = self._clamp_motor_centi(motor, max(low, min(high, candidate)))
                            candidate_delta = candidate / 100 - current
                            # Clipping into the band must never silently reverse
                            # the A6 direction or consume its uncertainty margin.
                            if (step_sign * candidate_delta > same_dir_uncertainty + 1e-9
                                    and abs(candidate / 100 - nominal_motor_deg) <= .85 * tolerance_motor_deg + 1e-9
                                    and abs(candidate / 100 - nominal_motor_deg) < active_offset_deg):
                                retract_latch_centi = candidate
                                pending_bias = candidate - nominal_target
                                pending_retracted = True
                    elif (not pending_retracted and stationary_confirmed
                          and allow_bias_axes is not None and allow_bias_axes[i]
                          and decision_at - self._last_target_sent_at[i] >= self.config["settling_bias_wait_s"]
                          and tolerance_motor_deg + self.config["settling_bias_trigger_margin_deg"] * motor["ratio"] < abs(nominal_delta) < abort_motor_deg):
                        cap = round(self.config["settling_bias_max_deg"] * motor["ratio"] * 100)
                        step = max(1, round(self.config["settling_bias_step_deg"] * motor["ratio"] * 100))
                        sign = 1 if nominal_delta > 0 else -1
                        if pending_bias * sign < 0:
                            pending_bias = 0  # Remove old bias before any reverse boost.
                        else:
                            pending_bias = sign * min(cap, abs(pending_bias) + step)

                # Explicit angle stage follows exhausted speed adjustments;
                # it cannot run on the old elapsed-time bias trigger.
                bias_mode = 0 if recovery_bias_modes is None else recovery_bias_modes[i]
                # A temporary drive target must be recalled against the ORIGINAL
                # node, independently of the healthy joint. Recall early while
                # approaching its band, or whenever it has crossed that node.
                old_bias = self._settling_bias_centi[i]
                axis_bias_retraction = bool(adaptive_mode and nominal_q_rad is None
                    and old_bias and (bias_mode == -2 or abs(nominal_delta) <= tolerance_motor_deg + 1e-9
                        or nominal_delta * old_bias <= 0
                        or (measured_speed > still_motor_speed
                            and abs(nominal_delta) <= tolerance_motor_deg + .25*motor["ratio"])))
                if bias_mode in (-1, 1) and nominal_q_rad is None and not axis_bias_retraction:
                    step = self.motion_policy["angle_step_deg"] * motor["ratio"]
                    cap = self.motion_policy["angle_max_deg"] * motor["ratio"]
                    levels = self.motion_policy['angle_steps_deg']
                    magnitude = abs(self._settling_bias_centi[i])/100
                    if bias_mode > 0:
                        sign = 1 if nominal_delta > 0 else -1
                        next_bias = (next((v*motor['ratio'] for v in levels
                            if v*motor['ratio'] > magnitude+.005), cap) if levels else magnitude+step)
                        desired_bias = sign * min(cap, next_bias)
                    elif self._settling_bias_centi[i]:
                        old_bias = self._settling_bias_centi[i]/100
                        smaller = (max((v*motor['ratio'] for v in levels
                            if v*motor['ratio'] < magnitude-.005), default=0.) if levels else max(0., magnitude-step))
                        desired_bias = math.copysign(smaller, old_bias)
                    else:
                        desired_bias = math.copysign(min(step, .5*abs(nominal_delta)), -nominal_delta)
                    pending_bias = round(desired_bias*100)
                    pending_retracted = False
                    retract_latch_centi = None
                    latch_locked = False
                if axis_bias_retraction:
                    pending_bias = 0
                    pending_retracted = True
                    retract_latch_centi = nominal_target
                    latch_locked = False
                if nominal_q_rad is not None:
                    target = self._clamp_motor_centi(motor, round(command_motor_deg * 100))
                    pending_bias = target - nominal_target
                    pending_retracted = external_retract
                elif latch_locked:
                    target = last_key[0]
                elif retract_latch_centi is not None:
                    target = retract_latch_centi
                else:
                    target = self._clamp_motor_centi(
                        motor, nominal_target + (pending_bias if axis_end else 0))
                if adaptive_mode and nominal_q_rad is None:
                    # Record the bounded target actually sent, not an unclipped
                    # requested offset. Subsequent .2-degree steps start here.
                    pending_bias = target - nominal_target
                if (adaptive_mode and (target != nominal_target or axis_bias_retraction)
                        and (last_key is None or target != last_key[0])
                        and (recovery_validator is None or not recovery_validator(i, target))):
                    if self._motion_trace is not None:
                        self._motion_trace.record_action({"motor": motor["name"], "status": "skipped",
                            "reason": "uncertified_recovery_angle_change", "target_motor_deg": target/100})
                    continue
                delta = target / 100 - current
                direction = 0 if delta > 0 else 1
                # Use same_dir_uncertainty when continuing in the same direction
                # only after confirmed stillness; first sends and reversals use the full budget.
                if (last_key is not None and direction == last_key[1]) or (last_key is None and stationary_confirmed):
                    uncertainty = same_dir_uncertainty
                else:
                    uncertainty = moving_uncertainty

                # A diagnosed stall with fresh, sustained stationary feedback
                # can resend the *same* target and direction at a higher tier.
                # The new cap is not evidence of motion before that write.
                stationary_stuck_upshift = (
                    stuck_upshift_axes is not None and stuck_upshift_axes[i]
                    and tier_changed and speed_tier > 0 and last_key is not None
                    and target == last_key[0] and direction == last_key[1]
                    and speed_floor > last_key[2]/100. + .005
                    and stationary_confirmed)
                # All other tier changes, reversals and angle adjustments keep
                # the original moving-direction uncertainty budget.
                if (tier_changed or (adaptive_mode and (target != nominal_target
                        or (last_key is not None and last_key[0] != nominal_target)))):
                    uncertainty = max(uncertainty, moving_uncertainty)
                stationary_stuck_angle_boost = bool(
                    adaptive_mode and stuck_angle_boost_axes is not None
                    and stuck_angle_boost_axes[i] and not axis_bias_retraction
                    and stationary_confirmed and last_key is not None
                    and direction == last_key[1] and target != last_key[0]
                    and abs(pending_bias) > abs(old_bias)
                    and (target-last_key[0])*(1 if direction == 0 else -1) > 0)
                stationary_bias_retraction = bool(axis_bias_retraction and stationary_confirmed)
                if (stationary_stuck_upshift or stationary_stuck_angle_boost
                        or stationary_bias_retraction):
                    uncertainty = stationary_uncertainty
                trace = self._motion_trace
                event = None if trace is None else {
                    "motor": motor["name"], "device": motor["device"],
                    "decision_monotonic_s": decision_at,
                    "reference_q_deg": math.degrees(q), "target_motor_deg": target / 100,
                    "nominal_target_motor_deg": nominal_motor_deg,
                    "nominal_delta_motor_deg": nominal_delta,
                    "stationary_direction_confirmed": stationary_confirmed,
                    "retraction_pending": retract_latch_centi is not None,
                    "settling_bias_motor_deg": (target - nominal_target) / 100,
                    "feedback_motor_deg": current, "delta_motor_deg": delta,
                    "feedback_speed_motor_deg_s": self._feedback[i].get("speed_deg_s"),
                    "speed_tier": speed_tier, "previous_speed_tier": previous_tier,
                    "axis_min_speed_deg_s": speed_floor, "command_speed_cap_deg_s": command_cap,
                    "angle_feedback_age_s": age, "direction_uncertainty_deg": uncertainty,
                    "stationary_stuck_upshift": bool(stationary_stuck_upshift),
                    "stationary_stuck_angle_boost": stationary_stuck_angle_boost,
                    "axis_bias_retraction": axis_bias_retraction,
                    "stationary_bias_retraction": stationary_bias_retraction,
                    "speed_feedback_age_s": (None if self._feedback[i].get("speed_stamp") is None
                                             else decision_at - self._feedback[i]["speed_stamp"]),
                }
                settling_change = speed_tier < 0 and tier_changed
                if settling_change and ((not at_endpoint and not adaptive_mode) or last_key is None
                        or target != last_key[0] or direction != last_key[1]
                        or not stationary_confirmed
                        or abs(delta) <= max(uncertainty,
                            self.recovery_config["settling_safe_distance_deg"] * motor["ratio"])):
                    if trace is not None:
                        trace.record_action({**event, "status": "skipped",
                                             "reason": "unsafe_settling_speed_change"})
                    continue
                if abs(delta) <= uncertainty or (
                        axis_end and nominal_q_rad is None and retract_latch_centi is None
                        and not settling_change
                        and abs(nominal_delta) <= tolerance_motor_deg + 1e-9
                        and last_key is not None):
                    if trace is not None:
                        trace.record_action({**event, "status": "skipped",
                                             "reason": "within_direction_uncertainty"})
                    continue
                planned_sign = (0 if planned_directions is None or i >= len(planned_directions)
                                else planned_directions[i])
                planned_motor_dir = (None if planned_sign == 0
                                     else (0 if motor["sign"] * planned_sign > 0 else 1))
                # Prevent mid-trajectory reverse A6 commands when an axis slightly
                # leads its interpolated reference along the planned motion direction.
                if (not at_endpoint and planned_motor_dir is not None
                        and not stationary_bias_retraction and direction != planned_motor_dir
                        and (abs(delta) <= lead_motor_deg or measured_speed > still_motor_speed)):
                    if trace is not None:
                        trace.record_action({**event, "status": "skipped",
                                             "reason": "within_direction_uncertainty"})
                    continue
                # On stationary segments or at the endpoint, never issue a reverse
                # A6 correction while the motor is still decelerating or already
                # within angle tolerance of the target.
                if ((axis_end or (planned_directions is not None and planned_motor_dir is None))
                        and not stationary_bias_retraction
                        and last_key is not None and direction != last_key[1]
                        and (abs(nominal_delta if axis_end else delta) <= tolerance_motor_deg
                             or measured_speed > still_motor_speed)):
                    if trace is not None:
                        trace.record_action({**event, "status": "skipped",
                                             "reason": "within_direction_uncertainty"})
                    continue
                velocity = (0.0 if planned_velocities is None else
                            abs(math.degrees(planned_velocities[i])) * motor["ratio"])
                desired_speed = (self.config["speed_feedforward_gain"] * velocity
                                 + self.config["speed_error_gain_s"] * abs(delta))
                # Continuous formula changes do not trigger retransmission.
                # A latched recovery floor survives all nodes of this monotonic run.
                base_speed = min(self.config["max_speed_deg_s"],
                                 max(self.config["min_speed_deg_s"], desired_speed))
                speed = max(1, round(min(command_cap, max(speed_floor, base_speed)) * 100))
                wire = position_frame(motor["id"], direction, target, speed)
                key = (target, direction, speed, speed_tier)
                if last_key is not None and last_key[0] == target:
                    # Only re-send an identical target if the previous 0xA6 reply
                    # was never received (unacknowledged frame loss). Once acked,
                    # repeating the exact same target every 500ms does not add torque.
                    stalled_unacked = (
                        direction == last_key[1]
                        and not self._last_target_acked[i]
                        and abs(nominal_delta if axis_end else delta) > max(uncertainty, tolerance_motor_deg)
                        and measured_speed <= still_motor_speed
                        and decision_at - self._last_target_sent_at[i] >= max(
                            0.25, 5 / self.config["command_rate_hz"])
                    )
                    # Never turn a same-position resend into a new forced
                    # A6 direction. Only a discrete tier transition may update
                    # an acknowledged target once; raw speed rounding is ignored.
                    same_direction_tier_change = tier_changed and direction == last_key[1]
                    if direction != last_key[1] or not (stalled_unacked or same_direction_tier_change):
                        if trace is not None:
                            trace.record_action({**event, "status": "skipped",
                                                 "reason": "unchanged_active_target"})
                        continue
                elif (coalesce_steps and not at_endpoint and retract_latch_centi is None
                      and not external_force and pending_bias == self._settling_bias_centi[i]
                      and last_key is not None
                      and direction == last_key[1]):
                    # Merge only temporarily; a changed target gets an exit.
                    step_deg = abs(target - last_key[0]) / 100.0
                    if (step_deg < self.config["coalesce_step_deg"] * motor["ratio"]
                            and decision_at - self._last_target_sent_at[i] < self.config["coalesce_wait_s"]):
                        if trace is not None:
                            trace.record_action({**event, "status": "skipped", "reason": "coalesced_small_step"})
                        continue
                try:
                    port.send(wire, cancel_event=self._halt_event)
                except BaseException as exc:
                    if trace is not None:
                        trace.record_action({**event, "status": "send_failed", "direction": direction,
                                             "frame_hex": wire.hex(" "), "error_type": type(exc).__name__,
                                             "error": str(exc), "call_ended_monotonic_s": time.monotonic()})
                    raise
                completed_at = time.monotonic()
                self._settling_bias_centi[i] = pending_bias if axis_end else 0
                self._settling_bias_retracted[i] = pending_retracted if axis_end else False
                self._stationary_probe[i] = None
                self._last_targets[i] = key
                self._last_target_sent_at[i] = completed_at
                self._last_target_acked[i] = False
                sent += 1
                if trace is not None:
                    trace.record_action({**event, "status": "sent", "direction": direction,
                                         "frame_hex": wire.hex(" "),
                                         "write_completed_monotonic_s": completed_at})
        return sent

    @staticmethod
    def _cartesian_target(meta, arm, nominal, lead_mm):
        direction = meta["direction"]
        start, end = meta["start_xy_mm"], meta["end_xy_mm"]
        edge = [b-a for a, b in zip(start, end)]
        length = math.hypot(*edge)
        if length <= 1e-9 or abs(lead_mm) <= 1e-12:
            return list(nominal)
        unit = [v/length for v in edge]
        centre = arm.slider_center(nominal, direction)
        progress = sum((v-a)*u for v, a, u in zip(centre, start, unit))
        progress = min(length, max(0., progress + lead_mm))
        target = [a + progress*u for a, u in zip(start, unit)]
        return list(arm.inverse(target, direction, seed=nominal))

    def _cartesian_pair_applied(self, command, arm, direction):
        """Accept a written pair or a negligible quantized difference.

        The two serial writes are not atomic. Keep the exact last-written
        targets in diagnostics; this only permits <=0.03 deg per-axis and
        <=0.1 mm model-space difference when one tiny update is suppressed.
        """
        retained = []
        for i, (motor, q) in enumerate(zip(self.motors, command)):
            key = self._last_targets[i]
            motor_deg = (self._feedback[i]["motor_deg"] if key is None else key[0]/100)
            model_deg = (motor_deg-motor["offset_deg"])/(motor["sign"]*motor["ratio"])
            if abs(model_deg-math.degrees(q)) > .03+1e-9:
                return False
            retained.append(math.radians(model_deg))
        a, b = arm.slider_center(retained, direction), arm.slider_center(command, direction)
        return math.hypot(a[0]-b[0], a[1]-b[1]) <= .1+1e-9

    def _send_timed_target(self, q_rad, trace):
        """Write both absolute A4 targets, without consulting tracking feedback."""
        sent = 0
        with self._io_lock:
            for i, (motor, port, q) in enumerate(zip(self.motors, self.ports, q_rad)):
                if self._halt_event.is_set():
                    raise HardwareError("motion_halted")
                target = round(self._motor_deg(motor, math.degrees(q)) * 100)
                speed = max(1, round(self.free_timed_speed * 100))
                wire = multi_turn_position_frame(motor["id"], target, speed)
                event = {"motor": motor["name"], "decision_monotonic_s": time.monotonic(),
                         "command": "0xa4", "target_motor_deg": target/100,
                         "nominal_target_motor_deg": target/100,
                         "speed_limit_deg_s": self.free_timed_speed,
                         "bias_motor_deg": 0., "tx_hex": wire.hex(" ")}
                try:
                    port.send(wire, cancel_event=self._halt_event)
                except BaseException:
                    trace.record_action({**event, "status": "send_failed"})
                    raise
                # The legacy A6 cache includes a forced direction; an A4 write
                # invalidates that cache before a later Push/withdraw phase.
                self._last_targets[i] = None
                self._last_target_sent_at[i] = time.monotonic()
                self._last_target_acked[i] = False
                trace.record_action({**event, "status": "sent",
                                     "write_completed_monotonic_s": time.monotonic()})
                sent += 1
        return sent

    def _move_timed(self, phase, health_check=None):
        """Send small A4 targets at a fixed period; qualify only the end."""
        points = timed_free_points(self._validate_phase(phase),
                                   self.free_timed_step, self.free_timed_period)
        trace = _MotionTrace(phase, points, self.config, self.motors)
        name = phase["phase"]
        controller = name + "_timed"
        if not self._move_lock.acquire(blocking=False):
            raise HardwareError("motion_already_running")
        self._moving = True
        self._halt_event.clear()
        self._active_axes = [True, True]
        self._settling_bias_centi = [0, 0]
        self._settling_bias_retracted = [False, False]
        self._stationary_probe = [None, None]
        self._motion_trace = trace
        state = target = elapsed = None
        sent_count = 0
        samples, schedule = deque(maxlen=10000), []
        sample_count = 0
        trace.execution.update(controller=controller, stage="tracking",
                               position_command="0xa4",
                               fixed_speed_deg_s=self.free_timed_speed,
                               max_step_deg=self.free_timed_step,
                               period_s=self.free_timed_period,
                               automatic_retry=False, schedule=schedule)
        try:
            state = self._read_state()
            if any(abs(math.degrees(a-b)) > self.config["angle_tolerance_deg"]
                   for a, b in zip(state["q_rad"], points[0][1])):
                raise HardwareError("trajectory_start_mismatch")
            started = time.monotonic()
            trace.started = started
            target, elapsed = points[0][1], 0.
            index = 1
            next_feedback = started
            last_sample_stamp = None
            dispatch_period = max(self.free_timed_period, 1/self.config["command_rate_hz"])
            last_write = started - dispatch_period
            end_at = None
            qualifier = GoalQualification(self.config["angle_tolerance_deg"],
                self.config["still_speed_deg_s"], self.config["goal_hold_s"], self.config["stable_s"])
            deadline = started + points[-1][0] + self.config["move_timeout_margin_s"]
            while True:
                if self._halt_event.is_set():
                    raise HardwareError("motion_halted")
                if health_check is not None:
                    health_check()
                now = time.monotonic()
                if now >= next_feedback:
                    self._request_feedback()
                    next_feedback = time.monotonic() + 1/self.config["feedback_rate_hz"]
                self._pump(0)
                state = self._snapshot()
                if state["stamp_monotonic_s"] != last_sample_stamp:
                    samples.append({"elapsed_s": time.monotonic()-started,
                        "axis_stamp_monotonic_s": list(state["axis_stamp_monotonic_s"]),
                        "speed_stamp_monotonic_s": list(state["speed_stamp_monotonic_s"]),
                        "q_deg": [math.degrees(q) for q in state["q_rad"]],
                        "speed_deg_s": [math.degrees(v) for v in state["speed_rad_s"]],
                        "output_raw": [r.get("current_raw") for r in self._feedback]})
                    sample_count += 1
                    last_sample_stamp = state["stamp_monotonic_s"]
                now = time.monotonic()
                if index < len(points) and now >= deadline:
                    raise HardwareError(controller + "_schedule_timeout")
                # No measured error, progress, speed, or acknowledgement gates
                # advancement. Late nodes stay ordered; never burst or drop them.
                if (index < len(points) and now-started >= points[index][0]
                        and now-last_write >= dispatch_period):
                    elapsed, target = points[index]
                    last_write = time.monotonic()
                    written = self._send_timed_target(target, trace)
                    sent_count += written
                    completed = time.monotonic()
                    schedule.append({"index": index, "planned_time_s": elapsed,
                        "write_started_elapsed_s": last_write-started,
                        "write_completed_elapsed_s": completed-started,
                        "lateness_s": completed-started-elapsed, "axes_written": written})
                    index += 1
                trace.execution.update(next_waypoint_index=index,
                    waypoints_dispatched=len(schedule), motion_commands_sent=sent_count)
                if index == len(points):
                    if end_at is None:
                        end_at = time.monotonic()
                        target, elapsed = points[-1][1], points[-1][0]
                        trace.execution.update(stage="settling", dispatch_completed_elapsed_s=end_at-started)
                    state = self._snapshot()
                    ready = all(t >= end_at for t in state["axis_stamp_monotonic_s"])
                    completed = qualifier.observe(state, target, ready)
                    trace.execution.update(goal_stable_s=qualifier.stable_s,
                        goal_stability_resets=qualifier.resets,
                        goal_speed_blips_tolerated=qualifier.blips_tolerated,
                        goal_stability_events=list(qualifier.events))
                    if completed:
                        trace.execution.update(stage="holding", goal_stable_s=qualifier.stable_s)
                        trace.execution.update(feedback_samples=list(samples),
                            dropped_feedback_samples=sample_count-len(samples))
                        return {"status": "completed", "phase": name, "feedback": state,
                            "motion_commands_sent": sent_count, "elapsed_s": time.monotonic()-started,
                            "motion_diagnostics": _capture_motion_trace(trace, "completed",
                                self._feedback, state, target, elapsed, None, None)}
                    if qualifier.failure:
                        raise HardwareError(qualifier.failure)
                    if qualifier.armed:
                        trace.execution["stage"] = "holding"
                    if not qualifier.armed and time.monotonic() >= end_at + self.config["goal_settle_timeout_s"]:
                        raise HardwareError("motion_goal_settle_timeout")
                    if time.monotonic() >= end_at + self.config["goal_settle_timeout_s"] + self.config["goal_hold_s"]:
                        raise HardwareError("motion_goal_hold_timeout" if qualifier.armed
                                            else "motion_goal_settle_timeout")
                wait_s = .005
                if index < len(points):
                    due = max(started+points[index][0], last_write+dispatch_period)
                    wait_s = min(wait_s, max(0., due-time.monotonic()))
                self._pump(wait_s)
        except BaseException as exc:
            trace.execution.update(feedback_samples=list(samples),
                dropped_feedback_samples=sample_count-len(samples))
            diagnostics = _capture_motion_trace(trace, "failed", self._feedback,
                                                state, target, elapsed, None, None)
            stop = self.halt()
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            details = {"phase": name, "controller": controller, "automatic_retry": False,
                       "motion_commands_sent": sent_count, "motion_diagnostics": diagnostics}
            if isinstance(exc, HardwareError):
                exc.details.update(details)
                exc.halt_result = stop
                raise
            raise HardwareError("motion_failed", {**details, "error": str(exc)}, stop) from exc
        finally:
            self._motion_trace = None
            self._active_axes = [True, True]
            self._moving = False
            self._move_lock.release()

    def move(self, phase, health_check=None):
        """Follow preplanned command nodes, then settle and hold the final goal.

        Free and Push use the same timed baseline when configured.
        Other phases retain their existing feedback-driven executor.
        """
        if phase.get("phase") in ("free", "push", "push_correction") and self.free_timed:
            return self._move_timed(phase, health_check)
        points = self._validate_phase(phase)
        trace = _MotionTrace(phase, points, self.config, self.motors)
        if not self._move_lock.acquire(blocking=False):
            raise HardwareError("motion_already_running")
        self._moving = True
        self._halt_event.clear()
        self._settling_bias_centi = [0, 0]
        self._settling_bias_retracted = [False, False]
        self._stationary_probe = [None, None]
        self._active_axes = [True, True]
        self._motion_trace = trace
        state = target = elapsed = progress_at = progress_q = None
        adaptive = self.motion_policy["enabled"]
        recovery = (MotionRecovery(self.recovery_config, self.motors, phase.get("phase"), self.motion_policy,
                        (self.config["min_speed_deg_s"],self.config["max_speed_deg_s"],self.config["still_speed_deg_s"]))
                    if adaptive else SpeedRecovery(self.recovery_config, self.motors, phase.get("phase")))
        sent_count = 0
        try:
            state = self._read_state()
            if any(abs(math.degrees(a-b)) > self.config["angle_tolerance_deg"]
                   for a, b in zip(state["q_rad"], points[0][1])):
                raise HardwareError("trajectory_start_mismatch", {"actual_q_rad": state["q_rad"],
                                                                  "planned_q_rad": points[0][1]})
            cursor = WaypointCursor(points, self.config["waypoint_tolerance_ratio"],
                                    1/self.config["command_rate_hz"],
                                    zero_span_tolerance_deg=self.config["angle_tolerance_deg"])
            # Quantization-equivalent paths need no physical command on that axis.
            self._active_axes = [len({round(self._motor_deg(motor, math.degrees(q[i]))*100)
                                      for _, q in points}) > 1
                                 for i, motor in enumerate(self.motors)]
            cursor.active_axes = tuple(self._active_axes)
            active_indices = [i for i, value in enumerate(self._active_axes) if value]
            passive_indices = [i for i, value in enumerate(self._active_axes) if not value]
            qualifier = GoalQualification(self.config["angle_tolerance_deg"],
                                          self.config["still_speed_deg_s"],
                                          self.config["goal_hold_s"], self.config["stable_s"],
                                          active_axes=self._active_axes)
            started = time.monotonic()
            trace.started = started
            target = cursor.target()
            elapsed = cursor.elapsed
            progress_at, progress_q = [started, started], list(state["q_rad"])
            passive_anchor = list(state["q_rad"])
            last_command = last_feedback = last_passive = started-10
            stage, stage_started = "tracking", started
            endpoint_at = endpoint_due = settling_deadline = endpoint_deadline = holding_at = None
            total_deadline = (started + points[-1][0] + self.config["move_timeout_margin_s"]
                              + len(points)*self.config["no_progress_s"])
            cart_meta = phase.get("cartesian_compensation")
            # The adaptive policy uses independent temporary joint targets.
            # Keep line metadata for nominal planning and face checks only.
            # Legacy non-adaptive execution retains its historical line mode.
            meta = None if adaptive else cart_meta
            cart_arm = None
            if cart_meta is not None:
                if phase.get("phase") not in ("push", "push_correction", "approach", "withdraw") or cart_meta.get("allow_endpoint_extension"):
                    raise ValueError("unsupported Cartesian compensation policy")
                cart_arm = PlanarArm(cart_meta["geometry"])
            cart_lead = 0.
            cart_retracted = False
            pending_cart = None

            geometry_context = phase.get("motion_recovery_geometry")
            geometry_scene = None
            if adaptive and geometry_context is not None:
                from .scene import SimpleScene
                geometry_scene = SimpleScene(geometry_context["scene"])

            def recovery_angle_safe(axis, target_centi):
                if geometry_scene is None and phase.get("phase") not in ("push", "push_correction"):
                    return False
                motor = self.motors[axis]
                desired = math.radians((target_centi/100-motor["offset_deg"])/(motor["sign"]*motor["ratio"]))
                if not motor["limits_deg"][0] <= math.degrees(desired) <= motor["limits_deg"][1]:
                    return False
                for other_from_command in (False, True):
                    start = list(state["q_rad"])
                    other = 1-axis
                    key = self._last_targets[other]
                    if other_from_command and key is not None:
                        m = self.motors[other]
                        start[other] = math.radians((key[0]/100-m["offset_deg"])/(m["sign"]*m["ratio"]))
                    end = list(start); end[axis] = desired
                    samples = max(1, math.ceil(abs(desired-start[axis])/math.radians(.1)))
                    for k in range(samples+1):
                        q = [a+(b-a)*k/samples for a,b in zip(start,end)]
                        if any(not m["limits_deg"][0] <= math.degrees(value) <= m["limits_deg"][1]
                               for m,value in zip(self.motors,q)):
                            return False
                        if cart_arm is not None and cart_arm.world_normal(q,cart_meta["direction"])[0] * (1 if cart_meta["direction"] == "+x" else -1) <= 0:
                            return False
                        if geometry_scene is not None and geometry_scene.check(q, geometry_context["sliders"], geometry_context["contact"]):
                            return False
                return True

            def commands_ready():
                for i in active_indices:
                    key = self._last_targets[i]
                    if key is None:
                        # An already acceptable physical position needs no dummy A6.
                        if abs(math.degrees(state["q_rad"][i]-points[-1][1][i])) > self.config["angle_tolerance_deg"]:
                            return False
                        continue
                    nominal = self._motor_deg(self.motors[i], math.degrees(points[-1][1][i]))
                    if adaptive and self._settling_bias_centi[i] != 0:
                        return False
                    if abs(key[0]/100-nominal) > .85*self.config["angle_tolerance_deg"]*self.motors[i]["ratio"]+1e-9:
                        return False
                    if state["axis_stamp_monotonic_s"][i] < max(endpoint_at, self._last_target_sent_at[i]):
                        return False
                return True

            def record(at):
                errors = [abs(math.degrees(a-b)) for a, b in zip(state["q_rad"], points[-1][1])]
                trace.execution.update({
                    "stage": stage, "stage_elapsed_s": at-stage_started,
                    "speed_recovery": recovery.snapshot(),
                    "wall_elapsed_s": at-started, "clock": "discrete_waypoints",
                    "waypoint_index": cursor.index, "waypoint_count": len(points),
                    "waypoint_tolerances_deg": list(cursor.tolerances_deg),
                    "waypoint_tolerance_ratio": cursor.tolerance_ratio,
                    "reference_elapsed_s": cursor.elapsed,
                    "active_axes": list(self._active_axes),
                    "inactive_feedback_hz": self.config["inactive_feedback_hz"],
                    "actual_q_deg": [math.degrees(q) for q in state["q_rad"]],
                    "actual_speed_deg_s": [math.degrees(v) for v in state["speed_rad_s"]],
                    "final_goal_errors_deg": errors,
                    "goal_arrived": qualifier.armed, "goal_stable_s": qualifier.stable_s,
                    "goal_guard_failure": qualifier.failure,
                    "goal_stability_resets": qualifier.resets,
                    "goal_speed_blips_tolerated": qualifier.blips_tolerated,
                    "goal_stability_events": list(qualifier.events),
                    "active_command_motor_deg": [None if k is None else k[0]/100 for k in self._last_targets],
                    "active_command_in_acceptance_band": self._active_command_in_acceptance_band(points[-1][1]),
                    "active_command_acknowledged": list(self._last_target_acked),
                    "settling_bias_deg": [b/100 for b in self._settling_bias_centi],
                    "settling_bias_retracted": list(self._settling_bias_retracted),
                    "bias_policy": "independent_axis_error_boost" if adaptive else "legacy",
                    "cartesian_bias_lead_mm": cart_lead,
                    "cartesian_bias_retracted": cart_retracted,
                    "cartesian_command_pending": pending_cart is not None,
                    "endpoint_feedback_ready": bool(endpoint_at is not None and commands_ready()),
                    "holding_commands_frozen": qualifier.armed,
                })

            trace.record_stage(stage, started, self._feedback, state, target, elapsed, progress_at, progress_q)
            if not active_indices:
                record(started)
                return {"status": "completed", "phase": phase.get("phase"), "feedback": state,
                        "motion_commands_sent": 0, "elapsed_s": 0., "no_motion": True,
                        "motion_diagnostics": _capture_motion_trace(trace, "completed", self._feedback,
                            state, target, elapsed, progress_at, progress_q)}
            while True:
                if health_check is not None:
                    health_check()
                now = time.monotonic()
                if self._halt_event.is_set():
                    raise HardwareError("motion_halted")
                queried = False
                if now-last_feedback >= 1/self.config["feedback_rate_hz"]:
                    self._request_feedback(axes=active_indices)
                    last_feedback = now
                    queried = True
                if passive_indices and now-last_passive >= 1/self.config["inactive_feedback_hz"]:
                    self._request_feedback(axes=passive_indices)
                    last_passive = now
                    queried = True
                self._pump(.006 if queried else 0.)
                state = self._snapshot()
                now = time.monotonic()
                for i in passive_indices:
                    if abs(math.degrees(state["q_rad"][i]-passive_anchor[i])) > self.config["inactive_drift_deg"]:
                        raise HardwareError("inactive_axis_drift", {"motor": self.motors[i]["name"]})
                    if abs(math.degrees(state["speed_rad_s"][i])) > 10*self.config["still_speed_deg_s"]:
                        raise HardwareError("inactive_axis_moving", {"motor": self.motors[i]["name"]})
                # Do not carry an out-of-band overdrive target into the next
                # node just because measured position briefly entered its band.
                point_tols = cursor.tolerances_deg
                bias_recovered = all(self._settling_bias_centi[i] == 0
                    or (not adaptive and self._last_targets[i] is not None
                        and abs(self._last_targets[i][0]/100-self._motor_deg(
                            self.motors[i], math.degrees(cursor.target()[i])))
                        <= .85*point_tols[i]*self.motors[i]["ratio"]+1e-9)
                    for i in active_indices)
                changed = (cursor.advance(now, state["q_rad"], state["axis_stamp_monotonic_s"])
                           if bias_recovered and pending_cart is None else False)
                target, elapsed = cursor.target(), cursor.elapsed
                if changed:
                    recovery.node_changed(now)
                    self._settling_bias_centi = [0, 0]
                    self._settling_bias_retracted = [False, False]
                    cart_lead = 0.
                    cart_retracted = False
                    pending_cart = None
                    progress_at, progress_q = [now, now], list(state["q_rad"])
                recovery.select_directions(cursor.planned_directions(), now)
                if cursor.at_endpoint and endpoint_at is None:
                    endpoint_at = now
                    endpoint_due = now+cursor.minimum_interval_s
                    settling_deadline = endpoint_due+self.config["goal_settle_timeout_s"]
                    endpoint_deadline = settling_deadline+self.config["goal_hold_s"]
                if endpoint_due is not None and now >= endpoint_due and stage == "tracking":
                    stage, stage_started = "settling", now
                    trace.record_stage(stage, now, self._feedback, state, target, elapsed, progress_at, progress_q)
                tolerances = ([self.config["angle_tolerance_deg"]]*2 if cursor.at_endpoint
                              else cursor.tolerances_deg)
                errors = [abs(math.degrees(a-b)) for a, b in zip(state["q_rad"], target)]
                for i in active_indices:
                    movement = abs(math.degrees(state["q_rad"][i]-progress_q[i]))
                    toward = (target[i]-progress_q[i])*(state["q_rad"][i]-progress_q[i]) >= 0
                    in_band = errors[i] <= tolerances[i]+1e-9
                    if in_band or (toward and movement >= self.recovery_config["progress_deg"]-1e-9):
                        progress_at[i], progress_q[i] = now, state["q_rad"][i]
                        recovery.progress(i, in_band)
                if adaptive:
                    exhausted = recovery.update(now, state, target, tolerances, progress_at,
                        active_indices, self._last_targets, [b/100 for b in self._settling_bias_centi])
                    if exhausted is not None:
                        raise HardwareError("motion_recovery_exhausted", {
                            "motor": self.motors[exhausted]["name"], "recovery_exhausted": True})
                    if qualifier.armed and any(recovery.pending):
                        qualifier.reset_after_command()
                        holding_at = None
                        stage, stage_started = "settling", now
                else:
                    if not qualifier.armed:
                        for i in active_indices:
                            if (errors[i] > tolerances[i]+1e-9
                                    and now-progress_at[i] >= self.recovery_config["trigger_s"]
                                    and abs(math.degrees(state["speed_rad_s"][i])) <= self.config["still_speed_deg_s"]):
                                key = self._last_targets[i]
                                outcome = recovery.stall(i, now, 0. if key is None else key[2]/100.)
                                if outcome == "exhausted":
                                    raise HardwareError("motion_speed_recovery_exhausted", {
                                        "motor": self.motors[i]["name"], "recovery_exhausted": True,
                                        "tracking_error_deg": errors[i]})
                        if cursor.at_endpoint and now >= endpoint_due:
                            recovery.observe_goal(state, points[-1][1])
                            for i in active_indices:
                                key = self._last_targets[i]
                                recovery.request_settling(i, now, 0. if key is None else key[2]/100.)
                if adaptive and recovery.deadline is not None:
                    total_deadline = max(total_deadline, recovery.deadline)
                if recovery.expired(now):
                    raise HardwareError("motion_speed_recovery_timeout", {"recovery_exhausted": True})
                if endpoint_at is not None and recovery.deadline is not None:
                    settling_deadline = max(settling_deadline, recovery.deadline)
                    endpoint_deadline = settling_deadline+self.config["goal_hold_s"]
                stalled = [self._active_axes[i] and now-progress_at[i] >= self.config["settling_bias_wait_s"]
                           for i in range(2)]
                command = list(target)
                new_lead = cart_lead
                external_retract = False
                if meta is not None and not qualifier.armed:
                    in_band = all(errors[i] <= tolerances[i] + 1e-9 for i in active_indices)
                    approaching = ((cart_lead > 0 or pending_cart is not None)
                        and any(abs(math.degrees(state["speed_rad_s"][i])) > self.config["still_speed_deg_s"]
                                for i in active_indices)
                        and all(errors[i] <= tolerances[i] + .25 for i in active_indices))
                    if in_band or approaching or cart_retracted:
                        new_lead = 0.
                        external_retract = cart_retracted or cart_lead > 0 or any(self._settling_bias_centi)
                        pending_cart = None
                    elif pending_cart is not None:
                        new_lead, command, external_retract = pending_cart
                    elif not adaptive and any(stalled) and all(abs(math.degrees(state["speed_rad_s"][i])) <= self.config["still_speed_deg_s"]
                                             for i in active_indices):
                        if now-max(self._last_target_sent_at[i] for i in active_indices) >= self.config["settling_bias_wait_s"]:
                            new_lead = min(meta["max_lead_mm"], cart_lead+meta["step_mm"])
                    if pending_cart is None:
                        command = self._cartesian_target(meta, cart_arm, target, new_lead)
                        # Ordinary nominal waypoints may advance by proximity;
                        # only a real lead/retraction needs pair completion.
                        if abs(new_lead) > 1e-12 or external_retract:
                            pending_cart = (new_lead, command, external_retract)
                    if any(abs(command[i]-target[i]) > 1e-9 for i in passive_indices):
                        raise HardwareError("cartesian_bias_requires_inactive_axis_motion")
                    for motor, q in zip(self.motors, command):
                        if not motor["limits_deg"][0] <= math.degrees(q) <= motor["limits_deg"][1]:
                            raise HardwareError("cartesian_bias_out_of_limits")
                if now-last_command >= 1/self.config["command_rate_hz"] and not qualifier.armed:
                    speed_floors, speed_tiers, speed_caps = recovery.profiles(self.config["max_speed_deg_s"])
                    previous_writes = list(self._last_target_sent_at)
                    written = self._send_targets(command, planned_directions=cursor.planned_directions(),
                        planned_velocities=cursor.planned_velocities(), at_endpoint=cursor.at_endpoint,
                        axis_at_endpoint=self._active_axes, coalesce_steps=True,
                        tolerances_deg=tolerances, command_axes=self._active_axes,
                        allow_bias_axes=([False, False] if meta is not None else
                            [not adaptive and stalled[i] and not recovery.eligible(i) for i in range(2)]),
                        adaptive_mode=adaptive,
                        stuck_upshift_axes=[bool(adaptive and request is not None
                            and request["kind"] == "up" and request.get("diagnosis") == "stuck")
                            for request in recovery.pending],
                        stuck_angle_boost_axes=[bool(adaptive and request is not None
                            and request["kind"] == "angle_up" and request.get("diagnosis") == "stuck")
                            for request in recovery.pending],
                        recovery_bias_modes=(recovery.angle_modes() if adaptive else None),
                        recovery_validator=recovery_angle_safe,
                        axis_min_speed_deg_s=speed_floors, speed_tiers=speed_tiers,
                        command_speed_caps_deg_s=speed_caps,
                        nominal_q_rad=(target if meta is not None else None),
                        external_force=(meta is not None and (new_lead != cart_lead or external_retract)),
                        external_retract=external_retract)
                    sent_count += written
                    for i in active_indices:
                        key = self._last_targets[i]
                        if key is not None and self._last_target_sent_at[i] > previous_writes[i]:
                            committed = recovery.wrote(i, key[3] if len(key) > 3 else 0, key[2]/100.,
                                self._last_target_sent_at[i], **({"bias": self._settling_bias_centi[i]/100} if adaptive else {}))
                            if adaptive:
                                recovery.last_bias[i] = self._settling_bias_centi[i]/100.
                            if committed:
                                qualifier.reset_after_command()
                    if meta is not None:
                        # A partial pair is retried, not counted as an applied
                        # Cartesian lead. A6 ACKs do not echo the target register.
                        retracted_in_band = external_retract and all(
                            self._last_targets[i] is None
                            or abs(self._last_targets[i][0]/100-self._motor_deg(
                                self.motors[i], math.degrees(target[i])))
                            <= .85*tolerances[i]*self.motors[i]["ratio"]+1e-9
                            for i in active_indices)
                        if retracted_in_band or self._cartesian_pair_applied(command, cart_arm, meta["direction"]):
                            cart_lead = new_lead
                            cart_retracted = cart_retracted or external_retract
                            pending_cart = None
                    last_command = time.monotonic()
                record(time.monotonic())
                ready = (cursor.at_endpoint and now >= endpoint_due and commands_ready()
                         and all(state["axis_stamp_monotonic_s"][i] >= endpoint_at for i in active_indices))
                completed = qualifier.observe(state, points[-1][1], ready)
                if qualifier.armed and holding_at is None:
                    for i in active_indices:
                        recovery.cancel_pending(i, now, "already_stable_keep_existing_hold")
                    holding_at = time.monotonic()
                    stage, stage_started = "holding", holding_at
                    trace.record_stage(stage, holding_at, self._feedback, state, target, elapsed, progress_at, progress_q)
                if qualifier.failure:
                    if adaptive and qualifier.failure == "motion_goal_hold_unstable":
                        # Keep collecting feedback for classification, without
                        # accepting the unstable hold or extending its deadline.
                        qualifier.reset_after_command()
                        holding_at = None
                        stage, stage_started = "settling", time.monotonic()
                    else:
                        raise HardwareError(qualifier.failure)
                if completed:
                    record(time.monotonic())
                    return {"status": "completed", "phase": phase.get("phase"), "feedback": state,
                        "motion_commands_sent": sent_count, "elapsed_s": time.monotonic()-started,
                        "motion_diagnostics": _capture_motion_trace(trace, "completed", self._feedback,
                            state, target, elapsed, progress_at, progress_q)}
                now = time.monotonic()
                if endpoint_at is not None:
                    if not qualifier.armed and now >= settling_deadline:
                        raise HardwareError("motion_goal_settle_timeout")
                    if qualifier.armed and now >= endpoint_deadline:
                        raise HardwareError("motion_goal_hold_timeout")
                else:
                    if now >= total_deadline:
                        raise HardwareError("motion_timeout")
                    for i in active_indices:
                        if now-progress_at[i] >= self.config["no_progress_s"]:
                            if recovery.eligible(i) and (adaptive or recovery.speed_tier[i] != -1):
                                continue  # Bounded in-motion recovery owns this timeout.
                            raise HardwareError("motion_no_progress", {"motor": self.motors[i]["name"],
                                                                       "tracking_error_deg": errors[i]})
                # Measured deviations remain bounded even though the virtual clock is gone.
                if any(errors[i] > max(self.config["tracking_abort_deg"],
                                      abs(math.degrees(points[max(0,cursor.index-1)][1][i]-target[i]))
                                      +self.config["tracking_abort_deg"])
                       for i in active_indices):
                    raise HardwareError("motion_tracking_error")
                self._pump(min(.01, max(.001, 1/self.config["command_rate_hz"]-(time.monotonic()-last_command))))
        except BaseException as exc:
            trace.execution["speed_recovery"] = recovery.snapshot()
            diagnostics = _capture_motion_trace(trace, "failed", self._feedback, state, target,
                                                elapsed, progress_at, progress_q)
            result = self.halt()
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            if isinstance(exc, HardwareError):
                exc.halt_result = result
                if recovery.started_at is not None:
                    exc.details["speed_recovery_attempted"] = True
                exc.details.setdefault("execution_stage", trace.execution.get("stage"))
                exc.details["motion_commands_sent"] = sent_count
                exc.details["phase"] = phase.get("phase")
                exc.details["motion_diagnostics"] = diagnostics
                raise
            raise HardwareError("motion_failed", {"error": str(exc), "phase": phase.get("phase"),
                                "motion_diagnostics": diagnostics}, result) from exc
        finally:
            self._active_axes = [True, True]
            self._motion_trace = None
            self._moving = False
            self._move_lock.release()
