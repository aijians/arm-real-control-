"""Virtual RS485 feedback/drive fixture; never opens a device."""
from contextlib import ExitStack
import math,unittest
from unittest import mock
from control import motors as m

def config(**extra):
    result = {
        "motors": [
            {"name": "upper", "device": "/tmp/never-opened-integration-upper",
             "id": 1, "sign": 1, "ratio": 1., "offset_deg": 0.,
             "limits_deg": [0., 180.48]},
            {"name": "lower", "device": "/tmp/never-opened-integration-lower",
             "id": 2, "sign": 1, "ratio": 1., "offset_deg": 0.,
             "limits_deg": [0., 93.82]},
        ],
        "angle_tolerance_deg": .3,
        "still_speed_deg_s": 1.,
        "stable_s": .1,
        "goal_hold_s": .4,
        "goal_settle_timeout_s": 3.,
        "feedback_timeout_s": .5,
        "feedback_rate_hz": 20.,
        "command_rate_hz": 50.,
        "no_progress_s": 2.,
        "move_timeout_margin_s": 8.,
    }
    result.update(extra)
    return result


def phase(points, times=None, name="free", **extra):
    if times is None:
        times = [float(i) for i in range(len(points))]
    return dict(phase=name, points=[
        {"time_s": t, "q_rad": list(map(math.radians, q))}
        for t, q in zip(times, points)], **extra)


class VirtualDrive:
    """Finite-speed independent motor with optional stiction/loss scenarios."""
    def __init__(self, model_q=(159., 20.), settings=None, *, ack=True,
                 immobile=False, start_friction_deg=0., query_loss=None,
                 disturbance=None, lost_targets=0, fail_write=None,
                 breakaway_once=False, load_model=None):
        self.pair = m.MotorPair(settings or config())
        self.now = 1000.
        self.origin = self.now
        self.q = [self.pair._motor_deg(motor, value)
                  for motor, value in zip(self.pair.motors, model_q)]
        self.target = list(self.q)
        self.velocity = [0., 0.]
        self.command_speed = [18., 18.]
        self.running = [False, False]
        self.ack = ack
        self.immobile = immobile
        self.start_friction_deg = start_friction_deg
        self.breakaway_once = breakaway_once
        self.has_moved = [False, False]
        self.query_loss = query_loss
        self.disturbance = disturbance
        self.lost_targets = lost_targets
        self.fail_write = fail_write
        # Non-uniform load: load_model(axis, motor_deg) -> lag (deg) the axis
        # needs before it moves at that angle (static friction + gravity). The
        # axis holds while |target - q| <= lag and otherwise tracks target - lag.
        self.load_model = load_model
        self.frames = []
        self.pending_acks = []
        self.halts = []
        self.unsafe_directions = []
        self.pumps = 0
        for axis, port in enumerate(self.pair.ports):
            port.send = lambda wire, *a, axis=axis, **k: self.send(axis, wire, **k)

    @property
    def elapsed(self):
        return self.now - self.origin

    def advance(self, seconds):
        seconds = max(0., float(seconds))
        for i in range(2):
            if self.load_model is not None and not self.immobile:
                lag = max(0., float(self.load_model(i, self.q[i])))
                delta = self.target[i] - self.q[i]
                if abs(delta) <= lag + 1e-12:
                    self.velocity[i] = 0.
                    continue
                goal = self.target[i] - math.copysign(lag, delta)
                step = math.copysign(min(abs(goal - self.q[i]), self.command_speed[i] * seconds), delta)
                self.q[i] += step
                self.has_moved[i] |= abs(step) > 1e-9
                self.velocity[i] = step / seconds if seconds else 0.
                continue
            delta = self.target[i] - self.q[i]
            friction = (self.start_friction_deg[i]
                        if isinstance(self.start_friction_deg, (list, tuple))
                        else self.start_friction_deg)
            if self.breakaway_once and self.has_moved[i]:
                friction = .05
            if self.immobile or (not self.running[i]
                                and abs(delta) <= friction):
                self.velocity[i] = 0.
                continue
            self.running[i] = abs(delta) > .000001
            step = math.copysign(min(abs(delta), self.command_speed[i] * seconds), delta)
            self.q[i] += step
            self.has_moved[i] |= abs(step) > 1e-9
            self.velocity[i] = step / seconds if seconds else 0.
            if abs(self.target[i] - self.q[i]) < 1e-8:
                self.running[i] = False
                self.velocity[i] = 0.
        self.now = round(self.now + seconds, 9)
        if self.disturbance:
            self.disturbance(self)

    def query_allowed(self, axis, command):
        return self.query_loss is None or not self.query_loss(axis, command, self.elapsed)

    def refresh(self, axis, command):
        if not self.query_allowed(axis, command):
            return
        reply = self.pair._feedback[axis]
        if command == 0x94:
            reply.update(motor_deg=round(self.q[axis], 2), angle_stamp=self.now,
                         angle_query_stamp=self.now)
        else:
            reply.update(speed_deg_s=float(round(self.velocity[axis])),
                         speed_stamp=self.now, current_raw=0)

    def send(self, axis, wire, cancel_event=None, **kwargs):
        if cancel_event is not None and cancel_event.is_set():
            raise m.HardwareError("motion_halted")
        port = self.pair.ports[axis]
        self.advance(max(.0001, getattr(port, "_next_send_at", 0.) - self.now))
        command = wire[1]
        if self.fail_write and self.fail_write(axis, wire, self):
            raise OSError("virtual write failure")
        self.frames.append((self.elapsed, axis, command, bytes(wire)))
        port._next_send_at = self.now + .005
        if command in (0x94, 0x9C):
            self.refresh(axis, command)
        elif command == 0xA6:
            target = int.from_bytes(wire[6:9], "little") / 100.
            direction = wire[5]
            if ((direction == 0 and target < self.q[axis] - .02)
                    or (direction == 1 and target > self.q[axis] + .02)):
                self.unsafe_directions.append((self.elapsed, axis, self.q[axis],
                                               target, direction))
            if self.lost_targets:
                self.lost_targets -= 1
                return
            self.target[axis] = target
            self.command_speed[axis] = int.from_bytes(wire[9:13], "little") / 100.
            if self.ack:
                self.pending_acks.append(axis)
        elif command == 0x81:
            self.target[axis] = self.q[axis]
            self.velocity[axis] = 0.
            self.running[axis] = False

    def pump(self, timeout_s=0.):
        self.pumps += 1
        if self.pumps > 40000:
            raise RuntimeError("virtual loop budget exceeded")
        self.advance(max(.001, timeout_s))
        for axis in self.pending_acks:
            self.pair._last_target_acked[axis] = True
            self.refresh(axis, 0xA6)
        self.pending_acks.clear()

    def halt(self):
        self.halts.append(self.elapsed)
        return {"stop_sent": [True, True], "stopped_confirmed": True}

    def run(self, planned, health_check=None, real_halt=False):
        self.pair._opened = True
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(m.time, "monotonic", lambda: self.now))
            stack.enter_context(mock.patch.object(m.time, "sleep", self.advance))
            stack.enter_context(mock.patch.object(self.pair, "_pump", self.pump))
            if not real_halt:
                stack.enter_context(mock.patch.object(self.pair, "halt", self.halt))
            return self.pair.move(planned, health_check=health_check)

    def halt_only(self):
        self.pair._opened = True
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(m.time, "monotonic", lambda: self.now))
            stack.enter_context(mock.patch.object(m.time, "sleep", self.advance))
            stack.enter_context(mock.patch.object(self.pair, "_pump", self.pump))
            return self.pair.halt()

    def a6(self, axis=None):
        return [item for item in self.frames
                if item[2] == 0xA6 and (axis is None or item[1] == axis)]


