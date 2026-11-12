"""Apply the configured PID RAM targets once before each hardware task.

Reuse the tuner's verified 0x42 transaction and acknowledgement rules. No ROM
save, enable, zeroing or position command is part of this initialization.
"""
from pid_tuner.protocol import encode_pid


def task_start_pid_settings(hardware):
    raw = hardware.get("task_start_pid", {"enabled": False})
    if not isinstance(raw, dict) or type(raw.get("enabled")) is not bool:
        raise ValueError("invalid hardware.task_start_pid.enabled")
    if set(raw) - {"enabled", "upper", "lower"}:
        raise ValueError("unknown hardware.task_start_pid setting")
    if not raw["enabled"]:
        return {"enabled": False}
    result = {"enabled": True}
    for axis in ("upper", "lower"):
        loops = raw.get(axis)
        if not isinstance(loops, dict) or set(loops) != {"angle", "speed"}:
            raise ValueError("task_start_pid requires angle and speed for " + axis)
        result[axis] = {}
        for loop in ("angle", "speed"):
            values = loops[loop]
            if not isinstance(values, dict) or set(values) != {"kp", "ki", "kd"}:
                raise ValueError("task_start_pid requires kp/ki/kd for " + axis + "." + loop)
            encode_pid(**values)  # Reject bools, floats and uint16 overflow before opening UART.
            result[axis][loop] = dict(values)
    return result


def initialize_task_pid(motors, hardware, report):
    """Borrow the task's open, exclusive ports; leave ownership with MotorPair.

    ``report`` is updated before every write so a partial/uncertain write is
    retained if initialization fails. Never retry writes or roll back guesses
    about the previous runtime RAM values.
    """
    from .motors import HardwareError
    from pid_tuner.device import SerialDevice

    cfg = task_start_pid_settings(hardware)
    report.update(status="disabled" if not cfg["enabled"] else "pending",
                  persistent=False, runtime_readback_verified=False, writes=[])
    if not cfg["enabled"]:
        return report
    if not motors._opened or motors._moving:
        raise HardwareError("task_pid_requires_idle_open_ports")
    if len(motors.motors) != 2 or {m["name"] for m in motors.motors} != {"upper", "lower"}:
        raise HardwareError("task_pid_requires_upper_and_lower")
    if not motors._move_lock.acquire(blocking=False):
        raise HardwareError("motion_already_running")
    device = None
    entry = None
    try:
        with motors._io_lock:
            report["stop_before_write"] = motors.halt()
            if not report["stop_before_write"].get("stopped_confirmed"):
                raise HardwareError("task_pid_stop_not_confirmed")
            report["status"] = "writing"
            for motor, port in zip(motors.motors, motors.ports):
                axis = motor["name"]
                device = SerialDevice({"device": motor["device"], "motor_id": motor["id"],
                    "baudrate": int(motors.config["baudrate"]), "parameter_protocol": "vendor_v235"},
                    lambda *args: None, timeout=.5)
                device.port = port  # Do not reopen or close the task-owned descriptor.
                for loop, param in (("angle", 0x96), ("speed", 0x97)):
                    values = cfg[axis][loop]
                    entry = {"axis": axis, "motor_id": motor["id"], "loop": loop,
                             "parameter_id": f"0x{param:02x}", "requested": dict(values),
                             "command": "0x42", "ack_ok": False}
                    report["writes"].append(entry)
                    device.write_parameter(param, encode_pid(**values), persistent=False)
                    entry.update(ack_ok=True, transaction=device.last_transaction)
                    print("[pid] %s %s RAM acknowledged: %s/%s/%s" %
                          (axis, loop, values["kp"], values["ki"], values["kd"]), flush=True)
            report["status"] = "acknowledged"
        return report
    except BaseException as exc:
        report.update(status="failed", reason=str(exc), error_type=type(exc).__name__)
        if entry is not None and not entry["ack_ok"]:
            entry.update(error=str(exc), transaction=None if device is None else device.last_transaction)
        report["stop_after_failure"] = motors.halt()
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise HardwareError("task_pid_initialization_failed", {"pid_initialization": report},
                            report["stop_after_failure"]) from exc
    finally:
        motors._move_lock.release()
