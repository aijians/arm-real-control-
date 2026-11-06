"""Pure framing and decoding for Lingkong RS485 protocol V2.35.

Parameter IDs and command IDs occupy separate namespaces: for example 0x99
below is a *parameter*, not a command. This module never opens a serial port.
"""

from control.protocol import (  # reuse the application's tested wire format
    Frame,
    angle_from_reply,
    extract_frame,
    make_frame,
    position_frame,
)


READ_PARAMETER = 0x40
WRITE_RAM = 0x42
WRITE_ROM = 0x44

PARAMS = {
    0x96: {"name": "angle_pid", "label": "角度环 PID", "kind": "pid", "unit": "raw"},
    0x97: {"name": "speed_pid", "label": "速度环 PID", "kind": "pid", "unit": "raw"},
    0x98: {"name": "current_pid", "label": "电流环 PID", "kind": "pid", "unit": "raw"},
    0x99: {"name": "max_torque_current", "label": "最大力矩电流", "kind": "int16", "unit": "raw"},
}

ERROR_FLAGS = {
    0: "low_voltage",
    1: "high_voltage",
    2: "driver_overtemperature",
    3: "motor_overtemperature",
    4: "overcurrent",
    5: "short_circuit",
    6: "stall",
    7: "input_timeout",
}


def _integer(value, minimum, maximum, label):
    # bool is an int subclass, but never a meaningful PID coefficient or ID.
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be an integer in {minimum}..{maximum}")
    return value


def _parameter(param):
    if type(param) is not int or param not in PARAMS:
        raise ValueError("Unsupported parameter ID")
    return param


def _payload(raw, size, label):
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raise ValueError(f"{label} must be bytes")
    result = bytes(raw)
    if len(result) != size:
        raise ValueError(f"{label} must contain exactly {size} bytes")
    return result


def read_parameter_frame(motor_id, param):
    """Build 0x40 with the required second, zero data byte (manual §24)."""
    _integer(motor_id, 1, 32, "Motor ID")
    return make_frame(READ_PARAMETER, motor_id, bytes((_parameter(param), 0)))


def write_parameter_frame(motor_id, param, raw6, persistent=False):
    """Write all six parameter bytes to RAM, or explicitly to ROM (§25–26).

    Callers editing 0x99 must preserve the four bytes the manual does not
    define. No zero padding or implicit construction of those bytes occurs.
    """
    _integer(motor_id, 1, 32, "Motor ID")
    _parameter(param)
    if type(persistent) is not bool:
        raise ValueError("persistent must be a bool")
    raw = _payload(raw6, 6, "Parameter data")
    return make_frame(WRITE_ROM if persistent else WRITE_RAM, motor_id,
                      bytes((param,)) + raw)


def encode_pid(kp, ki, kd):
    """Encode the three uint16 little-endian coefficients without truncation."""
    return b"".join(_integer(value, 0, 65535, label).to_bytes(2, "little")
                    for label, value in (("Kp", kp), ("Ki", ki), ("Kd", kd)))


def encode_max_torque_current(value, original_raw6):
    """Change only the signed int16 field; require original reserved bytes."""
    raw = _payload(original_raw6, 6, "Original parameter data")
    _integer(value, -32768, 32767, "Maximum torque current raw value")
    return value.to_bytes(2, "little", signed=True) + raw[2:]


def decode_parameter(param, raw6):
    """Decode documented fields only; never infer an ampere conversion."""
    _parameter(param)
    raw = _payload(raw6, 6, "Parameter data")
    if PARAMS[param]["kind"] == "pid":
        return {name: int.from_bytes(raw[offset:offset + 2], "little")
                for name, offset in (("kp", 0), ("ki", 2), ("kd", 4))}
    return {"value_raw": int.from_bytes(raw[:2], "little", signed=True),
            "reserved_hex": raw[2:].hex(" ")}


def decode_parameter_reply(frame, param):
    """Validate a 0x40 reply and return the six bytes for caching and editing.

    Motor ID matching belongs to the transaction layer, which knows the
    selected motor. The caller must already have verified the frame checksum
    through extract_frame (or control.protocol.parse_frame).
    """
    _parameter(param)
    if not isinstance(frame, Frame) or frame.command != READ_PARAMETER:
        raise ValueError("Unexpected parameter reply command")
    payload = _payload(frame.payload, 7, "Parameter reply")
    if payload[0] != param:
        raise ValueError("Parameter reply ID does not match the request")
    return payload[1:]


def validate_write_ack(frame, expected_cmd, param):
    """Raise ValueError for mismatched or failed acknowledgements.

    The manual's DATA[1] table cell describes success but its accompanying
    text has a DATA[0] typo. DATA[0] is the parameter ID; DATA[1] is status.
    """
    _parameter(param)
    if type(expected_cmd) is not int or expected_cmd not in (WRITE_RAM, WRITE_ROM):
        raise ValueError("Expected command must be WRITE_RAM or WRITE_ROM")
    if not isinstance(frame, Frame) or frame.command != expected_cmd:
        raise ValueError("Unexpected write acknowledgement command")
    payload = _payload(frame.payload, 2, "Write acknowledgement")
    if payload[0] != param:
        raise ValueError("Write acknowledgement parameter ID does not match")
    if payload[1] != 0:
        raise ValueError(f"Motor rejected parameter write (status {payload[1]})")


def decode_status1(payload7):
    """Decode 0x9A: signed temperature, 0.01 V, 0.01 A, status, errors."""
    raw = _payload(payload7, 7, "Status 1")
    return {
        "temperature_c": int.from_bytes(raw[:1], "little", signed=True),
        "voltage_v": int.from_bytes(raw[1:3], "little", signed=True) / 100.0,
        "bus_current_a": int.from_bytes(raw[3:5], "little", signed=True) / 100.0,
        "motor_state": raw[5],
        "error_state": raw[6],
        "error_flags": [name for bit, name in ERROR_FLAGS.items() if raw[6] & (1 << bit)],
    }


def decode_status2(payload7):
    """Decode 0x9C/A6, retaining model-dependent iq/power as a raw integer."""
    raw = _payload(payload7, 7, "Status 2")
    return {
        "temperature_c": int.from_bytes(raw[:1], "little", signed=True),
        "iq_or_power_raw": int.from_bytes(raw[1:3], "little", signed=True),
        "speed_deg_s": int.from_bytes(raw[3:5], "little", signed=True),
        "encoder_raw": int.from_bytes(raw[5:7], "little"),
    }


def decode_product_info(payload58):
    """Decode the full productInfo struct from manual section 27 (0x12)."""
    raw = _payload(payload58, 58, "Product information")
    def name(part):
        return part.split(b'\0', 1)[0].decode('utf-8', errors='replace').strip()
    result = {"driver_name": name(raw[:20]), "motor_name": name(raw[20:40]),
              "motor_serial": name(raw[40:52])}
    for label, offset in (("hardware_version", 52), ("motor_version", 54), ("firmware_version", 56)):
        value = int.from_bytes(raw[offset:offset+2], 'little')
        result[label+"_raw"] = value
        result[label] = value/10.
    return result
