"""The board's existing 0x3E motor protocol; no serial side effects."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Frame:
    command: int
    motor_id: int
    payload: bytes


def make_frame(command, motor_id, payload=b""):
    payload = bytes(payload)
    if not (isinstance(command, int) and 0 <= command <= 255
            and isinstance(motor_id, int) and 1 <= motor_id <= 32
            and len(payload) <= 100):
        raise ValueError("Invalid motor frame arguments")
    header = bytes((0x3E, command, motor_id, len(payload)))
    return (header + bytes((sum(header) & 255,)) + payload
            + (bytes((sum(payload) & 255,)) if payload else b""))


def position_frame(motor_id, direction, angle_centideg, speed_centideg_s):
    if direction not in (0, 1):
        raise ValueError("Direction must be 0 (increase) or 1 (decrease)")
    if not (isinstance(angle_centideg, int) and 0 <= angle_centideg <= 35999
            and isinstance(speed_centideg_s, int) and 0 < speed_centideg_s <= 36000):
        raise ValueError("A6 position or speed out of range")
    return make_frame(0xA6, motor_id, bytes((direction,))
                      + angle_centideg.to_bytes(3, "little")
                      + speed_centideg_s.to_bytes(4, "little"))


def parse_frame(data):
    if len(data) < 5 or data[0] != 0x3E or data[4] != sum(data[:4]) & 255:
        raise ValueError("Invalid frame header")
    size = data[3]
    if size > 100 or len(data) != 5 + size + bool(size):
        raise ValueError("Invalid frame size")
    if size and data[-1] != sum(data[5:-1]) & 255:
        raise ValueError("Invalid payload checksum")
    return Frame(data[1], data[2], bytes(data[5:-1]) if size else b"")


def extract_frame(buffer):
    """Remove noise/bad frames; retain an incomplete, valid-header frame."""
    while buffer:
        start = buffer.find(0x3E)
        if start < 0:
            buffer.clear()
            return None
        del buffer[:start]
        if len(buffer) < 5:
            return None
        # Check the header BEFORE trusting its length. Otherwise corrupt noise
        # with a plausible large length can indefinitely hide a complete reply.
        if buffer[4] != sum(buffer[:4]) & 255 or buffer[3] > 100:
            del buffer[0]
            continue
        size = 5 + buffer[3] + bool(buffer[3])
        if len(buffer) < size:
            return None
        try:
            frame = parse_frame(buffer[:size])
        except ValueError:
            del buffer[0]
            continue
        del buffer[:size]
        return frame
    return None


def angle_from_reply(frame):
    if frame.command != 0x94 or len(frame.payload) != 4:
        raise ValueError("Invalid single-turn angle reply")
    angle = int.from_bytes(frame.payload, "little") / 100.0
    if not 0 <= angle < 360:
        raise ValueError("Single-turn angle out of range")
    return angle


def state_from_reply(frame):
    if frame.command not in (0x9C, 0xA6) or len(frame.payload) != 7:
        raise ValueError("Invalid state-2 reply")
    p = frame.payload
    return {"temperature_c": int.from_bytes(p[:1], "little", signed=True),
            "current_raw": int.from_bytes(p[1:3], "little", signed=True),
            "speed_deg_s": int.from_bytes(p[3:5], "little", signed=True),
            "encoder_raw": int.from_bytes(p[5:7], "little")}
