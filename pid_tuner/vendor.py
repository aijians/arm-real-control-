"""LK Motor Tool V2.35 settings layout, verified on both DS40R7/MS4005 drives.

Keep this extended framing local to the tuner. The motion controller retains
its existing 100-byte limit. No hardware is accessed by this module.
"""
from control.protocol import Frame

SETTINGS_SIZE = 104
PID_SLICES = {0x96: slice(50, 56), 0x97: slice(56, 62), 0x98: slice(62, 68)}


def frame_bytes(command, motor_id, payload=b''):
    payload = bytes(payload)
    if not 1 <= motor_id <= 32 or not 0 <= command <= 255 or len(payload) > 104:
        raise ValueError('Invalid tuner frame')
    if len(payload) > 100 and (command not in (0x14, 0x15) or len(payload) != 104):
        raise ValueError('Only vendor settings frames may exceed 100 bytes')
    header = bytes((0x3E, command, motor_id, len(payload)))
    return header + bytes((sum(header) & 255,)) + payload + (bytes((sum(payload) & 255,)) if payload else b'')


def extract_reply(buffer):
    while buffer:
        start = buffer.find(0x3E)
        if start < 0:
            buffer.clear()
            return None
        del buffer[:start]
        if len(buffer) < 5:
            return None
        command, size = buffer[1], buffer[3]
        if (buffer[4] != sum(buffer[:4]) & 255 or not 1 <= buffer[2] <= 32
                or (size > 100 and not (command in (0x14, 0x15) and size == 104))):
            del buffer[0]
            continue
        total = 5 + size + bool(size)
        if len(buffer) < total:
            return None
        if size and buffer[total-1] != sum(buffer[5:total-1]) & 255:
            del buffer[0]
            continue
        frame = Frame(command, buffer[2], bytes(buffer[5:total-1]) if size else b'')
        del buffer[:total]
        return frame
    return None


def validate_settings(payload):
    raw = bytes(payload)
    if len(raw) != SETTINGS_SIZE or raw[100:104] != bytes.fromhex('78 56 AA 55'):
        raise ValueError('整组设置长度或有效标志不符，禁止使用未确认的布局')
    return raw


def parameter_from_settings(payload, param):
    raw = validate_settings(payload)
    if param in PID_SLICES:
        return raw[PID_SLICES[param]]
    if param == 0x99:
        return raw[68:70] + bytes(4)  # read-only; not a six-byte writable field
    raise ValueError('Unsupported settings parameter')


def check_settings_save(original, latest, acknowledged):
    """Allow only acknowledged PID targets to differ from the trusted snapshot."""
    original, latest = validate_settings(original), validate_settings(latest)
    if original[:50] + original[62:] != latest[:50] + latest[62:]:
        raise ValueError('非角度/速度PID设置已变化，禁止整组保存；请重新检查并建立备份')
    for name, param in (('angle', 0x96), ('speed', 0x97)):
        actual = parameter_from_settings(latest, param)
        if actual != parameter_from_settings(original, param) and acknowledged.get(name) != actual:
            raise ValueError('整组PID包含非本工具已应答目标的修改，禁止整组保存')
    return latest
