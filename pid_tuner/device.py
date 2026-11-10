"""One transaction at a time on the upper-arm port, plus a hardware-free demo."""
import math
import os
import select
import threading
import time

from control.motors import _Port
from control.protocol import Frame, angle_from_reply, extract_frame, make_frame, position_frame
from .protocol import (decode_parameter, decode_parameter_reply, decode_product_info, decode_status1, decode_status2,
                       encode_pid, read_parameter_frame, validate_write_ack,
                       write_parameter_frame, WRITE_RAM, WRITE_ROM)
from .vendor import extract_reply, frame_bytes, validate_settings, parameter_from_settings


class ReplyTimeout(TimeoutError):
    """Keep byte-level evidence when no acceptable response was obtained."""
    def __init__(self, motor_id, command, param, details):
        self.details = details
        suffix = f' 参数0x{param:02X}' if param is not None else ''
        received = details['received_bytes']
        evidence = '未接收到任何字节' if not received else f'收到{received}字节，但没有匹配的有效应答'
        super().__init__(f'电机 ID{motor_id} 命令0x{command:02X}{suffix}应答超时（{evidence}）')


def diagnostic_summary(steps):
    basic = [s for s in steps if s['command'] in ('0x94', '0x9C', '0x9A')]
    pid = [s for s in steps if s['command'] in ('0x40', '0x14')]
    basic_ok = [s['command'] for s in basic if s['ok']]
    summary = []
    if basic_ok:
        summary.append('基本状态查询已收到应答：'+', '.join(basic_ok)+'；相应命令的串口和ID通信正常。')
    else:
        summary.append('基本状态查询也未取得有效应答；请检查设备路径、ID、波特率、供电、接线及RS485收发方向。')
    if pid and all(s['ok'] for s in pid):
        summary.append('角度环和速度环PID均已成功读取；可重新连接建立备份。')
    elif basic_ok:
        summary.append('状态查询有应答，但PID参数读取未全部成功；需核对驱动型号、固件与V2.35参数协议是否匹配，不能仅凭超时断言固件不支持。')
    else:
        summary.append('PID尚未读取或备份，不能进入参数写入。诊断没有发送使能、停止、运动或参数写入命令。')
    return summary


class SerialDevice:
    def __init__(self, connection, logger, timeout=0.5):
        self.connection, self.log, self.timeout = connection, logger, timeout
        self.motor_id = connection['motor_id']
        self.port = _Port(connection['device'], connection['baudrate'])
        self.lock = threading.RLock()
        self.interrupted = threading.Event()
        self.last_transaction = None
        self.settings_raw = None
        self.vendor_mode = connection.get('parameter_protocol') == 'vendor_v235'

    def open(self):
        self.port.open()  # Opening a port never enables or moves the motor.

    def close(self):
        self.port.close()

    def interrupt(self):
        self.interrupted.set()

    def clear_interrupt(self):
        self.interrupted.clear()

    def _transaction(self, packet, command, param=None):
        with self.lock:
            self.last_transaction = None
            # Discard replies already queued before this request. Replies must
            # match motor ID, command and (for parameters) the parameter ID.
            self.port.buffer.clear()
            drain_until = time.monotonic()+0.02
            while time.monotonic() < drain_until:
                try:
                    old = os.read(self.port.fd, 4096)
                    if not old:
                        break
                    self.log('wire', 'RX_BEFORE_TX ' + old[:512].hex(' ').upper())
                except BlockingIOError:
                    break
            if self.interrupted.is_set() and command != 0x81:
                raise InterruptedError('操作被停止按钮中断')
            self.port.send(packet, cancel_event=None if command == 0x81 else self.interrupted)
            self.log('wire', 'TX ' + packet.hex(' ').upper())
            evidence = {'tx_hex': packet.hex(' ').upper(), 'received_bytes': 0,
                        'raw_rx_hex': '', 'unmatched_frames': [], 'incomplete_hex': ''}
            raw_received = bytearray()
            expected_length = {0x14: 104, 0x15: 0, 0x40: 7, 0x42: 2, 0x44: 2, 0x94: 4, 0x9C: 7,
                               0x9A: 7, 0x12: 58, 0x81: 0, 0x88: 0, 0xA6: 7}.get(command)
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                if self.interrupted.is_set() and command != 0x81:
                    raise InterruptedError('操作被停止按钮中断')
                if not select.select([self.port.fd], [], [], min(0.02, max(0, deadline-time.monotonic())))[0]:
                    continue
                try:
                    data = os.read(self.port.fd, 4096)
                except BlockingIOError:
                    continue
                if not data:
                    raise OSError('串口连接已断开')
                evidence['received_bytes'] += len(data)
                raw_received.extend(data)
                del raw_received[:-512]
                for offset in range(0, len(data), 512):
                    self.log('wire', 'RX_BYTES ' + data[offset:offset+512].hex(' ').upper())
                self.port.buffer.extend(data)
                while True:
                    frame = extract_reply(self.port.buffer)
                    if frame is None:
                        break
                    self.log('wire', 'RX ' + frame_bytes(frame.command, frame.motor_id, frame.payload).hex(' ').upper())
                    mismatch = None
                    if frame.motor_id != self.motor_id:
                        mismatch = 'motor_id'
                    elif frame.command != command:
                        mismatch = 'command'
                    elif param is not None and (not frame.payload or frame.payload[0] != param):
                        mismatch = 'parameter_id'
                    elif expected_length is not None and len(frame.payload) != expected_length:
                        mismatch = 'payload_length_or_echo'
                    if mismatch:
                        evidence['unmatched_frames'].append({'reason': mismatch, 'motor_id': frame.motor_id,
                            'command': f'0x{frame.command:02X}', 'payload_length': len(frame.payload),
                            'payload_hex': frame.payload.hex(' ').upper()})
                        evidence['unmatched_frames'] = evidence['unmatched_frames'][-8:]
                        continue
                    evidence.update(raw_rx_hex=raw_received.hex(' ').upper(), raw_rx_kept_bytes=len(raw_received),
                                    raw_rx_truncated=evidence['received_bytes'] > len(raw_received),
                                    rx_hex=frame_bytes(frame.command, frame.motor_id, frame.payload).hex(' ').upper(),
                                    incomplete_hex=bytes(self.port.buffer[-512:]).hex(' ').upper())
                    self.last_transaction = evidence
                    return frame
            evidence['raw_rx_hex'] = raw_received.hex(' ').upper()
            evidence['raw_rx_kept_bytes'] = len(raw_received)
            evidence['raw_rx_truncated'] = evidence['received_bytes'] > len(raw_received)
            evidence['incomplete_hex'] = bytes(self.port.buffer[-512:]).hex(' ').upper()
            self.last_transaction = evidence
            raise ReplyTimeout(self.motor_id, command, param, evidence)

    def read_parameter(self, param):
        if self.vendor_mode:
            return parameter_from_settings(self.read_settings(), param)
        frame = self._transaction(read_parameter_frame(self.motor_id, param), 0x40, param)
        return decode_parameter_reply(frame, param)

    def read_settings(self):
        frame = self._transaction(make_frame(0x14, self.motor_id), 0x14)
        self.settings_raw = validate_settings(frame.payload)
        return self.settings_raw

    def diagnose(self):
        """Only documented read operations, independently recorded on failure."""
        steps = []
        queries = [('单圈角度', 0x94), ('速度和电流原始值', 0x9C),
                   ('电压和故障状态', 0x9A), ('驱动型号和固件', 0x12),
                   ('角度环PID', 0x40, 0x96), ('速度环PID', 0x40, 0x97)]
        if self.vendor_mode:
            queries = queries[:4] + [('厂家整组设置和PID（104字节）', 0x14)]
        for query in queries:
            name, command = query[:2]
            param = query[2] if len(query) == 3 else None
            step = {'name': name, 'command': f'0x{command:02X}', 'ok': False}
            if param is not None:
                step['param_id'] = f'0x{param:02X}'
            try:
                if command == 0x14:
                    raw = self.read_settings()
                    step['decoded'] = {name: decode_parameter(p, parameter_from_settings(raw, p))
                                       for name, p in (('angle', 0x96), ('speed', 0x97), ('current', 0x98), ('limit_raw', 0x99))}
                elif param is not None:
                    step['decoded'] = decode_parameter(param, self.read_parameter(param))
                else:
                    frame = self._transaction(make_frame(command, self.motor_id), command)
                    if command == 0x94:
                        step['decoded'] = {'motor_angle_deg': angle_from_reply(frame)}
                    else:
                        decoder = {0x9C: decode_status2, 0x9A: decode_status1, 0x12: decode_product_info}[command]
                        step['decoded'] = decoder(frame.payload)
                step['ok'] = True
                step['details'] = dict(self.last_transaction or {})
            except InterruptedError:
                raise
            except Exception as exc:
                step.update(error=str(exc), details=getattr(exc, 'details', None) or self.last_transaction or {})
            steps.append(step)
        return {'steps': steps, 'summary': diagnostic_summary(steps)}

    def write_parameter(self, param, raw6, persistent=False):
        if self.vendor_mode and persistent:
            raw = validate_settings(self.settings_raw or b'')
            if param not in (0x96, 0x97) or parameter_from_settings(raw, param) != bytes(raw6):
                raise ValueError('整组保存必须与最新读回的已验证PID一致')
            self._transaction(frame_bytes(0x15, self.motor_id, raw), 0x15)
            return
        command = WRITE_ROM if persistent else WRITE_RAM
        frame = self._transaction(write_parameter_frame(self.motor_id, param, raw6, persistent), command, param)
        validate_write_ack(frame, command, param)

    def save_settings(self, payload):
        payload = validate_settings(payload)
        self._transaction(frame_bytes(0x15, self.motor_id, payload), 0x15)
        if self.read_settings() != payload:
            raise ValueError('整组保存已应答，但0x14读回与保存目标不一致；持久性未确认')

    def telemetry(self):
        angle = angle_from_reply(self._transaction(make_frame(0x94, self.motor_id), 0x94))
        state2 = decode_status2(self._transaction(make_frame(0x9C, self.motor_id), 0x9C).payload)
        state1 = decode_status1(self._transaction(make_frame(0x9A, self.motor_id), 0x9A).payload)
        sign, ratio, offset = (self.connection[k] for k in ('sign', 'ratio', 'offset_deg'))
        return {**state2, **state1, 'motor_angle_deg': angle,
                'angle_deg': (angle-offset)/(sign*ratio),
                'speed_deg_s': state2['speed_deg_s']/(sign*ratio)}

    def _simple(self, command):
        frame = self._transaction(make_frame(command, self.motor_id), command)
        if frame.payload:
            raise ValueError(f'命令0x{command:02X}应答长度不符')

    def stop(self):
        self._simple(0x81)

    def enable(self):
        self._simple(0x88)

    def move(self, target_deg, speed_deg_s, current_motor_deg):
        c = self.connection
        target = round((c['offset_deg']+c['sign']*c['ratio']*target_deg)*100)
        if not 0 <= target <= 35999:
            raise ValueError('电机目标超出单圈协议范围')
        speed = max(1, round(speed_deg_s*c['ratio']*100))
        if speed > 36000:
            raise ValueError('电机轴速度超出本工具支持范围')
        packet = position_frame(self.motor_id, 0 if target/100 > current_motor_deg else 1, target, speed)
        frame = self._transaction(packet, 0xA6)
        decode_status2(frame.payload)


class DemoDevice:
    """Synthetic responses, not a motor model or a prediction of tuned PID."""
    def __init__(self, connection, logger):
        self.connection, self.log = connection, logger
        self.params = {0x96: encode_pid(50, 0, 0), 0x97: encode_pid(100, 30, 0),
                       0x98: encode_pid(40, 20, 0), 0x99: (850).to_bytes(2, 'little', signed=True)+bytes(4)}
        self.rom = dict(self.params)
        initial = 45. if connection.get('axis') == 'lower' else 169.11
        self.angle, self.target, self.speed = initial, initial, 0.
        self.enabled = True
        self.last = time.monotonic()

    def open(self):
        self.last = time.monotonic()

    def close(self):
        pass

    def interrupt(self):
        pass

    def clear_interrupt(self):
        pass

    def _wire(self, packet, payload):
        command = packet[1]
        self.log('wire', 'TX ' + packet.hex(' ').upper())
        self.log('wire', 'RX ' + make_frame(command, self.connection['motor_id'], payload).hex(' ').upper())

    def read_parameter(self, param):
        self._wire(read_parameter_frame(self.connection['motor_id'], param), bytes((param,))+self.params[param])
        return self.params[param]

    def diagnose(self):
        state = self.telemetry()
        steps = [
            {'name': '单圈角度', 'command': '0x94', 'ok': True,
             'decoded': {'motor_angle_deg': state['motor_angle_deg']}},
            {'name': '速度和电流原始值', 'command': '0x9C', 'ok': True,
             'decoded': {k: state[k] for k in ('speed_deg_s', 'iq_or_power_raw')}},
            {'name': '电压和故障状态', 'command': '0x9A', 'ok': True,
             'decoded': {k: state[k] for k in ('voltage_v', 'bus_current_a', 'error_flags')}},
            {'name': '驱动型号和固件', 'command': '0x12', 'ok': True,
             'decoded': {'driver_name': 'DEMO', 'motor_name': '演示驱动，无硬件访问'}},
        ]
        for param, name in ((0x96, '角度环PID'), (0x97, '速度环PID')):
            steps.append({'name': name, 'command': '0x40', 'param_id': f'0x{param:02X}',
                          'ok': True, 'decoded': decode_parameter(param, self.read_parameter(param))})
        return {'steps': steps, 'summary': diagnostic_summary(steps)}

    def write_parameter(self, param, raw6, persistent=False):
        self._wire(write_parameter_frame(self.connection['motor_id'], param, raw6, persistent), bytes((param, 0)))
        if persistent:
            self.rom[param] = bytes(raw6)
        else:
            self.params[param] = bytes(raw6)

    def telemetry(self):
        now = time.monotonic()
        dt, self.last = min(0.3, now-self.last), now
        delta = self.target-self.angle
        velocity = max(-self.speed, min(self.speed, delta*5)) if self.enabled else 0.
        self.angle += math.copysign(min(abs(delta), abs(velocity)*dt), delta) if delta else 0.
        c = self.connection
        motor_angle = c['offset_deg']+c['sign']*c['ratio']*self.angle
        return {'angle_deg': self.angle, 'motor_angle_deg': motor_angle,
                'speed_deg_s': velocity, 'bus_current_a': round(0.25+abs(velocity)*0.08, 3),
                'voltage_v': 24., 'temperature_c': 32,
                'iq_or_power_raw': round(velocity*8), 'encoder_raw': round(motor_angle/360*16384),
                'motor_state': 0 if self.enabled else 0x10, 'error_state': 0, 'error_flags': []}

    def stop(self):
        self.target, self.speed = self.angle, 0.
        self._wire(make_frame(0x81, self.connection['motor_id']), b'')

    def enable(self):
        self.enabled = True
        self._wire(make_frame(0x88, self.connection['motor_id']), b'')

    def move(self, target_deg, speed_deg_s, current_motor_deg):
        c = self.connection
        target_motor = c['offset_deg']+c['sign']*c['ratio']*target_deg
        packet = position_frame(c['motor_id'], 0 if target_motor > current_motor_deg else 1,
                                round(target_motor*100), round(speed_deg_s*c['ratio']*100))
        self._wire(packet, bytes(7))
        self.target, self.speed = target_deg, speed_deg_s
