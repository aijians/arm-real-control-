"""Stateful tuning session; all motor operations are serialized and audited."""
from collections import deque
import copy
import csv
import io
import math
from pathlib import Path
import threading
import time
import uuid

from control.config import save_json
from control.runtime import instance_lock
from .device import DemoDevice, SerialDevice
from .protocol import decode_parameter, encode_pid
from .vendor import check_settings_save, parameter_from_settings, PID_SLICES

LOOPS = {'angle': 0x96, 'speed': 0x97}
PARAM_NAMES = {0x96: 'angle', 0x97: 'speed', 0x98: 'current', 0x99: 'torque_limit'}
TARGET_TOLERANCE_DEG = 0.2
SETTLE_SPEED_DEG_S = 1.
ARRIVAL_STABLE_S = 0.3
VERIFY_HOLD_S = 3.
VERIFY_TIMEOUT_S = 30.
HOLD_GUARD_DELAY_S = 0.5
HOLD_GUARD_MIN_SAMPLES = 3
HOLD_HARD_ERROR_DEG = 1.
HOLD_HARD_SPEED_DEG_S = 10.


def idle_motion():
    return {'active': False, 'target_deg': None, 'speed_deg_s': None,
            'result': 'idle', 'phase': 'idle', 'stop_required': False,
            'trial_id': None, 'trial_outcome': None, 'stop_reason': None,
            'start_deg': None, 'last_trial_error_deg': None,
            'last_trial_speed_deg_s': None, 'hold_peak_speed_deg_s': 0.,
            'hold_peak_error_deg': None, 'hold_started_at': None,
            'tolerance_deg': TARGET_TOLERANCE_DEG, 'verify_hold_s': VERIFY_HOLD_S,
            'max_hold_s': None, 'verify_timeout_s': VERIFY_TIMEOUT_S,
            'previous_trial': None, 'stop_trigger': None,
            'hold_warning': None, 'last_hold_warning': None, 'hold_observation_count': 0,
            'hold_guard_delay_s': HOLD_GUARD_DELAY_S, 'hold_guard_min_samples': HOLD_GUARD_MIN_SAMPLES,
            'hold_hard_error_deg': HOLD_HARD_ERROR_DEG, 'hold_hard_speed_deg_s': HOLD_HARD_SPEED_DEG_S,
            'settle_speed_deg_s': SETTLE_SPEED_DEG_S}


def number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(name+'必须是有限数字')
    return float(value)


AXIS_LABELS = {'upper': '大臂', 'lower': '小臂'}


def selected_motor(config, axis):
    if axis not in AXIS_LABELS:
        raise ValueError('关节必须为upper或lower')
    motors = [motor for motor in config['hardware']['motors'] if motor['name'] == axis]
    if len(motors) != 1:
        raise ValueError('配置中必须有且仅有一个'+axis+'关节')
    return motors[0]


class Tuner:
    def __init__(self, config, paths, mode='demo', device_factory=None, start_worker=True, axis='upper', lock_factory=None):
        motor = selected_motor(config, axis)
        self.axis, self.axis_label = axis, AXIS_LABELS[axis]
        self.mode = mode
        self.connection = {'axis': axis, 'axis_label': self.axis_label, 'device': motor['device'], 'motor_id': motor['id'],
                           'baudrate': int(config['hardware']['baudrate']),
                           **{k: motor[k] for k in ('sign', 'ratio', 'offset_deg')}}
        self.connection['parameter_protocol'] = config['hardware'].get('pid_parameter_protocol', 'vendor_v235')
        self._settings_backup = None
        self._settings_save_reference = None
        self.limits = {'min_deg': motor['limits_deg'][0], 'max_deg': motor['limits_deg'][1],
                       'max_test_delta_deg': 10., 'max_test_speed_deg_s': 60.,
                       'default_speed_deg_s': min(60., config['hardware']['max_speed_deg_s'])}
        self.runtime_dir = Path(paths['runtime_dir'])
        self.audit_dir = self.runtime_dir / 'pid_tuner' / mode
        self.factory = device_factory or (DemoDevice if mode == 'demo' else SerialDevice)
        self._ops = threading.RLock()
        self._state = threading.RLock()
        self._shutdown = threading.Event()
        self._stop_requested = threading.Event()
        self._stop_generation = 0
        self._device = None
        self._instance_lock = None
        self._lock_factory = lock_factory or instance_lock
        self.connected = False
        self.session_id = None
        self.params, self.baseline, self._baseline_raw, self._ram = {}, {}, {}, {}
        self.pid_status, self.latest = {}, None
        self.write_attempts = {}
        self.readings, self.logs = deque(maxlen=1800), deque(maxlen=500)
        self.events = []
        self._seq = self._log_seq = 0
        self._started = time.monotonic()
        self._last_client = time.monotonic()
        self._last_success = 0.
        self.last_error = None
        self.diagnostics = None
        self.motion = idle_motion()
        self._stable_since = None
        self._thread = None
        if start_worker:
            self._thread = threading.Thread(target=self._poll_loop, name='pid-monitor', daemon=True)
            self._thread.start()

    def log(self, level, message):
        with self._state:
            self._log_seq += 1
            self.logs.append({'seq': self._log_seq, 'time': round(time.monotonic()-self._started, 3),
                              'level': level, 'message': str(message)})

    def _event(self, action, **details):
        with self._state:
            self.events.append({'time': time.time(), 'action': action, **details})
        self.log('info', action)

    def _persist(self):
        if self.session_id:
            save_json(self.export(), self.audit_dir / (self.session_id+'.json'))

    def _require(self):
        if not self.connected or self._device is None:
            raise ValueError(f'请先连接{self.axis_label}')

    def _idle(self):
        self._require()
        self._check_cancel()
        if self.motion['active'] or self.motion.get('stop_required'):
            raise ValueError('请先停止试转，再修改参数')

    def _check_cancel(self, generation=None):
        if self._stop_requested.is_set() or (generation is not None and generation != self._stop_generation):
            raise InterruptedError('停止请求处理中，本次操作已取消')

    def _loop(self, loop):
        if loop not in LOOPS:
            raise ValueError('只允许修改角度环或速度环PID')
        return LOOPS[loop]

    def _read_one(self, param):
        raw = self._device.read_parameter(param)
        item = {**decode_parameter(param, raw), 'raw_hex': raw.hex(' ')}
        with self._state:
            self.params[PARAM_NAMES[param]] = item
        return raw

    def _read_all(self):
        for param in PARAM_NAMES:
            try:
                self._read_one(param)
            except Exception as exc:
                if param in (0x96, 0x97):
                    raise
                self.log('warning', f'只读参数0x{param:02X}不可用：{exc}')

    def connect(self):
        with self._ops:
            if self.connected:
                return
            self.session_id = uuid.uuid4().hex
            self._settings_backup = None
            self._settings_save_reference = None
            self.params, self.baseline, self._baseline_raw, self._ram = {}, {}, {}, {}
            self.pid_status = {}
            self.write_attempts = {}
            self.readings.clear()
            self.events = []
            self.latest = None
            self.last_error = None
            self.diagnostics = None
            self.motion = idle_motion()
            opened, phase = False, 'opening'
            try:
                if self.mode == 'real':
                    self._instance_lock = self._lock_factory(self.runtime_dir)
                    self._instance_lock.__enter__()
                self._device = self.factory(self.connection, self.log)
                self._device.open()
                opened, phase = True, 'parameters'
                self._read_all()
                self._settings_backup = getattr(self._device, 'settings_raw', None)
                self._settings_save_reference = self._settings_backup
                self.baseline = copy.deepcopy(self.params)
                self._baseline_raw = {k: bytes.fromhex(v['raw_hex']) for k, v in self.params.items()}
                self.pid_status = {k: 'read' for k in LOOPS}
                self.connected = True
                self._last_client = time.monotonic()
                phase = 'telemetry'
                self.sample()
                self._event('已连接并读取原始参数；尚未写入或使能', mode=self.mode, connection=self.connection)
                # Must succeed before any write or movement is made available.
                phase = 'backup'
                self._persist()
            except BaseException as exc:
                self.connected = False
                self.last_error = str(exc)
                # This fallback is read-only. Never turn a stop/interrupt or a
                # backup disk failure into another series of serial requests.
                if (opened and phase in ('parameters', 'telemetry') and isinstance(exc, Exception)
                        and not isinstance(exc, InterruptedError) and not self._stop_requested.is_set()
                        and hasattr(self._device, 'diagnose')):
                    try:
                        self.diagnostics = self._diagnostic_report()
                        self.diagnostics['initial_failure'] = {
                            'phase': phase, 'error': str(exc), 'details': getattr(exc, 'details', {})}
                    except Exception as diagnostic_exc:
                        self.log('warning', '只读诊断未完成：'+str(diagnostic_exc))
                try:
                    self._event('连接失败；没有写入任何参数', reason=str(exc),
                                details=getattr(exc, 'details', {}))
                    self._persist()
                except Exception as audit_exc:
                    self.log('error', '失败会话保存失败：'+str(audit_exc))
                finally:
                    self._release()
                raise

    def _diagnostic_report(self):
        report = self._device.diagnose()
        return {'created_at': time.time(), 'mode': self.mode,
                'connection': copy.deepcopy(self.connection), **report}

    def diagnose(self):
        generation = self._stop_generation
        with self._ops:
            self._check_cancel(generation)
            if self.motion['active'] or self.motion.get('stop_required'):
                raise ValueError(f'请先确认{self.axis_label}停止，再运行只读诊断')
            temporary = not self.connected
            if not self.session_id:
                self.session_id = uuid.uuid4().hex
            try:
                if temporary:
                    if self.mode == 'real':
                        self._instance_lock = self._lock_factory(self.runtime_dir)
                        self._instance_lock.__enter__()
                    self._device = self.factory(self.connection, self.log)
                    self._device.open()
                self._check_cancel(generation)
                self.diagnostics = self._diagnostic_report()
                self._event('只读通信诊断完成；未写入参数或改变电机运行状态')
            except InterruptedError:
                raise
            except Exception as exc:
                self.diagnostics = {'created_at': time.time(), 'mode': self.mode,
                    'connection': copy.deepcopy(self.connection),
                    'steps': [{'name': '只读诊断连接', 'command': 'open', 'ok': False,
                               'error': str(exc), 'details': getattr(exc, 'details', {})}],
                    'summary': ['诊断未完成，未发送任何参数写入、使能或运动命令。']}
                self.log('error', '只读诊断失败：'+str(exc))
            finally:
                if temporary:
                    self._release()
            self._persist()

    def _release(self):
        try:
            if self._device is not None:
                self._device.close()
        finally:
            self._device = None
            if self._instance_lock is not None:
                self._instance_lock.__exit__(None, None, None)
                self._instance_lock = None

    def disconnect(self):
        with self._ops:
            if not self.connected:
                return
            # Keep the port and monitor available when stopping fails, so the
            # operator and background worker can retry rather than abandon it.
            if self.motion['active']:
                self.motion['stop_reason'] = '断开连接'
                if self.motion.get('trial_outcome') == 'pending':
                    self.motion['trial_outcome'] = 'cancelled'
            self._stop_and_confirm()
            self._event(f'已停止并断开{self.axis_label}')
            self.connected = False
            self._release()
            self._persist()

    def read(self):
        with self._ops:
            self._idle()
            self._read_all()
            for loop in LOOPS:
                if (not self._vendor_mode() and loop in self._ram
                        and bytes.fromhex(self.params[loop]['raw_hex']) != self._ram[loop]):
                    self._ram.pop(loop)
                    self.pid_status[loop] = 'changed_on_device'
            self._event('已重新读取参数；连接时备份保持不变')
            self._persist()

    def sample(self):
        self._require()
        reading = self._device.telemetry()
        now = time.monotonic()
        with self._state:
            self._seq += 1
            target = self.motion['target_deg']
            row = {**reading, 'seq': self._seq, 't': round(now-self._started, 4),
                   'target_deg': target, 'motion_phase': self.motion.get('phase', 'idle'),
                   'trial_outcome': self.motion.get('trial_outcome'),
                   'trial_id': self.motion.get('trial_id'),
                   'target_error_deg': reading['angle_deg']-target if target is not None else None}
            self.latest = row
            self.readings.append(row)
            self._last_success = now
        return row

    def _stop_and_confirm(self):
        self._require()
        self._device.clear_interrupt()
        if self.motion.get('phase') == 'holding':
            self.motion['hold_elapsed_s'] = max(0., time.monotonic()-self.motion['hold_started_at'])
            stable_elapsed = max(0., time.monotonic()-self._stable_since) if self._stable_since is not None else 0.
            self.motion['hold_stable_s'] = VERIFY_HOLD_S if self.motion.get('trial_outcome') == 'passed' else min(VERIFY_HOLD_S, stable_elapsed)
        self.motion.update(active=False, stop_required=True, result='stopping', phase='stopping')
        try:
            self._device.stop()
            deadline, anchor, stable = time.monotonic()+2., None, None
            while time.monotonic() < deadline:
                row = self.sample()
                stationary = abs(row['speed_deg_s']) <= 1.
                if anchor is not None:
                    stationary = stationary and abs(row['angle_deg']-anchor) <= 0.2
                if stationary:
                    if stable is None:
                        stable, anchor = time.monotonic(), row['angle_deg']
                    if time.monotonic()-stable >= 0.3:
                        self.motion.update(result='stopped_confirmed', stop_required=False, phase='stopped')
                        self.motion.pop('stop_error', None)
                        return
                else:
                    stable, anchor = None, None
                time.sleep(0.05)
            raise TimeoutError('停止指令已发送，但未获得连续0.3秒停稳确认')
        except BaseException as exc:
            self.motion['result'] = 'stop_unconfirmed'
            self.motion['phase'] = 'stop_unconfirmed'
            self.motion['stop_error'] = str(exc)
            self.last_error = str(exc)
            raise

    def stop(self):
        self._stop_generation += 1
        self._stop_requested.set()
        device = self._device
        if device:
            device.interrupt()  # Let pending reads yield before waiting for the operation lock.
        with self._ops:
            try:
                if self.motion['active']:
                    self.motion['stop_reason'] = '手动停止'
                    if self.motion.get('trial_outcome') == 'pending':
                        self.motion['trial_outcome'] = 'cancelled'
                self._stop_and_confirm()
                self._event(f'停止{self.axis_label}：反馈已确认停稳')
                self._persist()
            finally:
                self._stop_requested.clear()

    def write(self, loop, values):
        generation = self._stop_generation
        param = self._loop(loop)
        if not isinstance(values, dict) or set(values) != {'kp', 'ki', 'kd'}:
            raise ValueError('必须同时提交整数kp、ki、kd')
        raw = encode_pid(values['kp'], values['ki'], values['kd'])
        with self._ops:
            self._idle()
            self._check_cancel(generation)
            self._stop_and_confirm()
            self._persist()  # Confirm the original backup is on disk before writing.
            previous_status = self.pid_status.get(loop, 'read')
            self.pid_status[loop] = 'write_pending'
            attempt = {'requested': copy.deepcopy(values), 'phase': 'preparing', 'ack_ok': False,
                       'started_at': time.time(), 'ok': False}
            self.write_attempts[loop] = attempt
            self._event('准备写入RAM', loop=loop, values=values)
            self._persist()
            try:
                self._check_cancel(generation)
                attempt['phase'] = 'write_ack'
                self._device.write_parameter(param, raw)
                attempt['ack_ok'] = True
                attempt['ack_details'] = copy.deepcopy(getattr(self._device, 'last_transaction', None))
                attempt['phase'] = 'readback'
                actual = self._read_one(param)
                attempt['readback_raw_hex'] = actual.hex(' ')
                attempt['readback_details'] = copy.deepcopy(getattr(self._device, 'last_transaction', None))
                if actual != raw and not self._vendor_mode():
                    raise ValueError('RAM写入应答成功，但读回数值不一致')
                if actual != raw:
                    attempt['warning'] = '整组设置读回仍与RAM目标不同；0x14不能作为运行RAM的独立读回证明'
            except BaseException as exc:
                if self._vendor_mode() and attempt['ack_ok'] and isinstance(exc, Exception) and not isinstance(exc, InterruptedError):
                    attempt['warning'] = 'RAM已应答，但整组设置读取失败：'+str(exc)
                    attempt['readback_details'] = copy.deepcopy(getattr(exc, 'details', None) or getattr(self._device, 'last_transaction', None))
                else:
                    attempt.update(error=str(exc), failed_phase=attempt['phase'],
                                   details=copy.deepcopy(getattr(exc, 'details', None) or
                                                         getattr(self._device, 'last_transaction', None)))
                    self.last_error = str(exc)
                    if attempt['phase'] == 'preparing':
                        self.pid_status[loop] = previous_status
                    else:
                        self.pid_status[loop] = 'write_unverified'
                        self._ram.pop(loop, None)
                    self._event('RAM写入未验证；请核对失败阶段和证据', loop=loop, attempt=copy.deepcopy(attempt))
                    self._persist()
                    raise
            attempt.update(ok=True, phase='acknowledged' if self._vendor_mode() else 'verified',
                           runtime_readback_verified=not self._vendor_mode())
            self.last_error = None
            self._ram[loop] = raw
            self.pid_status[loop] = 'ram_acknowledged' if self._vendor_mode() else 'ram_verified'
            self._event('RAM写入已应答；运行参数未独立读回' if self._vendor_mode() else 'RAM写入并读回一致；掉电失效', loop=loop, values=values)
            self._persist()

    def _vendor_mode(self):
        return self.mode == 'real' and self.connection['parameter_protocol'] == 'vendor_v235'

    def save(self, loop, confirm, expected=None):
        generation = self._stop_generation
        param = self._loop(loop)
        vendor = self._vendor_mode()
        if confirm != ('SAVE_ALL_ROM' if vendor else 'SAVE_ROM'):
            raise ValueError('需要明确确认保存ROM')
        with self._ops:
            self._idle()
            self._require_verified_parameters()
            self._check_cancel(generation)
            if loop not in self._ram:
                raise ValueError('请先完成RAM写入确认，再保存ROM')
            expected_raw = None
            if vendor:
                if not isinstance(expected, dict) or set(expected) != set(LOOPS):
                    raise ValueError('整组保存必须提交确认框中的两环目标值')
                expected_raw = {}
                for name in LOOPS:
                    values = expected[name]
                    if not isinstance(values, dict) or set(values) != {'kp', 'ki', 'kd'}:
                        raise ValueError('保存确认值字段不完整')
                    expected_raw[name] = encode_pid(**values)
                    intended = self._ram.get(name, bytes.fromhex(self.params[name]['raw_hex']))
                    if expected_raw[name] != intended:
                        raise ValueError('确认后的参数已变化，请重新查看确认框')
            self._stop_and_confirm()
            raw = self._read_one(param)
            if not vendor and raw != self._ram[loop]:
                self._ram.pop(loop)
                self.pid_status[loop] = 'changed_on_device'
                raise ValueError('电机当前参数已变化，请重新写入RAM验证')
            if vendor:
                snapshot = check_settings_save(self._settings_save_reference or b'', self._device.settings_raw or b'', self._ram)
                payload = bytearray(snapshot)
                for name, parameter in LOOPS.items():
                    if name not in self._ram and parameter_from_settings(snapshot, parameter) != expected_raw[name]:
                        raise ValueError('未写入RAM的另一环设置已变化，请重新读取后确认')
                    payload[PID_SLICES[parameter]] = expected_raw[name]
                raw = self._ram[loop]
            self._event('准备保存ROM', loop=loop, values=decode_parameter(param, raw))
            self._persist()
            sent = False
            try:
                self._check_cancel(generation)
                sent = True
                if vendor:
                    self._device.save_settings(bytes(payload))
                    self._read_all()
                    if self._device.settings_raw != bytes(payload):
                        raise ValueError('保存后整组设置再次变化；保存结果未确认')
                    self._settings_save_reference = bytes(payload)
                else:
                    self._device.write_parameter(param, raw, persistent=True)
            except BaseException as exc:
                if sent:
                    self.pid_status[loop] = 'rom_unverified'
                self.last_error = str(exc)
                self._event('ROM保存未确认', loop=loop, error=str(exc), command_may_have_been_sent=sent)
                self._persist()
                raise
            self.pid_status[loop] = 'rom_acknowledged'
            if vendor:
                for name in self._ram:
                    self.pid_status[name] = 'rom_acknowledged'
            self._event('ROM写入应答成功；需手动重上电后读回核验持久性', loop=loop,
                        values=decode_parameter(param, raw))
            self._persist()

    def restore(self, loop):
        self._loop(loop)
        if loop not in self.baseline:
            raise ValueError('没有本次连接的原始备份')
        values = {k: self.baseline[loop][k] for k in ('kp', 'ki', 'kd')}
        self.write(loop, values)
        self._event('本次连接备份已恢复到RAM；ROM没有自动改动', loop=loop)
        self._persist()

    def _require_verified_parameters(self):
        if any(v in ('write_pending', 'write_unverified', 'rom_unverified', 'changed_on_device')
               for v in self.pid_status.values()):
            raise ValueError('存在未确认的参数写入，禁止使能和试转；先查明写入应答及读回结果，不能靠重连确认运行参数')

    def enable(self, confirm):
        generation = self._stop_generation
        if confirm != 'ENABLE_MOTOR':
            raise ValueError(f'需要明确确认使能{self.axis_label}')
        with self._ops:
            self._idle()
            self._require_verified_parameters()
            self._check_cancel(generation)
            row = self.sample()
            if row['error_state']:
                raise ValueError('驱动器存在故障，不能使能：'+','.join(row['error_flags']))
            self._check_cancel(generation)
            self._device.enable()
            self.sample()
            self._event(f'已显式发送{self.axis_label}使能命令0x88')
            self._persist()

    def move(self, target_deg, speed_deg_s, *, start_gate=None):
        generation = self._stop_generation
        target = number(target_deg, '目标角度')
        speed = number(speed_deg_s, '试转速度')
        if not self.limits['min_deg'] <= target <= self.limits['max_deg']:
            raise ValueError(f'目标角度超出{self.axis_label}配置限位')
        if not 0 < speed <= self.limits['max_test_speed_deg_s']:
            raise ValueError('试转速度必须大于0且不超过60°/s')
        motor_target = round((self.connection['offset_deg'] + self.connection['sign']
                              * self.connection['ratio'] * target) * 100)
        if not 0 <= motor_target <= 35999:
            raise ValueError('电机目标超出本调参台A6单圈范围，不能下发负角目标')
        if round(speed * self.connection['ratio'] * 100) > 36000:
            raise ValueError('电机轴速度超出本工具支持范围')
        with self._ops:
            self._require()
            self._check_cancel()
            retarget = (self.motion['active'] and self.motion.get('phase') == 'holding'
                        and self.motion.get('trial_outcome') == 'passed'
                        and not self.motion.get('stop_required'))
            if not retarget:
                self._idle()
            self._require_verified_parameters()
            self._check_cancel(generation)
            try:
                row = self.sample()
            except BaseException as exc:
                if retarget:
                    self._abort_trial('下一角度反馈异常：'+str(exc))
                raise
            feedback_reason = None
            if row['error_state']:
                feedback_reason = '驱动器存在故障，不能试转：'+','.join(row['error_flags'])
            elif row['motor_state'] != 0:
                feedback_reason = f'{self.axis_label}处于关闭状态，请先显式使能'
            elif not self.limits['min_deg'] <= row['angle_deg'] <= self.limits['max_deg']:
                feedback_reason = '实测角度超出配置限位，请先停止检查'
            elif not retarget and abs(row['speed_deg_s']) > 1:
                feedback_reason = f'当前{self.axis_label}尚未停稳'
            elif retarget and time.monotonic()-self._last_client > 3:
                feedback_reason = '浏览器心跳中断超过3秒'
            if feedback_reason:
                if retarget:
                    old_error = row['angle_deg']-self.motion['target_deg']
                    self.motion['last_trial_error_deg'] = old_error
                    self.motion['hold_peak_error_deg'] = max(self.motion.get('hold_peak_error_deg') or 0., abs(old_error))
                    self._abort_trial(feedback_reason)
                raise ValueError(feedback_reason)
            if retarget:
                feedback_reason = self._hold_guard(row, time.monotonic())
                if feedback_reason:
                    self._abort_trial(feedback_reason)
                    raise ValueError(feedback_reason)
                if self.motion.get('hold_warning'):
                    raise ValueError('保持反馈正在波动观察，暂不切换下一角度；恢复稳定后再操作')
            delta = target-row['angle_deg']
            if abs(delta) > self.limits['max_test_delta_deg']:
                raise ValueError('单次试转与最新实测角度之差不能超过10°')
            previous = None
            if retarget:
                old_error = row['angle_deg']-self.motion['target_deg']
                previous = self._motion_snapshot()
                previous.pop('previous_trial', None)  # Keep repeated retargets from nesting snapshots.
                previous.update(active=False, phase='superseded', last_trial_error_deg=old_error)
                self._event('保持中切换下一角度；不发送停止或返回起点', previous_trial=previous,
                            from_deg=row['angle_deg'], target_deg=target, speed_deg_s=speed)
            self._event('准备单次试转', from_deg=row['angle_deg'], target_deg=target, speed_deg_s=speed)
            try:
                self._persist()
            except BaseException as exc:
                if retarget:
                    self._abort_trial('会话记录异常，取消下一角度：'+str(exc))
                raise
            # Mark ownership before sending: a missing A6 reply may still mean
            # that the motor accepted the command, so failure must attempt stop.
            self.motion = {**idle_motion(), 'active': True, 'target_deg': target, 'speed_deg_s': speed,
                           'start_deg': row['angle_deg'], 'trial_id': uuid.uuid4().hex,
                           'previous_trial': previous,
                           'trial_outcome': 'pending', 'result': 'running', 'phase': 'running',
                           'last_trial_error_deg': -delta,
                           'last_trial_speed_deg_s': row['speed_deg_s'],
                           'deadline': time.monotonic()+min(15., abs(delta)/speed*2+3)}
            self._stable_since = None
            self.last_error = None
            try:
                self._persist()  # Record ownership before the motion command can take effect.
                self._check_cancel(generation)
                if start_gate is not None:
                    start_gate()
                    self._check_cancel(generation)
                self._device.move(target, speed, row['motor_angle_deg'])
            except BaseException:
                self.motion.update(trial_outcome='failed', stop_reason='试转命令未确认')
                try:
                    self._stop_and_confirm()
                finally:
                    if not self.motion.get('stop_required'):
                        self.motion['result'] = 'command_failed'
                    self._persist()
                raise

    def _abort_trial(self, reason):
        self.motion['stop_trigger'] = {'reason': reason, 'sample': copy.deepcopy(self.latest),
                                       'hold_warning': copy.deepcopy(self.motion.get('hold_warning'))}
        self.motion.update(trial_outcome='failed', stop_reason=reason)
        try:
            self._stop_and_confirm()
            self.motion['result'] = 'aborted'
            self.last_error = reason
        finally:
            self._event('试转已停止' if not self.motion.get('stop_required') else '试转失败；停止尚未确认',
                        reason=reason, trial=self._motion_snapshot())
            self._persist()

    def _hold_guard(self, row, now):
        """Observe small transient feedback excursions before stopping a hold."""
        error = row['angle_deg']-self.motion['target_deg']
        speed = row['speed_deg_s']
        self.motion['last_trial_error_deg'] = error
        self.motion['last_trial_speed_deg_s'] = speed
        self.motion['hold_peak_error_deg'] = max(self.motion.get('hold_peak_error_deg') or 0., abs(error))
        self.motion['hold_peak_speed_deg_s'] = max(self.motion.get('hold_peak_speed_deg_s') or 0., abs(speed))
        position_bad = abs(error) > TARGET_TOLERANCE_DEG+1e-9
        speed_bad = abs(speed) > SETTLE_SPEED_DEG_S
        previous = self.motion.get('hold_warning')
        if not position_bad and not speed_bad:
            if previous:
                self.motion['last_hold_warning'] = {**previous, 'recovered': True,
                                                    'recovered_sample': copy.deepcopy(row)}
                self.log('info', '保持波动已恢复；未发送停止')
            self.motion['hold_warning'] = None
            if self._stable_since is None:
                self._stable_since = now
            return None
        if self.motion.get('trial_outcome') != 'passed':
            self._stable_since = None
        if previous:
            since, count = previous['since'], previous['samples']+1
        else:
            since, count = now, 1
            self.motion['hold_observation_count'] += 1
            self.log('warning', f'保持波动观察：误差{error:.3f}°，速度{speed:.2f}°/s；尚未停止')
        elapsed = max(0., now-since)
        warning = {'since': since, 'samples': count, 'elapsed_s': elapsed,
                   'error_deg': error, 'speed_deg_s': speed,
                   'position_exceeded': position_bad, 'speed_exceeded': speed_bad,
                   'sample': copy.deepcopy(row), 'recovered': False}
        self.motion['hold_warning'] = warning
        self.motion['last_hold_warning'] = copy.deepcopy(warning)
        if abs(error) >= HOLD_HARD_ERROR_DEG-1e-9:
            return f'保持位置明显偏离：误差{error:.3f}°达到±{HOLD_HARD_ERROR_DEG:g}°，立即停止'
        if abs(speed) >= HOLD_HARD_SPEED_DEG_S:
            return f'保持速度明显异常：速度{speed:.2f}°/s达到±{HOLD_HARD_SPEED_DEG_S:g}°/s，立即停止'
        if elapsed >= HOLD_GUARD_DELAY_S-1e-9 and count >= HOLD_GUARD_MIN_SAMPLES:
            exceeded = []
            if position_bad:
                exceeded.append(f'角度误差{error:.3f}°超过±{TARGET_TOLERANCE_DEG:g}°')
            if speed_bad:
                exceeded.append(f'速度{speed:.2f}°/s超过±{SETTLE_SPEED_DEG_S:g}°/s')
            return '保持持续异常：'+'；'.join(exceeded)+f'，持续{elapsed:.2f}秒/{count}个采样，停止'
        return None

    def _monitor_once(self):
        if self.motion.get('stop_required'):
            self._stop_and_confirm()
            self._event('重试停止：反馈现已确认停稳')
            self._persist()
            return
        row = self.sample()
        if not self.motion['active']:
            return
        now = time.monotonic()
        error = row['angle_deg']-self.motion['target_deg']
        self.motion['last_trial_error_deg'] = error
        self.motion['last_trial_speed_deg_s'] = row['speed_deg_s']
        in_position = abs(error) <= TARGET_TOLERANCE_DEG+1e-9 and abs(row['speed_deg_s']) <= SETTLE_SPEED_DEG_S
        reason = None
        if row['error_state']:
            reason = '驱动故障：'+','.join(row['error_flags'])
        elif row['motor_state'] != 0:
            reason = '电机已关闭，目标保持中断'
        elif not self.limits['min_deg'] <= row['angle_deg'] <= self.limits['max_deg']:
            reason = '实测角度超出配置限位'
        elif now-self._last_client > 3:
            reason = '浏览器心跳中断超过3秒'
        elif self.motion.get('phase') == 'holding':
            reason = self._hold_guard(row, now)
            if reason is None:
                if self.motion.get('trial_outcome') != 'passed' and now >= self.motion['verify_deadline']:
                    reason = '保持验证超过30秒仍未完成'
                elif (self.motion.get('trial_outcome') != 'passed' and not self.motion.get('hold_warning')
                      and self._stable_since is not None and now-self._stable_since >= VERIFY_HOLD_S-1e-9):
                    self.motion.update(result='completed', trial_outcome='passed',
                                       verification_sample=copy.deepcopy(row))
                    self._event('目标连续保持3秒验证成功；继续保持，尚未发送停止', trial=self._motion_snapshot())
                    self._persist()
                    return
        elif now > self.motion['deadline']:
            reason = '试转超过到位时间'
        elif in_position:
            if self._stable_since is None:
                self._stable_since = now
            if now-self._stable_since >= ARRIVAL_STABLE_S-1e-9:
                self.motion.update(phase='holding', result='observing', hold_started_at=now,
                                   verify_deadline=now+VERIFY_TIMEOUT_S, hold_peak_error_deg=abs(error),
                                   arrival_sample=copy.deepcopy(row))
                self._stable_since = now
                self._event('已到位，开始保持验证；尚未发送停止', trial=self._motion_snapshot())
                self._persist()
                return
        else:
            self._stable_since = None
        if reason:
            self._abort_trial(reason)

    def _poll_loop(self):
        while not self._shutdown.wait(0.1):
            if not self.connected or not self._ops.acquire(blocking=False):
                continue
            try:
                self._monitor_once()
            except Exception as exc:
                self.last_error = str(exc)
                self.log('error', str(exc))
                if self.motion['active'] or self.motion.get('stop_required'):
                    trial_failed = self.motion['active']
                    if trial_failed:
                        self.motion.update(trial_outcome='failed', stop_reason='反馈或会话记录异常：'+str(exc))
                    else:
                        self.motion['stop_error'] = str(exc)
                    try:
                        self._stop_and_confirm()
                    except Exception as stop_exc:
                        self.log('error', '停止未确认：'+str(stop_exc))
                    if trial_failed and not self.motion.get('stop_required'):
                        self.motion['result'] = 'feedback_failed'
                    try:
                        self._event('试验反馈或记录异常；验证失败' if trial_failed else '停止确认重试异常；保留试验结论',
                                    trial=self._motion_snapshot())
                        self._persist()
                    except Exception as audit_exc:
                        self.last_error = '会话记录保存失败：'+str(audit_exc)
                        self.log('error', self.last_error)
            finally:
                self._ops.release()

    def _motion_snapshot(self):
        motion = copy.deepcopy(self.motion)
        started = motion.get('hold_started_at')
        holding = motion.get('phase') == 'holding'
        now = time.monotonic()
        motion['hold_elapsed_s'] = max(0., now-started) if holding and started is not None else motion.get('hold_elapsed_s', 0.)
        motion['hold_remaining_s'] = (None if motion.get('trial_outcome') == 'passed'
                                      else max(0., motion.get('verify_deadline', now)-now)) if holding else 0.
        motion['hold_stable_s'] = min(VERIFY_HOLD_S, max(0., now-self._stable_since)) if holding and self._stable_since is not None else motion.get('hold_stable_s', VERIFY_HOLD_S if motion.get('trial_outcome') == 'passed' else 0.)
        return motion

    def state(self, since=0):
        self._last_client = time.monotonic()
        with self._state:
            return copy.deepcopy({'mode': self.mode, 'connected': self.connected,
                'connection': self.connection, 'limits': self.limits, 'session_id': self.session_id,
                'params': self.params, 'baseline': self.baseline, 'pid_status': self.pid_status,
                'write_attempts': self.write_attempts,
                'ram_ack_values': {k: decode_parameter(LOOPS[k], v) for k, v in self._ram.items()},
                'latest': self.latest, 'readings': [r for r in self.readings if r['seq'] > since],
                'logs': list(self.logs)[-100:], 'last_error': self.last_error,
                'diagnostics': self.diagnostics,
                'settings_backup_hex': self._settings_backup.hex(' ') if self._settings_backup else None,
                'save_scope': 'all_settings' if self.mode == 'real' and self.connection['parameter_protocol'] == 'vendor_v235' else 'single_loop',
                'motion': self._motion_snapshot(), 'last_feedback_age_s': (time.monotonic()-self._last_success if self.latest else None)})

    def export(self):
        with self._state:
            return copy.deepcopy({'format': 'lingkong-pid-session-v1', 'mode': self.mode,
                'session_id': self.session_id, 'connection': self.connection, 'limits': self.limits,
                'baseline': self.baseline, 'params': self.params, 'pid_status': self.pid_status,
                'write_attempts': self.write_attempts,
                'ram_ack_values': {k: decode_parameter(LOOPS[k], v) for k, v in self._ram.items()},
                'settings_backup_hex': self._settings_backup.hex(' ') if self._settings_backup else None,
                'events': self.events, 'readings': list(self.readings), 'logs': list(self.logs),
                'motion': self._motion_snapshot(), 'last_error': self.last_error, 'diagnostics': self.diagnostics})

    def csv(self):
        stream = io.StringIO()
        fields = ['axis', 'motor_id', 'seq', 't', 'target_deg', 'angle_deg', 'motor_angle_deg', 'speed_deg_s',
                  'motion_phase', 'trial_outcome', 'trial_id', 'target_error_deg',
                  'bus_current_a', 'iq_or_power_raw', 'voltage_v', 'temperature_c', 'error_state']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in self.export()['readings']:
            writer.writerow({**row, 'axis': self.axis, 'motor_id': self.connection['motor_id']})
        return '\ufeff'+stream.getvalue()

    def close(self):
        self._shutdown.set()
        if self._device:
            self._device.interrupt()
        if self._thread:
            self._thread.join(2)
        try:
            self.disconnect()
        finally:
            with self._ops:
                self.connected = False
                self._release()
