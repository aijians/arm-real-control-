"""Two existing tuners, one shared runtime lease and bounded paired commands."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import os
import threading
import time
import uuid

from control.runtime import instance_lock
from .service import AXIS_LABELS, Tuner, selected_motor


class SharedRuntimeLock:
    """A process-local lease; other controllers still see the original lock."""
    def __init__(self):
        self._guard = threading.Lock()
        self._count = 0
        self._lock = None

    @contextmanager
    def lease(self, directory):
        with self._guard:
            if not self._count:
                lock = instance_lock(directory)
                lock.__enter__()
                self._lock = lock
            self._count += 1
        try:
            yield
        finally:
            with self._guard:
                self._count -= 1
                if not self._count:
                    lock, self._lock = self._lock, None
                    lock.__exit__(None, None, None)


class DualTuner:
    def __init__(self, config, paths, mode='demo', device_factory=None, start_worker=True):
        motors = [selected_motor(config, axis) for axis in AXIS_LABELS]
        if len({os.path.realpath(m['device']) for m in motors}) != 2:
            raise ValueError('双轴调试要求两个不同的串口')
        self._lease = SharedRuntimeLock()
        self.axes = {axis: Tuner(config, paths, mode=mode, axis=axis,
            device_factory=device_factory, start_worker=start_worker,
            lock_factory=self._lease.lease) for axis in AXIS_LABELS}
        self.mode = mode
        self.last_error = None
        self.group = {'active': False, 'result': 'idle'}
        self._ops = threading.RLock()
        self._state = threading.RLock()
        self._stop_lock = threading.Lock()
        self._shutdown = threading.Event()
        self._cancel = threading.Event()
        self._generation = 0
        self._gate = None
        self._closed = False
        self._thread = None
        if start_worker:
            self._thread = threading.Thread(target=self._watch, name='pid-dual-monitor', daemon=True)
            self._thread.start()

    def select(self, axis):
        if axis not in self.axes:
            raise ValueError('请选择upper大臂或lower小臂')
        return self.axes[axis]

    def log(self, level, message):
        for tuner in self.axes.values():
            tuner.log(level, message)

    def state(self):
        result = {}
        for axis, tuner in self.axes.items():
            # Also refresh the heartbeat for the axis whose panel is hidden.
            state = tuner.state(2**63)
            result[axis] = {k: state[k] for k in ('connected', 'connection', 'limits',
                'latest', 'motion', 'pid_status', 'last_error', 'last_feedback_age_s')}
        with self._state:
            group = copy.deepcopy(self.group)
        return {'mode': self.mode, 'axes': result, 'group': group, 'last_error': self.last_error}

    @staticmethod
    def _parallel(actions):
        errors = {}
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix='pid-dual') as pool:
            futures = {axis: pool.submit(action) for axis, action in actions.items()}
            for axis, future in futures.items():
                try:
                    future.result()
                except BaseException as exc:
                    errors[axis] = exc
        return errors

    @staticmethod
    def _error(errors):
        return '; '.join(AXIS_LABELS[axis] + ': ' + str(exc) for axis, exc in errors.items())

    def connect_all(self):
        with self._ops:
            already = {axis: tuner.connected for axis, tuner in self.axes.items()}
            errors = self._parallel({axis: tuner.connect for axis, tuner in self.axes.items()})
            if errors:
                # Connection has only read parameters/telemetry. Roll back new
                # connections without emitting enable, movement or stop frames.
                for axis, tuner in self.axes.items():
                    if not already[axis] and tuner.connected:
                        with tuner._ops:
                            tuner.connected = False
                            tuner._release()
                raise RuntimeError(self._error(errors))

    def stop_all(self, reason='双轴手动停止'):
        self._generation += 1
        self._cancel.set()
        with self._state:
            # Claim the stop before either per-axis motion becomes inactive;
            # the watcher must not misclassify a requested stop as a fault.
            self.group.update(active=False, result='stopping', stop_reason=reason)
        gate = self._gate
        if gate is not None:
            gate.abort()
        # Do not wait for a group move/connection lock before requesting stop.
        for tuner in self.axes.values():
            if tuner._device:
                tuner._device.interrupt()
        with self._stop_lock:
            errors = self._parallel({axis: tuner.stop for axis, tuner in self.axes.items() if tuner.connected})
            with self._state:
                self.group.update(active=False, result='stop_unconfirmed' if errors else 'stopped',
                                  stop_reason=reason, stop_errors={a: str(e) for a, e in errors.items()})
            if errors:
                self.last_error = self._error(errors)
                raise RuntimeError(self.last_error)

    def stop_axis(self, axis):
        if self.group.get('active'):
            self.stop_all('联动期间单轴停止，停止双轴')
        else:
            self.select(axis).stop()

    def disconnect_axis(self, axis):
        if self.group.get('active'):
            self.stop_all('联动期间断开，停止双轴')
        self.select(axis).disconnect()

    def disconnect_all(self):
        self.stop_all('断开双轴')
        with self._ops:
            errors = self._parallel({axis: tuner.disconnect for axis, tuner in self.axes.items()})
            if errors:
                raise RuntimeError(self._error(errors))

    def move_both(self, targets):
        generation = self._generation
        if not isinstance(targets, dict) or set(targets) != set(self.axes):
            raise ValueError('必须同时提供upper和lower目标')
        for values in targets.values():
            if not isinstance(values, dict) or set(values) != {'target_deg', 'speed_deg_s'}:
                raise ValueError('每轴只接受target_deg与speed_deg_s')
        with self._ops:
            if generation != self._generation:
                raise InterruptedError('停止请求已取消排队的双轴试转')
            self._cancel.clear()
            gate = self._gate = threading.Barrier(2)
            with self._state:
                self.group = {'id': uuid.uuid4().hex, 'active': True, 'result': 'preparing',
                              'targets': copy.deepcopy(targets), 'requested_at': time.time()}
            def launch(axis):
                tuner = self.axes[axis]
                def ready():
                    if self._cancel.is_set():
                        raise InterruptedError('双轴试转已取消')
                    gate.wait(timeout=5.)
                    if self._cancel.is_set():
                        raise InterruptedError('双轴试转已取消')
                try:
                    tuner._event('双轴试转请求', group_id=self.group['id'])
                    tuner.move(**targets[axis], start_gate=ready)
                except BaseException:
                    gate.abort()
                    raise
            errors = self._parallel({axis: lambda axis=axis: launch(axis) for axis in self.axes})
            self._gate = None
            if errors or self._cancel.is_set():
                reason = self._error(errors) if errors else '双轴试转已取消'
                try:
                    self.stop_all('双轴下发失败：' + reason)
                finally:
                    self.last_error = reason
                raise RuntimeError(reason)
            with self._state:
                self.group.update(result='running')
            self.last_error = None

    def monitor_once(self):
        with self._state:
            if not self.group.get('active') or self.group.get('result') == 'preparing':
                return
        failed = []
        for axis, tuner in self.axes.items():
            with tuner._state:
                motion = copy.deepcopy(tuner.motion)
            if (not tuner.connected or not motion['active'] or motion.get('stop_required')
                    or motion.get('trial_outcome') in ('failed', 'cancelled')):
                failed.append(AXIS_LABELS[axis])
        if failed:
            with self._state:
                if not self.group.get('active'):
                    return
            self.stop_all('/'.join(failed) + '联动异常，停止双轴')
        elif all(t.motion.get('trial_outcome') == 'passed' for t in self.axes.values()):
            with self._state:
                self.group['result'] = 'holding'
        else:
            with self._state:
                self.group['result'] = 'running'

    def _watch(self):
        while not self._shutdown.wait(.1):
            try:
                self.monitor_once()
            except Exception as exc:
                self.last_error = str(exc)
                self.log('error', self.last_error)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._shutdown.set()
        if self._thread:
            self._thread.join(3)
        try:
            self.stop_all('关闭双轴调参台')
        finally:
            errors = self._parallel({axis: tuner.close for axis, tuner in self.axes.items()})
            if errors:
                raise RuntimeError(self._error(errors))
