"""Session interlocks and fault recovery using synthetic devices only."""
import copy
import csv
from contextlib import contextmanager
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from control.runtime import instance_lock
from pid_tuner.device import DemoDevice
from pid_tuner.protocol import encode_pid
from pid_tuner.service import Tuner


class Clock:
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class RecordingDevice(DemoDevice):
    """Faults may happen after a command took effect, as with a lost ACK."""
    def __init__(self, connection, logger):
        super().__init__(connection, logger)
        self.calls = []
        self.overrides = {}
        self.read_error = self.write_error = self.move_error = None
        self.telemetry_error = self.stop_error = None
        self.wrong_readback = False
        self.before_write = None

    def open(self):
        self.calls.append(('open', copy.deepcopy(self.connection)))
        super().open()

    def close(self):
        self.calls.append(('close',))

    def read_parameter(self, param):
        self.calls.append(('read', param))
        if self.read_error:
            raise self.read_error
        return super().read_parameter(param)

    def write_parameter(self, param, raw6, persistent=False):
        if self.before_write:
            self.before_write(param, raw6, persistent)
        self.calls.append(('write', param, raw6, persistent))
        super().write_parameter(param, raw6, persistent)
        if self.wrong_readback and not persistent:
            self.params[param] = encode_pid(999, 998, 997)
        if self.write_error:
            raise self.write_error

    def telemetry(self):
        self.calls.append(('telemetry',))
        if self.telemetry_error:
            raise self.telemetry_error
        return {**super().telemetry(), **self.overrides}

    def enable(self):
        self.calls.append(('enable',))
        super().enable()

    def stop(self):
        self.calls.append(('stop',))
        if self.stop_error:
            raise self.stop_error
        super().stop()

    def move(self, target_deg, speed_deg_s, current_motor_deg):
        self.calls.append(('move', target_deg, speed_deg_s, current_motor_deg))
        super().move(target_deg, speed_deg_s, current_motor_deg)
        if self.move_error:
            raise self.move_error


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.clock = Clock()
        self.addCleanup(patch.stopall)
        patch('pid_tuner.service.time.monotonic', side_effect=self.clock.monotonic).start()
        patch('pid_tuner.service.time.sleep', side_effect=self.clock.advance).start()
        self.config = {'hardware': {
            'baudrate': 115200, 'max_speed_deg_s': 18,
            'motors': [
                {'name': 'upper', 'device': '/synthetic/upper-only', 'id': 1,
                 'sign': 1, 'ratio': 1.0, 'offset_deg': 0.0, 'limits_deg': [0, 180]},
                {'name': 'lower', 'device': '/never/open/lower', 'id': 2},
            ]}}
        self.paths = {'runtime_dir': Path(self.directory.name)}
        self.instances = []

    def create(self, mode='demo', connect=True):
        def factory(connection, logger):
            device = RecordingDevice(connection, logger)
            self.instances.append(device)
            return device
        tuner = Tuner(self.config, self.paths, mode=mode,
                      device_factory=factory, start_worker=False)
        self.addCleanup(self.cleanup_tuner, tuner)
        if connect:
            tuner.connect()
        return tuner

    @staticmethod
    def cleanup_tuner(tuner):
        if tuner._device:
            device = tuner._device
            device.stop_error = device.telemetry_error = None
            device.overrides = {}
        tuner.close()

    @property
    def device(self):
        return self.instances[-1]

    def audit(self, tuner):
        return json.loads((tuner.audit_dir / (tuner.session_id + '.json')).read_text())

    def writes(self, persistent=None):
        return [call for call in self.device.calls
                if call[0] == 'write' and (persistent is None or call[3] == persistent)]

    def test_connect_reads_upper_port_and_persists_original_without_motor_commands(self):
        tuner = self.create()
        self.assertTrue(tuner.connected)
        self.assertEqual([call[0] for call in self.device.calls].count('open'), 1)
        self.assertEqual(self.device.connection['device'], '/synthetic/upper-only')
        self.assertEqual(self.device.connection['motor_id'], 1)
        self.assertEqual({call[1] for call in self.device.calls if call[0] == 'read'},
                         {0x96, 0x97, 0x98, 0x99})
        self.assertFalse(any(call[0] in ('write', 'enable', 'move', 'stop')
                             for call in self.device.calls))
        self.assertEqual(self.audit(tuner)['baseline']['speed']['kp'], 100)
        tuner.connect()
        self.assertEqual(len(self.instances), 1)

    def test_failed_initial_backup_closes_connection_and_prevents_writes(self):
        tuner = self.create(connect=False)
        with patch('pid_tuner.service.save_json', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                tuner.connect()
        self.assertFalse(tuner.connected)
        self.assertIsNone(tuner._device)
        self.assertIn(('close',), self.device.calls)
        with self.assertRaises(ValueError):
            tuner.write('speed', {'kp': 300, 'ki': 30, 'kd': 0})
        self.assertFalse(self.writes())

    def test_backup_exists_before_first_ram_write_and_original_survives_reread(self):
        tuner = self.create()
        original = copy.deepcopy(tuner.baseline)
        observed = []
        self.device.before_write = lambda *_: observed.append(self.audit(tuner)['baseline'])
        tuner.write('speed', {'kp': 300, 'ki': 30, 'kd': 0})
        tuner.read()
        self.assertEqual(observed, [original])
        self.assertEqual(tuner.baseline, original)
        self.assertEqual(tuner.params['speed']['kp'], 300)
        self.assertEqual(tuner.pid_status['speed'], 'ram_verified')

    def test_disk_failure_before_ram_command_keeps_device_unchanged(self):
        tuner = self.create()
        with patch('pid_tuner.service.save_json', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                tuner.write('speed', {'kp': 300, 'ki': 30, 'kd': 0})
        self.assertFalse(self.writes())
        self.assertEqual(self.device.params[0x97], encode_pid(100, 30, 0))

    def test_only_two_pid_loops_and_complete_integer_fields_are_writable(self):
        tuner = self.create()
        for loop in ('current', 'torque_limit', '', '0x98'):
            with self.subTest(loop=loop), self.assertRaises(ValueError):
                tuner.write(loop, {'kp': 1, 'ki': 2, 'kd': 3})
        for values in ({'kp': 1, 'ki': 2}, {'kp': 1, 'ki': 2, 'kd': 3, 'other': 4},
                       {'kp': True, 'ki': 2, 'kd': 3}, {'kp': 1.0, 'ki': 2, 'kd': 3},
                       {'kp': -1, 'ki': 2, 'kd': 3}, {'kp': 65536, 'ki': 2, 'kd': 3}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                tuner.write('speed', values)
        self.assertFalse(self.writes())

    def test_successful_ram_write_is_read_back_and_does_not_change_rom(self):
        tuner = self.create()
        original_rom = dict(self.device.rom)
        tuner.write('angle', {'kp': 50, 'ki': 0, 'kd': 9})
        self.assertEqual(tuner.pid_status['angle'], 'ram_verified')
        self.assertEqual(tuner.params['angle']['kd'], 9)
        write_index = next(i for i, call in enumerate(self.device.calls) if call[0] == 'write')
        self.assertIn(('read', 0x96), self.device.calls[write_index + 1:])
        self.assertEqual(self.device.rom, original_rom)
        self.assertEqual(len(self.writes(False)), 1)

    def test_lost_ram_ack_cannot_be_followed_by_rom_save(self):
        tuner = self.create()
        self.device.write_error = TimeoutError('RAM ACK lost')
        with self.assertRaises(TimeoutError):
            tuner.write('speed', {'kp': 300, 'ki': 30, 'kd': 0})
        self.assertEqual(self.device.params[0x97], encode_pid(300, 30, 0))
        self.assertEqual(tuner.pid_status['speed'], 'write_unverified')
        with self.assertRaises(ValueError):
            tuner.save('speed', 'SAVE_ROM')
        self.assertFalse(self.writes(True))

    def test_wrong_ram_readback_cannot_be_followed_by_rom_save(self):
        tuner = self.create()
        self.device.wrong_readback = True
        with self.assertRaisesRegex(ValueError, '读回'):
            tuner.write('speed', {'kp': 300, 'ki': 30, 'kd': 0})
        self.assertEqual(tuner.pid_status['speed'], 'write_unverified')
        with self.assertRaises(ValueError):
            tuner.save('speed', 'SAVE_ROM')
        self.assertFalse(self.writes(True))

    def test_rom_requires_ram_verification_and_exact_confirmation(self):
        tuner = self.create()
        with self.assertRaises(ValueError):
            tuner.save('speed', 'SAVE_ROM')
        tuner.write('speed', {'kp': 300, 'ki': 30, 'kd': 0})
        for confirm in (True, None, 'save_rom', ''):
            with self.subTest(confirm=confirm), self.assertRaises(ValueError):
                tuner.save('speed', confirm)
        self.assertFalse(self.writes(True))
        tuner.save('speed', 'SAVE_ROM')
        self.assertEqual(len(self.writes(True)), 1)
        self.assertEqual(self.device.rom[0x97], encode_pid(300, 30, 0))
        self.assertEqual(tuner.pid_status['speed'], 'rom_acknowledged')

    def test_changed_device_value_blocks_rom_and_invalidates_verification(self):
        tuner = self.create()
        tuner.write('speed', {'kp': 300, 'ki': 30, 'kd': 0})
        self.device.params[0x97] = encode_pid(400, 40, 0)
        with self.assertRaisesRegex(ValueError, '已变化'):
            tuner.save('speed', 'SAVE_ROM')
        self.assertEqual(tuner.pid_status['speed'], 'changed_on_device')
        self.assertFalse(self.writes(True))
        with self.assertRaises(ValueError):
            tuner.save('speed', 'SAVE_ROM')

    def test_manual_reread_invalidates_ram_verification_when_device_changes(self):
        tuner = self.create()
        tuner.write('speed', {'kp': 300, 'ki': 30, 'kd': 0})
        self.device.params[0x97] = encode_pid(400, 40, 0)
        tuner.read()
        self.assertEqual(tuner.pid_status['speed'], 'changed_on_device')
        with self.assertRaises(ValueError):
            tuner.save('speed', 'SAVE_ROM')
        self.assertEqual(tuner.baseline['speed']['kp'], 100)

    def test_lost_rom_ack_reports_uncertain_persistence(self):
        tuner = self.create()
        tuner.write('speed', {'kp': 300, 'ki': 30, 'kd': 0})
        self.device.write_error = TimeoutError('ROM ACK lost')
        with self.assertRaises(TimeoutError):
            tuner.save('speed', 'SAVE_ROM')
        self.assertEqual(tuner.pid_status['speed'], 'rom_unverified')
        self.assertEqual(self.audit(tuner)['pid_status']['speed'], 'rom_unverified')

    def test_restore_uses_connection_backup_only_in_ram(self):
        tuner = self.create()
        original_rom = dict(self.device.rom)
        tuner.write('speed', {'kp': 500, 'ki': 80, 'kd': 4})
        tuner.restore('speed')
        self.assertEqual(self.device.params[0x97], encode_pid(100, 30, 0))
        self.assertEqual(tuner.params['speed']['kp'], 100)
        self.assertEqual(tuner.pid_status['speed'], 'ram_verified')
        self.assertEqual(self.device.rom, original_rom)
        self.assertFalse(self.writes(True))

    def test_enable_requires_confirmation_and_rejects_fault(self):
        tuner = self.create()
        with self.assertRaises(ValueError):
            tuner.enable(True)
        self.device.overrides = {'error_state': 2, 'error_flags': ['overvoltage']}
        with self.assertRaisesRegex(ValueError, '故障'):
            tuner.enable('ENABLE_MOTOR')
        self.assertNotIn(('enable',), self.device.calls)
        self.device.overrides = {}
        tuner.enable('ENABLE_MOTOR')
        self.assertIn(('enable',), self.device.calls)

    def test_move_validates_finite_input_distance_speed_and_configured_limits(self):
        tuner = self.create()
        for target, speed in ((float('nan'), 5), (170, float('inf')), (True, 5),
                              ('170', 5), (170, True), (180.01, 5), (-0.01, 5),
                              (158, 5), (170, 0), (170, -1), (170, 60.01)):
            with self.subTest(target=target, speed=speed), self.assertRaises(ValueError):
                tuner.move(target, speed)
        self.assertFalse(any(call[0] == 'move' for call in self.device.calls))
        tuner.move(179.11, 60)
        self.assertTrue(tuner.motion['active'])

    def test_move_rejects_fault_motor_off_and_existing_motion(self):
        tuner = self.create()
        for override in ({'error_state': 1, 'error_flags': ['fault']},
                         {'motor_state': 0x10}, {'speed_deg_s': 1.01}):
            self.device.overrides = override
            with self.subTest(override=override), self.assertRaises(ValueError):
                tuner.move(170, 5)
        self.device.overrides = {}
        tuner.move(170, 5)
        with self.assertRaises(ValueError):
            tuner.move(171, 5)
        with self.assertRaises(ValueError):
            tuner.write('speed', {'kp': 300, 'ki': 30, 'kd': 0})
        self.assertEqual(len([call for call in self.device.calls if call[0] == 'move']), 1)

    def test_missing_motion_ack_still_attempts_and_confirms_stop(self):
        tuner = self.create()
        self.device.move_error = TimeoutError('A6 ACK lost after acceptance')
        with self.assertRaises(TimeoutError):
            tuner.move(175, 5)
        operations = [call[0] for call in self.device.calls]
        self.assertGreater(operations.index('stop'), operations.index('move'))
        self.assertFalse(tuner.motion['active'])
        self.assertEqual(tuner.motion['result'], 'command_failed')
        self.assertEqual(self.device.target, self.device.angle)
        self.assertEqual(self.device.speed, 0)

    def test_missing_motion_and_stop_acks_preserve_unconfirmed_stop_until_recovery(self):
        tuner = self.create()
        self.device.move_error = TimeoutError('A6 ACK lost after acceptance')
        self.device.stop_error = TimeoutError('stop ACK lost')
        with self.assertRaises(TimeoutError):
            tuner.move(175, 5)
        self.assertTrue(tuner.motion['stop_required'])
        self.assertEqual(tuner.motion['result'], 'stop_unconfirmed')
        self.assertEqual(self.audit(tuner)['motion']['result'], 'stop_unconfirmed')
        with self.assertRaises(ValueError):
            tuner.move(170, 1)
        self.device.stop_error = None
        tuner._monitor_once()
        self.assertFalse(tuner.motion['stop_required'])
        self.assertEqual(self.audit(tuner)['motion']['result'], 'stopped_confirmed')
        self.assertFalse(self.audit(tuner)['motion']['stop_required'])

    def test_operations_queued_before_completed_stop_cannot_issue_commands_afterward(self):
        tuner = self.create()
        tuner.write('speed', {'kp': 300, 'ki': 30, 'kd': 0})
        original_lock = tuner._ops

        @contextmanager
        def stop_before_lock_entry():
            # Simulate another HTTP request acquiring the lock and finishing
            # stop while this request, which already captured its generation,
            # was waiting. No real scheduling or serial connection is needed.
            with patch.object(tuner, '_ops', original_lock):
                tuner.stop()
            yield

        actions = {
            'write': lambda: tuner.write('angle', {'kp': 70, 'ki': 0, 'kd': 0}),
            'save': lambda: tuner.save('speed', 'SAVE_ROM'),
            'enable': lambda: tuner.enable('ENABLE_MOTOR'),
            'move': lambda: tuner.move(175, 5),
        }
        for name, action in actions.items():
            first_call = len(self.device.calls)
            with self.subTest(action=name), \
                    patch.object(tuner, '_ops', stop_before_lock_entry()), \
                    self.assertRaises(InterruptedError):
                action()
            new_calls = self.device.calls[first_call:]
            self.assertIn(('stop',), new_calls)
            self.assertFalse(any(call[0] in ('write', 'enable', 'move') for call in new_calls))
            self.assertFalse(tuner.motion['active'])

    def test_stop_after_interlocks_but_before_send_cancels_all_command_types(self):
        tuner = self.create()
        tuner.write('speed', {'kp': 300, 'ki': 30, 'kd': 0})
        actions = {
            'write': lambda: tuner.write('angle', {'kp': 70, 'ki': 0, 'kd': 0}),
            'save': lambda: tuner.save('speed', 'SAVE_ROM'),
            'enable': lambda: tuner.enable('ENABLE_MOTOR'),
            'move': lambda: tuner.move(175, 5),
        }
        for name, action in actions.items():
            original = tuner.sample if name == 'enable' else tuner._persist
            stopped = []

            def stop_during_preparation(*args, **kwargs):
                result = original(*args, **kwargs)
                if not stopped:
                    stopped.append(True)
                    tuner.stop()
                return result

            first_call = len(self.device.calls)
            checkpoint = 'sample' if name == 'enable' else '_persist'
            with self.subTest(action=name), \
                    patch.object(tuner, checkpoint, side_effect=stop_during_preparation), \
                    self.assertRaises(InterruptedError):
                action()
            self.assertEqual(stopped, [True])
            self.assertFalse(any(call[0] in ('write', 'enable', 'move')
                                 for call in self.device.calls[first_call:]))
            self.assertFalse(tuner.motion['active'])

    def test_browser_heartbeat_loss_stops_active_trial(self):
        tuner = self.create()
        tuner.move(175, 1)
        self.clock.advance(3.01)
        tuner._monitor_once()
        self.assertFalse(tuner.motion['active'])
        self.assertEqual(tuner.motion['result'], 'aborted')
        self.assertIn('心跳', tuner.last_error)
        self.assertIn(('stop',), self.device.calls)

    def test_state_poll_refreshes_heartbeat_and_does_not_return_shared_data(self):
        tuner = self.create()
        tuner.move(175, 1)
        self.clock.advance(2)
        snapshot = tuner.state()
        snapshot['params']['speed']['kp'] = 65535
        self.clock.advance(2)
        tuner._monitor_once()
        self.assertTrue(tuner.motion['active'])
        self.assertEqual(tuner.params['speed']['kp'], 100)
        previous = tuner.state()['latest']['seq']
        tuner.sample()
        self.assertTrue(all(row['seq'] > previous for row in tuner.state(previous)['readings']))

    def test_fault_during_trial_stops_and_records_reason(self):
        tuner = self.create()
        tuner.move(175, 1)
        self.device.overrides = {'error_state': 1, 'error_flags': ['overcurrent']}
        tuner._monitor_once()
        self.assertEqual(tuner.motion['result'], 'aborted')
        self.assertFalse(tuner.motion['active'])
        self.assertIn('overcurrent', tuner.last_error)
        self.assertIn(('stop',), self.device.calls)

    def test_stop_confirmation_times_out_when_feedback_says_motor_is_still_moving(self):
        tuner = self.create()
        tuner.move(175, 1)
        self.device.overrides = {'speed_deg_s': 2}
        with self.assertRaises(TimeoutError):
            tuner.stop()
        self.assertEqual(tuner.motion['result'], 'stop_unconfirmed')
        self.assertTrue(tuner.motion['stop_required'])
        self.assertIn(('stop',), self.device.calls)

    def test_lost_stop_ack_blocks_actions_until_monitor_confirms_recovery(self):
        tuner = self.create()
        tuner.move(175, 1)
        self.device.stop_error = TimeoutError('stop ACK lost')
        with self.assertRaises(TimeoutError):
            tuner.stop()
        self.assertTrue(tuner.motion['stop_required'])
        self.assertEqual(tuner.motion['result'], 'stop_unconfirmed')
        for action in (lambda: tuner.move(170, 1),
                       lambda: tuner.write('speed', {'kp': 300, 'ki': 30, 'kd': 0}),
                       lambda: tuner.enable('ENABLE_MOTOR')):
            with self.assertRaises(ValueError):
                action()
        self.device.stop_error = None
        tuner._monitor_once()
        self.assertFalse(tuner.motion['stop_required'])
        self.assertEqual(tuner.motion['result'], 'stopped_confirmed')
        self.assertEqual(len([call for call in self.device.calls if call[0] == 'stop']), 2)

    def test_stop_command_with_missing_feedback_remains_unconfirmed_until_retry(self):
        tuner = self.create()
        tuner.move(175, 1)
        self.device.telemetry_error = OSError('feedback disconnected')
        with self.assertRaises(OSError):
            tuner.stop()
        self.assertTrue(tuner.motion['stop_required'])
        self.assertEqual(tuner.motion['result'], 'stop_unconfirmed')
        self.device.telemetry_error = None
        tuner._monitor_once()
        self.assertFalse(tuner.motion['stop_required'])
        self.assertEqual(tuner.motion['result'], 'stopped_confirmed')

    def test_polling_survives_feedback_and_audit_failures_and_retries_stop(self):
        tuner = self.create()
        tuner.move(175, 1)
        self.device.telemetry_error = OSError('feedback disconnected')
        iterations = []

        def next_iteration(_timeout):
            iterations.append(1)
            if len(iterations) == 2:
                self.device.telemetry_error = None
            return len(iterations) > 2

        with patch.object(tuner._shutdown, 'wait', side_effect=next_iteration), \
                patch('pid_tuner.service.save_json', side_effect=OSError('disk full')):
            tuner._poll_loop()
        self.assertEqual(len(iterations), 3)
        self.assertFalse(tuner.motion['stop_required'])
        self.assertEqual(tuner.motion['result'], 'stopped_confirmed')
        self.assertEqual(len([call for call in self.device.calls if call[0] == 'stop']), 2)
        self.assertTrue(any('disk full' in log['message'] for log in tuner.logs))

    def test_trial_deadline_stops_even_with_live_browser(self):
        tuner = self.create()
        tuner.move(175, 1)
        self.clock.advance(16)
        tuner.state()
        tuner._monitor_once()
        self.assertEqual(tuner.motion['result'], 'aborted')
        self.assertIn('到位时间', tuner.last_error)

    def test_disconnect_stops_before_closing_and_reconnect_starts_new_backup(self):
        tuner = self.create()
        session_id = tuner.session_id
        tuner.write('angle', {'kp': 100, 'ki': 1, 'kd': 2})
        tuner.move(175, 1)
        tuner.disconnect()
        operations = [call[0] for call in self.device.calls]
        self.assertLess(max(i for i, value in enumerate(operations) if value == 'stop'),
                        operations.index('close'))
        self.assertFalse(tuner.connected)
        tuner.connect()
        self.assertNotEqual(tuner.session_id, session_id)
        self.assertEqual(tuner.baseline['angle']['kp'], 50)
        self.assertEqual(tuner.pid_status['angle'], 'read')

    def test_failed_disconnect_preserves_connection_and_real_lock_for_stop_retry(self):
        tuner = self.create(mode='real')
        device = self.device
        tuner.move(175, 1)
        device.stop_error = TimeoutError('stop ACK lost')
        with self.assertRaises(TimeoutError):
            tuner.disconnect()
        self.assertTrue(tuner.connected)
        self.assertIs(tuner._device, device)
        self.assertNotIn(('close',), device.calls)
        self.assertTrue(tuner.motion['stop_required'])
        with self.assertRaisesRegex(RuntimeError, 'runtime_already_running'):
            with instance_lock(self.paths['runtime_dir']):
                pass
        device.stop_error = None
        tuner._monitor_once()
        self.assertFalse(tuner.motion['stop_required'])
        self.assertEqual(self.audit(tuner)['motion']['result'], 'stopped_confirmed')
        tuner.disconnect()
        self.assertFalse(tuner.connected)
        self.assertIn(('close',), device.calls)
        with instance_lock(self.paths['runtime_dir']):
            pass

    def test_process_close_releases_port_and_real_lock_even_when_stop_fails(self):
        tuner = self.create(mode='real')
        device = self.device
        tuner.move(175, 1)
        device.stop_error = TimeoutError('stop ACK lost')
        with self.assertRaises(TimeoutError):
            tuner.close()
        self.assertFalse(tuner.connected)
        self.assertIsNone(tuner._device)
        self.assertIn(('close',), device.calls)
        self.assertTrue(tuner.motion['stop_required'])
        with instance_lock(self.paths['runtime_dir']):
            pass

    def test_real_session_respects_main_runtime_lock_before_constructing_device(self):
        tuner = self.create(mode='real', connect=False)
        with instance_lock(self.paths['runtime_dir']):
            with self.assertRaisesRegex(RuntimeError, 'runtime_already_running'):
                tuner.connect()
        self.assertFalse(self.instances)
        tuner.connect()
        self.assertTrue(tuner.connected)
        # The injected RecordingDevice remains synthetic in real-mode lock tests.
        with self.assertRaisesRegex(RuntimeError, 'runtime_already_running'):
            with instance_lock(self.paths['runtime_dir']):
                pass
        tuner.disconnect()
        with instance_lock(self.paths['runtime_dir']):
            pass

    def test_failed_real_connect_releases_runtime_lock(self):
        tuner = self.create(mode='real', connect=False)
        with patch('pid_tuner.service.save_json', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                tuner.connect()
        with instance_lock(self.paths['runtime_dir']):
            pass

    def test_export_and_csv_contain_feedback_and_original_snapshot(self):
        tuner = self.create()
        tuner.sample()
        exported = tuner.export()
        exported['baseline']['speed']['kp'] = 999
        self.assertEqual(tuner.baseline['speed']['kp'], 100)
        self.assertEqual(exported['format'], 'lingkong-pid-session-v1')
        rows = list(csv.DictReader(io.StringIO(tuner.csv().lstrip('\ufeff'))))
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(float(rows[0]['angle_deg']), 169.11)
        self.assertIn('bus_current_a', rows[0])


if __name__ == '__main__':
    unittest.main()
