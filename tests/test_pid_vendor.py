"""Real captured frames and offline tests; never access physical UARTs."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from control.protocol import Frame
from pid_tuner.device import DemoDevice
from pid_tuner.service import Tuner
from pid_tuner.protocol import encode_pid
from pid_tuner.vendor import (check_settings_save, extract_reply, frame_bytes,
                              parameter_from_settings, validate_settings)
from test_pid_transport import motor_emulator, send_reply

FRAMES = json.loads((Path(__file__).parent/'fixtures/vendor_settings_frames.json').read_text())['motors']
UPPER = bytes.fromhex(FRAMES['upper'])[5:-1]


class VendorTests(unittest.TestCase):
    def test_captured_frames_decode_and_reencode_exactly(self):
        for name, expected in (('upper', (300, 10)), ('lower', (500, 25))):
            raw = bytes.fromhex(FRAMES[name])
            frame = extract_reply(bytearray(raw))
            self.assertEqual(frame_bytes(frame.command, frame.motor_id, frame.payload), raw)
            self.assertEqual(len(validate_settings(frame.payload)), 104)
            self.assertEqual(parameter_from_settings(frame.payload, 0x96), encode_pid(50, 0, 0))
            self.assertEqual(parameter_from_settings(frame.payload, 0x97), encode_pid(*expected, 0))
            self.assertEqual(parameter_from_settings(frame.payload, 0x99)[:2], (850).to_bytes(2, 'little'))

    def test_fragmented_and_bad_checksum_settings_reply(self):
        raw = bytes.fromhex(FRAMES['upper'])
        bad = bytearray(raw); bad[-1] ^= 1
        buffer = bytearray(b'noise') + bad + raw[:60]
        self.assertIsNone(extract_reply(buffer))
        buffer.extend(raw[60:])
        self.assertEqual(extract_reply(buffer).payload, UPPER)
        self.assertFalse(buffer)

    def test_extended_frames_restricted_to_vendor_commands(self):
        with self.assertRaises(ValueError): frame_bytes(0xA6, 1, bytes(104))
        with self.assertRaises(ValueError): validate_settings(UPPER[:-1])
        bad = bytearray(UPPER); bad[-1] ^= 1
        with self.assertRaises(ValueError): validate_settings(bad)

    def test_real_transport_reads_104_bytes_instead_of_0x40(self):
        def respond(request, peer):
            self.assertEqual(request.command, 0x14)
            send_reply(peer, frame_bytes(0x14, 17, UPPER))
        with motor_emulator(respond) as (device, _, requests, _):
            device.vendor_mode = True
            self.assertEqual(device.read_parameter(0x97), encode_pid(300, 10, 0))
            self.assertEqual(device.settings_raw, UPPER)
            self.assertEqual([r.command for r in requests], [0x14])

    def test_persistent_write_is_whole_preserved_settings_with_empty_ack(self):
        with motor_emulator(lambda *_: None) as (device, _, _, _):
            device.vendor_mode = True
            device.settings_raw = UPPER
            device._transaction = Mock(return_value=Frame(0x15, 17, b''))
            device.write_parameter(0x97, encode_pid(300, 10, 0), persistent=True)
            packet, command = device._transaction.call_args.args
            self.assertEqual(command, 0x15)
            self.assertEqual(packet[5:-1], UPPER)
            self.assertEqual(len(packet), 110)

    def test_save_settings_rejects_old_group_after_successful_save_ack(self):
        target = bytearray(UPPER)
        target[56:62] = encode_pid(400, 10, 0)
        target = bytes(target)
        with motor_emulator(lambda *_: None) as (device, _, _, _):
            device.vendor_mode = True
            device._transaction = Mock(side_effect=[Frame(0x15, 17, b''), Frame(0x14, 17, UPPER)])
            with self.assertRaisesRegex(ValueError, '0x14读回'):
                device.save_settings(target)
            self.assertEqual([call.args[1] for call in device._transaction.call_args_list], [0x15, 0x14])
            self.assertEqual(device._transaction.call_args_list[0].args[0], frame_bytes(0x15, 17, target))

    def test_save_settings_does_not_hide_readback_timeout_after_save_ack(self):
        with motor_emulator(lambda *_: None) as (device, _, _, _):
            device.vendor_mode = True
            device._transaction = Mock(side_effect=[Frame(0x15, 17, b''), TimeoutError('0x14 timeout after save')])
            with self.assertRaisesRegex(TimeoutError, '0x14 timeout after save'):
                device.save_settings(UPPER)
            self.assertEqual([call.args[1] for call in device._transaction.call_args_list], [0x15, 0x14])

    def test_save_guard_rejects_unverified_other_loop_and_non_pid_changes(self):
        latest = bytearray(UPPER); latest[56:62] = encode_pid(350, 10, 0)
        with self.assertRaises(ValueError): check_settings_save(UPPER, latest, {})
        verified = {'speed': encode_pid(350, 10, 0)}
        self.assertEqual(check_settings_save(UPPER, latest, verified), latest)
        latest[2] ^= 1
        with self.assertRaises(ValueError): check_settings_save(UPPER, latest, verified)


class FakeVendor(DemoDevice):
    def __init__(self, connection, logger):
        super().__init__(connection, logger)
        self.settings_raw = UPPER
        self.persisted = []
        self.ack_targets = {}
        self.write_error = None

    def read_parameter(self, param):
        return parameter_from_settings(self.settings_raw, param)

    def write_parameter(self, param, raw6, persistent=False):
        if persistent:
            self.persisted.append(self.settings_raw)
        else:
            self.ack_targets[param] = raw6
            if self.write_error:
                raise self.write_error

    def save_settings(self, raw):
        self.persisted.append(raw)
        self.settings_raw = raw


class VendorServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        config = {'hardware': {'baudrate':115200, 'max_speed_deg_s':18, 'motors':[
            {'name':'upper','device':'/not/a/serial/device','id':1,'sign':1,'ratio':1.,'offset_deg':0.,'limits_deg':[0,180]}]}}
        self.tuner = Tuner(config, {'runtime_dir':Path(self.tmp.name)}, mode='real',
                           device_factory=FakeVendor, start_worker=False)
        self.tuner.connect()
        self.tuner._stop_and_confirm = Mock()
        self.addCleanup(self.tuner.close)

    def expected(self):
        state = self.tuner.state()
        return {name: state['ram_ack_values'].get(name, {k:state['params'][name][k] for k in ('kp','ki','kd')})
                for name in ('angle','speed')}

    def test_backup_and_explicit_whole_settings_save(self):
        t = self.tuner
        self.assertEqual(bytes.fromhex(t.export()['settings_backup_hex']), UPPER)
        self.assertEqual(t.state()['save_scope'], 'all_settings')
        t.write('speed', {'kp':350,'ki':10,'kd':0})
        with self.assertRaises(ValueError): t.save('speed', 'SAVE_ROM')
        t.save('speed', 'SAVE_ALL_ROM', self.expected())
        saved = t._device.persisted[0]
        self.assertEqual(saved[:56]+saved[62:], UPPER[:56]+UPPER[62:])
        self.assertEqual(saved[56:62], encode_pid(350, 10, 0))

    def test_external_config_change_prevents_save(self):
        t = self.tuner
        t.write('speed', {'kp':350,'ki':10,'kd':0})
        raw = bytearray(t._device.settings_raw); raw[0] ^= 1
        t._device.settings_raw = bytes(raw)
        with self.assertRaises(ValueError): t.save('speed', 'SAVE_ALL_ROM', self.expected())
        self.assertFalse(t._device.persisted)

    def test_external_other_pid_change_prevents_save(self):
        t = self.tuner
        t.write('speed', {'kp':350,'ki':10,'kd':0})
        raw = bytearray(t._device.settings_raw); raw[50:56] = encode_pid(51, 0, 0)
        t._device.settings_raw = bytes(raw)
        with self.assertRaises(ValueError): t.save('speed', 'SAVE_ALL_ROM', self.expected())
        self.assertFalse(t._device.persisted)

    def test_ack_success_with_old_group_readback_is_not_ram_failure(self):
        t = self.tuner
        t.write('speed', {'kp':400,'ki':10,'kd':0})
        self.assertEqual(t.pid_status['speed'], 'ram_acknowledged')
        self.assertEqual(t.params['speed']['kp'], 300)
        self.assertEqual(t.state()['ram_ack_values']['speed']['kp'], 400)
        self.assertFalse(t.write_attempts['speed']['runtime_readback_verified'])
        self.assertTrue(t.write_attempts['speed']['ack_ok'])
        self.assertEqual(t.write_attempts['speed']['readback_raw_hex'], encode_pid(300,10,0).hex(' '))
        t.read()
        self.assertEqual(t.pid_status['speed'], 'ram_acknowledged')
        self.assertEqual(t.state()['ram_ack_values']['speed']['kp'], 400)

    def test_ack_success_survives_group_readback_timeout_without_retry(self):
        t = self.tuner
        with patch.object(t._device, 'read_parameter', side_effect=TimeoutError('settings readback timeout')), \
                patch.object(t._device, 'write_parameter', wraps=t._device.write_parameter) as write:
            t.write('speed', {'kp':400, 'ki':10, 'kd':0})
        self.assertEqual(write.call_count, 1)
        self.assertEqual(t.pid_status['speed'], 'ram_acknowledged')
        self.assertEqual(t.state()['ram_ack_values']['speed'], {'kp':400, 'ki':10, 'kd':0})
        self.assertEqual(t.params['speed']['kp'], 300)
        attempt = t.write_attempts['speed']
        self.assertTrue(attempt['ack_ok'])
        self.assertFalse(attempt['runtime_readback_verified'])
        self.assertIn('settings readback timeout', attempt['warning'])
        self.assertNotIn('error', attempt)
        self.assertEqual(t._device.ack_targets, {0x97:encode_pid(400,10,0)})
        t.read()
        self.assertEqual(t.state()['ram_ack_values']['speed']['kp'], 400)

    def test_second_save_uses_previous_success_as_reference_and_preserves_original_backup(self):
        t = self.tuner
        for kp in (350, 400):
            t.write('speed', {'kp':kp, 'ki':10, 'kd':0})
            t.save('speed', 'SAVE_ALL_ROM', self.expected())
        self.assertEqual(len(t._device.persisted), 2)
        for saved, kp in zip(t._device.persisted, (350, 400)):
            self.assertEqual(saved[56:62], encode_pid(kp,10,0))
            self.assertEqual(saved[:56]+saved[62:], UPPER[:56]+UPPER[62:])
        self.assertEqual(bytes.fromhex(t.export()['settings_backup_hex']), UPPER)
        self.assertEqual(t.baseline['speed']['kp'], 300)

    def assert_save_failure_blocks_actions(self, failure):
        t = self.tuner
        t.write('speed', {'kp':400, 'ki':10, 'kd':0})
        with patch.object(t._device, 'save_settings', side_effect=failure) as save:
            with self.assertRaises(type(failure)):
                t.save('speed', 'SAVE_ALL_ROM', self.expected())
        self.assertEqual(save.call_count, 1)
        self.assertEqual(t.pid_status['speed'], 'rom_unverified')
        t.read()
        self.assertEqual(t.pid_status['speed'], 'rom_unverified')
        for action in (lambda:t.enable('ENABLE_MOTOR'), lambda:t.move(170,5),
                       lambda:t.save('speed','SAVE_ALL_ROM',self.expected())):
            with self.assertRaisesRegex(ValueError, '未确认'):
                action()
        audit = json.loads((t.audit_dir/(t.session_id+'.json')).read_text())
        self.assertEqual(audit['pid_status']['speed'], 'rom_unverified')
        self.assertEqual(audit['ram_ack_values']['speed']['kp'], 400)
        self.assertTrue(any(e.get('command_may_have_been_sent') for e in audit['events']))

    def test_save_readback_mismatch_globally_blocks_actions(self):
        self.assert_save_failure_blocks_actions(ValueError('0x14 readback mismatch after save ACK'))

    def test_save_readback_timeout_globally_blocks_actions(self):
        self.assert_save_failure_blocks_actions(TimeoutError('0x14 readback timeout after save ACK'))

    def test_missing_ack_records_error_and_globally_blocks_actions(self):
        t = self.tuner
        t._device.write_error = TimeoutError('ACK missing')
        with self.assertRaises(TimeoutError): t.write('speed', {'kp':400,'ki':10,'kd':0})
        t.read()
        for f in (lambda:t.enable('ENABLE_MOTOR'),lambda:t.move(170,5),lambda:t.save('angle','SAVE_ALL_ROM',self.expected())):
            with self.assertRaises(ValueError): f()
        audit=json.loads((t.audit_dir/(t.session_id+'.json')).read_text())
        self.assertEqual(audit['write_attempts']['speed']['error'],'ACK missing')
        self.assertEqual(audit['write_attempts']['speed']['failed_phase'],'write_ack')
        self.assertNotIn('speed',audit['ram_ack_values'])

    def test_stale_save_confirmation_cannot_persist_changed_targets(self):
        t=self.tuner
        t.write('speed',{'kp':350,'ki':10,'kd':0})
        expected=self.expected()
        t.write('speed',{'kp':400,'ki':10,'kd':0})
        with self.assertRaises(ValueError):t.save('speed','SAVE_ALL_ROM',expected)
        self.assertFalse(t._device.persisted)

    def test_acknowledged_targets_from_both_loops_are_saved_with_other_bytes_intact(self):
        t=self.tuner
        t.write('angle',{'kp':55,'ki':0,'kd':0})
        t.write('speed',{'kp':400,'ki':10,'kd':0})
        t.save('speed','SAVE_ALL_ROM',self.expected())
        saved=t._device.persisted[0]
        self.assertEqual(saved[:50]+saved[62:],UPPER[:50]+UPPER[62:])
        self.assertEqual(saved[50:56],encode_pid(55,0,0))
        self.assertEqual(saved[56:62],encode_pid(400,10,0))
