"""Failed connections keep evidence; diagnostics never imply write permission."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pid_tuner.device import DemoDevice, ReplyTimeout
from pid_tuner.service import Tuner


class ProbeDevice(DemoDevice):
    def __init__(self, connection, logger):
        super().__init__(connection, logger)
        self.read_error = None
        self.open_error = None
        self.probe_error = None
        self.calls = []

    def open(self):
        self.calls.append('open')
        if self.open_error:
            raise self.open_error

    def close(self):
        self.calls.append('close')

    def read_parameter(self, param):
        self.calls.append(('read', param))
        if self.read_error:
            raise self.read_error
        return super().read_parameter(param)

    def diagnose(self):
        self.calls.append('diagnose')
        if self.probe_error:
            raise self.probe_error
        return {'steps': [{'name': '单圈角度', 'command': '0x94', 'ok': True,
                           'decoded': {'motor_angle_deg': 169.11}},
                          {'name': '角度环PID', 'command': '0x40', 'param_id': '0x96',
                           'ok': False, 'error': 'silent parameter reply'}],
                'summary': ['状态有应答，PID查询无应答；需要进一步核对协议兼容性。']}

    def write_parameter(self, *args, **kwargs):
        raise AssertionError('A diagnostic must never write a parameter')

    def enable(self):
        raise AssertionError('A diagnostic must never enable the motor')

    def move(self, *args):
        raise AssertionError('A diagnostic must never issue motion')


class FailedSessionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(__file__).resolve().parents[1]
        self.config = json.loads((root/'config/robot.json').read_text())
        self.instances = []

    def tuner(self, read_error=None, open_error=None, probe_error=None):
        def factory(connection, logger):
            device = ProbeDevice(connection, logger)
            device.read_error, device.open_error, device.probe_error = read_error, open_error, probe_error
            self.instances.append(device)
            return device
        tuner = Tuner(self.config, {'runtime_dir': Path(self.directory.name)},
                      device_factory=factory, start_worker=False)
        self.addCleanup(tuner.close)
        return tuner

    def test_failed_parameter_read_is_probed_and_persisted_without_a_pid_backup(self):
        details = {'tx_hex': '3E 40 01 02 81 96 00 96', 'received_bytes': 0,
                   'raw_rx_hex': '', 'incomplete_hex': '', 'unmatched_frames': []}
        error = ReplyTimeout(1, 0x40, 0x96, details)
        tuner = self.tuner(read_error=error)
        with self.assertRaises(ReplyTimeout) as caught:
            tuner.connect()
        self.assertIs(caught.exception, error)
        self.assertEqual(tuner.baseline, {})
        self.assertFalse(tuner.connected)
        self.assertIsNone(tuner._device)
        self.assertIn('diagnose', self.instances[0].calls)
        report = tuner.export()
        self.assertEqual(report['diagnostics']['initial_failure']['details']['received_bytes'], 0)
        saved = json.loads((tuner.audit_dir/(tuner.session_id+'.json')).read_text())
        self.assertEqual(saved['diagnostics'], report['diagnostics'])
        self.assertIn('0x96', saved['last_error'])
        with self.assertRaises(ValueError):
            tuner.write('angle', {'kp': 50, 'ki': 0, 'kd': 0})

    def test_open_failure_and_interrupts_do_not_trigger_probes(self):
        for error, opening in ((OSError('port busy'), True),
                               (InterruptedError('stop requested'), False),
                               (KeyboardInterrupt(), False)):
            with self.subTest(error=type(error).__name__):
                tuner = self.tuner(open_error=error if opening else None,
                                   read_error=None if opening else error)
                with self.assertRaises(type(error)):
                    tuner.connect()
                self.assertNotIn('diagnose', self.instances[-1].calls)
                self.assertFalse(tuner.connected)

    def test_probe_failure_and_audit_failure_preserve_original_error(self):
        original = TimeoutError('original parameter timeout')
        tuner = self.tuner(read_error=original, probe_error=ValueError('probe failed'))
        with patch('pid_tuner.service.save_json', side_effect=OSError('disk full')):
            with self.assertRaises(TimeoutError) as caught:
                tuner.connect()
        self.assertIs(caught.exception, original)
        self.assertEqual(tuner.last_error, str(original))
        self.assertIn('close', self.instances[0].calls)

    def test_backup_failure_does_not_generate_additional_serial_probes(self):
        tuner = self.tuner()
        with patch('pid_tuner.service.save_json', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                tuner.connect()
        self.assertNotIn('diagnose', self.instances[0].calls)

    def test_standalone_probe_works_when_disconnected_and_never_establishes_a_backup(self):
        tuner = self.tuner()
        tuner.diagnose()
        self.assertEqual(self.instances[0].calls, ['open', 'diagnose', 'close'])
        self.assertFalse(tuner.connected)
        self.assertEqual(tuner.baseline, {})
        self.assertEqual(tuner.export()['diagnostics']['steps'][0]['command'], '0x94')
        self.assertTrue((tuner.audit_dir/(tuner.session_id+'.json')).exists())

    def test_connected_probe_preserves_backup_and_blocks_during_motion(self):
        tuner = self.tuner()
        tuner.connect()
        baseline = copy.deepcopy(tuner.baseline)
        tuner.diagnose()
        self.assertEqual(tuner.baseline, baseline)
        before = self.instances[0].calls.count('diagnose')
        tuner.motion['active'] = True
        with self.assertRaises(ValueError):
            tuner.diagnose()
        self.assertEqual(self.instances[0].calls.count('diagnose'), before)
        tuner.motion['active'] = False


if __name__ == '__main__':
    unittest.main()
