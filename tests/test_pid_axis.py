"""Joint selection checks; fake motors and HTTP server, no hardware access."""
from contextlib import nullcontext, redirect_stderr, redirect_stdout
import copy
import csv
import io
import tempfile
import unittest
from unittest.mock import patch

from control.protocol import parse_frame
from pid_tuner.device import DemoDevice
from pid_tuner.server import main
from pid_tuner.service import Tuner, selected_motor


def configuration():
    # Lower deliberately comes first: selection must use names, not list order.
    return {'hardware': {'baudrate': 115200, 'max_speed_deg_s': 18., 'motors': [
        {'name': 'lower', 'device': '/virtual/lower', 'id': 2,
         'sign': -1, 'ratio': 1.5, 'offset_deg': 150., 'limits_deg': [0., 90.]},
        {'name': 'upper', 'device': '/never/open/upper', 'id': 1,
         'sign': 1, 'ratio': 1., 'offset_deg': -1.75, 'limits_deg': [2., 180.]},
    ]}}


class AxisTests(unittest.TestCase):
    def test_lower_uses_its_port_mapping_limits_and_packet_address(self):
        created = []
        def factory(connection, logger):
            created.append(copy.deepcopy(connection))
            return DemoDevice(connection, logger)
        with tempfile.TemporaryDirectory() as runtime:
            tuner = Tuner(configuration(), {'runtime_dir': runtime}, axis='lower',
                          device_factory=factory, start_worker=False)
            try:
                self.assertEqual(created, [])
                tuner.connect()
                self.assertEqual(len(created), 1)
                self.assertEqual(created[0]['device'], '/virtual/lower')
                self.assertEqual(created[0]['motor_id'], 2)
                self.assertEqual(tuner.limits['max_deg'], 90.)
                row = tuner.sample()
                self.assertEqual(row['angle_deg'], 45.)
                self.assertEqual(row['motor_angle_deg'], 82.5)
                tuner.move(47., 2.)
                tx = [log['message'] for log in tuner.logs if log['message'].startswith('TX ')]
                moves = [parse_frame(bytes.fromhex(message[3:])) for message in tx
                         if bytes.fromhex(message[3:])[1] == 0xA6]
                self.assertEqual(len(moves), 1)
                self.assertEqual(moves[0].motor_id, 2)
                self.assertEqual(moves[0].payload[0], 1)
                self.assertEqual(int.from_bytes(moves[0].payload[1:4], 'little'), 7950)
                self.assertEqual(int.from_bytes(moves[0].payload[4:8], 'little'), 300)
                with self.assertRaisesRegex(ValueError, '小臂配置限位'):
                    tuner.move(91., 2.)
                self.assertEqual(tuner.state()['connection']['axis_label'], '小臂')
                self.assertEqual(tuner.export()['connection']['axis'], 'lower')
                rows = list(csv.DictReader(io.StringIO(tuner.csv().lstrip('\ufeff'))))
                self.assertTrue(rows)
                self.assertTrue(all(row['axis'] == 'lower' and row['motor_id'] == '2' for row in rows))
            finally:
                tuner.close()

    def test_default_upper_is_selected_by_name(self):
        with tempfile.TemporaryDirectory() as runtime:
            tuner = Tuner(configuration(), {'runtime_dir': runtime}, start_worker=False)
            try:
                self.assertEqual(tuner.connection['axis'], 'upper')
                self.assertEqual(tuner.connection['device'], '/never/open/upper')
            finally:
                tuner.close()

    def test_unknown_missing_and_duplicate_axes_fail_before_device_creation(self):
        for axis, motors in (('other', configuration()['hardware']['motors']),
                             ('lower', []),
                             ('lower', configuration()['hardware']['motors'] * 2)):
            config = configuration()
            config['hardware']['motors'] = motors
            with patch('pid_tuner.service.SerialDevice') as factory:
                with self.assertRaises(ValueError):
                    Tuner(config, {'runtime_dir': '/unused'}, axis=axis, mode='real')
                factory.assert_not_called()

    def test_cli_lower_overrides_and_sal_isolation_use_only_selected_axis(self):
        config = configuration()
        upper = copy.deepcopy(selected_motor(config, 'upper'))
        argv = ['pid_tuner', '--axis', 'lower', '--real', '--pause-sal',
                '--device', '/virtual/override', '--motor-id', '7']
        with patch('sys.argv', argv), patch('pid_tuner.server.load_config', return_value=(config, {})), \
             patch('pid_tuner.server.Tuner') as tuner, \
             patch('pid_tuner.server.TunerServer') as server, \
             patch('pid_tuner.server.isolate_sal', return_value=nullcontext()) as isolate, \
             redirect_stdout(io.StringIO()):
            server.return_value.serve_forever.side_effect = KeyboardInterrupt
            main()
            self.assertEqual(selected_motor(config, 'upper'), upper)
            self.assertEqual(selected_motor(config, 'lower')['id'], 7)
            self.assertEqual(tuner.call_args.kwargs, {'mode': 'real', 'axis': 'lower'})
            isolate.assert_called_once_with(['/virtual/override'])
            tuner.return_value.connect.assert_not_called()
            tuner.return_value.enable.assert_not_called()

    def test_invalid_cli_axis_does_not_load_config_or_create_tuner(self):
        with patch('sys.argv', ['pid_tuner', '--axis', 'other']), \
             patch('pid_tuner.server.load_config') as load, \
             patch('pid_tuner.server.Tuner') as tuner, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main()
            load.assert_not_called()
            tuner.assert_not_called()
