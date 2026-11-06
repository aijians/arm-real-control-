"""HTTP boundary checks without opening a network socket."""
from email.message import Message
import io
import json
from types import SimpleNamespace
import unittest

from pid_tuner.server import Handler, strict_json


class StubTuner:
    def __init__(self):
        self.calls = []
        self.last_error = None

    def __getattr__(self, name):
        def action(*args):
            self.calls.append((name, args))
            if name == 'state':
                return {'connected': False, 'readings': []}
            if name == 'export':
                return {'baseline': {'angle': {'kp': 50}}}
            if name == 'csv':
                return 't,angle_deg\n0,169.11\n'
        return action


class CaptureHandler(Handler):
    def __init__(self, path='/api/state', body=None, token='test-secret', origin=None):
        self.path = path
        self.server = SimpleNamespace(token='test-secret', tuner=StubTuner())
        self.headers = Message()
        self.headers['Host'] = '127.0.0.1:8765'
        if token is not None:
            self.headers['Authorization'] = 'Bearer '+token
        if origin:
            self.headers['Origin'] = origin
        raw = json.dumps(body).encode() if body is not None else b'{}'
        self.rfile = io.BytesIO(raw)
        self.headers['Content-Type'] = 'application/json'
        self.headers['Content-Length'] = str(len(raw))
        self.result = None

    def _reply(self, status, body, mime=None, filename=None):
        self.result = (status, body, mime, filename)


class HTTPTests(unittest.TestCase):
    def test_api_requires_token_and_unicode_token_is_safely_rejected(self):
        for token in (None, 'wrong', '非ASCII'):
            handler = CaptureHandler(token=token)
            handler.do_GET()
            self.assertEqual(handler.result[0], 401)
            self.assertEqual(handler.server.tuner.calls, [])

    def test_cross_origin_cannot_send_motor_commands(self):
        handler = CaptureHandler('/api/enable', {'confirm': 'ENABLE_MOTOR'}, origin='https://unrelated.invalid')
        handler.do_POST()
        self.assertEqual(handler.result[0], 403)
        self.assertEqual(handler.server.tuner.calls, [])

    def test_state_and_exports_are_authenticated(self):
        for path, method in (('/api/state?since=12', 'state'), ('/api/export.json', 'export'),
                             ('/api/export.csv', 'csv')):
            handler = CaptureHandler(path)
            handler.do_GET()
            self.assertEqual(handler.result[0], 200)
            self.assertEqual(handler.server.tuner.calls[0][0], method)
        self.assertEqual(CaptureHandler('/api/state').server.tuner.calls, [])

    def test_extra_persistent_flag_cannot_bypass_separate_rom_action(self):
        handler = CaptureHandler('/api/write', {'loop': 'angle', 'values': {'kp': 50, 'ki': 0, 'kd': 0},
                                               'persistent': True})
        handler.do_POST()
        self.assertEqual(handler.result[0], 400)
        self.assertEqual(handler.server.tuner.calls, [])

    def test_exact_post_schema_and_known_actions(self):
        handler = CaptureHandler('/api/write', {'loop': 'speed', 'values': {'kp': 100, 'ki': 30, 'kd': 0}})
        handler.do_POST()
        self.assertEqual(handler.result[0], 200)
        self.assertEqual(handler.server.tuner.calls, [('write', ('speed', {'kp': 100, 'ki': 30, 'kd': 0}))])
        handler = CaptureHandler('/api/reboot')
        handler.do_POST()
        self.assertEqual(handler.result[0], 404)
        self.assertEqual(handler.server.tuner.calls, [])

    def test_read_only_diagnostic_has_its_own_authenticated_empty_request(self):
        handler = CaptureHandler('/api/diagnose', {})
        handler.do_POST()
        self.assertEqual(handler.result[0], 200)
        self.assertEqual(handler.server.tuner.calls, [('diagnose', ())])

    def test_invalid_length_type_json_and_duplicate_fields_do_not_dispatch(self):
        for raw in (b'[]', b'{', b'{"loop":"angle","loop":"speed"}', b'{"x":NaN}'):
            handler = CaptureHandler('/api/write')
            handler.rfile = io.BytesIO(raw)
            handler.headers.replace_header('Content-Length', str(len(raw)))
            handler.do_POST()
            self.assertEqual(handler.result[0], 400)
            self.assertEqual(handler.server.tuner.calls, [])
        for name, value in (('Content-Length', '16385'), ('Content-Length', '-1'), ('Content-Type', 'text/plain')):
            handler = CaptureHandler('/api/stop')
            handler.headers.replace_header(name, value)
            handler.do_POST()
            self.assertEqual(handler.result[0], 400)
            self.assertEqual(handler.server.tuner.calls, [])

    def test_only_explicit_assets_are_served(self):
        handler = CaptureHandler('/../../config/robot.json', token=None)
        handler.do_GET()
        self.assertEqual(handler.result[0], 404)
        handler = CaptureHandler('/', token=None)
        handler.do_GET()
        self.assertEqual(handler.result[0], 200)
        self.assertIn(b'app.js', handler.result[1])

    def test_strict_json_preserves_integer_types(self):
        self.assertEqual(strict_json('{"kp":65535}'), {'kp': 65535})
        with self.assertRaises(ValueError):
            strict_json('{"kp":1,"kp":2}')


if __name__ == '__main__':
    unittest.main()
