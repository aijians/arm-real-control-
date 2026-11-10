"""Small local HTTP server. No web framework, CDN or internet is required."""
import argparse
from contextlib import nullcontext
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import signal
import threading
from urllib.parse import parse_qs, urlsplit

from control.config import load_config
from .service import AXIS_LABELS, Tuner, selected_motor
from .exclusive import isolate_sal

ROOT = Path(__file__).resolve().parent.parent
WEB = Path(__file__).resolve().parent / 'web'


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('JSON字段重复：'+key)
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError('不允许非有限数字：'+value)
    return json.loads(data, object_pairs_hook=pairs, parse_constant=nonfinite)


class TunerServer(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address, tuner, token):
        self.tuner, self.token = tuner, token
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    server_version = 'PIDWorkbench/1.0'

    def log_message(self, fmt, *args):
        # Do not put authentication tokens or full request URLs in logs.
        pass

    def _reply(self, status, body, mime='application/json; charset=utf-8', filename=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode('utf-8')
        elif isinstance(body, str):
            body = body.encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
        if filename:
            self.send_header('Content-Disposition', 'attachment; filename="'+filename+'"')
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _authorized(self):
        expected = 'Bearer '+self.server.token
        if not hmac.compare_digest(self.headers.get('Authorization', '').encode('utf-8'), expected.encode('utf-8')):
            self._reply(401, {'ok': False, 'error': '访问口令无效，请使用启动终端打印的完整链接打开'})
            return False
        origin = self.headers.get('Origin')
        if origin and origin not in ('http://'+self.headers.get('Host', ''), 'https://'+self.headers.get('Host', '')):
            self._reply(403, {'ok': False, 'error': '不接受跨站操作'})
            return False
        return True

    def do_GET(self):
        path = urlsplit(self.path).path
        assets = {'/': ('index.html', 'text/html; charset=utf-8'),
                  '/style.css': ('style.css', 'text/css; charset=utf-8'),
                  '/app.js': ('app.js', 'application/javascript; charset=utf-8')}
        if path in assets:
            name, mime = assets[path]
            try:
                self._reply(200, (WEB/name).read_bytes(), mime)
            except OSError:
                self._reply(503, {'ok': False, 'error': '界面资源尚未就绪'})
            return
        if not path.startswith('/api/'):
            self._reply(404, {'ok': False, 'error': '路径不存在'})
            return
        if not self._authorized():
            return
        try:
            if path == '/api/state':
                since = int(parse_qs(urlsplit(self.path).query).get('since', ['0'])[0])
                self._reply(200, self.server.tuner.state(max(0, since)))
            elif path == '/api/export.json':
                self._reply(200, self.server.tuner.export(), filename='pid-session.json')
            elif path == '/api/export.csv':
                self._reply(200, self.server.tuner.csv(), 'text/csv; charset=utf-8', 'pid-telemetry.csv')
            else:
                self._reply(404, {'ok': False, 'error': '路径不存在'})
        except (ValueError, TypeError) as exc:
            self._reply(400, {'ok': False, 'error': str(exc)})

    def do_POST(self):
        if not self._authorized():
            return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 16384:
                raise ValueError('请求体长度必须在1..16384字节内')
            if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                raise ValueError('请使用application/json')
            body = strict_json(self.rfile.read(size).decode('utf-8'))
            if not isinstance(body, dict):
                raise ValueError('请求必须是JSON对象')
            path = urlsplit(self.path).path
            tuner = self.server.tuner
            actions = {'/api/connect': (set(), lambda: tuner.connect()),
                       '/api/diagnose': (set(), lambda: tuner.diagnose()),
                       '/api/disconnect': (set(), lambda: tuner.disconnect()),
                       '/api/read': (set(), lambda: tuner.read()),
                       '/api/write': ({'loop', 'values'}, lambda: tuner.write(body['loop'], body['values'])),
                       '/api/save': ({'loop', 'confirm'} | ({'expected'} if 'expected' in body else set()), lambda: tuner.save(body['loop'], body['confirm'], body.get('expected'))),
                       '/api/restore': ({'loop'}, lambda: tuner.restore(body['loop'])),
                       '/api/enable': ({'confirm'}, lambda: tuner.enable(body['confirm'])),
                       '/api/move': ({'target_deg', 'speed_deg_s'}, lambda: tuner.move(body['target_deg'], body['speed_deg_s'])),
                       '/api/stop': (set(), lambda: tuner.stop())}
            if path not in actions:
                self._reply(404, {'ok': False, 'error': '路径不存在'})
                return
            keys, action = actions[path]
            if set(body) != keys:
                raise ValueError('请求字段不匹配：需要'+','.join(sorted(keys)))
            action()
            self._reply(200, {'ok': True})
        except (ValueError, TypeError, KeyError, UnicodeError) as exc:
            self.server.tuner.last_error = str(exc)
            self._reply(400, {'ok': False, 'error': str(exc)})
        except Exception as exc:
            self.server.tuner.last_error = str(exc)
            self.server.tuner.log('error', str(exc))
            self._reply(503, {'ok': False, 'error': str(exc)})


def main():
    parser = argparse.ArgumentParser(description='双关节PID调参台：默认演示，--real显式连接真实设备')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--real', action='store_true', help='允许通过界面连接所选关节')
    mode.add_argument('--demo', action='store_true', help='演示模式（默认，无硬件访问）')
    parser.add_argument('--host', default='127.0.0.1', help='局域网访问时使用0.0.0.0')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--axis', choices=tuple(AXIS_LABELS), default='upper',
                        help='选择关节：upper大臂（默认），lower小臂；切换需重启工具')
    parser.add_argument('--device', help='覆盖所选关节串口，如/dev/ttyUSB0')
    parser.add_argument('--motor-id', type=int, help='覆盖所选关节ID，1..32')
    parser.add_argument('--pause-sal', action='store_true', help='临时暂停板端sal串口扫描，退出时自动恢复（仅实机）')
    args = parser.parse_args()
    if args.pause_sal and not args.real:
        parser.error('--pause-sal只能与--real一起使用')
    config, paths = load_config(ROOT)
    try:
        motor = selected_motor(config, args.axis)
    except ValueError as exc:
        parser.error(str(exc))
    if args.device:
        motor['device'] = args.device
    if args.motor_id is not None:
        if not 1 <= args.motor_id <= 32:
            parser.error('--motor-id必须为1..32')
        motor['id'] = args.motor_id
    if not 0 <= args.port <= 65535:
        parser.error('--port必须为0..65535')
    token = secrets.token_urlsafe(32)
    tuner = Tuner(config, paths, mode='real' if args.real else 'demo', axis=args.axis)
    server = None
    isolation = isolate_sal([motor['device']]) if args.pause_sal else nullcontext()
    # Bind first: a port conflict must never leave the scanner suspended.
    try:
        server = TunerServer((args.host, args.port), tuner, token)
        host = args.host if args.host not in ('0.0.0.0', '::') else '<开发板IP>'
        print(AXIS_LABELS[args.axis]+f'PID调参台 [{args.axis}] {motor["device"]} ID{motor["id"]} / '+('真实设备模式，点击连接前不会打开串口' if args.real else '演示模式，无硬件访问'), flush=True)
        print(f'打开完整链接：http://{host}:{server.server_port}/#token={token}', flush=True)
        print('停止服务：Ctrl+C。实机模式请先退出主控制程序，串口不能共享。', flush=True)
        def stop(signum, frame):
            raise KeyboardInterrupt()
        old = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            with isolation:
                try:
                    server.serve_forever(poll_interval=0.2)
                finally:
                    tuner.close()  # release motor UART before resuming sal
        finally:
            for sig, handler in old.items():
                signal.signal(sig, handler)
    except KeyboardInterrupt:
        pass
    finally:
        if server:
            server.server_close()
        tuner.close()


if __name__ == '__main__':
    main()
