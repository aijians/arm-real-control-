"""Explicit, temporary isolation of the board vendor's UART scanner.

Never stop initrun.service. Resume only processes paused by this context,
checking executable and process start time to avoid PID reuse. An independent
pipe guardian resumes them if this process is killed or its connection dies.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time

SAL_EXE = '/userdata/satellware/bin/sal'


def identity(pid):
    proc = Path('/proc') / str(pid)
    if os.readlink(proc/'exe') != SAL_EXE:
        raise RuntimeError('sal进程身份已变化')
    return (proc/'stat').read_text().rsplit(')', 1)[1].split()[19]


def stopped(pid):
    states = []
    for task in (Path('/proc')/str(pid)/'task').iterdir():
        try:
            states.append(next(x.split()[1] for x in (task/'status').read_text().splitlines() if x.startswith('State:')))
        except OSError:
            continue
    return bool(states) and all(s in ('T', 't') for s in states)


def holds_uart(pid, devices):
    identities = {os.stat(d).st_rdev for d in devices}
    for fd in (Path('/proc')/str(pid)/'fd').iterdir():
        try:
            s = fd.stat()
            if stat.S_ISCHR(s.st_mode) and s.st_rdev in identities:
                return True
        except OSError:
            pass
    return False


def resume(records):
    for pid, start in records:
        try:
            if identity(pid) == start:
                os.kill(pid, signal.SIGCONT)
        except (OSError, RuntimeError):
            pass


@contextmanager
def isolate_sal(devices):
    records = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            pid = int(proc.name)
            start = identity(pid)
            if not stopped(pid):
                records.append((pid, start))
        except (OSError, RuntimeError):
            continue
    # EOF is delivered even on SIGKILL. No timeout that could resume the UART
    # scanner in the middle of an active test while the tuner is still alive.
    guard_code = ('import json,sys; from pid_tuner.exclusive import resume; '
                  'records=json.loads(sys.argv[1]); sys.stdin.buffer.read(); resume(records)')
    guardian = subprocess.Popen([sys.executable, '-c', guard_code, json.dumps(records)],
                                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True)
    try:
        for pid, start in records:
            deadline = time.monotonic()+8
            while True:
                if identity(pid) != start:
                    raise RuntimeError('sal进程身份已变化')
                os.kill(pid, signal.SIGSTOP)
                for _ in range(50):
                    if stopped(pid):
                        break
                    time.sleep(.01)
                if stopped(pid) and not holds_uart(pid, devices):
                    break
                os.kill(pid, signal.SIGCONT)
                if time.monotonic() > deadline:
                    raise RuntimeError('sal持续持有电机串口，未能建立独占窗口')
                time.sleep(.1)
        print('串口隔离：已临时暂停sal PID '+(', '.join(str(p) for p, _ in records) or '无运行实例')+
              '；退出调参程序时自动恢复。', flush=True)
        yield
    finally:
        resume(records)
        guardian.stdin.close()
        guardian.wait(timeout=5)
