"""Serial TXT inbox: claim once, save a terminal result, then delete the TXT."""
from contextlib import contextmanager
import fcntl
import glob
from pathlib import Path
import threading
import uuid

from .config import read_json, save_json, sync_directory


@contextmanager
def instance_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "task.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("runtime_already_running") from exc
        yield


class Inbox:
    def __init__(self, task_file, result_file):
        self.task_file, self.result_file = Path(task_file), Path(result_file)
        self.task_file.parent.mkdir(parents=True, exist_ok=True)
        self.prefix, self.suffix = "." + self.task_file.name + ".", ".processing"

    def recover(self):
        """Finalize an interrupted claim without repeating inference/motion."""
        pattern = glob.escape(self.prefix) + "*" + self.suffix
        for active in sorted(self.task_file.parent.glob(pattern)):
            task_id = active.name[len(self.prefix):-len(self.suffix)]
            if len(task_id) != 32 or any(c not in "0123456789abcdef" for c in task_id):
                continue
            try:
                previous = read_json(self.result_file)
            except (OSError, ValueError):
                previous = {}
            if not (isinstance(previous, dict) and previous.get("task_id") == task_id
                    and previous.get("status") in ("completed", "failed", "rejected")):
                save_json({"status": "failed", "reason": "runtime_restarted_during_task", "task_id": task_id}, self.result_file)
            active.unlink()
            sync_directory(active.parent)

    def claim(self):
        if self.task_file.exists() and not self.task_file.is_file():
            raise ValueError("task_file_is_not_a_file")
        task_id = uuid.uuid4().hex
        active = self.task_file.with_name(self.prefix + task_id + self.suffix)
        try:
            self.task_file.rename(active)
        except FileNotFoundError:
            return None
        sync_directory(active.parent)
        return task_id, active

    def finish(self, claim, result):
        task_id, active = claim
        # If saving fails, leave the claim in place and terminate the service;
        # a restart records interruption, never repeats motor commands.
        save_json({**result, "task_id": task_id}, self.result_file)
        active.unlink()
        sync_directory(active.parent)


class Runtime:
    def __init__(self, inbox, infer, execute, poll_interval_s=1.0):
        self.inbox, self.infer, self.execute = inbox, infer, execute
        self.poll_interval_s = poll_interval_s
        self.stopping = threading.Event()

    def process_one(self):
        claim = self.inbox.claim()
        if claim is None:
            return False
        task_id = claim[0]
        print(f"[task {task_id}] received", flush=True)
        result = None
        interrupted = False
        try:
            with claim[1].open("rb") as stream:
                data = stream.read(8193)
            if len(data) > 8192:
                raise ValueError("task_text_exceeds_8192_bytes")
            text = data.decode("utf-8-sig").strip()
            if not text:
                raise ValueError("empty_task_text")
            language = self.infer(text)
            print(f"[task {task_id}] parsed: {language['task']}", flush=True)
            result = self.execute(language["task"], claim[0])
            if not isinstance(result, dict) or result.get("status") not in ("completed", "failed", "rejected"):
                raise ValueError("invalid_task_result")
            result = {**result, "language_input": language}
        except KeyboardInterrupt:
            result = {"status": "failed", "reason": "runtime_stopped_during_task"}
            interrupted = True
        except Exception as exc:
            result = {"status": "failed", "reason": str(exc), "error_type": type(exc).__name__}
        self.inbox.finish(claim, result)
        print(f"[task {task_id}] {result['status']}: {result.get('reason', 'done')}; result: {self.inbox.result_file}", flush=True)
        if interrupted:
            raise KeyboardInterrupt()
        return True

    def run(self):
        self.inbox.recover()
        while not self.stopping.is_set():
            if not self.process_one():
                self.stopping.wait(self.poll_interval_s)
