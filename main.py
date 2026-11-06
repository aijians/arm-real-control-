#!/usr/bin/env python3
"""Fixed TXT inbox service. Start without command-line task parameters."""
from pathlib import Path
import signal
import sys

from control.config import load_config
from control.runtime import Inbox, Runtime, instance_lock

ROOT = Path(__file__).resolve().parent


def main():
    if len(sys.argv) != 1:
        print("启动入口不接收任务参数；请将自然语言TXT上传到配置的task_file。", file=sys.stderr)
        return 2
    vision = None
    service = None

    def stop(signum, frame):
        if service is not None:
            service.stopping.set()
        # Existing task/motor handlers stop both axes before propagating or
        # returning. Idle shutdown does not open UARTs or move the arm.
        raise KeyboardInterrupt()

    previous_signals = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    for signum in previous_signals:
        signal.signal(signum, stop)
    try:
        config, paths = load_config(ROOT)
        with instance_lock(paths["runtime_dir"]):
            from control.language import ResidentModel
            from control.planner import Planner
            from control.vision import Vision
            from control.motors import MotorPair
            from control.tasks import TaskRunner
            model = ResidentModel(paths["model_package"])
            planner = Planner(ROOT, config=config)
            vision = Vision(ROOT, {**config["vision"], "video_dir": str(paths["video_dir"])})

            def execute(task, task_id):
                result = {"status": "failed"}
                try:
                    print(f"[task {task_id}] camera starting", flush=True)
                    vision.start(task_id)
                    print(f"[task {task_id}] camera ready; executing", flush=True)
                    with MotorPair(config["hardware"]) as motors:
                        result = TaskRunner(planner, config, motors, vision).run(task)
                except KeyboardInterrupt:
                    result.update(status="failed", reason="task_interrupted")
                except Exception as exc:
                    result.update(status="failed", reason=str(exc), error_type=type(exc).__name__)
                finally:
                    result["video"] = vision.close()
                if result["video"]["status"] == "failed" and result["status"] == "completed":
                    result.update(status="failed", reason="video_recording_failed")
                return result

            # Initialization must succeed before any TXT is claimed.
            service = Runtime(Inbox(paths["task_file"], paths["result_file"]),
                              model.infer, execute, config["poll_interval_s"])
            print("Runtime ready: " + str(paths["task_file"]), flush=True)
            service.run()
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(type(exc).__name__ + ": " + str(exc), file=sys.stderr, flush=True)
        return 2
    finally:
        if vision is not None:
            vision.close()
        for signum, handler in previous_signals.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
