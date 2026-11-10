"""Three slider tasks composed from free/continuous-push/withdraw motions."""
import math
import time

from .targets import normalize_task, TargetResolver
from .config import push_correction_settings


class TaskError(RuntimeError):
    def __init__(self, reason, details=None):
        super().__init__(reason)
        self.details = details or {}


class TaskRunner:
    def __init__(self, planner, config, motors=None, vision=None):
        self.planner, self.config = planner, config
        self.motors, self.vision = motors, vision
        self.tolerance = float(config["verification"]["tolerance_mm"])
        self.correction_settings = push_correction_settings(config)
        self.targets = None
        self.contact = None
        self.withdraw_extension_mm = 0.
        self.tool_direction = "+x"
        self.return_q_deg = self.config.get("planning", {}).get("return_q_deg", [170., 4.])
        if (not isinstance(self.return_q_deg, (list, tuple)) or len(self.return_q_deg) != 2
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       for value in self.return_q_deg)):
            raise ValueError("planning.return_q_deg must contain two finite model angles")
        self.return_q_deg = list(self.return_q_deg)

    def legs(self, task, positions, slider_limits_x_mm):
        task = normalize_task(task)
        if slider_limits_x_mm is None:
            raise TaskError("slider_limits_unavailable")
        try:
            self.targets = TargetResolver({"sliders": self.config["sliders"],
                "feedback": {"slider_position_tolerance_mm": self.tolerance}}, slider_limits_x_mm)
        except (KeyError, TypeError, ValueError) as exc:
            raise TaskError("invalid_observed_slider_limits", {"error": str(exc)}) from exc
        color = task["color"]
        start = float(positions[color])
        lo, hi = self.targets.bounds(color)
        if not lo - self.tolerance <= start <= hi + self.tolerance:
            raise TaskError("slider_start_outside_observed_limits", {
                "color": color, "measured_x_mm": start,
                "x_limits_mm": [lo, hi], "tolerance_mm": self.tolerance,
                "outside_by_mm": max(lo - start, start - hi, 0.)})
        if task["intent"] == "push_distance":
            target = self.targets.distance(color, task["direction"], task["distance_mm"], positions)
            return [target.target_x_mm], {"requested_x_mm": target.requested_x_mm,
                                          "clipped": target.clipped}
        endpoint = self.targets.endpoint(color, positions).target_x_mm
        if task["intent"] == "push_to_end":
            return [endpoint], {}
        if abs(endpoint - start) <= self.tolerance:
            endpoint = lo if abs(endpoint - hi) <= self.tolerance else hi
        return_x = min(hi, max(lo, start))
        return [x for _ in range(task["round_trips"]) for x in (endpoint, return_x)], {
            "return_x_mm": start, "return_target_x_mm": return_x}

    def _snapshot(self, after=None):
        observed = self.vision.read(after_monotonic_s=time.monotonic() if after is None else after)
        joints = self.motors.read_state()
        return {**observed, **joints}

    def _move(self, phase, record):
        name = phase.get("phase", "unknown")
        print("[motion] start " + name, flush=True)
        try:
            motion = self.motors.move(phase, health_check=self.vision.check_health)
            record.setdefault("motions", []).append(motion)
            # Re-plan from measured angles, never the ideal endpoint.
            q = self.motors.read_state()["q_rad"]
        except Exception as exc:
            print("[motion] failed " + name + ": " + str(exc), flush=True)
            # Retain the pre-stop capture even when push recovery verifies
            # the slider position and does not propagate this exception.
            record.setdefault("motions", []).append({
                "status": "failed", "phase": phase.get("phase"),
                "reason": str(exc), "error_type": type(exc).__name__,
                "details": getattr(exc, "details", {}),
                "halt_result": getattr(exc, "halt_result", None)})
            raise
        print("[motion] completed " + name, flush=True)
        return q

    def _verify(self, color, goal, after):
        samples = []
        state = None
        for i in range(int(self.config["verification"]["samples"])):
            if i:
                time.sleep(float(self.config["verification"]["interval_s"]))
            state = self.vision.read(after_monotonic_s=after)
            error = float(state["sliders_x_mm"][color]) - goal
            samples.append({"error_mm": error, "stamp_monotonic_s": state["stamp_monotonic_s"]})
            after = state["stamp_monotonic_s"]
        return {"passed": bool(samples) and all(abs(s["error_mm"]) <= self.tolerance for s in samples),
                "samples": samples, "last_observation": state}

    def _withdraw(self, state, record):
        if self.contact is not None:
            c, d = self.contact
            options = ({"from_reference": True, "reference_extension_mm": self.withdraw_extension_mm}
                       if self.withdraw_extension_mm else {})
            phase = self.planner.withdraw(state["q_rad"], c, d, state["sliders_x_mm"], **options)
            state["q_rad"] = self._move(phase, record)
            self.contact = None
            self.withdraw_extension_mm = 0.
            # Withdrawal can also disturb a slider; use a new image before the
            # next free-space route rather than its pre-withdrawal positions.
            state.update(self.vision.read(after_monotonic_s=time.monotonic()))

    def _stopped_verification(self, color, goal, record):
        stop = self.motors.halt()
        record["stop"] = stop
        if not stop.get("stopped_confirmed"):
            raise TaskError("stop_not_confirmed")
        after = time.monotonic()
        record["verification_after_monotonic_s"] = after
        verification = self._verify(color, goal, after)
        record["verification"] = verification
        return verification

    def _stopped_push(self, phase, color, goal, record):
        error = None
        try:
            self._move(phase, record)
        except Exception as exc:
            error = str(exc)
            record["push_motor_error"] = error
        # Even a normally completed trajectory must be explicitly stopped
        # before obtaining the images used for task acceptance.
        return self._stopped_verification(color, goal, record), error

    def _push_with_corrections(self, color, goal, direction, state, record):
        sign = 1 if direction == "+x" else -1
        before = float(state["sliders_x_mm"][color])
        phase = self.planner.push(state["q_rad"], color, goal,
                                  state["sliders_x_mm"], self.targets.bounds(color))
        verification, error = self._stopped_push(phase, color, goal, record)
        corrections = record["corrections"] = []
        used = 0.
        settings = self.correction_settings
        while True:
            state.update(verification["last_observation"])
            actual = float(state["sliders_x_mm"][color])
            if verification["passed"]:
                # Preserve the established recovery: a stopped, visually
                # verified goal may succeed despite the arm's motion error.
                state["q_rad"] = self.motors.read_state()["q_rad"]
                self.withdraw_extension_mm = used
                return state
            if error is not None:
                raise TaskError(error)
            remaining = sign * (goal - actual)
            if remaining <= 0:
                raise TaskError("slider_overshot_target")
            if remaining <= self.tolerance:
                raise TaskError("slider_not_stably_at_target")
            progress = sign * (actual - before)
            if progress < settings["min_progress_mm"]:
                raise TaskError("push_no_progress", {"progress_mm": progress})
            if (len(corrections) >= settings["max_steps"]
                    or used >= settings["max_distance_mm"] - 1e-9):
                raise TaskError("push_correction_budget_exhausted", {
                    "steps": len(corrections), "extension_mm": used,
                    "remaining_mm": remaining})
            # Start from fresh measured joints, not the nominal end of the
            # previous phase or the visually observed slider's X coordinate.
            state["q_rad"] = self.motors.read_state()["q_rad"]
            distance = min(settings["step_mm"], settings["max_distance_mm"] - used, remaining)
            phase = self.planner.correction(state["q_rad"], color, goal, direction,
                state["sliders_x_mm"], self.targets.bounds(color), distance)
            used += distance
            correction = {"step": len(corrections) + 1, "extension_mm": distance,
                          "cumulative_extension_mm": used, "task_target_x_mm": goal,
                          "reference_start_mm": phase["reference_start_mm"],
                          "reference_target_mm": phase["reference_target_mm"],
                          "before_x_mm": actual}
            corrections.append(correction)
            before = actual
            verification, error = self._stopped_push(phase, color, goal, correction)
            correction["after_x_mm"] = float(verification["last_observation"]["sliders_x_mm"][color])
            correction["progress_mm"] = sign * (correction["after_x_mm"] - before)

    def _run_leg(self, color, goal):
        record = {"color": color, "target_x_mm": goal, "attempts": [], "status": "failed"}
        limit = int(self.config["retry_limit"])
        for number in range(limit + 1):
            attempt = {"attempt": number}
            record["attempts"].append(attempt)
            push_started = False
            try:
                if number:
                    stop = self.motors.halt()
                    attempt["stop_before_retry"] = stop
                    if not stop.get("stopped_confirmed"):
                        raise TaskError("stop_not_confirmed")
                state = self._snapshot()
                attempt["start_state"] = state
                self._withdraw(state, attempt)
                start = state["sliders_x_mm"][color]
                if abs(goal - start) <= self.tolerance:
                    verification = self._stopped_verification(color, goal, attempt)
                    if not verification["passed"]:
                        raise TaskError("slider_not_at_target")
                    state.update(verification["last_observation"])
                    attempt.update(status="completed", no_motion=True)
                    record["status"] = "completed"
                    record.pop("reason", None)
                    return record, state
                direction = "+x" if goal > start else "-x"
                self.tool_direction = direction
                pre = self.planner.precontact(color, direction, state["sliders_x_mm"], state["q_rad"])
                state["q_rad"] = self._move(self.planner.free(state["q_rad"], pre, state["sliders_x_mm"], direction=direction), attempt)
                # The single straight push includes crossing the precontact gap.
                # Remember its active face even when that motion fails partway.
                self.contact = color, direction
                self.withdraw_extension_mm = 0.
                push_started = True
                state = self._push_with_corrections(color, goal, direction, state, attempt)
                self._withdraw(state, attempt)
                attempt["status"] = "completed"
                record["status"] = "completed"
                record.pop("reason", None)
                return record, state
            except KeyboardInterrupt:
                attempt.update(status="interrupted", stop=self.motors.halt())
                record["reason"] = "task_interrupted"
                return record, None
            except Exception as exc:
                attempt.update(status="failed", reason=str(exc),
                               error_type=type(exc).__name__,
                               details=getattr(exc, "details", {}), stop=self.motors.halt())
                record["reason"] = str(exc)
                if getattr(exc, "details", {}).get("controller") == "free_timed":
                    return record, None
                # The current move already used its finite speed-recovery
                # budget. Replaying a whole leg must not restart that budget.
                if (getattr(exc, "details", {}).get("recovery_exhausted")
                        or getattr(exc, "details", {}).get("speed_recovery_attempted")):
                    return record, None
                # Retrying free entry can recover a stale observation or a
                # blocked entry. Never replay a complete push after the bounded
                # corrections, a push stall, or a failed stopped verification.
                if push_started:
                    return record, None
        return record, None

    def run(self, task):
        task = normalize_task(task)
        result = {"mode": "hardware_execution", "status": "failed", "task": task,
                  "segments": [], "completed_round_trips": 0,
                  "planned_round_trips": task.get("round_trips", 0)}
        try:
            result["environment_recovery"] = []
            initial = None
            for number in range(int(self.config["retry_limit"]) + 1):
                try:
                    stop = self.motors.halt()
                    if not stop.get("stopped_confirmed"):
                        raise TaskError("initial_stop_not_confirmed")
                    initial = self._snapshot()
                    break
                except Exception as exc:
                    result["environment_recovery"].append({"attempt": number, "reason": str(exc)})
            if initial is None:
                raise TaskError("fresh_stopped_environment_unavailable")
            # Preserve the observation even if task target validation fails.
            result["initial_state"] = initial
            # Keep the measured limits and goals fixed for this task; later
            # frames still provide live slider positions for verification.
            goals, extra = self.legs(task, initial["sliders_x_mm"], initial.get("slider_limits_x_mm"))
            result.update(extra)
            result["slider_limits_x_mm"] = {c: list(self.targets.bounds(c)) for c in self.config["sliders"]}
            for index, goal in enumerate(goals):
                leg, state = self._run_leg(task["color"], goal)
                result["segments"].append(leg)
                if leg["status"] != "completed":
                    result["reason"] = leg.get("reason", "segment_failed")
                    return result
                if task["intent"] == "push_round_trip" and index % 2:
                    result["completed_round_trips"] += 1
            result["return_motion"] = {"target_q_deg": list(self.return_q_deg)}
            return_q_rad = [math.radians(value) for value in self.return_q_deg]
            print("[motion] return target: upper=%.2f deg, lower=%.2f deg" % tuple(self.return_q_deg), flush=True)
            self._move(self.planner.free(state["q_rad"], return_q_rad, state["sliders_x_mm"], direction=self.tool_direction), result["return_motion"])
            result["status"] = "completed"
        except KeyboardInterrupt:
            result.update(reason="task_interrupted", stop=self.motors.halt())
        except Exception as exc:
            result.update(reason=str(exc), error_type=type(exc).__name__,
                          details=getattr(exc, "details", {}), stop=self.motors.halt())
        return result
