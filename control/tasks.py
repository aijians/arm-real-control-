"""Three slider tasks composed from free/continuous-push/withdraw motions."""
import copy
import math
import time

from .targets import normalize_task, TargetResolver
from .config import push_correction_settings


class TaskError(RuntimeError):
    def __init__(self, reason, details=None):
        super().__init__(reason)
        self.details = details or {}


def _finite_setting(value, name, minimum=0.):
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValueError("invalid " + name)
    return float(value)


class TaskRunner:
    _NO_RETRY = {"motion_halted", "stop_not_confirmed", "joint_feedback_out_of_limits",
                 "motion_goal_hold_hard_limit", "slider_overshot_target", "motion_already_running",
                 "verification_unreliable"}
    # 运动到位判定失败但机械臂已停在目标附近的错误；端点任务中可由模型推头到位窗口吸收。
    _ARRIVAL_ABSORBABLE = {"motion_goal_settle_timeout", "motion_goal_hold_timeout",
                           "motion_goal_hold_unstable", "motion_no_progress",
                           "motion_tracking_error", "motion_timeout"}

    def __init__(self, planner, config, motors=None, vision=None):
        self.planner, self.config = planner, config
        self.motors, self.vision = motors, vision
        verification = config["verification"]
        self.tolerance = float(verification["tolerance_mm"])
        window = verification.get("endpoint_window_mm", [self.tolerance, self.tolerance])
        if not isinstance(window, (list, tuple)) or len(window) != 2:
            raise ValueError("verification.endpoint_window_mm must be [under_mm, over_mm]")
        # 端点目标：允许欠推 under、越过 over（越过部分视为读数超出物理终点，按不可靠处理）。
        self.endpoint_window = tuple(_finite_setting(v, "verification.endpoint_window_mm")
                                     for v in window)
        self.stationary_drift_mm = _finite_setting(
            verification.get("stationary_drift_mm", 1.0), "verification.stationary_drift_mm")
        extra = verification.get("unreliable_extra_samples", 3)
        if type(extra) is not int or not 0 <= extra <= 10:
            raise ValueError("verification.unreliable_extra_samples must be an integer in 0..10")
        self.unreliable_extra_samples = extra
        force = verification.get("force_pass_unreliable", False)
        if not isinstance(force, bool):
            raise ValueError("verification.force_pass_unreliable must be true or false")
        self.force_pass_unreliable = force
        self.start_tolerance = _finite_setting(
            config.get("vision", {}).get("beyond_stop_tolerance_mm", 2.5),
            "vision.beyond_stop_tolerance_mm")
        self.arrival_window_mm = _finite_setting(
            config.get("planning", {}).get("arrival_window_mm", 0.), "planning.arrival_window_mm")
        self.reference_positions = {}
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
        self.return_upper_min_deg = self.config.get("planning", {}).get("return_upper_min_deg")
        if self.return_upper_min_deg is not None:
            self.return_upper_min_deg = _finite_setting(
                self.return_upper_min_deg, "planning.return_upper_min_deg")
            if self.return_upper_min_deg >= self.return_q_deg[0]:
                raise ValueError("planning.return_upper_min_deg must be below return upper target")

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
        # 行程为模型固定值；视觉读数与模型存在安装/标定偏差，起点按物理止挡容差检查。
        if not lo - self.start_tolerance <= start <= hi + self.start_tolerance:
            raise TaskError("slider_start_outside_observed_limits", {
                "color": color, "measured_x_mm": start,
                "x_limits_mm": [lo, hi], "tolerance_mm": self.start_tolerance,
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
        name = phase.get("route") if phase.get("route") == "observation_retreat" else phase.get("phase", "unknown")
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

    def _endpoint_sign(self, color, goal):
        """目标为行程端点时返回指向该端点的方向（+1 右端 / −1 左端），否则 None。"""
        targets = getattr(self, "targets", None)
        if targets is None:
            return None
        lo, hi = targets.bounds(color)
        if abs(goal - hi) <= 1e-6:
            return 1
        if abs(goal - lo) <= 1e-6:
            return -1
        return None

    def _band(self, color, goal):
        """返回 (under_mm, over_mm)：端点用端点窗口，中间目标用对称容差。"""
        if self._endpoint_sign(color, goal) is not None:
            return self.endpoint_window
        return self.tolerance, self.tolerance

    def _signed_error(self, color, goal, actual, sign=None):
        """沿推动方向的有符号误差：负值为欠推，正值为越过目标。"""
        if sign is None:
            sign = self._endpoint_sign(color, goal) or 1
        return sign * (float(actual) - goal)

    def _sample_reliability(self, state, color):
        """单帧可靠性：目标杆识别可靠，且另一根（本任务不推）杆相对初始读数未漂移。"""
        reliability = state.get("reliability") or {}
        own = reliability.get(color, {})
        reliable = bool(own.get("reliable", True))
        reasons = list(own.get("reasons", []))
        for other, reference in getattr(self, "reference_positions", {}).items():
            if other == color or other not in state.get("sliders_x_mm", {}):
                continue
            other_state = reliability.get(other, {})
            drift = abs(float(state["sliders_x_mm"][other]) - float(reference))
            if other_state.get("reliable", True) and drift > self.stationary_drift_mm:
                reliable = False
                reasons.append("stationary_slider_drift:%s:%.3f" % (other, drift))
            elif not other_state.get("reliable", True):
                reasons.append("stationary_slider_unreliable:%s" % other)
        return reliable, reasons

    def _verify(self, color, goal, after, sign=None):
        """连续取帧验收；只有可靠帧计入。不可靠帧最多补取 unreliable_extra_samples 帧。"""
        if sign is None:
            sign = self._endpoint_sign(color, goal) or 1
        under, over = self._band(color, goal)
        endpoint = self._endpoint_sign(color, goal) is not None
        needed = int(self.config["verification"]["samples"])
        max_reads = needed + getattr(self, "unreliable_extra_samples", 0)
        samples, reliable_states = [], []
        state = None
        for i in range(max_reads):
            if len(reliable_states) >= needed:
                break
            if i:
                time.sleep(float(self.config["verification"]["interval_s"]))
            state = self.vision.read(after_monotonic_s=after)
            actual = float(state["sliders_x_mm"][color])
            error = actual - goal
            reliable, reasons = self._sample_reliability(state, color)
            if reliable and endpoint and sign * error > over:
                # 端点任务读数越过物理终点余量：实体不可能越过止挡，按视觉不可靠处理。
                reliable = False
                reasons.append("reading_beyond_target_end_window")
            samples.append({"error_mm": error, "stamp_monotonic_s": state["stamp_monotonic_s"],
                            "reliable": reliable, "reasons": reasons})
            if reliable:
                reliable_states.append(state)
            after = state["stamp_monotonic_s"]
        result = {"samples": samples, "window_mm": [under, over], "direction_sign": sign,
                  "reliable_samples": len(reliable_states), "unreliable": False, "forced": False}
        if len(reliable_states) >= needed:
            errors = [sign * (float(s["sliders_x_mm"][color]) - goal) for s in reliable_states]
            result.update(passed=all(-under <= e <= over for e in errors),
                          last_observation=reliable_states[-1])
            return result
        result["unreliable_reasons"] = sorted({r for s in samples if not s["reliable"]
                                               for r in s["reasons"]})
        if getattr(self, "force_pass_unreliable", False) and state is not None:
            # 配置强制通过：视为已到目标并进入下一阶段，状态中的目标杆位置取任务目标。
            observation = copy.deepcopy(state)
            observation["sliders_x_mm"] = dict(observation["sliders_x_mm"])
            observation["forced_from_x_mm"] = observation["sliders_x_mm"][color]
            observation["sliders_x_mm"][color] = goal
            result.update(passed=True, forced=True, last_observation=observation)
            print("[vision] verification forced pass: %s" % result["unreliable_reasons"], flush=True)
            return result
        result.update(passed=False, unreliable=True, last_observation=state)
        return result

    def _raise_if_unreliable(self, verification):
        if verification.get("unreliable"):
            raise TaskError("verification_unreliable", {
                "reasons": verification.get("unreliable_reasons", []),
                "reliable_samples": verification.get("reliable_samples"),
                "samples": verification.get("samples")})

    def _stage(self, name, state, record, action):
        """Each stage owns its initial attempt plus retry_limit retries."""
        attempts = record.setdefault("stage_attempts", {}).setdefault(name, [])
        limit = int(self.config["retry_limit"])
        for number in range(limit + 1):
            attempt = {"attempt": number}
            attempts.append(attempt)
            try:
                if number:
                    stop = self.motors.halt()
                    attempt["stop_before_retry"] = stop
                    if not stop.get("stopped_confirmed"):
                        raise TaskError("stop_not_confirmed")
                    self._clear_for_observation(state, attempt)
                    state.update(self._snapshot())
                attempt["start_state"] = dict(state)
                result = action(attempt)
                attempt["status"] = "completed"
                return result
            except Exception as exc:
                stop = self.motors.halt()
                attempt.update(status="failed", reason=str(exc),
                               error_type=type(exc).__name__,
                               details=getattr(exc, "details", {}), stop=stop)
                # A requested halt, hard limit or unconfirmed stop must never
                # be cleared by the next call to MotorPair.move.
                if (number == limit or not stop.get("stopped_confirmed")
                        or str(exc) in self._NO_RETRY):
                    raise
                print("[motion] retry %s %d/%d: %s" %
                      (name, number + 1, limit, exc), flush=True)

    def _withdraw(self, state, record):
        if self.contact is not None:
            c, d = self.contact
            options = ({"from_reference": True, "reference_extension_mm": self.withdraw_extension_mm}
                       if self.withdraw_extension_mm else {})
            phase = None
            def withdraw(attempt):
                nonlocal phase
                if phase is None:
                    phase = self.planner.withdraw(state["q_rad"], c, d, state["sliders_x_mm"], **options)
                else:
                    phase = self.planner.retry_contact(phase, state["q_rad"], state["sliders_x_mm"])
                state["q_rad"] = self._move(phase, attempt)
                # A withdrawal can disturb a slider. Include its fresh image
                # in this stage, so observation failures use the same budget.
                state.update(self.vision.read(after_monotonic_s=time.monotonic()))
            self._stage("withdraw", state, record, withdraw)
            self.contact = None
            self.withdraw_extension_mm = 0.

    def _clear_for_observation(self, state, record):
        if self.contact is None:
            return
        color, direction = self.contact
        state.update(self.motors.read_state())
        retreat = record.setdefault("observation_retreats", [])
        entry = {"start_state": dict(state)}
        retreat.append(entry)
        limits = {c: self.targets.bounds(c) for c in self.config["sliders"]}
        phase = self.planner.observation_retreat(state["q_rad"], color, direction,
            state["sliders_x_mm"], limits[color], all_slider_limits=limits)
        entry["plan"] = phase
        if len(phase["points"]) > 1:
            state["q_rad"] = self._move(phase, entry)
        stop = self.motors.halt()
        entry["stop"] = stop
        if not stop.get("stopped_confirmed"):
            raise TaskError("stop_not_confirmed")
        state.update(self.motors.read_state())
        entry["all_slider_clearances"] = {c: self.planner.observation_pose(state["q_rad"],
            c, direction, state["sliders_x_mm"], bound) for c, bound in limits.items()}
        entry["measured_clearance"] = entry["all_slider_clearances"][color]
        entry["status"] = "completed"
        self.contact = None
        self.withdraw_extension_mm = 0.

    def _stopped_verification(self, color, goal, record, state):
        stop = self.motors.halt()
        record["stop"] = stop
        if not stop.get("stopped_confirmed"):
            raise TaskError("stop_not_confirmed")
        self._clear_for_observation(state, record)
        after = time.monotonic()
        record["verification_after_monotonic_s"] = after
        verification = self._verify(color, goal, after)
        record["verification"] = verification
        return verification

    def _model_tool_x(self, q, direction):
        return float(self.planner.arm.slider_center(q, direction)[0])

    def _arrival(self, phase, color, goal, record, state, error):
        """按模型推头 X 判定机械臂到位（端点任务 ±arrival_window_mm），并记录推头位置。"""
        direction = phase.get("direction", self.tool_direction)
        try:
            q = self.motors.read_state()["q_rad"]
            model_x = self._model_tool_x(q, direction)
        except Exception as exc:  # 反馈或模型不可用时不做任何吸收
            record["arrival"] = {"status": "unavailable", "error": str(exc)}
            return False
        endpoint = self._endpoint_sign(color, goal) is not None
        window = getattr(self, "arrival_window_mm", 0.)
        within = endpoint and window > 0 and abs(model_x - goal) <= window
        record["arrival"] = {"model_tool_x_mm": model_x, "goal_x_mm": goal,
                             "model_minus_goal_mm": model_x - goal,
                             "window_mm": window if endpoint else None,
                             "within_window": bool(within)}
        if error is not None and within and str(error) in self._ARRIVAL_ABSORBABLE:
            record["arrival"]["absorbed_motor_error"] = str(error)
            # 补推以实测停止姿态为参考。
            state["last_push_q_rad"] = list(q)
            return True
        return False

    def _stopped_push(self, phase, color, goal, record, state):
        error = None
        try:
            state["last_push_q_rad"] = self._move(phase, record)
        except Exception as exc:
            error = exc
            record["push_motor_error"] = str(error)
            if str(error) in self._NO_RETRY:
                raise
        if self._arrival(phase, color, goal, record, state, error):
            print("[motion] push arrival within window; motor error absorbed: %s" % error, flush=True)
            error = None
        # Even a normally completed trajectory must be explicitly stopped
        # before obtaining the images used for task acceptance.
        verification = self._stopped_verification(color, goal, record, state)
        self._raise_if_unreliable(verification)
        return verification, error

    def _push_stage(self, name, factory, color, goal, direction, state, record):
        under, over = self._band(color, goal)
        def push(attempt):
            sign = 1 if direction == "+x" else -1
            # A previous failed move may already have reached the slider goal.
            if attempt["attempt"]:
                verification = self._stopped_verification(color, goal, attempt, state)
                self._raise_if_unreliable(verification)
                state.update(verification["last_observation"])
                if verification["passed"]:
                    return verification, None
                signed = sign * (state["sliders_x_mm"][color] - goal)
                if signed > over:
                    raise TaskError("slider_overshot_target")
                if signed >= -under:
                    raise TaskError("slider_not_stably_at_target")
            if self.contact is None:
                # Verification now leaves the tool in the observation lane.
                # Both retries and bounded corrections need a fresh entry.
                state["q_rad"] = self.motors.read_state()["q_rad"]
                entry = attempt["reentry_free"] = {"start_state": dict(state)}
                self._free_entry(color, direction, state, entry)
            before = state["sliders_x_mm"][color]
            verification, error = self._stopped_push(factory(), color, goal, attempt, state)
            state.update(verification["last_observation"])
            if error is not None and str(error) in self._NO_RETRY:
                raise error
            if error is not None and not verification["passed"]:
                raise error
            if not verification["passed"]:
                actual = state["sliders_x_mm"][color]
                # 仅可靠帧可到达此处（不可靠已在 _stopped_push 中报 verification_unreliable）。
                if sign * (actual - goal) > over:
                    raise TaskError("slider_overshot_target")
                if sign * (actual - before) < self.correction_settings["min_progress_mm"]:
                    raise TaskError("push_no_progress")
            return verification, None
        result = self._stage(name, state, record, push)
        # Keep the established stopped-verification fields for result readers.
        last = record["stage_attempts"][name][-1]
        for key in ("stop", "verification", "verification_after_monotonic_s", "push_motor_error",
                    "arrival"):
            if key in last:
                record[key] = last[key]
        return result

    def _free_entry(self, color, direction, state, record):
        # Even a partial entry must clear the camera view before retry images.
        self.contact = color, direction
        pre = self.planner.precontact(color, direction, state["sliders_x_mm"], state["q_rad"])
        state["q_rad"] = self._move(self.planner.free(state["q_rad"], pre,
            state["sliders_x_mm"], direction=direction), record)

    def _push_with_corrections(self, color, goal, direction, state, record):
        sign = 1 if direction == "+x" else -1
        before = float(state["sliders_x_mm"][color])
        verification, error = self._push_stage("push", lambda: self.planner.push(
            state["q_rad"], color, goal, state["sliders_x_mm"], self.targets.bounds(color)),
            color, goal, direction, state, record)
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
                raise error
            under, over = self._band(color, goal)
            signed = sign * (actual - goal)
            remaining = -signed
            if signed > over:
                raise TaskError("slider_overshot_target")
            if signed >= -under:
                raise TaskError("slider_not_stably_at_target")
            progress = sign * (actual - before)
            if progress < settings["min_progress_mm"]:
                raise TaskError("push_no_progress", {"progress_mm": progress})
            if (len(corrections) >= settings["max_steps"]
                    or used >= settings["max_distance_mm"] - 1e-9):
                raise TaskError("push_correction_budget_exhausted", {
                    "steps": len(corrections), "extension_mm": used,
                    "remaining_mm": remaining})
            # 端点目标：补推后的模型推头不得越过模型固定行程终点（−2.06 已含预留间隙）。
            # 中间目标保持原逻辑（补推用于抵消 FK 与视觉的小偏差）。
            model_room = math.inf
            if self._endpoint_sign(color, goal) is not None:
                model_room = sign * (goal - self._model_tool_x(state["last_push_q_rad"], direction))
                if model_room <= 1e-6:
                    raise TaskError("push_correction_model_target_reached", {
                        "remaining_mm": remaining, "model_room_mm": model_room})
            # Keep the previous measured contact pose as the correction
            # reference; the current measured pose is now outside the rail.
            state["q_rad"] = self.motors.read_state()["q_rad"]
            distance = min(settings["step_mm"], settings["max_distance_mm"] - used, remaining,
                           model_room)
            used += distance
            correction = {"step": len(corrections) + 1, "extension_mm": distance,
                          "cumulative_extension_mm": used, "task_target_x_mm": goal,
                          "before_x_mm": actual}
            corrections.append(correction)
            before = actual
            reference_q = list(state["last_push_q_rad"])
            phase = None
            def correction_phase():
                nonlocal phase
                if phase is None:
                    phase = self.planner.correction(reference_q, color, goal, direction,
                        state["sliders_x_mm"], self.targets.bounds(color), distance)
                    correction.update(reference_start_mm=phase["reference_start_mm"],
                                      reference_target_mm=phase["reference_target_mm"])
                phase = self.planner.reenter_correction(phase, state["q_rad"], state["sliders_x_mm"])
                return phase
            verification, error = self._push_stage("push_correction", correction_phase,
                color, goal, direction, state, correction)
            correction["after_x_mm"] = float(verification["last_observation"]["sliders_x_mm"][color])
            correction["progress_mm"] = sign * (correction["after_x_mm"] - before)

    def _run_leg(self, color, goal):
        record = {"color": color, "target_x_mm": goal, "attempts": [], "status": "failed"}
        attempt = {"attempt": 0}
        record["attempts"].append(attempt)
        state = {}
        try:
            def observe(_):
                state.update(self._snapshot())
            self._stage("observe", state, attempt, observe)
            attempt["start_state"] = dict(state)
            self._withdraw(state, attempt)
            start = state["sliders_x_mm"][color]
            under, over = self._band(color, goal)
            if -under <= self._signed_error(color, goal, start) <= over:
                def verify(_):
                    verification = self._stopped_verification(color, goal, attempt, state)
                    self._raise_if_unreliable(verification)
                    if not verification["passed"]:
                        raise TaskError("slider_not_at_target")
                    state.update(verification["last_observation"])
                self._stage("verification", state, attempt, verify)
                attempt["no_motion"] = True
            else:
                direction = "+x" if goal > start else "-x"
                self.tool_direction = direction
                def free(stage_attempt):
                    self._free_entry(color, direction, state, stage_attempt)
                self._stage("free", state, attempt, free)
                self.contact = color, direction
                self.withdraw_extension_mm = 0.
                state = self._push_with_corrections(color, goal, direction, state, attempt)
                self._withdraw(state, attempt)
            attempt["status"] = record["status"] = "completed"
            return record, state
        except KeyboardInterrupt:
            attempt.update(status="interrupted", stop=self.motors.halt())
            record["reason"] = "task_interrupted"
        except Exception as exc:
            attempt.update(status="failed", reason=str(exc), error_type=type(exc).__name__,
                           details=getattr(exc, "details", {}), stop=self.motors.halt())
            record["reason"] = str(exc)
        return record, None

    def _initial_environment(self, result):
        """Confirm fresh, reliable and stationary slider readings before freezing any goal.

        行程范围为模型固定值；这里要求连续 3 帧两色识别均可靠，且两杆位置波动 ≤ 1 mm。
        """
        required, max_spread = 3, 1.0
        limit = max(required, int(self.config["retry_limit"]) + 1)
        confirmation = result["initial_limits_confirmation"] = {
            "status": "unconfirmed", "required_samples": required,
            "max_spread_mm": max_spread, "max_observations": limit, "observations": []}
        accepted = []
        for number in range(limit):
            try:
                stop = self.motors.halt()
                if not stop.get("stopped_confirmed"):
                    raise TaskError("initial_stop_not_confirmed")
                observed = self._snapshot()
                result["initial_state"] = observed
                limits = observed.get("slider_limits_x_mm")
                reliability = observed.get("reliability") or {}
                confirmation["observations"].append({
                    "attempt": number, "stamp_monotonic_s": observed.get("stamp_monotonic_s"),
                    "sliders_x_mm": observed.get("sliders_x_mm"),
                    "slider_limits_x_mm": limits, "reliability": reliability,
                    "rail_observation": observed.get("rail_observation", {})})
                if limits is None:
                    raise TaskError("slider_limits_unavailable", {
                        "rail_observation": observed.get("rail_observation", {})})
                unreliable = {c: r.get("reasons", []) for c, r in reliability.items()
                              if not r.get("reliable", True)}
                if unreliable:
                    print("[vision] initial frame unreliable: %s" % unreliable, flush=True)
                    raise TaskError("initial_vision_unreliable", {"reasons": unreliable})
                # Validate both colors before using their numbers for stability.
                TargetResolver({"sliders": self.config["sliders"],
                    "feedback": {"slider_position_tolerance_mm": self.tolerance}}, limits)
                accepted.append(observed)
                spread = max(max(s["sliders_x_mm"][c] for s in accepted)
                             - min(s["sliders_x_mm"][c] for s in accepted)
                             for c in self.config["sliders"])
                if spread > max_spread:
                    raise TaskError("initial_positions_unstable", {"spread_mm": spread,
                                                                   "max_spread_mm": max_spread})
                if len(accepted) >= required:
                    confirmation.update(status="confirmed", spread_mm=spread)
                    return observed
                print("[vision] initial confirmation %d/%d" % (len(accepted), required), flush=True)
            except Exception as exc:
                accepted = []
                result["environment_recovery"].append({"attempt": number, "reason": str(exc),
                                                       "details": getattr(exc, "details", {})})
                if str(exc) in self._NO_RETRY or str(exc) == "initial_stop_not_confirmed":
                    raise
                if number == limit - 1:
                    raise
                print("[vision] retry initial environment %d: %s" % (number + 1, exc), flush=True)
        raise TaskError("initial_positions_unstable", {"confirmed_samples": len(accepted),
                                                       "required_samples": required})

    def run(self, task):
        task = normalize_task(task)
        result = {"mode": "hardware_execution", "status": "failed", "task": task,
                  "segments": [], "completed_round_trips": 0,
                  "planned_round_trips": task.get("round_trips", 0)}
        try:
            result["environment_recovery"] = []
            initial = self._initial_environment(result)
            # Preserve the observation even if task target validation fails.
            result["initial_state"] = initial
            # 非目标杆的初始读数用作验收时的静止漂移基准。
            self.reference_positions = {c: float(x) for c, x in initial["sliders_x_mm"].items()
                                        if c != task["color"]}
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
            if self.return_upper_min_deg is not None:
                result["return_motion"]["upper_min_deg_exclusive"] = self.return_upper_min_deg
                print("[motion] return acceptance: upper > %.2f deg; lower target=%.2f deg; still/hold required" %
                      (self.return_upper_min_deg, self.return_q_deg[1]), flush=True)
            def return_home(attempt):
                phase = self.planner.free(state["q_rad"], return_q_rad,
                    state["sliders_x_mm"], direction=self.tool_direction)
                phase["route"] = "return_home"
                if self.return_upper_min_deg is not None:
                    phase["return_upper_min_deg"] = self.return_upper_min_deg
                state["q_rad"] = self._move(phase, attempt)
            self._stage("return", state, result["return_motion"], return_home)
            result["status"] = "completed"
        except KeyboardInterrupt:
            result.update(reason="task_interrupted", stop=self.motors.halt())
        except Exception as exc:
            result.update(reason=str(exc), error_type=type(exc).__name__,
                          details=getattr(exc, "details", {}), stop=self.motors.halt())
        return result
