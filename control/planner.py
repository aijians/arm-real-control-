"""Fixed-scene Cartesian lines, with one configured obstacle passage.

All positions are mm in the shoulder frame; q2 is the physical BAC angle.
Free motion uses the selected face's rigid equivalent point, including the
pin radius. Collision checks use only scene.json primitives. Pushes retain
the agreed real-arm policy: straight IK interpolation without wall rejection.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import time

from .config import resolve, push_correction_settings
from .kinematics import PlanarArm
from .scene import SimpleScene


class PlanError(RuntimeError):
    pass


class Planner:
    def __init__(self, root, config=None):
        self.root = Path(root).resolve()
        self.config = (json.loads((self.root / "config/robot.json").read_text())
                       if config is None else config)
        self.settings = self.config["planning"]
        self.correction_settings = push_correction_settings(self.config)
        for key in ("free_speed_deg_s", "free_speed_mm_s", "push_speed_mm_s",
                    "contact_speed_mm_s", "linear_step_mm",
                    "contact_position_tolerance_mm", "collision_step_deg"):
            value = self.settings[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise PlanError("invalid_planning_parameter:" + key)
        minimum = self.settings.get("waypoint_min_step_deg", 1.2)
        if (isinstance(minimum, bool) or not isinstance(minimum, (int, float))
                or not math.isfinite(minimum) or minimum <= 0):
            raise PlanError("invalid_planning_parameter:waypoint_min_step_deg")
        self.waypoint_min_step_deg = float(minimum)
        axis_minimum = self.settings.get("waypoint_min_axis_step_deg", 0.)
        if (type(axis_minimum) not in (int,float) or not math.isfinite(axis_minimum)
                or not (axis_minimum == 0 or .05 <= axis_minimum <= .5)):
            raise PlanError("invalid_planning_parameter:waypoint_min_axis_step_deg")
        self.waypoint_min_axis_step_deg = axis_minimum
        self.arm = PlanarArm(self.config["kinematics"])
        scene_config = json.loads(resolve(self.root, self.config["paths"]["scene"]).read_text())
        self.scene = SimpleScene(scene_config)
        self.routes = scene_config["routes"]
        self.offsets = {d: float(self.config["kinematics"]["faces"][d]["approach_offset_mm"]) for d in ("+x", "-x")}
        if any(not math.isfinite(v) or v <= 0 for v in self.offsets.values()):
            raise PlanError("invalid_approach_offset")
        hardware = self.config.get("hardware", {})
        ratios = [abs(float(m.get("ratio", 1))) for m in hardware.get("motors", [])] or [1.]
        self.joint_rate = math.radians(min(float(self.settings["free_speed_deg_s"]),
            float(hardware.get("max_speed_deg_s", self.settings["free_speed_deg_s"])) / max(ratios)))
        if not math.isfinite(self.joint_rate) or self.joint_rate <= 0:
            raise PlanError("invalid_joint_rate")

    def _q(self, value):
        try:
            q = [float(v) for v in value]
            if len(q) != 2 or not all(math.isfinite(v) for v in q) or not self.arm.within_limits(q):
                raise ValueError()
            return q
        except (TypeError, ValueError):
            raise PlanError("joint_angles_outside_model_limits") from None

    def _positions(self, sliders):
        try:
            result = {color: float(sliders[color]) for color in self.config["sliders"]}
            if not all(math.isfinite(v) for v in result.values()):
                raise ValueError()
            return result
        except (KeyError, TypeError, ValueError):
            raise PlanError("two_finite_slider_positions_required") from None

    def _target(self, color, sliders):
        if color not in self.config["sliders"]:
            raise PlanError("unknown_slider_color")
        return [sliders[color], float(self.config["sliders"][color]["y_mm"])]

    def _inverse(self, target, direction, seed, contact=True):
        try:
            q = list(self.arm.inverse(target, direction, seed=seed, require_direction=contact))
        except ValueError as exc:
            raise PlanError("inverse_failed:" + str(exc)) from exc
        if max(abs(a-b) for a, b in zip(q, seed)) > math.radians(5):
            raise PlanError("inverse_branch_discontinuity")
        return self._q(q)

    @staticmethod
    def _inverse_smoothstep(fraction):
        if fraction <= 0:
            return 0.
        if fraction >= 1:
            return 1.
        lo, hi = 0., 1.
        for _ in range(45):
            mid = (lo + hi) / 2
            smooth = mid ** 3 * (10 + mid * (-15 + 6 * mid))
            if smooth < fraction:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def _command_chord_clear(self, name, a, b, direction, collision_context):
        """Check the actual joint interpolation introduced by decimation.

        Dense IK checks are retained, but they do not certify a longer joint
        chord. Contact pushes retain their established wall policy; every
        phase still checks joint limits, Cartesian line deviation and (except
        free) the selected pushing normal. 0.1 mm is a model approximation
        limit here, not a claim about real-arm positioning accuracy.
        """
        if name not in ("push", "push_correction"):
            if collision_context is None:
                return False
            positions, contact = collision_context
            try:
                self._checked_edges([a, b], positions, contact)
            except PlanError:
                return False
        start, end = self.arm.slider_center(a, direction), self.arm.slider_center(b, direction)
        edge = [v-u for u, v in zip(start, end)]
        length2 = sum(v*v for v in edge)
        count = max(1, math.ceil(max(abs(y-x) for x, y in zip(a, b)) /
                                math.radians(self.settings["collision_step_deg"])))
        sign = 1 if direction == "+x" else -1
        for index in range(count + 1):
            q = [x+(y-x)*index/count for x, y in zip(a, b)]
            if not self.arm.within_limits(q):
                return False
            centre = self.arm.slider_center(q, direction)
            if length2 <= 1e-16:
                error = math.dist(centre, start)
            else:
                fraction = sum((v-u)*d for v, u, d in zip(centre, start, edge)) / length2
                nearest = [u + min(1., max(0., fraction))*d for u, d in zip(start, edge)]
                error = math.dist(centre, nearest)
            if error > .1 + 1e-9:
                return False
            if name != "free" and self.arm.world_normal(q, direction)[0] * sign <= 0:
                return False
        return True

    def _command_points(self, name, points, direction, collision_context):
        """Select coarse command nodes while retaining reversals and corners."""
        if len(points) <= 2:
            return points, list(range(len(points)))
        path = [point["q_rad"] for point in points]
        centres = [self.arm.slider_center(q, direction) for q in path]
        mandatory = {0, len(points)-1}
        for axis in range(2):
            previous_sign, previous_end = 0, None
            for index in range(1, len(path)):
                delta = path[index][axis] - path[index-1][axis]
                if abs(delta) <= 1e-10:
                    continue
                sign = 1 if delta > 0 else -1
                if previous_sign and sign != previous_sign:
                    # Keep both ends of an intervening static plateau as
                    # well as the ordinary single-node turning point.
                    mandatory.update((previous_end, index-1))
                previous_sign, previous_end = sign, index
        for index in range(1, len(points)-1):
            left = [b-a for a, b in zip(centres[index-1], centres[index])]
            right = [b-a for a, b in zip(centres[index], centres[index+1])]
            scale = math.hypot(*left) * math.hypot(*right)
            if scale > 1e-12:
                cross = abs(left[0]*right[1]-left[1]*right[0])
                dot = sum(a*b for a, b in zip(left, right))
                if cross > 1e-5*scale or dot < 0:
                    mandatory.add(index)
        selected, last = [0], 0
        minimum = math.radians(self.waypoint_min_step_deg)
        for index in range(1, len(points)):
            span = max(abs(y-x) for x, y in zip(path[last], path[index]))
            axis_steps = [abs(math.degrees(y-x)) for x,y in zip(path[last],path[index])]
            micro = any(1e-7 < step < self.waypoint_min_axis_step_deg-1e-9 for step in axis_steps)
            if index not in mandatory and (span < minimum or micro):
                continue
            if index == last + 1 or self._command_chord_clear(
                    name, path[last], path[index], direction, collision_context):
                selected.append(index)
            else:
                # A forbidden longer chord restores its original checked
                # dense samples; no route search or geometric shortcut.
                selected.extend(range(last + 1, index + 1))
            last = index
        # Geometry fallback and protected corners can reintroduce tiny
        # spans. Remove only dispensable nodes whose new chord is certified.
        changed = True
        while changed:
            changed = False
            for position in range(1,len(selected)-1):
                a,b,c = selected[position-1:position+2]
                if b in mandatory:
                    continue
                spans = [abs(math.degrees(path[y][axis]-path[x][axis]))
                         for x,y in ((a,b),(b,c)) for axis in range(2)]
                if (any(1e-7 < value < self.waypoint_min_axis_step_deg-1e-9 for value in spans)
                        and self._command_chord_clear(name,path[a],path[c],direction,collision_context)):
                    selected.pop(position)
                    changed = True
                    break
        return [points[index] for index in selected], selected

    def _phase(self, name, path, distances, speed, *, _collision_context=None, **metadata):
        retained, costs = [list(path[0])], [0.]
        for i, q in enumerate(path[1:]):
            joint_time = max(abs(b-a) for a, b in zip(path[i], q)) / self.joint_rate
            cost = max(joint_time, float(distances[i]) / speed)
            if cost > 1e-12:
                retained.append(list(q))
                costs.append(costs[-1] + cost)
        if costs[-1] <= 0:
            points = [{"q_rad": retained[0], "time_s": 0.}]
        else:
            duration = 1.875 * costs[-1]
            points = [{"q_rad": q, "time_s": duration * self._inverse_smoothstep(c / costs[-1])}
                      for q, c in zip(retained, costs)]
        dense_count = len(points)
        points, source_indices = self._command_points(
            name, points, metadata.get("direction", "+x"), _collision_context)
        recovery_geometry = None
        policy_enabled = self.config.get("hardware",{}).get("motion_recovery",{}).get("enabled",False)
        if policy_enabled and _collision_context is not None:
            positions, contact = _collision_context
            recovery_geometry = {"scene": copy.deepcopy(self.scene.config),
                                 "sliders": copy.deepcopy(positions), "contact": copy.deepcopy(contact)}
        if policy_enabled and name in ("approach", "withdraw"):
            metadata["cartesian_compensation"] = self._cartesian_compensation(
                self.arm.slider_center(path[0],metadata.get("direction","+x")),
                self.arm.slider_center(path[-1],metadata.get("direction","+x")),metadata.get("direction","+x"))
        protected_micro_steps = [{"index":i,"axis":axis,"step_deg":abs(math.degrees(b["q_rad"][axis]-a["q_rad"][axis]))}
            for i,(a,b) in enumerate(zip(points,points[1:]),1) for axis in range(2)
            if 1e-7 < abs(math.degrees(b["q_rad"][axis]-a["q_rad"][axis])) < self.waypoint_min_axis_step_deg-1e-9]
        return {"phase": name, "points": points,
                "motion_recovery_geometry": recovery_geometry,
                "waypoint_min_axis_step_deg": self.waypoint_min_axis_step_deg,
                "protected_micro_steps": protected_micro_steps, "angle_convention": "model_q1_world_bearing_q2_BAC",
                "time_parameterization": "quintic_progress_on_joint_polyline",
                "dense_point_count": dense_count, "source_point_indices": source_indices,
                "command_point_count": len(points),
                "waypoint_min_step_deg": self.waypoint_min_step_deg,
                "command_chord_cartesian_error_limit_mm": .1, **metadata}

    def _clear(self, q, sliders, contact=None):
        hit = self.scene.check(q, sliders, contact=contact)
        if hit:
            raise PlanError("collision:" + json.dumps(hit, ensure_ascii=False, sort_keys=True))

    def _checked_edges(self, path, sliders, contact=None):
        # Motors interpolate angles between Cartesian IK samples. Check that
        # short joint chord too, using only the lightweight scene primitives.
        step = math.radians(self.settings["collision_step_deg"])
        self._clear(path[0], sliders, contact)
        for a, b in zip(path, path[1:]):
            count = max(1, math.ceil(max(abs(y-x) for x, y in zip(a, b)) / step))
            for i in range(1, count + 1):
                self._clear([x+(y-x)*i/count for x, y in zip(a, b)], sliders, contact)

    def _line(self, q_start, target, direction, contact=True, q_goal=None):
        centre = list(self.arm.slider_center(q_start, direction))
        count = max(1, math.ceil(math.dist(centre, target) / self.settings["linear_step_mm"]))
        path, distances, previous = [list(q_start)], [], centre
        for i in range(1, count + 1):
            c = [a+(b-a)*i/count for a, b in zip(centre, target)]
            nxt = self._inverse(c, direction, path[-1], contact=contact)
            if q_goal is not None and i == count:
                if max(abs(a-b) for a, b in zip(nxt, q_goal)) > 1e-6:
                    raise PlanError("goal_on_different_inverse_branch")
                nxt = list(q_goal)
            distances.append(math.dist(previous, c))
            path.append(nxt)
            previous = c
        return path, distances

    def _free_route(self, start, targets, positions, direction):
        path, distances = [start], []
        for goal in targets:
            part, steps = self._line(path[-1], self.arm.slider_center(goal, direction),
                                     direction, contact=False, q_goal=goal)
            self._checked_edges(part, positions)
            path.extend(part[1:])
            distances.extend(steps)
        return path, distances

    def free(self, q_start, q_goal, sliders, direction="+x"):
        started = time.monotonic()
        start, goal, positions = self._q(q_start), self._q(q_goal), self._positions(sliders)
        if direction not in ("+x", "-x"):
            raise PlanError("invalid_contact_direction")
        self._clear(start, positions)
        self._clear(goal, positions)
        passage = self._q([math.radians(a) for a in self.routes["passage_q_deg"]])
        # Finite, fixed recipes only. Never search a configuration grid.
        via = [passage, goal]
        right = self.arm.slider_center(goal, direction)[0] >= self.routes["right_region_x_min_mm"]
        candidates = [("passage", via), ("direct", [goal])] if right else [("direct", [goal]), ("passage", via)]
        # A face change near a pin needs room below the rail before crossing
        # to its other side. This is one fixed three-line recipe, not a search.
        entry_y = self.routes["entry_y_mm"]
        entry = []
        seed = start
        for pose in (start, goal):
            point = list(self.arm.slider_center(pose, direction))
            if point[1] > entry_y:
                point[1] = entry_y
                try:
                    seed = list(self.arm.inverse(point, direction, seed=seed, require_direction=False))
                    entry.append(seed)
                except ValueError:
                    break
        else:
            candidates.append(("rail_entry", entry + [goal]))
        failures = []
        for route_name, targets in candidates:
            try:
                path, distances = self._free_route(start, targets, positions, direction)
                return self._phase("free", path, distances, self.settings["free_speed_mm_s"],
                    _collision_context=(positions, None),
                    direction=direction, route=route_name, collision_checked=True,
                    slider_positions_x_mm=positions, planning_elapsed_s=time.monotonic()-started,
                    cartesian_waypoints_mm=[list(self.arm.slider_center(q, direction)) for q in [start] + targets])
            except PlanError as exc:
                failures.append(str(exc))
        raise PlanError("free_path_not_found:" + " | ".join(failures))

    def precontact(self, color, direction, sliders, seed):
        positions, seed = self._positions(sliders), self._q(seed)
        target = self._target(color, positions)
        target[0] -= (1 if direction == "+x" else -1) * self.offsets[direction]
        try:
            q = self._q(self.arm.inverse(target, direction, seed=seed))
        except ValueError as exc:
            raise PlanError("precontact_inverse_failed:" + str(exc)) from exc
        self._clear(q, positions)
        return q

    def _contact_phase(self, name, q_start, color, direction, sliders):
        q, positions = self._q(q_start), self._positions(sliders)
        target = self._target(color, positions)
        centre = list(self.arm.slider_center(q, direction))
        sign = 1 if direction == "+x" else -1
        if self.arm.world_normal(q, direction)[0] * sign <= 0:
            raise PlanError("contact_face_points_away_from_slider")
        gap, tangential = sign * (target[0]-centre[0]), abs(target[1]-centre[1])
        tolerance = self.settings["contact_position_tolerance_mm"]
        if tangential > tolerance or gap < -tolerance:
            raise PlanError("contact_start_position_error_exceeds_tolerance")
        if gap > self.offsets[direction] + tolerance:
            raise PlanError("contact_start_gap_exceeds_tolerance")
        if name == "withdraw":
            target[0] -= sign * self.offsets[direction]
        path, distances = self._line(q, target, direction)
        # The scene exemption is limited to the selected face patch against
        # the selected target stem, not the entire pin or other tool parts.
        contact = {"color": color, "direction": direction,
                   "tolerance_mm": self.offsets[direction] + tolerance}
        self._checked_edges(path, positions, contact)
        return self._phase(name, path, distances, self.settings["contact_speed_mm_s"],
            _collision_context=(positions, contact),
            color=color, direction=direction, collision_checked=True,
            start_gap_mm=gap, end_gap_mm=(self.offsets[direction] if name == "withdraw" else 0.),
            start_tangential_error_mm=tangential, offset_axis="world_x")

    def approach(self, q_start, color, direction, sliders):
        return self._contact_phase("approach", q_start, color, direction, sliders)

    def withdraw(self, q_start, color, direction, sliders, *, from_reference=False,
                 reference_extension_mm=0.):
        if not from_reference:
            return self._contact_phase("withdraw", q_start, color, direction, sliders)
        q, positions = self._q(q_start), self._positions(sliders)
        if direction not in ("+x", "-x"):
            raise PlanError("invalid_contact_direction")
        if (type(reference_extension_mm) not in (int, float)
                or not math.isfinite(reference_extension_mm)
                or not 0 < reference_extension_mm <= self.correction_settings["max_distance_mm"]):
            raise PlanError("invalid_reference_withdraw_extension")
        sign = 1 if direction == "+x" else -1
        centre = list(self.arm.slider_center(q, direction))
        target = self._target(color, positions)
        y_error = abs(centre[1] - target[1])
        if y_error > self.settings["contact_position_tolerance_mm"]:
            raise PlanError("reference_withdraw_y_error_exceeds_tolerance")
        if self.arm.world_normal(q, direction)[0] * sign <= 0:
            raise PlanError("contact_face_points_away_from_slider")
        if abs(centre[0] - target[0]) > reference_extension_mm + self.config["verification"]["tolerance_mm"] + 1e-9:
            raise PlanError("reference_withdraw_x_error_exceeds_budget")
        # Undo the accumulated reference extension before opening the normal
        # approach gap; otherwise a 2 mm retreat after a 3 mm correction may
        # still leave the model touching the pin and block the next free move.
        retreat = self.offsets[direction] + reference_extension_mm
        end = [centre[0] - sign * retreat, centre[1]]
        path, distances = self._line(q, end, direction)
        contact = {"color": color, "direction": direction,
                   "reference_contact_tolerance_mm": reference_extension_mm}
        self._checked_edges(path, positions, contact)
        self._clear(path[-1], positions)
        return self._phase("withdraw", path, distances, self.settings["contact_speed_mm_s"],
            _collision_context=(positions, contact),
            color=color, direction=direction, collision_checked=True, from_reference=True,
            reference_start_mm=centre, reference_target_mm=end,
            reference_extension_mm=reference_extension_mm,
            withdraw_distance_mm=retreat,
            reference_contact_tolerance_mm=reference_extension_mm,
            start_tangential_error_mm=y_error, offset_axis="world_x")

    @staticmethod
    def _push_target(target_x_mm, limits_x_mm):
        try:
            goal = float(target_x_mm)
        except (TypeError, ValueError):
            raise PlanError("invalid_push_target") from None
        try:
            lo, hi = limits_x_mm
            lo, hi = float(lo), float(hi)
            if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
                raise ValueError()
        except (TypeError, ValueError):
            raise PlanError("invalid_observed_slider_limits") from None
        if not math.isfinite(goal) or not lo <= goal <= hi:
            raise PlanError("push_target_outside_observed_limits")
        return goal, lo, hi

    def _cartesian_compensation(self, start, end, direction):
        """Describe bounded in-path push assistance without changing its goal.

        The executor advances a scalar distance on this same Cartesian line
        and solves both joint commands together using the pure-math PlanarArm.
        No per-joint push bias and no implicit extension past ``end`` is
        permitted. The task layer alone owns its stopped-observation extension
        budget (push_correction); even those phases stop at their own endpoint.
        A detached JSON-compatible geometry snapshot makes saved phases usable
        without importing Planner or consulting a subsequently edited config.
        """
        return {
            "version": 1,
            "direction": direction,
            "start_xy_mm": list(start),
            "end_xy_mm": list(end),
            "geometry": copy.deepcopy(self.config["kinematics"]),
            "step_mm": .2,
            "max_lead_mm": .5,
            "allow_endpoint_extension": False,
        }

    def push(self, q_start, color, target_x_mm, sliders, limits_x_mm):
        q, positions = self._q(q_start), self._positions(sliders)
        start = self._target(color, positions)
        goal, lo, hi = self._push_target(target_x_mm, limits_x_mm)
        direction = "+x" if goal >= start[0] else "-x"
        centre = list(self.arm.slider_center(q, direction))
        # Start from measured free-motion feedback and cross the configured
        # precontact gap within this same straight push. Contact is a point
        # on the path, with no separate stop or contact-arrival claim.
        error = math.dist(centre, start)
        sign = 1 if direction == "+x" else -1
        gap = sign * (start[0] - centre[0])
        tangential = abs(start[1] - centre[1])
        tolerance = self.settings["contact_position_tolerance_mm"]
        if tangential > tolerance or gap < -tolerance:
            raise PlanError("push_start_contact_error_exceeds_tolerance")
        if gap > self.offsets[direction] + tolerance:
            raise PlanError("push_start_gap_exceeds_tolerance")
        if self.arm.world_normal(q, direction)[0] * sign <= 0:
            raise PlanError("contact_face_points_away_from_slider")
        end = [goal, start[1]]
        path, distances = self._line(q, end, direction)
        return self._phase("push", path, distances, self.settings["push_speed_mm_s"],
            color=color, direction=direction, start_x_mm=start[0], target_x_mm=goal,
            feasible_x_limits_mm=[lo, hi],
            start_equivalent_center_mm=centre, target_center_mm=end,
            start_contact_error_mm=error, start_gap_mm=gap,
            precontact_offset_mm=self.offsets[direction],
            start_tangential_error_mm=tangential,
            entry_mode="continuous_from_precontact", collision_checked=False,
            fixed_point_canonical_xy_mm=list(self.arm.faces[direction]["point_canonical_xy_mm"]),
            linear_step_mm=self.settings["linear_step_mm"],
            cartesian_compensation=self._cartesian_compensation(centre, end, direction))

    def correction(self, q_start, color, task_goal_x_mm, direction, sliders,
                   limits_x_mm, distance_mm):
        """Extend the measured tool reference; never replace the task goal.

        A small X mismatch between FK and vision is precisely why this phase
        exists. Y alignment, the pushing face and continuous IK still apply.
        The caller owns the per-leg count and cumulative extension budget.
        """
        q, positions = self._q(q_start), self._positions(sliders)
        goal, lo, hi = self._push_target(task_goal_x_mm, limits_x_mm)
        if direction not in ("+x", "-x"):
            raise PlanError("invalid_contact_direction")
        settings = self.correction_settings
        if (type(distance_mm) not in (int, float) or not math.isfinite(distance_mm)
                or not 0 < distance_mm <= settings["step_mm"]):
            raise PlanError("invalid_push_correction_distance")
        sign = 1 if direction == "+x" else -1
        start = list(self.arm.slider_center(q, direction))
        y_error = abs(start[1] - self._target(color, positions)[1])
        if y_error > self.settings["contact_position_tolerance_mm"]:
            raise PlanError("push_correction_y_error_exceeds_tolerance")
        if self.arm.world_normal(q, direction)[0] * sign <= 0:
            raise PlanError("contact_face_points_away_from_slider")
        end = [start[0] + sign * distance_mm, start[1]]
        path, distances = self._line(q, end, direction)
        return self._phase("push_correction", path, distances, settings["speed_mm_s"],
            color=color, direction=direction, task_target_x_mm=goal,
            feasible_x_limits_mm=[lo, hi],
            reference_start_mm=start, reference_target_mm=end,
            extension_mm=distance_mm, start_tangential_error_mm=y_error,
            collision_checked=False,
            fixed_point_canonical_xy_mm=list(self.arm.faces[direction]["point_canonical_xy_mm"]),
            linear_step_mm=self.settings["linear_step_mm"],
            cartesian_compensation=self._cartesian_compensation(start, end, direction))
