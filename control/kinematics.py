"""Planar kinematics using the measured model dimensions in robot.json.

All lengths are millimetres and all angles are radians. In this delivery,
q1 is the shoulder-to-elbow world bearing and q2 is the physical BAC angle.  ``canonical`` points
already include the elbow's static yaw: k = R(elbow_static_yaw) * local_point.
Do not add that yaw again when using this module.

The IK target is the SLIDER CENTRE, never the centre of a contact face. The
selected surface point plus the stem radius along its toward-slider normal
is rigidly attached to the forearm and is the point solved by IK.  This is a
contact-point model; it does not check the rest of the arm for collisions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

XY = tuple[float, float]
JointPair = tuple[float, float]
_ANGLE_TOL = 1e-9
_REACH_TOL = 1e-10
_NORMAL_X_TOL = 1e-9
_POSITION_TOL_MM = 1e-6


class GeometryError(ValueError):
    """Invalid geometry or malformed numeric input."""


class IKError(ValueError):
    """No permitted contact configuration solves a requested slider centre."""


@dataclass(frozen=True)
class IKCandidate:
    """A solution inside the joint limits, including wrong-side solutions.

    ``direction_valid`` requires the toward-slider normal to have the same
    world-X sign as the requested push direction.  A candidate with False is
    diagnostic evidence only; ``inverse`` never returns it.
    """

    joint_angles_rad: JointPair
    world_normal_xy: XY
    direction_valid: bool
    position_error_mm: float

    @property
    def q(self) -> JointPair:
        return self.joint_angles_rad


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GeometryError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise GeometryError(f"{name} must be a finite number")
    return result


def _pair(value: Any, name: str) -> XY:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise GeometryError(f"{name} must contain exactly two numbers")
    return _number(value[0], f"{name}[0]"), _number(value[1], f"{name}[1]")


def _rotate(angle: float, xy: XY) -> XY:
    c, s = math.cos(angle), math.sin(angle)
    return c * xy[0] - s * xy[1], s * xy[0] + c * xy[1]


def _angle_copies(angle: float, lower: float, upper: float) -> list[float]:
    """Enumerate equivalent angles that satisfy the configured joint limits."""
    first = math.ceil((lower - angle - _ANGLE_TOL) / math.tau)
    last = math.floor((upper - angle + _ANGLE_TOL) / math.tau)
    result = []
    for turns in range(first, last + 1):
        q = angle + turns * math.tau
        if lower - _ANGLE_TOL <= q <= upper + _ANGLE_TOL:
            result.append(min(upper, max(lower, q)))
    return result


class PlanarArm:
    """Two revolute joints, two explicit user-selected contact faces.

    forward(q, point) = R(a + q1) d + R(a + q1 + q2) point.

    ``forward(q)`` returns the elbow axis position (point=(0, 0)).
    ``forward(q, face_point)`` returns the selected contact point.  ``slider_center`` adds
    the cylinder's radius along the face normal.
    Shoulder translation is zero in this model's world frame.
    """

    def __init__(self, config: Mapping[str, Any]):
        try:
            arm = config["arm"]
            self.shoulder_static_yaw_rad = _number(
                arm["shoulder_static_yaw_rad"], "shoulder_static_yaw_rad"
            )
            self.elbow_translation_mm = _pair(
                arm["elbow_translation_mm"], "elbow_translation_mm"
            )
            # Retained for provenance; face points are already canonical.
            self.elbow_static_yaw_rad = _number(
                arm["elbow_static_yaw_rad"], "elbow_static_yaw_rad"
            )
            limits = arm["joint_limits_rad"]
            if not isinstance(limits, (list, tuple)) or len(limits) != 2:
                raise GeometryError("joint_limits_rad must contain two intervals")
            self.joint_limits_rad = tuple(
                _pair(interval, f"joint_limits_rad[{i}]")
                for i, interval in enumerate(limits)
            )
            self.stem_radius_mm = _number(
                config["slider"]["stem_radius_mm"], "stem_radius_mm"
            )
            self.faces: dict[str, dict[str, XY]] = {}
            for direction in ("+x", "-x"):
                face = config["faces"][direction]
                point = _pair(
                    face["point_canonical_xy_mm"],
                    f"faces[{direction}].point_canonical_xy_mm",
                )
                normal = _pair(
                    face["normal_canonical_xy"],
                    f"faces[{direction}].normal_canonical_xy",
                )
                length = math.hypot(*normal)
                if length <= 1e-12:
                    raise GeometryError(f"faces[{direction}] normal cannot be zero")
                if "contact_segment_canonical_xy_mm" in face:
                    a, b = (_pair(p, "contact segment") for p in face["contact_segment_canonical_xy_mm"])
                    edge = (b[0]-a[0], b[1]-a[1])
                    squared = edge[0]**2+edge[1]**2
                    if squared <= 0:
                        raise GeometryError("contact segment is empty")
                    fraction = ((point[0]-a[0])*edge[0]+(point[1]-a[1])*edge[1])/squared
                    distance = abs(edge[0]*(point[1]-a[1])-edge[1]*(point[0]-a[0]))/math.sqrt(squared)
                    if not 0 <= fraction <= 1 or distance > .05:
                        raise GeometryError(f"faces[{direction}] point outside the contact segment")
                self.faces[direction] = {
                    "point_canonical_xy_mm": point,
                    "normal_canonical_xy": (normal[0] / length, normal[1] / length),
                }
        except KeyError as exc:
            raise GeometryError(f"Missing geometry key: {exc}") from exc
        except TypeError as exc:
            raise GeometryError("Geometry sections must be mappings") from exc
        if math.hypot(*self.elbow_translation_mm) <= 1e-12:
            raise GeometryError("Shoulder-to-elbow length must be positive")
        if self.stem_radius_mm <= 0:
            raise GeometryError("stem_radius_mm must be positive")
        for lower, upper in self.joint_limits_rad:
            if lower > upper or upper - lower > 4 * math.tau:
                raise GeometryError("Joint limits must be ordered and span <= 4 turns")

    def _direction(self, direction: str) -> int:
        if direction not in self.faces:
            raise GeometryError("direction must be '+x' or '-x'")
        return 1 if direction == "+x" else -1

    def _joints(self, q: Sequence[float]) -> JointPair:
        return _pair(q, "joint angles")

    def within_limits(self, q: Sequence[float]) -> bool:
        angles = self._joints(q)
        return all(
            lo - _ANGLE_TOL <= value <= hi + _ANGLE_TOL
            for value, (lo, hi) in zip(angles, self.joint_limits_rad)
        )

    def equivalent_point(self, direction: str) -> XY:
        """Rigid canonical point representing the slider centre at contact."""
        self._direction(direction)
        face = self.faces[direction]
        point = face["point_canonical_xy_mm"]
        normal = face["normal_canonical_xy"]
        offset = self.stem_radius_mm
        return point[0] + offset * normal[0], point[1] + offset * normal[1]

    def forward(
        self,
        q: Sequence[float],
        point_canonical_xy_mm: Sequence[float] | None = None,
    ) -> XY:
        """World XY of a forearm canonical point; default is the elbow axis."""
        q1, q2 = self._joints(q)
        point = (
            (0.0, 0.0)
            if point_canonical_xy_mm is None
            else _pair(point_canonical_xy_mm, "point_canonical_xy_mm")
        )
        elbow = _rotate(self.shoulder_static_yaw_rad + q1, self.elbow_translation_mm)
        tip = _rotate(self.shoulder_static_yaw_rad + q1 + q2, point)
        return elbow[0] + tip[0], elbow[1] + tip[1]

    def slider_center(
        self, q: Sequence[float], direction: str
    ) -> XY:
        """World cylinder centre; the radius offset rotates with the arm."""
        return self.forward(q, self.equivalent_point(direction))

    def world_normal(self, q: Sequence[float], direction: str) -> XY:
        """World normal pointing from the selected face towards the slider."""
        self._direction(direction)
        q1, q2 = self._joints(q)
        return _rotate(
            self.shoulder_static_yaw_rad + q1 + q2,
            self.faces[direction]["normal_canonical_xy"],
        )

    def inverse_candidates(
        self,
        target_xy_mm: Sequence[float],
        direction: str,
        seed: Sequence[float] | None = None,
    ) -> list[IKCandidate]:
        """All solutions inside limits, including invalid push-normal signs.

        The optional seed also fixes the shoulder angle for the exceptional
        equal-length, fully folded target at the shoulder origin.
        """
        target = _pair(target_xy_mm, "target_xy_mm")
        sign = self._direction(direction)
        seed_q = self._joints(seed) if seed is not None else None
        d = self.elbow_translation_mm
        k = self.equivalent_point(direction)
        l1, l2, radius = math.hypot(*d), math.hypot(*k), math.hypot(*target)
        if l2 <= 1e-12:
            raise GeometryError("Effective contact point cannot be the elbow axis")
        cosine = (radius * radius - l1 * l1 - l2 * l2) / (2 * l1 * l2)
        if cosine < -1 - _REACH_TOL or cosine > 1 + _REACH_TOL:
            return []
        relative = math.acos(min(1.0, max(-1.0, cosine)))
        d_angle, k_angle = math.atan2(d[1], d[0]), math.atan2(k[1], k[0])
        target_angle = math.atan2(target[1], target[0])
        result: list[IKCandidate] = []
        for phi in (relative, -relative):
            q2_base = phi - k_angle + d_angle
            for q2 in _angle_copies(q2_base, *self.joint_limits_rad[1]):
                if radius <= _POSITION_TOL_MM and abs(l1 - l2) <= _POSITION_TOL_MM:
                    # At a perfectly folded singularity the shoulder is free.
                    q1_choices = [
                        min(
                            self.joint_limits_rad[0][1],
                            max(
                                self.joint_limits_rad[0][0],
                                seed_q[0] if seed_q else self.joint_limits_rad[0][0],
                            ),
                        )
                    ]
                else:
                    q1_base = (
                        target_angle
                        - self.shoulder_static_yaw_rad
                        - d_angle
                        - math.atan2(l2 * math.sin(phi), l1 + l2 * math.cos(phi))
                    )
                    q1_choices = _angle_copies(q1_base, *self.joint_limits_rad[0])
                for q1 in q1_choices:
                    q = q1, q2
                    if any(math.hypot(q1 - c.q[0], q2 - c.q[1]) < _ANGLE_TOL for c in result):
                        continue
                    actual = self.slider_center(q, direction)
                    error = math.hypot(actual[0] - target[0], actual[1] - target[1])
                    if error > _POSITION_TOL_MM:
                        continue
                    normal = self.world_normal(q, direction)
                    result.append(
                        IKCandidate(q, normal, sign * normal[0] > _NORMAL_X_TOL, error)
                    )
        return sorted(result, key=lambda candidate: candidate.q)

    def inverse(
        self,
        target_xy_mm: Sequence[float],
        direction: str,
        seed: Sequence[float] | None = None,
        require_direction: bool = True,
    ) -> JointPair:
        """Solve a world SLIDER CENTRE using the specified face and push sign.

        An optional previous joint pair chooses the closest continuous
        solution in absolute joint coordinates.  No periodic wrap is used for
        this distance because the joints have physical stops.
        """
        seed_q = self._joints(seed) if seed is not None else None
        candidates = self.inverse_candidates(target_xy_mm, direction, seed_q)
        permitted = [candidate for candidate in candidates
                     if candidate.direction_valid or not require_direction]
        if not permitted:
            if candidates:
                signs = ", ".join(f"{candidate.world_normal_xy[0]:+.6f}" for candidate in candidates)
                raise IKError(
                    f"Selected {direction} face reaches slider centre {tuple(target_xy_mm)} "
                    f"but pushes from the wrong side (normal world X: {signs})"
                )
            raise IKError(
                f"Slider centre {tuple(target_xy_mm)} has no {direction} contact solution "
                "inside configured joint limits"
            )
        if seed_q is None:
            return permitted[0].q
        return min(
            permitted,
            key=lambda candidate: sum((q - previous) ** 2 for q, previous in zip(candidate.q, seed_q)),
        ).q
