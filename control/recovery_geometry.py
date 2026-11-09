"""Incremental, move-scoped recovery envelopes at the existing 0.1 deg resolution."""
from collections import OrderedDict
import math
import time


class RecoveryGeometry:
    """A sampled envelope, never a cached assertion about an unbounded path.

    Certificates cover the moving axis interval and the other axis band. Point
    results belong to one immutable scene/contact context and one move only.
    None means pending: the caller must return to feedback, not wait here.
    """
    STEP_DEG = .1
    MAX_POINTS = 8192
    MAX_CERTIFICATES = 64

    def __init__(self, motors, feedback, last_targets, check, nominal_targets=None):
        self.motors = motors
        self.feedback = feedback
        self.last_targets = last_targets
        self.check = check
        self.nominal_targets = nominal_targets
        self.points = OrderedDict()
        self.certificates = OrderedDict()
        self.checked = self.hits = self.pending = 0
        self.longest_check_s = 0.

    def _model(self, axis, motor_deg):
        m = self.motors[axis]
        return (motor_deg-m['offset_deg'])/(m['sign']*m['ratio'])

    def _grid(self, axis, low, high):
        limits = self.motors[axis]['limits_deg']
        low = max(limits[0], math.floor((low-1e-8)/self.STEP_DEG)*self.STEP_DEG)
        high = min(limits[1], math.ceil((high+1e-8)/self.STEP_DEG)*self.STEP_DEG)
        return low, high

    def _values(self, axis, low, high):
        values = [low, high]
        values.extend(k*self.STEP_DEG for k in range(
            math.ceil(low/self.STEP_DEG), math.floor(high/self.STEP_DEG)+1))
        return sorted(set(round(v, 9) for v in values))

    def _envelope(self, axis, target_centi):
        other = 1-axis
        actual = [self._model(i, self.feedback[i]['motor_deg']) for i in (0, 1)]
        if any(not self.motors[i]['limits_deg'][0] <= value <= self.motors[i]['limits_deg'][1]
               for i, value in enumerate(actual)):
            return None
        desired = self._model(axis, target_centi/100.)
        if not self.motors[axis]['limits_deg'][0] <= desired <= self.motors[axis]['limits_deg'][1]:
            return None
        key_other = self.last_targets[other]
        commanded = actual[other] if key_other is None else self._model(other, key_other[0]/100.)
        band = self._grid(other, min(actual[other], commanded), max(actual[other], commanded))
        interval = self._grid(axis, min(actual[axis], desired), max(actual[axis], desired))
        key = (axis, target_centi, band)
        return key, interval, desired

    def contains(self, axis, target_centi):
        envelope = self._envelope(axis, target_centi)
        if envelope is None:
            return False
        key, interval, desired = envelope
        # A different target can share a completed rectangle only when its
        # exact target line was also checked (nominal is warmed with each bias).
        for certified, bounds in reversed(self.certificates.items()):
            if certified[0] != axis or certified[2] != key[2]:
                continue
            if not bounds[0] <= interval[0] or not interval[1] <= bounds[1]:
                continue
            for value in self._values(1-axis, *key[2]):
                point = [0., 0.]
                point[axis], point[1-axis] = round(desired, 9), value
                if not self.points.get(tuple(point), False):
                    break
            else:
                self.hits += 1
                return True
        return False

    def __call__(self, axis, target_centi):
        envelope = self._envelope(axis, target_centi)
        if envelope is None:
            return False
        key, interval, desired = envelope
        band = key[2]
        other = 1-axis
        if self.contains(axis, target_centi):
            return True
        old = self.certificates.get(key)
        if old is not None and old[0] <= interval[0] and interval[1] <= old[1]:
            self.certificates.move_to_end(key)
            self.hits += 1
            return True
        moving = self._values(axis, *interval)
        # Include the exact new target in addition to the fixed sampling grid.
        moving = sorted(set(moving + [round(desired, 9)]))
        if self.nominal_targets is not None:
            nominal = self._model(axis, self.nominal_targets()[axis]/100.)
            if interval[0] <= nominal <= interval[1]:
                moving = sorted(set(moving + [round(nominal, 9)]))
        others = self._values(other, *band)
        if len(moving)*len(others) > self.MAX_POINTS:
            return False
        # At most one expensive scene call per invocation. Even cold cache
        # construction yields to the regular feedback/abort loop after it.
        did_check = False
        for a in moving:
            for b in others:
                q = [0., 0.]
                q[axis], q[other] = a, b
                point = tuple(q)
                if point not in self.points:
                    if did_check:
                        self.pending += 1
                        return None
                    began = time.monotonic()
                    safe = self.check([math.radians(v) for v in point])
                    self.longest_check_s = max(self.longest_check_s, time.monotonic()-began)
                    self.points[point] = safe
                    self.checked += 1
                    did_check = True
                    if len(self.points) > self.MAX_POINTS:
                        self.points.popitem(last=False)
                if not self.points[point]:
                    return False
                self.points.move_to_end(point)
        self.certificates[key] = interval
        self.certificates.move_to_end(key)
        if len(self.certificates) > self.MAX_CERTIFICATES:
            self.certificates.popitem(last=False)
        return True

    def snapshot(self):
        return {'checked_points': self.checked, 'certificate_hits': self.hits,
                'pending_yields': self.pending, 'longest_scene_check_s': self.longest_check_s,
                'cached_points': len(self.points), 'cached_certificates': len(self.certificates)}
