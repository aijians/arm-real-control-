"""Feedback classification and bounded speed-before-angle recovery in move."""
import math

from .recovery import SpeedRecovery


DEFAULTS = {
    'enabled': False, 'max_downshifts': 2, 'max_angle_changes': 3,
    'angle_step_deg': .2, 'angle_max_deg': .6,
    'angle_steps_deg': None, 'final_speed_deg_s': None,
    'cartesian_step_mm': .2, 'cartesian_max_mm': .4,
    'ordinary_speed_steps_deg_s': [6., 12., 18.],
    'downshift_steps_deg_s': [2., 6., 18., 30.],
}


def motion_policy_settings(hardware):
    raw = hardware.get('motion_recovery', {})
    if not isinstance(raw, dict) or set(raw)-set(DEFAULTS):
        raise ValueError('invalid motion_recovery settings')
    result = {**DEFAULTS, **raw}
    if type(result['enabled']) is not bool:
        raise ValueError('motion_recovery.enabled must be bool')
    for key in ('max_downshifts', 'max_angle_changes'):
        if type(result[key]) is not int or not 1 <= result[key] <= 4:
            raise ValueError('invalid motion_recovery.'+key)
    for key, maximum in (('angle_step_deg', .4), ('angle_max_deg', 1.2),
                         ('cartesian_step_mm', .3), ('cartesian_max_mm', .5)):
        value = result[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
            raise ValueError('invalid motion_recovery.'+key)
    if (result['angle_step_deg'] > result['angle_max_deg']
            or result['cartesian_step_mm'] > result['cartesian_max_mm']):
        raise ValueError('motion_recovery step exceeds cap')
    for key, maximum in (('ordinary_speed_steps_deg_s', 18.), ('downshift_steps_deg_s', 60.)):
        values = result[key]
        if (not isinstance(values, list) or not 1 <= len(values) <= 4
                or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 or v > maximum for v in values)
                or any(a >= b for a,b in zip(values, values[1:]))):
            raise ValueError('invalid motion_recovery.'+key)
    levels = result['angle_steps_deg']
    if levels is not None and (not isinstance(levels, list) or not 1 <= len(levels) <= 4
            or any(type(v) not in (int, float) or not math.isfinite(v)
                   or not 0 < v <= result['angle_max_deg'] for v in levels)
            or any(a >= b for a,b in zip(levels, levels[1:]))):
        raise ValueError('invalid motion_recovery.angle_steps_deg')
    final = result['final_speed_deg_s']
    steps = hardware.get('speed_recovery', {}).get('speed_steps_deg_s', [18.,30.])
    if final is not None and (type(final) not in (int,float) or not math.isfinite(final)
            or final != 90. or final != steps[-1]
            or levels is None or abs(levels[-1]-result['angle_max_deg']) > 1e-9
            or result['max_angle_changes'] < len(levels)):
        raise ValueError('invalid motion_recovery.final_speed_deg_s')
    return result


class MotionRecovery(SpeedRecovery):
    """One budget per move; per-axis diagnosis precedes every intervention.

    All active axes, directions and phases share the same configured tiers.
    Legacy ordinary/cartesian-mm keys are read for snapshot compatibility;
    common speed_steps and angle_step/angle_max drive the current policy.
    Angles here are requests, committed only after a changed target is written.
    """
    def __init__(self, settings, motors, phase, policy, limits=(2., 18., 1.)):
        super().__init__(settings, motors, phase)
        self.policy = policy
        self.minimum_speed, self.base_cap, self.still_speed = limits
        n = len(motors)
        self.diagnoses = ['normal']*n
        self.downshifts = [0]*n
        self.angle_changes = [0]*n
        self.last_bias = [0.]*n
        self.final_speed_ready = [False]*n
        self.final_speed_used = [False]*n

    def eligible(self, axis):
        return self.policy['enabled'] and self.phase in (
            'free', 'approach', 'push', 'push_correction', 'withdraw') and self.directions[axis] != 0

    def profiles(self, base_cap):
        floors, tiers = list(self.axis_min_speed_deg_s), list(self.speed_tier)
        caps = [max(base_cap, floor) if tier > 0 else (floor if tier < 0 else base_cap)
                for floor, tier in zip(floors, tiers)]
        return floors, tiers, caps

    def node_changed(self, now):
        super().node_changed(now)
        self.diagnoses = ['normal']*len(self.motors)
        self.final_speed_ready = [False]*len(self.motors)

    def angle_modes(self):
        return [0 if request is None else
                (1 if request['kind'] == 'angle_up' else (-1 if request['kind'] == 'angle_down' else
                 (-2 if request['kind'] == 'bias_clear' else 0)))
                for request in self.pending]

    def classify(self, axis, now, state, goal, tolerance, progress_at):
        # Position reversals with measurable amplitude take precedence over
        # no-progress. Repeated wobble must not be mistaken for useful travel.
        if self.oscillating(axis):
            return 'oscillating'
        error = abs(math.degrees(state['q_rad'][axis]-goal[axis]))
        if (error > tolerance+1e-9 and now-progress_at >= self.settings['trigger_s']
                and abs(math.degrees(state['speed_rad_s'][axis])) <= self.still_speed):
            return 'stuck'
        return 'normal'

    def update(self, now, state, goal, tolerances, progress_at, active, keys, biases):
        self.observe_goal(state, goal)
        for axis in active:
            if not self.eligible(axis):
                continue
            diagnosis = self.classify(axis, now, state, goal, tolerances[axis], progress_at[axis])
            previous = self.diagnoses[axis]
            if diagnosis != previous:
                self.diagnoses[axis] = diagnosis
                if diagnosis != 'normal':
                    errors = [value for _,value in self.history[axis]]
                    self.event(now, 'motion_diagnosed', axis, diagnosis=diagnosis,
                               actual_q_deg=math.degrees(state['q_rad'][axis]),
                               nominal_target_q_deg=math.degrees(goal[axis]),
                               window_error_min_deg=min(errors), window_error_max_deg=max(errors),
                               since_progress_s=now-progress_at[axis])
                request = self.pending[axis]
                if request is not None and diagnosis != 'normal' and request.get('diagnosis') != diagnosis:
                    self.cancel_pending(axis, now, 'diagnosis_changed')
            if diagnosis == 'normal':
                request = self.pending[axis]
                if (request is not None and request['kind'] in ('up','angle_up')
                        and progress_at[axis] > request['at']+1e-9):
                    self.cancel_pending(axis, now, 'fresh_progress_before_write')
                continue
            self.recovering[axis] = True
            if self.deadline is not None and now >= self.deadline:
                return axis
            if self.pending[axis] is not None or now-self.evaluation_at[axis] < self.settings['observe_s']:
                continue
            key = keys[axis]
            if key is None:
                # There is no known issued target to tune safely.
                continue
            actual_speed = key[2]/100.
            if diagnosis == 'stuck':
                steps = self.settings['speed_steps_deg_s']
                final = self.policy['final_speed_deg_s']
                candidates = [(tier, speed) for tier, speed in enumerate(steps, 1)
                              if speed > actual_speed+.005 and speed != final]
                if candidates and self.upshifts[axis] < self.settings['max_upshifts']:
                    tier, speed = candidates[0]
                    self._request(axis, now, tier, speed, 'up')
                elif final is not None and not self.final_speed_used[axis]:
                    # The last tier is gated by an actually written maximum
                    # bias, a no-progress window, and a separate nominal recall.
                    if self.final_speed_ready[axis] and abs(biases[axis]) < .005:
                        self._request(axis, now, len(steps), final, 'up')
                    elif abs(biases[axis]) >= self.policy['angle_max_deg']*self.motors[axis]['ratio']-.005:
                        self._request(axis, now, self.speed_tier[axis],
                                      self.axis_min_speed_deg_s[axis], 'bias_clear')
                        self.pending[axis]['bias_before'] = biases[axis]
                    else:
                        self._angle_or_exhaust(axis, now, biases[axis], 'angle_up')
                else:
                    self._angle_or_exhaust(axis, now, biases[axis], 'angle_up')
            else:
                candidates = [speed for speed in self.policy['downshift_steps_deg_s']
                              if self.minimum_speed <= speed < actual_speed-.005]
                if candidates and self.downshifts[axis] < self.policy['max_downshifts']:
                    self._request(axis, now, -(self.downshifts[axis]+1), max(candidates), 'settle')
                else:
                    self._angle_or_exhaust(axis, now, biases[axis], 'angle_down')
            if self.pending[axis] is not None:
                self.pending[axis]['diagnosis'] = diagnosis
            if self.no_gain_windows[axis] >= self.settings['max_no_gain_windows']:
                return axis
        return None

    def _angle_or_exhaust(self, axis, now, bias, kind):
        if (self.angle_changes[axis] < self.policy['max_angle_changes']
                and not (kind == 'angle_up' and self.final_speed_used[axis])):
            self._request(axis, now, self.speed_tier[axis], self.axis_min_speed_deg_s[axis], kind)
            self.pending[axis]['bias_before'] = bias
        else:
            self._begin(now)
            self.evaluation_at[axis] = now
            self.no_gain_windows[axis] += 1
            self.event(now, 'angle_ceiling_no_gain', axis, windows=self.no_gain_windows[axis])

    def wrote(self, axis, tier, speed, at, bias=None):
        request = self.pending[axis]
        if request is not None and request['kind'] == 'bias_clear':
            if bias is None or abs(bias) >= .005 or at < request['at']:
                return False
            self.last_bias[axis] = 0.
            self.pending[axis] = None
            self.final_speed_ready[axis] = True
            # Nominal recall was already preceded by a full no-progress
            # window. Re-evaluate feedback before the next (separate) write.
            self.evaluation_at[axis] = at-self.settings['observe_s']
            self.history[axis].clear()
            self.event(at, 'bias_cleared_before_final_speed', axis,
                       bias_motor_deg=0., previous_bias_motor_deg=request['bias_before'])
            return True
        if request is not None and request['kind'].startswith('angle_'):
            if bias is None or at < request['at'] or abs(bias-request['bias_before']) < .005:
                return False
            self.angle_changes[axis] += 1
            self.last_bias[axis] = bias
            self.pending[axis] = None
            self.evaluation_at[axis] = at
            self.history[axis].clear()
            self.event(at, 'angle_change_written', axis, kind=request['kind'],
                       bias_motor_deg=bias, diagnosis=request.get('diagnosis'))
            return True
        down = request is not None and request['kind'] == 'settle'
        committed = super().wrote(axis, tier, speed, at)
        if committed and down:
            self.downshifts[axis] += 1
        if committed and self.policy['final_speed_deg_s'] is not None and speed >= self.policy['final_speed_deg_s']-.005:
            self.final_speed_used[axis] = True
            self.final_speed_ready[axis] = False
            self.event(at, 'final_speed_written', axis, speed_deg_s=speed)
        if committed:
            self.history[axis].clear()
        return committed

    def snapshot(self):
        return {**super().snapshot(), 'policy': 'classify_speed_then_angle',
                'diagnoses': list(self.diagnoses), 'downshifts': list(self.downshifts),
                'angle_changes': list(self.angle_changes), 'angle_bias_motor_deg': list(self.last_bias),
                'angle_steps_deg': self.policy['angle_steps_deg'],
                'final_speed_deg_s': self.policy['final_speed_deg_s'],
                'final_speed_used': list(self.final_speed_used)}


def contact_angle_step(target_for_lead, nominal, current_lead, modes,
                       step_deg, max_deg, backward_mm, forward_mm):
    """Choose the first along-line lead giving <=step additional joint bias.

    The infinity norm bounds *both* joint offsets, including when only one
    joint requests assistance. The original line endpoints bound the search.
    Reduction retracts existing forward lead before requesting backward lead.
    """
    def magnitude(lead):
        return max(abs(math.degrees(a-b)) for a,b in zip(target_for_lead(lead),nominal))
    decrease = any(mode < 0 for mode in modes)
    if decrease and current_lead > 1e-9:
        goal=max(0.,magnitude(current_lead)-step_deg)
        low,high=0.,current_lead
        for _ in range(32):
            mid=(low+high)/2
            if magnitude(mid) <= goal:low=mid
            else:high=mid
        return low
    sign=-1 if decrease else 1
    previous=max(0.,sign*current_lead)
    limit=backward_mm if decrease else forward_mm
    goal=min(max_deg,magnitude(sign*previous)+step_deg)
    if limit <= 1e-9 or goal <= 1e-9:return 0.
    low=0.
    # Small local brackets avoid assuming a whole contact line is monotonic.
    for i in range(1,math.ceil(limit/.05)+1):
        high=min(limit,i*.05)
        if magnitude(sign*high) >= goal:
            for _ in range(32):
                mid=(low+high)/2
                if magnitude(sign*mid)<=goal:low=mid
                else:high=mid
            return sign*low
        low=high
    return sign*limit
