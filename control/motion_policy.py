"""Feedback classification and bounded speed-before-angle recovery in move."""
import math

from .recovery import SpeedRecovery


DEFAULTS = {
    'enabled': False, 'max_downshifts': 2, 'max_angle_changes': 3,
    'angle_step_deg': .2, 'angle_max_deg': .6,
    'angle_steps_deg': None, 'final_speed_deg_s': None,
    'max_recovery_rounds': 1,
    'overshoot_return_enabled': False, 'overshoot_return_speed_deg_s': 18.,
    'bias_retraction_speed_deg_s': None,
    'bias_retraction_requires_crossing': False,
    'axis_angle_bias': {},
    'cartesian_step_mm': .2, 'cartesian_max_mm': .4,
    'ordinary_speed_steps_deg_s': [6., 12., 18.],
    'downshift_steps_deg_s': [2., 6., 18., 30.],
}


def motion_policy_settings(hardware):
    raw = hardware.get('motion_recovery', {})
    if not isinstance(raw, dict) or set(raw)-set(DEFAULTS):
        raise ValueError('invalid motion_recovery settings')
    result = {**DEFAULTS, **raw}
    if (type(result['enabled']) is not bool or type(result['overshoot_return_enabled']) is not bool
            or type(result['bias_retraction_requires_crossing']) is not bool):
        raise ValueError('motion_recovery.enabled must be bool')
    for key in ('max_downshifts', 'max_angle_changes', 'max_recovery_rounds'):
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
    if result['max_recovery_rounds'] > 3:
        raise ValueError('motion_recovery.max_recovery_rounds must be 1..3')
    speed = result['overshoot_return_speed_deg_s']
    if type(speed) not in (int,float) or not math.isfinite(speed) or not hardware.get('min_speed_deg_s',2.) <= speed <= 18.:
        raise ValueError('invalid motion_recovery.overshoot_return_speed_deg_s')
    recall_speed = result['bias_retraction_speed_deg_s']
    if recall_speed is not None and (type(recall_speed) not in (int, float)
            or not math.isfinite(recall_speed)
            or not hardware.get('min_speed_deg_s', 2.) <= recall_speed <= 18.):
        raise ValueError('invalid motion_recovery.bias_retraction_speed_deg_s')
    rules = result['axis_angle_bias']
    names = {m['name'] for m in hardware.get('motors', [])}
    if not isinstance(rules, dict) or set(rules)-names:
        raise ValueError('invalid motion_recovery.axis_angle_bias')
    for rule in rules.values():
        if not isinstance(rule, dict) or set(rule) != {'below_model_deg', 'max_deg'}:
            raise ValueError('invalid axis angle-bias rule')
        for key, maximum in (('below_model_deg', 360.), ('max_deg', result['angle_max_deg'])):
            value = rule[key]
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
                raise ValueError('invalid axis angle-bias '+key)
    return result


def angle_bias_profile(policy, motor, actual_deg, nominal_deg):
    """Model joint degrees, independent of motor offset/sign/reduction."""
    rule = policy['axis_angle_bias'].get(motor['name'])
    cap = policy['angle_max_deg'] if rule is None else rule['max_deg']
    allowed = rule is None or max(actual_deg, nominal_deg) < rule['below_model_deg']-1e-9
    levels = policy['angle_steps_deg']
    if levels is not None:
        levels = [v for v in levels if v <= cap+1e-9]
        if not levels or levels[-1] < cap-1e-9:
            levels.append(cap)
    return allowed, cap, levels


class MotionRecovery(SpeedRecovery):
    """A bounded budget per waypoint; speed/bias steps belong to one round.

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
        self.node_index = 0
        self.round_index = 0
        self.total_rounds_started = 0
        self.move_recovery_started_at = None
        self.returning = [False]*n
        self.return_attempted = [False]*n
        self.return_written_at = [-math.inf]*n
        self.nominal_goals_deg = [None]*n
        self.bias_allowed = [True]*n
        self.bias_caps_deg = [self.policy['angle_max_deg']]*n

    def eligible(self, axis):
        return self.policy['enabled'] and self.phase in (
            'free', 'approach', 'push', 'push_correction', 'withdraw') and self.directions[axis] != 0

    def profiles(self, base_cap):
        floors, tiers = list(self.axis_min_speed_deg_s), list(self.speed_tier)
        caps = [max(base_cap, floor) if tier > 0 else (floor if tier < 0 else base_cap)
                for floor, tier in zip(floors, tiers)]
        return floors, tiers, caps

    def event(self, now, event, axis=None, **values):
        values.setdefault('waypoint_index', self.node_index)
        values.setdefault('recovery_round', max(1, self.round_index))
        super().event(now, event, axis, **values)

    def _begin(self, now):
        if self.move_recovery_started_at is None:
            self.move_recovery_started_at = now
        super()._begin(now)

    def _reset_round_actions(self, now):
        n = len(self.motors)
        for i in range(n):
            self.cancel_pending(i, now, 'recovery_round_changed')
            self.history[i].clear()
        self.upshifts = [0]*n
        self.downshifts = [0]*n
        self.downshifted = [False]*n
        self.angle_changes = [0]*n
        self.no_gain_windows = [0]*n
        self.final_speed_ready = [False]*n
        self.final_speed_used = [False]*n
        self.evaluation_at = [now]*n
        self.returning = [False]*n
        self.return_attempted = [False]*n
        self.return_written_at = [-math.inf]*n

    def node_changed(self, now, node_index=None):
        super().node_changed(now)
        self.node_index = self.node_index+1 if node_index is None else node_index
        self._reset_round_actions(now)
        self.diagnoses = ['normal']*len(self.motors)
        self.last_bias = [0.]*len(self.motors)
        self.round_index = 0
        self.started_at = self.deadline = None
        final = self.policy['final_speed_deg_s']
        if final is not None:
            steps = self.settings['speed_steps_deg_s']
            for i, floor in enumerate(self.axis_min_speed_deg_s):
                if floor >= final-.005:
                    self.axis_min_speed_deg_s[i] = steps[-2]
                    self.speed_tier[i] = len(steps)-1
        self.event(now, 'waypoint_recovery_budget_reset', recovery_round=0,
                   max_rounds=self.policy['max_recovery_rounds'])

    def _round_written(self, at):
        if self.round_index == 0:
            self.round_index = 1
            self.total_rounds_started += 1
            self.event(at, 'node_recovery_round_started')

    def nominal_retry_axes(self):
        return [r is not None and r['kind'] == 'round_restart' for r in self.pending]

    def nominal_return_axes(self):
        return [r is not None and r['kind'] == 'overshoot_return' for r in self.pending]

    def node_ready(self):
        return not any(self.returning) and not any(self.nominal_return_axes())

    def _angle_window_still(self, axis, state, window):
        stamp = state['axis_stamp_monotonic_s'][axis]
        points = list(self.history[axis])
        start = 0
        while start+1 < len(points) and points[start+1][0] <= stamp-window+1e-9:
            start += 1
        points = points[start:]
        return (len(points) >= 2 and points[-1][0]-points[0][0] >= .8*window
                and max(e for _,e in points)-min(e for _,e in points) <= self.settings['progress_deg']+1e-9
                and abs(math.degrees(state['speed_rad_s'][axis])) <= self.still_speed)

    def _request_return(self, axis, now, bias, next_round=None):
        number = max(1,self.round_index) if next_round is None else next_round
        self._request(axis, now, -number, self.policy['overshoot_return_speed_deg_s'], 'overshoot_return')
        self.pending[axis].update(bias_before=bias, diagnosis='overshoot',
                                  nominal_goal_deg=self.nominal_goals_deg[axis])
        if next_round is not None:
            self.pending[axis]['next_round'] = next_round

    def _retry_or_exhaust(self, axis, now, bias):
        if max(1, self.round_index) >= self.policy['max_recovery_rounds']:
            return axis
        # Retain the achieved ordinary floor. The final 90 tier is recalled
        # to the ordinary ceiling before any new biased command is allowed.
        if self.diagnoses[axis] == 'overshoot':
            for i in range(len(self.motors)):
                self.cancel_pending(i, now, 'recovery_round_retry')
            self._request_return(axis, now, bias, max(1,self.round_index)+1)
            self.event(now, 'node_recovery_round_retry_requested', axis,
                       next_round=self.pending[axis]['next_round'])
            return None
        steps = self.settings['speed_steps_deg_s']
        ordinary = [v for v in steps if v != self.policy['final_speed_deg_s']]
        floor = min(max(self.base_cap, self.axis_min_speed_deg_s[axis]), ordinary[-1])
        tier = steps.index(floor)+1 if floor in steps else 0
        for i in range(len(self.motors)):
            self.cancel_pending(i, now, 'recovery_round_retry')
        self._request(axis, now, tier, floor, 'round_restart')
        self.pending[axis].update(bias_before=bias, next_round=max(1,self.round_index)+1,
                                  diagnosis=self.diagnoses[axis])
        self.event(now, 'node_recovery_round_retry_requested', axis,
                   next_round=self.pending[axis]['next_round'])
        return None

    def angle_modes(self):
        return [0 if request is None else
                (1 if request['kind'] == 'angle_up' else (-1 if request['kind'] == 'angle_down' else
                 (-2 if request['kind'] in ('bias_clear','round_restart','overshoot_return') else 0)))
                for request in self.pending]

    def classify(self, axis, now, state, goal, tolerance, progress_at):
        # Position reversals with measurable amplitude take precedence over
        # no-progress. Repeated wobble must not be mistaken for useful travel.
        signed_error = math.degrees(state['q_rad'][axis]-goal[axis])
        if (self.policy['overshoot_return_enabled'] and self.returning[axis]
                and abs(signed_error) > tolerance+1e-9):
            return 'overshoot'
        if self.oscillating(axis):
            return 'oscillating'
        if (self.policy['overshoot_return_enabled']
                and self.directions[axis]*signed_error > tolerance+1e-9):
            return 'overshoot'
        error = abs(signed_error)
        if (error > tolerance+1e-9 and now-progress_at >= self.settings['trigger_s']
                and abs(math.degrees(state['speed_rad_s'][axis])) <= self.still_speed):
            return 'stuck'
        return 'normal'

    def update(self, now, state, goal, tolerances, progress_at, active, keys, biases):
        self.nominal_goals_deg = [math.degrees(q) for q in goal]
        self.observe_goal(state, goal)
        for axis in active:
            allowed, cap, _ = angle_bias_profile(self.policy, self.motors[axis],
                math.degrees(state['q_rad'][axis]), math.degrees(goal[axis]))
            self.bias_allowed[axis], self.bias_caps_deg[axis] = allowed, cap
            if not self.eligible(axis):
                continue
            request = self.pending[axis]
            if not allowed and request is not None and request['kind'].startswith('angle_'):
                self.cancel_pending(axis, now, 'angle_bias_disabled_for_pose')
            if (self.returning[axis] and abs(math.degrees(state['q_rad'][axis]-goal[axis])) <= tolerances[axis]+1e-9
                    and state['axis_stamp_monotonic_s'][axis] >= self.return_written_at[axis]+.1
                    and self._angle_window_still(axis, state, .1)):
                self.returning[axis] = False
                self.recovering[axis] = False
                self.event(now, 'overshoot_return_completed', axis,
                           actual_q_deg=math.degrees(state['q_rad'][axis]))
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
                if (request is not None and request['kind'] == 'overshoot_return'
                        and abs(math.degrees(state['q_rad'][axis]-goal[axis])) <= tolerances[axis]+1e-9):
                    self.cancel_pending(axis, now, 'nominal_band_recovered_before_return_write')
                if (request is not None and request['kind'] in ('up','angle_up','round_restart')
                        and progress_at[axis] > request['at']+1e-9):
                    self.cancel_pending(axis, now, 'fresh_progress_before_write')
                continue
            self.recovering[axis] = True
            if self.deadline is not None and now >= self.deadline:
                return axis
            if any(r is not None and 'next_round' in r for r in self.pending):
                continue
            if self.pending[axis] is not None or now-self.evaluation_at[axis] < self.settings['observe_s']:
                continue
            key = keys[axis]
            if key is None:
                # There is no known issued target to tune safely.
                continue
            actual_speed = key[2]/100.
            if diagnosis == 'overshoot':
                if not self.return_attempted[axis]:
                    self._request_return(axis, now, biases[axis])
                    self.pending[axis]['nominal_goal_deg'] = math.degrees(goal[axis])
                elif self._angle_window_still(axis, state, self.settings['trigger_s']):
                    self._begin(now)
                    self.evaluation_at[axis] = now
                    self.no_gain_windows[axis] += 1
                    self.event(now, 'overshoot_return_no_gain', axis, windows=self.no_gain_windows[axis])
            elif diagnosis == 'stuck':
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
                    if not allowed and abs(biases[axis]) < .005:
                        self._request(axis, now, len(steps), final, 'up')
                        self.event(now, 'final_speed_requested_without_angle_bias', axis)
                    elif self.final_speed_ready[axis] and abs(biases[axis]) < .005:
                        self._request(axis, now, len(steps), final, 'up')
                    elif abs(biases[axis]) >= cap*self.motors[axis]['ratio']-.005:
                        if self.policy['bias_retraction_requires_crossing']:
                            # Keep the certified drive target. Recalling it on
                            # the near side just to unlock 90 reintroduces the
                            # old/new/actual direction conflict.
                            self._request(axis, now, len(steps), final, 'up')
                            self.event(now, 'final_speed_requested_keep_bias', axis,
                                       bias_motor_deg=biases[axis])
                        else:
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
                    tier = -(self.downshifts[axis]+1)
                    if tier == self.speed_tier[axis]:
                        tier = -4 if tier != -4 else -3
                    self._request(axis, now, tier, max(candidates), 'settle')
                else:
                    self._angle_or_exhaust(axis, now, biases[axis], 'angle_down')
            if self.pending[axis] is not None:
                self.pending[axis]['diagnosis'] = diagnosis
            if self.no_gain_windows[axis] >= self.settings['max_no_gain_windows']:
                return self._retry_or_exhaust(axis, now, biases[axis])
        return None

    def _angle_or_exhaust(self, axis, now, bias, kind):
        # A waypoint can enter the permitted region after a speed-only 90
        # recovery above the threshold. Allow its first small bias there,
        # without re-injecting a bias after an already completed bias cycle.
        first_zone_bias = (self.motors[axis]['name'] in self.policy['axis_angle_bias']
                           and self.angle_changes[axis] == 0 and abs(bias) < .005)
        angle_room = (kind != 'angle_up'
                      or abs(bias) < self.bias_caps_deg[axis]*self.motors[axis]['ratio']-.005)
        if (self.bias_allowed[axis] and self.angle_changes[axis] < self.policy['max_angle_changes']
                and angle_room
                and not (kind == 'angle_up' and self.final_speed_used[axis] and not first_zone_bias)):
            self._request(axis, now, self.speed_tier[axis], self.axis_min_speed_deg_s[axis], kind)
            self.pending[axis]['bias_before'] = bias
        else:
            self._begin(now)
            self.evaluation_at[axis] = now
            self.no_gain_windows[axis] += 1
            self.event(now, 'angle_ceiling_no_gain', axis, windows=self.no_gain_windows[axis])

    def wrote(self, axis, tier, speed, at, bias=None):
        request = self.pending[axis]
        if request is not None and request['kind'] == 'overshoot_return':
            if (bias is None or abs(bias) >= .005 or at < request['at']
                    or tier != request['tier'] or speed > self.policy['overshoot_return_speed_deg_s']+.005):
                return False
            next_round = request.get('next_round')
            self.pending[axis] = None
            if next_round is not None:
                self._reset_round_actions(at)
                self.round_index = next_round
                self.total_rounds_started += 1
                self.event(at, 'node_recovery_round_started', axis)
            else:
                self._round_written(at)
            self.returning[axis] = self.return_attempted[axis] = True
            self.return_written_at[axis] = self.evaluation_at[axis] = at
            self.last_bias[axis] = 0.
            self.history[axis].clear()
            self.event(at, 'overshoot_return_written', axis, speed_deg_s=speed,
                       nominal_goal_deg=request.get('nominal_goal_deg'), bias_motor_deg=0.)
            return True
        if request is not None and request['kind'] == 'round_restart':
            if bias is None or abs(bias) >= .005 or at < request['at']:
                return False
            next_round = request['next_round']
            self.pending[axis] = None
            self._reset_round_actions(at)
            self.round_index = next_round
            self.total_rounds_started += 1
            self.last_bias[axis] = 0.
            self.event(at, 'node_recovery_round_started', axis)
            return True
        if request is not None and request['kind'] == 'bias_clear':
            if bias is None or abs(bias) >= .005 or at < request['at']:
                return False
            self._round_written(at)
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
            self._round_written(at)
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
        if committed:
            self._round_written(at)
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
                'axis_angle_bias_allowed': list(self.bias_allowed),
                'axis_angle_bias_caps_deg': list(self.bias_caps_deg),
                'final_speed_deg_s': self.policy['final_speed_deg_s'],
                'final_speed_used': list(self.final_speed_used),
                'waypoint_index': self.node_index, 'recovery_round': self.round_index,
                'max_recovery_rounds': self.policy['max_recovery_rounds'],
                'total_rounds_started': self.total_rounds_started,
                'recovery_budget_scope': 'waypoint',
                'move_recovery_started_at': self.move_recovery_started_at,
                'overshoot_return_active': list(self.returning),
                'overshoot_return_attempted': list(self.return_attempted)}


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
