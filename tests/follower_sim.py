"""Closed-loop simulation of RailFollower against a non-uniform load (tests only).

Plant (model degrees): each axis needs a lag ``lag_fn(axis, q)`` before it
moves (static friction + gravity); it then tracks ``target - lag`` at most at
its A6 speed field. Feedback is sampled at 20 Hz, control runs at 50 Hz, and
writes pass an emulation of MotorPair's A6 direction gate.
"""
import math

from control.follower import RailFollower


class LoadedPlant:
    def __init__(self, q0, lag_fn):
        self.q = list(map(float, q0))
        self.v = [0., 0.]
        self.target = list(self.q)
        self.speed = [5., 5.]
        self.lag_fn = lag_fn

    def step(self, dt):
        for i in range(2):
            lag = max(0., self.lag_fn(i, self.q[i]))
            e = self.target[i] - self.q[i]
            if abs(e) <= lag + 1e-12:
                self.v[i] = 0.
                continue
            goal = self.target[i] - math.copysign(lag, e)
            step = math.copysign(min(abs(goal - self.q[i]), self.speed[i] * dt), e)
            self.q[i] += step
            self.v[i] = step / dt


def simulate(path, mode, lag_fn, settings=None, *, t_max=60., control_dt=.02,
             feedback_dt=.05, physics_dt=.005, end_tolerance=.3):
    follower = RailFollower(path, mode, settings)
    plant = LoadedPlant(path[0], lag_fn)
    written = list(map(float, path[0]))
    fb_q, fb_v, fb_t = list(plant.q), [0., 0.], 0.
    still_since = [0., 0.]
    trace, unsafe, steps = [], [], []
    t, result = 0., None
    while t < t_max:
        if t - fb_t >= feedback_dt - 1e-9 or t == 0.:
            fb_q, fb_v, fb_t = list(plant.q), list(plant.v), t
            for i in range(2):
                if abs(fb_v[i]) > 1e-9:
                    still_since[i] = None
                elif still_since[i] is None:
                    still_since[i] = t
        out = follower.update(t, fb_q, written, fb_v)
        for i in out.get("rebased_axes", []):
            written[i] = fb_q[i]            # MotorPair re-anchors its step check here
        if out["failure"]:
            result = dict(out["failure"], status="failed")
            break
        for i in range(2):
            want = out["targets_deg"][i]
            if abs(want - written[i]) <= 1e-12:
                continue
            age = t - fb_t + .0012
            stationary = still_since[i] is not None and t - still_since[i] >= .1
            bound = (1. if stationary else max(out["speeds_deg_s"][i], abs(fb_v[i]))) * age + .015
            if abs(want - fb_q[i]) <= bound:
                continue                                   # MotorPair would skip this write
            direction = 1 if want > fb_q[i] else -1
            steps.append((i, want - written[i]))
            written[i] = want
            plant.target[i] = want
            plant.speed[i] = out["speeds_deg_s"][i]
            # A6 direction byte check against the *true* position at arrival.
            if direction * (want - plant.q[i]) < -1e-9:
                unsafe.append((t, i, plant.q[i], want, direction))
        trace.append({"t": t, "q": list(plant.q), "fb": list(fb_q), "written": list(written),
                      "segment": out["segment"], "lead": out["lead_deg"],
                      "lead_state": out["lead_state_deg"], "master": out["master"]})
        if out["at_end"] and all(abs(plant.q[i] - path[-1][i]) <= max(end_tolerance, lag_fn(i, plant.q[i]) + .05)
                                 for i in range(2)):
            result = {"status": "completed", "t": t}
            break
        for _ in range(int(round(control_dt / physics_dt))):
            plant.step(physics_dt)
        t = round(t + control_dt, 9)
    if result is None:
        result = {"status": "timeout", "t": t}
    result.update(trace=trace, unsafe=unsafe, steps=steps, follower=follower)
    return result
