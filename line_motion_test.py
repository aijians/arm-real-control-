"""Two user-specified motor-angle legs; no camera, task inbox or PID writes."""
import argparse
from contextlib import nullcontext
from datetime import datetime
import json
import math
from pathlib import Path
import signal

from control.config import load_config, save_json
from control.kinematics import PlanarArm
from control.runtime import instance_lock
from control.trajectory import timed_free_points


MOTOR_TARGETS = ((164.96, 53.09), (146.0, 51.06),
                 (144.99, 47.45), (158.16, 53.02))


class LineTestPlan:
    def __init__(self, root):
        self.config, self.paths = load_config(root)
        self.arm = PlanarArm(self.config['kinematics'])
        self.motors = self.config['hardware']['motors']
        if [m['name'] for m in self.motors] != ['upper', 'lower']:
            raise ValueError('需要按 upper、lower 排列的双轴配置')
        if not self.config['hardware'].get('free_timed_motion', {}).get('enabled'):
            raise ValueError('本脚本需要现有定时 A4 模式')
        self.targets = [self.from_motor(p) for p in MOTOR_TARGETS]
        self.home = self.check([math.radians(v) for v in
                                self.config['planning']['return_q_deg']])
        scene = json.loads(self.paths['scene'].read_text())
        self.departure_y = float(scene['routes']['entry_y_mm'])

    def check(self, q):
        if len(q) != 2 or not all(math.isfinite(v) for v in q):
            raise ValueError('关节角必须是两个有限数值')
        if not self.arm.within_limits(q):
            raise ValueError('角度超出几何模型限位')
        for v, m in zip(q, self.motors):
            if not m['limits_deg'][0]-1e-9 <= math.degrees(v) <= m['limits_deg'][1]+1e-9:
                raise ValueError('角度超出电机配置限位：'+m['name'])
        return list(q)

    def from_motor(self, values):
        return self.check([math.radians((v-m['offset_deg'])/(m['sign']*m['ratio']))
                           for v, m in zip(values, self.motors)])

    def to_motor(self, q):
        return [m['offset_deg']+m['sign']*m['ratio']*math.degrees(v)
                for v, m in zip(q, self.motors)]

    def line(self, start, goal, face):
        """Linear Cartesian reference, continuous seeded IK, exact joint endpoint.

        Use the same rigid equivalent contact point as production pushing.
        Face +x belongs to the first rightward leg, -x to the second leftward leg.
        No slider positions are fabricated for this manual motion test.
        """
        start, goal = self.check(start), self.check(goal)
        a, b = self.arm.slider_center(start, face), self.arm.slider_center(goal, face)
        count = max(1, math.ceil(math.dist(a, b)/0.25))
        path = [start]
        for i in range(1, count+1):
            target = [x+(y-x)*i/count for x, y in zip(a, b)]
            q = self.check(self.arm.inverse(target, face, seed=path[-1], require_direction=False))
            if max(abs(x-y) for x, y in zip(q, path[-1])) > math.radians(5):
                raise ValueError('直线逆解分支不连续')
            if i == count:
                if max(abs(x-y) for x, y in zip(q, goal)) > 1e-6:
                    raise ValueError('终点角度不在同一连续逆解分支')
                q = goal
            path.append(q)
        return path

    def departure(self, q):
        point = list(self.arm.slider_center(q, '+x'))
        if point[1] <= self.departure_y:
            return self.check(q)
        point[1] = self.departure_y
        return self.check(self.arm.inverse(point, '+x', seed=q, require_direction=False))

    def phase(self, label, path, *, home=False):
        timed = self.config['hardware']['free_timed_motion']
        # MotorPair repeats this subdivision idempotently. Dense Cartesian
        # vertices remain intact; unlike two angle lerps, the reference is straight.
        points = timed_free_points([(float(i), self.check(q)) for i, q in enumerate(path)],
                                   timed['max_step_deg'], timed['period_s'])
        result = {'phase': 'free', 'route': label,
                  'points': [{'time_s': t, 'q_rad': q} for t, q in points],
                  'collision_checked': False, 'camera_used': False}
        if home:
            result['route'] = 'return_home'
            minimum = self.config['planning'].get('return_upper_min_deg')
            if minimum is not None:
                result['return_upper_min_deg'] = minimum
        return result

    def stages(self):
        a, b, c, d = self.targets
        # Build every nominal leg before any serial port is opened.
        departure = self.departure(b)
        return [
            ('到第一起点', a, None, False),
            ('第一段直线 +x', b, '+x', False),
            ('复位撤离', departure, '+x', False),
            ('复位', self.home, None, True),
            ('到第二起点', c, None, False),
            ('第二段直线 -x', d, '-x', False),
        ]

    def preview(self):
        stages = self.stages()
        previous = self.home
        result = []
        for label, goal, face, home in stages:
            path = self.line(previous, goal, face) if face else [previous, goal]
            phase = self.phase(label, path, home=home)
            result.append({'stage': label, 'target_motor_deg': self.to_motor(goal),
                           'target_model_deg': list(map(math.degrees, goal)),
                           'reference_face': face, 'nodes': len(phase['points']),
                           'nominal_duration_s': phase['points'][-1]['time_s']})
            previous = goal
        return result


def execute(plan, record, log_path, *, pair_factory=None, lock_factory=instance_lock,
            isolation=None):
    if pair_factory is None:
        from control.motors import MotorPair
        pair_factory = MotorPair
    with lock_factory(plan.paths['runtime_dir']), (isolation or nullcontext()):
        with pair_factory(plan.config['hardware']) as pair:
            for label, goal, face, home in plan.stages():
                state = pair.read_state()
                start = plan.check(state['q_rad'])
                path = plan.line(start, goal, face) if face else [start, goal]
                phase = plan.phase(label, path, home=home)
                entry = {'stage': label, 'status': 'running', 'plan': phase,
                         'start_motor_deg': plan.to_motor(start),
                         'target_motor_deg': plan.to_motor(goal)}
                record['stages'].append(entry)
                save_json(record, log_path)
                print('[测试] '+label+' → 电机角 '+str(entry['target_motor_deg']), flush=True)
                try:
                    entry['result'] = pair.move(phase)
                except BaseException as exc:
                    entry.update(status='failed', reason=str(exc),
                                 details=getattr(exc, 'details', {}),
                                 halt_result=getattr(exc, 'halt_result', None))
                    save_json(record, log_path)
                    raise
                entry['status'] = 'completed'
                entry['actual_motor_deg'] = plan.to_motor(entry['result']['feedback']['q_rad'])
                save_json(record, log_path)
                print('[测试] 完成 '+label+'，反馈电机角 '+str(entry['actual_motor_deg']), flush=True)


def main():
    parser = argparse.ArgumentParser(description='无相机双段直线测试；角度顺序为大臂、小臂电机角')
    parser.add_argument('--execute', action='store_true', help='实际执行；省略时仅离线预览')
    parser.add_argument('--isolate-sal', action='store_true', help='运行期间临时暂停厂家串口扫描器，退出自动恢复')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    plan = LineTestPlan(root)
    preview = plan.preview()
    print(json.dumps(preview, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        print('以上为离线预览，没有访问串口或相机。实际运行加 --execute。')
        return 0
    log_path = plan.paths['runtime_dir']/'line_test'/(
        datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json')
    record = {'status': 'running', 'angle_space': 'motor_deg', 'camera_used': False,
              'pid_written': False, 'preview': preview, 'stages': []}
    isolation = None
    if args.isolate_sal:
        from pid_tuner.exclusive import isolate_sal
        isolation = isolate_sal([m['device'] for m in plan.motors])
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt('收到退出信号')
    old_term = signal.signal(signal.SIGTERM, interrupted)
    save_json(record, log_path)
    print('日志：'+str(log_path), flush=True)
    try:
        execute(plan, record, log_path, isolation=isolation)
        record['status'] = 'completed'
        print('两段完成，保持第二终点；未额外复位。', flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        record.update(status='failed', reason=str(exc))
        print('测试中止：'+str(exc)+'；不再执行后续动作。', flush=True)
        return 1
    finally:
        save_json(record, log_path)
        signal.signal(signal.SIGTERM, old_term)
        print('日志：'+str(log_path), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
