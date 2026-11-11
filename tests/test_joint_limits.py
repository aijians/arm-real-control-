"""Offline configured boundaries, signed A4 targets and wrapped feedback."""
import math
import time
import unittest
from pathlib import Path
from control.config import load_config
from control.motors import MotorPair, HardwareError
from control.planner import Planner

ROOT=Path(__file__).resolve().parents[1]

class JointLimitTests(unittest.TestCase):
    def pair(self):
        config,_=load_config(ROOT)
        return MotorPair(config['hardware'])

    def test_loaded_and_geometric_limits_match_requested_intervals(self):
        config,_=load_config(ROOT)
        self.assertEqual([(m['offset_deg'],m['limits_deg']) for m in config['hardware']['motors']],
                         [(0.,[90.,182.]),(0.,[-1.,90.])])
        planner=Planner(ROOT,config)
        self.assertEqual(planner.scene.limits.tolist(),
                         config['kinematics']['arm']['joint_limits_rad'])
        for arm in (planner.arm,):
            for q in ([90.,-1.],[182.,90.]):
                self.assertTrue(arm.within_limits(list(map(math.radians,q))))
            for q in ([89.,0.],[183.,0.],[100.,-2.],[100.,91.]):
                self.assertFalse(arm.within_limits(list(map(math.radians,q))))

    def test_a4_accepts_negative_boundary_and_a6_does_not_wrap_target(self):
        pair=self.pair()
        points=[{'time_s':0.,'q_rad':list(map(math.radians,[100.,-1.]))}]
        self.assertEqual(pair._validate_phase({'phase':'free','points':points})[0][1],points[0]['q_rad'])
        with self.assertRaisesRegex(ValueError,'outside A6 range'):
            pair._validate_phase({'phase':'withdraw','points':points})
        points[0]['q_rad'][1]=math.radians(-1.1)
        with self.assertRaisesRegex(ValueError,'outside model joint limits'):
            pair._validate_phase({'phase':'free','points':points})

    def test_single_turn_negative_feedback_preserves_raw_value(self):
        pair=self.pair();now=time.monotonic()
        pair._feedback=[{'motor_deg':100.,'speed_deg_s':0.,'angle_stamp':now,'speed_stamp':now},
                        {'motor_deg':359.5,'speed_deg_s':0.,'angle_stamp':now,'speed_stamp':now}]
        state=pair._snapshot()
        self.assertEqual(state['motor_deg'],[100.,-.5])
        self.assertEqual(state['raw_motor_deg'],[100.,359.5])
        self.assertAlmostEqual(math.degrees(state['q_rad'][1]),-.5)
        pair._feedback[1]['motor_deg']=358.
        with self.assertRaisesRegex(HardwareError,'joint_feedback_out_of_limits'):
            pair._snapshot()

if __name__=='__main__':unittest.main()
