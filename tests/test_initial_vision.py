"""Initial confirmation: fixed model limits, reliable fresh frames and no unsafe motion."""
import copy
import unittest
from unittest.mock import Mock

from control.tasks import TaskRunner, TaskError

FIXED = {"蓝": [-46.56, -2.06], "橙": [-31.06, 13.44]}


def state(shift=0., reliable=True, orange=12., limits=True):
    reliability = {"蓝": {"reliable": reliable, "reasons": [] if reliable else ["rod_width_inconsistent_with_model"]},
                   "橙": {"reliable": True, "reasons": []}}
    return {"sliders_x_mm": {"蓝": -47. + shift, "橙": orange}, "q_rad": [3., .1],
            "slider_limits_x_mm": copy.deepcopy(FIXED) if limits else None,
            "reliability": reliability,
            "rail_observation": {"limits_source": "fixed_model"}}


class InitialVisionTests(unittest.TestCase):
    def runner(self, states):
        config={"sliders": {"蓝": {"y_mm":155., "end":"right"},
                            "橙": {"y_mm":155., "end":"left"}},
                "verification":{"tolerance_mm":.5,"samples":3,"interval_s":.2},
                "retry_limit":3}
        runner=TaskRunner(Mock(),config,Mock(),Mock())
        runner.motors.halt.return_value={"stopped_confirmed":True}
        runner._snapshot=Mock(side_effect=copy.deepcopy(states))
        return runner

    def test_unreliable_first_frame_recovers_with_three_stable_frames(self):
        runner=self.runner([state(reliable=False),state(),state(.1),state(.2)])
        result={"environment_recovery":[]}
        final=runner._initial_environment(result)
        self.assertEqual(final['slider_limits_x_mm'],FIXED)
        self.assertEqual(result['initial_limits_confirmation']['status'],'confirmed')
        self.assertEqual(len(result['environment_recovery']),1)
        self.assertEqual(result['environment_recovery'][0]['reason'],'initial_vision_unreliable')
        self.assertAlmostEqual(result['initial_limits_confirmation']['spread_mm'],.2)
        runner.motors.move.assert_not_called()

    def test_persistent_unreliable_frames_retain_reason_and_last_observation(self):
        runner=self.runner([state(reliable=False)]*4)
        result=runner.run({'intent':'push_to_end','color':'蓝'})
        self.assertEqual(result['reason'],'initial_vision_unreliable')
        self.assertEqual(result['details']['reasons'],{'蓝':['rod_width_inconsistent_with_model']})
        self.assertEqual(len(result['environment_recovery']),4)
        self.assertEqual(result['segments'],[])
        runner.motors.move.assert_not_called()
        self.assertEqual(runner.planner.mock_calls,[])

    def test_missing_limits_still_fail_without_motion(self):
        runner=self.runner([state(limits=False)]*4)
        result=runner.run({'intent':'push_to_end','color':'蓝'})
        self.assertEqual(result['reason'],'slider_limits_unavailable')
        runner.motors.move.assert_not_called()

    def test_a_valid_lucky_frame_is_not_enough(self):
        runner=self.runner([state(reliable=False)]*3+[state()])
        result=runner.run({'intent':'push_to_end','color':'蓝'})
        self.assertEqual(result['reason'],'initial_positions_unstable')
        runner.motors.move.assert_not_called()

    def test_jumping_positions_never_freeze_a_target(self):
        runner=self.runner([state(),state(2.),state(),state(2.)])
        result=runner.run({'intent':'push_to_end','color':'蓝'})
        self.assertEqual(result['reason'],'initial_positions_unstable')
        runner.motors.move.assert_not_called()

    def test_stationary_orange_jump_also_blocks_confirmation(self):
        runner=self.runner([state(),state(orange=13.5),state(),state(orange=13.5)])
        result=runner.run({'intent':'push_to_end','color':'蓝'})
        self.assertEqual(result['reason'],'initial_positions_unstable')

    def test_unreliable_frame_resets_consecutive_confirmation(self):
        runner=self.runner([state(),state(reliable=False),state(),state()])
        result=runner.run({'intent':'push_to_end','color':'蓝'})
        self.assertEqual(result['reason'],'initial_positions_unstable')
        self.assertEqual(result['details']['confirmed_samples'],2)

    def test_unconfirmed_stop_does_not_read_or_retry_vision(self):
        runner=self.runner([state()]*4)
        runner.motors.halt.return_value={'stopped_confirmed':False}
        result=runner.run({'intent':'push_to_end','color':'蓝'})
        self.assertEqual(result['reason'],'initial_stop_not_confirmed')
        runner._snapshot.assert_not_called()
        runner.motors.move.assert_not_called()

    def test_fixed_model_end_is_the_push_to_end_goal(self):
        runner=self.runner([state(),state(.1),state(.2)])
        result={'environment_recovery':[]}
        initial=runner._initial_environment(result)
        goals,_=runner.legs({'intent':'push_to_end','color':'蓝'},
                           initial['sliders_x_mm'],initial['slider_limits_x_mm'])
        self.assertEqual(goals,[-2.06])
        self.assertEqual(runner.targets.bounds('蓝'),(-46.56,-2.06))

    def test_start_reading_within_stop_tolerance_is_accepted(self):
        # 视觉读数在左止挡外 1.4 mm（地面实测约 −47.9），在 2.5 mm 止挡容差内仍可开始。
        runner=self.runner([state(-.9)]*3)
        result={'environment_recovery':[]}
        initial=runner._initial_environment(result)
        goals,_=runner.legs({'intent':'push_to_end','color':'蓝'},
                           initial['sliders_x_mm'],initial['slider_limits_x_mm'])
        self.assertEqual(goals,[-2.06])

    def test_active_slider_outside_motion_limits_still_fails(self):
        runner=self.runner([state(orange=16.5)]*3)
        result=runner.run({'intent':'push_to_end','color':'橙'})
        self.assertEqual(result['reason'],'slider_start_outside_observed_limits')
        runner.motors.move.assert_not_called()


if __name__=='__main__': unittest.main()
