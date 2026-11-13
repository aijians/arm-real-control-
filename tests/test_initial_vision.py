"""Initial rail confirmation: fresh retries, fixed limits and no unsafe motion."""
import copy
import unittest
from unittest.mock import Mock

from control.tasks import TaskRunner, TaskError


def state(shift=0., valid=True):
    return {"sliders_x_mm": {"蓝": -47., "橙": 12.}, "q_rad": [3., .1],
            "slider_limits_x_mm": {"蓝": [-51.+shift, -5.+shift],
                                   "橙": [-35.5+shift, 10.5+shift]} if valid else None,
            "rail_observation": {"reason": "ok" if valid else "rail_endpoints_missing"}}


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

    def test_missing_first_frame_recovers_with_three_stable_frames(self):
        runner=self.runner([state(valid=False),state(),state(.1),state(.2)])
        result={"environment_recovery":[]}
        final=runner._initial_environment(result)
        self.assertEqual(final['slider_limits_x_mm']['蓝'][1],-4.8)
        self.assertEqual(result['initial_limits_confirmation']['status'],'confirmed')
        self.assertEqual(len(result['environment_recovery']),1)
        self.assertAlmostEqual(result['initial_limits_confirmation']['spread_mm'],.2)
        runner.motors.move.assert_not_called()

    def test_persistent_missing_limits_retain_reason_and_last_observation(self):
        runner=self.runner([state(valid=False)]*4)
        result=runner.run({'intent':'push_to_end','color':'蓝'})
        self.assertEqual(result['reason'],'slider_limits_unavailable')
        self.assertEqual(result['details']['rail_observation']['reason'],'rail_endpoints_missing')
        self.assertEqual(len(result['environment_recovery']),4)
        self.assertIsNone(result['initial_state']['slider_limits_x_mm'])
        self.assertEqual(result['segments'],[])
        runner.motors.move.assert_not_called()
        self.assertEqual(runner.planner.mock_calls,[])

    def test_a_valid_lucky_frame_is_not_enough(self):
        runner=self.runner([state(valid=False)]*3+[state()])
        result=runner.run({'intent':'push_to_end','color':'蓝'})
        self.assertEqual(result['reason'],'slider_limits_unstable')
        runner.motors.move.assert_not_called()

    def test_jumping_endpoints_never_freeze_a_target(self):
        runner=self.runner([state(),state(2.),state(),state(2.)])
        result=runner.run({'intent':'push_to_end','color':'蓝'})
        self.assertEqual(result['reason'],'slider_limits_unstable')
        runner.motors.move.assert_not_called()

    def test_missing_frame_resets_consecutive_confirmation(self):
        runner=self.runner([state(),state(valid=False),state(),state()])
        result=runner.run({'intent':'push_to_end','color':'蓝'})
        self.assertEqual(result['reason'],'slider_limits_unstable')
        self.assertEqual(result['details']['confirmed_samples'],2)

    def test_unconfirmed_stop_does_not_read_or_retry_vision(self):
        runner=self.runner([state()]*4)
        runner.motors.halt.return_value={'stopped_confirmed':False}
        result=runner.run({'intent':'push_to_end','color':'蓝'})
        self.assertEqual(result['reason'],'initial_stop_not_confirmed')
        runner._snapshot.assert_not_called()
        runner.motors.move.assert_not_called()

    def test_confirmed_last_bounds_are_frozen_without_moving_them_outwards(self):
        runner=self.runner([state(),state(.1),state(.2)])
        result={'environment_recovery':[]}
        initial=runner._initial_environment(result)
        goals,_=runner.legs({'intent':'push_to_end','color':'蓝'},
                           initial['sliders_x_mm'],initial['slider_limits_x_mm'])
        self.assertEqual(goals,[-4.8])
        self.assertEqual(runner.targets.bounds('蓝'),(-50.8,-4.8))

    def test_active_slider_outside_motion_limits_still_fails(self):
        runner=self.runner([state()]*3)
        result=runner.run({'intent':'push_to_end','color':'橙'})
        self.assertEqual(result['reason'],'slider_start_outside_observed_limits')
        runner.motors.move.assert_not_called()


if __name__=='__main__': unittest.main()
