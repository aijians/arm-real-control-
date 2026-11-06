"""A waypoint owns three recovery rounds, not three speed/bias changes."""
import copy,json,math,unittest
from pathlib import Path
from test_recovery90 import fast_config,VirtualDrive,phase,scene_context,MotionRecovery,policy_module,recovery_module,m,LastResortDrive

def three_round_config():
    c=fast_config();c['motion_recovery']['max_recovery_rounds']=3;return c

class WaypointRoundsTests(unittest.TestCase):
    def test_stalled_node_gets_three_full_rounds_before_exhaustion(self):
        d=VirtualDrive(immobile=True,settings=three_round_config())
        with self.assertRaises(m.HardwareError) as caught:
            d.run(phase([[159.,20.],[157.,20.]],motion_recovery_geometry=scene_context()))
        self.assertEqual(caught.exception.reason,'motion_recovery_exhausted')
        r=caught.exception.details['motion_diagnostics']['execution']['speed_recovery']
        self.assertEqual(r['recovery_round'],3)
        self.assertEqual([e['recovery_round'] for e in r['events'] if e['event']=='node_recovery_round_started'],[1,2,3])
        for number in (1,2,3):
            ev=[e for e in r['events'] if e['recovery_round']==number]
            self.assertEqual([e['bias_motor_deg'] for e in ev if e['event']=='angle_change_written'],[-.2,-.6,-1.2])
            self.assertEqual(sum(e['event']=='final_speed_written' for e in ev),1)
        self.assertLess(d.elapsed,24.)
        self.assertEqual(d.unsafe_directions,[])
    def test_second_round_can_succeed_without_replaying_the_whole_path(self):
        class SecondRoundLoad(LastResortDrive):
            def advance(self,seconds):
                # Release on the second distinct final-speed command.
                final=sum(int.from_bytes(w[9:13],'little')==9000 for _,_,_,w in self.a6(0))
                original=self.release_speed;self.release_speed=90. if final>=2 else 1000.
                super().advance(seconds);self.release_speed=original
        d=SecondRoundLoad(settings=three_round_config())
        result=d.run(phase([[159.,20.],[157.,20.]],motion_recovery_geometry=scene_context()))
        r=result['motion_diagnostics']['execution']['speed_recovery']
        self.assertEqual(r['recovery_round'],2)
        self.assertEqual(sum(e['event']=='final_speed_written' for e in r['events']),2)
        self.assertEqual(d.pair._settling_bias_centi,[0,0]);self.assertEqual(d.unsafe_directions,[])
    def test_new_node_resets_counts_and_deadline_while_retaining_ordinary_floor(self):
        c=three_round_config();r=MotionRecovery(recovery_module.recovery_settings(c),c['motors'],'push',policy_module.motion_policy_settings(c))
        r.select_directions([-1,0],0.);r.axis_min_speed_deg_s[0]=60.;r.speed_tier[0]=3
        r.round_index=3;r.upshifts[0]=4;r.angle_changes[0]=3;r.downshifts[0]=2
        r.started_at=1.;r.deadline=25.;r.node_changed(26.,3)
        self.assertEqual(r.node_index,3);self.assertEqual(r.round_index,0)
        self.assertEqual(r.angle_changes,[0,0]);self.assertEqual(r.upshifts,[0,0]);self.assertIsNone(r.deadline)
        self.assertEqual(r.axis_min_speed_deg_s[0],60.)
        r._request(0,27.,3,60.,'angle_up');r.pending[0]['bias_before']=0.
        self.assertTrue(r.wrote(0,3,60.,27.1,bias=-.2))
        self.assertEqual(r.round_index,1);self.assertEqual(r.deadline,51.)
    def test_speed_changes_and_bias_levels_stay_in_the_same_round(self):
        c=three_round_config();r=MotionRecovery(recovery_module.recovery_settings(c),c['motors'],'push',policy_module.motion_policy_settings(c))
        r.select_directions([-1,0],0.)
        for i,s in enumerate((18.,30.,60.),1):
            r._request(0,float(i),i,s,'up');self.assertTrue(r.wrote(0,i,s,i+.1,bias=0.))
            self.assertEqual(r.round_index,1)
        for i,bias in enumerate((-.2,-.6,-1.2),4):
            r._request(0,float(i),3,60.,'angle_up');r.pending[0]['bias_before']=r.last_bias[0]
            self.assertTrue(r.wrote(0,3,60.,i+.1,bias=bias));self.assertEqual(r.round_index,1)
        self.assertEqual(r.angle_changes[0],3);self.assertEqual(r.total_rounds_started,1)
    def test_unwritten_round_restart_does_not_consume_next_round(self):
        c=three_round_config();r=MotionRecovery(recovery_module.recovery_settings(c),c['motors'],'push',policy_module.motion_policy_settings(c))
        r.round_index=1;r.axis_min_speed_deg_s[0]=90.;r.speed_tier[0]=4;r.diagnoses[0]='stuck'
        r._retry_or_exhaust(0,1.,-1.2)
        self.assertFalse(r.wrote(0,3,60.,1.1,bias=-1.2));self.assertEqual(r.round_index,1)
        self.assertTrue(r.wrote(0,3,60.,1.2,bias=0.));self.assertEqual(r.round_index,2)
    def test_two_waypoints_each_can_use_their_own_second_round(self):
        class PerPointLoad(VirtualDrive):
            def advance(self,seconds):
                old=self.q[0];moved=self.has_moved[0]
                super().advance(seconds)
                trace=self.pair._motion_trace
                index=0 if trace is None else trace.execution.get('waypoint_index',0)
                if index not in (1,2):return
                target=15700 if index==1 else 15500
                final=sum(int.from_bytes(w[9:13],'little')==9000
                          and int.from_bytes(w[6:9],'little')==target for _,_,_,w in self.a6(0))
                if final<2:
                    self.q[0],self.velocity[0],self.running[0],self.has_moved[0]=old,0.,False,moved
        d=PerPointLoad(settings=three_round_config())
        result=d.run(phase([[159.,20.],[157.,20.],[155.,20.]],motion_recovery_geometry=scene_context()))
        r=result['motion_diagnostics']['execution']['speed_recovery']
        started=[(e['waypoint_index'],e['recovery_round']) for e in r['events'] if e['event']=='node_recovery_round_started']
        self.assertEqual(started,[(1,1),(1,2),(2,1),(2,2)])
        self.assertEqual(d.unsafe_directions,[])
    def test_production_has_three_rounds(self):
        c=json.loads((Path(__file__).resolve().parents[1]/'config/robot.json').read_text())
        self.assertEqual(c['hardware']['motion_recovery']['max_recovery_rounds'],3)
        for n in (0,4,True):
            c['hardware']['motion_recovery']['max_recovery_rounds']=n
            with self.assertRaises(ValueError):policy_module.motion_policy_settings(c['hardware'])

if __name__=='__main__':unittest.main(verbosity=2)
