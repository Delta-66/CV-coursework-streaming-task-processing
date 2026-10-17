"""Synthetic logic regressions; does not claim real visual accuracy."""
from copy import deepcopy
from dataclasses import replace
import unittest
from streaming_recovery.__main__ import load_config
from streaming_recovery.engine import Monitor
from streaming_recovery.tests.test_system import SyntheticVision, facts, observation
from streaming_recovery.recovery import validate_consistency

class FrozenLearningTests(unittest.TestCase):
    def setUp(self):
        self.config=load_config();self.config['rules']['observe_work_exit']=True
        self.vision=SyntheticVision();self.monitor=Monitor(self.config,self.vision)

    def feed(self,t,work=True,piece='unknown',valid=True,raw=.005):
        self.vision.current=facts(t,location='gripper',gripper='closed',motion='at_board' if work else 'to_board',
            contact='between_fingers',released='no',cut_piece=piece,valid=valid,
            measurements={'robot_at_board':work,'cut_layout':{'clear':not work,'timestamp':float(t),
                'ref':f'obs-{int(t*25):06d}','camera':'cam_high','frame':'cam_high_image_pixels','image_size':[64,48],'small':[]}})
        obs=observation(t,raw);velocity=[0.]*14;velocity[7]=.3
        obs.modalities['observation.velocity']={'status':'available','timestamp':float(t)}
        return self.monitor.step(replace(obs,observation_velocity=velocity))

    def establish(self):
        self.feed(0);return self.feed(1)

    def test_in_work_unfinished_is_not_failure(self):
        for t in range(20):self.assertIsNone(self.feed(t)['failure_event'])
        self.assertIsNone(self.monitor.end_of_stream()['task_success'])

    def test_exit_needs_repeat_and_keeps_event_with_held_knife(self):
        self.establish();a=self.feed(2,False,'no');b=self.feed(3,False,'no');c=self.feed(4,False,'no')
        self.assertEqual(a['execution_status'],'suspicious')
        self.assertEqual(b['execution_status'],'confirmed_failure')
        self.assertEqual(b['failure_event']['event_id'],c['failure_event']['event_id'])
        self.assertFalse(c['notify']);self.assertFalse(c['state']['recovered'])
        self.assertEqual(c['correction']['low_level']['recovery_phase'],'verify_cut_alignment')
        validate_consistency(c)

    def test_unknown_layout_or_gap_does_not_complete_failure(self):
        self.establish();self.assertIsNone(self.feed(2,False,'unknown')['failure_event'])
        self.assertIsNone(self.feed(20,False,'no')['failure_event'])

    def test_occluded_work_cannot_become_fresh_exit(self):
        self.establish();self.feed(2,valid=False)
        self.assertIsNone(self.feed(3,False,'no')['failure_event'])

    def test_prior_advice_cannot_mutate_fact_context(self):
        self.establish();self.feed(2,False,'no');self.feed(3,False,'no')
        before=deepcopy(vars(self.monitor.recovery))
        for phase in ('open','close','move_to_future','already_successful'):
            self.monitor.recovery.observe_request({'low_level':{'recovery_phase':phase}})
        self.assertEqual(before,vars(self.monitor.recovery))

    def test_reset_removes_work_event_and_wait(self):
        self.establish();self.feed(2,False,'no');self.feed(3,False,'no');self.monitor.reset()
        self.assertIsNone(self.monitor.tracker.work_cycle)
        self.assertEqual(self.monitor.recovery.wait_count,0)
        self.assertIsNone(self.feed(0,False,'no')['failure_event'])

    def test_contradictory_grasp_recovery_action_rejected_for_effect_event(self):
        self.establish();self.feed(2,False,'no');r=self.feed(3,False,'no')
        r['correction']['low_level']['action_type']='open_gripper'
        with self.assertRaisesRegex(ValueError,'effect_action_goal_mismatch'):validate_consistency(r)

    def test_positive_piece_is_not_negative_effect(self):
        self.establish()
        for t in (2,3,4):self.assertIsNone(self.feed(t,False,'yes')['failure_event'])

    def test_actual_regrasp_starts_new_work_cycle_without_old_negative_effect(self):
        self.establish();self.feed(2,False,'no');self.feed(3,False,'no')
        for t,raw,gripper in ((4,.06,'open'),(5,.005,'closed')):
            self.vision.current=facts(t,location='holder',gripper=gripper,motion='at_holder',contact='separate',
                released='yes',cut_piece='no',measurements={'robot_at_board':False,'cut_layout':{'clear':True}})
            r=self.monitor.step(observation(t,raw))
        self.assertTrue(r['state']['new_attempt_observed'])
        self.assertIsNone(r['state']['work_cycle'])
        self.assertIsNone(r['state']['candidate'])
        self.assertEqual(r['failure_event']['state'],'closed_for_retry')

if __name__=='__main__':unittest.main()
