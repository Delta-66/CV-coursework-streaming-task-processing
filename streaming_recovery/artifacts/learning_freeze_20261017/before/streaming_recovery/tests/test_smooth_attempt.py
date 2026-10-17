"""完整 Monitor 的平滑闭爪合成语义回归，不是自然重试/真实闭爪定位验证。"""
from dataclasses import replace
import unittest

from streaming_recovery.__main__ import load_config
from streaming_recovery.engine import Monitor
from streaming_recovery.tests.test_system import SyntheticVision, facts, observation


class SmoothAttemptTests(unittest.TestCase):
    def setUp(self):
        self.config=load_config()
        self.vision=SyntheticVision()
        self.monitor=Monitor(self.config,self.vision)

    def feed(self,t,raw=.06,near=True,valid=True,stale=False,missing=False,**kw):
        gripper="open" if raw>=self.config['rules']['gripper_open_raw'] else (
            "closed" if raw<=self.config['rules']['gripper_closed_raw'] else "unknown")
        fields={"location":"board" if near else "holder","gripper":gripper,
                "motion":"at_board","contact":"separate","released":"yes",
                "knife_handle_xy":[700,500] if near else None,
                "gripper_xy":[400,500] if near else None,
                "gripper_source":"observation.state[13]"}
        fields.update(kw)
        visibility=fields.pop('visibility','clear')
        self.vision.current=facts(t,valid=valid,**fields)
        self.vision.current.visibility=visibility
        obs=observation(t,raw)
        velocity=[0.]*14;velocity[7]=.3
        obs.modalities['observation.velocity']={"status":"available","timestamp":float(t),"age_s":0.,"reason":None}
        if stale:
            obs.modalities['observation.state'].update(status="stale",timestamp=float(t)-1.,age_s=1.)
        if missing:
            obs.modalities['observation.state'].update(status="missing",timestamp=None,reason="not_arrived")
        return self.monitor.step(replace(obs,observation_velocity=velocity))

    def held_then_lost(self):
        for t in (0,1):
            held=self.feed(t,.005,location="gripper",contact="between_fingers",released="no",motion="to_board")
        self.assertTrue(held['state']['preconditions']['reliable_grasp'])
        for t in (2,3):lost=self.feed(t,.005)
        self.assertEqual(lost['failure_event']['kind'],'lost_grasp')
        self.assertEqual(lost['failure_event']['state'],'confirmed')
        return lost

    def test_smooth_close_after_lost_grasp_creates_new_attempt_and_event(self):
        lost=self.held_then_lost()
        rows=[self.feed(t,raw) for t,raw in enumerate((.06,.045,.035,.025,.015,.005,.005),4)]
        self.assertEqual([r['timestamp']for r in rows if r['state']['new_attempt_observed']],[7.])
        self.assertEqual(rows[-1]['state']['attempt'],1)
        self.assertEqual(rows[-1]['state']['attempt_context']['support_location'],'board')
        self.assertEqual(rows[-1]['failure_event']['kind'],'grasp_not_established')
        self.assertEqual(rows[-1]['failure_event']['state'],'confirmed')
        self.assertNotEqual(rows[-1]['failure_event']['event_id'],lost['failure_event']['event_id'])
        self.assertEqual(rows[-1]['recovery_event']['event_id'],rows[-1]['failure_event']['event_id'])
        self.assertTrue(all(not r['execution']['actually_executed']for r in rows))

    def test_large_close_is_one_attempt(self):
        self.feed(0,.06)
        r=self.feed(1,.005)
        self.assertTrue(r['state']['new_attempt_observed'])
        self.assertEqual(r['state']['attempt'],1)

    def test_threshold_jitter_does_not_rearm(self):
        rows=[self.feed(t,raw) for t,raw in enumerate((.06,.024,.026,.023,.025,.027,.024))]
        self.assertEqual(sum(r['state']['new_attempt_observed']for r in rows),1)
        self.assertEqual(rows[-1]['state']['attempt'],1)

    def test_long_closed_hold_does_not_add_attempts(self):
        self.feed(0,.06)
        rows=[self.feed(t,.005) for t in range(1,61)]
        self.assertEqual(sum(r['state']['new_attempt_observed']for r in rows),1)
        self.assertEqual(rows[-1]['state']['attempt'],1)

    def test_real_reopen_then_smooth_close_creates_second_attempt(self):
        values=(.06,.005,.005,.06,.045,.035,.025,.015,.005)
        rows=[self.feed(t,raw) for t,raw in enumerate(values)]
        self.assertEqual([r['timestamp']for r in rows if r['state']['new_attempt_observed']],[1.,6.])
        self.assertEqual(rows[-1]['state']['attempt'],2)

    def test_no_observed_open_history_is_not_an_attempt(self):
        rows=[self.feed(t,raw) for t,raw in enumerate((.035,.03,.025,.015,.005))]
        self.assertFalse(any(r['state']['new_attempt_observed']for r in rows))

    def test_long_gap_invalidates_open_history(self):
        self.feed(0,.06);self.feed(1,.035)
        r=self.feed(10,.005)
        self.assertFalse(r['state']['new_attempt_observed']);self.assertEqual(r['state']['attempt'],0)

    def test_occlusion_requires_fresh_open_history(self):
        self.feed(0,.06);self.feed(1,.035)
        self.feed(2,.03,valid=False,visibility="occluded")
        self.feed(3,.035)
        r=self.feed(4,.005)
        self.assertFalse(r['state']['new_attempt_observed']);self.assertEqual(r['state']['attempt'],0)

    def test_stale_or_missing_raw_cannot_keep_derived_open_latch(self):
        for status in ('stale','missing'):
            with self.subTest(status=status):
                self.setUp();self.feed(0,.06);self.feed(1,.035)
                self.feed(2,.06,**{status:True})
                self.feed(3,.035)
                r=self.feed(4,.005)
                self.assertFalse(r['state']['new_attempt_observed']);self.assertEqual(r['state']['attempt'],0)

    def test_close_away_from_target_consumes_latch(self):
        self.feed(0,.06,near=False)
        away=self.feed(1,.025,near=False)
        near=self.feed(2,.005,near=True)
        self.assertFalse(away['state']['new_attempt_observed'])
        self.assertFalse(near['state']['new_attempt_observed'])
        self.assertEqual(near['state']['attempt'],0)

    def test_close_gripper_suggestion_does_not_increment_attempt(self):
        # 模拟先前失败后已打开、准备证据完备，但实际raw继续维持打开。
        self.held_then_lost()
        self.feed(4,.06)
        self.feed(5,.005)
        self.feed(6,.005);self.feed(7,.005);self.feed(8,.005)
        before=self.monitor.tracker.attempt
        for t in (9,10):
            ready={"depth":"confirmed","orientation":"confirmed","contact_region":"confirmed",
                   "source":"synthetic_independent_depth_contact","timestamp":float(t),"ref":f"ready-{t}"}
            r=self.feed(t,.06,knife_handle_xy=[500,500],gripper_xy=[500,500],grasp_readiness=ready)
            self.assertEqual(r['correction']['low_level']['action_type'],'close_gripper')
            self.assertFalse(r['state']['new_attempt_observed'])
            self.assertEqual(r['state']['attempt'],before)

    def test_reset_removes_open_history(self):
        self.feed(0,.06);self.monitor.reset()
        r=self.feed(0,.005)
        self.assertEqual(r['state']['attempt'],0);self.assertFalse(r['state']['new_attempt_observed'])

if __name__=='__main__':unittest.main()
