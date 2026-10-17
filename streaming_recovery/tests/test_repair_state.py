"""2026-10-17 组合反例：合成语义逻辑，不代表真实像素/自然重试验证。"""
from dataclasses import replace
import unittest

from streaming_recovery.__main__ import load_config
from streaming_recovery.engine import Monitor
from streaming_recovery.tests.test_system import SyntheticVision, facts, observation


class RepairStateTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.vision = SyntheticVision()
        self.monitor = Monitor(self.config, self.vision)

    def feed(self, t, raw=.06, moving=False, **kw):
        self.vision.current = facts(t, **kw)
        obs = observation(t, raw)
        velocity = [0.] * 14
        velocity[7] = .3 if moving else 0.
        obs.modalities["observation.velocity"] = {"status":"available", "timestamp":float(t), "age_s":0., "reason":None}
        return self.monitor.step(replace(obs, observation_velocity=velocity))

    def held(self, start=0):
        for t in (start, start+1):
            result = self.feed(t, .005, moving=True, location="gripper", gripper="closed",
                               contact="between_fingers", released="no", motion="to_board")
        self.assertTrue(result["state"]["preconditions"]["reliable_grasp"])
        return result

    def lost_then_board_retry(self, support="board"):
        self.held()
        for t in (2, 3):
            lost = self.feed(t, .005, location=support, gripper="closed", contact="separate", motion="at_board")
        self.assertEqual(lost["failure_event"]["kind"], "lost_grasp")
        self.assertEqual(lost["failure_event"]["state"], "confirmed")
        self.feed(4, location=support, gripper="open", contact="separate", motion="at_board",
                  knife_handle_xy=[700,500], gripper_xy=[400,500])
        retry = self.feed(5, .005, location=support, gripper="closed", contact="separate", motion="at_board",
                          knife_handle_xy=[700,500], gripper_xy=[400,500])
        self.assertTrue(retry["state"]["new_attempt_observed"])
        self.assertEqual(retry["failure_event"]["state"], "closed_for_retry")
        return lost, retry

    def test_A_explicit_release_overrides_stale_positive_proxies(self):
        self.held()
        for t in (2, 3):
            result = self.feed(t, .005, location="holder", gripper="closed", contact="separate",
                               released="yes", motion="at_board",
                               measurements={"tool_on_board":True, "robot_at_board":True})
            with self.subTest(timestamp=t):
                self.assertFalse(result["state"]["preconditions"]["reliable_grasp"])
                self.assertNotIn("grasp", result["state"]["anchors"])
                self.assertNotEqual(result["state"]["stage"], "cutting")
                self.assertTrue(result["state"].get("evidence_conflicts"))
        self.assertTrue(result["state"]["preconditions"]["tool_returned"])
        result = self.feed(4, .005, location="board", gripper="closed", contact="separate", motion="at_board")
        self.assertNotEqual(result["state"]["candidate"], "lost_grasp")

    def test_B_early_release_then_empty_motion_invalidates_current_precondition(self):
        self.feed(0)
        self.feed(1, .005, gripper="closed")
        self.held(2)
        for t in (4, 5):
            released = self.feed(t, location="holder", gripper="open", contact="separate", released="yes", motion="at_holder")
            self.assertIsNone(released["state"]["candidate"], "正常释放本身不能被判失败")
        for t in (6, 7):
            result = self.feed(t, moving=True, location="holder", gripper="open", contact="separate",
                               released="yes", motion="at_board")
        self.assertTrue(result["state"]["attempt_held"], "保留本次曾成功的历史语义")
        self.assertFalse(result["state"]["preconditions"]["reliable_grasp"])
        self.assertEqual(result["state"]["candidate"], "prerequisite_unmet")
        self.assertEqual(result["state"]["stage"], "blocked_precondition")
        self.assertEqual(result["execution_status"], "confirmed_failure")

    def test_B_return_without_subsequent_action_is_not_failure(self):
        self.held()
        for t in range(2, 8):
            result = self.feed(t, location="holder", gripper="open", contact="separate", released="yes", motion="away")
            self.assertIsNone(result["state"]["candidate"])
            self.assertFalse(result["notify"])

    def test_C_failed_board_or_elsewhere_retry_creates_fresh_event(self):
        for support in ("board", "elsewhere"):
            with self.subTest(support=support):
                self.setUp()
                lost, retry = self.lost_then_board_retry(support)
                for t in (6, 7, 8):
                    result = self.feed(t, .005, location=support, gripper="closed", contact="separate", motion="at_board",
                                       knife_handle_xy=[700,500], gripper_xy=[400,500])
                self.assertIsNotNone(result["failure_event"])
                self.assertEqual(result["failure_event"]["kind"], "grasp_not_established")
                self.assertEqual(result["failure_event"]["state"], "confirmed")
                self.assertNotEqual(result["failure_event"]["event_id"], lost["failure_event"]["event_id"])
                self.assertEqual(result["recovery_event"]["event_id"], result["failure_event"]["event_id"])
                self.assertEqual(result["correction"]["low_level"]["action_type"], "open_gripper")
                self.assertEqual(result["state"]["attempt_context"]["support_location"], support)
                self.assertEqual(result["state"]["attempt_context"]["object"], "knife")
                self.assertFalse(result["execution"]["actually_executed"])

    def test_C_closed_waiting_or_aligned_points_are_not_failure_evidence(self):
        for contact in ("unknown", "separate"):
            with self.subTest(contact=contact):
                self.setUp()
                self.lost_then_board_retry()
                for t in range(6, 16):
                    result = self.feed(t, .005, location="board", gripper="closed", contact=contact, motion="at_board",
                                       knife_handle_xy=[500,500], gripper_xy=[500,500])
                    self.assertIsNone(result["state"]["candidate"])
                    self.assertIsNone(result["failure_event"])
                    self.assertEqual(result["correction"]["low_level"]["recovery_phase"], "verify_grasp_effect")
                    self.assertFalse(result["execution"]["actually_executed"])

    def test_C_occluded_retry_waits_without_opening_or_false_resolution(self):
        self.lost_then_board_retry()
        result = self.feed(6, .005, location="unknown", gripper="closed", contact="unknown", valid=False, error="occluded")
        self.assertEqual(result["execution_status"], "information_insufficient")
        self.assertEqual(result["recovery_event"]["observation_status"], "unobservable")
        self.assertEqual(result["correction"]["low_level"]["action_type"], "hold_and_observe")

    def test_C_positive_retry_resolves_context_without_execution_attribution(self):
        self.lost_then_board_retry()
        for t in (6, 7):
            result = self.feed(t, .005, moving=True, location="gripper", gripper="closed", contact="between_fingers", motion="to_board")
        self.assertTrue(result["state"]["preconditions"]["reliable_grasp"])
        self.assertIsNone(result["recovery_event"])
        self.assertIsNone(result["correction"])
        self.assertFalse(result["execution"]["outcome_attributed_to_suggestion"])

    def test_C_actual_reopen_returns_to_preparation_without_inventing_attempt(self):
        self.lost_then_board_retry()
        attempt=self.monitor.tracker.attempt
        result=self.feed(6,location="board",gripper="open",contact="separate",motion="at_board",
                         knife_handle_xy=[700,500],gripper_xy=[400,500])
        self.assertEqual(result["state"]["attempt"],attempt)
        self.assertFalse(result["state"]["new_attempt_observed"])
        self.assertEqual(result["correction"]["low_level"]["action_type"],"align_in_image")
        self.assertEqual(result["correction"]["low_level"]["recovery_phase"],"align_image")
        self.assertFalse(result["execution"]["actually_executed"])

    def test_D_recovery_condition_keeps_connection_inside_gripper(self):
        self.lost_then_board_retry()
        result = self.feed(6, .005, moving=True, location="gripper", gripper="closed", contact="between_fingers", motion="to_board")
        self.assertIsNotNone(result["correction"])
        resume_when = result["correction"]["high_level"]["resume_when"]
        self.assertNotIn("离开夹爪内", resume_when)
        self.assertIn("保持连接", resume_when)
        self.assertIn("砧板", resume_when)

    def test_visible_joint_localization_keeps_alignment_stage_without_readiness_claim(self):
        result = self.feed(0, moving=True, location="holder", contact="separate", motion="at_holder")
        self.assertEqual(result["state"]["stage"], "approach_tool")
        result = self.feed(1, moving=True, location="holder", contact="separate", motion="at_holder",
                           knife_handle_xy=[700,500], gripper_xy=[400,500])
        self.assertEqual(result["state"]["stage"], "align_grasp")
        self.assertFalse(result["state"]["preconditions"]["reliable_grasp"])
        self.assertEqual(result["visual_facts"]["grasp_readiness"], {})
        self.assertIsNone(result["correction"])


if __name__ == "__main__":
    unittest.main()
