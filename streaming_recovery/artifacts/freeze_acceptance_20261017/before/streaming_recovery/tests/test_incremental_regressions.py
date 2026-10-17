"""审查问题的合成语义回归。先在修改前运行；不代表视觉识别精度。"""
from copy import deepcopy
from dataclasses import replace
import unittest

from streaming_recovery.tests import test_system as _support
from streaming_recovery.tests.test_system import facts, observation
from streaming_recovery.recovery import validate_consistency


class IncrementalRegressions(_support.SystemTests):
    # 不重复继承原测试，仅复用其准备/输入辅助函数。
    def test_A_occluded_failure_cannot_open(self):
        self.failed_attempt()
        self.vision.current = facts(4, gripper="closed", contact="separate")
        self.vision.current.visibility = "occluded"
        r = self.monitor.step(observation(4, .005))
        self.assertFalse(r["visual_facts"]["effect_usable"])
        self.assertEqual(r["correction"]["low_level"]["action_type"], "hold_and_observe")

    def test_B_second_attempt_failure_after_historical_grasp(self):
        self.feed(0)
        self.feed(1,.005,gripper="closed")
        for t in (2,3):
            self.feed(t,.005,location="gripper",gripper="closed",contact="between_fingers",motion="to_board")
        for t in (4,5):
            lost = self.feed(t,.005,location="board",gripper="closed",contact="separate",motion="at_board")
        self.feed(6,.06,gripper="open",contact="separate")
        self.feed(7,.005,gripper="closed")
        for t in (8,9):
            r = self.feed(t,.005,gripper="closed",contact="separate",motion="at_board")
        self.assertEqual(r["state"]["attempt"],2)
        self.assertIsNotNone(r["failure_event"])
        self.assertEqual(r["failure_event"]["kind"],"grasp_not_established")
        self.assertNotEqual(r["failure_event"]["event_id"],lost["failure_event"]["event_id"])

    def test_C_unknown_gap_preserves_recent_grasp_evidence(self):
        for t in (0,1):
            self.feed(t,.005,location="gripper",gripper="closed",contact="between_fingers",motion="to_board")
        r=self.feed(2,.005,location="unknown",gripper="closed",contact="unknown",motion="unknown")
        self.assertIsNone(r["failure_event"])
        for t in (3,4):
            r=self.feed(t,.005,location="board",gripper="closed",contact="separate",motion="at_board")
        self.assertIsNotNone(r["failure_event"])
        self.assertEqual(r["failure_event"]["kind"],"lost_grasp")
        self.assertEqual(r["execution_status"],"confirmed_failure")

    def test_D_preexisting_piece_is_not_this_cut(self):
        for t in (0,1):
            self.feed(t,cut_piece="yes",contact="separate",released="yes")
        for t in (2,3):
            self.feed(t,.005,location="gripper",gripper="closed",motion="at_board",contact="between_fingers",cut_piece="yes")
        for t in (4,5):
            r=self.feed(t,contact="separate",released="yes",cut_piece="yes")
        self.assertFalse(r["state"]["preconditions"]["cut_effect_confirmed"])
        self.assertFalse(r["state"]["task_completed"])

    def genuine_cut(self):
        self.feed(0,cut_piece="no",contact="separate",released="yes")
        self.feed(1,.005,gripper="closed",cut_piece="no")
        for t in (2,3):
            self.feed(t,.005,location="gripper",gripper="closed",contact="between_fingers",motion="at_board",cut_piece="no",
                      measurements={"cut_operation":"yes"})
        for t in (4,5):
            self.feed(t,.005,location="gripper",gripper="closed",contact="between_fingers",motion="at_board",cut_piece="yes",
                      measurements={"cut_operation":"yes", "new_cut_effect":"yes", "cut_reference_timestamp":0.0})
        for t in (6,7):
            r=self.feed(t,contact="separate",released="yes",cut_piece="yes")
        return r

    def test_E_completion_does_not_freeze_current_return(self):
        r=self.genuine_cut()
        self.assertTrue(r["state"]["task_completed"])
        r=self.feed(8,.005,location="board",gripper="closed",contact="separate",motion="at_board",cut_piece="yes")
        self.assertFalse(r["state"]["preconditions"]["tool_returned"])
        self.assertNotEqual(r["state"]["stage"],"completed")
        self.assertIsNone(r["state"]["candidate"],"已明确释放的历史连接不能继续作为失抓前件")

    def test_validator_rejects_unusable_and_unfounded_spatial_values(self):
        _,r=self.failed_attempt()
        for mutation in ("occluded", "stale", "conflict", "translation", "rotation", "object", "frame", "force", "precondition"):
            with self.subTest(mutation=mutation):
                bad=deepcopy(r)
                if mutation=="occluded":
                    bad["visual_facts"].update(valid=False,visibility="occluded",effect_usable=False)
                elif mutation=="stale":
                    bad["visual_facts"]["timestamp"]-=100
                elif mutation=="conflict":
                    bad["state"]["modality_conflict"]=True
                elif mutation in ("translation","rotation"):
                    bad["correction"]["low_level"][mutation]={"values":[.2,0,0],"units":"m"}
                elif mutation=="object":
                    bad["correction"]["high_level"]["object"]="sausage"
                    bad["correction"]["low_level"]["object"]="sausage"
                elif mutation=="force":
                    bad["correction"]["low_level"]["force"]=99
                elif mutation=="precondition":
                    bad["correction"]["low_level"]["preconditions"][1]["status"]="supported_by_current_observation"
                else:
                    bad["correction"]["low_level"]["coordinate_frame"]="robot_base"
                with self.assertRaises(ValueError):
                    validate_consistency(bad)

    def test_2d_alignment_waits_then_independent_readiness_allows_close(self):
        self.failed_attempt()
        r=self.feed(4,contact="separate",knife_handle_xy=[500,500],gripper_xy=[500,500])
        self.assertNotEqual(r["correction"]["low_level"]["action_type"],"align_in_image")
        self.assertNotEqual(r["correction"]["low_level"]["action_type"],"close_gripper")
        # 向后兼容扩展：旧 VisualFacts 上也能挂载，但旧系统并不消费此证据。
        self.vision.current=facts(5,contact="separate",knife_handle_xy=[500,500],gripper_xy=[500,500])
        self.vision.current.grasp_readiness={"depth":"confirmed","orientation":"confirmed","contact_region":"confirmed",
            "source":"synthetic_independent_depth_contact","timestamp":5.,"ref":"ready-5"}
        r=self.monitor.step(observation(5))
        self.assertEqual(r["correction"]["low_level"]["action_type"],"close_gripper")
        self.assertEqual(r["state"]["attempt"],1)
        self.assertFalse(r["execution"]["actually_executed"])

    def test_negated_and_other_instructions_rejected(self):
        for text in ("Do not use the right arm; leave the knife in the holder and do not cut the sausage.",
                     "Use the right arm to move the sausage near the knife holder."):
            with self.subTest(instruction=text):
                self.monitor.reset()
                self.vision.current=facts(0)
                with self.assertRaises(ValueError):
                    self.monitor.step(replace(observation(0),instruction=text))

    def test_observed_close_advances_then_effect_resolves(self):
        self.failed_attempt()
        self.feed(4,contact="separate",knife_handle_xy=[500,500],gripper_xy=[500,500])
        r=self.feed(5,.005,gripper="closed",contact="unknown")
        self.assertEqual(r["state"]["attempt"],2)
        self.assertEqual(r["correction"]["low_level"]["recovery_phase"],"verify_grasp_effect")
        r=self.feed(6,.005,location="gripper",gripper="closed",contact="between_fingers",motion="to_board")
        self.assertIsNot(r["state"]["preconditions"]["reliable_grasp"],True)
        r=self.feed(7,.005,location="gripper",gripper="closed",contact="between_fingers",motion="to_board")
        self.assertTrue(r["state"]["preconditions"]["reliable_grasp"])
        self.assertIsNone(r["correction"])
        self.assertFalse(r["execution"]["actually_executed"])

    def test_old_grasp_expires_and_unknown_does_not_prove_loss(self):
        for t in (0,1):
            self.feed(t,.005,location="gripper",gripper="closed",contact="between_fingers",motion="to_board")
        for t in range(2,8):
            r=self.feed(t,.005,location="unknown",gripper="closed",contact="unknown",motion="unknown")
            self.assertIsNone(r["failure_event"])
        r=self.feed(8,.005,location="board",gripper="closed",contact="separate",motion="at_board")
        self.assertIsNone(r["failure_event"])

    def test_close_after_release_away_is_not_a_retry(self):
        self.genuine_cut()
        old=self.monitor.tracker.attempt
        r=self.feed(8,.005,gripper="closed",contact="separate",released="yes",motion="away",cut_piece="yes")
        self.assertEqual(r["state"]["attempt"],old)
        self.assertIsNone(r["failure_event"])

    def test_qwen_holder_and_legacy_rack_boundary_without_gpu(self):
        from unittest.mock import Mock
        import json
        from streaming_recovery.vision import QwenProcess, PROMPT
        from streaming_recovery.contracts import ENUMS
        self.assertIn("knife_location: holder",PROMPT)
        for location in ("holder","rack"):
            model=QwenProcess.__new__(QwenProcess)
            model.config=self.config["model"];model.error=None;model.calls=0
            model.incoming=Mock();model.outgoing=Mock()
            payload={k:"unknown" for k in ENUMS}|{"knife_location":location,"knife_handle_xy":None,"gripper_xy":None,"description":"format fixture"}
            model.outgoing.get.return_value={"raw":json.dumps(payload)}
            out=model.observe(observation(0),None)
            self.assertTrue(out.valid,out.error)
            self.assertEqual(out.knife_location,"holder")


# unittest 不重复继承的 17 项；它们仍由原测试模块独立执行。
for _name in tuple(vars(_support.SystemTests)):
    if _name.startswith("test_") and _name not in IncrementalRegressions.__dict__:
        setattr(IncrementalRegressions,_name,None)

if __name__=="__main__":
    unittest.main()
