"""合成语义输入只验证逻辑，不作为真实视觉识别结果。"""
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import unittest
from PIL import Image
from streaming_recovery.__main__ import load_config
from streaming_recovery.contracts import Observation, Sample, VisualFacts, ENUMS
from streaming_recovery.source import CausalAligner
from streaming_recovery.engine import Monitor
from streaming_recovery.recovery import validate_consistency
from streaming_recovery.vision import make_request, strict_json, QwenProcess


INSTRUCTION="Use the right arm to pick up the knife from the knife holder, cut a small piece of sausage and put it back on the knife holder."


def observation(t, gripper=.06, image=True):
    state=[0.]*14
    state[13]=gripper
    modalities={k:{"status":"available","timestamp":t,"age_s":0.,"reason":None} for k in ("cam_high","observation.state")}
    if not image:
        modalities["cam_high"]["status"]="missing"
    return Observation(float(t),INSTRUCTION,{"cam_high":Image.new("RGB",(64,48),(40,60,80))} if image else {},state,modalities,f"obs-{int(t*25):06d}")


def facts(t,location="holder",gripper="open",motion="at_holder",contact="unknown",valid=True,**kw):
    return VisualFacts(float(t),f"obs-{int(t*25):06d}",visibility="clear",knife_location=location,gripper=gripper,
        motion=motion,contact=contact,description="合成语义测试输入，非真实视觉",valid=valid,
        provenance="synthetic_test_double",evidence_refs=[{"timestamp":float(t),"ref":f"obs-{int(t*25):06d}","camera":"cam_high"}],**kw)


class SyntheticVision:
    # 无完整未来序列，只持有当前外部测试输入。
    def __init__(self):
        self.current=None
        self.last_metrics={"provenance":"synthetic_test_double"}

    def observe(self,observation,previous):
        return self.current


class SystemTests(unittest.TestCase):
    def setUp(self):
        self.config=load_config()
        self.vision=SyntheticVision()
        self.monitor=Monitor(self.config,self.vision)

    def feed(self,t,raw=.06,**kw):
        self.vision.current=facts(t,**kw)
        return self.monitor.step(observation(t,raw))

    def failed_attempt(self,start=0):
        self.feed(start)
        self.feed(start+1,.005,gripper="closed")
        suspected=self.feed(start+2,.005,gripper="closed",motion="at_board",contact="separate")
        confirmed=self.feed(start+3,.005,gripper="closed",motion="at_board",contact="separate")
        return suspected,confirmed

    def test_normal_incomplete_is_not_failure(self):
        for t in range(20):
            r=self.feed(t)
            self.assertFalse(r["notify"])
            self.assertEqual(r["execution_status"],"executing")
        self.assertIsNone(self.monitor.end_of_stream()["task_success"])

    def test_closed_gripper_not_success_then_evidence_failure(self):
        self.feed(0)
        r=self.feed(1,.005,gripper="closed")
        self.assertIsNot(r["state"]["preconditions"]["reliable_grasp"],True)
        self.assertIsNone(r["failure_event"])
        s=self.feed(2,.005,gripper="closed",motion="at_board",contact="separate")
        c=self.feed(3,.005,gripper="closed",motion="at_board",contact="separate")
        self.assertEqual(s["execution_status"],"suspicious")
        self.assertEqual(c["execution_status"],"confirmed_failure")
        self.assertEqual(c["diagnosis"]["type"],"grasp_not_established")

    def test_event_dedup_and_actual_retry(self):
        s,c=self.failed_attempt()
        for t in (4,5,6):
            r=self.feed(t,.005,gripper="closed",motion="at_board",contact="separate")
            self.assertFalse(r["notify"])
            self.assertEqual(r["failure_event"]["event_id"],c["failure_event"]["event_id"])
            self.assertEqual(r["state"]["attempt"],1)
        self.feed(7,.06,gripper="open",motion="at_holder")
        self.feed(8,.005,gripper="closed",motion="at_holder")
        r=self.feed(9,.005,gripper="closed",motion="at_board",contact="separate")
        self.assertNotEqual(r["failure_event"]["event_id"],c["failure_event"]["event_id"])
        self.assertEqual(r["state"]["attempt"],2)

    def test_missing_conflict_and_recovery_not_execution(self):
        _,c=self.failed_attempt()
        r=self.feed(4,.005,valid=False,error="occluded")
        self.assertEqual(r["execution_status"],"information_insufficient")
        self.assertEqual(r["failure_event"]["event_id"],c["failure_event"]["event_id"])
        self.assertNotEqual(r["failure_event"]["state"],"resolved")
        r=self.feed(5,.06,gripper="closed",location="gripper",contact="between_fingers",motion="at_board")
        self.assertTrue(r["state"]["modality_conflict"])
        self.assertIsNot(r["state"]["preconditions"]["reliable_grasp"],True)
        for t in (6,7):
            r=self.feed(t,.005,gripper="closed",location="gripper",contact="between_fingers",motion="at_board")
        self.assertEqual(r["execution_status"],"recovered")
        self.assertFalse(r["execution"]["actually_executed"])
        self.assertIsNone(r["execution"]["execution_receipt"])
        self.assertIsNone(r["correction"])

    def test_loss_of_grasp_has_repeated_evidence(self):
        for t in (0,1):
            self.feed(t,.005,gripper="closed",location="gripper",contact="between_fingers",motion="to_board")
        a=self.feed(2,.005,gripper="closed",location="board",contact="separate",motion="at_board")
        b=self.feed(3,.005,gripper="closed",location="board",contact="separate",motion="at_board")
        self.assertEqual(a["execution_status"],"suspicious")
        self.assertEqual(b["diagnosis"]["type"],"lost_grasp")
        self.assertEqual(b["correction"]["low_level"]["action_type"],"hold_and_observe")

    def test_leaving_visible_work_area_does_not_invent_a_retry_stage(self):
        for t in (0,1):
            self.feed(t,.005,gripper="closed",location="gripper",contact="between_fingers",motion="at_board")
        r=self.feed(2,.005,gripper="closed",location="unknown",contact="unknown",motion="unknown")
        self.assertEqual(r["state"]["stage"],"unknown")
        self.assertEqual(r["execution_status"],"information_insufficient")
        self.assertIn("曾建立刀具夹持",r["state"]["confirmed_subgoals"])
        self.assertIsNone(r["failure_event"])

    def test_missing_breaks_suspicion_continuity(self):
        self.feed(0)
        self.feed(1,.005,gripper="closed")
        self.feed(2,.005,gripper="closed",contact="separate",motion="at_board")
        self.feed(3,.005,valid=False,error="timeout")
        r=self.feed(4,.005,gripper="closed",contact="separate",motion="at_board")
        self.assertEqual(r["failure_event"]["state"],"suspected")

    def test_large_time_gap_cannot_make_new_attempt(self):
        self.feed(0)
        r=self.feed(100,.005,gripper="closed")
        self.assertEqual(r["state"]["attempt"],0)

    def test_two_layer_consistency_coordinates_and_next_action(self):
        s,r=self.failed_attempt()
        self.assertEqual(s["correction"]["low_level"]["action_type"],"hold_and_observe")
        self.assertEqual(r["correction"]["low_level"]["action_type"],"open_gripper")
        low=r["correction"]["low_level"]
        self.assertIsNone(low["numeric_command"])
        self.assertFalse(low["directly_executable"])
        self.assertTrue(low["post_checks"])
        broken=deepcopy(r)
        broken["correction"]["low_level"]["recovery_goal_id"]="other"
        with self.assertRaises(ValueError):
            validate_consistency(broken)
        r=self.feed(4,.06,gripper="open",contact="separate",motion="at_board",knife_handle_xy=[800,600],gripper_xy=[400,300])
        low=r["correction"]["low_level"]
        self.assertEqual(low["action_type"],"align_in_image")
        self.assertEqual(low["coordinate_frame"],"cam_high_image_x_right_y_down")
        self.assertIn("图像右侧",low["translation"]["direction"])
        self.assertIsNone(low["translation"]["magnitude"])

    def test_reset_matches_fresh_monitor(self):
        self.failed_attempt()
        self.monitor.reset()
        a=self.feed(0)
        fresh=Monitor(self.config,self.vision)
        b=fresh.step(observation(0))
        self.assertEqual(a,b)
        self.assertEqual(a["state"]["attempt"],0)
        self.assertEqual(a["memory_usage"]["events"],0)

    def test_model_request_has_only_bounded_past_images(self):
        a,b=observation(1),observation(2)
        req=make_request(b,a)
        self.assertEqual(len(req["frames"]),2)
        self.assertFalse(any(k in req for k in ("episode","path","label","duration")))
        with self.assertRaises(ValueError):
            make_request(a,b)

    def test_visual_schema_rejects_hallucinated_coordinates_and_conflict(self):
        data={k:"unknown" for k in ENUMS}|{"knife_handle_xy":None,"gripper_xy":None,"description":""}
        VisualFacts.parse(data,0,"obs","synthetic_test_double",[])
        for value in ([float("nan"),20],[1001,2],[True,1],[1,2,3]):
            bad=data|{"knife_handle_xy":value}
            with self.assertRaises(ValueError):
                VisualFacts.parse(bad,0,"obs","test",[])
        with self.assertRaises(ValueError):
            VisualFacts.parse(data|{"knife_location":"holder","co_motion":"yes"},0,"obs","test",[])

    def test_model_invalid_output_and_timeout_degrade_explicitly(self):
        from queue import Empty
        from unittest.mock import Mock
        for raw in ('{"x":1,"x":2}', '{"x":NaN}', 'not json'):
            with self.assertRaises(ValueError):
                strict_json(raw)
        model=QwenProcess.__new__(QwenProcess)
        model.config=self.config["model"]
        model.incoming=Mock()
        model.outgoing=Mock()
        model.outgoing.get.side_effect=Empty
        model.error=None
        model.calls=0
        model.close=Mock()
        r=model.observe(observation(0),None)
        self.assertFalse(r.valid)
        self.assertEqual(r.error,"model_timeout_worker_terminated")
        model.close.assert_called_once()
        model.outgoing.get.assert_called_once()
        # worker 已关闭后再次调用仍明确未知，不再次发送请求或伪造结果。
        r=model.observe(observation(1),observation(0))
        self.assertFalse(r.valid)
        self.assertEqual(model.incoming.put.call_count,1)

    def test_aligner_future_stale_invalid_and_out_of_order(self):
        a=CausalAligner(["camera","joints"],.2)
        with self.assertRaises(ValueError):
            a.push(Sample("camera",1,1,"future"),0)
        a.push(Sample("camera",0,0,"first"),0)
        values,status=a.snapshot(.1)
        self.assertEqual(values["camera"],"first")
        self.assertEqual(status["joints"]["status"],"missing")
        values,status=a.snapshot(.5)
        self.assertNotIn("camera",values)
        self.assertEqual(status["camera"]["status"],"stale")
        a.push(Sample("joints",.5,.5,None,False,"nan"),.5)
        self.assertEqual(a.snapshot(.5)[1]["joints"]["status"],"invalid")
        with self.assertRaises(ValueError):
            a.snapshot(.4)

    def test_duplicate_timestamps_and_unsupported_task_rejected(self):
        self.feed(0)
        with self.assertRaises(ValueError):
            self.feed(0)
        self.monitor.reset()
        with self.assertRaises(ValueError):
            self.monitor.step(replace(observation(0),instruction="unplug charger"))

    def test_completion_requires_effect_evidence_not_initial_layout(self):
        for t in (0,1):
            self.feed(t,.005,gripper="closed",contact="separate",released="yes",cut_piece="yes")
        self.assertFalse(self.monitor.tracker.completed)
        for t in (2,3):
            self.feed(t,.005,location="gripper",gripper="closed",motion="at_board",contact="between_fingers",cut_piece="yes")
        for t in (4,5):
            r=self.feed(t,.06,location="holder",gripper="open",contact="separate",released="yes",cut_piece="yes")
        self.assertTrue(r["state"]["task_completed"])
        self.assertFalse(r["execution"]["actually_executed"])

    def test_long_stream_every_history_bound(self):
        maxima={}
        for t in range(10000):
            k=t%8
            if k in (0,1):
                r=self.feed(t,.06,gripper="open")
            elif k==2:
                r=self.feed(t,.005,gripper="closed")
            elif k in (3,4,5):
                r=self.feed(t,.005,gripper="closed",contact="separate",motion="at_board")
            else:
                r=self.feed(t,.005,location="gripper",gripper="closed",contact="between_fingers",motion="at_board")
            for key,value in r["memory_usage"].items():
                maxima[key]=max(maxima.get(key,0),value)
        self.assertLessEqual(maxima["events"],4)
        self.assertLessEqual(maxima["event_evidence"],16)
        self.assertLessEqual(maxima["features"],6)
        self.assertLessEqual(maxima["anchors"],8)
        self.assertEqual(maxima["log_reads"],0)
        self.assertLessEqual(maxima["observations"],4)


if __name__=="__main__":
    unittest.main()
