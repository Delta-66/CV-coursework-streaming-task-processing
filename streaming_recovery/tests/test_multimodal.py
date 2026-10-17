"""合成输入的多模态因果/功能回归；不代表真实视觉识别结果。"""
from dataclasses import replace
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from streaming_recovery.__main__ import load_config
from streaming_recovery.dynamics import JointInterval, joint_motion
from streaming_recovery.engine import Monitor
from streaming_recovery.source import LeRobotStream
from streaming_recovery.tests.test_system import observation, facts, SyntheticVision, INSTRUCTION


def measured(t, velocity=0., position=0., gripper=.005, native=False):
    obs=observation(t,gripper)
    obs.observation_state[7]=position
    obs.observation_velocity=[0.]*14
    obs.observation_velocity[7]=velocity
    obs.modalities["observation.velocity"]={"status":"available","timestamp":float(t),"age_s":0.,"reason":None}
    if native:
        # 显式合成整个区间，不能再用两个静止端点冒充连续静止。
        interval=JointInterval()
        start=max(0.,float(t)-1.)
        for i in range(round((float(t)-start)*25)+1):
            timestamp=start+i/25
            interval.push(timestamp,obs.observation_state,obs.observation_velocity,f"synthetic-{timestamp:.2f}")
        obs.joint_interval=interval.snapshot()
    return obs


class MultimodalTests(unittest.TestCase):
    def sequence(self,velocity=0.,position_step=0.,**kw):
        vision=SyntheticVision()
        monitor=Monitor(load_config(),vision)
        rows=[]
        for t in (0,1,2):
            raw=.06 if kw.get("gripper")=="open" else .005
            vision.current=facts(t,**deepcopy(kw))
            if t==0 and vision.current.measurements.get("cut_operation")=="yes":
                # 相对运动证据要先建立参考，首个样本尚无区间。
                vision.current.measurements["cut_operation"]="unknown"
            rows.append(monitor.step(measured(t,velocity,t*position_step,raw,native=True)))
        return rows,monitor

    def test_measured_motion_changes_stage_without_changing_grasp_evidence(self):
        kw={"location":"gripper","gripper":"closed","motion":"to_board","contact":"between_fingers"}
        stopped,_=self.sequence(**kw)
        moving,_=self.sequence(velocity=.4,**kw)
        self.assertEqual(stopped[-1]["state"]["stage"],"holding_tool")
        self.assertEqual(moving[-1]["state"]["stage"],"transport_tool")
        self.assertTrue(stopped[-1]["state"]["preconditions"]["reliable_grasp"])
        self.assertTrue(moving[-1]["state"]["preconditions"]["reliable_grasp"])

    def test_position_change_counts_as_recent_not_instantaneous_motion(self):
        result=joint_motion(measured(1,0.,.1),measured(0),load_config()["rules"])
        self.assertEqual(result["status"],"recent_motion")
        self.assertEqual(result["instantaneous"],"below_motion_threshold")
        self.assertEqual(len(result["evidence_refs"]),3)

    def test_gripper_motion_is_excluded_from_joint_motion(self):
        a,b=measured(0,gripper=.06),measured(1,gripper=0.,native=True)
        b.observation_velocity[13]=5.
        result=joint_motion(b,a,load_config()["rules"])
        self.assertEqual(result["status"],"stationary")

    def test_open_empty_arm_moving_over_work_area_supports_missing_precondition(self):
        kw={"location":"holder","gripper":"open","motion":"at_board","contact":"separate"}
        moving,_=self.sequence(velocity=.4,**kw)
        stopped,_=self.sequence(**kw)
        self.assertEqual(moving[-1]["diagnosis"]["type"],"prerequisite_unmet")
        self.assertEqual(moving[-1]["execution_status"],"confirmed_failure")
        self.assertIsNone(stopped[-1]["failure_event"])
        self.assertEqual(moving[-1]["state"]["attempt"],0)

    def test_moving_closed_gripper_does_not_establish_grasp(self):
        rows,_=self.sequence(velocity=.4,location="unknown",gripper="closed",motion="to_board",contact="unknown")
        self.assertTrue(all(r["state"]["preconditions"]["reliable_grasp"] is not True for r in rows))
        self.assertTrue(all(r["failure_event"] is None for r in rows))

    def test_stationary_joint_evidence_vetoes_visual_cut_motion(self):
        kw={"location":"gripper","gripper":"closed","motion":"at_board","contact":"between_fingers",
            "measurements":{"cut_operation":"yes"},"cut_piece":"no"}
        # 独立字典避免测试替身在前一个引擎内被修改。
        stopped,monitor=self.sequence(**kw)
        self.assertIsNone(monitor.tracker.operation_at)
        self.assertTrue(all(r["visual_facts"]["measurements"].get("cut_operation")!="yes" for r in stopped[1:]))
        self.assertEqual(stopped[-1]["visual_facts"]["measurements"]["cut_motion_check"],"joint_stationary_visual_motion_unconfirmed")
        kw["measurements"]={"cut_operation":"yes"}
        moving,monitor=self.sequence(velocity=.4,**kw)
        self.assertIsNotNone(monitor.tracker.operation_at)
        self.assertFalse(moving[-1]["state"]["task_completed"])

    def test_stale_and_invalid_velocity_not_used_and_missing_does_not_fake_zero(self):
        for value,status,timestamp in (([4.]*14,"stale",0.),([float('nan')]*14,"available",1.)):
            vision=SyntheticVision();monitor=Monitor(load_config(),vision)
            obs=measured(1)
            obs.observation_velocity=value
            obs.modalities["observation.velocity"].update(status=status,timestamp=timestamp)
            vision.current=facts(1,gripper="closed")
            result=monitor.step(obs)
            self.assertNotIn("observation.velocity[7:13]",result["modalities_used"])
            self.assertEqual(result["state"]["arm_activity"],"unknown")
            self.assertIsNone(result["observation_inputs"]["observation.velocity"])

    def test_future_velocity_rejected_and_old_position_not_used_across_gap(self):
        vision=SyntheticVision();monitor=Monitor(load_config(),vision)
        obs=measured(0);obs.modalities["observation.velocity"]["timestamp"]=1.
        vision.current=facts(0)
        with self.assertRaisesRegex(ValueError,"future modality"):
            monitor.step(obs)
        result=joint_motion(measured(10,0.,3.),measured(0),load_config()["rules"])
        self.assertIsNone(result["max_abs_position_change_raw"])
        self.assertEqual(result["status"],"unknown")

    def test_departed_work_area_without_cut_effect_does_not_restart_transport(self):
        vision=SyntheticVision();monitor=Monitor(load_config(),vision)
        for t in (0,1,2):
            vision.current=facts(t,location="gripper",gripper="closed",contact="between_fingers",
                                 motion="at_board" if t<2 else "to_board")
            result=monitor.step(measured(t,.4))
        self.assertEqual(result["state"]["stage"],"verify_cut")
        self.assertEqual(result["execution_status"],"information_insufficient")
        self.assertIsNone(result["failure_event"])

    def test_observed_return_and_release_do_not_require_confirmed_cut(self):
        vision=SyntheticVision();monitor=Monitor(load_config(),vision)
        for t in (0,1):
            vision.current=facts(t,location="gripper",gripper="closed",contact="between_fingers",motion="at_board")
            monitor.step(measured(t,.4))
        vision.current=facts(2,location="gripper",gripper="closed",contact="between_fingers",motion="returning_holder")
        result=monitor.step(measured(2,.4))
        self.assertEqual(result["state"]["stage"],"return_tool")
        vision.current=facts(3,location="holder",gripper="open",contact="separate",motion="away",released="yes")
        result=monitor.step(measured(3,.4,gripper=.06))
        self.assertEqual(result["state"]["stage"],"verify_completion")
        self.assertFalse(result["state"]["task_completed"])

    def test_optional_missing_columns_and_invalid_velocity_in_actual_reader(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        base=Path(__file__).resolve().parents[1]/"artifacts/test_tmp"
        base.mkdir(exist_ok=True)
        for include in (False,True):
            with tempfile.TemporaryDirectory(dir=base) as tmp:
                root=Path(tmp);(root/"meta").mkdir();(root/"data/chunk-000").mkdir(parents=True)
                (root/"meta/info.json").write_text(json.dumps({"codebase_version":"v2.1","fps":25,"chunks_size":1000}))
                row={"timestamp":0.,"frame_index":0,"observation.state":[0.]*14,"action":[999.]*14}
                if include:
                    row["observation.velocity"]=[float('nan')]*14
                pq.write_table(pa.Table.from_pylist([row]),root/"data/chunk-000/episode_000000.parquet")
                stream=LeRobotStream(root,0,["cam_high"],instruction=INSTRUCTION)
                obs=list(stream)[0]
                self.assertEqual(obs.modalities["observation.velocity"]["status"],"invalid" if include else "missing")
                self.assertIsNone(obs.observation_velocity)
                self.assertNotIn("action",stream.audit["columns_read"])


if __name__=="__main__":
    unittest.main()
