"""25 FPS 合成关节往复输入用于混叠回归，不是机器人视觉验证。"""
import json
import math
from pathlib import Path
import tempfile
import unittest
import pyarrow as pa
import pyarrow.parquet as pq
from streaming_recovery.__main__ import load_config
from streaming_recovery.source import LeRobotStream
from streaming_recovery.dynamics import JointInterval, joint_motion, valid_interval
from streaming_recovery.engine import Monitor
from streaming_recovery.tests.test_multimodal import measured
from streaming_recovery.tests.test_system import SyntheticVision, facts
from streaming_recovery.tests.test_system import INSTRUCTION


class JointIntervalTests(unittest.TestCase):
    def test_direct_positive_position_evidence_cannot_be_overridden_by_injected_static_summary(self):
        obs=measured(1,position=.2,native=True)
        result=joint_motion(obs,measured(0),load_config()["rules"])
        self.assertEqual(result["status"],"recent_motion")

    def test_out_and_back_native_motion_is_not_stationary_from_equal_endpoints(self):
        base=Path(__file__).resolve().parents[1]/"artifacts/test_tmp"
        base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            root=Path(tmp);(root/"meta").mkdir();(root/"data/chunk-000").mkdir(parents=True)
            (root/"meta/info.json").write_text(json.dumps({"codebase_version":"v2.1","fps":25,"chunks_size":1000}))
            rows=[]
            for i in range(26):
                t=i/25
                state=[0.]*14;velocity=[0.]*14
                state[7]=.2*math.sin(math.pi*t)**2
                velocity[7]=.2*math.pi*math.sin(2*math.pi*t)
                rows.append({"timestamp":t,"frame_index":i,"observation.state":state,"observation.velocity":velocity})
            pq.write_table(pa.Table.from_pylist(rows),root/"data/chunk-000/episode_000000.parquet")
            stream=LeRobotStream(root,0,[],sample_every=25,instruction=INSTRUCTION)
            observations=list(stream)
            result=joint_motion(observations[-1],observations[0],load_config()["rules"])
            self.assertNotEqual(result["status"],"stationary", "端点相同不能否认区间内的实际往复运动")
            self.assertEqual(result["interval_coverage"],"native_complete")
            self.assertEqual(result["native_interval"]["sample_count"],26)
            self.assertGreater(result["native_interval"]["max_joint_path_raw"],.35)
            again=list(stream)
            self.assertEqual(again[0].joint_interval["sample_count"],1)
            self.assertEqual(again[-1].joint_interval,observations[-1].joint_interval)
            self.assertEqual(stream.audit["rows_read"],26)

    def test_missing_native_sample_or_modality_cannot_prove_stationarity(self):
        for fault in ("gap","position","velocity"):
            collector=JointInterval()
            for i in range(26):
                if fault=="gap" and i==12:
                    continue
                collector.push(i/25,None if fault=="position" and i==12 else [0.]*14,
                               None if fault=="velocity" and i==12 else [0.]*14,f"f{i}")
            obs=measured(1);obs.joint_interval=collector.snapshot()
            result=joint_motion(obs,measured(0),load_config()["rules"])
            self.assertFalse(obs.joint_interval["complete"])
            self.assertNotEqual(result["status"],"stationary")

    def test_low_endpoints_do_not_veto_current_visual_operation(self):
        vision=SyntheticVision();monitor=Monitor(load_config(),vision)
        for t in (0,1):
            vision.current=facts(t,location="gripper",gripper="closed",motion="at_board",contact="between_fingers",
                                 measurements={"cut_operation":"yes" if t else "unknown"})
            result=monitor.step(measured(t))
        self.assertEqual(result["visual_facts"]["measurements"]["joint_motion"]["status"],"low_motion_support")
        self.assertEqual(result["visual_facts"]["measurements"]["cut_operation"],"yes")

    def test_aggregate_future_rejected_invalid_explicit_and_storage_bounded(self):
        collector=JointInterval()
        for i in range(10001):
            collector.push(i/25,[0.]*14,[0.]*14,f"f{i}")
        usage=collector.usage()
        self.assertEqual(usage["native_joint_history_samples"],0)
        self.assertEqual(usage["native_joint_previous_values"],6)
        self.assertEqual(usage["native_joint_path_values"],6)
        vision=SyntheticVision();monitor=Monitor(load_config(),vision)
        obs=measured(0);obs.joint_interval=collector.snapshot();vision.current=facts(0)
        with self.assertRaisesRegex(ValueError,"future"):
            monitor.step(obs)
        collector.reset()
        collector.push(0,[0.]*14,[0.]*14,"f0")
        self.assertEqual(collector.snapshot()["sample_count"],1)
        obs.joint_interval=collector.snapshot();obs.joint_interval["max_joint_path_raw"]=float("nan")
        self.assertFalse(valid_interval(obs.joint_interval,0))
        result=Monitor(load_config(),vision).step(obs)
        self.assertEqual(result["observation_inputs"]["joint_interval_status"],"invalid_or_stale")
        self.assertIsNone(result["observation_inputs"]["joint_interval"])


if __name__=="__main__":
    unittest.main()
