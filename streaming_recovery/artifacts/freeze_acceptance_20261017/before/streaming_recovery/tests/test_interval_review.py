"""独立审查补充：原生区间只覆盖末端一小段时，不得否定整个决策间隔的运动。"""
import unittest
from streaming_recovery.__main__ import load_config
from streaming_recovery.dynamics import JointInterval, joint_motion, valid_interval
from streaming_recovery.engine import Monitor
from streaming_recovery.tests.test_multimodal import measured
from streaming_recovery.tests.test_system import SyntheticVision, facts


class IntervalReviewTests(unittest.TestCase):
    def native(self, start=.96, end=1., moving=False):
        collector=JointInterval()
        velocity=[0.]*14
        velocity[7]=.3 if moving else 0.
        collector.push(start,[0.]*14,velocity,"native-start")
        collector.push(end,[0.]*14,[0.]*14,"native-end")
        return collector.snapshot()

    def test_short_complete_suffix_does_not_veto_longer_visual_interval(self):
        vision=SyntheticVision();monitor=Monitor(load_config(),vision)
        vision.current=facts(0,location="gripper",gripper="closed",motion="at_board",contact="between_fingers")
        monitor.step(measured(0))
        observation=measured(1);observation.joint_interval=self.native()
        vision.current=facts(1,location="gripper",gripper="closed",motion="at_board",contact="between_fingers",
                             measurements={"cut_operation":"yes"})
        result=monitor.step(observation)
        motion=result["visual_facts"]["measurements"]["joint_motion"]
        self.assertNotEqual(motion["status"],"stationary")
        self.assertEqual(result["visual_facts"]["measurements"]["cut_operation"],"yes")

    def test_positive_short_interval_is_still_evidence_of_recent_motion(self):
        observation=measured(1);observation.joint_interval=self.native(moving=True)
        motion=joint_motion(observation,measured(0),load_config()["rules"])
        self.assertEqual(motion["status"],"recent_motion")
        self.assertLessEqual(len(motion["evidence_refs"]),3)
        self.assertTrue(any(ref.get("field")=="native_measured_joint_interval" and
                            ref.get("start_ref")=="native-start" and ref.get("ref")=="native-end"
                            for ref in motion["evidence_refs"]))

    def test_stationarity_requires_a_previous_interval_within_gap(self):
        observation=measured(1);observation.joint_interval=self.native()
        for previous in (None,measured(-3)):
            with self.subTest(previous=previous.timestamp if previous else None):
                motion=joint_motion(observation,previous,load_config()["rules"])
                self.assertNotEqual(motion["status"],"stationary")

    def test_single_sample_cannot_claim_complete_nonzero_interval(self):
        native=self.native()
        native.update(start_timestamp=0.,span_s=1.,sample_count=1,expected_sample_count=1,
                      valid_position_samples=1,valid_velocity_samples=1)
        self.assertFalse(valid_interval(native,1.))


if __name__=="__main__":
    unittest.main()
