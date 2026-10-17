"""效果顺序的合成语义反例；不作为真实像素准确率或自然恢复验证。"""
import unittest
from streaming_recovery.__main__ import load_config
from streaming_recovery.engine import Monitor
from streaming_recovery.tests.test_system import SyntheticVision, facts, observation


class EffectOrderTests(unittest.TestCase):
    def setUp(self):
        self.vision=SyntheticVision()
        self.monitor=Monitor(load_config(),self.vision)

    def feed(self,t,raw=.06,**kw):
        visibility=kw.pop("visibility","clear")
        self.vision.current=facts(t,**kw)
        self.vision.current.visibility=visibility
        return self.monitor.step(observation(t,raw))

    def layout(self,t,pieces):
        return {"timestamp":float(t),"ref":f"obs-{int(t*25):06d}","camera":"cam_high",
                "frame":"cam_high_image_pixels","image_size":[512,384],"clear":True,"small":pieces}

    def held(self,t,piece="no",operation=False,measurements=None):
        m=dict(measurements or {})
        m.setdefault("cut_operation","yes" if operation else "unknown")
        return self.feed(t,.005,location="gripper",gripper="closed",contact="between_fingers",
                         released="no",motion="at_board" if operation else "to_board",cut_piece=piece,measurements=m)

    def ordered_start(self):
        self.feed(0,cut_piece="no",contact="separate",released="yes")
        self.feed(1,.005,gripper="closed",cut_piece="no")
        self.held(2)
        return self.held(3,operation=True)

    def test_effect_before_operation_cannot_be_retroactively_attributed(self):
        self.feed(0,cut_piece="no",contact="separate",released="yes")
        self.feed(1,cut_piece="yes",contact="separate",released="yes")
        self.held(2,"yes")
        self.held(3,"yes",True,{"new_cut_effect":"unknown"})
        r=self.held(4,"yes",measurements={"new_cut_effect":"unknown"})
        self.assertFalse(r["state"]["preconditions"]["cut_effect_confirmed"])
        for t in (5,6):
            r=self.feed(t,cut_piece="yes",contact="separate",released="yes")
        self.assertFalse(r["state"]["task_completed"])

    def test_normal_binary_transition_after_operation_remains_supported(self):
        self.ordered_start()
        self.held(4,"yes")
        r=self.held(5,"yes")
        self.assertTrue(r["state"]["preconditions"]["cut_effect_confirmed"])

    def test_same_frame_first_effect_and_operation_is_not_ordered(self):
        self.feed(0,cut_piece="no")
        self.held(1)
        self.held(2)
        self.held(3,"yes",True)
        r=self.held(4,"yes")
        self.assertFalse(r["state"]["preconditions"]["cut_effect_confirmed"])

    def test_preexisting_piece_then_distinct_new_piece_after_operation(self):
        old=[[100,100,10,10]]
        for t in (0,1):
            self.feed(t,cut_piece="yes",measurements={"cut_layout":self.layout(t,old)})
        self.held(2,"yes",measurements={"cut_layout":self.layout(2,old)})
        self.held(3,"yes",True,{"cut_layout":self.layout(3,old)})
        for t in (4,5):
            r=self.held(t,"yes",measurements={"cut_layout":self.layout(t,old+[[200,200,12,12]]),
                          "new_cut_effect":"yes","cut_reference_timestamp":0.})
        self.assertTrue(r["state"]["preconditions"]["cut_effect_confirmed"])

    def test_initial_reference_novelty_does_not_override_preoperation_layout(self):
        old=[[100,100,10,10]]
        self.feed(0,cut_piece="no",measurements={"cut_layout":self.layout(0,[])})
        self.feed(1,cut_piece="yes",measurements={"cut_layout":self.layout(1,old),"new_cut_effect":"yes","cut_reference_timestamp":0.})
        self.held(2,"yes",measurements={"cut_layout":self.layout(2,old)})
        self.held(3,"yes",True,{"cut_layout":self.layout(3,old),"new_cut_effect":"yes","cut_reference_timestamp":0.})
        r=self.held(4,"yes",measurements={"cut_layout":self.layout(4,old),"new_cut_effect":"yes","cut_reference_timestamp":0.})
        self.assertFalse(r["state"]["preconditions"]["cut_effect_confirmed"])

    def test_invalid_observation_or_long_gap_invalidates_unconfirmed_link(self):
        for fault in ("invalid","occluded","gap"):
            with self.subTest(fault=fault):
                self.setUp();self.ordered_start()
                if fault!="gap":
                    self.feed(4,.005,valid=fault!="invalid",visibility="occluded" if fault=="occluded" else "clear",
                              gripper="closed",cut_piece="unknown")
                for t in (8,9) if fault=="gap" else (5,6):
                    r=self.held(t,"yes")
                self.assertFalse(r["state"]["preconditions"]["cut_effect_confirmed"])

    def test_valid_local_food_unknown_can_wait_through_continuous_operation(self):
        self.ordered_start()
        for t in (4,5,6):
            self.held(t,"unknown",True)
        self.held(7,"yes")
        r=self.held(8,"yes")
        self.assertTrue(r["state"]["preconditions"]["cut_effect_confirmed"])

    def test_invalid_layout_cannot_be_treated_as_empty_or_binary_fallback(self):
        self.ordered_start()
        for t in (4,5):
            bad=self.layout(t,[[1,2]])
            r=self.held(t,"yes",measurements={"cut_layout":bad})
        self.assertFalse(r["state"]["preconditions"]["cut_effect_confirmed"])

    def test_preexisting_piece_moved_or_unmatched_is_not_a_new_piece(self):
        old=[[100,100,10,10]]
        for moved in ([[200,200,10,10]],[[200,200,10,10],[300,200,12,12]]):
            with self.subTest(moved=moved):
                self.setUp()
                self.feed(0,cut_piece="yes",measurements={"cut_layout":self.layout(0,old)})
                self.held(1,"yes",measurements={"cut_layout":self.layout(1,old)})
                self.held(2,"yes",True,{"cut_layout":self.layout(2,old)})
                for t in (3,4):
                    r=self.held(t,"yes",measurements={"cut_layout":self.layout(t,moved),"new_cut_effect":"yes","cut_reference_timestamp":0.})
                self.assertFalse(r["state"]["preconditions"]["cut_effect_confirmed"])

    def test_layout_binding_shape_and_capacity_rejected(self):
        for fault in ("camera","frame","timestamp","ref","capacity","size","bounds","empty_yes"):
            with self.subTest(fault=fault):
                self.setUp();self.ordered_start()
                for t in (4,5):
                    layout=self.layout(t,[[100,100,10,10]])
                    if fault=="camera":layout["camera"]="cam_right_wrist"
                    elif fault=="frame":layout["frame"]="robot_base"
                    elif fault=="timestamp":layout["timestamp"]=float(t+1)
                    elif fault=="ref":layout["ref"]="other-observation"
                    elif fault=="capacity":layout["small"]*=5
                    elif fault=="size":layout["image_size"]=[100000,384]
                    elif fault=="bounds":layout["small"]=[[510,10,20,10]]
                    else:layout["small"]=[]
                    r=self.held(t,"yes",measurements={"cut_layout":layout})
                self.assertFalse(r["state"]["preconditions"]["cut_effect_confirmed"])
                self.assertEqual(r["state"]["cut_effect_order"]["reason"],"invalid_layout_contract")

    def test_first_change_operation_and_baseline_refs_are_explicit_and_bounded(self):
        self.ordered_start();self.held(4,"yes")
        r=self.held(5,"yes")
        ordered=r["state"]["cut_effect_order"]
        self.assertTrue(ordered["supports_current_operation"])
        self.assertEqual(ordered["baseline_ref"]["timestamp"],2.)
        self.assertEqual(ordered["operation_ref"],{"timestamp":3.,"ref":"obs-000075"})
        self.assertEqual(ordered["first_change"]["timestamp"],4.)
        self.assertEqual(ordered["first_change"]["operation_timestamp"],3.)
        usage=r["memory_usage"]
        self.assertLessEqual(usage["effect_layout_records"],2)
        self.assertLessEqual(usage["effect_layout_boxes"],8)
        self.assertLessEqual(usage["effect_evidence_refs"],6)
        self.assertLessEqual(usage["effect_state_chars"],6000)

    def test_confirmed_first_change_does_not_drift_after_later_layout_noise(self):
        self.ordered_start();self.held(4,"yes");self.held(5,"yes")
        self.held(6,"no")
        r=self.held(7,"yes")
        self.assertTrue(r["state"]["preconditions"]["cut_effect_confirmed"])
        self.assertEqual(r["state"]["cut_effect_order"]["first_change"]["timestamp"],4.)
        self.assertEqual(r["state"]["cut_effect_order"]["reason"],"historical_effect_already_confirmed")
        r=self.feed(8,.005,valid=False,gripper="closed",cut_piece="unknown")
        self.assertTrue(r["state"]["preconditions"]["cut_effect_confirmed"])
        self.assertIsNone(r["state"]["preconditions"]["reliable_grasp"])
        self.assertEqual(r["state"]["cut_effect_order"]["first_change"]["timestamp"],4.)
        self.assertEqual(r["state"]["cut_effect_order"]["reason"],"historical_confirmed_effect_current_observation_unavailable")


if __name__=="__main__":
    unittest.main()
