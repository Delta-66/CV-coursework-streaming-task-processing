"""Behavioral checks for causal, bounded, evidence-gated diagnosis tracking."""
from copy import deepcopy
import unittest

from diagnostic_tracking import DiagnosticTracker, FAILURE_TYPES, KEY_STATE_VALUES


def observation(frame_index, stage="grasping", failure_type=None, achieved=False):
    failure = failure_type is not None
    return {
        "stage": stage, "status": "failure" if failure else "in_progress", "failure": failure,
        "evidence": ["The gripper is closed beside the object." if failure else "The object is held above the table."],
        "high_level_advice": "Reposition and grasp." if failure else None,
        "low_level_advice": {"action": "Align with the object.", "reference_frame": "image",
                             "completion_check": "Check the grip."} if failure else None,
        "key_states": {"grasp": "not_held" if failure else "held", "target_relation": "unknown", "task_goal": "unknown"},
        "expected_effect": "The object is held.", "observed_effect": "The gripper is beside the object." if failure else "The object is held.",
        "action_outcome": "not_achieved" if failure else "achieved" if achieved else "uncertain",
        "failure_type": failure_type or "none", "evidence_frame_indices": [frame_index],
    }


def uncertain(frame_index):
    result = observation(frame_index, stage="unknown")
    result.update(status="insufficient_evidence", failure=None, evidence=[], evidence_frame_indices=[])
    result["key_states"] = {name: "unknown" for name in KEY_STATE_VALUES}
    return result


class DiagnosticTrackerTests(unittest.TestCase):
    def test_stage_needs_consecutive_support_and_stale_is_explicit(self):
        tracker = DiagnosticTracker()
        first = tracker.update(0, 0, 0, observation(0))
        self.assertEqual(first["tracked_stage"], "unknown")
        self.assertFalse(first["stage_confirmed"])
        confirmed = tracker.update(0, 1, 1, observation(1))
        self.assertEqual(confirmed["tracked_stage"], "grasping")
        self.assertFalse(confirmed["stage_stale"])
        unknown = tracker.update(0, 2, 2, uncertain(2))
        self.assertEqual(unknown["observed_stage"], "unknown")
        self.assertEqual(unknown["tracked_stage"], "grasping")
        self.assertTrue(unknown["stage_stale"])
        self.assertEqual(unknown["stage_candidate_count"], 0)
        changed = tracker.update(0, 3, 3, observation(3, stage="transporting"))
        self.assertEqual(changed["tracked_stage"], "grasping")
        self.assertTrue(changed["stage_stale"])
        confirmed = tracker.update(0, 4, 4, observation(4, stage="transporting"))
        self.assertEqual(confirmed["tracked_stage"], "transporting")
        self.assertTrue(confirmed["stage_confirmed"])

    def test_one_alert_per_unresolved_incident_even_after_long_gap(self):
        tracker = DiagnosticTracker(cooldown_s=1)
        first = tracker.update(0, 0, 0, observation(0, failure_type="grasp_miss"))
        self.assertFalse(first["alert"])
        self.assertEqual(first["failure_state"], "suspected")
        second = tracker.update(0, 1, 1, observation(1, failure_type="grasp_miss"))
        self.assertTrue(second["alert"])
        third = tracker.update(0, 2, 999, observation(2, failure_type="grasp_miss"))
        self.assertFalse(third["alert"])
        self.assertEqual(third["incident_id"], second["incident_id"])
        self.assertEqual(third["suppression_reason"], "duplicate_unresolved_incident")

    def test_key_states_need_independent_consecutive_confirmation(self):
        tracker = DiagnosticTracker()
        first = observation(0)
        first["key_states"] = {"grasp": "held", "target_relation": "away_from_target", "task_goal": "not_met"}
        state = tracker.update(0, 0, 0, first)
        self.assertTrue(all(state["key_state_stale"].values()))
        self.assertEqual(state["tracked_key_states"], {name: "unknown" for name in KEY_STATE_VALUES})
        second = deepcopy(first)
        second["evidence_frame_indices"] = [1]
        second["key_states"]["target_relation"] = "at_target"
        state = tracker.update(0, 1, 1, second)
        self.assertEqual(state["tracked_key_states"], {"grasp": "held", "target_relation": "unknown", "task_goal": "not_met"})
        self.assertEqual(state["key_state_stale"], {"grasp": False, "target_relation": True, "task_goal": False})
        self.assertEqual(state["key_state_candidate_counts"], {"grasp": 2, "target_relation": 1, "task_goal": 2})
        state = tracker.update(0, 2, 2, uncertain(2))
        self.assertEqual(state["tracked_key_states"]["grasp"], "held")
        self.assertTrue(all(state["key_state_stale"].values()))
        state = tracker.update(0, 3, 3, observation(3))
        self.assertEqual(state["key_state_candidate_counts"]["grasp"], 1)
        self.assertTrue(state["key_state_stale"]["grasp"])
        state = tracker.update(0, 4, 4, None)
        self.assertEqual(state["observed_key_states"], {name: "unknown" for name in KEY_STATE_VALUES})
        self.assertEqual(state["tracked_key_states"]["grasp"], "held")
        self.assertEqual(state["key_state_candidate_counts"], {name: 0 for name in KEY_STATE_VALUES})
        self.assertTrue(all(state["key_state_stale"].values()))

    def test_unrelated_achieved_action_cannot_clear_failure(self):
        cases = (
            ("grasp_miss", {"grasp": "not_held", "target_relation": "at_target", "task_goal": "not_met"}),
            ("object_dropped", {"grasp": "unknown", "target_relation": "at_target", "task_goal": "not_met"}),
            ("placement_miss", {"grasp": "held", "target_relation": "away_from_target", "task_goal": "not_met"}),
            ("obstructed", {"grasp": "held", "target_relation": "at_target", "task_goal": "not_met"}),
            ("task_deviation", {"grasp": "held", "target_relation": "at_target", "task_goal": "unknown"}),
            ("other", {"grasp": "held", "target_relation": "at_target", "task_goal": "not_met"}),
        )
        for failure_type, key_states in cases:
            with self.subTest(failure_type=failure_type):
                tracker = DiagnosticTracker(failure_confirmation=1)
                first = tracker.update(0, 0, 0, observation(0, failure_type=failure_type))
                for frame_index in (1, 2, 3):
                    result = observation(frame_index, achieved=True)
                    result["key_states"] = key_states.copy()
                    state = tracker.update(0, frame_index, frame_index, result)
                    self.assertTrue(state["recovery_action_supported"])
                    self.assertFalse(state["recovery_key_state_matches"][failure_type])
                    self.assertFalse(state["recovery_evidence_supported"])
                    self.assertFalse(state["recovery_confirmed"])
                    self.assertEqual(state["incident_id"], first["incident_id"])
                    self.assertEqual(state["recovery_count"], 0)

    def test_recovery_requires_matching_all_unresolved_failures(self):
        tracker = DiagnosticTracker(failure_confirmation=1, cooldown_s=0)
        first = tracker.update(0, 0, 0, observation(0, failure_type="object_dropped"))
        second = tracker.update(0, 1, 1, observation(1, failure_type="placement_miss"))
        for frame_index in (2, 3):
            state = tracker.update(0, frame_index, frame_index, observation(frame_index, achieved=True))
            self.assertEqual(state["recovery_key_state_matches"], {"object_dropped": True, "placement_miss": False})
            self.assertEqual(state["unresolved_failure_types"], ["object_dropped", "placement_miss"])
            self.assertFalse(state["recovery_confirmed"])
        for frame_index in (4, 5):
            result = observation(frame_index, achieved=True)
            result["key_states"]["target_relation"] = "at_target"
            state = tracker.update(0, frame_index, frame_index, result)
        self.assertTrue(state["recovery_confirmed"])
        self.assertEqual(state["recovered_incident_ids"], [first["incident_id"], second["incident_id"]])
        self.assertEqual(state["recovery_policy"], "uncalibrated_key_state_heuristic")

    def test_task_goal_met_can_recover_every_type_with_current_evidence(self):
        for failure_type in FAILURE_TYPES:
            with self.subTest(failure_type=failure_type):
                tracker = DiagnosticTracker(failure_confirmation=1)
                tracker.update(0, 0, 0, observation(0, failure_type=failure_type))
                for frame_index in (1, 2):
                    result = observation(frame_index, stage="completed", achieved=True)
                    result["status"] = "completed"
                    result["key_states"] = {"grasp": "unknown", "target_relation": "unknown", "task_goal": "met"}
                    state = tracker.update(0, frame_index, frame_index, result)
                self.assertTrue(state["recovery_confirmed"])
                self.assertEqual(state["unresolved_failure_types"], [])

    def test_stale_matching_key_state_cannot_be_used_as_current_recovery_evidence(self):
        tracker = DiagnosticTracker(failure_confirmation=1, cooldown_s=0)
        tracker.update(0, 0, 0, observation(0))
        state = tracker.update(0, 1, 1, observation(1))
        self.assertEqual(state["tracked_key_states"]["grasp"], "held")
        result = observation(2, failure_type="object_dropped")
        result["key_states"]["grasp"] = "unknown"
        tracker.update(0, 2, 2, result)
        for frame_index in (3, 4):
            result = observation(frame_index, achieved=True)
            result["key_states"]["grasp"] = "unknown"
            state = tracker.update(0, frame_index, frame_index, result)
            self.assertEqual(state["tracked_key_states"]["grasp"], "held")
            self.assertTrue(state["key_state_stale"]["grasp"])
            self.assertFalse(state["recovery_evidence_supported"])
            self.assertFalse(state["recovery_confirmed"])

    def test_uncertainty_invalid_and_plain_progress_cannot_recover(self):
        tracker = DiagnosticTracker(failure_confirmation=1)
        alerted = tracker.update(0, 0, 0, observation(0, failure_type="grasp_miss"))
        for frame_index, result in enumerate((uncertain(1), None, observation(3)), 1):
            state = tracker.update(0, frame_index, frame_index, result)
            self.assertEqual(state["incident_id"], alerted["incident_id"])
            self.assertEqual(state["failure_state"], "confirmed")
            self.assertFalse(state["recovery_confirmed"])
            self.assertFalse(state["alert"])

    def test_recovery_needs_consecutive_current_evidence_and_allows_new_event(self):
        tracker = DiagnosticTracker(cooldown_s=0)
        tracker.update(0, 0, 0, observation(0, failure_type="object_dropped"))
        first = tracker.update(0, 1, 1, observation(1, failure_type="object_dropped"))
        pending = tracker.update(0, 2, 2, observation(2, achieved=True))
        self.assertEqual(pending["failure_state"], "recovery_pending")
        tracker.update(0, 3, 3, uncertain(3))
        pending = tracker.update(0, 4, 4, observation(4, achieved=True))
        self.assertFalse(pending["recovery_confirmed"])
        recovered = tracker.update(0, 5, 5, observation(5, achieved=True))
        self.assertTrue(recovered["recovery_confirmed"])
        self.assertEqual(recovered["recovered_incident_ids"], [first["incident_id"]])
        self.assertIsNone(recovered["active_failure_type"])
        tracker.update(0, 6, 6, observation(6, failure_type="object_dropped"))
        second = tracker.update(0, 7, 7, observation(7, failure_type="object_dropped"))
        self.assertTrue(second["alert"])
        self.assertNotEqual(first["incident_id"], second["incident_id"])

    def test_different_failure_waits_for_source_time_cooldown(self):
        tracker = DiagnosticTracker(cooldown_s=8)
        tracker.update(0, 0, 0, observation(0, failure_type="grasp_miss"))
        first = tracker.update(0, 1, 1, observation(1, failure_type="grasp_miss"))
        tracker.update(0, 2, 2, observation(2, failure_type="placement_miss"))
        pending = tracker.update(0, 3, 3, observation(3, failure_type="placement_miss"))
        self.assertEqual(pending["suppression_reason"], "cooldown")
        self.assertFalse(pending["alert"])
        self.assertEqual(pending["cooldown_remaining_s"], 6)
        self.assertNotEqual(first["incident_id"], pending["incident_id"])
        pending = tracker.update(0, 4, 8.9, observation(4, failure_type="placement_miss"))
        self.assertFalse(pending["alert"])
        second = tracker.update(0, 5, 9, observation(5, failure_type="placement_miss"))
        self.assertTrue(second["alert"])
        tracker.update(0, 6, 18, observation(6, failure_type="grasp_miss"))
        repeated = tracker.update(0, 7, 19, observation(7, failure_type="grasp_miss"))
        self.assertEqual(repeated["incident_id"], first["incident_id"])
        self.assertFalse(repeated["alert"])

    def test_invalid_output_breaks_candidates_without_inventing_observation(self):
        tracker = DiagnosticTracker(cooldown_s=0)
        tracker.update(0, 0, 0, observation(0, failure_type="obstructed"))
        invalid = tracker.update(0, 1, 1, None)
        self.assertEqual(invalid["observed_stage"], "unknown")
        self.assertEqual(invalid["candidate_count"], 0)
        self.assertEqual(invalid["stage_candidate_count"], 0)
        self.assertEqual(invalid["suppression_reason"], "invalid_model_output")
        self.assertEqual(invalid["failure_state"], "unknown")
        resumed = tracker.update(0, 2, 2, observation(2, failure_type="obstructed"))
        self.assertFalse(resumed["alert"])
        confirmed = tracker.update(0, 3, 3, observation(3, failure_type="obstructed"))
        self.assertTrue(confirmed["alert"])

    def test_pending_cooldown_never_emits_without_fresh_support(self):
        tracker = DiagnosticTracker(failure_confirmation=1, cooldown_s=8)
        tracker.update(0, 0, 0, observation(0, failure_type="grasp_miss"))
        pending = tracker.update(0, 1, 1, observation(1, failure_type="object_dropped"))
        self.assertEqual(pending["suppression_reason"], "cooldown")
        after_cooldown = tracker.update(0, 2, 20, uncertain(2))
        self.assertFalse(after_cooldown["alert"])
        self.assertEqual(after_cooldown["failure_state"], "confirmed")
        fresh = tracker.update(0, 3, 21, observation(3, failure_type="object_dropped"))
        self.assertTrue(fresh["alert"])
        self.assertEqual(fresh["incident_id"], pending["incident_id"])

    def test_same_failure_after_recovery_still_obeys_cooldown(self):
        tracker = DiagnosticTracker(failure_confirmation=1, recovery_confirmation=1, cooldown_s=8)
        first = tracker.update(0, 0, 0, observation(0, failure_type="grasp_miss"))
        recovered = tracker.update(0, 1, 1, observation(1, achieved=True))
        self.assertTrue(recovered["recovery_confirmed"])
        pending = tracker.update(0, 2, 2, observation(2, failure_type="grasp_miss"))
        self.assertFalse(pending["alert"])
        self.assertNotEqual(pending["incident_id"], first["incident_id"])
        resumed = tracker.update(0, 3, 8, observation(3, failure_type="grasp_miss"))
        self.assertTrue(resumed["alert"])
        self.assertEqual(resumed["incident_id"], pending["incident_id"])

    def test_failure_type_change_restarts_confirmation(self):
        tracker = DiagnosticTracker()
        tracker.update(0, 0, 0, observation(0, failure_type="grasp_miss"))
        state = tracker.update(0, 1, 1, observation(1, failure_type="object_dropped"))
        self.assertEqual(state["candidate_count"], 1)
        self.assertFalse(state["alert"])

    def test_missing_or_future_evidence_and_uncertain_outcome_cannot_alert(self):
        changes = ({"evidence_frame_indices": []}, {"evidence_frame_indices": [0]},
                   {"evidence_frame_indices": [1, 2]}, {"evidence": []},
                   {"evidence": ["  "]}, {"action_outcome": "uncertain"},
                   {"failure_type": "unknown"}, {"failure_type": "none"})
        for change in changes:
            with self.subTest(change=change):
                tracker = DiagnosticTracker(failure_confirmation=1)
                result = observation(1, failure_type="grasp_miss")
                result.update(change)
                state = tracker.update(0, 1, 1, result)
                self.assertFalse(state["alert"])
                self.assertEqual(state["candidate_count"], 0)

    def test_past_only_recovery_evidence_cannot_clear_incident(self):
        tracker = DiagnosticTracker(failure_confirmation=1, recovery_confirmation=1)
        tracker.update(0, 0, 0, observation(0, failure_type="obstructed"))
        result = observation(1, achieved=True)
        result["evidence_frame_indices"] = [0]
        state = tracker.update(0, 1, 1, result)
        self.assertFalse(state["recovery_confirmed"])
        self.assertEqual(state["failure_state"], "confirmed")

    def test_time_and_frame_regressions_rejected_without_state_mutation(self):
        for frame_index, timestamp in ((2, 1), (1, 2), (0, 3), (3, 0)):
            with self.subTest(frame_index=frame_index, timestamp=timestamp):
                tracker = DiagnosticTracker()
                tracker.update(0, 1, 1, observation(1))
                saved = deepcopy(tracker.__dict__)
                with self.assertRaises(ValueError):
                    tracker.update(0, frame_index, timestamp, observation(frame_index))
                self.assertEqual(tracker.__dict__, saved)

    def test_episode_change_resets_stage_failure_and_cooldown(self):
        tracker = DiagnosticTracker(stage_confirmation=1, failure_confirmation=1)
        tracker.update(0, 8, 8, observation(8, failure_type="grasp_miss"))
        next_episode = tracker.update(1, 0, 0, uncertain(0))
        self.assertEqual(next_episode["tracked_stage"], "unknown")
        self.assertEqual(next_episode["unresolved_failure_types"], [])
        self.assertEqual(next_episode["cooldown_remaining_s"], 0)
        self.assertEqual(next_episode["tracked_key_states"], {name: "unknown" for name in KEY_STATE_VALUES})
        self.assertEqual(next_episode["key_state_candidate_counts"], {name: 0 for name in KEY_STATE_VALUES})
        confirmed = tracker.update(1, 1, 1, observation(1, failure_type="grasp_miss"))
        self.assertTrue(confirmed["alert"])

    def test_configuration_and_input_boundaries(self):
        for name in ("stage_confirmation", "failure_confirmation", "recovery_confirmation"):
            for value in (0, -1, 1.5, True, "2"):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    DiagnosticTracker(**{name: value})
        for value in (-1, float("inf"), float("nan"), True, "2"):
            with self.subTest(cooldown=value), self.assertRaises(ValueError):
                DiagnosticTracker(cooldown_s=value)
        tracker = DiagnosticTracker(1, 1, 1, 0)
        for args in ((True, 0, 0, None), (0, -1, 0, None), (0, 0, float("nan"), None),
                     (0, 0, float("inf"), None), (0, 0, -1, None), (0, 0, True, None), (0, 0, 0, [])):
            with self.subTest(args=args), self.assertRaises(ValueError):
                tracker.update(*args)
        state = tracker.update(0, 0, 0, observation(0, failure_type="grasp_miss"))
        self.assertTrue(state["alert"])
        self.assertTrue(state["stage_confirmed"])

    def test_state_size_is_bounded_and_results_are_not_retained_or_mutated(self):
        tracker = DiagnosticTracker(failure_confirmation=1, cooldown_s=0)
        fixed_attribute_names = set(tracker.__dict__)
        for frame_index in range(120):
            failure_type = sorted(FAILURE_TYPES)[frame_index % len(FAILURE_TYPES)]
            result = observation(frame_index, failure_type=failure_type)
            saved = deepcopy(result)
            state = tracker.update(0, frame_index, frame_index, result)
            self.assertEqual(result, saved)
            self.assertLessEqual(len(tracker._incidents), len(FAILURE_TYPES))
            self.assertLessEqual(state["stage_candidate_count"], tracker.stage_confirmation)
            self.assertLessEqual(state["candidate_count"], tracker.failure_confirmation)
            self.assertEqual(set(state["tracked_key_states"]), set(KEY_STATE_VALUES))
            self.assertTrue(all(value <= tracker.stage_confirmation for value in state["key_state_candidate_counts"].values()))
            self.assertEqual(set(tracker.__dict__), fixed_attribute_names)
            self.assertFalse(state["policy_calibrated"])
        self.assertEqual(len(tracker._incidents), len(FAILURE_TYPES))
        state["tracked_key_states"]["grasp"] = "tampered"
        state["key_state_candidates"]["grasp"] = "tampered"
        state["key_state_candidate_counts"]["grasp"] = 999
        self.assertNotEqual(tracker.tracked_key_states["grasp"], "tampered")
        self.assertNotEqual(tracker.key_state_candidates["grasp"], "tampered")
        self.assertLessEqual(tracker.key_state_candidate_counts["grasp"], tracker.stage_confirmation)


if __name__ == "__main__":
    unittest.main()
