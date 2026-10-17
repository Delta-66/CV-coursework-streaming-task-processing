import unittest
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from streaming_recovery.evaluation import (Outcome, parse_outcome, notification_kind,
                                          outcome_statistics, replay_audit, check_manual_moments,
                                          conditional_direction_checks, row_fingerprint)


def row(state, status, reason, notify=True):
    return {"notify": notify, "notification_reason": reason, "execution_status": status,
            "failure_event": {"event_id": "event-0001", "state": state}}


class EvaluationTests(unittest.TestCase):
    def test_explicit_label_enums(self):
        self.assertIs(parse_outcome("success"), Outcome.SUCCESS)
        self.assertIs(parse_outcome("failure"), Outcome.FAILURE)
        for invalid in (True, False, None, 0, 1, "", "unknown", "SUCCESS", "failure "):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_outcome(invalid)

    def test_result_combinations_are_not_string_truthiness(self):
        confirmed = [row("confirmed", "confirmed_failure", "evidence_confirmed")]
        for label, records, missed, false_alarm in (
            ("success", [], None, False), ("success", confirmed, None, True),
            ("failure", [], True, None), ("failure", confirmed, False, None)):
            with self.subTest(label=label, records=bool(records)):
                result = outcome_statistics(label, records, None)
                self.assertEqual(result["official_outcome"], label)
                self.assertEqual(result["failure_missed_at_episode_level"], missed)
                self.assertEqual(result["false_failure_alarm_on_normal"], false_alarm)

    def test_suspicion_does_not_count_as_confirmation(self):
        records = [row("suspected", "suspicious", "new_suspicion")]
        failure = outcome_statistics("failure", records, None)
        self.assertTrue(failure["failure_missed_at_episode_level"])
        self.assertEqual(failure["notification_counts"]["suspicion"], 1)
        normal = outcome_statistics("success", records, None)
        self.assertTrue(normal["false_failure_alarm_on_normal"])
        self.assertFalse(normal["false_confirmed_failure_on_normal"])

    def test_recovery_retry_and_silent_persistence(self):
        records = [row("resolved", "recovered", "recovery_observed_without_execution_attribution"),
                   row("closed_for_retry", "executing", "new_attempt_observed")]
        stats = outcome_statistics("success", records, True)
        self.assertFalse(stats["false_failure_alarm_on_normal"])
        self.assertEqual(stats["notification_counts"]["recovery"], 1)
        self.assertEqual(stats["notification_counts"]["retry"], 1)
        silent = row("persistent", "confirmed_failure", None, False)
        self.assertIsNone(notification_kind(silent))
        self.assertTrue(outcome_statistics("failure", [silent], None)["confirmed_failure"])

    def test_invalid_label_is_rejected_before_counting(self):
        with self.assertRaises(ValueError):
            outcome_statistics("maybe", [], None)

    def test_metadata_short_read_and_camera_truncation_fail(self):
        audit = {"rows_read": 100, "frames_decoded": {"cam_high": 100, "cam_right_wrist": 100}, "invalid_video_timestamps": 0}
        args = (100, 100, audit, [0., 1., 2., 3.], 25, 25, ["cam_high", "cam_right_wrist"])
        self.assertTrue(replay_audit(*args)["full_replay"])
        self.assertFalse(replay_audit(101, *args[1:])["full_replay"])
        audit["rows_read"] = 99
        self.assertFalse(replay_audit(*args)["full_replay"])
        audit["rows_read"] = 100
        audit["frames_decoded"]["cam_right_wrist"] = 99
        self.assertFalse(replay_audit(*args)["full_replay"])

    def test_full_native_read_does_not_imply_decision_coverage(self):
        audit = {"rows_read": 100, "frames_decoded": {"a": 100, "b": 100}, "invalid_video_timestamps": 0}
        result = replay_audit(100, 100, audit, [0., 1., 2.], 25, 25, ["a", "b"])
        self.assertTrue(result["full_replay"])
        self.assertFalse(result["decision_sample_coverage"])

    def test_phase_checks_use_timestamp_and_predicates(self):
        rows = [{"timestamp": 2., "state": {"stage": "locate_tool", "attempt": 0}}]
        checks = [{"timestamp": 2., "expected_stages": ["align_grasp"], "expected_fields": {"state.attempt": 1}},
                  {"timestamp": 5., "expected_stages": ["locate_tool"]}]
        result = check_manual_moments(rows, checks)
        self.assertEqual(result[0]["status"], "gap")
        self.assertEqual(len(result[0]["differences"]), 2)
        self.assertEqual(result[1]["status"], "not_observed")

    def test_phase_window_does_not_hide_failed_all_or_invalid_expectations(self):
        rows = [{"timestamp": 1., "state": {"stage": "verify_grasp"}},
                {"timestamp": 2., "state": {"stage": "cutting"}}]
        check = {"start_s": 1., "end_s": 2., "expected_stages": ["cutting"]}
        self.assertEqual(check_manual_moments(rows, [check])[0]["status"], "passed")
        self.assertEqual(check_manual_moments(rows, [{**check, "mode": "all"}])[0]["status"], "gap")
        with self.assertRaises(ValueError):
            check_manual_moments(rows, [{"timestamp": 1.}])

    def test_conditional_direction_binding_and_visual_review_are_required(self):
        """纯契约故障注入；测试文件与坐标是合成输入，不是视觉识别验证。"""
        row = {"timestamp": 2., "observation_ref": "obs-000050",
               "visual_facts": {"valid": True, "effect_usable": True, "visibility": "clear",
                   "provenance": "real_multiview_effect_geometry", "gripper": "open", "contact": "separate",
                   "knife_handle_xy": [400., 700.], "gripper_xy": [500., 700.],
                   "localization_frame": "cam_right_wrist_image_x_right_y_down",
                   "measurements": {"localization": {"status": "localized", "method": "fixture_only"}}},
               "modalities": {"cam_right_wrist": {"status": "available", "timestamp": 2.}},
               "decision_rules": {"direction_deadband_normalized": 50}}
        relative = "trajectory_000002/frames/obs-000050_cam_right_wrist.jpg"
        image = b"synthetic contract fixture; not real image validation"
        record = {"episode_for_audit_only": 2, "timestamp": 2., "observation_ref": "obs-000050",
                  "row_sha256": row_fingerprint(row), "image_path": relative,
                  "image_sha256": hashlib.sha256(image).hexdigest(), "camera": "cam_right_wrist",
                  "localization_frame": row["visual_facts"]["localization_frame"],
                  "knife_handle_xy": [400., 700.], "gripper_xy": [500., 700.], "localization_method": "fixture_only",
                  "expected_directions": ["图像左侧"], "reviewer": "synthetic_test",
                  "observed_features": "synthetic fixture only", "reviewed_handle_neck": True,
                  "reviewed_two_finger_edges": True, "coordinate_claim": "image_plane_only_no_robot_mapping"}
        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory)/relative
            path.parent.mkdir(parents=True)
            path.write_bytes(image)
            audit = lambda item: conditional_direction_checks({2: [row]}, directory, [item])[0]
            result = audit(record)
            self.assertEqual(result["status"], "passed")
            self.assertFalse(result["natural_alignment_action"])
            self.assertFalse(result["actually_executed"])
            self.assertIsNone(result["generated_direction"]["numeric_command"])
            for key, invalid in (("row_sha256", "bad"), ("image_sha256", "bad"),
                                 ("image_path", "../wrong.jpg"), ("knife_handle_xy", [100., 700.])):
                with self.subTest(key=key):
                    self.assertEqual(audit({**record, key: invalid})["status"], "invalid")
            for key, missing in (("reviewer", ""), ("observed_features", ""), ("reviewed_handle_neck", False),
                                 ("reviewed_two_finger_edges", False), ("expected_directions", ["图像右侧"])):
                with self.subTest(key=key):
                    self.assertEqual(audit({**record, key: missing})["status"], "gap")


if __name__ == "__main__":
    unittest.main()
