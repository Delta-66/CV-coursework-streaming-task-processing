"""Offline evaluation tests using serialized fixtures, never data downloads or GPUs."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evaluate_temporal_benchmark import evaluate, fingerprint, sample_indices, sha256


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class TemporalEvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.metadata = self.root / "metadata"
        self.splits = self.root / "splits"
        self.benchmark = self.root / "benchmark"
        self.lengths = {index: 101 if index != 499 else 18 for index in range(500)}
        episodes = [{"episode_index": index, "length": length, "tasks": ["Synthetic task"]}
                    for index, length in self.lengths.items()]
        annotations = [{"episode_index": index, "num_frames": length,
                        "episode_success": "failure" if index % 2 == 0 else "success",
                        "avoid_keyframe_indices": [25] if index % 2 == 0 else [],
                        "correct_keyframe_indices": [75] if index % 2 == 0 else []}
                       for index, length in self.lengths.items()]
        write_json(self.metadata / "meta/info.json", {"fps": 25})
        write_jsonl(self.metadata / "meta/episodes.jsonl", episodes)
        write_jsonl(self.metadata / "meta/vifailback_annotations.jsonl", annotations)
        write_jsonl(self.splits / "test.jsonl", [{"episode_index": index} for index in range(500)])
        split_protocol = {
            "canonical_detection_benchmark_episode_indices": list(range(500)),
            "artifacts": {"test.jsonl": {"sha256": sha256(self.splits / "test.jsonl")}},
            "sources": {"metadata": {"files": {name: {"sha256": sha256(self.metadata / name)}
                         for name in ("meta/info.json", "meta/episodes.jsonl", "meta/vifailback_annotations.jsonl")}}},
        }
        write_json(self.splits / "protocol.json", split_protocol)
        self.config = {
            "source": {"format": "lerobot_v2.1"},
            "monitor": {"sample_every": 25, "include_last_frame": True,
                        "max_model_calls": None, "max_samples_per_episode": None, "history_size": 2},
            "tracking": {"failure_confirmation": 2}, "model": {"device": "cuda:0"},
        }
        self.protocol = {
            "episode_indices": list(range(500)), "sample_every": 25, "include_last_frame": True,
            "config": self.config, "test_backend": False,
            "split_manifest_sha256": sha256(self.splits / "test.jsonl"),
            "split_protocol_sha256": sha256(self.splits / "protocol.json"), "created_at": "fixture",
        }
        self.protocol["protocol_sha256"] = fingerprint({key: value for key, value in self.protocol.items()
                                                       if key != "created_at"})
        write_json(self.benchmark / "protocol.json", self.protocol)
        self.plan = {"episodes": [{"episode_index": index, "num_frames": length,
                                   "expected_model_calls": len(sample_indices(length, 25, True))}
                                  for index, length in self.lengths.items()],
                     "total_frames": sum(self.lengths.values()), "total_model_calls": 2497}
        write_json(self.benchmark / "plan.json", self.plan)
        self.protocol.update(plan_sha256=fingerprint(self.plan), prompt_sha256="fixture-prompt",
                             model={"repo_id": "fixture/model", "revision": "fixed", "files": []},
                             program_sha256={"temporal_diagnosis.py": "fixture-code"})
        self.protocol["protocol_sha256"] = fingerprint({key: value for key, value in self.protocol.items()
                                                       if key not in ("protocol_sha256", "created_at")})
        write_json(self.benchmark / "protocol.json", self.protocol)
        self.checkpoint = {"status": "running", "protocol_sha256": self.protocol["protocol_sha256"],
                           "episodes": {str(index): {"status": "pending"} for index in range(500)}}
        self.save_checkpoint()

    def save_checkpoint(self):
        write_json(self.benchmark / "checkpoint.json", self.checkpoint)

    def completed(self, index, alerts=(), invalid=(), uncertain=(), duplicate_incident=False,
                  all_invalid_failure=False):
        """Serialize production-shaped records; no backend/model is executed."""
        run_dir = self.benchmark / "episodes" / str(index)
        config = deepcopy(self.config)
        config["source"]["episodes"] = [index]
        frames = sample_indices(self.lengths[index], 25, True)
        rows = []
        for frame in frames:
            valid = not all_invalid_failure and frame not in invalid
            alert = valid and frame in alerts
            status = "failure" if alert else "insufficient_evidence" if frame in uncertain else "in_progress"
            incident = 1 if duplicate_incident else list(alerts).index(frame) + 1 if alert else None
            result = {"status": status, "failure": True if alert else None if status == "insufficient_evidence" else False}
            rows.append({"episode_index": index, "frame_index": frame, "timestamp": frame / 25,
                         "result_valid": valid, "result": result if valid else None,
                         "monitor": {"alert": alert, "incident_id": incident if alert else None},
                         "metrics": {"compute_device": "cuda:0", "inference_ms": 100, "gpu_wait_time_s": 0}})
        alert_rows = [row for row in rows if row["monitor"]["alert"]]
        summary = {"status": "monitor_run_failed" if all_invalid_failure else "monitor_run_complete",
                   "model_calls": len(rows), "valid_outputs": sum(row["result_valid"] for row in rows),
                   "invalid_outputs": sum(not row["result_valid"] for row in rows), "failure_alerts": len(alert_rows),
                   "episodes": [{"episode_index": index, "frames_seen": self.lengths[index],
                                 "model_calls": len(rows), "source_frames": self.lengths[index], "full_episode_read": True}]}
        write_json(run_dir / "config.json", config)
        write_json(run_dir / "summary.json", summary)
        write_json(run_dir / "provenance.json", {"offline_annotations_used": False, "recorded_targets_used": False,
                                               "cpu_fallback": False, "compute_device": "cuda:0",
                                               "prompt_sha256": self.protocol["prompt_sha256"],
                                               "model": self.protocol["model"],
                                               "program_sha256": self.protocol["program_sha256"]})
        write_jsonl(run_dir / "predictions.jsonl", rows)
        write_jsonl(run_dir / "alerts.jsonl", alert_rows)
        entry = {"status": "failed" if all_invalid_failure else "complete",
                 "run_dir": run_dir.relative_to(self.benchmark).as_posix(), "expected_model_calls": len(rows),
                 "artifact_sha256": {name: sha256(run_dir / name) for name in
                                     ("summary.json", "config.json", "provenance.json", "predictions.jsonl", "alerts.jsonl")}}
        self.checkpoint["episodes"][str(index)] = entry
        self.save_checkpoint()
        return run_dir, rows

    def audit(self):
        with patch("builtins.print"):
            return evaluate(self.benchmark, self.metadata, self.splits)

    def update_hash(self, index, filename):
        entry = self.checkpoint["episodes"][str(index)]
        entry["artifact_sha256"][filename] = sha256(self.benchmark / entry["run_dir"] / filename)
        self.save_checkpoint()

    def test_interim_metrics_use_only_fully_executed_episodes(self):
        self.completed(0, alerts=(50,))
        self.completed(1)
        self.completed(2)
        self.completed(3, alerts=(50,))
        self.checkpoint["episodes"]["4"]["status"] = "running"
        self.checkpoint["episodes"]["5"]["status"] = "interrupted"
        self.save_checkpoint()
        report = self.audit()
        self.assertEqual(report["status"], "interim")
        self.assertEqual(report["coverage"]["fully_executed_episodes"], 4)
        self.assertEqual(report["coverage"]["excluded_episodes"], 496)
        detection = report["metrics"]["episode_failure_detection_rate"]
        self.assertEqual((detection["numerator"], detection["denominator"], detection["value"]), (1, 2, 0.5))
        self.assertEqual(detection["excluded_from_denominator"], 248)
        false_alarm = report["metrics"]["success_episode_false_alarm_rate"]
        self.assertEqual((false_alarm["numerator"], false_alarm["denominator"]), (1, 2))
        self.assertEqual(report["plan"]["total_model_calls"], 2497)
        self.assertEqual(report["plan"]["too_short_for_confirmation_episode_indices"], [499])

    def test_fully_executed_all_invalid_failure_is_not_silently_excluded(self):
        self.completed(0, alerts=(50,))
        self.completed(2, all_invalid_failure=True)
        self.completed(1, uncertain=(0, 25, 50, 75, 100))
        report = self.audit()
        self.assertEqual(report["coverage"]["fully_executed_episodes"], 3)
        self.assertEqual(report["coverage"]["successful_complete_runs"], 2)
        self.assertEqual(report["coverage"]["all_invalid_full_execution_failures"], 1)
        detection = report["metrics"]["episode_failure_detection_rate"]
        self.assertEqual((detection["numerator"], detection["denominator"]), (1, 2))
        self.assertEqual(report["metrics"]["invalid_output_calls"], 5)
        self.assertEqual(report["metrics"]["entirely_abstaining_episodes"], 2)
        invalid_episode = next(row for row in report["episodes"] if row["episode_index"] == 2)
        self.assertFalse(invalid_episode["failure_detected"])
        self.assertEqual(invalid_episode["execution_class"], "all_invalid_full_execution_failure")

    def test_no_complete_run_has_unavailable_rates_not_zero_accuracy(self):
        self.checkpoint["episodes"]["0"] = {"status": "failed"}
        self.save_checkpoint()
        report = self.audit()
        self.assertIsNone(report["metrics"]["episode_failure_detection_rate"]["value"])
        self.assertIsNone(report["metrics"]["success_episode_false_alarm_rate"]["value"])
        self.assertEqual(report["coverage"]["fully_executed_episodes"], 0)
        self.assertEqual(report["coverage"]["checkpoint_status_counts"]["failed"], 1)
        self.assertEqual(report["excluded"][0]["status"], "failed_not_fully_verified")
        self.assertIn("unavailable", report["unavailable_metrics"]["stage_accuracy"])

    def test_keyframe_offset_is_signed_and_missing_detections_are_not_zero_delay(self):
        self.completed(0, alerts=(50,))
        self.completed(2)
        report = self.audit()
        offset = report["correct_keyframe_reference_offset"]
        self.assertEqual(offset["matched_detected_failed_episodes"], 1)
        self.assertEqual(offset["negative_offsets"], 1)
        self.assertEqual(offset["mean_signed_offset_s"], -1.0)
        self.assertIn("not the first detectable", offset["limitation"])
        self.assertIsNone(next(row for row in report["episodes"] if row["episode_index"] == 2)["first_alert_minus_correct_keyframe_s"])

    def test_alert_counts_and_repeated_incident_ids_are_not_real_event_ground_truth(self):
        self.completed(0, alerts=(50, 75), duplicate_incident=True)
        self.completed(2, alerts=(50, 75))
        report = self.audit()
        self.assertEqual(report["metrics"]["total_alerts"], 4)
        self.assertEqual(report["metrics"]["duplicate_same_incident_alerts"], 1)
        self.assertEqual(report["metrics"]["additional_alerts_after_first_per_episode"], 2)
        self.assertIn("unavailable", report["unavailable_metrics"]["failure_event_recall_and_true_duplicate_event_rate"])

    def test_hash_tampering_and_incomplete_sampling_never_enter_denominators(self):
        run_dir, rows = self.completed(0, alerts=(50,))
        rows.pop(1)
        write_jsonl(run_dir / "predictions.jsonl", rows)
        report = self.audit()
        self.assertEqual(report["coverage"]["fully_executed_episodes"], 0)
        self.assertIn("hash differs", report["excluded"][0]["reason"])
        self.update_hash(0, "predictions.jsonl")
        report = self.audit()
        self.assertEqual(report["coverage"]["fully_executed_episodes"], 0)
        self.assertIn("sampled frames", report["excluded"][0]["reason"])

    def test_partial_all_invalid_attempt_is_still_excluded(self):
        run_dir, rows = self.completed(0, all_invalid_failure=True)
        write_jsonl(run_dir / "predictions.jsonl", rows[:-1])
        self.update_hash(0, "predictions.jsonl")
        report = self.audit()
        self.assertEqual(report["coverage"]["all_invalid_full_execution_failures"], 0)
        self.assertEqual(report["metrics"]["episode_failure_detection_rate"]["denominator"], 0)

    def test_protocol_or_synthetic_backend_cannot_claim_formal_results(self):
        self.protocol["sample_every"] = 100
        write_json(self.benchmark / "protocol.json", self.protocol)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.audit()
        self.protocol["sample_every"] = 25
        self.protocol["test_backend"] = True
        payload = {key: value for key, value in self.protocol.items() if key not in ("protocol_sha256", "created_at")}
        self.protocol["protocol_sha256"] = fingerprint(payload)
        write_json(self.benchmark / "protocol.json", self.protocol)
        with self.assertRaisesRegex(ValueError, "Synthetic backends"):
            self.audit()

    def test_plan_and_offline_metadata_must_match_frozen_sources(self):
        self.plan["total_model_calls"] -= 1
        write_json(self.benchmark / "plan.json", self.plan)
        with self.assertRaisesRegex(ValueError, "plan fingerprint"):
            self.audit()
        self.plan["total_model_calls"] += 1
        write_json(self.benchmark / "plan.json", self.plan)
        path = self.metadata / "meta/vifailback_annotations.jsonl"
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Metadata differs"):
            self.audit()

    def test_final_frame_sampling_has_no_duplicates(self):
        self.assertEqual(sample_indices(51, 25, True), [0, 25, 50])
        self.assertEqual(sample_indices(50, 25, True), [0, 25, 49])
        self.assertEqual(sample_indices(18, 25, True), [0, 17])


if __name__ == "__main__":
    unittest.main()
