"""Reject corrupted monitor evidence using synthetic sources and no model or GPU."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import yaml

from run_task_monitor import run
from verify_task_monitor import read_json, read_jsonl, verify, verify_metrics


class SyntheticSource:
    cameras = ["cam"]
    length = 7

    def __init__(self, episode_index):
        self.episode_index = episode_index

    def iter_frames(self):
        for frame_index in range(self.length):
            yield SimpleNamespace(
                episode_index=self.episode_index, frame_index=frame_index,
                timestamp=frame_index / 25, timestamp_source="synthetic clock",
                task_instruction="Place the block in the tray.",
                images={"cam": np.full((40, 80, 3), frame_index + self.episode_index * 20, np.uint8)},
                observations={"observation.state": np.arange(14, dtype=np.float32) + frame_index,
                              "observation.velocity": np.zeros(14, dtype=np.float32),
                              "observation.effort": np.ones(14, dtype=np.float32)},
                source_references={"cam": {"frame_index": frame_index,
                                           "video_timestamp": frame_index / 25}})


class SyntheticBackend:
    device = "test_backend"

    def predict(self, request):
        current = request.frames[-1]
        if current.episode_index == 1 and current.frame_index == 6:
            return "invalid json", {"compute_device": self.device, "inference_ms": 1.0}
        result = {
            "stage": "unknown", "status": "insufficient_evidence", "failure": None,
            "evidence": [], "high_level_advice": None, "low_level_advice": None,
            "key_states": {"grasp": "unknown", "target_relation": "unknown", "task_goal": "unknown"},
            "expected_effect": "The block should be inside the tray.",
            "observed_effect": "The action effect is unclear.",
            "action_outcome": "uncertain", "failure_type": "unknown", "evidence_frame_indices": [],
        }
        if len(request.frames) > 1:
            result.update(
                stage="placing", status="failure", failure=True,
                evidence=["Synthetic fixture: released block remains outside the tray."],
                high_level_advice="Recover and place the block in the tray.",
                low_level_advice={"action": "Align in the image.", "reference_frame": "image",
                                  "completion_check": "Check alignment."},
                observed_effect="The released block remains outside the target.",
                action_outcome="not_achieved", failure_type="placement_miss",
                evidence_frame_indices=[request.frames[-2].frame_index, current.frame_index])
            result["key_states"] = {"grasp": "not_held", "target_relation": "away_from_target",
                                    "task_goal": "not_met"}
        return json.dumps(result), {"compute_device": self.device, "inference_ms": 1.0}


class TaskMonitorVerifierTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.config_path = self.folder / "config.yaml"
        config = {
            "source": {"format": "synthetic"},
            "monitor": {"camera": "cam", "sample_every": 2, "max_image_edge": 64,
                        "history_size": 2, "max_history_s": 8.0,
                        "max_samples_per_episode": None, "max_model_calls": None},
            "tracking": {"stage_confirmation": 2, "failure_confirmation": 2,
                         "recovery_confirmation": 2, "cooldown_s": 8.0},
            "model": {"path": "unused", "device": "cuda:0", "max_new_tokens": 768,
                      "max_image_pixels": 98304},
            "output_dir": str(self.folder / "runs"),
        }
        self.config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        with patch("run_task_monitor.load_sources", return_value=self.sources()), patch("builtins.print"):
            self.run_dir = run(self.config_path, SyntheticBackend())
        self.predictions_path = self.run_dir / "predictions.jsonl"
        self.rows = read_jsonl(self.predictions_path)

    @staticmethod
    def sources():
        return [SyntheticSource(0), SyntheticSource(1)]

    def audit(self, output=None):
        with patch("verify_task_monitor.load_sources", return_value=self.sources()), patch("builtins.print"):
            return verify(self.run_dir, self.config_path, output)

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    @staticmethod
    def write_jsonl(path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def assert_rejected_without_overwriting_report(self, pattern):
        output = self.folder / "validation.json"
        output.write_text("previous valid report", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, pattern):
            self.audit(output)
        self.assertEqual(output.read_text(encoding="utf-8"), "previous valid report")

    def test_roundtrip_verifies_windows_invalid_output_and_deduplicated_alerts(self):
        output = self.folder / "validation.json"
        report = self.audit(output)
        self.assertEqual(report["validation_purpose"], "test_backend_functional_check")
        self.assertEqual((report["model_calls"], report["valid_outputs"], report["invalid_outputs"]), (8, 7, 1))
        self.assertEqual(report["failure_alerts"], 2)
        self.assertEqual(report["source_images_verified"], 8)
        self.assertEqual(report["window_image_references_verified"], 18)
        self.assertEqual(read_json(output), report)
        self.assertEqual([row["frame_index"] for row in read_jsonl(self.run_dir / "alerts.jsonl")], [4, 4])
        self.assertFalse(self.rows[3]["monitor"]["alert"])
        self.assertFalse(self.rows[-1]["result_valid"])
        self.assertEqual(self.rows[-1]["raw_response"], "invalid json")
        self.assertIn("not rechecked", report["model_weight_integrity"])

    def test_terminal_frame_roundtrip_and_sampling_coverage_cannot_be_forged(self):
        config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        config["monitor"].update(sample_every=4, include_last_frame=True)
        self.config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        with patch("run_task_monitor.load_sources", return_value=self.sources()), patch("builtins.print"):
            self.run_dir = run(self.config_path, SyntheticBackend())
        report = self.audit()
        self.assertEqual(report["model_calls"], 6)
        rows = read_jsonl(self.run_dir / "predictions.jsonl")
        self.assertEqual([row["frame_index"] for row in rows], [0, 4, 6, 0, 4, 6])
        summary_path = self.run_dir / "summary.json"
        summary = read_json(summary_path)
        summary["sampling_coverage"]["expected_model_calls"] = 2
        self.write_json(summary_path, summary)
        self.assert_rejected_without_overwriting_report("Sampling coverage")

    def test_rejects_causal_window_and_original_state_corruption(self):
        changes = [
            ("future timestamp", lambda rows: rows[1]["window"][0].update(timestamp=999), "Window timestamp"),
            ("state value", lambda rows: rows[1]["window"][-1]["observations"]["observation.state"].__setitem__(0, 999), "Window observations"),
            ("gripper value", lambda rows: rows[1]["window"][-1]["gripper_state"].update(left=999), "Window gripper_state"),
            ("wrong past frame", lambda rows: rows[2]["window"][0].update(frame_index=2), "Window frame_index"),
            ("omitted window image", lambda rows: rows[2]["window"].pop(0), "causally rebuilt window"),
        ]
        for name, mutate, error in changes:
            with self.subTest(name=name):
                rows = deepcopy(self.rows)
                mutate(rows)
                self.write_jsonl(self.predictions_path, rows)
                self.assert_rejected_without_overwriting_report(error)
        self.write_jsonl(self.predictions_path, self.rows)

    def test_rejects_image_pixels_hash_and_path_escape(self):
        image_path = self.run_dir / self.rows[0]["window"][0]["image_path"]
        original = image_path.read_bytes()
        image = cv2.imread(str(image_path))
        image[0, 0, 0] ^= 255
        self.assertTrue(cv2.imwrite(str(image_path), image))
        self.assert_rejected_without_overwriting_report("pixels differ")
        image_path.write_bytes(original)
        rows = deepcopy(self.rows)
        rows[0]["window"][0]["image_sha256"] = "0" * 64
        self.write_jsonl(self.predictions_path, rows)
        self.assert_rejected_without_overwriting_report("hash differs")
        rows[0]["window"][0].update(image_sha256=self.rows[0]["window"][0]["image_sha256"],
                                    image_path="../../outside.png")
        self.write_jsonl(self.predictions_path, rows)
        self.assert_rejected_without_overwriting_report("escapes")

    def test_rejects_sampling_and_raw_result_inconsistencies(self):
        changes = [
            ("missing sample", lambda rows: rows.pop(1), "sequence does not match"),
            ("duplicate sample", lambda rows: rows.insert(1, deepcopy(rows[0])), "sequence does not match"),
            ("raw output", lambda rows: rows[1].update(raw_response="{}"), "validity differs"),
            ("parsed result", lambda rows: rows[1]["result"].update(observed_effect="Fabricated replacement."), "parsed raw response"),
            ("invalid error", lambda rows: rows[-1].update(validation_error="different failure"), "validation error differs"),
        ]
        for name, mutate, error in changes:
            with self.subTest(name=name):
                rows = deepcopy(self.rows)
                mutate(rows)
                self.write_jsonl(self.predictions_path, rows)
                self.assert_rejected_without_overwriting_report(error)
        self.write_jsonl(self.predictions_path, self.rows)

    def test_rejects_tracker_alert_log_and_summary_corruption(self):
        rows = deepcopy(self.rows)
        rows[2]["monitor"]["alert"] = False
        self.write_jsonl(self.predictions_path, rows)
        self.assert_rejected_without_overwriting_report("tracking differs")
        self.write_jsonl(self.predictions_path, self.rows)
        alerts_path = self.run_dir / "alerts.jsonl"
        original_alerts = alerts_path.read_bytes()
        for changed in ([], [self.rows[2], self.rows[2], self.rows[6]]):
            with self.subTest(alert_count=len(changed)):
                self.write_jsonl(alerts_path, changed)
                self.assert_rejected_without_overwriting_report("Alert log differs")
        alerts_path.write_bytes(original_alerts)
        summary_path = self.run_dir / "summary.json"
        summary = read_json(summary_path)
        summary["model_calls"] += 1
        self.write_json(summary_path, summary)
        self.assert_rejected_without_overwriting_report("Summary model_calls")

    def test_rejects_code_hash_prompt_and_schema_changes(self):
        provenance_path = self.run_dir / "provenance.json"
        original = read_json(provenance_path)
        changed = deepcopy(original)
        changed["program_sha256"]["temporal_diagnosis.py"] = "0" * 64
        self.write_json(provenance_path, changed)
        self.assert_rejected_without_overwriting_report("code revision")
        self.write_json(provenance_path, original)
        prompt_path = self.run_dir / "prompt.txt"
        original_prompt = prompt_path.read_text(encoding="utf-8")
        altered_prompt = original_prompt + "\nDifferent instruction."
        prompt_path.write_text(altered_prompt, encoding="utf-8")
        self.assert_rejected_without_overwriting_report("provenance hash")
        changed = deepcopy(original)
        changed["prompt_sha256"] = hashlib.sha256(altered_prompt.encode()).hexdigest()
        self.write_json(provenance_path, changed)
        self.assert_rejected_without_overwriting_report("current prompt.*matching code revision")
        self.write_json(provenance_path, original)
        prompt_path.write_text(original_prompt, encoding="utf-8")
        schema_path = self.run_dir / "output_schema.json"
        schema = read_json(schema_path)
        schema["anyOf"][0]["additionalProperties"] = True
        self.write_json(schema_path, schema)
        self.assert_rejected_without_overwriting_report("schema.*matching code revision")

    def test_gpu_metric_validation_requires_every_encoded_window_image(self):
        config = {"model": {"device": "cuda:0"}}
        provenance = {"compute_device": "cuda:0"}
        metrics = {"compute_device": "cuda:0", "inference_ms": 10.0,
                   "image_grid_thw": [[1, 4, 4], [1, 4, 4]], "oom_retries": 0,
                   "peak_cuda_allocated_mib": 4000.0, "peak_cuda_reserved_mib": 4100.0}
        self.assertEqual(verify_metrics(metrics, config, provenance, 2), "cuda:0")
        changes = [
            ("missing image", {"image_grid_thw": [[1, 4, 4]]}, "every supplied window image"),
            ("zero grid", {"image_grid_thw": [[1, 0, 4], [1, 4, 4]]}, "Invalid model image token grid"),
            ("device fallback", {"compute_device": "cpu"}, "compute device differs"),
            ("negative wait", {"gpu_wait_time_s": -1}, "gpu_wait_time_s"),
            ("NaN duration", {"inference_ms": float("nan")}, "inference_ms"),
            ("negative memory", {"peak_cuda_allocated_mib": -1}, "peak_cuda_allocated_mib"),
            ("invalid retries", {"oom_retries": True}, "oom_retries"),
        ]
        for name, update, error in changes:
            with self.subTest(name=name):
                changed = deepcopy(metrics)
                changed.update(update)
                with self.assertRaisesRegex(ValueError, error):
                    verify_metrics(changed, config, provenance, 2)


if __name__ == "__main__":
    unittest.main()
