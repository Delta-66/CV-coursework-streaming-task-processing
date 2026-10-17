"""Exercise the temporal pipeline with known inputs, budgets and model failures."""
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

from run_task_monitor import load_config, run, sampled_frame, scheduled_calls


def diagnosis(request, failure=False):
    result = {"stage": "unknown", "status": "insufficient_evidence", "failure": None, "evidence": [],
              "high_level_advice": None, "low_level_advice": None,
              "key_states": {"grasp": "unknown", "target_relation": "unknown", "task_goal": "unknown"},
              "expected_effect": "The block should be placed in the tray.",
              "observed_effect": "The image does not establish the action effect.",
              "action_outcome": "uncertain", "failure_type": "unknown", "evidence_frame_indices": []}
    if failure:
        result.update(stage="placing", status="failure", failure=True,
                      evidence=["The released block is visibly outside the tray."],
                      high_level_advice="Recover and place the block in the tray.",
                      low_level_advice={"action": "Align with the block to the left in the image.",
                                        "reference_frame": "image", "completion_check": "Verify alignment."},
                      observed_effect="The block remains outside the tray after release.",
                      action_outcome="not_achieved", failure_type="placement_miss",
                      evidence_frame_indices=[frame.frame_index for frame in request.frames])
        result["key_states"].update(grasp="not_held", target_relation="away_from_target", task_goal="not_met")
    return result


class OnlinePacket(SimpleNamespace):
    def __getattr__(self, name):
        if name in {"recorded_targets", "extra_records", "offline_labels", "offline_annotations"}:
            raise AssertionError(f"The online runner read a forbidden field: {name}")
        raise AttributeError(name)


class FakeSource:
    cameras = ["cam"]

    def __init__(self, episode_index, length=7):
        self.episode_index = episode_index
        self.length = length
        self.last_yielded = -1
        self.closed = False

    def iter_frames(self):
        try:
            for frame_index in range(self.length):
                self.last_yielded = frame_index
                yield OnlinePacket(
                    episode_index=self.episode_index, frame_index=frame_index,
                    timestamp=frame_index / 25, timestamp_source="fixture",
                    task_instruction="Place the block in the tray.",
                    images={"cam": np.full((40, 80, 3), frame_index, np.uint8)},
                    observations={"observation.state": np.arange(14, dtype=float) + frame_index,
                                  "observation.velocity": np.zeros(14),
                                  "observation.effort": np.ones(14)},
                    source_references={"cam": {"frame_index": frame_index, "video_timestamp": frame_index / 25}})
        finally:
            self.closed = True


class FakeBackend:
    def __init__(self, sources, invalid_calls=(), fail_calls=(), crash_call=None):
        self.sources = {source.episode_index: source for source in sources}
        self.requests = []
        self.invalid_calls = set(invalid_calls)
        self.fail_calls = set(fail_calls)
        self.crash_call = crash_call

    def predict(self, request):
        # A model call must finish before the replay iterator exposes its future.
        current = request.frames[-1]
        if self.sources[current.episode_index].last_yielded != current.frame_index:
            raise AssertionError("The runner read future packets before completing inference")
        self.requests.append(request)
        call = len(self.requests)
        if call == self.crash_call:
            raise RuntimeError("Synthetic model failure")
        raw_text = "invalid json" if call in self.invalid_calls else json.dumps(diagnosis(request, call in self.fail_calls))
        return raw_text, {"inference_ms": 1.0, "compute_device": "test_backend"}


def config_for(folder):
    return {"source": {"format": "lerobot_v2.1"},
            "monitor": {"camera": "cam", "sample_every": 2, "max_image_edge": 64,
                        "history_size": 2, "max_history_s": 8.0,
                        "max_samples_per_episode": None, "max_model_calls": None},
            "tracking": {"stage_confirmation": 2, "failure_confirmation": 2,
                         "recovery_confirmation": 2, "cooldown_s": 8.0},
            "model": {"path": "unused-test-model", "device": "cuda:0", "max_new_tokens": 768,
                      "max_image_pixels": 98304}, "output_dir": str(Path(folder) / "runs")}


def save_config(folder, config):
    path = Path(folder) / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def read_rows(run_dir, filename="predictions.jsonl"):
    return [json.loads(line) for line in (run_dir / filename).read_text(encoding="utf-8").splitlines()]


class TaskMonitorRunnerTests(unittest.TestCase):
    def test_terminal_sampling_covers_short_and_exact_boundary_trajectories_once(self):
        monitor = {"sample_every": 25, "include_last_frame": True}
        cases = {1: [0], 25: [0, 24], 26: [0, 25], 27: [0, 25, 26],
                 300: [*range(0, 300, 25), 299], 519: [*range(0, 519, 25), 518]}
        for length, expected in cases.items():
            with self.subTest(length=length):
                actual = [index for index in range(length) if sampled_frame(index, length, monitor)]
                self.assertEqual(actual, expected)
                self.assertEqual(scheduled_calls(length, monitor), len(expected))

    def test_full_sequence_retains_tail_and_reports_sampling_not_just_read_coverage(self):
        with TemporaryDirectory() as folder:
            config = config_for(folder)
            config["monitor"]["include_last_frame"] = True
            sources = [FakeSource(5, length=6)]
            backend = FakeBackend(sources)
            with patch("run_task_monitor.load_sources", return_value=sources), patch("builtins.print"):
                run_dir = run(save_config(folder, config), backend)
            self.assertEqual([row["frame_index"] for row in read_rows(run_dir)], [0, 2, 4, 5])
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(summary["sampling_coverage"], {"expected_model_calls": 4,
                "recorded_model_calls": 4, "complete": True, "include_last_frame": True})
            self.assertEqual([frame.frame_index for frame in backend.requests[-1].frames], [2, 4, 5])

    def test_budget_invalid_response_and_episode_boundary_are_retained(self):
        with TemporaryDirectory() as folder:
            config = config_for(folder)
            config["monitor"].update(max_samples_per_episode=2, max_model_calls=3)
            config_path = save_config(folder, config)
            sources = [FakeSource(0), FakeSource(1), FakeSource(2)]
            backend = FakeBackend(sources, invalid_calls=[2])
            with patch("run_task_monitor.load_sources", return_value=sources), patch("builtins.print"):
                run_dir = run(config_path, backend)
            rows = read_rows(run_dir)
            self.assertEqual([(row["episode_index"], row["frame_index"]) for row in rows], [(0, 0), (0, 2), (1, 0)])
            self.assertEqual([row["history_frames"] for row in rows], [0, 1, 0])
            self.assertEqual([len(request.frames) for request in backend.requests], [1, 2, 1])
            self.assertFalse(rows[1]["result_valid"])
            self.assertIsNone(rows[1]["result"])
            self.assertEqual(rows[1]["raw_response"], "invalid json")
            self.assertTrue(rows[1]["validation_error"])
            self.assertFalse(rows[1]["monitor"]["alert"])
            self.assertEqual(read_rows(run_dir, "alerts.jsonl"), [])
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(summary["status"], "monitor_run_complete")
            self.assertEqual((summary["valid_outputs"], summary["invalid_outputs"]), (2, 1))
            self.assertTrue(all(item["full_episode_read"] for item in summary["episodes"]))
            self.assertFalse(summary["sampling_coverage"]["complete"])
            self.assertEqual([item["frames_seen"] for item in summary["episodes"]], [7, 7])
            self.assertEqual(sources[2].last_yielded, -1)
            self.assertTrue(sources[0].closed and sources[1].closed)
            provenance = json.loads((run_dir / "provenance.json").read_text())
            self.assertFalse(provenance["offline_annotations_used"])
            self.assertFalse(provenance["recorded_targets_used"])
            self.assertFalse(provenance["cpu_fallback"])
            self.assertEqual(provenance["max_frames_per_call"], 3)
            self.assertEqual(provenance["compute_device"], "test_backend")

    def test_bounded_windows_and_saved_images_match_exact_model_inputs(self):
        with TemporaryDirectory() as folder:
            config_path = save_config(folder, config_for(folder))
            sources = [FakeSource(0)]
            backend = FakeBackend(sources)
            with patch("run_task_monitor.load_sources", return_value=sources), patch("builtins.print"):
                run_dir = run(config_path, backend)
            rows = read_rows(run_dir)
            self.assertEqual([[item["frame_index"] for item in row["window"]] for row in rows],
                             [[0], [0, 2], [0, 2, 4], [2, 4, 6]])
            self.assertEqual(len(list((run_dir / "frames").rglob("*.png"))), 4)
            for row, request in zip(rows, backend.requests):
                for saved, frame in zip(row["window"], request.frames):
                    image = cv2.cvtColor(cv2.imread(str(run_dir / saved["image_path"])), cv2.COLOR_BGR2RGB)
                    np.testing.assert_array_equal(image, frame.image)
                    self.assertEqual(saved["image_sha256"], hashlib.sha256(frame.image.tobytes()).hexdigest())
                    self.assertEqual(saved["observations"], frame.observations)
                    self.assertLessEqual(saved["timestamp"], row["timestamp"])
                    self.assertEqual(saved["episode_index"], row["episode_index"])

    def test_only_confirmed_first_incident_is_written_to_alert_log(self):
        with TemporaryDirectory() as folder:
            config_path = save_config(folder, config_for(folder))
            sources = [FakeSource(0)]
            backend = FakeBackend(sources, fail_calls=[2, 3, 4])
            with patch("run_task_monitor.load_sources", return_value=sources), patch("builtins.print"):
                run_dir = run(config_path, backend)
            rows = read_rows(run_dir)
            alerts = read_rows(run_dir, "alerts.jsonl")
            self.assertEqual([row["monitor"]["alert"] for row in rows], [False, False, True, False])
            self.assertEqual(alerts, [rows[2]])
            self.assertEqual(json.loads((run_dir / "summary.json").read_text())["failure_alerts"], 1)

    def test_all_invalid_responses_fail_and_keep_raw_records(self):
        with TemporaryDirectory() as folder:
            config_path = save_config(folder, config_for(folder))
            sources = [FakeSource(0)]
            backend = FakeBackend(sources, invalid_calls=[1, 2, 3, 4])
            with patch("run_task_monitor.load_sources", return_value=sources), patch("builtins.print"):
                with self.assertRaisesRegex(RuntimeError, "No valid model responses"):
                    run(config_path, backend)
            run_dir = next((Path(folder) / "runs").iterdir())
            rows = read_rows(run_dir)
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(not row["result_valid"] and row["result"] is None for row in rows))
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(summary["status"], "monitor_run_failed")
            self.assertEqual(summary["invalid_outputs"], 4)
            self.assertTrue(summary["episodes"][0]["full_episode_read"])

    def test_exception_records_partial_coverage_and_closes_source(self):
        with TemporaryDirectory() as folder:
            config_path = save_config(folder, config_for(folder))
            sources = [FakeSource(0)]
            backend = FakeBackend(sources, crash_call=2)
            with patch("run_task_monitor.load_sources", return_value=sources), patch("builtins.print"):
                with self.assertRaisesRegex(RuntimeError, "Synthetic model failure"):
                    run(config_path, backend)
            run_dir = next((Path(folder) / "runs").iterdir())
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(summary["status"], "monitor_run_failed")
            self.assertEqual(summary["model_calls"], 1)
            self.assertEqual(summary["episodes"][0]["frames_seen"], 3)
            self.assertFalse(summary["episodes"][0]["full_episode_read"])
            self.assertEqual(len(read_rows(run_dir)), 1)
            self.assertTrue(sources[0].closed)

    def test_invalid_configuration_and_overrides_fail_before_inference(self):
        with TemporaryDirectory() as folder:
            cases = [("monitor", "history_size", -1), ("monitor", "history_size", True),
                     ("monitor", "history_size", 33),
                     ("monitor", "max_history_s", 0), ("monitor", "max_history_s", float("inf")),
                     ("monitor", "sample_every", 0), ("monitor", "max_model_calls", 0),
                     ("monitor", "include_last_frame", "true"),
                     ("model", "max_image_pixels", 1), ("model", "gpu_poll_interval_s", 61),
                     ("tracking", "failure_confirmation", 0), ("tracking", "cooldown_s", -1)]
            for section, key, value in cases:
                with self.subTest(section=section, key=key, value=value):
                    config = config_for(folder)
                    config[section][key] = value
                    path = save_config(folder, config)
                    with self.assertRaises(ValueError):
                        load_config(path)
            path = save_config(folder, config_for(folder))
            for override in (0, -1, True):
                with self.assertRaisesRegex(ValueError, "override must be positive"):
                    run(path, max_model_calls=override)


if __name__ == "__main__":
    unittest.main()
