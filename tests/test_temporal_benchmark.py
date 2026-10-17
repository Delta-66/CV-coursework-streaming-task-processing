"""Test full-cohort scheduling, frozen resume and explicit execution failures."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

from download_vifailback import sha256
from run_temporal_benchmark import atomic_json, build_plan, load_config, read_json, run, sample_indices, validate_protocol


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(folder):
    root = Path(folder)
    data = root / "data"
    info = {"fps": 25, "chunks_size": 1000, "data_path": "data/episode_{episode_index:06d}.parquet",
            "video_path": "videos/{video_key}/episode_{episode_index:06d}.mp4"}
    write_json(data / "meta/info.json", info)
    (data / "meta/episodes.jsonl").write_text('\n'.join(json.dumps({"episode_index": index, "length": length}) for index, length in [(2, 26), (3, 51), (9, 10)]))
    (data / "meta/tasks.jsonl").write_text('{"task_index":0,"task":"Fixture task"}\n')
    test_rows = [{"episode_index": index, "num_frames": length, "split": "test", "exposure_reason": None,
                  "benchmark_cohort": "canonical_detection" if index != 9 else "extra_media_reference"}
                 for index, length in [(2, 26), (3, 51), (9, 10)]]
    split = root / "splits/test.jsonl"
    split.parent.mkdir()
    split.write_text('\n'.join(map(json.dumps, test_rows)))
    write_json(root / "splits/protocol.json", {"artifacts": {"test.jsonl": {"sha256": sha256(split)}},
               "canonical_detection_benchmark_episode_indices": [2, 3], "benchmark_exposure_episode_indices": [],
               "sources": {"metadata": {"revision": "fixture-revision"}}})
    episodes = []
    for index, length in [(2, 26), (3, 51)]:
        files = []
        for relative in [f"data/episode_{index:06d}.parquet", f"videos/cam/episode_{index:06d}.mp4"]:
            path = data / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
            files.append({"path": relative, "size": path.stat().st_size, "remote_sha256": sha256(path)})
        episodes.append({"episode_index": index, "num_frames": length, "files": files})
    write_json(root / "download_plan.json", {"revision": "fixture-revision", "split_manifest_sha256": sha256(split), "episodes": episodes})
    model = root / "model"
    model.mkdir()
    (model / "weights.bin").write_bytes(b"fixture-weights")
    write_json(model / "model_manifest.json", {"files": [{"path": "weights.bin", "size": (model / "weights.bin").stat().st_size,
                                                          "sha256": sha256(model / "weights.bin")}], "repo_id": "fixture", "revision": "fixture"})
    config = {"benchmark": {"split_manifest": "splits/test.jsonl", "split_protocol": "splits/protocol.json", "download_plan": "download_plan.json"},
              "source": {"format": "lerobot_v2.1", "root": "data", "cameras": ["cam"]},
              "monitor": {"camera": "cam", "sample_every": 25, "include_last_frame": True, "max_samples_per_episode": None,
                          "max_model_calls": None, "max_image_edge": 64, "history_size": 2, "max_history_s": 8},
              "tracking": {"stage_confirmation": 2, "failure_confirmation": 2, "recovery_confirmation": 2, "cooldown_s": 8.0},
              "model": {"path": "model", "device": "cuda:0", "max_new_tokens": 768, "max_image_pixels": 98304},
              "output_dir": "benchmark"}
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path, config


class FakeRunner:
    def __init__(self, fail_indices=(), all_invalid_indices=(), truncate_indices=()):
        self.calls = []
        self.backends = []
        self.fail_indices = set(fail_indices)
        self.all_invalid_indices = set(all_invalid_indices)
        self.truncate_indices = set(truncate_indices)

    def __call__(self, path, backend=None):
        config = read_json(path)
        index = config["source"]["episodes"][0]
        self.calls.append(index)
        self.backends.append(backend)
        checkpoint_path = Path(config["output_dir"]).parents[2] / "checkpoint.json"
        if not read_json(checkpoint_path)["episodes"][str(index)].get("input_sha256"):
            raise AssertionError("Input hashes were not persisted before inference")
        if index in self.fail_indices:
            raise RuntimeError("Synthetic interrupted trajectory")
        length = {2: 26, 3: 51}[index]
        indices = sample_indices(length)
        if index in self.truncate_indices:
            indices = indices[:-1]
        target = Path(config["output_dir"]) / f"attempt_{len(self.calls):03d}"
        target.mkdir(parents=True)
        invalid = index in self.all_invalid_indices
        summary = {"status": "monitor_run_failed" if invalid else "monitor_run_complete", "model_calls": len(indices),
                   "valid_outputs": 0 if invalid else len(indices), "invalid_outputs": len(indices) if invalid else 0,
                   "episodes": [{"episode_index": index, "frames_seen": length, "source_frames": length,
                                 "model_calls": len(indices), "full_episode_read": True}]}
        write_json(target / "summary.json", summary)
        write_json(target / "config.json", config)
        write_json(target / "provenance.json", {"fixture": True})
        (target / "predictions.jsonl").write_text('\n'.join(json.dumps({"episode_index": index, "frame_index": frame,
                                                                        "result_valid": not invalid}) for frame in indices))
        (target / "alerts.jsonl").write_text("")
        if invalid:
            raise RuntimeError("No valid model responses")
        return target


def fake_verify(run_path, config_path, output):
    write_json(Path(output), {"status": "passed", "test_fixture": True})


class TemporalBenchmarkTests(unittest.TestCase):
    def test_checkpoint_replace_retries_transient_windows_lock_without_truncation(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.json"
            path.write_text('{"previous": true}', encoding="utf-8")
            replace = Path.replace
            calls = []
            def transient(source, target):
                calls.append(source)
                if len(calls) == 1:
                    self.assertEqual(read_json(path), {"previous": True})
                    raise PermissionError("Simulated short-lived Windows sharing lock")
                return replace(source, target)
            with patch.object(Path, "replace", transient), patch("run_temporal_benchmark.time.sleep") as sleep:
                atomic_json(path, {"complete": True})
            self.assertEqual(read_json(path), {"complete": True})
            self.assertEqual(len(calls), 2)
            sleep.assert_called_once_with(0.05)

    def test_sampling_includes_tail_once(self):
        self.assertEqual(sample_indices(1), [0])
        self.assertEqual(sample_indices(25), [0, 24])
        self.assertEqual(sample_indices(26), [0, 25])
        self.assertEqual(sample_indices(51), [0, 25, 50])

    def test_plan_uses_full_canonical_cohort_and_real_file_sizes(self):
        with TemporaryDirectory() as folder:
            config_path, _ = fixture(folder)
            plan = build_plan(load_config(config_path))
            self.assertEqual([row["episode_index"] for row in plan["episodes"]], [2, 3])
            self.assertEqual((plan["total_frames"], plan["total_model_calls"], plan["total_bytes"]), (77, 5, 28))
            self.assertEqual(plan["unknown_size_files"], 0)

    def test_complete_run_reuses_backend_and_resume_skips_completed_episodes(self):
        with TemporaryDirectory() as folder, patch("run_temporal_benchmark.PROGRAM_FILES", ()), patch("builtins.print"):
            config_path, _ = fixture(folder)
            backend = SimpleNamespace(device="test_backend")
            runner = FakeRunner()
            prepared = []
            run_dir = run(config_path, backend=backend, episode_runner=runner, episode_verifier=fake_verify, ensure_files=prepared.append)
            self.assertEqual(runner.calls, [2, 3])
            self.assertTrue(all(item is backend for item in runner.backends))
            self.assertEqual(prepared, [2, 3])
            checkpoint = read_json(run_dir / "checkpoint.json")
            self.assertEqual(checkpoint["status"], "complete")
            self.assertTrue(all(item["execution_complete"] for item in checkpoint["episodes"].values()))
            validate_protocol(read_json(run_dir / "protocol.json"))
            run(config_path, backend=backend, episode_runner=runner, episode_verifier=fake_verify, ensure_files=prepared.append)
            self.assertEqual(runner.calls, [2, 3])

    def test_failure_retained_and_resume_retries_only_failed_episode(self):
        with TemporaryDirectory() as folder, patch("run_temporal_benchmark.PROGRAM_FILES", ()), patch("builtins.print"):
            config_path, _ = fixture(folder)
            backend = SimpleNamespace(device="test_backend")
            runner = FakeRunner(fail_indices=[3])
            run_dir = run(config_path, backend=backend, episode_runner=runner, episode_verifier=fake_verify)
            self.assertEqual(read_json(run_dir / "checkpoint.json")["status"], "incomplete")
            runner.fail_indices.clear()
            run(config_path, backend=backend, episode_runner=runner, episode_verifier=fake_verify)
            self.assertEqual(runner.calls, [2, 3, 3])
            self.assertEqual(read_json(run_dir / "checkpoint.json")["status"], "complete")

    def test_all_invalid_complete_execution_is_retained_for_failure_accounting(self):
        with TemporaryDirectory() as folder, patch("run_temporal_benchmark.PROGRAM_FILES", ()), patch("builtins.print"):
            config_path, _ = fixture(folder)
            runner = FakeRunner(all_invalid_indices=[2])
            run_dir = run(config_path, backend=SimpleNamespace(device="test_backend"), episode_runner=runner, episode_verifier=fake_verify)
            entry = read_json(run_dir / "checkpoint.json")["episodes"]["2"]
            self.assertEqual(entry["status"], "failed")
            self.assertEqual(entry["failure_kind"], "all_invalid_model_outputs")
            self.assertTrue(entry["execution_complete"])
            self.assertEqual(entry["invalid_outputs"], 2)
            self.assertEqual(len(entry["artifact_sha256"]), 5)

    def test_truncated_trajectory_cannot_be_marked_complete(self):
        with TemporaryDirectory() as folder, patch("run_temporal_benchmark.PROGRAM_FILES", ()), patch("builtins.print"):
            config_path, _ = fixture(folder)
            run_dir = run(config_path, backend=SimpleNamespace(device="test_backend"), episode_runner=FakeRunner(truncate_indices=[2]), episode_verifier=fake_verify)
            entry = read_json(run_dir / "checkpoint.json")["episodes"]["2"]
            self.assertEqual(entry["status"], "failed")
            self.assertFalse(entry["execution_complete"])

    def test_resume_preserves_first_all_invalid_execution_without_selective_retry(self):
        with TemporaryDirectory() as folder, patch("run_temporal_benchmark.PROGRAM_FILES", ()), patch("builtins.print"):
            config_path, _ = fixture(folder)
            backend = SimpleNamespace(device="test_backend")
            runner = FakeRunner(all_invalid_indices=[2])
            run_dir = run(config_path, backend=backend, episode_runner=runner, episode_verifier=fake_verify)
            first = read_json(run_dir / "checkpoint.json")["episodes"]["2"]
            runner.all_invalid_indices.clear()  # A later attempt could look better; it must not run.
            run(config_path, backend=backend, episode_runner=runner, episode_verifier=fake_verify)
            second = read_json(run_dir / "checkpoint.json")["episodes"]["2"]
            self.assertEqual(runner.calls, [2, 3])
            self.assertEqual(first, second)
            self.assertEqual(second["failure_kind"], "all_invalid_model_outputs")
            (run_dir / second["run_dir"] / "provenance.json").write_text('{"tampered":true}')
            with self.assertRaisesRegex(ValueError, "artifacts changed"):
                run(config_path, backend=backend, episode_runner=runner, episode_verifier=fake_verify)
            self.assertEqual(runner.calls, [2, 3])

    def test_protocol_change_or_completed_artifact_tampering_blocks_resume(self):
        with TemporaryDirectory() as folder, patch("run_temporal_benchmark.PROGRAM_FILES", ()), patch("builtins.print"):
            config_path, config = fixture(folder)
            backend = SimpleNamespace(device="test_backend")
            runner = FakeRunner()
            run_dir = run(config_path, backend=backend, episode_runner=runner, episode_verifier=fake_verify)
            config["model"]["max_new_tokens"] = 1024
            config_path.write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(ValueError, "Cannot resume"):
                run(config_path, backend=backend, episode_runner=runner, episode_verifier=fake_verify)
            config["model"]["max_new_tokens"] = 768
            config_path.write_text(yaml.safe_dump(config))
            checkpoint = read_json(run_dir / "checkpoint.json")
            path = run_dir / checkpoint["episodes"]["2"]["run_dir"] / "provenance.json"
            path.write_text('{"tampered":true}')
            with self.assertRaisesRegex(ValueError, "artifacts changed"):
                run(config_path, backend=backend, episode_runner=runner, episode_verifier=fake_verify)
            self.assertEqual(runner.calls, [2, 3])

    def test_plan_only_freezes_without_model_loading_and_caps_or_cpu_are_rejected(self):
        with TemporaryDirectory() as folder, patch("run_temporal_benchmark.PROGRAM_FILES", ()), patch("builtins.print"):
            config_path, config = fixture(folder)
            with patch("run_temporal_benchmark.QwenTemporalDiagnosis") as model:
                run_dir = run(config_path, plan_only=True)
                model.assert_not_called()
            self.assertEqual(read_json(run_dir / "checkpoint.json")["status"], "planned")
            for section, key, value in [("monitor", "max_model_calls", 9), ("monitor", "include_last_frame", False),
                                        ("monitor", "sample_every", 100), ("model", "device", "cpu")]:
                changed = deepcopy(config)
                changed[section][key] = value
                config_path.write_text(yaml.safe_dump(changed))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    load_config(config_path)

    def test_orphan_completed_output_is_recovered_with_verified_input_hashes(self):
        with TemporaryDirectory() as folder, patch("run_temporal_benchmark.PROGRAM_FILES", ()), patch("builtins.print"):
            config_path, _ = fixture(folder)
            backend = SimpleNamespace(device="test_backend")
            runner = FakeRunner()
            run_dir = run(config_path, backend=backend, episode_runner=runner, episode_verifier=fake_verify)
            checkpoint_path = run_dir / "checkpoint.json"
            checkpoint = read_json(checkpoint_path)
            entry = checkpoint["episodes"]["2"]
            entry["status"] = "running"
            entry["attempts"][-1]["status"] = "running"
            for name in ("run_dir", "artifact_sha256", "execution_complete", "input_sha256"):
                entry.pop(name)
            write_json(checkpoint_path, checkpoint)
            run(config_path, backend=backend, episode_runner=runner, episode_verifier=fake_verify)
            self.assertEqual(runner.calls, [2, 3])
            recovered = read_json(checkpoint_path)["episodes"]["2"]
            self.assertTrue(recovered["recovered_after_interrupt"])
            self.assertEqual(len(recovered["input_sha256"]), 2)
            self.assertEqual(recovered["attempts"][0]["status"], "interrupted")

    def test_completed_episode_source_tampering_blocks_resume_without_inference(self):
        with TemporaryDirectory() as folder, patch("run_temporal_benchmark.PROGRAM_FILES", ()), patch("builtins.print"):
            config_path, _ = fixture(folder)
            backend = SimpleNamespace(device="test_backend")
            runner = FakeRunner()
            run(config_path, backend=backend, episode_runner=runner, episode_verifier=fake_verify)
            (Path(folder) / "data/data/episode_000002.parquet").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "input SHA256 mismatch"):
                run(config_path, backend=backend, episode_runner=runner, episode_verifier=fake_verify)
            self.assertEqual(runner.calls, [2, 3])

    def test_orphan_all_invalid_output_is_recovered_without_repeating_inference(self):
        with TemporaryDirectory() as folder, patch("run_temporal_benchmark.PROGRAM_FILES", ()), patch("builtins.print"):
            config_path, _ = fixture(folder)
            backend = SimpleNamespace(device="test_backend")
            runner = FakeRunner(all_invalid_indices=[2])
            run_dir = run(config_path, backend=backend, episode_runner=runner, episode_verifier=fake_verify)
            checkpoint_path = run_dir / "checkpoint.json"
            checkpoint = read_json(checkpoint_path)
            entry = checkpoint["episodes"]["2"]
            entry["status"] = "running"
            entry["attempts"][-1]["status"] = "running"
            for name in ("run_dir", "artifact_sha256", "execution_complete", "input_sha256"):
                entry.pop(name)
            write_json(checkpoint_path, checkpoint)
            runner.all_invalid_indices.clear()
            run(config_path, backend=backend, episode_runner=runner, episode_verifier=fake_verify)
            recovered = read_json(checkpoint_path)["episodes"]["2"]
            self.assertEqual(runner.calls, [2, 3])
            self.assertEqual(recovered["status"], "failed")
            self.assertTrue(recovered["execution_complete"])
            self.assertTrue(recovered["recovered_after_interrupt"])
            self.assertEqual(recovered["failure_kind"], "all_invalid_model_outputs")
            self.assertEqual(len(recovered["input_sha256"]), 2)


if __name__ == "__main__":
    unittest.main()
