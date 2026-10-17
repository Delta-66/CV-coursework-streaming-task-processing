"""Tests for current-only input isolation, result validity, and inference budgets."""
import importlib.util
import json
import subprocess
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from current_observation import CurrentObservation, QwenCurrentObservation, make_request, output_schema, parse_result, read_cuda_memory, wait_for_cuda_memory
from run_current_observation import load_config, run


def diagnosis(status="insufficient_evidence"):
    result = {"stage": "unknown", "status": status, "failure": None, "evidence": [],
              "high_level_advice": None, "low_level_advice": None}
    if status == "in_progress":
        result.update(stage="approaching", failure=False, evidence=["The gripper is beside the object."])
    if status == "completed":
        result.update(stage="completed", failure=False, evidence=["All specified final conditions are visible."])
    if status == "failure":
        result.update(stage="placing", failure=True, evidence=["The object is visibly outside the target tray."],
                      high_level_advice="Recover the object and place it in the tray.",
                      low_level_advice={"action": "Align with the object to the left in the image.",
                                        "reference_frame": "image", "completion_check": "Verify alignment with the object."})
    return result


class PoisonedPacket:
    def __init__(self):
        self.task_instruction = "Place the block in the tray."
        self.images = {"cam": np.full((80, 160, 3), 123, np.uint8)}

    def __getattr__(self, name):
        raise AssertionError(f"Current-only request accessed forbidden field: {name}")


class FakeSource:
    cameras = ["cam"]
    length = 7

    def __init__(self, episode_index):
        self.episode_index = episode_index

    def iter_frames(self):
        for frame_index in range(self.length):
            yield SimpleNamespace(episode_index=self.episode_index, frame_index=frame_index,
                                  timestamp=frame_index/25, timestamp_source="fixture", task_instruction="Task",
                                  images={"cam": np.full((8, 8, 3), frame_index, np.uint8)},
                                  source_references={"cam": {"frame_index": frame_index}},
                                  recorded_targets={"action": [999]}, offline_labels={"future_failure": True})


class FakeBackend:
    def __init__(self):
        self.requests = []

    def predict(self, request):
        self.requests.append(request)
        return ("bad json" if len(self.requests) == 2 else json.dumps(diagnosis())), {"inference_ms": 1}


class BaselineTests(unittest.TestCase):
    def test_windows_memory_uses_lower_cuda_or_physical_free_value(self):
        device_uuid = "b54bcf49-7ee9-1a96-9695-21e4d58abb4a"
        for cuda_free, expected in ((6650, 1787), (1200, 1200)):
            with self.subTest(cuda_free=cuda_free):
                torch = SimpleNamespace(cuda=SimpleNamespace(
                    mem_get_info=lambda device: (cuda_free * 1024**2, 8192 * 1024**2),
                    get_device_properties=lambda device: SimpleNamespace(uuid=device_uuid)))
                with patch("current_observation.sys.platform", "win32"), \
                     patch("current_observation.subprocess.run", return_value=SimpleNamespace(
                         stdout=f"GPU-{device_uuid}, 1787, 8188\n")) as command:
                    self.assertEqual(read_cuda_memory(torch, "cuda:0"), (expected * 1024**2, 8188 * 1024**2))
                self.assertEqual(command.call_args.args[0], ["nvidia-smi", "--query-gpu=uuid,memory.free,memory.total",
                                                            "--format=csv,noheader,nounits"])
                self.assertTrue(command.call_args.kwargs["capture_output"])
                self.assertTrue(command.call_args.kwargs["text"])
                self.assertTrue(command.call_args.kwargs["check"])
                self.assertLessEqual(command.call_args.kwargs["timeout"], 5)

    def test_windows_memory_matches_uuid_instead_of_logical_gpu_index(self):
        selected = "b54bcf49-7ee9-1a96-9695-21e4d58abb4a"
        other = "63c601bd-290b-4e5c-b775-f4f8145ae718"
        torch = SimpleNamespace(cuda=SimpleNamespace(
            mem_get_info=lambda device: (7000 * 1024**2, 8192 * 1024**2),
            get_device_properties=lambda device: SimpleNamespace(uuid="GPU-" + selected.upper())))
        report = f"GPU-{other}, 7500, 8192\nGPU-{selected}, 400, 8192\n"
        with patch("current_observation.sys.platform", "win32"), \
             patch("current_observation.subprocess.run", return_value=SimpleNamespace(stdout=report)):
            self.assertEqual(read_cuda_memory(torch, "cuda:0"), (400 * 1024**2, 8192 * 1024**2))

    def test_windows_missing_or_invalid_physical_report_stops_inference(self):
        selected = "b54bcf49-7ee9-1a96-9695-21e4d58abb4a"
        torch = SimpleNamespace(cuda=SimpleNamespace(
            mem_get_info=lambda device: (7000 * 1024**2, 8192 * 1024**2),
            get_device_properties=lambda device: SimpleNamespace(uuid=selected)))
        errors = [FileNotFoundError("nvidia-smi unavailable"), subprocess.CalledProcessError(1, "nvidia-smi"),
                  subprocess.TimeoutExpired("nvidia-smi", 5)]
        for error in errors:
            with self.subTest(error=type(error).__name__), patch("current_observation.sys.platform", "win32"), \
                 patch("current_observation.subprocess.run", side_effect=error), \
                 self.assertRaisesRegex(RuntimeError, "refusing CUDA inference"):
                read_cuda_memory(torch, "cuda:0")
        reports = ["", "GPU-other, 7000, 8192\n", f"GPU-{selected}, N/A, 8192\n",
                   f"GPU-{selected}, nan, 8192\n", f"GPU-{selected}, 0, 0\n",
                   f"GPU-{selected}, -1, 8192\n", f"GPU-{selected}, 9000, 8192\n",
                   f"GPU-{selected}, 7000\n", f"GPU-{selected}, 7000, 8192\n" * 2]
        for report in reports:
            with self.subTest(report=report), patch("current_observation.sys.platform", "win32"), \
                 patch("current_observation.subprocess.run", return_value=SimpleNamespace(stdout=report)), \
                 self.assertRaisesRegex(RuntimeError, "refusing CUDA inference"):
                read_cuda_memory(torch, "cuda:0")

    def test_nonwindows_uses_cuda_api_and_rejects_invalid_memory(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda device: (4096, 8192)))
        with patch("current_observation.sys.platform", "linux"), patch("current_observation.subprocess.run") as command:
            self.assertEqual(read_cuda_memory(torch, "cuda:0"), (4096, 8192))
        command.assert_not_called()
        for free, total in ((-1, 8192), (8193, 8192), (0, 0), (float("nan"), 8192), (1, float("inf"))):
            torch.cuda.mem_get_info = lambda device: (free, total)
            with self.subTest(free=free, total=total), patch("current_observation.sys.platform", "linux"), \
                 self.assertRaisesRegex(RuntimeError, "Invalid CUDA"):
                read_cuda_memory(torch, "cuda:0")

    def test_windows_physical_headroom_releases_weights_and_waits_for_gpu(self):
        selected = "b54bcf49-7ee9-1a96-9695-21e4d58abb4a"
        reports = [SimpleNamespace(stdout=f"GPU-{selected}, {free}, 8192\n") for free in (200, 3000, 7000)]
        destinations = []
        backend = QwenCurrentObservation.__new__(QwenCurrentObservation)
        backend.device = "cuda:0"
        backend.min_free_cuda_mib = 6500
        backend.min_inference_free_cuda_mib = 1024
        backend.gpu_poll_interval_s = 10
        backend.model = SimpleNamespace(to=destinations.append)
        backend.torch = SimpleNamespace(cuda=SimpleNamespace(
            mem_get_info=lambda device: (7000 * 1024**2, 8192 * 1024**2),
            get_device_properties=lambda device: SimpleNamespace(uuid=selected),
            empty_cache=lambda: None, synchronize=lambda device: None, OutOfMemoryError=MemoryError))
        with patch("current_observation.sys.platform", "win32"), \
             patch("current_observation.subprocess.run", side_effect=reports), \
             patch("current_observation.time.sleep") as sleep, patch("builtins.print"):
            backend._ensure_cuda_ready()
        self.assertEqual(destinations, ["cpu", "cuda:0"])
        self.assertEqual(backend.model.hf_device_map, {"": "cuda:0"})
        sleep.assert_called_once_with(10)

    def test_gpu_memory_waits_until_available_without_busy_polling(self):
        available = iter([200, 4500, 6000])
        clock = [0.0]
        def sleep(seconds):
            clock[0] += seconds
        torch = SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda device: (next(available)*1024**2, 8192*1024**2)))
        with patch("current_observation.sys.platform", "linux"), patch("current_observation.time.sleep", side_effect=sleep), \
             patch("current_observation.time.perf_counter", side_effect=lambda: clock[0]), patch("builtins.print"):
            self.assertEqual(wait_for_cuda_memory(torch, "cuda:0", 5000, 10), 20.0)

    def test_impossible_gpu_memory_requirement_is_rejected(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda device: (4096*1024**2, 4096*1024**2)))
        with patch("current_observation.sys.platform", "linux"), self.assertRaisesRegex(ValueError, "total capacity"):
            wait_for_cuda_memory(torch, "cuda:0", 5000)

    def test_low_headroom_releases_weights_and_resumes_only_on_gpu(self):
        available = iter([200, 3000, 6000])
        destinations = []
        backend = QwenCurrentObservation.__new__(QwenCurrentObservation)
        backend.device = "cuda:0"
        backend.min_free_cuda_mib = 5000
        backend.min_inference_free_cuda_mib = 1024
        backend.gpu_poll_interval_s = 10
        backend.model = SimpleNamespace(to=destinations.append)
        backend.torch = SimpleNamespace(cuda=SimpleNamespace(
            mem_get_info=lambda device: (next(available)*1024**2, 8192*1024**2),
            empty_cache=lambda: None, synchronize=lambda device: None, OutOfMemoryError=MemoryError))
        with patch("current_observation.sys.platform", "linux"), patch("current_observation.time.sleep") as sleep, patch("builtins.print"):
            backend._ensure_cuda_ready()
        self.assertEqual(destinations, ["cpu", "cuda:0"])
        self.assertEqual(backend.model.hf_device_map, {"": "cuda:0"})
        sleep.assert_called_once_with(10)

    def test_oom_during_gpu_transfer_retries_after_releasing_weights(self):
        destinations = []
        def move(device):
            destinations.append(device)
            if len(destinations) == 1:
                raise MemoryError("Another task consumed GPU memory")
        backend = QwenCurrentObservation.__new__(QwenCurrentObservation)
        backend.device = "cuda:0"
        backend.min_free_cuda_mib = 5000
        backend.gpu_poll_interval_s = 10
        backend.model = SimpleNamespace(to=move)
        backend.torch = SimpleNamespace(cuda=SimpleNamespace(
            mem_get_info=lambda device: (6000*1024**2, 8192*1024**2),
            empty_cache=lambda: None, synchronize=lambda device: None, OutOfMemoryError=MemoryError))
        with patch("current_observation.sys.platform", "linux"), patch("current_observation.time.sleep") as sleep, patch("builtins.print"):
            backend._acquire_cuda()
        self.assertEqual(destinations, ["cuda:0", "cpu", "cuda:0"])
        sleep.assert_called_once_with(10)

    @unittest.skipUnless(importlib.util.find_spec("lmformatenforcer"), "Optional baseline decoding dependency")
    def test_generation_oom_retries_same_image_on_gpu_without_cpu_inference(self):
        destinations = []
        messages_seen = []
        calls = []
        class Inputs(dict):
            def to(self, device):
                self["device"] = device
                return self
        def prepare(messages, **kwargs):
            messages_seen.append(messages)
            return Inputs(input_ids=np.array([[1]]), image_grid_thw=np.array([[1, 4, 4]]))
        def generate(**kwargs):
            calls.append(kwargs["device"])
            if len(calls) == 1:
                raise MemoryError("GPU memory was taken during generation")
            return np.array([[1, 2, 3]])
        backend = QwenCurrentObservation.__new__(QwenCurrentObservation)
        backend.device = "cuda:0"
        backend.max_new_tokens = 128
        backend.max_inference_s = 120
        backend.gpu_poll_interval_s = 10
        backend.tokenizer_data = None
        backend.model = SimpleNamespace(generate=generate, to=destinations.append)
        backend.processor = SimpleNamespace(apply_chat_template=prepare,
                                            batch_decode=lambda *args, **kwargs: [json.dumps(diagnosis())])
        backend.torch = SimpleNamespace(inference_mode=nullcontext, cuda=SimpleNamespace(
            empty_cache=lambda: None, synchronize=lambda device: None, reset_peak_memory_stats=lambda device: None,
            max_memory_allocated=lambda device: 0, max_memory_reserved=lambda device: 0, OutOfMemoryError=MemoryError))
        def resume():
            backend.model.to("cuda:0")
            return 20.0, 1.0
        request = make_request(PoisonedPacket(), "cam")
        with patch.object(backend, "_ensure_cuda_ready", return_value=(0.0, 0.0)), \
             patch.object(backend, "_acquire_cuda", side_effect=resume), \
             patch("lmformatenforcer.integrations.transformers.build_transformers_prefix_allowed_tokens_fn"), \
             patch("current_observation.time.sleep"), patch("builtins.print"):
            raw_text, metrics = backend.predict(request)
        self.assertEqual(parse_result(raw_text), diagnosis())
        self.assertEqual(calls, ["cuda:0", "cuda:0"])
        self.assertEqual(destinations, ["cpu", "cuda:0"])
        self.assertIs(messages_seen[0], messages_seen[1])
        self.assertEqual(metrics["oom_retries"], 1)
        self.assertGreaterEqual(metrics["gpu_wait_time_s"], 20.0)
        self.assertEqual(metrics["compute_device"], "cuda:0")

    def test_accepts_complete_results_and_fences(self):
        for status in ("in_progress", "failure", "completed", "insufficient_evidence"):
            with self.subTest(status=status):
                result = diagnosis(status)
                self.assertEqual(parse_result(json.dumps(result)), result)
        self.assertEqual(parse_result("```json\n"+json.dumps(diagnosis())+"\n```"), diagnosis())

    def test_rejects_contradictory_states_and_missing_fields(self):
        invalid = []
        result = diagnosis("in_progress")
        result["failure"] = None
        invalid.append(result)
        result = diagnosis("failure")
        result["evidence"] = []
        invalid.append(result)
        result = diagnosis("failure")
        result["low_level_advice"]["reference_frame"] = "robot_base"
        invalid.append(result)
        result = diagnosis()
        del result["high_level_advice"]
        invalid.append(result)
        result = diagnosis("in_progress")
        result["low_level_advice"] = diagnosis("failure")["low_level_advice"]
        invalid.append(result)
        result = diagnosis()
        result["stage"] = []
        invalid.append(result)
        for result in invalid:
            with self.subTest(result=result):
                with self.assertRaises(ValueError):
                    parse_result(json.dumps(result))

    def test_rejects_duplicate_keys_and_non_json_constants(self):
        text = json.dumps(diagnosis()).replace('"failure": null', '"failure": null, "failure": null')
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            parse_result(text)
        with self.assertRaises(ValueError):
            parse_result(json.dumps(diagnosis()).replace('"failure": null', '"failure": NaN'))
        with self.assertRaises(ValueError):
            parse_result("```{} ```")

    def test_request_uses_only_current_image_and_task(self):
        packet = PoisonedPacket()
        request = make_request(packet, "cam", 64)
        self.assertIsInstance(request, CurrentObservation)
        self.assertEqual(request.image.shape, (32, 64, 3))
        self.assertFalse(request.image.flags.writeable)
        packet.images["cam"][:] = 0
        self.assertTrue((request.image == 123).all())
        self.assertEqual(set(request.__dict__), {"task_instruction", "camera", "image"})

    def test_missing_camera_and_invalid_rgb_are_errors(self):
        packet = PoisonedPacket()
        with self.assertRaises(ValueError):
            make_request(packet, "other")
        packet.images["cam"] = np.zeros((3, 8, 8), np.float32)
        with self.assertRaises(ValueError):
            make_request(packet, "cam")

    def test_budget_and_invalid_response_are_retained(self):
        with TemporaryDirectory() as folder:
            config = {"source": {"format": "lerobot_v2.1"},
                      "baseline": {"camera": "cam", "sample_every": 2, "max_image_edge": 64,
                                   "max_samples_per_episode": 2, "max_model_calls": 3},
                      "model": {"max_new_tokens": 128, "max_image_pixels": 65536}, "output_dir": folder}
            config_path = Path(folder)/"config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            backend = FakeBackend()
            with patch("run_current_observation.load_sources", return_value=[FakeSource(0), FakeSource(1)]):
                run_dir = run(config_path, backend)
            records = [json.loads(line) for line in (run_dir/"predictions.jsonl").read_text().splitlines()]
            self.assertEqual([(row["episode_index"], row["frame_index"]) for row in records], [(0, 0), (0, 2), (1, 0)])
            self.assertEqual(len(backend.requests), 3)
            self.assertTrue(all(row["history_frames"] == 0 for row in records))
            self.assertFalse(records[1]["result_valid"])
            self.assertIsNone(records[1]["result"])
            self.assertEqual(records[1]["raw_response"], "bad json")
            summary = json.loads((run_dir/"summary.json").read_text())
            self.assertEqual(summary["valid_outputs"], 2)
            self.assertEqual(summary["invalid_outputs"], 1)

    def test_nonpositive_budget_is_rejected(self):
        with TemporaryDirectory() as folder:
            config = {"source": {}, "model": {}, "baseline": {"sample_every": 1, "max_image_edge": 64,
                                                                 "max_model_calls": 0}}
            path = Path(folder)/"config.yaml"
            path.write_text(yaml.safe_dump(config))
            with self.assertRaises(ValueError):
                load_config(path)

    def test_invalid_generation_time_budget_is_rejected(self):
        with TemporaryDirectory() as folder:
            config = {"source": {}, "baseline": {"camera": "cam", "sample_every": 1, "max_image_edge": 64},
                      "model": {"max_new_tokens": 128, "max_image_pixels": 65536, "max_inference_s": float("nan")}}
            path = Path(folder)/"config.yaml"
            path.write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(ValueError, "max_inference_s"):
                load_config(path)

    @unittest.skipUnless(importlib.util.find_spec("lmformatenforcer"), "Optional baseline decoding dependency")
    def test_constrained_schema_accepts_all_four_statuses(self):
        from lmformatenforcer import JsonSchemaParser
        for status in ("in_progress", "failure", "completed", "insufficient_evidence"):
            with self.subTest(status=status):
                parser = JsonSchemaParser(output_schema())
                for character in json.dumps(diagnosis(status)):
                    self.assertIn(character, parser.get_allowed_characters())
                    parser = parser.add_character(character)
                self.assertTrue(parser.can_end())


if __name__ == "__main__":
    unittest.main()
