"""Bounded GPU resource probe on already-seen development trajectories only.

This never trains weights or runs the reserved test set. An allocation failure
ends the probe immediately; unlike the benchmark it does not wait and retry.
"""
import argparse
from contextlib import ExitStack, closing
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from unittest.mock import patch

from current_observation import read_cuda_memory
from run_task_monitor import load_config, sampled_frame, save_window
from stream_dataset import load_sources
from temporal_diagnosis import (CausalWindow, QwenTemporalDiagnosis, SYSTEM_PROMPT,
                                build_messages, output_schema, parse_result)


ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


class OneShotModel(QwenTemporalDiagnosis):
    def _acquire_cuda(self):
        started = time.perf_counter()
        self.model.to(self.device)
        self.torch.cuda.synchronize(self.device)
        self.model.hf_device_map = {"": self.device}
        return 0.0, time.perf_counter() - started

    def predict_once(self, request):
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
        torch = self.torch
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        inputs = self.processor.apply_chat_template(
            build_messages(request), tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt").to(self.device)
        prefix_allowed_tokens_fn = build_transformers_prefix_allowed_tokens_fn(
            self.tokenizer_data, JsonSchemaParser(output_schema()))
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                use_cache=True, max_time=self.max_inference_s,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn)
        torch.cuda.synchronize(self.device)
        generated_ids = generated[:, inputs["input_ids"].shape[1]:]
        raw_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return raw_text, {
            "inference_s": time.perf_counter() - started,
            "input_tokens": inputs["input_ids"].shape[1],
            "generated_tokens": generated_ids.shape[1],
            "at_token_limit": generated_ids.shape[1] >= self.max_new_tokens,
            "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(self.device) / 1024**2,
            "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved(self.device) / 1024**2,
        }


def start_windows_memory_sampler(path):
    """Sample this PID's WDDM counters; do not confuse allocator with total VRAM."""
    if os.name != "nt":
        return None, None
    command = f"""
$ErrorActionPreference = 'Stop'
while ($true) {{
  $rows = @(Get-CimInstance -ClassName Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory -Filter "Name LIKE 'pid_{os.getpid()}_%'")
  $dedicated = ($rows | Measure-Object -Property DedicatedUsage -Sum).Sum
  $shared = ($rows | Measure-Object -Property SharedUsage -Sum).Sum
  [ordered]@{{time = [DateTimeOffset]::Now.ToString('o'); dedicated_bytes = $dedicated; shared_bytes = $shared}} | ConvertTo-Json -Compress
  Start-Sleep -Milliseconds 1000
}}
"""
    log = path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
    return process, log


def run(args):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    if args.allocator_mib <= 0 or args.max_calls <= 0:
        raise ValueError("allocator-mib and max-calls must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no CPU inference fallback")
    config_path = ROOT / "configs/task_monitor.yaml"
    config = load_config(config_path)
    if config["source"]["episodes"] != [0, 18]:
        raise ValueError("Probe is restricted to already-seen development episodes 0 and 18")
    if config["model"]["device"] != "cuda:0":
        raise ValueError("Probe currently requires cuda:0")
    output_dir = args.output_dir or ROOT / "output/memory_probes" / (
        datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.attention}_{args.allocator_mib}mib")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    free_bytes, total_bytes = read_cuda_memory(torch, "cuda:0")
    budget_bytes = args.allocator_mib * 1024**2
    summary = {
        "started_at": datetime.now().astimezone().isoformat(), "pid": os.getpid(),
        "status": "running", "mode": "development_resource_probe_not_training",
        "attention": args.attention, "allocator_limit_mib": args.allocator_mib,
        "allocator_limit_excludes_cuda_context_and_library_allocations": True,
        "gpu_name": torch.cuda.get_device_name(0), "gpu_free_before_mib": free_bytes / 1024**2,
        "gpu_total_mib": total_bytes / 1024**2, "config": config,
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "lm-format-enforcer")},
        "program_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                           for name in ("probe_temporal_resources.py", "current_observation.py",
                                        "temporal_diagnosis.py", "run_task_monitor.py")},
        "calls": [], "error": None,
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "prompt.txt").write_text(SYSTEM_PROMPT, encoding="utf-8")
    write_json(output_dir / "output_schema.json", output_schema())
    print(json.dumps({"output_dir": str(output_dir), "pid": os.getpid(),
                      "free_mib": free_bytes / 1024**2}), flush=True)
    if free_bytes < budget_bytes + 256 * 1024**2 or budget_bytes >= total_bytes:
        summary.update(status="insufficient_free_memory", error="Need allocator budget plus 256 MiB free headroom")
        write_json(output_dir / "summary.json", summary)
        return 2
    torch.cuda.set_per_process_memory_fraction(
        budget_bytes / torch.cuda.get_device_properties(0).total_memory, 0)
    sampler, log = start_windows_memory_sampler(output_dir / "process_memory.jsonl")
    started = time.perf_counter()
    try:
        with ExitStack() as stack:
            if args.attention == "efficient":
                # Process-local only: explicit repeat_kv permits the available
                # efficient kernel; no edit to installed transformers or core code.
                # A Mock would retain every tensor argument and invalidate memory
                # measurements. A plain callable holds no history between calls.
                stack.enter_context(patch("transformers.integrations.sdpa_attention.use_gqa_in_sdpa",
                                          new=lambda attention_mask, key: False))
                stack.enter_context(sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION))
            model = config["model"]
            backend = OneShotModel(
                (config_path.parent / model["path"]).resolve(), model["device"],
                model["max_new_tokens"], model["max_image_pixels"], model["max_inference_s"])
            summary["model_load_s"] = backend.load_time_s
            summary["model_manifest"] = backend.manifest
            monitor = config["monitor"]
            window = CausalWindow(monitor["history_size"], monitor["max_history_s"], monitor["max_image_edge"])
            with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as predictions:
                for source in load_sources(config, config_path):
                    with closing(source.iter_frames()) as frames:
                        for packet in frames:
                            if not sampled_frame(packet.frame_index, source.length, monitor):
                                continue
                            if len(summary["calls"]) >= args.max_calls:
                                break
                            request = window.append(packet, monitor["camera"])
                            row = {"episode_index": packet.episode_index, "frame_index": packet.frame_index,
                                   "images": len(request.frames), "task_instruction": request.task_instruction,
                                   "window": save_window(output_dir, request)}
                            summary["current_input"] = {key: row[key] for key in ("episode_index", "frame_index", "images")}
                            write_json(output_dir / "summary.json", summary)
                            raw_text, metrics = backend.predict_once(request)
                            row.update(metrics=metrics, raw_response=raw_text, validation_error=None)
                            try:
                                row["result"] = parse_result(raw_text, request)
                            except (ValueError, TypeError) as error:
                                row["result"] = None
                                row["validation_error"] = str(error)
                            row["result_valid"] = row["result"] is not None
                            predictions.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                            predictions.flush()
                            summary["calls"].append({**summary["current_input"], **metrics,
                                                     "result_valid": row["result_valid"]})
                            print(json.dumps(summary["calls"][-1]), flush=True)
                    if len(summary["calls"]) >= args.max_calls:
                        break
            summary["status"] = "completed"
    except torch.cuda.OutOfMemoryError as error:
        summary.update(status="cuda_out_of_memory", error=str(error))
        print("CUDA out of memory: probe stopped; no retry or CPU fallback.", flush=True)
    except (RuntimeError, ValueError) as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
        print(summary["error"], flush=True)
    finally:
        summary["elapsed_s"] = time.perf_counter() - started
        if sampler is not None:
            sampler.terminate()
            sampler.wait(timeout=10)
            log.close()
            samples = []
            for line in (output_dir / "process_memory.jsonl").read_text(encoding="utf-8-sig", errors="replace").splitlines():
                try:
                    samples.append(json.loads(line))
                except ValueError:
                    pass
            summary["process_memory_samples"] = len(samples)
            summary["sampled_peak_process_dedicated_mib"] = max(
                ((item.get("dedicated_bytes") or 0) / 1024**2 for item in samples), default=None)
            summary["sampled_peak_process_shared_mib"] = max(
                ((item.get("shared_bytes") or 0) / 1024**2 for item in samples), default=None)
            summary["process_memory_note"] = "WDDM per-PID counters sampled about once per second; may miss short peaks"
        calls = summary["calls"]
        summary["successful_inferences"] = len(calls)
        summary["valid_outputs"] = sum(item["result_valid"] for item in calls)
        summary["mean_inference_s"] = statistics.mean(item["inference_s"] for item in calls) if calls else None
        plan_path = ROOT / "output/temporal_benchmark/canonical_detection_v1/plan.json"
        if plan_path.is_file():
            episodes = json.loads(plan_path.read_text(encoding="utf-8"))["episodes"]
            counts = {images: sum(max(item["expected_model_calls"] - 2, 0) if images == 3
                                 else int(item["expected_model_calls"] >= images) for item in episodes)
                      for images in (1, 2, 3)}
            means = {images: statistics.mean(item["inference_s"] for item in calls if item["images"] == images)
                     for images in (1, 2, 3) if any(item["images"] == images for item in calls)}
            summary["benchmark_call_counts_by_images"] = counts
            summary["mean_inference_s_by_images"] = means
            if all(images in means for images in counts) and summary["status"] == "completed":
                estimate = sum(counts[images] * means[images] for images in counts) / 3600
                summary["estimated_benchmark_inference_hours"] = estimate
                summary["estimated_hours_with_20_percent_margin"] = estimate * 1.2
                summary["estimate_caveat"] = "Two seen trajectories only; excludes downloads, GPU waiting, and task-dependent output length"
        summary["finished_at"] = datetime.now().astimezone().isoformat()
        write_json(output_dir / "summary.json", summary)
        print(json.dumps({key: value for key, value in summary.items()
                          if key not in ("calls", "config", "model_manifest", "program_sha256")}, ensure_ascii=False), flush=True)
    return 0 if summary["status"] == "completed" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention", choices=("default", "efficient"), default="default")
    parser.add_argument("--allocator-mib", type=int, default=5120)
    parser.add_argument("--max-calls", type=int, default=35)
    parser.add_argument("--output-dir", type=Path)
    raise SystemExit(run(parser.parse_args()))
