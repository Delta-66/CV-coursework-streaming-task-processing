"""Verify causal monitoring records against source frames without model inference."""
import argparse
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import statistics

import cv2
import numpy as np

from diagnostic_tracking import DiagnosticTracker
from run_task_monitor import sampled_frame, scheduled_calls
from stream_dataset import load_sources
from temporal_diagnosis import CausalWindow, SYSTEM_PROMPT, output_schema, parse_result


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs/task_monitor.yaml"
DEFAULT_REPORT = ROOT / "docs/task-monitor-validation.json"
PROGRAM_FILES = ("run_task_monitor.py", "temporal_diagnosis.py", "diagnostic_tracking.py",
                 "current_observation.py", "dataset_inputs.py", "stream_dataset.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON value in {path}: {value}")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                      parse_constant=reject_constant, object_pairs_hook=unique_keys)


def read_jsonl(path):
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON value in {path}: {value}")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    return [json.loads(line, parse_constant=reject_constant, object_pairs_hook=unique_keys)
            for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def finite_nonnegative(value, name, positive=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), f"Invalid {name}: {value!r}")


def checked_image_path(run_dir, relative):
    require(isinstance(relative, str) and relative, "An image path is required")
    path = (run_dir / relative).resolve()
    require(path.is_relative_to(run_dir / "frames"), "Saved image path escapes the run frames directory")
    require(path.is_file(), f"Missing saved image: {path}")
    return path


def verify_runtime_artifacts(run_dir, provenance):
    """Require the code and model instructions used by this recorded run."""
    program_hashes = provenance.get("program_sha256")
    require(isinstance(program_hashes, dict) and set(program_hashes) == set(PROGRAM_FILES),
            "Recorded program hashes are incomplete; verify with the matching code revision")
    for name in PROGRAM_FILES:
        current_hash = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        require(program_hashes[name] == current_hash,
                f"Recorded code differs from current {name}; verify with the code revision used for this run")
    prompt = (run_dir / "prompt.txt").read_text(encoding="utf-8")
    require(provenance.get("prompt_sha256") == hashlib.sha256(prompt.encode()).hexdigest(),
            "Saved prompt differs from its provenance hash")
    require(prompt == SYSTEM_PROMPT,
            "Recorded prompt differs from the current prompt; verify with the matching code revision")
    require(read_json(run_dir / "output_schema.json") == output_schema(),
            "Recorded output schema differs from the current schema; verify with the matching code revision")


def verify_metrics(metrics, config, provenance, frame_count):
    require(isinstance(metrics, dict), "metrics must be an object")
    device = metrics.get("compute_device")
    require(isinstance(device, str) and bool(device), "Missing compute_device")
    if provenance.get("compute_device") != "test_backend":
        require(device == config["model"]["device"], "Actual compute device differs from run configuration")
        require(device == provenance.get("compute_device"), "Actual compute device differs from provenance")
    finite_nonnegative(metrics.get("inference_ms"), "inference_ms", positive=True)
    for name in ("gpu_wait_time_s", "gpu_resume_time_s", "oom_retry_time_s"):
        finite_nonnegative(metrics.get(name, 0), name)
    retries = metrics.get("oom_retries", 0)
    require(type(retries) is int and retries >= 0, "oom_retries must be a nonnegative integer")
    for name in ("input_tokens", "generated_tokens"):
        if name in metrics:
            require(type(metrics[name]) is int and metrics[name] > 0, f"Invalid {name}")
    if device.startswith("cuda"):
        for name in ("peak_cuda_allocated_mib", "peak_cuda_reserved_mib"):
            finite_nonnegative(metrics.get(name), name)
        require(metrics["peak_cuda_reserved_mib"] >= metrics["peak_cuda_allocated_mib"],
                "Reserved GPU peak is below allocated GPU peak")
    if provenance.get("compute_device") != "test_backend":
        grid = metrics.get("image_grid_thw")
        require(isinstance(grid, list) and len(grid) == frame_count,
                "The model image token grid does not cover every supplied window image")
        require(all(isinstance(item, list) and len(item) == 3
                    and all(type(value) is int and value > 0 for value in item) for item in grid),
                "Invalid model image token grid")
    return device


def verify(run_dir, config_path=DEFAULT_CONFIG, output=DEFAULT_REPORT):
    """Rebuild sampled windows and decisions; write a report only after all checks pass.

    Set ``output=None`` for a fully read-only audit. The saved config determines
    sampling; config_path supplies its original directory for relative data paths.
    """
    run_dir = Path(run_dir).resolve()
    config_path = Path(config_path).resolve()
    config = read_json(run_dir / "config.json")
    summary = read_json(run_dir / "summary.json")
    provenance = read_json(run_dir / "provenance.json")
    records = read_jsonl(run_dir / "predictions.jsonl")
    alerts = read_jsonl(run_dir / "alerts.jsonl")
    require(summary.get("status") == "monitor_run_complete", "The monitor run did not complete")
    require(bool(records), "No monitoring predictions were recorded")
    require(provenance.get("offline_annotations_used") is False, "Offline annotation isolation is not established")
    require(provenance.get("recorded_targets_used") is False, "Recorded target isolation is not established")
    require(provenance.get("state_vectors_used") is True, "State vector use is not recorded")
    require(provenance.get("cpu_fallback") is False, "CPU fallback policy is not recorded")
    require(provenance.get("previous_model_outputs_used_as_input") is False,
            "Isolation from previous model assertions is not established")
    verify_runtime_artifacts(run_dir, provenance)
    monitor = config["monitor"]
    require(provenance.get("history_size") == monitor["history_size"], "History capacity differs from provenance")
    require(provenance.get("max_frames_per_call") == monitor["history_size"] + 1,
            "Maximum frame count differs from provenance")
    require(provenance.get("max_history_s") == monitor["max_history_s"], "History duration differs from provenance")
    require(provenance.get("tracking") == config["tracking"], "Tracking configuration differs from provenance")
    window = CausalWindow(monitor["history_size"], monitor["max_history_s"], monitor["max_image_edge"])
    tracker = DiagnosticTracker(**config["tracking"])
    record_index = 0
    valid = 0
    references_checked = 0
    seen = set()
    source_images = set()
    devices = set()
    observed_episodes = []
    expected_alerts = []
    sources = load_sources(config, config_path)
    for source in sources:
        cap = monitor.get("max_model_calls")
        if cap is not None and record_index >= cap:
            break
        count = 0
        frames_seen = 0
        with closing(source.iter_frames()) as frames:
            for packet in frames:
                frames_seen += 1
                if not sampled_frame(packet.frame_index, source.length, monitor):
                    continue
                episode_cap = monitor.get("max_samples_per_episode")
                if ((episode_cap is not None and count >= episode_cap)
                        or (cap is not None and record_index >= cap)):
                    continue
                require(record_index < len(records), "Missing prediction for an expected sampled source frame")
                row = records[record_index]
                identity = (packet.episode_index, packet.frame_index)
                require(identity not in seen, "Duplicate sampled frame")
                seen.add(identity)
                require((row.get("episode_index"), row.get("frame_index")) == identity,
                        "Prediction sequence does not match configured sampling")
                require(row.get("timestamp") == packet.timestamp, "Prediction timestamp differs from source")
                require(row.get("timestamp_source") == packet.timestamp_source, "Timestamp source differs")
                request = window.append(packet, monitor["camera"])
                require(row.get("task_instruction") == request.task_instruction, "Task instruction differs from source")
                require(row.get("camera") == request.camera, "Camera differs from configured source")
                require(row.get("history_frames") == len(request.frames) - 1, "Wrong history frame count")
                saved_window = row.get("window")
                require(isinstance(saved_window, list) and len(saved_window) == len(request.frames),
                        "Saved window differs from the causally rebuilt window")
                for saved, frame in zip(saved_window, request.frames):
                    for name in ("episode_index", "frame_index", "timestamp", "timestamp_source",
                                 "observations", "gripper_state", "source_reference"):
                        require(saved.get(name) == getattr(frame, name), f"Window {name} differs from source")
                    require(frame.episode_index == packet.episode_index and frame.timestamp <= packet.timestamp,
                            "Window contains a different episode or future observation")
                    require(saved.get("image_shape") == list(frame.image.shape), "Saved image shape differs")
                    image_path = checked_image_path(run_dir, saved.get("image_path"))
                    saved_image = cv2.imread(str(image_path))
                    require(saved_image is not None, f"Cannot decode saved RGB image: {image_path}")
                    saved_image = cv2.cvtColor(saved_image, cv2.COLOR_BGR2RGB)
                    require(np.array_equal(saved_image, frame.image), "Saved image pixels differ from current source")
                    require(saved.get("image_sha256") == hashlib.sha256(frame.image.tobytes()).hexdigest(),
                            "Saved image hash differs from source RGB")
                    source_images.add((frame.episode_index, frame.frame_index))
                    references_checked += 1
                result = None
                validation_error = None
                try:
                    result = parse_result(row["raw_response"], request)
                except (ValueError, TypeError) as error:
                    validation_error = str(error)
                require(row.get("result_valid") is (result is not None), "Recorded result validity differs")
                require(row.get("result") == result, "Recorded result differs from the parsed raw response")
                require(row.get("validation_error") == validation_error, "Recorded validation error differs")
                valid += int(result is not None)
                tracked = tracker.update(packet.episode_index, packet.frame_index, packet.timestamp, result)
                require(row.get("monitor") == tracked, "Stage or failure tracking differs from deterministic replay")
                if tracked["alert"]:
                    expected_alerts.append(row)
                devices.add(verify_metrics(row["metrics"], config, provenance, len(request.frames)))
                count += 1
                record_index += 1
        observed_episodes.append({"episode_index": source.episode_index, "frames_seen": frames_seen,
                                  "model_calls": count, "source_frames": source.length,
                                  "full_episode_read": frames_seen == source.length})
    require(record_index == len(records), "Unexpected predictions outside the configured sampling budget")
    require(valid > 0, "No valid monitor responses were recorded")
    require(alerts == expected_alerts, "Alert log differs from confirmed, deduplicated monitoring decisions")
    require(summary.get("episodes") == observed_episodes, "Episode coverage summary differs from source replay")
    if "sampling_coverage" in summary:
        expected_coverage = {
            "expected_model_calls": sum(scheduled_calls(source.length, monitor) for source in sources),
            "recorded_model_calls": record_index,
            "complete": len(observed_episodes) == len(sources) and all(
                episode["full_episode_read"] and episode["model_calls"] == scheduled_calls(source.length, monitor)
                for source, episode in zip(sources, observed_episodes)),
            "include_last_frame": monitor.get("include_last_frame", False),
        }
        require(summary["sampling_coverage"] == expected_coverage, "Sampling coverage differs from source replay")
    require(summary.get("max_frames_per_call") == monitor["history_size"] + 1,
            "Summary maximum frame count differs from configuration")
    expected_counts = {"model_calls": len(records), "valid_outputs": valid,
                       "invalid_outputs": len(records) - valid, "failure_alerts": len(alerts)}
    for name, value in expected_counts.items():
        require(summary.get(name) == value, f"Summary {name} differs from verified records")
    finite_nonnegative(summary.get("wall_time_s"), "wall_time_s", positive=True)
    finite_nonnegative(provenance.get("startup_gpu_wait_time_s", 0) or 0, "startup_gpu_wait_time_s")
    gpu_only = all(device.startswith("cuda") for device in devices)
    test_backend = provenance.get("compute_device") == "test_backend"
    report = {
        "status": "passed",
        "run_dir": run_dir.relative_to(ROOT).as_posix() if run_dir.is_relative_to(ROOT) else str(run_dir),
        "validation_purpose": "test_backend_functional_check" if test_backend else
                              "gpu_monitor_smoke_check" if gpu_only else "cpu_functional_check",
        "compute_devices": sorted(devices),
        **expected_counts,
        "source_images_verified": len(source_images),
        "window_image_references_verified": references_checked,
        "sampled_sequence_verified": True,
        "recorded_program_prompt_and_schema_verified": True,
        "causal_windows_verified": True,
        "state_vectors_and_gripper_values_verified": True,
        "raw_responses_verified": True,
        "tracking_and_alert_deduplication_verified": True,
        "mean_inference_s": statistics.mean(row["metrics"]["inference_ms"] / 1000 for row in records),
        "gpu_wait_time_s": (provenance.get("startup_gpu_wait_time_s") or 0)
                           + sum(row["metrics"].get("gpu_wait_time_s", 0) for row in records),
        "summary": summary,
        "provenance": provenance,
        "predictions": [{name: row[name] for name in
                         ("episode_index", "frame_index", "timestamp", "history_frames", "result_valid",
                          "result", "monitor", "metrics")} for row in records],
        "quality_evaluation": "not performed; independent human temporal labels are required",
        "model_weight_integrity": "not rechecked; this verifier does not hash model weights or verify which weights generated the logs",
        "observed_limitations": [
            "Source/log/causality checks do not establish correctness of model evidence or task stages.",
            "Repeated model assertions are not independent proof of a failure or recovery.",
            "Alert thresholds are uncalibrated; no accuracy, recall, or false-alarm rate is claimed.",
            "Recorded inference time is not online alert latency.",
        ],
    }
    if output is not None:
        output = Path(output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print("PASS: source images/states, causal windows, raw responses, sampling, tracking, alerts and totals")
    if output is not None:
        print("Validation report:", output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--no-write", action="store_true", help="Audit without writing a validation report")
    args = parser.parse_args()
    run_dir = args.run_dir
    if run_dir is None:
        parent = ROOT / "output/monitor_runs"
        candidates = [path for path in parent.iterdir()
                      if path.is_dir() and (path / "summary.json").is_file()
                      and read_json(path / "summary.json").get("status") == "monitor_run_complete"] if parent.is_dir() else []
        if not candidates:
            parser.error("No complete monitor run; run run_task_monitor.py first")
        run_dir = sorted(candidates)[-1]
    verify(run_dir, args.config, None if args.no_write else args.output)
