"""Replay causal observation windows and retain task progress and failure alerts."""
import argparse
from contextlib import closing
from datetime import datetime
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import cv2
import yaml

from diagnostic_tracking import DiagnosticTracker
from stream_dataset import load_sources
from temporal_diagnosis import CausalWindow, QwenTemporalDiagnosis, SYSTEM_PROMPT, output_schema, parse_result


def load_config(config_path):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8-sig"))
    names = ("source", "monitor", "tracking", "model")
    if not isinstance(config, dict) or any(not isinstance(config.get(name), dict) for name in names):
        raise ValueError("source, monitor, tracking and model configurations are required")
    monitor = config["monitor"]
    if type(monitor.get("include_last_frame", False)) is not bool:
        raise ValueError("monitor.include_last_frame must be boolean")
    for name, minimum in (("sample_every", 1), ("max_image_edge", 32), ("history_size", 0)):
        if type(monitor.get(name)) is not int or monitor[name] < minimum:
            raise ValueError(f"Invalid monitor.{name}")
    if monitor["history_size"] > 32:
        raise ValueError("monitor.history_size must be <= 32")
    for name in ("max_samples_per_episode", "max_model_calls"):
        value = monitor.get(name)
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"monitor.{name} must be positive or null")
    if not isinstance(monitor.get("camera"), str) or not monitor["camera"].strip():
        raise ValueError("An explicit monitor.camera is required")
    model = config["model"]
    for name in ("max_new_tokens", "max_image_pixels"):
        if type(model.get(name)) is not int or model[name] <= 0:
            raise ValueError(f"Invalid model.{name}")
    if model["max_image_pixels"] < 65536:
        raise ValueError("model.max_image_pixels must be >= 65536")
    numeric = [("monitor.max_history_s", monitor.get("max_history_s"), False)]
    for name, default in (("max_inference_s", 120), ("min_free_cuda_mib", 5000),
                          ("gpu_poll_interval_s", 10), ("min_inference_free_cuda_mib", 1024)):
        numeric.append((f"model.{name}", model.get(name, default), False))
    tracking = config["tracking"]
    allowed = {"stage_confirmation", "failure_confirmation", "recovery_confirmation", "cooldown_s"}
    if set(tracking) != allowed:
        raise ValueError("tracking must define stage_confirmation, failure_confirmation, recovery_confirmation and cooldown_s")
    for name in allowed - {"cooldown_s"}:
        if type(tracking[name]) is not int or tracking[name] < 1:
            raise ValueError(f"Invalid tracking.{name}")
    numeric.append(("tracking.cooldown_s", tracking["cooldown_s"], True))
    for name, value, allow_zero in numeric:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
            raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    if model.get("gpu_poll_interval_s", 10) > 60:
        raise ValueError("model.gpu_poll_interval_s must be <= 60")
    for name in ("path", "device"):
        if not isinstance(model.get(name), str) or not model[name].strip():
            raise ValueError(f"An explicit model.{name} is required")
    if not isinstance(config.get("output_dir"), str) or not config["output_dir"].strip():
        raise ValueError("output_dir must be a nonempty path")
    return config


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")


def sampled_frame(frame_index, length, monitor):
    """Fixed causal cadence plus the terminal observation, never future pixels."""
    return frame_index % monitor["sample_every"] == 0 or (
        monitor.get("include_last_frame", False) and frame_index == length - 1)


def scheduled_calls(length, monitor):
    return len(range(0, length, monitor["sample_every"])) + int(
        length > 0 and monitor.get("include_last_frame", False)
        and (length - 1) % monitor["sample_every"] != 0)


def save_window(run_dir, request):
    """Save the exact resized RGB images; previous windows reuse the same PNGs."""
    window = []
    for frame in request.frames:
        folder = run_dir / "frames" / f"episode_{frame.episode_index:06d}"
        folder.mkdir(parents=True, exist_ok=True)
        image_path = folder / f"{frame.frame_index:06d}.png"
        if not image_path.exists():
            if not cv2.imwrite(str(image_path), cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR)):
                raise IOError("Cannot save actual model input image")
        window.append({"episode_index": frame.episode_index, "frame_index": frame.frame_index,
                       "timestamp": frame.timestamp, "timestamp_source": frame.timestamp_source,
                       "observations": frame.observations, "gripper_state": frame.gripper_state,
                       "image_path": image_path.relative_to(run_dir).as_posix(),
                       "image_shape": list(frame.image.shape),
                       "image_sha256": hashlib.sha256(frame.image.tobytes()).hexdigest(),
                       "source_reference": frame.source_reference})
    return window


def provenance(config, sources, backend):
    versions = {}
    for name in ("torch", "transformers", "accelerate", "tokenizers", "lm-format-enforcer", "numpy", "av", "pyarrow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    manifests = []
    for source in sources:
        root = getattr(source, "root", None)
        if root is None and getattr(source, "path", None) is not None:
            root = source.path.parents[2]
        manifest_path = Path(root) / "download_manifest.json" if root else None
        if manifest_path and manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifests.append({"episode_index": source.episode_index, "repo_id": manifest["repo_id"],
                              "revision": manifest["revision"]})
    code_hashes = {}
    for name in ("run_task_monitor.py", "temporal_diagnosis.py", "diagnostic_tracking.py", "current_observation.py", "dataset_inputs.py", "stream_dataset.py"):
        path = Path(__file__).resolve().parent / name
        code_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    monitor = config["monitor"]
    return {"mode": "causal_window_task_progress_and_failure_diagnosis",
            "model": getattr(backend, "manifest", {"backend": type(backend).__name__}),
            "model_load_time_s": getattr(backend, "load_time_s", None),
            "startup_gpu_wait_time_s": getattr(backend, "startup_gpu_wait_time_s", None),
            "compute_device": getattr(backend, "device", "test_backend"),
            "gpu_memory_policy": "pause_release_weights_wait_and_resume_same_input_on_gpu",
            "cpu_fallback": False, "versions": versions, "datasets": manifests,
            "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            "program_sha256": code_hashes, "history_size": monitor["history_size"],
            "max_frames_per_call": monitor["history_size"] + 1, "max_history_s": monitor["max_history_s"],
            "offline_annotations_used": False, "recorded_targets_used": False,
            "state_vectors_used": True, "state_policy": "whitelisted measured observations only; original field names and raw units retained",
            "previous_model_outputs_used_as_input": False, "tracking": config["tracking"]}


def run(config_path, backend=None, device=None, max_model_calls=None):
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    if device is not None:
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device override must be nonempty")
        config["model"]["device"] = device
    if max_model_calls is not None:
        if type(max_model_calls) is not int or max_model_calls <= 0:
            raise ValueError("max_model_calls override must be positive")
        config["monitor"]["max_model_calls"] = max_model_calls
    sources = load_sources(config, config_path)
    monitor = config["monitor"]
    for source in sources:
        if monitor["camera"] not in source.cameras:
            raise ValueError("Monitor camera must be selected in the input source")
    window = CausalWindow(monitor["history_size"], monitor["max_history_s"], monitor["max_image_edge"])
    tracker = DiagnosticTracker(**config["tracking"])
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    run_dir = (config_path.parent / config["output_dir"]).resolve() / now.strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "config.json", config)
    (run_dir / "prompt.txt").write_text(SYSTEM_PROMPT, encoding="utf-8")
    write_json(run_dir / "output_schema.json", output_schema())
    model_calls = 0
    valid_outputs = 0
    failure_alerts = 0
    observed_episodes = []
    started = time.perf_counter()
    try:
        if backend is None:
            model = config["model"]
            backend = QwenTemporalDiagnosis((config_path.parent / model["path"]).resolve(), model["device"],
                                             model["max_new_tokens"], model["max_image_pixels"],
                                             model.get("max_inference_s", 120), model.get("min_free_cuda_mib", 5000),
                                             model.get("gpu_poll_interval_s", 10), model.get("min_inference_free_cuda_mib", 1024))
        write_json(run_dir / "provenance.json", provenance(config, sources, backend))
        with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as file, \
             (run_dir / "alerts.jsonl").open("w", encoding="utf-8") as alerts:
            for source in sources:
                cap = monitor.get("max_model_calls")
                if cap is not None and model_calls >= cap:
                    break
                episode = {"episode_index": source.episode_index, "frames_seen": 0, "model_calls": 0,
                           "source_frames": source.length, "full_episode_read": False}
                observed_episodes.append(episode)
                with closing(source.iter_frames()) as frames:
                    for packet in frames:
                        episode["frames_seen"] += 1
                        if not sampled_frame(packet.frame_index, source.length, monitor):
                            continue
                        episode_cap = monitor.get("max_samples_per_episode")
                        if (episode_cap is not None and episode["model_calls"] >= episode_cap) or (cap is not None and model_calls >= cap):
                            continue
                        request = window.append(packet, monitor["camera"])
                        saved_window = save_window(run_dir, request)
                        # Complete this prediction before requesting another packet from the source.
                        raw_text, metrics = backend.predict(request)
                        model_calls += 1
                        episode["model_calls"] += 1
                        result = None
                        validation_error = None
                        try:
                            result = parse_result(raw_text, request)
                            valid_outputs += 1
                        except (ValueError, TypeError) as error:
                            validation_error = str(error)
                        tracked = tracker.update(packet.episode_index, packet.frame_index, packet.timestamp, result)
                        failure_alerts += int(tracked["alert"])
                        row = {"episode_index": packet.episode_index, "frame_index": packet.frame_index,
                               "timestamp": packet.timestamp, "timestamp_source": packet.timestamp_source,
                               "task_instruction": request.task_instruction, "camera": request.camera,
                               "history_frames": len(request.frames) - 1, "window": saved_window,
                               "raw_response": raw_text, "result_valid": result is not None, "result": result,
                               "validation_error": validation_error, "monitor": tracked, "metrics": metrics,
                               "evidence_review_status": "not_human_verified"}
                        line = json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                        file.write(line)
                        file.flush()
                        if tracked["alert"]:
                            alerts.write(line)
                            alerts.flush()
                        status = result["status"] if result else "invalid_output"
                        print(f"episode={packet.episode_index} frame={packet.frame_index} time={packet.timestamp:.2f}s "
                              f"status={status} tracked_stage={tracked['tracked_stage']} alert={tracked['alert']}", flush=True)
                episode["full_episode_read"] = episode["frames_seen"] == source.length
        if model_calls == 0:
            raise RuntimeError("No model calls were made")
        if valid_outputs == 0:
            raise RuntimeError("No valid model responses; inspect retained raw responses and validation errors")
        summary = {"status": "monitor_run_complete", "started_at": now.isoformat(), "model_calls": model_calls,
                   "valid_outputs": valid_outputs, "invalid_outputs": model_calls - valid_outputs,
                   "failure_alerts": failure_alerts, "episodes": observed_episodes,
                   "max_frames_per_call": monitor["history_size"] + 1, "wall_time_s": time.perf_counter() - started,
                   "evaluation_status": "pipeline and temporal alert rules only; model accuracy, missed failures and detection delay require labeled evaluation"}
        summary["sampling_coverage"] = {
            "expected_model_calls": sum(scheduled_calls(source.length, monitor) for source in sources),
            "recorded_model_calls": model_calls,
            "complete": len(observed_episodes) == len(sources) and all(
                episode["full_episode_read"] and episode["model_calls"] == scheduled_calls(source.length, monitor)
                for source, episode in zip(sources, observed_episodes)),
            "include_last_frame": monitor.get("include_last_frame", False),
        }
        write_json(run_dir / "summary.json", summary)
    except (Exception, KeyboardInterrupt) as error:
        write_json(run_dir / "summary.json", {
            "status": "monitor_run_interrupted" if isinstance(error, KeyboardInterrupt) else "monitor_run_failed",
            "started_at": now.isoformat(), "error": str(error) or type(error).__name__,
            "model_calls": model_calls, "valid_outputs": valid_outputs, "invalid_outputs": model_calls - valid_outputs,
            "failure_alerts": failure_alerts, "episodes": observed_episodes, "wall_time_s": time.perf_counter() - started})
        raise
    print("Task monitor results:", run_dir, flush=True)
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "configs/task_monitor.yaml")
    parser.add_argument("--device", help="Explicit device override, e.g. cpu or cuda:0; no automatic CPU fallback")
    parser.add_argument("--max-model-calls", type=int, help="Explicit positive call budget override")
    args = parser.parse_args()
    run(args.config, device=args.device, max_model_calls=args.max_model_calls)
