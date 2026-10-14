"""Run a bounded current-observation baseline and retain raw, validated outputs."""
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

from current_observation import SYSTEM_PROMPT, QwenCurrentObservation, make_request, output_schema, parse_result
from stream_dataset import load_sources


def load_config(config_path):
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict) or any(not isinstance(config.get(name), dict) for name in ("source", "baseline", "model")):
        raise ValueError("source, baseline and model configurations are required")
    baseline = config["baseline"]
    for name, minimum in (("sample_every", 1), ("max_image_edge", 32)):
        if type(baseline.get(name)) is not int or baseline[name] < minimum:
            raise ValueError(f"Invalid baseline.{name}")
    for name in ("max_samples_per_episode", "max_model_calls"):
        value = baseline.get(name)
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"baseline.{name} must be positive or null")
    model = config["model"]
    for name in ("max_new_tokens", "max_image_pixels"):
        if type(model.get(name)) is not int or model[name] <= 0:
            raise ValueError(f"Invalid model.{name}")
    if model["max_image_pixels"] < 65536:
        raise ValueError("max_image_pixels must be >= 65536")
    for name, default in (("max_inference_s", 120), ("min_free_cuda_mib", 5000),
                          ("gpu_poll_interval_s", 10), ("min_inference_free_cuda_mib", 1024)):
        value = model.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"model.{name} must be finite and positive")
    if model.get("gpu_poll_interval_s", 10) > 60:
        raise ValueError("gpu_poll_interval_s must be <= 60")
    if not isinstance(baseline.get("camera"), str) or not baseline["camera"]:
        raise ValueError("An explicit camera is required")
    return config


def run(config_path, backend=None, device=None, max_model_calls=None):
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    if device is not None:
        config["model"]["device"] = device
    if max_model_calls is not None:
        if type(max_model_calls) is not int or max_model_calls <= 0:
            raise ValueError("max_model_calls override must be positive")
        config["baseline"]["max_model_calls"] = max_model_calls
    sources = load_sources(config, config_path)
    baseline = config["baseline"]
    for source in sources:
        if baseline["camera"] not in source.cameras:
            raise ValueError("Baseline camera must be selected in the input source")
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    run_dir = (config_path.parent / config["output_dir"]).resolve() / now.strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "prompt.txt").write_text(SYSTEM_PROMPT, encoding="utf-8")
    (run_dir / "output_schema.json").write_text(json.dumps(output_schema(), ensure_ascii=False, indent=2), encoding="utf-8")
    model_calls = 0
    valid_outputs = 0
    failure_alerts = 0
    observed_episodes = []
    started = time.perf_counter()
    try:
        if backend is None:
            model = config["model"]
            backend = QwenCurrentObservation((config_path.parent / model["path"]).resolve(), model["device"],
                                             model["max_new_tokens"], model["max_image_pixels"],
                                             model.get("max_inference_s", 120), model.get("min_free_cuda_mib", 5000),
                                             model.get("gpu_poll_interval_s", 10), model.get("min_inference_free_cuda_mib", 1024))
        versions = {}
        for name in ("torch", "transformers", "accelerate", "tokenizers", "lm-format-enforcer", "numpy", "av", "pyarrow"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        manifests = []
        for source in sources:
            root = getattr(source, "root", None)
            manifest_path = root / "download_manifest.json" if root else None
            if manifest_path and manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifests.append({"episode_index": source.episode_index, "repo_id": manifest["repo_id"], "revision": manifest["revision"]})
        provenance = {"mode": "current_observation_image_only", "model": getattr(backend, "manifest", {"backend": type(backend).__name__}),
                      "model_load_time_s": getattr(backend, "load_time_s", None),
                      "startup_gpu_wait_time_s": getattr(backend, "startup_gpu_wait_time_s", None),
                      "compute_device": getattr(backend, "device", "test_backend"),
                      "gpu_memory_policy": "pause_release_weights_wait_and_resume_same_input_on_gpu",
                      "cpu_fallback": False, "versions": versions,
                      "datasets": manifests, "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
                      "history_frames_per_call": 0, "offline_annotations_used": False,
                      "recorded_targets_used": False, "state_vectors_used": False}
        (run_dir / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
        with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as file:
            for source in sources:
                count = 0
                seen = 0
                cap = baseline.get("max_model_calls")
                if cap is not None and model_calls >= cap:
                    break
                with closing(source.iter_frames()) as frames:
                    for packet in frames:
                        seen += 1
                        if packet.frame_index % baseline["sample_every"]:
                            continue
                        episode_cap = baseline.get("max_samples_per_episode")
                        if (episode_cap is not None and count >= episode_cap) or (cap is not None and model_calls >= cap):
                            continue
                        request = make_request(packet, baseline["camera"], baseline["max_image_edge"])
                        folder = run_dir / "images" / f"episode_{packet.episode_index:06d}"
                        folder.mkdir(parents=True, exist_ok=True)
                        image_path = folder / f"{packet.frame_index:06d}.png"
                        if not cv2.imwrite(str(image_path), cv2.cvtColor(request.image, cv2.COLOR_RGB2BGR)):
                            raise IOError("Cannot save actual model input image")
                        raw_text, metrics = backend.predict(request)
                        model_calls += 1
                        result = None
                        validation_error = None
                        try:
                            result = parse_result(raw_text)
                            valid_outputs += 1
                        except (ValueError, TypeError) as error:
                            validation_error = str(error)
                        alert = result is not None and result["failure"] is True
                        failure_alerts += int(alert)
                        row = {"episode_index": packet.episode_index, "frame_index": packet.frame_index,
                               "timestamp": packet.timestamp, "timestamp_source": packet.timestamp_source,
                               "task_instruction": request.task_instruction, "camera": request.camera,
                               "model_image_path": image_path.relative_to(run_dir).as_posix(),
                               "model_image_shape": list(request.image.shape),
                               "model_image_sha256": hashlib.sha256(request.image.tobytes()).hexdigest(),
                               "source_image_reference": packet.source_references[request.camera],
                               "history_frames": 0, "raw_response": raw_text, "result": result,
                               "result_valid": result is not None, "validation_error": validation_error,
                               "evidence_review_status": "not_human_verified",
                               "alert": alert, "metrics": metrics}
                        file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                        file.flush()
                        count += 1
                        status = result["status"] if result else "invalid_output"
                        print(f"episode={packet.episode_index} frame={packet.frame_index} time={packet.timestamp:.2f}s status={status}", flush=True)
                observed_episodes.append({"episode_index": source.episode_index, "frames_seen": seen,
                                          "model_calls": count, "source_frames": source.length,
                                          "full_episode_read": seen == source.length})
        if model_calls == 0:
            raise RuntimeError("No model calls were made")
        if valid_outputs == 0:
            raise RuntimeError("No valid model responses; inspect retained raw responses and validation errors")
        summary = {"status": "baseline_run_complete", "started_at": now.isoformat(), "model_calls": model_calls,
                   "valid_outputs": valid_outputs, "invalid_outputs": model_calls-valid_outputs,
                   "failure_alerts": failure_alerts, "episodes": observed_episodes,
                   "history_frames_per_call": 0, "wall_time_s": time.perf_counter()-started,
                   "evaluation_status": "prediction pipeline only; accuracy and detection delay not evaluated"}
        (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    except (Exception, KeyboardInterrupt) as error:
        (run_dir / "summary.json").write_text(json.dumps({"status": "baseline_run_interrupted" if isinstance(error, KeyboardInterrupt) else "baseline_run_failed", "error": str(error) or type(error).__name__,
                                                         "model_calls": model_calls}, ensure_ascii=False, indent=2), encoding="utf-8")
        raise
    print("Baseline results:", run_dir, flush=True)
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "configs/current_observation.yaml")
    parser.add_argument("--device", help="Explicit device override, e.g. cpu or cuda:0")
    parser.add_argument("--max-model-calls", type=int, help="Explicit positive call budget override")
    args = parser.parse_args()
    run(args.config, device=args.device, max_model_calls=args.max_model_calls)
