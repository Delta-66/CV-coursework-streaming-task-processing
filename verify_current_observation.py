"""Verify baseline logs against the actual current source frames; no model calls."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import cv2
import numpy as np

from current_observation import make_request, parse_result
from stream_dataset import load_sources


ROOT = Path(__file__).resolve().parent


def verify(run_dir, config_path):
    run_dir = Path(run_dir).resolve()
    config_path = Path(config_path).resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert summary["status"] == "baseline_run_complete"
    assert summary["model_calls"] == len(records) > 0
    assert not provenance["offline_annotations_used"] and not provenance["recorded_targets_used"]
    assert provenance["history_frames_per_call"] == 0 and not provenance["state_vectors_used"]
    seen = set()
    valid = 0
    for row in records:
        identity = (row["episode_index"], row["frame_index"])
        assert identity not in seen
        seen.add(identity)
        assert row["history_frames"] == 0
        assert row["metrics"]["compute_device"] == config["model"]["device"]
        for name in ("gpu_wait_time_s", "gpu_resume_time_s", "oom_retry_time_s", "oom_retries"):
            assert row["metrics"].get(name, 0) >= 0
        if row["result_valid"]:
            assert parse_result(row["raw_response"]) == row["result"]
            valid += 1
    assert valid == summary["valid_outputs"]
    by_identity = {(row["episode_index"], row["frame_index"]): row for row in records}
    checked = 0
    for source in load_sources(config, config_path):
        for packet in source.iter_frames():
            row = by_identity.get((packet.episode_index, packet.frame_index))
            if row is None:
                continue
            assert row["timestamp"] == packet.timestamp
            assert row["source_image_reference"] == packet.source_references[row["camera"]]
            request = make_request(packet, row["camera"], config["baseline"]["max_image_edge"])
            assert request.task_instruction == row["task_instruction"]
            saved = cv2.imread(str(run_dir / row["model_image_path"]))
            assert saved is not None
            np.testing.assert_array_equal(cv2.cvtColor(saved, cv2.COLOR_BGR2RGB), request.image)
            assert hashlib.sha256(request.image.tobytes()).hexdigest() == row["model_image_sha256"]
            checked += 1
    assert checked == len(records)
    devices = sorted({row["metrics"]["compute_device"] for row in records})
    gpu_only = all(device.startswith("cuda") for device in devices)
    limitations = ["Model evidence and stages have not been human verified.",
                   "Recorded inference time is not online alert latency."]
    if any(row["result"] and row["result"]["stage"] == "approaching" for row in records):
        limitations.append("Approaching predictions require human review against visible task stages.")
    report = {"status": "passed", "run_dir": run_dir.relative_to(ROOT).as_posix(),
              "validation_purpose": "gpu_baseline_smoke_check" if gpu_only else "cpu_functional_check",
              "compute_devices": devices,
              "model_calls": len(records), "valid_outputs": valid,
              "current_source_images_verified": checked,
              "mean_inference_s": statistics.mean(row["metrics"]["inference_ms"]/1000 for row in records),
              "gpu_wait_time_s": (provenance.get("startup_gpu_wait_time_s") or 0) + sum(row["metrics"].get("gpu_wait_time_s", 0) for row in records),
              "summary": summary, "provenance": provenance,
              "predictions": [{"episode_index": row["episode_index"], "frame_index": row["frame_index"],
                               "timestamp": row["timestamp"], "result": row["result"],
                               "metrics": row["metrics"]} for row in records],
              "quality_evaluation": "not performed; no human first-detectable-time labels",
              "observed_limitations": limitations}
    output = ROOT / ("docs/current-observation-validation.json" if gpu_only else "docs/current-observation-functional-validation.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("PASS: current source images, raw response integrity, strict result validation and input isolation")
    print("Validation report:", output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/current_observation.yaml")
    args = parser.parse_args()
    run_dir = args.run_dir
    if run_dir is None:
        candidates = [path for path in (ROOT / "output/baseline_runs").iterdir()
                      if path.is_dir() and (path / "summary.json").is_file()
                      and json.loads((path / "summary.json").read_text(encoding="utf-8"))["status"] == "baseline_run_complete"]
        if not candidates:
            raise FileNotFoundError("No complete baseline run; run run_current_observation.py first")
        run_dir = sorted(candidates)[-1]
    verify(run_dir, args.config)
