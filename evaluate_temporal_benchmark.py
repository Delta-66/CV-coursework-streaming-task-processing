"""Offline episode-level evaluation of the frozen canonical temporal benchmark.

Labels are joined only after inference. Missing, interrupted, or rejected runs
never enter the completed-episode metric denominators. No VQA answers are read.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parent
CANONICAL_EPISODES = 500
REQUIRED_ARTIFACTS = ("summary.json", "config.json", "provenance.json", "predictions.jsonl", "alerts.jsonl")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def decode_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def read_json(path):
    return decode_json(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    return [decode_json(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def inside(root, relative):
    require(isinstance(relative, str) and bool(relative), "A recorded relative path is required")
    path = (root / relative).resolve()
    require(path.is_relative_to(root), "Recorded path escapes the benchmark directory")
    return path


def sample_indices(length, sample_every, include_last_frame):
    require(type(length) is int and length > 0, "Episode length must be positive")
    require(type(sample_every) is int and sample_every > 0, "sample_every must be positive")
    require(type(include_last_frame) is bool, "include_last_frame must be boolean")
    indices = set(range(0, length, sample_every))
    if include_last_frame:
        indices.add(length - 1)
    return sorted(indices)


def fraction(numerator, denominator, expected_denominator=None):
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None,
            "expected_denominator": expected_denominator,
            "excluded_from_denominator": expected_denominator - denominator if expected_denominator is not None else None}


def load_reference(metadata_root, split_dir):
    """Read original whole-episode labels; never assign them to individual frames."""
    metadata_root, split_dir = Path(metadata_root).resolve(), Path(split_dir).resolve()
    split_protocol = read_json(split_dir / "protocol.json")
    test_path = split_dir / "test.jsonl"
    require(sha256(test_path) == split_protocol["artifacts"]["test.jsonl"]["sha256"],
            "Test manifest differs from the frozen split")
    for name, description in split_protocol["sources"]["metadata"]["files"].items():
        path = inside(metadata_root, name)
        require(sha256(path) == description["sha256"], "Metadata differs from the frozen split source")
    canonical = split_protocol["canonical_detection_benchmark_episode_indices"]
    require(isinstance(canonical, list) and len(canonical) == CANONICAL_EPISODES
            and all(type(index) is int and index >= 0 for index in canonical)
            and len(set(canonical)) == len(canonical), "Expected the complete canonical 500-episode cohort")
    test = {row["episode_index"] for row in read_jsonl(test_path)}
    require(set(canonical) <= test, "A canonical episode is exposed or absent from the frozen test set")
    info = read_json(metadata_root / "meta/info.json")
    fps = info["fps"]
    require(type(fps) in (int, float) and math.isfinite(fps) and fps > 0, "Invalid metadata FPS")
    episodes = read_jsonl(metadata_root / "meta/episodes.jsonl")
    annotations = read_jsonl(metadata_root / "meta/vifailback_annotations.jsonl")
    episode_by_id = {item["episode_index"]: item for item in episodes}
    annotation_by_id = {item["episode_index"]: item for item in annotations}
    require(len(episode_by_id) == len(episodes) and len(annotation_by_id) == len(annotations),
            "Duplicate episode metadata or annotations")
    reference = {}
    for index in sorted(canonical):
        require(index in episode_by_id and index in annotation_by_id, "A canonical episode lacks reference metadata")
        episode, annotation = episode_by_id[index], annotation_by_id[index]
        length = episode["length"]
        require(type(length) is int and length > 0 and annotation["num_frames"] == length,
                "Reference episode lengths disagree")
        outcome = annotation["episode_success"]
        require(outcome in ("failure", "success"), "Unknown official episode outcome")
        correct = annotation.get("correct_keyframe_indices", [])
        avoid = annotation.get("avoid_keyframe_indices", [])
        for indices in (correct, avoid):
            require(isinstance(indices, list) and all(type(value) is int and 0 <= value < length for value in indices),
                    "Official sparse keyframe index is outside its trajectory")
        reference[index] = {"episode_index": index, "num_frames": length, "outcome": outcome,
                            "correct_keyframe_indices": correct, "avoid_keyframe_indices": avoid}
    return reference, float(fps), split_protocol


def check_run_protocol(benchmark_dir, split_dir, reference, fps):
    protocol = read_json(benchmark_dir / "protocol.json")
    payload = {key: value for key, value in protocol.items() if key not in ("created_at", "protocol_sha256")}
    require(protocol.get("protocol_sha256") == fingerprint(payload), "Frozen benchmark protocol fingerprint differs")
    require(protocol.get("test_backend") is False, "Synthetic backends cannot produce formal test metrics")
    require(protocol.get("episode_indices") == sorted(reference), "Run cohort differs from the complete canonical test cohort")
    require(protocol.get("split_manifest_sha256") == sha256(split_dir / "test.jsonl"), "Run uses a different test manifest")
    require(protocol.get("split_protocol_sha256") == sha256(split_dir / "protocol.json"), "Run uses a different split protocol")
    config = protocol["config"]
    monitor = config["monitor"]
    require(monitor.get("max_model_calls") is None and monitor.get("max_samples_per_episode") is None,
            "Full-trajectory evaluation requires uncapped model calls")
    require(protocol.get("sample_every") == monitor["sample_every"]
            and protocol.get("include_last_frame") == monitor.get("include_last_frame"),
            "Frozen sampling declarations disagree")
    require(monitor["sample_every"] == 25 and monitor.get("include_last_frame") is True,
            "This full temporal benchmark is frozen to every 25 frames plus the final frame")
    require(isinstance(config["model"]["device"], str) and config["model"]["device"].startswith("cuda"),
            "The frozen benchmark must use GPU inference")
    require(type(config["tracking"]["failure_confirmation"]) is int and config["tracking"]["failure_confirmation"] > 0,
            "Invalid failure confirmation policy")
    checkpoint = read_json(benchmark_dir / "checkpoint.json")
    require(checkpoint.get("protocol_sha256") == protocol["protocol_sha256"], "Checkpoint belongs to another protocol")
    entries = checkpoint.get("episodes")
    require(isinstance(entries, dict), "Checkpoint must contain an episode mapping")
    require(set(entries) <= {str(index) for index in reference}, "Checkpoint includes episodes outside the frozen cohort")
    plan = read_json(benchmark_dir / "plan.json")
    require(protocol.get("plan_sha256") == fingerprint(plan), "Frozen plan fingerprint differs")
    planned = plan.get("episodes")
    require(isinstance(planned, list), "Missing episode plan")
    by_id = {item["episode_index"]: item for item in planned}
    require(len(by_id) == len(planned) and set(by_id) == set(reference), "Episode plan differs from the complete test cohort")
    total_calls = 0
    minimum_confirmation_calls = config["tracking"]["failure_confirmation"] + 1
    too_short = []
    for index, item in reference.items():
        indices = sample_indices(item["num_frames"], monitor["sample_every"], monitor["include_last_frame"])
        require(by_id[index].get("num_frames") == item["num_frames"]
                and by_id[index].get("expected_model_calls") == len(indices), "Planned frame or call counts differ from source metadata")
        total_calls += len(indices)
        if len(indices) < minimum_confirmation_calls:
            too_short.append(index)
    require(plan.get("total_frames") == sum(item["num_frames"] for item in reference.values())
            and plan.get("total_model_calls") == total_calls, "Plan totals differ from the complete source cohort")
    return protocol, checkpoint, {"episodes": len(reference), "total_frames": plan["total_frames"],
                                  "total_model_calls": total_calls, "source_duration_s": plan["total_frames"] / fps,
                                  "minimum_calls_for_confirmed_failure": minimum_confirmation_calls,
                                  "too_short_for_confirmation_episode_indices": too_short,
                                  "too_short_for_confirmation_episodes": len(too_short),
                                  "too_short_for_confirmation_failed_episodes": sum(reference[index]["outcome"] == "failure" for index in too_short),
                                  "confirmation_limit_note": "A temporal failure first requires past and current observations, followed by the frozen consecutive-confirmation count."}


def inspect_completed_episode(benchmark_dir, entry, reference, protocol, fps, allow_all_invalid_failure=False):
    index = reference["episode_index"]
    run_dir = inside(benchmark_dir, entry.get("run_dir"))
    hashes = entry.get("artifact_sha256")
    require(isinstance(hashes, dict) and set(REQUIRED_ARTIFACTS) <= set(hashes), "Completed entry lacks required artifact hashes")
    for name, digest in hashes.items():
        require(isinstance(digest, str), "Invalid artifact hash")
        require(sha256(inside(run_dir, name)) == digest, f"Completed artifact hash differs: {name}")
    summary = read_json(run_dir / "summary.json")
    config = read_json(run_dir / "config.json")
    provenance = read_json(run_dir / "provenance.json")
    records = read_jsonl(run_dir / "predictions.jsonl")
    alerts = read_jsonl(run_dir / "alerts.jsonl")
    run_complete = summary.get("status") == "monitor_run_complete"
    require(run_complete or (allow_all_invalid_failure and summary.get("status") == "monitor_run_failed"),
            "Episode run did not complete")
    require(config["source"].get("episodes") == [index], "Episode run config has a different source episode")
    for section in ("monitor", "tracking", "model"):
        require(config.get(section) == protocol["config"][section], f"Episode {section} differs from the frozen protocol")
    require(provenance.get("prompt_sha256") == protocol["prompt_sha256"], "Episode prompt differs from the frozen protocol")
    require(provenance.get("model") == protocol["model"], "Episode model manifest differs from the frozen protocol")
    code_hashes = provenance.get("program_sha256")
    require(isinstance(code_hashes, dict) and bool(code_hashes)
            and all(protocol["program_sha256"].get(name) == value for name, value in code_hashes.items()),
            "Episode program hashes differ from the frozen protocol")
    require(provenance.get("offline_annotations_used") is False and provenance.get("recorded_targets_used") is False,
            "Online label isolation is not established")
    require(provenance.get("cpu_fallback") is False
            and provenance.get("compute_device") == protocol["config"]["model"]["device"], "GPU inference policy differs")
    monitor = protocol["config"]["monitor"]
    expected = sample_indices(reference["num_frames"], monitor["sample_every"], monitor["include_last_frame"])
    require(entry.get("expected_model_calls") == len(expected), "Checkpoint expected call count differs")
    require([row.get("frame_index") for row in records] == expected, "Completed episode is missing or duplicates expected sampled frames")
    require(summary.get("episodes") == [{"episode_index": index, "frames_seen": reference["num_frames"],
                                        "model_calls": len(expected), "source_frames": reference["num_frames"],
                                        "full_episode_read": True}], "Completed episode does not establish full source replay")
    valid = 0
    statuses = Counter()
    expected_alerts = []
    inference_s = []
    wait_s = 0.0
    previous_time = -1.0
    for row in records:
        require(row.get("episode_index") == index, "Prediction belongs to another episode")
        timestamp = row.get("timestamp")
        require(type(timestamp) in (int, float) and math.isfinite(timestamp) and timestamp > previous_time,
                "Prediction timestamps must strictly increase")
        require(abs(timestamp - row["frame_index"] / fps) <= 1e-4, "Prediction timestamp differs from the source frame grid")
        previous_time = timestamp
        result = row.get("result")
        require(type(row.get("result_valid")) is bool, "Result validity must be explicit")
        if row["result_valid"]:
            require(isinstance(result, dict) and result.get("status") in
                    ("failure", "in_progress", "completed", "insufficient_evidence"), "Invalid validated result status")
            statuses[result["status"]] += 1
            valid += 1
        else:
            require(result is None, "Invalid output was converted into a diagnosis")
            statuses["invalid_output"] += 1
        tracked = row.get("monitor")
        require(isinstance(tracked, dict) and type(tracked.get("alert")) is bool, "Missing explicit alert decision")
        if tracked["alert"]:
            require(row["result_valid"] and result["status"] == "failure" and result.get("failure") is True,
                    "An invalid or uncertain result was converted into a failure alert")
            require(type(tracked.get("incident_id")) is int and tracked["incident_id"] > 0, "Alert lacks a traceable incident ID")
            expected_alerts.append(row)
        metrics = row["metrics"]
        require(metrics.get("compute_device") == protocol["config"]["model"]["device"], "A prediction used another compute device")
        duration = metrics.get("inference_ms")
        require(type(duration) in (int, float) and math.isfinite(duration) and duration > 0, "Invalid inference duration")
        inference_s.append(duration / 1000)
        waiting = metrics.get("gpu_wait_time_s", 0)
        require(type(waiting) in (int, float) and math.isfinite(waiting) and waiting >= 0, "Invalid GPU wait duration")
        wait_s += waiting
    require(alerts == expected_alerts, "Alert log differs from prediction decisions")
    counts = {"model_calls": len(expected), "valid_outputs": valid, "invalid_outputs": len(expected) - valid,
              "failure_alerts": len(alerts)}
    require(all(summary.get(name) == value for name, value in counts.items()), "Episode summary totals differ from records")
    require(not run_complete or valid > 0, "A completed run with zero valid outputs misrepresents a model-output failure")
    require(run_complete or (valid == 0 and not alerts),
            "Failed run is not an artifact-verified, fully executed all-invalid model-output failure")
    incidents = Counter(row["monitor"]["incident_id"] for row in alerts)
    abstentions = statuses["invalid_output"] + statuses["insufficient_evidence"]
    first_alert = alerts[0]["timestamp"] if alerts else None
    correct = reference["correct_keyframe_indices"]
    anchor_offset = (first_alert - correct[0] / fps if first_alert is not None
                     and reference["outcome"] == "failure" and len(correct) == 1 else None)
    return {"episode_index": index, "outcome": reference["outcome"], "model_calls": len(expected),
            "execution_class": "successful_complete_run" if run_complete else "all_invalid_full_execution_failure",
            "valid_outputs": valid, "invalid_outputs": len(expected) - valid,
            "insufficient_evidence_outputs": statuses["insufficient_evidence"],
            "abstaining_calls": abstentions, "all_calls_abstained": abstentions == len(expected),
            "alert_count": len(alerts), "failure_detected": bool(alerts),
            "first_alert_timestamp_s": first_alert,
            "additional_alerts_after_first": max(0, len(alerts) - 1),
            "duplicate_incident_alerts": sum(count - 1 for count in incidents.values()),
            "correct_keyframe_index": correct[0] if len(correct) == 1 else None,
            "first_alert_minus_correct_keyframe_s": anchor_offset,
            "inference_total_s": sum(inference_s), "gpu_wait_time_s": wait_s,
            "interpretation": "No alert means not detected; it does not establish a successful task."}


def evaluate(benchmark_dir, metadata_root=ROOT / "data/vifailback_lerobot", split_dir=ROOT / "configs/splits", output=None):
    benchmark_dir, split_dir = Path(benchmark_dir).resolve(), Path(split_dir).resolve()
    reference, fps, split_protocol = load_reference(metadata_root, split_dir)
    protocol, checkpoint, plan = check_run_protocol(benchmark_dir, split_dir, reference, fps)
    included, excluded = [], []
    checkpoint_statuses = Counter()
    for index in sorted(reference):
        entry = checkpoint["episodes"].get(str(index), {"status": "pending"})
        status = entry.get("status")
        require(status in ("pending", "running", "complete", "failed", "interrupted"), "Unknown checkpoint episode status")
        checkpoint_statuses[status] += 1
        if status not in ("complete", "failed"):
            excluded.append({"episode_index": index, "status": status, "reason": "Not a complete episode; excluded from all outcome metric denominators"})
            continue
        try:
            included.append(inspect_completed_episode(benchmark_dir, entry, reference[index], protocol, fps,
                                                       allow_all_invalid_failure=status == "failed"))
        except (ValueError, KeyError, TypeError, OSError) as error:
            excluded.append({"episode_index": index,
                             "status": "rejected_complete_record" if status == "complete" else "failed_not_fully_verified",
                             "checkpoint_status": status, "reason": str(error)})
    failures = [row for row in included if row["outcome"] == "failure"]
    successes = [row for row in included if row["outcome"] == "success"]
    offsets = [row["first_alert_minus_correct_keyframe_s"] for row in failures
               if row["first_alert_minus_correct_keyframe_s"] is not None]
    completed_calls = sum(row["model_calls"] for row in included)
    total_alerts = sum(row["alert_count"] for row in included)
    expected_failures = sum(row["outcome"] == "failure" for row in reference.values())
    expected_successes = len(reference) - expected_failures
    all_invalid_failures = sum(row["execution_class"] == "all_invalid_full_execution_failure" for row in included)
    report = {
        "schema_version": 1, "status": "complete" if len(included) == len(reference) else "interim",
        "protocol_sha256": protocol["protocol_sha256"],
        "cohort": "Frozen canonical task_success_detection 500 trajectories; six extra-reference trajectories excluded",
        "reference_label_source": "Original episode_success from meta/vifailback_annotations.jsonl joined to meta/episodes.jsonl",
        "vqa_answers_read": False,
        "plan": plan,
        "coverage": {"expected_episodes": len(reference), "fully_executed_episodes": len(included),
                     "successful_complete_runs": len(included) - all_invalid_failures,
                     "all_invalid_full_execution_failures": all_invalid_failures,
                     "excluded_episodes": len(excluded), "fraction": len(included) / len(reference),
                     "checkpoint_status_counts": dict(sorted(checkpoint_statuses.items())),
                     "completed_model_calls": completed_calls, "expected_model_calls": plan["total_model_calls"],
                     "completed_failure_episodes": len(failures), "expected_failure_episodes": expected_failures,
                     "completed_success_episodes": len(successes), "expected_success_episodes": expected_successes,
                     "denominator_policy": "Only artifact-verified, fully replayed and fully sampled episodes; fully executed all-invalid failures included conservatively as no alert and explicit abstentions; unfinished or unverified episodes excluded"},
        "metrics": {
            "population": {"included_episodes": len(included), "expected_episodes": len(reference),
                           "excluded_episodes": len(excluded), "all_invalid_full_execution_failures": all_invalid_failures},
            "episode_failure_detection_rate": fraction(sum(row["failure_detected"] for row in failures), len(failures), expected_failures),
            "success_episode_false_alarm_rate": fraction(sum(row["failure_detected"] for row in successes), len(successes), expected_successes),
            "failed_episodes_without_alert": sum(not row["failure_detected"] for row in failures),
            "total_alerts": total_alerts,
            "duplicate_same_incident_alerts": sum(row["duplicate_incident_alerts"] for row in included),
            "episodes_with_duplicate_same_incident_alerts": sum(row["duplicate_incident_alerts"] > 0 for row in included),
            "additional_alerts_after_first_per_episode": sum(row["additional_alerts_after_first"] for row in included),
            "invalid_output_calls": sum(row["invalid_outputs"] for row in included),
            "insufficient_evidence_calls": sum(row["insufficient_evidence_outputs"] for row in included),
            "abstaining_call_fraction": fraction(sum(row["abstaining_calls"] for row in included), completed_calls, plan["total_model_calls"]),
            "entirely_abstaining_episodes": sum(row["all_calls_abstained"] for row in included),
            "mean_inference_s": sum(row["inference_total_s"] for row in included) / completed_calls if completed_calls else None,
            "gpu_wait_time_s_in_completed_predictions": sum(row["gpu_wait_time_s"] for row in included),
        },
        "correct_keyframe_reference_offset": {
            "status": "available_for_detected_failed_episodes" if offsets else "no_detected_failed_episode_with_one_anchor",
            "reference": "Official sparse correct_keyframe_indices converted by the source 25 FPS frame grid",
            "definition": "first emitted alert source timestamp minus official correct keyframe source timestamp; signed seconds",
            "matched_detected_failed_episodes": len(offsets),
            "excluded_or_undetected_failed_episodes": expected_failures - len(offsets),
            "completed_failed_episodes_with_one_anchor": sum(len(reference[row["episode_index"]]["correct_keyframe_indices"]) == 1 for row in failures),
            "mean_signed_offset_s": statistics.mean(offsets) if offsets else None,
            "median_signed_offset_s": statistics.median(offsets) if offsets else None,
            "negative_offsets": sum(value < 0 for value in offsets),
            "limitation": "Sparse correction annotations are not the first detectable failure time; this is not detection-delay ground truth. Missed episodes have no offset and are never assigned zero delay."},
        "unavailable_metrics": {
            "stage_accuracy": "unavailable: no independent frame-level stage ground truth",
            "first_detectable_failure_latency": "unavailable: no human first-detectable-time annotations",
            "failure_event_recall_and_true_duplicate_event_rate": "unavailable: no per-event temporal boundaries or event IDs in the source labels",
            "frame_failure_accuracy": "unavailable: episode labels must not be copied onto each frame",
        },
        "interpretation": [
            "These are episode-level streaming alert metrics, not the official VQA benchmark score.",
            "Invalid and insufficient-evidence outputs are abstentions, not predictions of task success.",
            "Fully executed failed episodes with no alert remain end-to-end missed detections, including all-invalid technical/model failures; incomplete runs never enter the denominator.",
            "Duplicate incident IDs measure repeated system alerts; additional alerts may correspond to distinct real events and are not automatically errors.",
            "Partial results describe the completed subset only and cannot be claimed as full-test performance.",
            "Artifact hashes establish record integrity, not truth of the model evidence or identity of the loaded weight bytes.",
        ],
        "episodes": included, "excluded": excluded,
    }
    if output is not None:
        output = Path(output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(output)
    print(json.dumps({"status": report["status"], "included_episodes": len(included), "expected_episodes": len(reference),
                      "expected_model_calls": plan["total_model_calls"], "metrics": report["metrics"]}, ensure_ascii=False, allow_nan=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, default=ROOT / "data/vifailback_lerobot")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "configs/splits")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    evaluate(args.benchmark_dir, args.metadata_root, args.split_dir,
             None if args.no_write else args.output or args.benchmark_dir / "evaluation.json")
