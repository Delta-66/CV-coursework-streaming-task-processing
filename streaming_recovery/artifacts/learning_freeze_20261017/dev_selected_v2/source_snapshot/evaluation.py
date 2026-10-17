"""离线验收：标签仅在已有推理输出生成后读取，绝不送回监测器。"""
from __future__ import annotations

import argparse
from collections import Counter
from enum import Enum
import hashlib
import json
from pathlib import Path
from .review_binding import row_fingerprint, validate_review_bindings


class Outcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"


def parse_outcome(value):
    """精确枚举，拒绝字符串真值、bool、缺失与未知拼写。"""
    if not isinstance(value, str) or value not in {"success", "failure"}:
        raise ValueError(f"invalid episode_success: {value!r}")
    return Outcome(value)


def notification_kind(row):
    if not row.get("notify", False):
        return None
    reason = row.get("notification_reason")
    event = row.get("failure_event") or {}
    if reason == "recovery_observed_without_execution_attribution" or event.get("state") in {"resolved", "retracted"}:
        return "recovery"
    if reason == "new_attempt_observed" or event.get("state") == "closed_for_retry":
        return "retry"
    if reason == "evidence_confirmed" or event.get("state") in {"confirmed", "persistent"}:
        return "failure_confirmation"
    if reason == "new_suspicion" or event.get("state") == "suspected":
        return "suspicion"
    return "other"


def outcome_statistics(label, rows, task_success):
    outcome = parse_outcome(label)
    counts = Counter(notification_kind(row) for row in rows if row.get("notify"))
    confirmed_events = sorted({(row.get("failure_event") or {}).get("event_id", "unidentified")
                               for row in rows if row.get("execution_status") == "confirmed_failure"
                               or (row.get("failure_event") or {}).get("state") in {"confirmed", "persistent"}})
    confirmed = bool(confirmed_events)
    failure_alerts = counts["suspicion"] + counts["failure_confirmation"]
    return {
        "official_outcome": outcome.value,
        "confirmed_failure": confirmed,
        "confirmed_event_ids": confirmed_events,
        "failure_missed_at_episode_level": not confirmed if outcome is Outcome.FAILURE else None,
        "false_failure_alarm_on_normal": bool(failure_alerts) if outcome is Outcome.SUCCESS else None,
        "false_confirmed_failure_on_normal": confirmed if outcome is Outcome.SUCCESS else None,
        "completion_missed": task_success is not True if outcome is Outcome.SUCCESS else None,
        "false_task_success_on_failure": task_success is True if outcome is Outcome.FAILURE else None,
        "final_task_success": task_success,
        "notification_counts": {key: counts[key] for key in
                                ("suspicion", "failure_confirmation", "recovery", "retry", "other")},
    }


def replay_audit(expected_rows, parquet_rows, source_audit, timestamps, sample_every, fps, cameras):
    """分别检查完整源读取与决策采样；不可把 reader 的消费量当成期待长度。"""
    if isinstance(sample_every, bool) or not isinstance(sample_every, int) or sample_every < 1:
        raise ValueError("sample_every must be a positive integer")
    expected_times = [i / fps for i in range(0, parquet_rows, sample_every)]
    actual_frames = source_audit.get("frames_decoded", {})
    consumption = {
        "metadata_matches_parquet": expected_rows == parquet_rows,
        "parquet_consumed_completely": source_audit.get("rows_read") == parquet_rows,
        "both_videos_consumed_completely": all(actual_frames.get(camera) == parquet_rows for camera in cameras),
        "video_timestamps_valid": source_audit.get("invalid_video_timestamps") == 0,
    }
    coverage = len(timestamps) == len(expected_times) and all(abs(a-b) <= 1e-5 for a, b in zip(timestamps, expected_times))
    return {"metadata_expected_rows": expected_rows, "parquet_actual_rows": parquet_rows,
            "consumed_parquet_rows": source_audit.get("rows_read"),
            "consumed_video_frames": {camera: actual_frames.get(camera) for camera in cameras},
            "full_replay": all(consumption.values()), "full_replay_checks": consumption,
            "decision_sample_coverage": coverage, "expected_decision_count": len(expected_times),
            "actual_decision_count": len(timestamps), "sample_every_native_frames": sample_every,
            "last_expected_decision_timestamp": expected_times[-1] if expected_times else None,
            "last_actual_decision_timestamp": timestamps[-1] if timestamps else None}


def _path_value(value, dotted):
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def check_manual_moments(rows, checkpoints):
    """外部目视记录与日志独立比对；不把阶段名出现过当成时序正确。"""
    results = []
    for checkpoint in checkpoints:
        if "timestamp" in checkpoint:
            requested = checkpoint["timestamp"]
            row = min(rows, key=lambda row: abs(row["timestamp"]-requested)) if rows else None
            tolerance = checkpoint.get("tolerance_s", 0.001)
            candidates = [row] if row is not None and abs(row["timestamp"]-requested) <= tolerance else []
        else:
            start, end = checkpoint["start_s"], checkpoint["end_s"]
            if end < start:
                raise ValueError("manual checkpoint end precedes start")
            candidates = [row for row in rows if start <= row["timestamp"] <= end]
        mode = checkpoint.get("mode", "any")
        if mode not in {"any", "all"}:
            raise ValueError("manual checkpoint mode must be any or all")
        if not checkpoint.get("expected_stages") and not checkpoint.get("expected_fields"):
            raise ValueError("manual checkpoint requires explicit stage or predicate expectation")
        if not candidates:
            results.append({**checkpoint, "status": "not_observed", "actual_timestamp": None,
                            "differences": ["no decision within declared timestamp tolerance"]})
            continue
        comparisons = []
        for row in candidates:
            differences = []
            if "expected_stages" in checkpoint and row["state"]["stage"] not in checkpoint["expected_stages"]:
                differences.append(f"stage={row['state']['stage']}; expected one of {checkpoint['expected_stages']}")
            for key, expected in checkpoint.get("expected_fields", {}).items():
                actual = _path_value(row, key)
                if actual != expected:
                    differences.append(f"{key}={actual!r}; expected {expected!r}")
            comparisons.append({"timestamp": row["timestamp"], "stage": row["state"]["stage"],
                                "observation_ref": row.get("observation_ref"), "differences": differences})
        passing = [item for item in comparisons if not item["differences"]]
        passed = bool(passing) if mode == "any" else len(passing) == len(comparisons)
        row_result = passing[0] if passed else comparisons[0]
        differences = [] if passed else [difference for item in comparisons for difference in item["differences"]]
        results.append({**checkpoint, "status": "passed" if not differences else "gap",
                        "actual_timestamp": row_result["timestamp"], "actual_stage": row_result["stage"],
                        "observation_ref": row_result["observation_ref"], "differences": differences,
                        "matched_timestamps": [item["timestamp"] for item in passing], "comparisons": comparisons})
    return results


def capacity_audit(row, config):
    """复用在线容量契约，并检查所有图像字节与禁止历史；不是仅检查帧窗口。"""
    from types import SimpleNamespace
    from .engine import Monitor
    violations = []
    try:
        Monitor.check_capacity(SimpleNamespace(config=config), row)
    except (ValueError, KeyError, TypeError) as error:
        violations.append(str(error))
    usage, memory = row["memory_usage"], config["memory"]
    edge = config["model"]["image_edge"]
    additional_limits = {"image_bytes": 2*memory["frames"]*edge*edge*3,
                         "visual_reference_bytes": config["geometry"]["max_reference_images"]*edge*edge*3,
                         "visual_mask_bytes": config["geometry"]["max_masks"]*edge*edge,
                         "visual_calibration_entries": 2, "active_events": 1,
                         "model_request_queue_capacity": 1, "model_response_queue_capacity": 1,
                         "model_input_images": 3}
    violations += [f"capacity_exceeded:{key}" for key, limit in additional_limits.items() if usage.get(key, 0) > limit]
    return violations


def completion_chain(rows):
    reference = next((row["state"].get("cut_reference") for row in rows if row["state"].get("cut_reference")), None)
    operations = [row["timestamp"] for row in rows
                  if row["visual_facts"].get("measurements", {}).get("cut_operation") == "yes"
                  and row["state"]["preconditions"].get("reliable_grasp") is True]
    effects = [row["timestamp"] for row in rows if row["state"]["preconditions"].get("cut_effect_confirmed") is True]
    returned = [row["timestamp"] for row in rows if row["state"]["preconditions"].get("tool_returned") is True]
    complete = [row["timestamp"] for row in rows if row["state"].get("task_completed") is True]
    ordered = bool(reference and operations and effects and returned and complete
                   and reference["timestamp"] < operations[0] <= effects[0] <= complete[0]
                   and returned[0] <= complete[0])
    return {"reference": reference, "operation_times": operations, "confirmed_effect_times": effects,
            "current_return_confirmed_times": returned, "completion_times": complete,
            "reference_operation_effect_completion_ordered": ordered}


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while data := stream.read(1024*1024):
            value.update(data)
    return value.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def conditional_direction_checks(rows_by_episode, run_dir, records):
    """独立真实像素方向核查，不构造失败事件，不计作自然恢复或建议已执行。"""
    if not records:
        return []
    from .recovery import image_alignment_directions
    run_dir = Path(run_dir).resolve()
    results = []
    for record in records:
        binding_errors, review_gaps = [], []
        rows = rows_by_episode.get(record.get("episode_for_audit_only"), [])
        matching = [row for row in rows if row["timestamp"] == record.get("timestamp")
                    and row.get("observation_ref") == record.get("observation_ref")]
        directions, coordinate_frame = [], None
        if len(matching) != 1:
            binding_errors.append("unique_real_decision_not_found")
        else:
            row = matching[0]
            facts = row["visual_facts"]
            loc = facts.get("measurements", {}).get("localization", {})
            coordinate_frame = facts.get("localization_frame")
            camera = record.get("camera")
            expected_image = (f"trajectory_{record['episode_for_audit_only']:06d}/frames/"
                              f"{row['observation_ref']}_{camera}.jpg")
            if record.get("row_sha256") != row_fingerprint(row):
                binding_errors.append("real_decision_hash_mismatch")
            if record.get("image_path") != expected_image:
                binding_errors.append("image_does_not_belong_to_bound_real_decision")
            else:
                path = (run_dir/expected_image).resolve()
                if run_dir not in path.parents or not path.is_file() or record.get("image_sha256") != _sha(path):
                    binding_errors.append("real_image_hash_or_path_mismatch")
            if not facts.get("valid") or not facts.get("effect_usable") or facts.get("visibility") not in {"clear", "partial"}:
                binding_errors.append("current_visual_evidence_unusable")
            if facts.get("provenance") not in ("real_multiview_effect_geometry", "real_fixed_supervised_fusion") or loc.get("status") != "localized":
                binding_errors.append("real_joint_localization_not_available")
            if record.get("localization_method") != loc.get("method") or not loc.get("method"):
                binding_errors.append("localization_source_mismatch")
            for key in ("knife_handle_xy", "gripper_xy", "localization_frame"):
                if record.get(key) != facts.get(key) or facts.get(key) is None:
                    binding_errors.append("bound_localization_mismatch:"+key)
            if coordinate_frame != f"{camera}_image_x_right_y_down":
                binding_errors.append("image_coordinate_frame_mismatch")
            modality = row.get("modalities", {}).get(camera, {})
            if modality.get("status") != "available" or modality.get("timestamp") != row["timestamp"]:
                binding_errors.append("current_image_not_available")
            if facts.get("gripper") != "open" or facts.get("contact") != "separate":
                review_gaps.append("conditional_alignment_requires_current_open_and_separate_evidence")
            if not binding_errors:
                directions = image_alignment_directions(facts["knife_handle_xy"], facts["gripper_xy"],
                                                         row["decision_rules"]["direction_deadband_normalized"])
                if not directions:
                    review_gaps.append("no_direction_outside_declared_deadband")
                if record.get("expected_directions") != directions:
                    review_gaps.append("shared_direction_rule_disagrees_with_visual_review")
        if not record.get("reviewer") or not isinstance(record.get("observed_features"), str) or not record["observed_features"].strip():
            review_gaps.append("visual_review_text_or_reviewer_missing")
        if record.get("reviewed_handle_neck") is not True or record.get("reviewed_two_finger_edges") is not True:
            review_gaps.append("handle_and_two_finger_edges_not_both_visually_reviewed")
        if record.get("coordinate_claim") != "image_plane_only_no_robot_mapping":
            review_gaps.append("coordinate_scope_not_explicitly_image_only")
        results.append({**record, "status": "invalid" if binding_errors else "gap" if review_gaps else "passed",
                        "binding_errors": binding_errors, "review_gaps": review_gaps,
                        "generated_direction": {"coordinate_frame": coordinate_frame,
                            "translation": {"kind": "qualitative_image_plane", "direction": directions, "magnitude": None},
                            "rotation": None, "numeric_command": None, "directly_executable": False,
                            "preconditions": ["current knife handle and both finger edges located", "current gripper open and object separate",
                                              "independent depth, orientation, calibration and safe path before physical motion"],
                            "post_checks": ["new image confirms finger midpoint and handle align", "independent grasp readiness remains required"]},
                        "evidence_kind": "conditional_direction_from_bound_real_pixels_and_visual_review",
                        "natural_failure_event_created": False, "natural_alignment_action": False,
                        "actually_executed": False, "execution_receipt": None, "outcome_attributed_to_suggestion": False})
    return results


def evaluate_run(config, run_dir, manual_review=None):
    """读取任意配置所选非测试轨迹；不运行模型、不修改历史输出。"""
    import pyarrow.parquet as pq
    from .recovery import validate_consistency

    run_dir, root = Path(run_dir), Path(config["source"]["root"])
    summary = _read(run_dir/"summary.json")
    snapshot = _read(run_dir/"config_snapshot.json")
    annotations = {row["episode_index"]: row for row in _jsonl(root/"meta/vifailback_annotations.jsonl")}
    metadata = {row["episode_index"]: row for row in _jsonl(root/"meta/episodes.jsonl")}
    info = _read(root/"meta/info.json")
    selected = config["source"]["episodes"]
    by_episode = {row["episode_for_audit_only"]: row for row in summary["episodes"]}
    checks = {"config_episode_selection_matches_run": set(selected) == set(by_episode),
              "config_matches_snapshot": config == snapshot,
              "real_data_declaration": summary.get("actual_data") is True and summary.get("synthetic") is False,
              "advisory_replay_mode": summary.get("execution_mode") == "advisory_replay",
              "selected_camera_pair": set(config["source"]["cameras"]) == {"cam_high", "cam_right_wrist"}}
    split_path = Path(__file__).resolve().parents[1]/"configs/splits/test.jsonl"
    frozen = {item["episode_index"] for item in _jsonl(split_path)}
    checks["no_frozen_test_episode"] = not (set(selected) & frozen)
    manual_review = manual_review or {"review_kind": "not_provided", "checkpoints": []}
    checkpoints = manual_review.get("checkpoints", [])
    per_episode = []
    rows_by_episode = {episode: _jsonl(run_dir/item["output_subdir"]/"decisions.jsonl")
                       for episode, item in by_episode.items()}
    bindings = validate_review_bindings(manual_review, rows_by_episode, run_dir)
    checks["manual_review_evidence_bindings_valid"] = bindings["valid"]
    capability_gaps = []
    for episode in selected:
        if episode not in by_episode:
            capability_gaps.append(f"episode {episode}: run missing")
            continue
        item = by_episode[episode]
        path = run_dir/item["output_subdir"]/"decisions.jsonl"
        rows = rows_by_episode[episode]
        chunks_size = info.get("chunks_size", 1000)
        parquet = root/f"data/chunk-{episode//chunks_size:03d}/episode_{episode:06d}.parquet"
        replay = replay_audit(metadata[episode]["length"], pq.ParquetFile(parquet).metadata.num_rows,
                              item["source_audit"], [row["timestamp"] for row in rows],
                              config["source"]["sample_every"], info["fps"], config["source"]["cameras"])
        stats = outcome_statistics(annotations[episode]["episode_success"], rows, item["end"]["task_success"])
        moments = []
        for checkpoint, binding in zip(checkpoints, bindings["checkpoints"]):
            if checkpoint.get("episode_for_audit_only") != episode:
                continue
            if binding["status"] == "invalid":
                moments.append({**checkpoint, "status": "invalid", "binding_errors": binding["errors"],
                                "differences": [], "semantic_comparison": "not_assessed_invalid_evidence"})
            else:
                moments.extend(check_manual_moments(rows, [checkpoint]))
        chain = completion_chain(rows)
        corrections, violations = Counter(), []
        for row in rows:
            if row.get("correction"):
                corrections[row["correction"]["low_level"]["action_type"]] += 1
                try:
                    validate_consistency(row)
                except (ValueError, KeyError, TypeError) as error:
                    violations.append({"timestamp": row["timestamp"], "error": str(error)})
        future = []
        for row in rows:
            refs = list(row["visual_facts"].get("evidence_refs", []))
            refs += row["visual_facts"].get("measurements", {}).get("joint_motion", {}).get("evidence_refs", [])
            future += [{"decision_timestamp": row["timestamp"], "reference": ref} for ref in refs if ref["timestamp"] > row["timestamp"]]
        execution_safe = all(row["execution"].get("actually_executed") is False
                             and row["execution"].get("execution_receipt") is None
                             and row["execution"].get("outcome_attributed_to_suggestion") is False for row in rows)
        capacity_violations = [{"timestamp": row["timestamp"], "errors": errors} for row in rows
                               if (errors := capacity_audit(row, config))]
        source_valid = len(item["source_files"]) == len(config["source"]["cameras"])+1 and all(
            Path(source["path"]).is_file() and _sha(source["path"]) == source["sha256"] for source in item["source_files"])
        local_checks = {"full_replay": replay["full_replay"], "sample_coverage": replay["decision_sample_coverage"],
                        "source_files_match_snapshot": source_valid, "advisory_semantics": execution_safe,
                        "logged_capacities_respected": not capacity_violations, "no_future_references": not future,
                        "correction_contracts_valid": not violations,
                        "notification_types_known": stats["notification_counts"]["other"] == 0}
        checks.update({f"episode_{episode}:{key}": value for key, value in local_checks.items()})
        gaps = []
        if stats["failure_missed_at_episode_level"]:
            gaps.append("failure_not_confirmed")
        if stats["false_failure_alarm_on_normal"] or stats["false_confirmed_failure_on_normal"]:
            gaps.append("failure_alarm_on_success")
        if stats["completion_missed"]:
            gaps.append("task_completion_not_confirmed")
        if stats["false_task_success_on_failure"]:
            gaps.append("task_success_claimed_on_failure")
        if stats["official_outcome"] == "success" and not chain["reference_operation_effect_completion_ordered"]:
            gaps.append("reference_operation_new_effect_completion_chain_incomplete")
        if not moments:
            gaps.append("manual_phase_checks_missing")
        elif any(m["status"] in {"gap", "not_observed"} for m in moments):
            gaps.append("manual_key_phase_mismatch_or_unsampled")
        capability_gaps += [f"episode {episode}: {gap}" for gap in gaps]
        per_episode.append({"episode_for_audit_only": episode, **stats, **replay,
                            "manual_moments": moments, "completion_evidence_chain": chain,
                            "semantic_status": "not_assessable" if any(m["status"] == "invalid" for m in moments) else "partial" if gaps else "passed", "capability_gaps": gaps,
                            "engineering_checks": local_checks, "correction_contract_violations": violations,
                            "capacity_violations": capacity_violations,
                            "action_counts": dict(corrections),
                            "joint_localization_decisions": sum(row["visual_facts"].get("knife_handle_xy") is not None
                                                                and row["visual_facts"].get("gripper_xy") is not None for row in rows),
                            "independent_readiness_decisions": sum(
                                all(row["visual_facts"].get("grasp_readiness", {}).get(key) == "confirmed"
                                    for key in ("depth", "orientation", "contact_region"))
                                and row["visual_facts"].get("grasp_readiness", {}).get("source") in
                                {"measured_depth_and_contact", "validated_controller_observation"} for row in rows),
                            "unknown_decision_timestamps": [row["timestamp"] for row in rows if row["execution_status"] == "information_insufficient"],
                            "unknown_stage_timestamps": [row["timestamp"] for row in rows if row["state"]["stage"] == "unknown"],
                            "elapsed_wall_s": item["elapsed_wall_s"], "decisions_sha256": _sha(path),
                            "alerts": [{"timestamp": row["timestamp"], "type": notification_kind(row),
                                        "event": row.get("failure_event"), "reason": row.get("notification_reason")}
                                       for row in rows if row.get("notify")]})
    localization = sum(row["joint_localization_decisions"] for row in per_episode)
    directions = sum(row["action_counts"].get("align_in_image", 0) for row in per_episode)
    conditional = conditional_direction_checks(rows_by_episode, run_dir, manual_review.get("conditional_directions", []))
    for item, binding in zip(conditional, bindings["conditional_directions"]):
        if binding["status"] == "invalid":
            item["status"] = "invalid"
            item["binding_errors"] = sorted(set(item["binding_errors"]+binding["errors"]))
    conditional_passed = sum(item["status"] == "passed" for item in conditional)
    checks["conditional_direction_real_evidence_bindings_valid"] = all(item["status"] != "invalid" for item in conditional)
    invalid_evidence = not bindings["valid"] or any(item["status"] == "invalid" for item in conditional)
    if not localization:
        capability_gaps.append("real_handle_and_two_finger_localization_not_observed")
    if not invalid_evidence and not directions and not conditional_passed:
        capability_gaps.append("neither_natural_nor_verified_conditional_real_direction_available")
    reviewed_directions = sum(c.get("reviewed_directional_action") is True and c["status"] == "passed"
                              and c.get("expected_fields", {}).get("correction.low_level.action_type") == "align_in_image"
                              for item in per_episode for c in item["manual_moments"])
    if not invalid_evidence and not reviewed_directions and not conditional_passed:
        capability_gaps.append("directional_action_visual_review_not_provided")
    if any(item["status"] == "gap" for item in conditional):
        capability_gaps.append("conditional_direction_visual_or_evidence_gap")
    engineering = all(checks.values())
    return {"report_version": 3, "run": str(run_dir.resolve()), "offline_only": True,
            "independent_labels_read_after_inference": True, "inference_repeated_for_this_report": False,
            "engineering_status": "passed" if engineering else "failed",
            "evidence_status": "invalid" if invalid_evidence else "valid",
            "semantic_status": "not_assessable" if invalid_evidence else "partial" if capability_gaps else "passed",
            "overall_status": "invalid" if invalid_evidence else "failed" if not engineering else "partial" if capability_gaps else "passed",
            "checks": checks, "per_episode": per_episode, "capability_gaps": capability_gaps,
            "binding_report": bindings,
            "conditional_direction_checks": conditional,
            "direction_evidence_scope": "Natural align counts are separate from conditional real-pixel direction checks; neither implies a natural retry or physical execution.",
            "manual_review_kind": manual_review.get("review_kind", "not_provided"),
            "manual_review_sha256": hashlib.sha256(json.dumps(manual_review, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest(),
            "elapsed_wall_s_from_original_run": summary["elapsed_wall_s"],
            "totals": {"native_rows": sum(r["consumed_parquet_rows"] for r in per_episode),
                       "decoded_images": sum(sum(r["consumed_video_frames"].values()) for r in per_episode),
                       "decisions": sum(r["actual_decision_count"] for r in per_episode),
                       "confirmed_failure_episodes": sum(r["confirmed_failure"] for r in per_episode),
                       "joint_localization_decisions": localization, "directional_action_decisions": directions,
                       "reviewed_directional_checkpoints": reviewed_directions,
                       "verified_conditional_real_directions": conditional_passed},
            "run_summary_sha256": _sha(run_dir/"summary.json"), "run_config_sha256": _sha(run_dir/"config_snapshot.json"),
            "run_code_sha256": summary.get("code_sha256", {}), "evaluator_sha256": _sha(__file__),
            "frozen_test_split_sha256": _sha(split_path),
            "detection_latency": None, "latency_reason": "no independently annotated first-detectable failure timestamp",
            "generalization_claim": False, "sample_exposure": "regression: all six trajectories have now been inspected",
            "not_verified_by_offline_log_audit": ["causal counterfactual full-chain test", "synthetic state regressions",
                "natural retry and slip perception", "physical execution", "independent depth/pose/contact readiness"]}


def main():
    parser = argparse.ArgumentParser(description="参数化真实回放离线验收；不重复推理、不向推理输入标签")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--manual-review")
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-complete", action="store_true", help="能力缺口也返回非零退出码")
    args = parser.parse_args()
    from .__main__ import load_config
    report = evaluate_run(load_config(args.config), args.run, _read(args.manual_review) if args.manual_review else None)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"report": str(output), "engineering_status": report["engineering_status"],
                      "semantic_status": report["semantic_status"], "totals": report["totals"]}, ensure_ascii=False))
    if report["engineering_status"] != "passed" or args.require_complete and report["overall_status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
