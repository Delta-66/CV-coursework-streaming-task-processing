"""离线目视记录的版本化证据绑定；此模块不被在线监测器导入。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath

BINDING_VERSION = 2
FINGERPRINT_VERSION = "canonical-json-full-row-v1"
FINGERPRINT_DEFINITION = {"algorithm": "sha256", "encoding": "utf-8", "sort_keys": True,
                          "separators": [",", ":"], "ensure_ascii": False, "allow_nan": False,
                          "scope": "entire_decision_row_including_processing; reruns require explicit rebinding"}


def row_fingerprint(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                    allow_nan=False).encode("utf-8")).hexdigest()


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while data := stream.read(1024*1024):
            value.update(data)
    return value.hexdigest()


def load_run_rows(run_dir):
    run_dir = Path(run_dir)
    summary = json.loads((run_dir/"summary.json").read_text(encoding="utf-8"))
    result = {}
    for episode in summary["episodes"]:
        ep = episode["episode_for_audit_only"]
        if ep in result or episode["output_subdir"] != f"trajectory_{ep:06d}":
            raise ValueError("duplicate episode or unexpected run trajectory path")
        result[ep] = [json.loads(line) for line in (run_dir/episode["output_subdir"]/"decisions.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip()]
    return result


def source_binding_errors(review, run_dir, legacy=False):
    errors = []
    run_dir = Path(run_dir).resolve()
    source = review.get("source_run")
    if not isinstance(source, str) or not source or Path(source).resolve() != run_dir:
        errors.append("source_run_mismatch_or_missing")
    if not legacy:
        if review.get("binding_version") != BINDING_VERSION:
            errors.append("binding_version_missing_or_unsupported; use explicit migrate command")
        if review.get("row_fingerprint_version") != FINGERPRINT_VERSION:
            errors.append("row_fingerprint_version_missing_or_unsupported")
        if review.get("source_run_summary_sha256") != file_sha256(run_dir/"summary.json"):
            errors.append("source_run_summary_hash_mismatch_or_missing")
        if review.get("row_fingerprint_definition") != FINGERPRINT_DEFINITION:
            errors.append("row_fingerprint_definition_missing_or_inconsistent")
    elif review.get("binding_version") not in (None, BINDING_VERSION):
        errors.append("legacy_binding_version_unsupported")
    if legacy and "source_run_summary_sha256" in review and review["source_run_summary_sha256"] != file_sha256(run_dir/"summary.json"):
        errors.append("existing_source_run_summary_hash_mismatch")
    if legacy and "row_fingerprint_version" in review and review["row_fingerprint_version"] != FINGERPRINT_VERSION:
        errors.append("existing_row_fingerprint_version_unsupported")
    return errors


def _images(record, direction=False):
    if direction:
        return [record.get("image_path")], {record.get("image_path"): record.get("image_sha256")}
    return record.get("image_refs"), record.get("image_sha256")


def record_binding_errors(record, rows, run_dir, direction=False, legacy=False):
    """返回 errors,row；legacy仅供显式迁移验证，仍检查旧记录中所有已有绑定。"""
    errors = []
    ep, timestamp, ref = record.get("episode_for_audit_only"), record.get("timestamp"), record.get("observation_ref")
    if isinstance(ep, bool) or not isinstance(ep, int) or ep < 0:
        return ["invalid_episode_key"], None
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
        return ["invalid_timestamp_key"], None
    matched = [row for row in rows if row.get("timestamp") == timestamp]
    # 即使两个重复时间点的ref不同也拒绝：一个时刻必须映射到唯一决策。
    if len(matched) != 1:
        return ["unique_real_decision_not_found"], None
    row = matched[0]
    if ref is None and legacy:
        ref = row.get("observation_ref")
    if not isinstance(ref, str) or ref != row.get("observation_ref"):
        errors.append("observation_ref_mismatch_or_missing")
    if not legacy or "row_sha256" in record:
        if record.get("row_sha256") != row_fingerprint(row):
            errors.append("real_decision_hash_mismatch_or_missing")
    if not legacy or "row_fingerprint_version" in record:
        if record.get("row_fingerprint_version") != FINGERPRINT_VERSION:
            errors.append("row_fingerprint_version_missing_or_unsupported")
    image_refs, image_hashes = _images(record, direction)
    if not isinstance(image_refs, list) or not image_refs or not all(isinstance(path, str) and path for path in image_refs):
        return errors+["image_refs_missing_or_invalid"], row
    if len(image_refs) != len(set(image_refs)) or not isinstance(image_hashes, dict) or set(image_refs) != set(image_hashes):
        return errors+["image_hash_keys_mismatch_or_duplicate_refs"], row
    root = Path(run_dir).resolve()
    available_refs = row.get("visual_facts", {}).get("evidence_refs", [])
    historical = record.get("allowed_history_refs", [])
    if not isinstance(historical, list):
        return errors+["invalid_allowed_history_refs"], row
    for relative in image_refs:
        normalized = relative.replace("\\", "/")
        pure = PurePosixPath(normalized)
        if pure.is_absolute() or ".." in pure.parts or ":" in normalized or normalized != relative:
            errors.append("image_path_not_safe_relative:"+relative)
            continue
        path = (root/normalized).resolve()
        if root not in path.parents:
            errors.append("image_path_outside_run:"+relative)
            continue
        camera = next((name for name in ("cam_high", "cam_right_wrist") if normalized.endswith("_"+name+".jpg")), None)
        current = f"trajectory_{ep:06d}/frames/{ref}_{camera}.jpg"
        modality = row.get("modalities", {}).get(camera, {})
        allowed = normalized == current and modality.get("status") == "available" and modality.get("timestamp") == timestamp
        if not allowed:
            for item in historical:
                if not isinstance(item, dict) or not {"ref", "camera", "timestamp"} <= set(item):
                    continue
                expected = f"trajectory_{ep:06d}/frames/{item['ref']}_{item['camera']}.jpg"
                causal = isinstance(item["timestamp"], (float, int)) and item["timestamp"] <= timestamp
                referenced = any(all(candidate.get(key) == item[key] for key in ("ref", "camera", "timestamp")) for candidate in available_refs)
                allowed = allowed or normalized == expected and camera == item["camera"] and causal and referenced
        if not allowed:
            errors.append("image_not_current_or_explicit_causal_reference:"+relative)
        if not path.is_file():
            errors.append("image_missing:"+relative)
        elif image_hashes.get(relative) != file_sha256(path):
            errors.append("image_sha256_mismatch:"+relative)
    return errors, row


def validate_review_bindings(review, rows_by_episode, run_dir):
    source_errors = source_binding_errors(review, run_dir)
    result = {"source_errors": source_errors, "checkpoints": [], "conditional_directions": []}
    for key, direction in (("checkpoints", False), ("conditional_directions", True)):
        for record in review.get(key, []):
            errors, row = record_binding_errors(record, rows_by_episode.get(record.get("episode_for_audit_only"), []), run_dir, direction)
            result[key].append({"status": "invalid" if source_errors or errors else "valid",
                                "errors": source_errors+errors, "timestamp": record.get("timestamp"),
                                "episode_for_audit_only": record.get("episode_for_audit_only"),
                                "observation_ref": row.get("observation_ref") if row else None})
    result["valid"] = not source_errors and all(item["status"] == "valid" for key in
                                                ("checkpoints", "conditional_directions") for item in result[key])
    return result


def _stamp(review, run_dir):
    review.update(binding_version=BINDING_VERSION, row_fingerprint_version=FINGERPRINT_VERSION,
                  row_fingerprint_definition=FINGERPRINT_DEFINITION, source_run=str(Path(run_dir).resolve()),
                  source_run_summary_sha256=file_sha256(Path(run_dir)/"summary.json"))


def migrate_review(review, run_dir):
    """仅验证通过的旧绑定可补行指纹；不静默修复错误路径、SHA或已有错误指纹。"""
    rows = load_run_rows(run_dir)
    errors = source_binding_errors(review, run_dir, legacy=True)
    if review.get("binding_version") == BINDING_VERSION:
        audit = validate_review_bindings(review, rows, run_dir)
        if not audit["valid"]:
            raise ValueError(json.dumps(audit, ensure_ascii=False))
        return deepcopy(review)
    migrated = deepcopy(review)
    for key, direction in (("checkpoints", False), ("conditional_directions", True)):
        for record in migrated.get(key, []):
            problems, row = record_binding_errors(record, rows.get(record.get("episode_for_audit_only"), []), run_dir, direction, legacy=True)
            errors += problems
            if not problems:
                record.update(observation_ref=row["observation_ref"], row_sha256=row_fingerprint(row), row_fingerprint_version=FINGERPRINT_VERSION)
    if errors:
        raise ValueError("legacy migration rejected: "+json.dumps(errors, ensure_ascii=False))
    _stamp(migrated, run_dir)
    migrated["binding_migration"] = {"from_version": "unversioned_legacy", "existing_images_and_hashes_verified": True,
                                     "existing_row_fingerprints_verified_when_present": True,
                                     "inference_repeated": False, "semantic_expectations_changed": False}
    return migrated


def rebind_review(review, old_run, new_run, reviewer, note):
    """同像素新回放的显式重绑定：保持人工预期，记录完整新行；不会替换不一致图像。"""
    if not reviewer.strip() or not note.strip():
        raise ValueError("explicit reviewer and review note required")
    old_rows, new_rows = load_run_rows(old_run), load_run_rows(new_run)
    old_audit = validate_review_bindings(review, old_rows, old_run)
    if not old_audit["valid"]:
        raise ValueError("old review must be valid v2 before rebinding: "+json.dumps(old_audit, ensure_ascii=False))
    rebound = deepcopy(review)
    changed = []
    for key, direction in (("checkpoints", False), ("conditional_directions", True)):
        for record in rebound.get(key, []):
            ep = record["episode_for_audit_only"]
            # 保留旧image SHA校验新图；只对已明确允许改变的行指纹解除旧值。
            candidate = deepcopy(record)
            candidate.pop("row_sha256")
            errors, row = record_binding_errors(candidate, new_rows.get(ep, []), new_run, direction, legacy=True)
            if errors:
                raise ValueError("new run does not match reviewed pixels/identity: "+json.dumps(errors, ensure_ascii=False))
            previous = next(item for item in old_rows[ep] if item["timestamp"] == record["timestamp"])
            changed.append({"episode_for_audit_only": ep, "timestamp": row["timestamp"], "old_row_sha256": record["row_sha256"],
                            "new_row_sha256": row_fingerprint(row),
                            "changed_top_level_fields": sorted(k for k in set(previous)|set(row) if previous.get(k) != row.get(k))})
            record["row_sha256"] = row_fingerprint(row)
    _stamp(rebound, new_run)
    rebound["binding_rebind"] = {"old_source_run": str(Path(old_run).resolve()), "reviewer": reviewer, "note": note,
                                 "same_reviewed_image_bytes": True, "semantic_expectations_changed": False,
                                 "inference_repeated": False, "changed_rows": changed}
    return rebound


def main():
    parser = argparse.ArgumentParser(description="迁移或显式重绑定离线目视记录；保留旧文件，不运行推理")
    parser.add_argument("mode", choices=("migrate", "rebind"))
    parser.add_argument("--review", required=True)
    parser.add_argument("--run", required=True, help="原记录绑定的运行")
    parser.add_argument("--new-run")
    parser.add_argument("--reviewer")
    parser.add_argument("--note")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    review = json.loads(Path(args.review).read_text(encoding="utf-8"))
    if args.mode == "migrate":
        result = migrate_review(review, args.run)
    else:
        if not args.new_run or not args.reviewer or not args.note:
            parser.error("rebind requires --new-run, --reviewer and --note")
        result = rebind_review(review, args.run, args.new_run, args.reviewer, args.note)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"output": str(output), "binding_version": BINDING_VERSION, "inference_repeated": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
