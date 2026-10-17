"""Build project trajectory splits without reading VQA answer fields or running inference.

All benchmark image/video references are reserved together, including references
outside the task_success_detection cohort. This is a project split, not a claim
that the separately published official training VQA file has been audited.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re

from download_vifailback import HubDownload, LEROBOT_REPO, RAW_REPO, sha256


ROOT = Path(__file__).resolve().parent
LEROBOT_REVISION = "619791b94f61667333fe179d471051e0bca0e06f"
RAW_REVISION = "21df46c2585c52e5a62df5f4b946bae6cbc6fed7"
BENCHMARK_FILE = "VQA/ViFailback_VQA_benchmark.json"
BENCHMARK_SHA256 = "211c7047577ee4327d232727bd92fddc6456383ed3751b4b940267b0a1058218"
DEFAULT_EXPOSURES = {0: "prior_model_development", 1: "annotation_review_this_audit", 18: "prior_model_development"}
SPLIT_NAMES = ("train", "validation", "dev_seen", "test", "exposed_benchmark")
METADATA_FILES = ("meta/info.json", "meta/tasks.jsonl", "meta/episodes.jsonl", "meta/vifailback_annotations.jsonl")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def normalized_path(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Dataset references must be nonempty POSIX paths")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or ":" in value:
        raise ValueError("Dataset reference must remain inside its repository")
    return path.as_posix()


def source_path_for_reference(value):
    """Bind raw HDF5, every camera, and every frame to the same trajectory key."""
    path = normalized_path(value)
    if re.fullmatch(r"raw_data/[^/]+/episode_[0-9]+\.hdf5", path):
        return path
    match = re.fullmatch(r"annotated_data/([^/]+)/frames/(episode_[0-9]+)_cam_[^/]+/[^/]+", path)
    if match is None:
        match = re.fullmatch(r"annotated_data/([^/]+)/videos/(episode_[0-9]+)_cam_[^/]+\.mp4", path)
    if match is None:
        raise ValueError(f"Unsupported benchmark reference path: {path}")
    return f"raw_data/{match[1]}/{match[2]}.hdf5"


def load_metadata(root):
    """Use original task instructions to merge collection folders into logical tasks."""
    root = Path(root)
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8-sig"))
    tasks = read_jsonl(root / "meta/tasks.jsonl")
    episodes = read_jsonl(root / "meta/episodes.jsonl")
    annotations = read_jsonl(root / "meta/vifailback_annotations.jsonl")
    task_by_instruction = {}
    task_ids = set()
    for task in tasks:
        if (type(task.get("task_index")) is not int or task["task_index"] < 0
                or not isinstance(task.get("task"), str) or not task["task"].strip()
                or task["task_index"] in task_ids or task["task"] in task_by_instruction):
            raise ValueError("Task metadata must have distinct integer IDs and complete instructions")
        task_by_instruction[task["task"]] = task["task_index"]
        task_ids.add(task["task_index"])
    episode_by_id = {}
    for episode in episodes:
        index = episode.get("episode_index")
        if type(index) is not int or index < 0 or index in episode_by_id:
            raise ValueError("Episode metadata has invalid or duplicate IDs")
        episode_by_id[index] = episode
    rows = {}
    source_paths = set()
    for annotation in annotations:
        index = annotation.get("episode_index")
        if type(index) is not int or index not in episode_by_id or index in rows:
            raise ValueError("Annotations must map once to each metadata episode")
        source_path = normalized_path(annotation["source_hdf5"])
        expected = f"raw_data/{annotation['source_task']}/{annotation['source_episode']}.hdf5"
        if source_path != expected or source_path_for_reference(source_path) != source_path or source_path in source_paths:
            raise ValueError("Raw source paths must identify distinct, consistently named trajectories")
        if annotation.get("source_repo_id") != RAW_REPO:
            raise ValueError("Unexpected raw source repository")
        source_paths.add(source_path)
        detail = annotation["annotation"]
        instruction = detail.get("task")
        if instruction not in task_by_instruction or episode_by_id[index].get("tasks") != [instruction]:
            raise ValueError("Annotation task must match the complete instruction in both task and episode metadata")
        num_frames = annotation.get("num_frames")
        if type(num_frames) is not int or num_frames <= 0 or num_frames != episode_by_id[index].get("length"):
            raise ValueError("Annotation frame count differs from episode metadata")
        outcome = annotation.get("episode_success")
        if outcome not in ("success", "failure"):
            raise ValueError("Unknown episode outcome; do not silently invent a stratification label")
        failure_type = detail.get("failure_type") if outcome == "failure" else "none"
        if not isinstance(failure_type, str) or not failure_type.strip():
            raise ValueError("Failure episodes need an explicit failure_type for stratification")
        rows[index] = {"episode_index": index, "source_hdf5": source_path,
                       "source_task": annotation["source_task"], "source_episode": annotation["source_episode"],
                       "task_index": task_by_instruction[instruction], "task_instruction": instruction,
                       "logical_task_sha256": hashlib.sha256(instruction.encode()).hexdigest(),
                       "num_frames": num_frames, "outcome": outcome, "failure_type": failure_type}
    if set(rows) != set(episode_by_id) or len(rows) != info.get("total_episodes") or len(tasks) != info.get("total_tasks"):
        raise ValueError("Metadata coverage differs between info, episodes, tasks and annotations")
    if sum(row["num_frames"] for row in rows.values()) != info.get("total_frames"):
        raise ValueError("Total source frame count differs from info.json")
    return rows, task_by_instruction


def map_benchmark(records, rows):
    """Inspect only top-level keys, tags and media paths; never inspect messages/answers."""
    if not isinstance(records, list) or not records:
        raise ValueError("Benchmark must be a nonempty record list")
    source_to_id = {row["source_hdf5"]: index for index, row in rows.items()}
    record_indices = defaultdict(list)
    tag_episodes = defaultdict(set)
    tag_counts = Counter()
    top_keys = set()
    path_count = 0
    for record_index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError("Benchmark record must be an object")
        top_keys.update(record)
        tag = record.get("tag")
        if not isinstance(tag, str) or not tag:
            raise ValueError("Benchmark record needs a nonempty tag")
        images = record.get("images", [])
        video = record.get("video", [])
        if isinstance(video, str):
            video = [video]
        if not isinstance(images, list) or not isinstance(video, list):
            raise ValueError("Benchmark images/video must contain path strings")
        paths = images + video
        if not paths:
            raise ValueError("Benchmark record has no media references")
        episode_ids = set()
        for path in paths:
            source_path = source_path_for_reference(path)
            if source_path not in source_to_id:
                raise ValueError(f"Benchmark reference has no local metadata mapping: {source_path}")
            episode_ids.add(source_to_id[source_path])
        if len(episode_ids) != 1:
            raise ValueError("One VQA record references multiple trajectories; split assignment is ambiguous")
        index = next(iter(episode_ids))
        record_indices[index].append(record_index)
        tag_episodes[tag].add(index)
        tag_counts[tag] += 1
        path_count += len(paths)
    canonical = tag_episodes.get("task_success_detection", set())
    if not canonical:
        raise ValueError("Cannot identify the detection cohort: task_success_detection tag is absent")
    return {"episode_indices": set(record_indices), "canonical_episode_indices": set(canonical),
            "extra_episode_indices": set(record_indices) - canonical,
            "record_indices": dict(record_indices), "record_count": len(records), "path_count": path_count,
            "top_level_keys": sorted(top_keys), "tag_record_counts": dict(sorted(tag_counts.items())),
            "tag_trajectory_counts": {tag: len(ids) for tag, ids in sorted(tag_episodes.items())}}


def rank(seed, value):
    return hashlib.sha256(json.dumps([seed, value], ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def stratified_partition(rows, indices, seed, validation_fraction):
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if (isinstance(validation_fraction, bool) or not isinstance(validation_fraction, (int, float))
            or not math.isfinite(validation_fraction) or not 0 < validation_fraction < 1):
        raise ValueError("validation_fraction must be finite and between zero and one")
    groups = defaultdict(list)
    for index in sorted(indices):
        row = rows[index]
        groups[(row["task_instruction"], row["outcome"], row["failure_type"])].append(index)
    keys = sorted(groups)
    minimum = {key: int(len(groups[key]) >= 2) for key in keys}
    maximum = {key: max(0, len(groups[key]) - 1) for key in keys}
    ideal = {key: len(groups[key]) * validation_fraction for key in keys}
    quota = {key: min(maximum[key], max(minimum[key], math.floor(ideal[key]))) for key in keys}
    target = min(sum(maximum.values()), max(sum(minimum.values()), math.floor(len(indices) * validation_fraction + 0.5)))
    while sum(quota.values()) != target:
        if sum(quota.values()) < target:
            eligible = [key for key in keys if quota[key] < maximum[key]]
            selected = min(eligible, key=lambda key: (quota[key] - ideal[key], rank(seed, key)))
            quota[selected] += 1
        else:
            eligible = [key for key in keys if quota[key] > minimum[key]]
            selected = min(eligible, key=lambda key: (ideal[key] - quota[key], rank(seed, key)))
            quota[selected] -= 1
    train, validation = set(), set()
    strata = []
    for key in keys:
        ordered = sorted(groups[key], key=lambda index: rank(seed, rows[index]["source_hdf5"]))
        validation.update(ordered[:quota[key]])
        train.update(ordered[quota[key]:])
        strata.append({"task_index": rows[ordered[0]]["task_index"], "outcome": key[1], "failure_type": key[2],
                       "total": len(ordered), "train": len(ordered) - quota[key], "validation": quota[key]})
    return train, validation, strata


def build_splits(rows, benchmark, exposures, seed=20261016, validation_fraction=0.2):
    if any(type(index) is not int or index not in rows or not isinstance(reason, str) or not reason
           for index, reason in exposures.items()):
        raise ValueError("Exposure ledger contains unknown episodes or invalid reasons")
    universe = set(rows)
    reserved = set(benchmark["episode_indices"])
    if not reserved <= universe:
        raise ValueError("Benchmark contains an unknown trajectory")
    development = universe - reserved
    dev_seen = development & set(exposures)
    exposed_benchmark = reserved & set(exposures)
    train, validation, strata = stratified_partition(rows, development - dev_seen, seed, validation_fraction)
    splits = {"train": train, "validation": validation, "dev_seen": dev_seen,
              "test": reserved - exposed_benchmark, "exposed_benchmark": exposed_benchmark}
    assigned = set()
    for name in SPLIT_NAMES:
        if assigned & splits[name]:
            raise ValueError("Split overlap")
        assigned.update(splits[name])
    if assigned != universe or set(exposures) & (validation | splits["test"]):
        raise ValueError("Split coverage or exposure isolation failed")
    return splits, strata


def verify_manifest(root, names, repo_id, revision):
    root = Path(root).resolve()
    manifest = json.loads((root / "download_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("repo_id") != repo_id or manifest.get("revision") != revision:
        raise ValueError("Download manifest differs from the pinned source revision")
    by_name = {item["path"]: item for item in manifest["files"]}
    result = {}
    for name in names:
        path = root / name
        digest = sha256(path)
        if name not in by_name or by_name[name].get("sha256") != digest or by_name[name].get("size") != path.stat().st_size:
            raise ValueError(f"Source file is absent from or disagrees with its download manifest: {name}")
        result[name] = {"sha256": digest, "bytes": path.stat().st_size}
    return {"repo_id": repo_id, "revision": revision, "files": result}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def check_existing_protocol(output_dir, sources, seed, validation_fraction, exposure_ledger, expected_contents=None):
    """Refuse silent holdout changes or overwriting a modified/incomplete manifest."""
    path = output_dir / "protocol.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        expected = {"sources": sources, "seed": seed, "validation_fraction_requested": validation_fraction,
                    "exposure_ledger": exposure_ledger}
        if any(existing.get(key) != value for key, value in expected.items()):
            raise FileExistsError("Frozen split protocol differs; use a new output directory and explicitly review a new protocol")
        for name, artifact in existing["artifacts"].items():
            artifact_path = output_dir / normalized_path(name)
            if not artifact_path.is_file() or sha256(artifact_path) != artifact["sha256"]:
                raise FileExistsError("Existing split artifacts were modified or are incomplete; refusing to overwrite")
        if expected_contents is not None:
            if set(expected_contents) != set(existing["artifacts"]):
                raise FileExistsError("Recomputed split artifacts differ from the frozen protocol; refusing to overwrite")
            for name, content in expected_contents.items():
                artifact_path = output_dir / normalized_path(name)
                saved = read_jsonl(artifact_path) if name.endswith(".jsonl") else json.loads(artifact_path.read_text(encoding="utf-8"))
                if content != saved:
                    raise FileExistsError("Recomputed split artifacts differ from the frozen protocol; refusing to overwrite")
    elif any((output_dir / name).exists() for name in [*(name + ".jsonl" for name in SPLIT_NAMES), "task_groups.json", "strata.json"]):
        raise FileExistsError("Output contains split files without a complete protocol; use a new output directory")


def prepare_splits(metadata_root, protocol_root, output_dir, summary_path, seed=20261016,
                   validation_fraction=0.2, exposures=None):
    metadata_root, protocol_root, output_dir = map(Path, (metadata_root, protocol_root, output_dir))
    exposures = dict(DEFAULT_EXPOSURES if exposures is None else exposures)
    sources = {"metadata": verify_manifest(metadata_root, METADATA_FILES, LEROBOT_REPO, LEROBOT_REVISION),
               "benchmark": verify_manifest(protocol_root, [BENCHMARK_FILE], RAW_REPO, RAW_REVISION)}
    if sources["benchmark"]["files"][BENCHMARK_FILE]["sha256"] != BENCHMARK_SHA256:
        raise ValueError("Benchmark SHA256 differs from the independently pinned digest")
    rows, task_by_instruction = load_metadata(metadata_root)
    # messages/answers are never accessed, printed, copied into manifests or sent to a model.
    records = json.loads((protocol_root / BENCHMARK_FILE).read_text(encoding="utf-8"))
    benchmark = map_benchmark(records, rows)
    del records
    splits, strata = build_splits(rows, benchmark, exposures, seed, validation_fraction)
    exposure_ledger = [{"episode_index": index, "reason": reason,
                        "assigned_split": "exposed_benchmark" if index in benchmark["episode_indices"] else "dev_seen"}
                       for index, reason in sorted(exposures.items())]
    development_tasks = {rows[index]["task_index"] for index in set(rows) - benchmark["episode_indices"]}
    shared = {index for index in benchmark["episode_indices"] if rows[index]["task_index"] in development_tasks}
    held_out = benchmark["episode_indices"] - shared
    # Construct every artifact before touching the output, so algorithm changes
    # cannot silently replace the held-out membership under the same seed.
    artifact_contents = {}
    for name in SPLIT_NAMES:
        artifact_contents[f"{name}.jsonl"] = []
        for index in sorted(splits[name]):
            row = rows[index]
            item = {key: row[key] for key in ("episode_index", "source_hdf5", "source_task", "source_episode",
                                              "task_index", "logical_task_sha256", "num_frames")}
            item.update(split=name, exposure_reason=exposures.get(index))
            if index in benchmark["episode_indices"]:
                item.update(benchmark_record_indices=benchmark["record_indices"][index],
                            benchmark_cohort="canonical_detection" if index in benchmark["canonical_episode_indices"] else "extra_media_reference",
                            domain_relative_to_nonbenchmark_pool="shared_task" if index in shared else "held_out_task")
            else:
                item["stratification"] = {"outcome": row["outcome"], "failure_type": row["failure_type"]}
            artifact_contents[f"{name}.jsonl"].append(item)
    task_groups = []
    for instruction, task_index in sorted(task_by_instruction.items(), key=lambda item: item[1]):
        task_ids = {index for index, row in rows.items() if row["task_index"] == task_index}
        task_groups.append({"task_index": task_index, "task_instruction": instruction,
                            "logical_task_sha256": hashlib.sha256(instruction.encode()).hexdigest(),
                            "source_tasks": sorted({rows[index]["source_task"] for index in task_ids}),
                            "trajectory_counts": {name: len(task_ids & ids) for name, ids in splits.items()}})
    artifact_contents["task_groups.json"] = task_groups
    artifact_contents["strata.json"] = strata
    check_existing_protocol(output_dir, sources, seed, validation_fraction, exposure_ledger, artifact_contents)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_files = {}
    for name, content in artifact_contents.items():
        path = output_dir / name
        if name.endswith(".jsonl"):
            with path.open("w", encoding="utf-8") as file:
                for item in content:
                    file.write(json.dumps(item, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            artifact_files[name] = {"sha256": sha256(path), "trajectories": len(content)}
        else:
            write_json(path, content)
            artifact_files[name] = {"sha256": sha256(path)}
    summary = {
        "status": "verified_project_split_created", "schema_version": 1,
        "split_authority": "Project trajectory split using every media reference in the pinned official benchmark as reserved data",
        "sources": sources, "script_sha256": sha256(Path(__file__)),
        "seed": seed, "validation_fraction_requested": validation_fraction,
        "algorithm": "SHA256(seed, source_hdf5) order within exact instruction/outcome/failure_type strata; bounded largest-remainder quotas; singleton strata remain in train",
        "grouping_unit": "source_hdf5 trajectory; all cameras, formats, frames, windows and VQA references stay together",
        "complete_metadata": {"trajectories": len(rows), "frames": sum(row["num_frames"] for row in rows.values()),
                              "source_task_directories": len({row["source_task"] for row in rows.values()}),
                              "logical_tasks": len(task_by_instruction)},
        "benchmark": {"vqa_records": benchmark["record_count"], "media_references": benchmark["path_count"],
                      "mapped_trajectories": len(benchmark["episode_indices"]), "unmapped_references": 0,
                      "ambiguous_records": 0, "top_level_keys": benchmark["top_level_keys"],
                      "tag_record_counts": benchmark["tag_record_counts"], "tag_trajectory_counts": benchmark["tag_trajectory_counts"],
                      "canonical_detection_trajectories": len(benchmark["canonical_episode_indices"]),
                      "extra_referenced_trajectories": len(benchmark["extra_episode_indices"]),
                      "answers_inspected_or_exported": False},
        "canonical_detection_benchmark_episode_indices": sorted(benchmark["canonical_episode_indices"]),
        "benchmark_extra_episode_indices": sorted(benchmark["extra_episode_indices"]),
        "benchmark_extra_source_hdf5": [rows[index]["source_hdf5"] for index in sorted(benchmark["extra_episode_indices"])],
        "nonbenchmark_development_pool": len(rows) - len(benchmark["episode_indices"]),
        "unexposed_development_pool": len(splits["train"] | splits["validation"]),
        "split_trajectory_counts": {name: len(ids) for name, ids in splits.items()},
        "exposure_ledger": exposure_ledger,
        "benchmark_exposure_episode_indices": sorted(splits["exposed_benchmark"]),
        "local_domain_comparison": {
            "authority": "Computed from task instruction equality and the nonbenchmark pool; not official ID/OOD labels",
            "shared_task_indices": sorted({rows[index]["task_index"] for index in shared}),
            "held_out_task_indices": sorted({rows[index]["task_index"] for index in held_out}),
            "shared_task_trajectories": len(shared), "held_out_task_trajectories": len(held_out)},
        "validation_fraction_actual": len(splits["validation"]) / len(splits["train"] | splits["validation"]) if splits["train"] | splits["validation"] else None,
        "strata_count": len(strata), "singleton_strata_in_train": sum(item["total"] == 1 for item in strata),
        "checks": {"pairwise_disjoint": True, "complete_trajectory_coverage": True,
                   "exposed_samples_absent_from_validation_and_test": True, "all_benchmark_references_reserved": True},
        "artifacts": artifact_files,
        "official_training_vqa": {"downloaded": False, "independently_verified": False,
                                  "note": "The nonbenchmark development pool is not asserted to equal the published training VQA list"},
        "usage_rules": [
            "Fit models on train only; dev_seen may be used for debugging and must never count as clean validation/test.",
            "Use validation for prompt/threshold selection; freeze choices before a one-time held-out test.",
            "Detection benchmark reporting must use the canonical detection cohort, subtract any exposed members, and report the extra media-reference cohort separately.",
            "A project streaming evaluation on these held-out trajectories is not automatically the official VQA benchmark metric.",
            "No model training, test inference, full-video download or test answer export is performed by this tool.",
        ],
        "count_reconciliation": "The detection-tag cohort and the union of all benchmark media references differ; reserve the union to prevent cross-task VQA leakage, rather than force published headline counts.",
    }
    write_json(output_dir / "protocol.json", summary)
    write_json(Path(summary_path), summary)
    print(json.dumps({"status": summary["status"], "metadata_trajectories": len(rows),
                      "benchmark_reference_trajectories": len(benchmark["episode_indices"]),
                      "canonical_detection_trajectories": len(benchmark["canonical_episode_indices"]),
                      "split_trajectory_counts": summary["split_trajectory_counts"]}, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, default=ROOT / "data/vifailback_lerobot")
    parser.add_argument("--protocol-root", type=Path, default=ROOT / "data/vifailback_protocol")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "configs/splits")
    parser.add_argument("--summary", type=Path, default=ROOT / "docs/data-split-summary.json")
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--download-benchmark", action="store_true", help="Download only the pinned benchmark JSON using the existing downloader")
    parser.add_argument("--proxy", help="Optional proxy for this download only")
    args = parser.parse_args()
    if args.download_benchmark:
        hub = HubDownload(RAW_REPO, args.protocol_root, RAW_REVISION, args.proxy)
        path = hub.download(BENCHMARK_FILE)
        if sha256(path) != BENCHMARK_SHA256:
            raise ValueError("Downloaded benchmark SHA256 differs from the pinned digest")
        hub.save_manifest([])
    prepare_splits(args.metadata_root, args.protocol_root, args.output_dir, args.summary,
                   args.seed, args.validation_fraction)
