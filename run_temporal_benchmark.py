"""Run every frozen canonical test trajectory with resumable episode checkpoints."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

from download_vifailback import sha256
from run_task_monitor import load_config as load_monitor_config, run as run_monitor
from temporal_diagnosis import QwenTemporalDiagnosis, SYSTEM_PROMPT, output_schema
from verify_task_monitor import verify as verify_monitor


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ("summary.json", "config.json", "provenance.json", "predictions.jsonl", "alerts.jsonl")
PROGRAM_FILES = ("run_temporal_benchmark.py", "run_task_monitor.py", "verify_task_monitor.py",
                 "temporal_diagnosis.py", "diagnostic_tracking.py", "current_observation.py",
                 "dataset_inputs.py", "stream_dataset.py", "download_temporal_benchmark.py",
                 "evaluate_temporal_benchmark.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    # Windows readers/indexers can briefly hold the old file without delete sharing.
    # Retry only the atomic rename; never truncate the last valid checkpoint.
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * 2 ** attempt)


def sample_indices(num_frames, sample_every=25):
    if type(num_frames) is not int or num_frames <= 0 or type(sample_every) is not int or sample_every <= 0:
        raise ValueError("Positive integer frame count and sampling interval are required")
    return sorted(set(range(0, num_frames, sample_every)) | {num_frames - 1})


def safe_relative(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Artifact path escapes its run directory")
    return path


def load_config(config_path):
    config_path = Path(config_path).resolve()
    config = load_monitor_config(config_path)
    benchmark = config.get("benchmark")
    if not isinstance(benchmark, dict):
        raise ValueError("benchmark configuration is required")
    for name in ("split_manifest", "split_protocol", "download_plan"):
        if not isinstance(benchmark.get(name), str) or not benchmark[name]:
            raise ValueError(f"benchmark.{name} must be a path")
        benchmark[name] = str((config_path.parent / benchmark[name]).resolve())
    for section, name in (("source", "root"), ("model", "path")):
        config[section][name] = str((config_path.parent / config[section][name]).resolve())
    config["output_dir"] = str((config_path.parent / config["output_dir"]).resolve())
    monitor = config["monitor"]
    if (monitor["sample_every"] != 25 or monitor.get("include_last_frame") is not True
            or monitor.get("max_samples_per_episode") is not None or monitor.get("max_model_calls") is not None):
        raise ValueError("Full temporal benchmark requires every 25th frame plus the last frame, without episode or global call caps")
    if not config["model"]["device"].startswith("cuda"):
        raise ValueError("Temporal benchmark must use CUDA; CPU inference is not a fallback")
    if config["source"].get("format") != "lerobot_v2.1" or config["source"].get("cameras") != [monitor["camera"]]:
        raise ValueError("Benchmark source must select exactly the configured LeRobot camera")
    return config


def build_plan(config):
    benchmark = config["benchmark"]
    split_path = Path(benchmark["split_manifest"])
    split_protocol = read_json(benchmark["split_protocol"])
    if sha256(split_path) != split_protocol["artifacts"][split_path.name]["sha256"]:
        raise ValueError("Test manifest differs from the frozen split protocol")
    rows = read_jsonl(split_path)
    if len({row["episode_index"] for row in rows}) != len(rows):
        raise ValueError("Test manifest contains duplicate episodes")
    selected = sorted((row for row in rows if row.get("benchmark_cohort") == "canonical_detection"), key=lambda row: row["episode_index"])
    expected = set(split_protocol["canonical_detection_benchmark_episode_indices"]) - set(split_protocol["benchmark_exposure_episode_indices"])
    if not selected or {row["episode_index"] for row in selected} != expected:
        raise ValueError("Selected episodes do not match the entire unexposed canonical detection cohort")
    if any(row.get("split") != "test" or row.get("exposure_reason") is not None for row in selected):
        raise ValueError("Only clean test trajectories can enter the benchmark")
    root = Path(config["source"]["root"])
    info = read_json(root / "meta/info.json")
    if info.get("fps") != 25:
        raise ValueError("The one-second sampling protocol requires a 25 FPS source")
    lengths = {row["episode_index"]: row["length"] for row in read_jsonl(root / "meta/episodes.jsonl")}
    download_path = Path(benchmark["download_plan"])
    download = read_json(download_path) if download_path.is_file() else None
    download_rows = {}
    if download is not None:
        if download.get("split_manifest_sha256") != sha256(split_path):
            raise ValueError("Download plan refers to a different test split")
        if download.get("revision") != split_protocol["sources"]["metadata"]["revision"]:
            raise ValueError("Download plan uses a different dataset revision")
        download_rows = {row["episode_index"]: row for row in download["episodes"]}
        if len(download_rows) != len(download["episodes"]) or set(download_rows) != expected:
            raise ValueError("Download plan must cover exactly the canonical test cohort")
    episodes = []
    for row in selected:
        index, length = row["episode_index"], row["num_frames"]
        if lengths.get(index) != length:
            raise ValueError("Test trajectory length differs from source metadata")
        values = {"episode_index": index, "episode_chunk": index // info["chunks_size"]}
        paths = [info["data_path"].format(**values), info["video_path"].format(video_key=config["monitor"]["camera"], **values)]
        remote = {item["path"]: item for item in download_rows.get(index, {}).get("files", [])}
        if download is not None and (set(remote) != set(paths) or download_rows[index]["num_frames"] != length):
            raise ValueError("Download plan camera/files/length differ from the monitoring protocol")
        files = []
        for relative in paths:
            path = safe_relative(root, relative)
            size = remote[relative]["size"] if relative in remote else path.stat().st_size if path.is_file() else None
            if size is not None and (type(size) is not int or size <= 0):
                raise ValueError("Input file size must be positive")
            files.append({"path": relative, "size": size, "remote_sha256": remote.get(relative, {}).get("remote_sha256")})
        episodes.append({"episode_index": index, "num_frames": length, "expected_model_calls": len(sample_indices(length)), "files": files})
    sizes = [item["size"] for episode in episodes for item in episode["files"]]
    return {"cohort": "canonical_detection", "episode_count": len(episodes), "episodes": episodes,
            "sample_every": 25, "include_last_frame": True, "total_frames": sum(row["num_frames"] for row in episodes),
            "total_model_calls": sum(row["expected_model_calls"] for row in episodes),
            "total_bytes": sum(sizes) if all(size is not None for size in sizes) else None,
            "known_bytes": sum(size for size in sizes if size is not None), "unknown_size_files": sum(size is None for size in sizes),
            "download_plan_sha256": sha256(download_path) if download is not None else None}


def make_protocol(config, plan, test_backend=False):
    model_path = Path(config["model"]["path"])
    model_manifest = read_json(model_path / "model_manifest.json")
    for item in model_manifest["files"]:
        path = safe_relative(model_path, item["path"])
        if not path.is_file() or path.stat().st_size != item["size"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Model file differs from its frozen manifest: {item['path']}")
    versions = {}
    for name in ("torch", "transformers", "accelerate", "tokenizers", "lm-format-enforcer", "numpy", "av", "pyarrow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    protocol = {"schema_version": 1, "config": config, "test_backend": test_backend,
                "episode_indices": [row["episode_index"] for row in plan["episodes"]],
                "sample_every": 25, "include_last_frame": True,
                "split_manifest_sha256": sha256(Path(config["benchmark"]["split_manifest"])),
                "split_protocol_sha256": sha256(Path(config["benchmark"]["split_protocol"])),
                "plan_sha256": fingerprint(plan), "model_manifest_sha256": sha256(model_path / "model_manifest.json"),
                "model": model_manifest, "versions": versions,
                "source_metadata_sha256": {name: sha256(Path(config["source"]["root"]) / "meta" / name)
                                           for name in ("info.json", "episodes.jsonl", "tasks.jsonl")},
                "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
                "output_schema_sha256": fingerprint(output_schema()),
                "program_sha256": {name: sha256(ROOT / name) for name in PROGRAM_FILES},
                "cpu_fallback": False, "offline_annotations_used_by_model": False}
    protocol["protocol_sha256"] = fingerprint(protocol)
    protocol["created_at"] = now()
    return protocol


def validate_protocol(protocol):
    actual = fingerprint({key: value for key, value in protocol.items() if key not in ("protocol_sha256", "created_at")})
    if actual != protocol.get("protocol_sha256"):
        raise ValueError("Frozen benchmark protocol hash is invalid")
    return actual


@contextmanager
def exclusive_run(run_dir):
    """OS locks are released on process death, allowing a safe later resume."""
    path = Path(run_dir) / ".run.lock"
    with path.open("a+b") as lock:
        if path.stat().st_size == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("This benchmark run is already active in another process") from error
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def check_complete(run_path, episode, config, allow_failed=False):
    summary = read_json(run_path / "summary.json")
    records = read_jsonl(run_path / "predictions.jsonl")
    expected = sample_indices(episode["num_frames"])
    expected_coverage = [{"episode_index": episode["episode_index"], "frames_seen": episode["num_frames"],
                          "source_frames": episode["num_frames"], "model_calls": len(expected), "full_episode_read": True}]
    allowed_statuses = ("monitor_run_complete", "monitor_run_failed") if allow_failed else ("monitor_run_complete",)
    if summary.get("status") not in allowed_statuses or summary.get("episodes") != expected_coverage:
        raise ValueError("Episode run did not read the complete trajectory")
    if [(row["episode_index"], row["frame_index"]) for row in records] != [(episode["episode_index"], index) for index in expected]:
        raise ValueError("Episode predictions do not cover every required sample and the final frame")
    if (summary.get("model_calls") != len(expected) or summary.get("valid_outputs") + summary.get("invalid_outputs") != len(expected)
            or read_json(run_path / "config.json") != config):
        raise ValueError("Episode counts or configuration differ from the frozen protocol")
    return summary, {name: sha256(run_path / name) for name in ARTIFACTS}


def progress(checkpoint):
    episodes = list(checkpoint["episodes"].values())
    complete = [item for item in episodes if item["status"] == "complete"]
    return {"status": checkpoint["status"], "completed_episodes": len(complete), "total_episodes": len(episodes),
            "completed_model_calls": sum(item["expected_model_calls"] for item in complete),
            "total_model_calls": sum(item["expected_model_calls"] for item in episodes),
            "failed_episodes": sum(item["status"] == "failed" for item in episodes),
            "current_episode": checkpoint.get("current_episode"), "updated_at": checkpoint["updated_at"]}


def verify_episode_inputs(episode, root, expected_hashes=None):
    digests = {}
    for item in episode["files"]:
        path = safe_relative(root, item["path"])
        if not path.is_file() or (item["size"] is not None and path.stat().st_size != item["size"]):
            raise FileNotFoundError(f"Missing or incomplete trajectory input: {item['path']}")
        digests[item["path"]] = sha256(path)
        if item["remote_sha256"] and digests[item["path"]] != item["remote_sha256"]:
            raise ValueError(f"Trajectory input SHA256 mismatch: {item['path']}")
    if expected_hashes is not None and digests != expected_hashes:
        raise ValueError("Trajectory inputs differ from the recorded inference inputs")
    return digests


def run(config_path, run_dir=None, backend=None, ensure_files=None, episode_runner=None, episode_verifier=None, plan_only=False):
    config = load_config(config_path)
    plan = build_plan(config)
    protocol = make_protocol(config, plan, test_backend=backend is not None and not isinstance(backend, QwenTemporalDiagnosis))
    run_dir = Path(run_dir).resolve() if run_dir is not None else Path(config["output_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    episode_runner = run_monitor if episode_runner is None else episode_runner
    episode_verifier = verify_monitor if episode_verifier is None else episode_verifier
    with exclusive_run(run_dir):
        protocol_path, checkpoint_path = run_dir / "protocol.json", run_dir / "checkpoint.json"
        if protocol_path.is_file():
            existing = read_json(protocol_path)
            if validate_protocol(existing) != protocol["protocol_sha256"]:
                raise ValueError("Cannot resume after model, prompt, rules, code, data plan or split protocol changes")
            if fingerprint(read_json(run_dir / "plan.json")) != existing["plan_sha256"]:
                raise ValueError("Saved benchmark plan was changed")
            checkpoint = read_json(checkpoint_path)
            if checkpoint.get("protocol_sha256") != existing["protocol_sha256"]:
                raise ValueError("Checkpoint belongs to a different protocol")
        else:
            if checkpoint_path.exists():
                raise ValueError("Checkpoint exists without its frozen protocol")
            atomic_json(protocol_path, protocol)
            atomic_json(run_dir / "plan.json", plan)
            checkpoint = {"schema_version": 1, "protocol_sha256": protocol["protocol_sha256"], "status": "planned" if plan_only else "running",
                          "created_at": now(), "updated_at": now(), "episodes": {
                              str(item["episode_index"]): {"episode_index": item["episode_index"], "status": "pending",
                                                           "expected_model_calls": item["expected_model_calls"], "attempts": []}
                              for item in plan["episodes"]}}
            atomic_json(checkpoint_path, checkpoint)
        if set(checkpoint["episodes"]) != {str(item["episode_index"]) for item in plan["episodes"]}:
            raise ValueError("Checkpoint episode membership differs from the frozen plan")

        def persist(status="running"):
            checkpoint["status"] = status
            checkpoint["updated_at"] = now()
            atomic_json(checkpoint_path, checkpoint)
            atomic_json(run_dir / "progress.json", progress(checkpoint))

        if plan_only:
            persist(checkpoint["status"])
            print(json.dumps({key: value for key, value in plan.items() if key != "episodes"}, ensure_ascii=False, indent=2))
            return run_dir

        try:
            for episode in plan["episodes"]:
                index = episode["episode_index"]
                entry = checkpoint["episodes"][str(index)]
                episode_config = {name: deepcopy(config[name]) for name in ("source", "monitor", "tracking", "model")}
                episode_config["source"]["episodes"] = [index]
                episode_config["output_dir"] = str(run_dir / "episodes" / f"episode_{index:06d}" / "runs")
                config_path_episode = run_dir / "configs" / f"episode_{index:06d}.json"
                if entry.get("run_dir") and entry.get("artifact_sha256"):
                    previous_path = safe_relative(run_dir, entry["run_dir"])
                    if {name: sha256(previous_path / name) for name in ARTIFACTS} != entry["artifact_sha256"]:
                        raise ValueError("Completed or failed episode artifacts changed; refusing silent recomputation")
                if entry["status"] == "complete" or entry.get("execution_complete") is True:
                    input_sha256 = verify_episode_inputs(episode, config["source"]["root"], entry.get("input_sha256"))
                    run_path = safe_relative(run_dir, entry["run_dir"])
                    _, digests = check_complete(run_path, episode, episode_config, allow_failed=entry["status"] != "complete")
                    if digests != entry["artifact_sha256"]:
                        raise ValueError("Completed episode artifacts changed; refusing silent recomputation")
                    entry["input_sha256"] = input_sha256
                    persist(checkpoint["status"])
                    continue
                if entry["status"] == "running" and entry["attempts"] and entry["attempts"][-1]["status"] == "running":
                    entry["attempts"][-1].update(status="interrupted", ended_at=now(), error="Previous process stopped before checkpoint completion")
                checkpoint["current_episode"] = index
                entry["status"] = "running"
                entry["config_path"] = config_path_episode.relative_to(run_dir).as_posix()
                atomic_json(config_path_episode, episode_config)
                attempt = {"started_at": now(), "status": "running"}
                entry["attempts"].append(attempt)
                persist()
                print(f"Benchmark episode={index}; {progress(checkpoint)['completed_episodes']}/{plan['episode_count']} complete", flush=True)
                started = time.perf_counter()
                previous_outputs = set(Path(episode_config["output_dir"]).glob("*/summary.json"))
                try:
                    # Recover a complete output written immediately before a prior process died.
                    candidates = sorted(Path(episode_config["output_dir"]).glob("*/summary.json"))
                    recovered = None
                    for candidate in candidates:
                        candidate_summary = read_json(candidate)
                        recoverable = candidate_summary.get("status") == "monitor_run_complete" or (
                            candidate_summary.get("status") == "monitor_run_failed"
                            and candidate_summary.get("valid_outputs") == 0
                            and candidate_summary.get("invalid_outputs") == episode["expected_model_calls"])
                        if recoverable:
                            try:
                                check_complete(candidate.parent, episode, episode_config, allow_failed=True)
                            except (ValueError, OSError, KeyError, TypeError):
                                continue
                            recovered = candidate.parent
                            break
                    if ensure_files is not None:
                        ensure_files(index)
                    entry["input_sha256"] = verify_episode_inputs(episode, config["source"]["root"], entry.get("input_sha256"))
                    persist()  # Persist the actual inputs before model construction or inference.
                    if recovered is None:
                        if backend is None:
                            model = config["model"]
                            backend = QwenTemporalDiagnosis(model["path"], model["device"], model["max_new_tokens"],
                                                             model["max_image_pixels"], model.get("max_inference_s", 120),
                                                             model.get("min_free_cuda_mib", 6500), model.get("gpu_poll_interval_s", 10),
                                                             model.get("min_inference_free_cuda_mib", 1024))
                        run_path = Path(episode_runner(config_path_episode, backend=backend)).resolve()
                    else:
                        run_path = recovered.resolve()
                    if not run_path.is_relative_to(run_dir):
                        raise ValueError("Episode runner returned a result outside the benchmark run")
                    summary, digests = check_complete(run_path, episode, episode_config, allow_failed=recovered is not None)
                    if summary["status"] == "monitor_run_failed":
                        entry.update(status="failed", run_dir=run_path.relative_to(run_dir).as_posix(), artifact_sha256=digests,
                                     recovered_after_interrupt=True, valid_outputs=0, invalid_outputs=summary["invalid_outputs"],
                                     execution_complete=True, failure_kind="all_invalid_model_outputs")
                        attempt.update(status="failed", run_dir=entry["run_dir"], error="Recovered complete execution with all model outputs invalid")
                    else:
                        episode_verifier(run_path, config_path_episode, output=run_path / "validation.json")
                        entry.update(status="complete", run_dir=run_path.relative_to(run_dir).as_posix(),
                                     artifact_sha256=digests, completed_at=now(), recovered_after_interrupt=recovered is not None,
                                     valid_outputs=summary["valid_outputs"], invalid_outputs=summary["invalid_outputs"],
                                     execution_complete=True, failure_kind=None)
                        attempt.update(status="complete", run_dir=entry["run_dir"])
                except KeyboardInterrupt:
                    entry["status"] = attempt["status"] = "interrupted"
                    attempt["error"] = "KeyboardInterrupt"
                    raise
                except Exception as error:
                    entry["status"] = attempt["status"] = "failed"
                    attempt["error"] = f"{type(error).__name__}: {error}"
                    new_outputs = sorted(set(Path(episode_config["output_dir"]).glob("*/summary.json")) - previous_outputs)
                    if new_outputs:
                        failed_path = new_outputs[-1].parent
                        if all((failed_path / name).is_file() for name in ARTIFACTS):
                            failed_summary = read_json(failed_path / "summary.json")
                            failed_rows = read_jsonl(failed_path / "predictions.jsonl")
                            complete_execution = (
                                [(row.get("episode_index"), row.get("frame_index")) for row in failed_rows]
                                == [(index, frame) for frame in sample_indices(episode["num_frames"])]
                                and failed_summary.get("episodes") == [{"episode_index": index, "frames_seen": episode["num_frames"],
                                    "source_frames": episode["num_frames"], "model_calls": episode["expected_model_calls"], "full_episode_read": True}])
                            entry.update(run_dir=failed_path.relative_to(run_dir).as_posix(),
                                         artifact_sha256={name: sha256(failed_path / name) for name in ARTIFACTS},
                                         execution_complete=complete_execution,
                                         valid_outputs=failed_summary.get("valid_outputs", 0), invalid_outputs=failed_summary.get("invalid_outputs", 0),
                                         failure_kind="all_invalid_model_outputs" if complete_execution and failed_summary.get("valid_outputs") == 0 else "episode_execution_error")
                            attempt["run_dir"] = entry["run_dir"]
                    print(f"Benchmark episode={index} failed: {attempt['error']}", flush=True)
                finally:
                    attempt["ended_at"] = now()
                    attempt["wall_time_s"] = time.perf_counter() - started
                    persist()
            checkpoint["current_episode"] = None
            persist("complete" if all(item["status"] == "complete" for item in checkpoint["episodes"].values()) else "incomplete")
        except KeyboardInterrupt:
            persist("interrupted")
            raise
        except Exception:
            persist("incomplete")
            raise
        if not protocol["test_backend"]:
            try:
                from evaluate_temporal_benchmark import evaluate
                evaluate(run_dir, metadata_root=Path(config["source"]["root"]),
                         split_dir=Path(config["benchmark"]["split_manifest"]).parent,
                         output=run_dir / "evaluation.json")
                checkpoint["evaluation"] = {"status": "complete", "path": "evaluation.json", "updated_at": now()}
            except Exception as error:
                checkpoint["evaluation_error"] = f"{type(error).__name__}: {error}"
                print("Benchmark evaluation failed:", checkpoint["evaluation_error"], flush=True)
            persist(checkpoint["status"])
    print(json.dumps(progress(checkpoint), ensure_ascii=False), flush=True)
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/temporal_benchmark.yaml")
    parser.add_argument("--run-dir", type=Path, help="Resume this run; completed trajectories are verified and skipped")
    parser.add_argument("--plan-only", "--plan", dest="plan", action="store_true", help="Freeze the protocol and full plan without loading a GPU model")
    parser.add_argument("--status", type=Path, nargs="?", const=Path("__default__"), help="Read progress; defaults to the configured benchmark run directory")
    parser.add_argument("--download", action="store_true", help="Prepare each episode's pinned files before running it")
    parser.add_argument("--proxy", help="Optional per-download proxy; no global settings change")
    args = parser.parse_args()
    if args.status:
        status_dir = Path(load_config(args.config)["output_dir"]) if args.status == Path("__default__") else args.status
        print(json.dumps(progress(read_json(status_dir / "checkpoint.json")), ensure_ascii=False, indent=2))
    elif args.plan:
        run(args.config, args.run_dir, plan_only=True)
    else:
        ensure_files = None
        if args.download:
            from download_temporal_benchmark import ensure_episode
            config = load_config(args.config)
            ensure_files = lambda index: ensure_episode(config["source"]["root"], index,
                                                        plan_path=config["benchmark"]["download_plan"], proxy=args.proxy)
        run(args.config, args.run_dir, ensure_files=ensure_files)
