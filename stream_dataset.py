"""Replay and record ViFailback inputs. No baseline, model or diagnosis is run."""
import argparse
import base64
from contextlib import ExitStack, closing
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import yaml

from dataset_inputs import HDF5Episode, LeRobotEpisode, jsonl


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"encoding": "base64", "data": base64.b64encode(value).decode("ascii")}
    raise TypeError(f"Unsupported recorded value: {type(value)}")


def dump(value):
    return json.dumps(value, ensure_ascii=False, default=serializable, allow_nan=False)


def load_sources(config, config_path):
    source = config["source"]
    def resolve(value):
        return (config_path.parent / value).resolve()
    if source["format"] == "lerobot_v2.1":
        episodes = source["episodes"]
        if not isinstance(episodes, list) or not episodes or len(set(episodes)) != len(episodes) or any(type(item) is not int or item < 0 for item in episodes):
            raise ValueError("episodes must contain distinct nonnegative integer IDs")
        return [LeRobotEpisode(resolve(source["root"]), episode, source.get("cameras")) for episode in episodes]
    if source["format"] == "vifailback_hdf5":
        annotation = None
        if source.get("annotation_path"):
            annotation = next((item for item in jsonl(resolve(source["annotation_path"]))
                               if item["episode_index"] == source["episode_index"]), None)
            if annotation is None:
                raise ValueError("No raw episode mapping in annotation_path")
            path = resolve(source["path"])
            suffix = Path(annotation["source_hdf5"]).parts
            if path.parts[-len(suffix):] != suffix:
                raise ValueError("Annotation source_hdf5 does not match the selected HDF5 path")
        task_instruction = source.get("task_instruction") or (annotation["annotation"]["task"] if annotation else None)
        return [HDF5Episode(resolve(source["path"]), task_instruction, source.get("episode_index", 0),
                            source.get("fps"), source.get("timestamp_path"), source.get("cameras"),
                            source.get("include_depth", True), source.get("rgb_storage_order", "RGB"), annotation)]
    raise ValueError("source.format must be lerobot_v2.1 or vifailback_hdf5")


def describe_image(run_dir, packet, name, image, depth, save):
    path = None
    if save:
        identifier = hashlib.sha256(name.encode()).hexdigest()[:12]
        folder = run_dir / "arrays" / f"episode_{packet.episode_index:06d}"
        folder.mkdir(parents=True, exist_ok=True)
        destination = folder / f"{packet.frame_index:06d}_{identifier}{'.npy' if depth else '.png'}"
        if depth:
            np.save(destination, image, allow_pickle=False)
        elif not cv2.imwrite(str(destination), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
            raise IOError("Cannot save RGB frame")
        path = destination.relative_to(run_dir).as_posix()
    return {"shape": list(image.shape), "dtype": str(image.dtype),
            "color_order": "native single channel" if depth else "RGB",
            "source": packet.source_references[name], "artifact_path": path,
            "units": "source depth units; not calibrated here" if depth else "uint8 RGB"}


def run(config_path):
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict) or not isinstance(config.get("source"), dict):
        raise ValueError("Configuration must contain a source mapping")
    for name in ("sample_every", "save_arrays_every"):
        if type(config.get(name)) is not int or config[name] < (1 if name == "sample_every" else 0):
            raise ValueError(f"Invalid {name}")
    if type(config.get("realtime")) is not bool:
        raise ValueError("realtime must be true or false")
    speed = config.get("playback_speed", 1.0)
    if isinstance(speed, bool) or not isinstance(speed, (float, int)) or not math.isfinite(speed) or speed <= 0:
        raise ValueError("playback_speed must be positive")
    sources = load_sources(config, config_path)
    required = config.get("required_fields", [])
    if not isinstance(required, list) or any(not isinstance(name, str) for name in required):
        raise ValueError("required_fields must be a list of original field names")
    reports = [source.describe() for source in sources]
    for report in reports:
        absent = set(required) - report["fields"].keys()
        if absent:
            raise ValueError(f"Required source fields absent: {sorted(absent)}")
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    run_dir = (config_path.parent / config["output_dir"]).resolve() / now.strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(dump(config), encoding="utf-8")
    (run_dir / "schema.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    provenance = []
    for source in sources:
        manifest_path = (source.root if isinstance(source, LeRobotEpisode) else source.path.parents[2]) / "download_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            provenance.append({"episode_index": source.episode_index, "manifest_path": str(manifest_path),
                               "repo_id": manifest["repo_id"], "revision": manifest["revision"]})
    (run_dir / "provenance.json").write_text(dump(provenance), encoding="utf-8")
    counts = []
    started = time.perf_counter()
    try:
        with ExitStack() as stack:
            events = stack.enter_context((run_dir / "inputs.jsonl").open("w", encoding="utf-8"))
            targets = stack.enter_context((run_dir / "recorded_targets.jsonl").open("w", encoding="utf-8"))
            extras = stack.enter_context((run_dir / "extra_records.jsonl").open("w", encoding="utf-8"))
            for source in sources:
                emitted = 0
                seen = 0
                episode_start = time.perf_counter()
                initial_timestamp = None
                with closing(source.iter_frames()) as packets:
                    for packet in packets:
                        if initial_timestamp is None:
                            initial_timestamp = packet.timestamp
                        if config["realtime"]:
                            deadline = episode_start + (packet.timestamp - initial_timestamp) / speed
                            while True:
                                remaining = deadline-time.perf_counter()
                                if remaining <= 0:
                                    break
                                time.sleep(min(0.1, remaining))
                        seen += 1
                        if packet.frame_index % config["sample_every"]:
                            continue
                        frame_start = time.perf_counter()
                        save = config["save_arrays_every"] > 0 and packet.frame_index % config["save_arrays_every"] == 0
                        event = packet.online_input()
                        event["images"] = {name: describe_image(run_dir, packet, name, image, False, save)
                                           for name, image in packet.images.items()}
                        event["depth"] = {name: describe_image(run_dir, packet, name, image, True, save)
                                          for name, image in packet.depth.items()}
                        event["recording_ms"] = (time.perf_counter()-frame_start)*1000
                        events.write(dump(event) + "\n")
                        identity = {"episode_index": packet.episode_index, "frame_index": packet.frame_index,
                                    "timestamp": packet.timestamp}
                        targets.write(dump({**identity, "recorded_targets": packet.recorded_targets,
                                            "online_availability": "not established; excluded from online_input"}) + "\n")
                        extras.write(dump({**identity, "extra_records": packet.extra_records}) + "\n")
                        emitted += 1
                counts.append({"episode_index": source.episode_index, "decoded_frames": seen,
                               "recorded_frames": emitted, "expected_frames": source.length})
        # Ground truth is exported after replay and never passed to online_input.
        offline_dir = run_dir / "offline_annotations"
        offline_dir.mkdir()
        for source in sources:
            annotation = source.offline_annotations()
            if annotation is not None:
                (offline_dir / f"episode_{source.episode_index:06d}.json").write_text(dump(annotation), encoding="utf-8")
            with (offline_dir / f"flags_{source.episode_index:06d}.jsonl").open("w", encoding="utf-8") as file:
                for row in source.iter_offline_flags():
                    file.write(dump(row) + "\n")
        summary = {"status": "input_replay_complete", "started_at": now.isoformat(), "episodes": counts,
                   "model_calls": 0, "baseline_constructed": False, "wall_time_s": time.perf_counter()-started}
        (run_dir / "summary.json").write_text(dump(summary), encoding="utf-8")
    except Exception as error:
        (run_dir / "summary.json").write_text(dump({"status": "input_replay_failed", "error": str(error),
                                                   "model_calls": 0, "baseline_constructed": False}), encoding="utf-8")
        raise
    print("Input replay complete:", run_dir)
    print("Episodes:", counts, "Model calls: 0. Baseline construction: disabled.")
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "configs/vifailback.yaml")
    args = parser.parse_args()
    run(args.config)
