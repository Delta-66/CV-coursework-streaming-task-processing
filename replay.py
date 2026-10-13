"""Replay a constant-frame-rate video causally and record a bounded history.

The placeholder backend does not perform task understanding or failure detection.
"""
import argparse
from collections import deque
from datetime import datetime
import json
import math
from pathlib import Path
from time import perf_counter

import cv2
import yaml


def load_config(config_path):
    with config_path.open(encoding="utf-8-sig") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    for name, minimum in (("sample_every", 1), ("history_size", 0), ("max_image_width", 1)):
        value = config.get(name)
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not isinstance(config.get("task_instruction"), str) or not config["task_instruction"].strip():
        raise ValueError("task_instruction must be a nonempty string")
    if type(config.get("save_frames")) is not bool:
        raise ValueError("save_frames must be true or false")
    for name in ("video_path", "output_dir"):
        value = config.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a nonempty path string")
        config[name] = (config_path.parent / value).resolve()
    return config


def diagnose(task_instruction, current, history):
    """Replace this function when a real model is selected.

    current contains the current sampled frame; history contains only previous
    sampled frames, with frame_index, timestamp_s and image in each entry.
    """
    return {
        "backend": "placeholder",
        "stage": "unknown",
        "failure": None,
        "evidence": [],
        "high_level_advice": None,
        "low_level_advice": None,
        "alert": None,
        "note": "模型尚未接入；以上字段不代表检测结果。",
    }


def replay(config_path):
    config = load_config(config_path)
    if not config["video_path"].is_file():
        raise FileNotFoundError(f"Video not found: {config['video_path']}. Run create_demo.py first.")
    reader = cv2.VideoCapture(str(config["video_path"]))
    try:
        if not reader.isOpened():
            raise RuntimeError("Cannot open video")
        fps = reader.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video has no valid FPS; use a source with explicit timestamps")
        run_dir = config["output_dir"] / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir.mkdir(parents=True, exist_ok=False)
        if config["save_frames"]:
            (run_dir / "frames").mkdir()
        saved_config = {name: str(value) if isinstance(value, Path) else value
                        for name, value in config.items()}
        with (run_dir / "config.json").open("w", encoding="utf-8") as file:
            json.dump(saved_config, file, ensure_ascii=False, indent=2)
        history = deque(maxlen=config["history_size"])
        frame_index = 0
        observation_count = 0
        peak_history_bytes = 0
        start = perf_counter()
        with (run_dir / "events.jsonl").open("w", encoding="utf-8") as file:
            while True:
                success, frame = reader.read()
                if not success:
                    break
                if frame_index % config["sample_every"] == 0:
                    event_start = perf_counter()
                    if frame.shape[1] > config["max_image_width"]:
                        width = config["max_image_width"]
                        height = max(1, round(frame.shape[0] * width / frame.shape[1]))
                        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                    current = {"frame_index": frame_index, "timestamp_s": frame_index / fps, "image": frame}
                    previous = tuple(history)
                    result = diagnose(config["task_instruction"], current, previous)
                    frame_path = None
                    if config["save_frames"]:
                        frame_path = f"frames/{frame_index:06d}.png"
                        if not cv2.imwrite(str(run_dir / frame_path), frame):
                            raise RuntimeError("Cannot save frame image")
                    history.append(current)
                    history_bytes = sum(item["image"].nbytes for item in history)
                    peak_history_bytes = max(peak_history_bytes, history_bytes)
                    event = {
                        "frame_index": frame_index,
                        "timestamp_s": current["timestamp_s"],
                        "timestamp_source": "frame_index/fps (constant frame rate)",
                        "task_instruction": config["task_instruction"],
                        "history_frame_indices": [item["frame_index"] for item in previous],
                        "history_timestamp_s": [item["timestamp_s"] for item in previous],
                        "history_size_limit": config["history_size"],
                        "retained_image_bytes": history_bytes,
                        "frame_path": frame_path,
                        "result": result,
                        "processing_ms": (perf_counter() - event_start) * 1000,
                    }
                    file.write(json.dumps(event, ensure_ascii=False) + "\n")
                    observation_count += 1
                frame_index += 1
        if frame_index == 0:
            raise RuntimeError("Video contains no decodable frames")
        summary = {
            "status": "replay_only_model_not_connected",
            "decoded_frames": frame_index,
            "sampled_observations": observation_count,
            "fps": fps,
            "history_size_limit": config["history_size"],
            "peak_retained_image_bytes": peak_history_bytes,
            "model_calls": 0,
            "wall_time_s": perf_counter() - start,
        }
        with (run_dir / "summary.json").open("w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)
        print(f"Replayed {frame_index} frames; logged {observation_count} observations.")
        print("Results:", run_dir)
        print("Placeholder backend: no model inference or failure detection.")
        return run_dir
    finally:
        reader.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "configs" / "demo.yaml")
    args = parser.parse_args()
    replay(args.config.resolve())
