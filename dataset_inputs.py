"""Causal, input-only adapters for the official ViFailback trajectory formats.

Original field names are preserved. Targets and offline labels are separate from
online observations. Images are HWC RGB; depth retains its native dtype.
"""
from contextlib import ExitStack, closing
from dataclasses import dataclass, field
import json
import math
from pathlib import Path

import av
import cv2
import h5py
import numpy as np
import pyarrow.parquet as pq


TARGET_FIELDS = {"action", "action_eef", "action_leader", "base_action",
                 "complementary_info.action_eef", "complementary_info.base_action"}
OFFLINE_FIELD = "complementary_info.vifailback_flags"
INDEX_FIELDS = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Dataset path escapes its root: {relative}")
    return path


def finite(value, name, dtype=None):
    array = np.asarray(value, dtype=dtype)
    if array.dtype.kind in "fci" and not np.isfinite(array).all():
        raise ValueError(f"Non-finite values in {name}")
    return array


def field_role(name):
    if name == OFFLINE_FIELD or name.startswith(("annotation", "labels", "failure")):
        return "offline_annotation"
    if name in INDEX_FIELDS:
        return "index"
    if name.startswith(("observations/images_depth/", "observation.images_depth.")):
        return "depth"
    if name.startswith(("observations/images/", "observation.images.")):
        return "rgb"
    if name.startswith(("observations/", "observation.")):
        return "observation"
    if name in TARGET_FIELDS:
        return "recorded_target"
    return "extra_record"


def attribute_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return {"encoding": "hex", "data": value.hex()}
    return value


@dataclass
class InputFrame:
    episode_index: int
    frame_index: int
    timestamp: float
    timestamp_source: str
    task_instruction: str
    task_index: int | None
    observations: dict
    images: dict
    depth: dict
    recorded_targets: dict
    extra_records: dict
    source_references: dict
    source_indices: dict = field(default_factory=dict)

    def online_input(self):
        """Use this mapping for future monitoring; excludes targets and labels."""
        return {"episode_index": self.episode_index, "frame_index": self.frame_index,
                "timestamp": self.timestamp, "timestamp_source": self.timestamp_source,
                "task_instruction": self.task_instruction, "task_index": self.task_index,
                "source_indices": self.source_indices,
                "observations": self.observations, "images": self.images, "depth": self.depth,
                "gripper_state": self.gripper_state()}

    def gripper_state(self):
        for name in ("observation.state", "observations/qpos"):
            if name in self.observations and np.asarray(self.observations[name]).shape == (14,):
                return {"left": float(self.observations[name][6]),
                        "right": float(self.observations[name][13]), "source_field": name,
                        "source_indices": [6, 13], "units": "source units; not normalized"}
        return None


class LeRobotEpisode:
    def __init__(self, root, episode_index, cameras=None):
        self.root = Path(root).resolve()
        self.episode_index = episode_index
        self.info = json.loads((self.root / "meta/info.json").read_text(encoding="utf-8-sig"))
        if self.info.get("codebase_version") != "v2.1":
            raise ValueError("Only the official LeRobot v2.1 layout is supported")
        self.fps = float(self.info["fps"])
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Invalid dataset FPS")
        self.tasks = {item["task_index"]: item["task"] for item in jsonl(self.root / "meta/tasks.jsonl")}
        self.episode = next((item for item in jsonl(self.root / "meta/episodes.jsonl")
                             if item["episode_index"] == episode_index), None)
        if self.episode is None:
            raise ValueError(f"Unknown episode: {episode_index}")
        values = {"episode_index": episode_index,
                  "episode_chunk": episode_index // self.info["chunks_size"]}
        self.path = safe_path(self.root, self.info["data_path"].format(**values))
        with closing(pq.ParquetFile(self.path)) as self.parquet:
            self.length = self.parquet.metadata.num_rows
            self.columns = set(self.parquet.schema_arrow.names)
        if self.length <= 0 or self.length != self.episode["length"]:
            raise ValueError("Parquet length disagrees with episode metadata")
        self.features = self.info["features"]
        required = {"episode_index", "frame_index", "timestamp", "task_index"}
        if not required <= self.columns:
            raise ValueError(f"Missing required columns: {sorted(required-self.columns)}")
        available = [name for name, feature in self.features.items() if feature["dtype"] == "video"]
        self.cameras = available if cameras is None else cameras
        if not isinstance(self.cameras, list) or any(name not in available for name in self.cameras):
            raise ValueError("Requested camera is absent from meta/info.json")
        self.video_paths = {name: safe_path(self.root, self.info["video_path"].format(video_key=name, **values))
                            for name in self.cameras}
        if any(not path.is_file() for path in self.video_paths.values()):
            raise FileNotFoundError("A selected camera video has not been downloaded")
        absent = {name for name, feature in self.features.items() if feature["dtype"] != "video"} - self.columns
        if absent:
            raise ValueError(f"Metadata features absent from Parquet: {sorted(absent)}")

    def describe(self):
        return {"format": "lerobot_v2.1", "episode_index": self.episode_index, "frames": self.length,
                "fps": self.fps, "timestamp_source": "Parquet timestamp (conversion timebase)",
                "fields": {name: {**feature, "role": field_role(name)} for name, feature in self.features.items()},
                "selected_cameras": self.cameras, "depth_available": False,
                "observed_eef_pose_available": False, "wrench_6d_available": False,
                "action_leader_available": False, "target_availability_at_t": "not established",
                "depth_rgb_alignment": "not applicable; depth omitted by conversion"}

    def _rows(self, columns=None):
        with closing(pq.ParquetFile(self.path)) as parquet:
            for batch in parquet.iter_batches(batch_size=64, columns=columns):
                yield from batch.to_pylist()

    def iter_frames(self):
        with ExitStack() as stack:
            streams = {}
            for name, path in self.video_paths.items():
                container = stack.enter_context(av.open(str(path)))
                streams[name] = iter(container.decode(video=0))
            previous_timestamp = -math.inf
            first_global_index = None
            for expected_index, row in enumerate(self._rows()):
                if row["frame_index"] != expected_index or row["episode_index"] != self.episode_index:
                    raise ValueError("Rows must be contiguous and ordered within one episode")
                timestamp = float(row["timestamp"])
                if not math.isfinite(timestamp) or timestamp <= previous_timestamp or timestamp < 0:
                    raise ValueError("Timestamps must be finite, nonnegative and strictly increasing")
                if abs(timestamp - expected_index / self.fps) > 0.5 / self.fps + 1e-5:
                    raise ValueError("Parquet timestamp disagrees with the video frame grid")
                previous_timestamp = timestamp
                if "index" in row:
                    if first_global_index is None:
                        first_global_index = row["index"]
                    if row["index"] != first_global_index + expected_index:
                        raise ValueError("Global frame index is not contiguous")
                task_index = row["task_index"]
                if task_index not in self.tasks:
                    raise ValueError(f"Unknown task_index: {task_index}")
                observations, recorded_targets, extra_records = {}, {}, {}
                for name, value in row.items():
                    feature = self.features.get(name)
                    dtype = feature["dtype"] if feature and feature["dtype"] in ("float32", "float64", "int64", "int32", "bool", "uint8", "uint16") else None
                    array = finite(value, name, dtype)
                    if feature and name not in INDEX_FIELDS and list(array.shape) != feature["shape"]:
                        raise ValueError(f"Shape mismatch for {name}: {array.shape}")
                    role = field_role(name)
                    if role == "observation":
                        observations[name] = array
                    elif role == "recorded_target":
                        recorded_targets[name] = array
                    elif role == "extra_record":
                        extra_records[name] = value
                images, references = {}, {}
                for name, stream in streams.items():
                    try:
                        frame = next(stream)
                    except StopIteration:
                        raise ValueError(f"Video ended before state data: {name}") from None
                    video_timestamp = float(frame.pts * frame.time_base) if frame.pts is not None else None
                    if video_timestamp is None or abs(video_timestamp-timestamp) > 0.5/self.fps + 1e-5:
                        raise ValueError(f"Video PTS disagrees with state timestamp: {name}")
                    if video_timestamp > timestamp + 1e-5:
                        raise ValueError(f"Camera frame is in the future relative to state timestamp: {name}")
                    image = frame.to_ndarray(format="rgb24")
                    if list(image.transpose(2, 0, 1).shape) != self.features[name]["shape"]:
                        raise ValueError(f"Video dimensions disagree with metadata: {name}")
                    images[name] = image
                    references[name] = {"path": str(self.video_paths[name]), "frame_index": expected_index,
                                        "video_timestamp": video_timestamp}
                yield InputFrame(self.episode_index, expected_index, timestamp, "Parquet timestamp",
                                 self.tasks[task_index], task_index, observations, images, {},
                                 recorded_targets, extra_records, references,
                                 {name: row[name] for name in INDEX_FIELDS if name in row})
            for name, stream in streams.items():
                if next(stream, None) is not None:
                    raise ValueError(f"Video contains extra frames after state data: {name}")

    def offline_annotations(self):
        path = self.root / "meta/vifailback_annotations.jsonl"
        if not path.is_file():
            return None
        return next((item for item in jsonl(path) if item["episode_index"] == self.episode_index), None)

    def iter_offline_flags(self):
        if OFFLINE_FIELD in self.columns:
            yield from self._rows(["episode_index", "frame_index", "timestamp", OFFLINE_FIELD])


def decode_hdf_image(value, depth=False, storage_order="RGB"):
    array = np.asarray(value)
    if array.ndim == 0 and array.dtype.kind in "SO":
        encoded = np.frombuffer(bytes(value), np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED if depth else cv2.IMREAD_COLOR)
    elif array.ndim == 1 and array.dtype == np.uint8:
        image = cv2.imdecode(array, cv2.IMREAD_UNCHANGED if depth else cv2.IMREAD_COLOR)
    else:
        image = array
    if image is None or image.size == 0:
        raise ValueError("Cannot decode HDF5 image bytes")
    if depth:
        if image.ndim != 2:
            raise ValueError("Depth must be a single-channel array")
    else:
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("RGB must decode to an HWC uint8 three-channel image")
        if storage_order == "BGR":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


class HDF5Episode:
    def __init__(self, path, task_instruction, episode_index=0, fps=None, timestamp_path=None,
                 cameras=None, include_depth=True, storage_order="RGB", annotation=None):
        self.path = Path(path).resolve()
        self.task_instruction = task_instruction
        if not isinstance(task_instruction, str) or not task_instruction.strip():
            raise ValueError("Raw HDF5 requires an explicit task instruction or a metadata mapping")
        if storage_order not in ("RGB", "BGR"):
            raise ValueError("storage_order must be RGB or BGR")
        self.storage_order = storage_order
        self.episode_index = episode_index
        self.include_depth = include_depth
        self.annotation = annotation
        self.fps = fps
        self.fields = {}
        with h5py.File(self.path, "r") as file:
            self.attributes = {name: attribute_value(value) for name, value in file.attrs.items()}
            def inspect(name, dataset):
                if isinstance(dataset, h5py.Dataset):
                    self.fields[name] = {"shape": list(dataset.shape), "dtype": str(dataset.dtype),
                                         "role": field_role(name),
                                         "attributes": {key: attribute_value(value) for key, value in dataset.attrs.items()}}
            file.visititems(inspect)
            if "observations/qpos" not in file:
                raise ValueError("Raw ViFailback file is missing observations/qpos")
            self.length = len(file["observations/qpos"])
            if self.length <= 0:
                raise ValueError("Empty HDF5 episode")
            available = [name for name in self.fields if field_role(name) == "rgb"]
            self.cameras = available if cameras is None else cameras
            if not isinstance(self.cameras, list) or any(name not in available for name in self.cameras):
                raise ValueError("Requested RGB field is absent from HDF5")
            self.depth_keys = [name for name in self.fields if field_role(name) == "depth"] if include_depth else []
            self.value_keys = [name for name in self.fields if field_role(name) in ("observation", "recorded_target")]
            self.extra_keys = [name for name in self.fields if field_role(name) == "extra_record"
                               and self.fields[name]["shape"] and self.fields[name]["shape"][0] == self.length]
            for name in self.cameras + self.depth_keys + self.value_keys:
                if len(file[name]) != self.length:
                    raise ValueError(f"HDF5 stream length mismatch: {name}")
            for name, dimension in (("observations/qpos", 14), ("observations/qvel", 14),
                                    ("observations/effort", 14), ("action", 14),
                                    ("action_eef", 16), ("base_action", 2)):
                if name in self.fields and self.fields[name]["shape"] != [self.length, dimension]:
                    raise ValueError(f"Unexpected vector dimensions: {name}")
            candidates = [name for name in ("timestamp", "timestamps", "observations/timestamp") if name in file]
            if timestamp_path is None and len(candidates) > 1:
                raise ValueError("Multiple timestamp fields; specify timestamp_path")
            self.timestamp_path = timestamp_path or (candidates[0] if candidates else None)
            if self.timestamp_path:
                if self.timestamp_path not in file or file[self.timestamp_path].shape not in ((self.length,), (self.length, 1)):
                    raise ValueError("Timestamp dataset must have one scalar per frame")
            else:
                if fps is None or not math.isfinite(float(fps)) or float(fps) <= 0:
                    raise ValueError("No raw timestamps; explicitly configure a verified FPS")
                self.fps = float(fps)
        if annotation is not None and (annotation["episode_index"] != episode_index or annotation["num_frames"] != self.length):
            raise ValueError("Raw annotation mapping disagrees with the selected episode")

    def describe(self):
        return {"format": "vifailback_hdf5", "episode_index": self.episode_index, "frames": self.length,
                "fps": self.fps, "timestamp_source": self.timestamp_path or "frame_index/configured_fps; not sensor timestamps",
                "fields": self.fields, "attributes": self.attributes, "selected_cameras": self.cameras,
                "depth_available": any(field_role(name) == "depth" for name in self.fields),
                "selected_depth_fields": self.depth_keys,
                "observed_eef_pose_available": False, "wrench_6d_available": False,
                "action_leader_available": "action_leader" in self.fields,
                "rgb_storage_order": self.storage_order, "depth_rgb_alignment": "not guaranteed; no fusion performed",
                "depth_units": "not established by this interface", "target_availability_at_t": "not established"}

    def iter_frames(self):
        with h5py.File(self.path, "r") as file:
            previous_timestamp = -math.inf
            for frame_index in range(self.length):
                timestamp = float(np.asarray(file[self.timestamp_path][frame_index]).item()) if self.timestamp_path else frame_index / self.fps
                if not math.isfinite(timestamp) or timestamp <= previous_timestamp or timestamp < 0:
                    raise ValueError("Timestamps must be finite, nonnegative and strictly increasing")
                previous_timestamp = timestamp
                observations, recorded_targets = {}, {}
                for name in self.value_keys:
                    value = finite(file[name][frame_index], name)
                    (observations if field_role(name) == "observation" else recorded_targets)[name] = value
                images = {name: decode_hdf_image(file[name][frame_index], storage_order=self.storage_order)
                          for name in self.cameras}
                depth = {name: decode_hdf_image(file[name][frame_index], depth=True) for name in self.depth_keys}
                extra_records = {name: file[name][frame_index] for name in self.extra_keys}
                references = {name: {"path": str(self.path), "dataset": name, "frame_index": frame_index}
                              for name in self.cameras + self.depth_keys}
                yield InputFrame(self.episode_index, frame_index, timestamp,
                                 self.timestamp_path or "frame_index/configured_fps", self.task_instruction,
                                 None, observations, images, depth, recorded_targets, extra_records, references,
                                 {"frame_index": frame_index, "episode_index": self.episode_index})

    def offline_annotations(self):
        return self.annotation

    def iter_offline_flags(self):
        return iter(())
