"""Interface tests: synchronization, preserved values, missing fields and leakage."""
import json
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import MagicMock, patch

import av
import cv2
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from dataset_inputs import HDF5Episode, LeRobotEpisode, OFFLINE_FIELD
from stream_dataset import run


def make_hdf(path, timestamps=(0.0, 0.05, 0.12)):
    rgb = np.zeros((3, 8, 12, 3), np.uint8)
    rgb[..., 0] = 200
    depth = np.arange(3*8*12, dtype=np.uint16).reshape(3, 8, 12) + 1000
    with h5py.File(path, "w") as file:
        file["observations/qpos"] = np.arange(42, dtype=np.float32).reshape(3, 14)
        file["observations/qvel"] = np.ones((3, 14), np.float32)
        file["observations/effort"] = np.zeros((3, 14), np.float32)
        file["action"] = np.full((3, 14), 999, np.float32)
        file["action_eef"] = np.ones((3, 16), np.float32)
        file["base_action"] = np.ones((3, 2), np.float32)
        file["observations/images/cam_high"] = rgb
        file["observations/images_depth/cam_high"] = depth
        if timestamps is not None:
            file["timestamp"] = np.asarray(timestamps)
    return rgb, depth


def make_lerobot(root, timestamps=(0.0, 0.1, 0.2), video_frames=3):
    (root / "meta").mkdir()
    (root / "data").mkdir()
    (root / "videos").mkdir()
    features = {
        "observation.state": {"dtype": "float32", "shape": [14]},
        "action": {"dtype": "float32", "shape": [14]},
        OFFLINE_FIELD: {"dtype": "bool", "shape": [3]},
        "observation.images.cam_high": {"dtype": "video", "shape": [3, 8, 12]},
        **{name: {"dtype": "float32" if name == "timestamp" else "int64", "shape": [1]}
           for name in ("timestamp", "frame_index", "episode_index", "task_index", "index")},
    }
    info = {"codebase_version": "v2.1", "fps": 10, "chunks_size": 1000,
            "data_path": "data/{episode_index}.parquet", "video_path": "videos/{episode_index}.mp4",
            "features": features}
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "Move the block"})+'\n')
    (root / "meta/episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": 3})+'\n')
    data = {"observation.state": pa.array([list(range(14))]*3, type=pa.list_(pa.float32(), 14)),
            "action": pa.array([[999.0]*14]*3, type=pa.list_(pa.float32(), 14)),
            OFFLINE_FIELD: pa.array([[True, False, False]]*3, type=pa.list_(pa.bool_(), 3)),
            "timestamp": pa.array(timestamps, type=pa.float32()), "frame_index": [0, 1, 2],
            "episode_index": [0]*3, "task_index": [0]*3, "index": [100, 101, 102]}
    pq.write_table(pa.table(data), root / "data/0.parquet")
    with av.open(str(root / "videos/0.mp4"), "w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 12, 8, "yuv420p"
        for index in range(video_frames):
            image = np.full((8, 12, 3), index*50, np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class InputTests(unittest.TestCase):
    def test_hdf_values_timestamps_depth_and_targets(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "episode.hdf5"
            rgb, depth = make_hdf(path)
            frames = list(HDF5Episode(path, "Move the block").iter_frames())
            self.assertEqual([frame.timestamp for frame in frames], [0.0, 0.05, 0.12])
            np.testing.assert_array_equal(frames[1].images["observations/images/cam_high"], rgb[1])
            np.testing.assert_array_equal(frames[1].depth["observations/images_depth/cam_high"], depth[1])
            self.assertEqual(frames[1].depth["observations/images_depth/cam_high"].dtype, np.uint16)
            self.assertNotIn("action", frames[1].online_input())
            self.assertEqual(frames[1].recorded_targets["action"][0], 999)
            self.assertEqual(frames[1].gripper_state()["left"], 20)

    def test_raw_without_timestamps_requires_explicit_fps(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "episode.hdf5"
            make_hdf(path, None)
            with self.assertRaisesRegex(ValueError, "configure a verified FPS"):
                HDF5Episode(path, "task")
            self.assertEqual([frame.timestamp for frame in HDF5Episode(path, "task", fps=25).iter_frames()], [0, 0.04, 0.08])

    def test_rejects_nonmonotonic_time_and_length_mismatch(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "episode.hdf5"
            make_hdf(path, (0, 0.05, 0.04))
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                list(HDF5Episode(path, "task").iter_frames())
            with h5py.File(path, "a") as file:
                del file["observations/qvel"]
                file["observations/qvel"] = np.zeros((2, 14), np.float32)
            with self.assertRaisesRegex(ValueError, "length mismatch"):
                HDF5Episode(path, "task")

    def test_compressed_rgb_and_depth(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "episode.hdf5"
            rgb, depth = make_hdf(path)
            with h5py.File(path, "a") as file:
                for name, values in (("observations/images/cam_high", rgb), ("observations/images_depth/cam_high", depth)):
                    del file[name]
                    encoded = [cv2.imencode(".png", image)[1].tobytes() for image in values]
                    file[name] = np.asarray(encoded, dtype=f"S{max(map(len, encoded))}")
            frames = list(HDF5Episode(path, "task").iter_frames())
            np.testing.assert_array_equal(frames[2].images["observations/images/cam_high"], rgb[2])
            np.testing.assert_array_equal(frames[2].depth["observations/images_depth/cam_high"], depth[2])

    def test_lerobot_sync_and_offline_flag_isolation(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            make_lerobot(root)
            source = LeRobotEpisode(root, 0)
            frames = list(source.iter_frames())
            self.assertEqual(len(frames), 3)
            self.assertEqual(frames[0].observations["observation.state"].dtype, np.float32)
            self.assertEqual(frames[0].source_indices["index"], 100)
            self.assertNotIn(OFFLINE_FIELD, frames[0].observations)
            self.assertNotIn(OFFLINE_FIELD, frames[0].extra_records)
            self.assertNotIn("recorded_targets", frames[0].online_input())
            self.assertEqual(next(source.iter_offline_flags())[OFFLINE_FIELD], [True, False, False])

    def test_rejects_short_video_and_wrong_timestamp(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            make_lerobot(root, video_frames=2)
            with self.assertRaisesRegex(ValueError, "Video ended before"):
                list(LeRobotEpisode(root, 0).iter_frames())

    def test_rejects_camera_lookahead_inside_sync_tolerance(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            make_lerobot(root)
            frame = MagicMock()
            frame.pts = 2
            frame.time_base = Fraction(1, 100)
            container = MagicMock()
            container.__enter__.return_value = container
            container.decode.return_value = iter([frame])
            with patch("dataset_inputs.av.open", return_value=container):
                with self.assertRaisesRegex(ValueError, "in the future"):
                    list(LeRobotEpisode(root, 0).iter_frames())
        with TemporaryDirectory() as folder:
            root = Path(folder)
            make_lerobot(root, timestamps=(0, 0.12, 0.05))
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                list(LeRobotEpisode(root, 0).iter_frames())

    def test_recording_roundtrip_and_realtime(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "episode.hdf5"
            rgb, depth = make_hdf(path, (0, 0.01, 0.02))
            config = {"source": {"format": "vifailback_hdf5", "path": str(path), "task_instruction": "task"},
                      "output_dir": str(root / "runs"), "sample_every": 1, "save_arrays_every": 1,
                      "realtime": True, "playback_speed": 1}
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            started = time.perf_counter()
            run_dir = run(config_path)
            self.assertGreaterEqual(time.perf_counter()-started, 0.02)
            events = [json.loads(line) for line in (run_dir / "inputs.jsonl").read_text(encoding="utf-8").splitlines()]
            image_path = run_dir / events[2]["images"]["observations/images/cam_high"]["artifact_path"]
            saved_rgb = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
            np.testing.assert_array_equal(saved_rgb, rgb[2])
            depth_path = run_dir / events[2]["depth"]["observations/images_depth/cam_high"]["artifact_path"]
            np.testing.assert_array_equal(np.load(depth_path), depth[2])
            self.assertEqual(json.loads((run_dir/"summary.json").read_text())["model_calls"], 0)
            self.assertNotIn("failure", events[0])


if __name__ == "__main__":
    unittest.main()
