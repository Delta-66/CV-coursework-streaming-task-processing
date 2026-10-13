"""Validate downloaded real samples and their recorded artifacts, without a model."""
from contextlib import closing
import json
from pathlib import Path

import cv2
import numpy as np

from dataset_inputs import HDF5Episode, LeRobotEpisode, OFFLINE_FIELD


ROOT = Path(__file__).resolve().parent


def latest_run(kind):
    folder = ROOT / "output/input_runs" / kind
    directories = sorted(path for path in folder.iterdir() if path.is_dir())
    if not directories:
        raise FileNotFoundError(f"Run stream_dataset.py for {kind} first")
    return directories[-1]


def verify_recorded_run(run_dir, expected_counts):
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "input_replay_complete"
    assert summary["model_calls"] == 0 and summary["baseline_constructed"] is False
    counts = {}
    previous = {}
    saved_arrays = 0
    with (run_dir / "inputs.jsonl").open(encoding="utf-8") as file:
        for line in file:
            event = json.loads(line)
            episode = event["episode_index"]
            assert event["timestamp"] > previous.get(episode, -1)
            previous[episode] = event["timestamp"]
            assert event["frame_index"] == counts.get(episode, 0)
            assert "failure" not in event and "result" not in event and "recorded_targets" not in event
            assert OFFLINE_FIELD not in event["observations"]
            for record in list(event["images"].values()) + list(event["depth"].values()):
                if record["artifact_path"]:
                    path = run_dir / record["artifact_path"]
                    image = np.load(path, allow_pickle=False) if path.suffix == ".npy" else cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                    assert image is not None and list(image.shape) == record["shape"]
                    assert str(image.dtype) == record["dtype"]
                    saved_arrays += 1
            counts[episode] = counts.get(episode, 0) + 1
    assert counts == expected_counts
    with (run_dir / "recorded_targets.jsonl").open(encoding="utf-8") as file:
        assert sum(1 for _ in file) == sum(counts.values())
    return {"run_dir": str(run_dir), "frames": counts, "saved_arrays_read_back": saved_arrays}


def main():
    lerobot_root = ROOT / "data/vifailback_lerobot"
    converted = LeRobotEpisode(lerobot_root, 0)
    annotation = converted.offline_annotations()
    raw = HDF5Episode(ROOT / "data/vifailback_raw" / annotation["source_hdf5"],
                      annotation["annotation"]["task"], fps=25)
    mapping = {"observations/qpos": "observation.state", "observations/qvel": "observation.velocity",
               "observations/effort": "observation.effort", "action": "action",
               "action_eef": "complementary_info.action_eef", "base_action": "complementary_info.base_action"}
    minimum_psnr = float("inf")
    compared = 0
    first_raw = None
    with closing(raw.iter_frames()) as raw_frames, closing(converted.iter_frames()) as converted_frames:
        for current_raw, current_converted in zip(raw_frames, converted_frames, strict=True):
            if first_raw is None:
                first_raw = current_raw
            assert current_raw.frame_index == current_converted.frame_index
            assert abs(current_raw.timestamp-current_converted.timestamp) < 1e-5
            for raw_name, converted_name in mapping.items():
                raw_values = current_raw.observations if raw_name.startswith("observations/") else current_raw.recorded_targets
                converted_values = current_converted.observations if converted_name.startswith("observation.") else current_converted.recorded_targets
                np.testing.assert_array_equal(raw_values[raw_name], converted_values[converted_name])
            for name, image in current_raw.images.items():
                converted_image = current_converted.images[name.replace("observations/images/", "observation.images.")]
                psnr = cv2.PSNR(image, converted_image)
                assert psnr > 20, "Unexpected RGB channel order or severe conversion difference"
                minimum_psnr = min(minimum_psnr, psnr)
            assert all(image.dtype == np.uint16 and image.shape == (480, 640) for image in current_raw.depth.values())
            compared += 1
    assert compared == 300
    full_counts = {}
    for episode_index in (0, 18):
        source = LeRobotEpisode(lerobot_root, episode_index)
        full_counts[episode_index] = sum(1 for _ in source.iter_frames())
        annotation = source.offline_annotations()
        flags = list(source.iter_offline_flags())
        assert len(flags) == full_counts[episode_index]
        assert all(row[OFFLINE_FIELD][0] == (annotation["episode_success"] == "failure") for row in flags)
        assert [row["frame_index"] for row in flags if row[OFFLINE_FIELD][1]] == annotation["avoid_keyframe_indices"]
        assert [row["frame_index"] for row in flags if row[OFFLINE_FIELD][2]] == annotation["correct_keyframe_indices"]
    lerobot_run = verify_recorded_run(latest_run("lerobot"), {0: 300, 18: 519})
    raw_dir = latest_run("hdf5")
    raw_run = verify_recorded_run(raw_dir, {0: 300})
    first_event = json.loads((raw_dir / "inputs.jsonl").read_text(encoding="utf-8").splitlines()[0])
    for name, image in first_raw.depth.items():
        np.testing.assert_array_equal(np.load(raw_dir / first_event["depth"][name]["artifact_path"]), image)
    for name, image in first_raw.images.items():
        saved = cv2.imread(str(raw_dir / first_event["images"][name]["artifact_path"]))
        np.testing.assert_array_equal(cv2.cvtColor(saved, cv2.COLOR_BGR2RGB), image)
    report = {"status": "passed", "lerobot_frames_verified": full_counts,
              "hdf5_frames_verified": compared, "numeric_fields_compared_exactly": mapping,
              "minimum_raw_to_av1_rgb_psnr_db": minimum_psnr,
              "video_pts_state_timestamps_and_frame_indices": "passed",
              "rgb_and_uint16_depth_artifact_roundtrip": "passed",
              "offline_labels_and_recorded_targets_excluded_from_online_input": True,
              "runs": [lerobot_run, raw_run], "model_calls": 0, "baseline_constructed": False}
    output = ROOT / "output/input_validation.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("PASS: real multi-camera inputs, temporal alignment, original numeric values, depth and separated annotations")
    print("Validation report:", output)


if __name__ == "__main__":
    main()
