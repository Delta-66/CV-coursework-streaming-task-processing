"""Check trajectory grouping, holdout isolation and deterministic split construction."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from prepare_dataset_splits import (BENCHMARK_FILE, BENCHMARK_SHA256, RAW_REPO, build_splits,
                                    check_existing_protocol, load_metadata, map_benchmark, prepare_splits,
                                    source_path_for_reference, stratified_partition)


def fixture_rows(count=40):
    rows = {}
    for index in range(count):
        task_index = index // 20
        instruction = f"Task instruction {task_index}"
        rows[index] = {"episode_index": index, "source_hdf5": f"raw_data/task_{task_index}/episode_{index}.hdf5",
                       "source_task": f"task_{task_index}", "source_episode": f"episode_{index}",
                       "task_index": task_index, "task_instruction": instruction,
                       "logical_task_sha256": hashlib.sha256(instruction.encode()).hexdigest(),
                       "num_frames": 100, "outcome": "failure" if index % 2 else "success",
                       "failure_type": "gripper state" if index % 2 else "none"}
    return rows


def reference(rows, index, camera="cam_high"):
    return f"annotated_data/{rows[index]['source_task']}/frames/episode_{index}_{camera}/25.jpg"


class AnswerAccessForbidden(dict):
    def __getitem__(self, key):
        if key in ("messages", "answers"):
            raise AssertionError("The split tool must not inspect benchmark answers")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in ("messages", "answers"):
            raise AssertionError("The split tool must not inspect benchmark answers")
        return super().get(key, default)


class DatasetSplitTests(unittest.TestCase):
    def test_all_cameras_frames_and_formats_map_to_one_source_trajectory(self):
        expected = "raw_data/task_a/episode_3.hdf5"
        for path in (expected, "annotated_data/task_a/frames/episode_3_cam_high/0.jpg",
                     "annotated_data/task_a/frames/episode_3_cam_left_wrist/25.jpg",
                     "annotated_data/task_a/videos/episode_3_cam_right_wrist.mp4"):
            self.assertEqual(source_path_for_reference(path), expected)
        for path in ("../raw_data/task_a/episode_3.hdf5", "/raw_data/task_a/episode_3.hdf5",
                     "C:/data/episode_3.hdf5", "arbitrary.jpg", "annotated_data/task_a/../frames/episode_3_cam_high/0.jpg"):
            with self.assertRaises(ValueError):
                source_path_for_reference(path)

    def test_benchmark_union_reserves_extra_references_without_reading_answers(self):
        rows = fixture_rows()
        records = [AnswerAccessForbidden(tag="task_success_detection", images=[reference(rows, 2)], messages=object()),
                   AnswerAccessForbidden(tag="failure_type", images=[reference(rows, 2, "cam_left_wrist")], messages=object()),
                   AnswerAccessForbidden(tag="high_level_reason", images=[reference(rows, 4)], messages=object())]
        benchmark = map_benchmark(records, rows)
        self.assertEqual(benchmark["episode_indices"], {2, 4})
        self.assertEqual(benchmark["canonical_episode_indices"], {2})
        self.assertEqual(benchmark["extra_episode_indices"], {4})
        self.assertEqual(benchmark["record_indices"], {2: [0, 1], 4: [2]})

    def test_unknown_and_ambiguous_benchmark_references_fail_closed(self):
        rows = fixture_rows()
        for paths in (["annotated_data/unknown/frames/episode_2_cam_high/0.jpg"],
                      [reference(rows, 2), reference(rows, 3)], []):
            with self.subTest(paths=paths), self.assertRaises(ValueError):
                map_benchmark([{"tag": "task_success_detection", "images": paths}], rows)
        with self.assertRaisesRegex(ValueError, "detection cohort"):
            map_benchmark([{"tag": "other", "images": [reference(rows, 2)]}], rows)

    def test_disjoint_complete_and_seen_samples_never_enter_clean_holdouts(self):
        rows = fixture_rows()
        benchmark = {"episode_indices": {0, 2, 4, 6}}
        exposures = {0: "prior_model_development", 1: "annotation_review", 18: "prior_model_development"}
        splits, strata = build_splits(rows, benchmark, exposures)
        self.assertEqual(splits["exposed_benchmark"], {0})
        self.assertEqual(splits["dev_seen"], {1, 18})
        self.assertEqual(splits["test"], {2, 4, 6})
        self.assertEqual(set().union(*splits.values()), set(rows))
        self.assertEqual(sum(map(len, splits.values())), len(rows))
        self.assertFalse(set(exposures) & (splits["validation"] | splits["test"]))
        self.assertTrue(all(item["total"] == item["train"] + item["validation"] for item in strata))
        with self.assertRaisesRegex(ValueError, "unknown"):
            build_splits(rows, benchmark, {999: "reviewed"})

    def test_seed_reproducible_independent_of_input_order_and_changes_selection(self):
        rows = fixture_rows(100)
        indices = set(rows)
        first = stratified_partition(rows, indices, 42, 0.2)
        reversed_rows = dict(reversed(list(rows.items())))
        self.assertEqual(first, stratified_partition(reversed_rows, indices, 42, 0.2))
        self.assertEqual(len(first[1]), 20)
        self.assertNotEqual(first[1], stratified_partition(rows, indices, 43, 0.2)[1])
        self.assertFalse(first[0] & first[1])
        self.assertEqual(first[0] | first[1], indices)

    def test_small_strata_keep_train_members_and_report_actual_quota(self):
        rows = fixture_rows(3)
        rows[0]["task_instruction"] = "Unique singleton instruction"
        rows[1]["outcome"] = rows[2]["outcome"] = "success"
        rows[1]["failure_type"] = rows[2]["failure_type"] = "none"
        train, validation, strata = stratified_partition(rows, set(rows), 42, 0.2)
        self.assertIn(0, train)
        self.assertEqual((len(train), len(validation)), (2, 1))
        self.assertTrue(all(item["train"] >= 1 for item in strata))
        for value in (0, 1, -1, float("nan"), True):
            with self.assertRaises(ValueError):
                stratified_partition(rows, set(rows), 42, value)

    def test_logical_task_uses_complete_instruction_and_metadata_consistency(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "meta").mkdir()
            instruction = "Place the cube into the drawer."
            annotations = [{"episode_index": index, "source_hdf5": f"raw_data/collection_ep{index + 1}/episode_0.hdf5",
                            "source_task": f"collection_ep{index + 1}", "source_episode": "episode_0",
                            "source_repo_id": RAW_REPO, "num_frames": 10, "episode_success": "success",
                            "annotation": {"task": instruction}} for index in range(2)]
            (root / "meta/info.json").write_text(json.dumps({"total_episodes": 2, "total_tasks": 1, "total_frames": 20}))
            (root / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 7, "task": instruction}) + "\n")
            (root / "meta/episodes.jsonl").write_text("\n".join(json.dumps({"episode_index": index, "tasks": [instruction], "length": 10}) for index in range(2)))
            annotation_path = root / "meta/vifailback_annotations.jsonl"
            annotation_path.write_text("\n".join(map(json.dumps, annotations)))
            rows, tasks = load_metadata(root)
            self.assertEqual({row["task_index"] for row in rows.values()}, {7})
            self.assertEqual(len(tasks), 1)
            annotations[1]["episode_success"] = "unlabeled"
            annotation_path.write_text("\n".join(map(json.dumps, annotations)))
            with self.assertRaisesRegex(ValueError, "Unknown episode outcome"):
                load_metadata(root)
            annotations[1]["episode_success"] = "success"
            annotations[1]["annotation"]["task"] = "A different instruction."
            annotation_path.write_text("\n".join(map(json.dumps, annotations)))
            with self.assertRaisesRegex(ValueError, "complete instruction"):
                load_metadata(root)

    def test_frozen_protocol_refuses_changed_seed_or_tampered_artifacts(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            artifact = root / "train.jsonl"
            artifact.write_text('{"episode_index":1}\n')
            sources = {"test_fixture": "sha256"}
            ledger = [{"episode_index": 0, "reason": "seen", "assigned_split": "dev_seen"}]
            existing = {"sources": sources, "seed": 42, "validation_fraction_requested": 0.2,
                        "exposure_ledger": ledger, "artifacts": {"train.jsonl": {"sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}}}
            (root / "protocol.json").write_text(json.dumps(existing))
            check_existing_protocol(root, sources, 42, 0.2, ledger)
            with self.assertRaisesRegex(FileExistsError, "Frozen split protocol differs"):
                check_existing_protocol(root, sources, 43, 0.2, ledger)
            artifact.write_text('{"episode_index":2}\n')
            with self.assertRaisesRegex(FileExistsError, "modified or are incomplete"):
                check_existing_protocol(root, sources, 42, 0.2, ledger)

    def test_algorithm_membership_change_is_rejected_before_any_file_write(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            rows = fixture_rows()
            tasks = {row["task_instruction"]: row["task_index"] for row in rows.values()}
            protocol_root = root / "source"
            benchmark_path = protocol_root / BENCHMARK_FILE
            benchmark_path.parent.mkdir(parents=True)
            benchmark_path.write_text(json.dumps([
                {"tag": "task_success_detection", "images": [reference(rows, 38)]},
                {"tag": "task_success_detection", "images": [reference(rows, 39)]}]))
            manifest = {"files": {BENCHMARK_FILE: {"sha256": BENCHMARK_SHA256}}}
            output_dir = root / "splits"
            summary_path = root / "summary.json"
            with patch("prepare_dataset_splits.verify_manifest", return_value=manifest), \
                 patch("prepare_dataset_splits.load_metadata", return_value=(rows, tasks)), patch("builtins.print"):
                prepare_splits(root, protocol_root, output_dir, summary_path, seed=42, exposures={})
                before = {path: path.read_bytes() for path in [*output_dir.iterdir(), summary_path]}

                def changed_algorithm(*args, **kwargs):
                    splits, strata = build_splits(*args, **kwargs)
                    validation_index = min(splits["validation"])
                    stratum = tuple(rows[validation_index][key] for key in ("task_index", "outcome", "failure_type"))
                    train_index = next(index for index in sorted(splits["train"])
                                       if tuple(rows[index][key] for key in ("task_index", "outcome", "failure_type")) == stratum)
                    splits["validation"].remove(validation_index)
                    splits["validation"].add(train_index)
                    splits["train"].remove(train_index)
                    splits["train"].add(validation_index)
                    return splits, strata

                with patch("prepare_dataset_splits.build_splits", side_effect=changed_algorithm), \
                     self.assertRaisesRegex(FileExistsError, "Recomputed split artifacts differ"):
                    prepare_splits(root, protocol_root, output_dir, summary_path, seed=42, exposures={})
                self.assertEqual(before, {path: path.read_bytes() for path in before})
                # A different code/provenance hash alone does not lock out an
                # implementation repair when the computed membership is equal.
                protocol_path = output_dir / "protocol.json"
                protocol = json.loads(protocol_path.read_text())
                protocol["script_sha256"] = "previous-implementation-hash"
                protocol_path.write_text(json.dumps(protocol))
                prepare_splits(root, protocol_root, output_dir, summary_path, seed=42, exposures={})
                for path, content in before.items():
                    if path != protocol_path and path != summary_path:
                        self.assertEqual(content, path.read_bytes())


if __name__ == "__main__":
    unittest.main()
