"""离线绑定故障注入，文件内容为测试替身；不作为真实视觉成绩。"""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from streaming_recovery.review_binding import (FINGERPRINT_DEFINITION, FINGERPRINT_VERSION, file_sha256, load_run_rows,
    migrate_review, rebind_review, record_binding_errors, row_fingerprint, validate_review_bindings)


class ReviewBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root/"run"
        self.row = {"timestamp": 1., "observation_ref": "obs-000025", "state": {"stage": "locate_tool"},
                    "processing": {"decision_wall_s": .1}, "visual_facts": {"evidence_refs": []},
                    "modalities": {"cam_high": {"status": "available", "timestamp": 1.},
                                   "cam_right_wrist": {"status": "available", "timestamp": 1.}}}
        self.paths = [f"trajectory_000002/frames/obs-000025_{camera}.jpg" for camera in ("cam_high", "cam_right_wrist")]
        self.write_run(self.run)
        ordinary = {"episode_for_audit_only": 2, "timestamp": 1., "expected_stages": ["locate_tool"],
                    "image_refs": self.paths, "image_sha256": {path: file_sha256(self.run/path) for path in self.paths}}
        direction = {"episode_for_audit_only": 2, "timestamp": 1., "observation_ref": "obs-000025",
                     "row_sha256": row_fingerprint(self.row), "camera": "cam_right_wrist", "image_path": self.paths[1],
                     "image_sha256": file_sha256(self.run/self.paths[1])}
        self.legacy = {"source_run": str(self.run.resolve()), "checkpoints": [ordinary], "conditional_directions": [direction]}
        self.review = migrate_review(self.legacy, self.run)

    def write_run(self, run, row=None):
        row = self.row if row is None else row
        for path in self.paths:
            target = run/path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(("synthetic file fixture "+path).encode())
        (run/"trajectory_000002/decisions.jsonl").write_text(json.dumps(row)+"\n", encoding="utf-8")
        (run/"summary.json").write_text(json.dumps({"episodes": [{"episode_for_audit_only": 2,
                                                                "output_subdir": "trajectory_000002"}]}), encoding="utf-8")

    def audit(self, review=None, rows=None):
        return validate_review_bindings(self.review if review is None else review,
                                       load_run_rows(self.run) if rows is None else rows, self.run)

    def test_valid_legacy_migration_adds_versioned_entire_row_binding(self):
        self.assertTrue(self.audit()["valid"])
        self.assertNotIn("row_sha256", self.legacy["checkpoints"][0])
        record = self.review["checkpoints"][0]
        self.assertEqual(record["row_fingerprint_version"], FINGERPRINT_VERSION)
        self.assertEqual(record["row_sha256"], row_fingerprint(self.row))

    def test_wrong_source_invalidates_ordinary_and_direction(self):
        review = deepcopy(self.review)
        review["source_run"] = str(self.root/"missing")
        audit = self.audit(review)
        self.assertFalse(audit["valid"])
        self.assertEqual(audit["checkpoints"][0]["status"], "invalid")
        self.assertEqual(audit["conditional_directions"][0]["status"], "invalid")

    def test_missing_fields_do_not_auto_pass_and_window_is_not_unique_point(self):
        for field in ("row_sha256", "observation_ref", "row_fingerprint_version", "image_refs", "image_sha256", "timestamp"):
            with self.subTest(field=field):
                review = deepcopy(self.review)
                review["checkpoints"][0].pop(field)
                self.assertFalse(self.audit(review)["valid"])
        review = deepcopy(self.review)
        point = review["checkpoints"][0]
        point.pop("timestamp")
        point.update(start_s=0., end_s=2.)
        self.assertFalse(self.audit(review)["valid"])

    def test_bad_sha_missing_image_and_wrong_current_image_rejected(self):
        review = deepcopy(self.review)
        review["checkpoints"][0]["image_sha256"][self.paths[0]] = "bad"
        self.assertFalse(self.audit(review)["valid"])
        wrong = "trajectory_000002/frames/obs-000000_cam_high.jpg"
        (self.run/wrong).write_bytes(b"other real moment fixture")
        review = deepcopy(self.review)
        review["checkpoints"][0].update(image_refs=[wrong], image_sha256={wrong: file_sha256(self.run/wrong)})
        self.assertFalse(self.audit(review)["valid"])
        (self.run/self.paths[0]).unlink()
        self.assertFalse(self.audit()["valid"])

    def test_path_escape_rejected_even_with_existing_file_and_correct_hash(self):
        outside = self.root/"outside.jpg"
        outside.write_bytes(b"outside fixture")
        review = deepcopy(self.review)
        review["checkpoints"][0].update(image_refs=["../outside.jpg"], image_sha256={"../outside.jpg": file_sha256(outside)})
        self.assertFalse(self.audit(review)["valid"])

    def test_duplicate_or_changed_row_rejected_including_processing(self):
        self.assertFalse(self.audit(rows={2: [self.row, deepcopy(self.row)]})["valid"])
        changed = deepcopy(self.row)
        changed["processing"]["decision_wall_s"] = .2
        self.assertFalse(self.audit(rows={2: [changed]})["valid"])
        changed = deepcopy(self.row)
        changed["observation_ref"] = "another"
        self.assertFalse(self.audit(rows={2: [changed]})["valid"])

    def test_explicit_causal_history_reference_only(self):
        row = deepcopy(self.row)
        reference = {"ref": "obs-000000", "timestamp": 0., "camera": "cam_high"}
        row["visual_facts"]["evidence_refs"] = [reference]
        path = "trajectory_000002/frames/obs-000000_cam_high.jpg"
        (self.run/path).write_bytes(b"historical fixture")
        record = deepcopy(self.review["checkpoints"][0])
        record.update(row_sha256=row_fingerprint(row), image_refs=[path], image_sha256={path: file_sha256(self.run/path)})
        self.assertTrue(record_binding_errors(record, [row], self.run)[0])
        record["allowed_history_refs"] = [reference]
        self.assertFalse(record_binding_errors(record, [row], self.run)[0])
        record["allowed_history_refs"] = [{**reference, "timestamp": 2.}]
        self.assertTrue(record_binding_errors(record, [row], self.run)[0])

    def test_migration_refuses_bad_existing_bindings_instead_of_overwriting(self):
        for kind in ("source", "image", "direction_row", "ordinary_existing_row", "existing_definition", "explicit_null_ref"):
            review = deepcopy(self.legacy)
            if kind == "source": review["source_run"] = str(self.root/"wrong")
            if kind == "image": review["checkpoints"][0]["image_sha256"][self.paths[0]] = "bad"
            if kind == "direction_row": review["conditional_directions"][0]["row_sha256"] = "bad"
            if kind == "ordinary_existing_row": review["checkpoints"][0]["row_sha256"] = "bad"
            if kind == "existing_definition": review["row_fingerprint_definition"] = {"scope": "wrong"}
            if kind == "explicit_null_ref": review["checkpoints"][0]["observation_ref"] = None
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                migrate_review(review, self.run)

    def test_returned_contract_does_not_alias_module_contract(self):
        original = deepcopy(FINGERPRINT_DEFINITION)
        try:
            self.review["row_fingerprint_definition"]["scope"] = "external mutation"
            self.assertEqual(FINGERPRINT_DEFINITION, original)
        finally:
            FINGERPRINT_DEFINITION.clear()
            FINGERPRINT_DEFINITION.update(original)

    def test_explicit_rebind_same_pixels_preserves_expectations_and_records_new_row(self):
        new = self.root/"rerun"
        row = deepcopy(self.row)
        row["processing"]["decision_wall_s"] = .2
        self.write_run(new, row)
        self.assertFalse(validate_review_bindings(self.review, load_run_rows(new), new)["valid"])
        rebound = rebind_review(self.review, self.run, new, "test reviewer", "reviewed identical pixels and changed timing")
        self.assertTrue(validate_review_bindings(rebound, load_run_rows(new), new)["valid"])
        self.assertEqual(rebound["checkpoints"][0]["expected_stages"], self.review["checkpoints"][0]["expected_stages"])
        self.assertEqual(rebound["binding_rebind"]["changed_rows"][0]["changed_top_level_fields"], ["processing"])
        (new/self.paths[0]).write_bytes(b"changed pixels")
        with self.assertRaises(ValueError):
            rebind_review(self.review, self.run, new, "reviewer", "must fail")


if __name__ == "__main__":
    unittest.main()
