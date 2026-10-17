"""Exercise the real pinned downloader with local byte streams, never live HTTP."""
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

from download_temporal_benchmark import CAMERA, ensure_episode
from prepare_dataset_splits import LEROBOT_REPO, LEROBOT_REVISION


class ChunkedResponse(io.BytesIO):
    """Return short HTTP-like reads so the downloader must consume the stream."""

    def read(self, size=-1):
        return super().read(min(size, 5) if size >= 0 else 5)


def digest(content):
    return hashlib.sha256(content).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


class TemporalDownloadTests(unittest.TestCase):
    def setUp(self):
        # Work within the repository's writable scratch directory on Windows.
        scratch = Path(__file__).resolve().parents[1] / "tmp"
        scratch.mkdir(exist_ok=True)
        self.temporary = TemporaryDirectory(prefix="temporal-download-test-", dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name).resolve()
        self.root = self.folder / "dataset"
        self.root.mkdir()
        self.episode_index = 42
        self.payloads = {
            "data/chunk-000/episode_000042.parquet": b"synthetic parquet byte fixture, no task answers",
            f"videos/chunk-000/{CAMERA}/episode_000042.mp4": b"synthetic video byte fixture, no task answers",
        }
        self.legacy = {"path": "meta/legacy.json", "size": 2,
                       "sha256": digest(b"{}"), "remote_sha256": digest(b"{}")}
        legacy_path = self.root / self.legacy["path"]
        legacy_path.parent.mkdir(parents=True)
        legacy_path.write_bytes(b"{}")
        self.manifest_path = self.root / "download_manifest.json"
        write_json(self.manifest_path, {
            "repo_id": LEROBOT_REPO, "revision": LEROBOT_REVISION,
            "selected_episodes": [1], "files": [self.legacy],
        })
        self.split_path = self.folder / "test.jsonl"
        self.split_path.write_text(json.dumps({"episode_index": self.episode_index,
                                              "benchmark_cohort": "canonical_detection"}) + "\n",
                                   encoding="utf-8")
        self.plan = {
            "repo_id": LEROBOT_REPO, "revision": LEROBOT_REVISION, "camera": CAMERA,
            "split_path": str(self.split_path),
            "split_manifest_sha256": digest(self.split_path.read_bytes()),
            "episodes": [{"episode_index": self.episode_index, "num_frames": 101,
                          "files": [{"path": name, "size": len(payload), "remote_sha256": digest(payload)}
                                    for name, payload in self.payloads.items()]}],
        }
        self.plan_path = self.folder / "plan.json"
        write_json(self.plan_path, self.plan)

    def network(self, payloads=()):
        responses = [ChunkedResponse(payload) for payload in payloads]
        opener = SimpleNamespace(open=Mock(side_effect=responses if responses else
                                         AssertionError("Unexpected network access")))
        return opener, patch("download_vifailback.build_opener", return_value=opener), responses

    def save_payload(self, name, payload):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def test_verified_cached_files_require_no_network_and_preserve_previous_manifest_entries(self):
        for name, payload in self.payloads.items():
            self.save_payload(name, payload)
        opener, transport, _ = self.network()
        with transport, patch("download_temporal_benchmark.time.sleep") as sleep:
            records = ensure_episode(self.root, self.episode_index, self.plan_path)
        opener.open.assert_not_called()
        sleep.assert_not_called()
        self.assertEqual({item["path"]: item["sha256"] for item in records},
                         {name: digest(payload) for name, payload in self.payloads.items()})
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["selected_episodes"], [1, self.episode_index])
        self.assertEqual(manifest["files"][0], self.legacy)
        self.assertEqual(len(manifest["files"]), 3)
        self.assertFalse(list(self.root.rglob("*.part")))
        for name, payload in self.payloads.items():
            self.assertEqual((self.root / name).read_bytes(), payload)

    def test_chunked_download_checks_hashes_and_atomically_merges_manifest(self):
        opener, transport, responses = self.network(self.payloads.values())
        with transport, patch("download_temporal_benchmark.time.sleep") as sleep, patch("builtins.print"):
            records = ensure_episode(self.root, self.episode_index, self.plan_path)
        self.assertEqual(opener.open.call_count, len(self.payloads))
        sleep.assert_not_called()
        self.assertTrue(all(response.closed for response in responses))
        for request_call, (name, payload), record in zip(opener.open.call_args_list, self.payloads.items(), records):
            request = request_call.args[0]
            self.assertIn(f"/resolve/{LEROBOT_REVISION}/", request.full_url)
            self.assertTrue(request.full_url.endswith(name))
            self.assertEqual(request_call.kwargs["timeout"], 60)
            self.assertEqual((self.root / name).read_bytes(), payload)
            self.assertEqual(record["sha256"], digest(payload))
            self.assertEqual(record["remote_sha256"], digest(payload))
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["files"][0], self.legacy)
        self.assertEqual(manifest["selected_episodes"], [1, self.episode_index])
        self.assertFalse(list(self.root.rglob("*.part")))
        self.assertFalse(self.manifest_path.with_name("download_manifest.json.tmp").exists())

    def test_same_size_bad_sha_preserves_old_file_and_does_not_write_manifest(self):
        names = list(self.payloads)
        self.save_payload(names[0], self.payloads[names[0]])
        old_payload = b"the previously downloaded file must remain intact"
        destination = self.save_payload(names[1], old_payload)
        wrong_payload = b"x" * len(self.payloads[names[1]])
        self.assertEqual(len(wrong_payload), len(self.payloads[names[1]]))
        self.assertNotEqual(digest(wrong_payload), digest(self.payloads[names[1]]))
        original_manifest = self.manifest_path.read_bytes()
        opener, transport, responses = self.network([wrong_payload] * 3)
        with transport, patch("download_temporal_benchmark.time.sleep") as sleep, \
                patch("download_temporal_benchmark.write_json") as write_manifest, patch("builtins.print"):
            with self.assertRaisesRegex(OSError, "size/SHA256"):
                ensure_episode(self.root, self.episode_index, self.plan_path)
        self.assertEqual(opener.open.call_count, 3)
        self.assertEqual(sleep.call_args_list, [call(2), call(4)])
        write_manifest.assert_not_called()
        self.assertEqual(self.manifest_path.read_bytes(), original_manifest)
        self.assertEqual(destination.read_bytes(), old_payload)
        self.assertEqual((self.root / names[0]).read_bytes(), self.payloads[names[0]])
        self.assertEqual(destination.with_name(destination.name + ".part").read_bytes(), wrong_payload)
        self.assertTrue(all(response.closed for response in responses))

    def test_episode_outside_prepared_plan_is_rejected_before_network_or_writes(self):
        original_manifest = self.manifest_path.read_bytes()
        opener, transport, _ = self.network()
        with transport as build_opener:
            with self.assertRaisesRegex(ValueError, "not uniquely present"):
                ensure_episode(self.root, self.episode_index + 1, self.plan_path)
        build_opener.assert_not_called()
        opener.open.assert_not_called()
        self.assertEqual(self.manifest_path.read_bytes(), original_manifest)
        self.assertFalse((self.root / "data").exists())
        self.assertFalse((self.root / "videos").exists())

    def test_path_escape_is_rejected_without_touching_outside_file(self):
        outside = self.folder / "outside.bin"
        outside.write_bytes(b"outside dataset root")
        original_manifest = self.manifest_path.read_bytes()
        paths = ["../outside.bin", "data/../../outside.bin", outside.as_posix()]
        # A backslash traversal is meaningful on the Windows host, not POSIX.
        if __import__("os").name == "nt":
            paths.append("..\\outside.bin")
        for name in paths:
            with self.subTest(path=name):
                self.plan["episodes"][0]["files"] = [{
                    "path": name, "size": len(b"replacement"), "remote_sha256": digest(b"replacement"),
                }]
                write_json(self.plan_path, self.plan)
                opener, transport, _ = self.network()
                with transport:
                    with self.assertRaisesRegex(ValueError, "escapes dataset root"):
                        ensure_episode(self.root, self.episode_index, self.plan_path)
                opener.open.assert_not_called()
                self.assertEqual(outside.read_bytes(), b"outside dataset root")
                self.assertEqual(self.manifest_path.read_bytes(), original_manifest)
                self.assertFalse(list(self.folder.rglob("*.part")))

    def test_changed_frozen_split_is_rejected_before_network(self):
        self.split_path.write_text("changed membership\n", encoding="utf-8")
        original_manifest = self.manifest_path.read_bytes()
        opener, transport, _ = self.network()
        with transport as build_opener:
            with self.assertRaisesRegex(ValueError, "Split membership changed"):
                ensure_episode(self.root, self.episode_index, self.plan_path)
        build_opener.assert_not_called()
        opener.open.assert_not_called()
        self.assertEqual(self.manifest_path.read_bytes(), original_manifest)


if __name__ == "__main__":
    unittest.main()
