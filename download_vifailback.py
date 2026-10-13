"""Download selected official episodes, pinned to a Hub commit, with checksums."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from urllib.request import ProxyHandler, Request, build_opener


LEROBOT_REPO = "sii-rhos-ai/vifailback-dataset-lerobot"
RAW_REPO = "sii-rhos-ai/ViFailback-Dataset"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class HubDownload:
    def __init__(self, repo_id, root, revision=None, proxy=None):
        self.repo_id = repo_id
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.opener = build_opener(ProxyHandler({"http": proxy, "https": proxy})) if proxy else build_opener()
        self.revision = revision or self.json(f"https://huggingface.co/api/datasets/{repo_id}")["sha"]
        manifest_path = self.root / "download_manifest.json"
        if manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing["repo_id"] != repo_id or existing["revision"] != self.revision:
                raise ValueError("Use a new dataset root for a different repository/revision")
        self.files = []

    def json(self, url, data=None):
        request = Request(url, data=json.dumps(data).encode() if data is not None else None,
                          headers={"User-Agent": "cv-streaming-input/1.0", "Content-Type": "application/json"})
        with self.opener.open(request, timeout=60) as response:
            return json.load(response)

    def download(self, relative_path):
        parts = PurePosixPath(relative_path)
        if parts.is_absolute() or ".." in parts.parts:
            raise ValueError("Remote paths must stay inside the dataset root")
        path = self.root.joinpath(*parts.parts)
        info = self.json(f"https://huggingface.co/api/datasets/{self.repo_id}/paths-info/{self.revision}",
                         {"paths": [relative_path]})
        if len(info) != 1 or info[0].get("type") != "file":
            raise FileNotFoundError(f"Remote dataset file missing: {relative_path}")
        expected_size = info[0]["size"]
        expected_hash = (info[0].get("lfs") or {}).get("oid")
        if not path.is_file() or path.stat().st_size != expected_size or (expected_hash and sha256(path) != expected_hash):
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".part")
            url = f"https://huggingface.co/datasets/{self.repo_id}/resolve/{self.revision}/{quote(relative_path, safe='/')}"
            print("Downloading:", relative_path, f"({expected_size / 1024**2:.1f} MiB)", flush=True)
            request = Request(url, headers={"User-Agent": "cv-streaming-input/1.0"})
            with self.opener.open(request, timeout=60) as response, temporary.open("wb") as file:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    file.write(block)
            if temporary.stat().st_size != expected_size:
                raise IOError(f"Incomplete download: {relative_path}")
            if expected_hash and sha256(temporary) != expected_hash:
                raise IOError(f"SHA256 mismatch: {relative_path}")
            temporary.replace(path)
        self.files.append({"path": relative_path, "size": expected_size, "sha256": sha256(path),
                           "remote_sha256": expected_hash})
        return path

    def save_manifest(self, episodes):
        manifest_path = self.root / "download_manifest.json"
        previous = []
        previous_episodes = []
        if manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing["repo_id"] != self.repo_id or existing["revision"] != self.revision:
                raise ValueError("Dataset root already contains a different repository/revision")
            previous = existing["files"]
            previous_episodes = existing.get("selected_episodes", [])
        merged = {entry["path"]: entry for entry in previous + self.files}
        manifest = {"repo_id": self.repo_id, "revision": self.revision,
                    "selected_episodes": sorted(set(previous_episodes + list(episodes))),
                    "files": list(merged.values())}
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def download_episodes(root, episodes, proxy=None, revision=None, raw_root=None, raw_revision=None):
    root = Path(root)
    manifest_path = root / "download_manifest.json"
    if manifest_path.is_file() and revision is None:
        revision = json.loads(manifest_path.read_text(encoding="utf-8"))["revision"]
    hub = HubDownload(LEROBOT_REPO, root, revision, proxy)
    for name in ("README.md", "meta/info.json", "meta/tasks.jsonl", "meta/episodes.jsonl",
                 "meta/vifailback_annotations.jsonl"):
        hub.download(name)
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    if info["codebase_version"] != "v2.1":
        raise ValueError("This downloader supports the official LeRobot v2.1 layout")
    available = {item["episode_index"] for item in read_jsonl(root / "meta/episodes.jsonl")}
    if any(type(episode) is not int or episode not in available for episode in episodes):
        raise ValueError("Selected episode is absent from metadata")
    cameras = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    for episode_index in episodes:
        values = {"episode_index": episode_index, "episode_chunk": episode_index // info["chunks_size"]}
        hub.download(info["data_path"].format(**values))
        for video_key in cameras:
            hub.download(info["video_path"].format(video_key=video_key, **values))
    hub.save_manifest(episodes)
    if raw_root:
        raw_manifest = Path(raw_root) / "download_manifest.json"
        if raw_manifest.is_file() and raw_revision is None:
            raw_revision = json.loads(raw_manifest.read_text(encoding="utf-8"))["revision"]
        raw = HubDownload(RAW_REPO, raw_root, raw_revision, proxy)
        raw.download("README.md")
        selected = set(episodes)
        for item in read_jsonl(root / "meta/vifailback_annotations.jsonl"):
            if item["episode_index"] in selected:
                raw.download(item["source_hdf5"])
        raw.save_manifest(episodes)
    print("Downloaded and verified episodes:", episodes)
    print("Dataset:", root.resolve(), "revision:", hub.revision)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/vifailback_lerobot"))
    parser.add_argument("--episodes", nargs="+", type=int, default=[0, 18])
    parser.add_argument("--revision")
    parser.add_argument("--proxy", help="Optional proxy for this download only; no system settings are changed")
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--raw-revision")
    args = parser.parse_args()
    download_episodes(args.root, args.episodes, args.proxy, args.revision, args.raw_root, args.raw_revision)
