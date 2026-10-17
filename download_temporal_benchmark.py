"""Prepare and download the frozen detection cohort, one full trajectory at a time."""
import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import time
from urllib.parse import quote
from urllib.request import Request

from download_vifailback import HubDownload, sha256
from prepare_dataset_splits import LEROBOT_REPO, LEROBOT_REVISION, METADATA_FILES, verify_manifest

ROOT = Path(__file__).resolve().parent
DEFAULT_PLAN = ROOT / "data/vifailback_temporal_download_plan.json"
CAMERA = "observation.images.cam_high"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_plan(root, split_path=ROOT / "configs/splits/test.jsonl", plan_path=DEFAULT_PLAN, proxy=None):
    root, split_path, plan_path = Path(root), Path(split_path), Path(plan_path)
    verify_manifest(root, METADATA_FILES, LEROBOT_REPO, LEROBOT_REVISION)
    protocol = json.loads((split_path.parent / "protocol.json").read_text(encoding="utf-8"))
    if sha256(split_path) != protocol["artifacts"][split_path.name]["sha256"]:
        raise ValueError("Test membership does not match the frozen split protocol")
    rows = [json.loads(line) for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row["benchmark_cohort"] == "canonical_detection"]
    if len(rows) != 500 or len({row["episode_index"] for row in rows}) != 500:
        raise ValueError("Expected all 500 unique, unexposed detection benchmark trajectories")
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    episodes, paths = [], []
    for row in rows:
        values = {"episode_index": row["episode_index"], "episode_chunk": row["episode_index"] // info["chunks_size"]}
        names = [info["data_path"].format(**values), info["video_path"].format(video_key=CAMERA, **values)]
        episodes.append({"episode_index": row["episode_index"], "num_frames": row["num_frames"], "files": names})
        paths.extend(names)
    hub = HubDownload(LEROBOT_REPO, root, LEROBOT_REVISION, proxy)
    remote = {}
    for offset in range(0, len(paths), 100):
        batch = paths[offset:offset + 100]
        values = hub.json(f"https://huggingface.co/api/datasets/{LEROBOT_REPO}/paths-info/{LEROBOT_REVISION}", {"paths": batch})
        if {item["path"] for item in values} != set(batch) or any(item.get("type") != "file" for item in values):
            raise ValueError("Remote file listing does not cover the planned source files")
        for item in values:
            remote[item["path"]] = {"path": item["path"], "size": item["size"],
                                     "remote_sha256": (item.get("lfs") or {}).get("oid")}
        print(f"Source sizes verified: {offset + len(batch)}/{len(paths)}", flush=True)
    for episode in episodes:
        episode["files"] = [remote[name] for name in episode["files"]]
    plan = {"repo_id": LEROBOT_REPO, "revision": LEROBOT_REVISION, "camera": CAMERA,
            "split_manifest_sha256": sha256(split_path), "split_path": str(split_path.resolve()),
            "episodes": episodes, "total_bytes": sum(item["size"] for item in remote.values()),
            "total_frames": sum(row["num_frames"] for row in rows), "total_episodes": len(rows),
            "file_count": len(remote), "offline_labels_used_for_selection": False}
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise FileExistsError("Existing download plan differs; review a new destination explicitly")
    write_json(plan_path, plan)
    print(json.dumps({key: plan[key] for key in ("total_episodes", "total_frames", "total_bytes", "file_count")}), flush=True)
    return plan


def ensure_episode(root, episode_index, plan_path=DEFAULT_PLAN, proxy=None):
    root = Path(root).resolve()
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    if plan["repo_id"] != LEROBOT_REPO or plan["revision"] != LEROBOT_REVISION or plan["camera"] != CAMERA:
        raise ValueError("Download plan source differs from the frozen source")
    if sha256(Path(plan["split_path"])) != plan["split_manifest_sha256"]:
        raise ValueError("Split membership changed after preparing download plan")
    entries = [item for item in plan["episodes"] if item["episode_index"] == episode_index]
    if len(entries) != 1:
        raise ValueError("Episode is not uniquely present in the prepared benchmark")
    hub = HubDownload(LEROBOT_REPO, root, LEROBOT_REVISION, proxy)
    records = []
    for item in entries[0]["files"]:
        if not isinstance(item["path"], str) or not item["path"] or ":" in item["path"] or "\\" in item["path"]:
            raise ValueError("Download path escapes dataset root or is not a relative POSIX path")
        parts = PurePosixPath(item["path"])
        path = root.joinpath(*parts.parts).resolve()
        if parts.is_absolute() or ".." in parts.parts or not path.is_relative_to(root):
            raise ValueError("Download path escapes dataset root")
        expected_hash = item["remote_sha256"]
        valid = path.is_file() and path.stat().st_size == item["size"] and (not expected_hash or sha256(path) == expected_hash)
        if not valid:
            if shutil.disk_usage(root).free < item["size"] + 2 * 1024**3:
                raise OSError("Insufficient disk space; preserve 2 GiB for system and result logs")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".part")
            url = f"https://huggingface.co/datasets/{LEROBOT_REPO}/resolve/{LEROBOT_REVISION}/{quote(item['path'], safe='/')}"
            for attempt in range(3):
                try:
                    print(f"Downloading episode {episode_index}: {item['path']} ({item['size']} bytes)", flush=True)
                    request = Request(url, headers={"User-Agent": "cv-streaming-input/1.0"})
                    with hub.opener.open(request, timeout=60) as response, temporary.open("wb") as file:
                        for block in iter(lambda: response.read(1024 * 1024), b""):
                            file.write(block)
                    if temporary.stat().st_size != item["size"] or (expected_hash and sha256(temporary) != expected_hash):
                        raise IOError("Downloaded file does not match pinned size/SHA256")
                    temporary.replace(path)
                    break
                except (OSError, ValueError):
                    if attempt == 2:
                        raise
                    time.sleep(2 ** (attempt + 1))
        records.append({**item, "sha256": sha256(path)})
    # Only this process writes the manifest; preserve all earlier project samples.
    manifest_path = root / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    merged = {item["path"]: item for item in manifest["files"]}
    merged.update({item["path"]: item for item in records})
    manifest["files"] = list(merged.values())
    manifest["selected_episodes"] = sorted(set(manifest.get("selected_episodes", [])) | {episode_index})
    write_json(manifest_path, manifest)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data/vifailback_lerobot")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--proxy")
    parser.add_argument("--download", action="store_true", help="Download every planned trajectory, without model inference")
    args = parser.parse_args()
    plan = prepare_plan(args.root, plan_path=args.plan, proxy=args.proxy)
    if args.download:
        for item in plan["episodes"]:
            ensure_episode(args.root, item["episode_index"], args.plan, args.proxy)
