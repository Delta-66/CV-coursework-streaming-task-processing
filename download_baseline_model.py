"""Download a pinned official Qwen3-VL model with size/SHA256 verification."""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
from urllib.parse import quote
from urllib.request import ProxyHandler, Request, build_opener

from download_vifailback import sha256


def download_model(root, repo_id="Qwen/Qwen3-VL-2B-Instruct", revision=None, proxy=None):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    opener = build_opener(ProxyHandler({"http": proxy, "https": proxy})) if proxy else build_opener()
    def request_json(url, data=None):
        request = Request(url, data=json.dumps(data).encode() if data is not None else None,
                          headers={"Content-Type": "application/json", "User-Agent": "cv-baseline/1.0"})
        with opener.open(request, timeout=60) as response:
            return json.load(response)
    existing = root / "model_manifest.json"
    if existing.is_file():
        saved = json.loads(existing.read_text(encoding="utf-8"))
        revision = revision or saved["revision"]
        if saved["repo_id"] != repo_id or saved["revision"] != revision:
            raise ValueError("Use a new model directory for a different model/revision")
    info = request_json(f"https://huggingface.co/api/models/{repo_id}" + (f"/revision/{revision}" if revision else ""))
    revision = info["sha"]
    paths = [item["rfilename"] for item in info["siblings"]
             if item["rfilename"].endswith((".json", ".jinja", ".safetensors"))
             or item["rfilename"] in ("merges.txt", "README.md")]
    if not any(path.endswith(".safetensors") for path in paths):
        raise ValueError("Only Safetensors model weights are accepted")
    details = request_json(f"https://huggingface.co/api/models/{repo_id}/paths-info/{revision}", {"paths": paths})
    sizes = {item["path"]: item for item in details}
    files = []
    for relative_path in paths:
        parts = PurePosixPath(relative_path)
        if parts.is_absolute() or ".." in parts.parts:
            raise ValueError("Invalid model path")
        path = root.joinpath(*parts.parts)
        detail = sizes[relative_path]
        expected_hash = (detail.get("lfs") or {}).get("oid")
        valid = path.is_file() and path.stat().st_size == detail["size"]
        if valid and expected_hash:
            valid = sha256(path) == expected_hash
        if not valid:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".part")
            url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{quote(relative_path, safe='/')}"
            command = ["curl.exe" if os.name == "nt" else "curl", "--fail", "--location", "--silent", "--show-error",
                       "--retry", "3", "--connect-timeout", "30", "--speed-time", "60", "--speed-limit", "1024",
                       "--continue-at", "-", "--output", str(temporary)]
            if proxy:
                command.extend(["--proxy", proxy])
            print("Downloading:", relative_path, f"({detail['size']/1024**3:.3f} GiB)", flush=True)
            subprocess.run(command + [url], check=True)
            if temporary.stat().st_size != detail["size"]:
                raise IOError(f"Incomplete model file: {relative_path}")
            if expected_hash and sha256(temporary) != expected_hash:
                raise IOError(f"SHA256 mismatch: {relative_path}")
            temporary.replace(path)
        files.append({"path": relative_path, "size": detail["size"], "sha256": sha256(path),
                      "remote_sha256": expected_hash})
    manifest = {"repo_id": repo_id, "revision": revision, "files": files}
    existing.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Model downloaded and verified:", root, revision)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("models/Qwen3-VL-2B-Instruct"))
    parser.add_argument("--repo-id", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--revision", default="89644892e4d85e24eaac8bacfd4f463576704203")
    parser.add_argument("--proxy")
    args = parser.parse_args()
    download_model(args.root, args.repo_id, args.revision, args.proxy)
