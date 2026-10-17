"""只获取两条非测试补充轨迹；冻结划分不变，不运行推理。"""
import json
import argparse
from pathlib import Path
from download_vifailback import HubDownload, LEROBOT_REPO

def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes",type=int,nargs="+",choices=[2,6,19],default=[2,19])
    episodes=parser.parse_args().episodes
    selected=[("train" if ep==19 else "validation",ep) for ep in episodes]
    test={json.loads(s)["episode_index"] for s in (root/"configs/splits/test.jsonl").read_text(encoding="utf-8").splitlines()}
    for split,episode in selected:
        members={json.loads(s)["episode_index"] for s in (root/f"configs/splits/{split}.jsonl").read_text(encoding="utf-8").splitlines()}
        if episode not in members or episode in test:
            raise ValueError("frozen non-test selection no longer matches")
    data=root/"data/vifailback_lerobot"
    revision=json.loads((data/"download_manifest.json").read_text(encoding="utf-8"))["revision"]
    hub=HubDownload(LEROBOT_REPO,data,revision)
    for _,episode in selected:
        paths=[f"data/chunk-000/episode_{episode:06d}.parquet"]+[f"videos/chunk-000/observation.images.{c}/episode_{episode:06d}.mp4" for c in ("cam_high","cam_right_wrist")]
        for path in paths:
            hub.download(path)
    hub.save_manifest(episodes)
    output=root/("streaming_recovery/artifacts/incremental_20261017/download_"+"_".join(map(str,episodes))+".json")
    output.write_text(json.dumps({"repo":LEROBOT_REPO,"revision":revision,"files":hub.files},indent=2),encoding="utf-8")
    print("Downloaded and hash-verified non-test episodes:",episodes)

if __name__=="__main__":
    main()
