"""实图模型接通检查，供独立复现；不把输出当作人工真值。"""
import json
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from .source import LeRobotStream, instruction_for_episode
from .vision import QwenProcess


def main():
    base = Path(__file__).resolve().parent
    config = json.loads((base / "config.json").read_text(encoding="utf-8"))
    config["model"]["path"] = str((base / config["model"]["path"]).resolve())
    root = (base / config["source"]["root"]).resolve()
    parser=argparse.ArgumentParser(description="实验性 Qwen 原始视觉输出检查；不是验收主方法")
    parser.add_argument("--output")
    args=parser.parse_args()
    path=Path(args.output) if args.output else base/"artifacts"/f'probe_{datetime.now():%Y%m%d_%H%M%S}.jsonl'
    if path.exists():
        raise FileExistsError(path)
    model = QwenProcess(config["model"])
    if model.error:
        raise RuntimeError(model.error)
    try:
        with path.open("x", encoding="utf-8") as f:
            for episode in (0,18):
                previous = None
                stream = LeRobotStream(root, episode, ["cam_high","cam_right_wrist"], sample_every=50,
                                        instruction=instruction_for_episode(root, episode))
                for observation in stream:
                    facts = model.observe(observation, previous)
                    row = {"episode_for_audit_only":episode,"facts":asdict(facts),"metrics":model.last_metrics}
                    f.write(json.dumps(row, ensure_ascii=False)+"\n");f.flush()
                    print(json.dumps({"episode":episode,"t":facts.timestamp,"location":facts.knife_location,
                                      "contact":facts.contact,"motion":facts.motion,"co_motion":facts.co_motion,
                                      "cut":facts.cut_piece,"error":facts.error,"description":facts.description},ensure_ascii=False),flush=True)
                    previous = observation
    finally:
        model.close()


if __name__ == "__main__":
    main()
