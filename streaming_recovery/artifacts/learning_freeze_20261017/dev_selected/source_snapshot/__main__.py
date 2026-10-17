from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import shutil
from pathlib import Path
import sys
import time
from .source import LeRobotStream, instruction_for_episode
from .geometry import GeometryVision
from .engine import Monitor


BASE=Path(__file__).resolve().parent


def load_config(path=None):
    path=Path(path) if path else BASE/"config.json"
    config=json.loads(path.read_text(encoding="utf-8"))
    for section,key in (("source","root"),("model","path")):
        config[section][key]=str((path.parent/config[section][key]).resolve())
    config["output"]=str((path.parent/config["output"]).resolve())
    if "fusion" in config:
        config["fusion"]["path"]=str((path.parent/config["fusion"]["path"]).resolve())
    if any(config["memory"][k]<1 for k in ("frames","features","events","event_evidence","anchors","objects")):
        raise ValueError("memory capacities must be positive")
    if config["memory"]["objects"]<3 or config["memory"]["window_s"]<=0:
        raise ValueError("profile requires 3 object slots and positive window")
    if config["rules"]["confirm_samples"]<2 or config["rules"]["confirm_span_s"]<=0:
        raise ValueError("failure confirmation requires repeated independent observations")
    if config["memory"]["features"]<config["rules"]["confirm_samples"] or config["memory"]["event_evidence"]<2:
        raise ValueError("insufficient feature/evidence capacity for confirmation")
    if config["memory"]["anchors"]<6 or config["memory"]["window_s"]<config["rules"]["confirm_span_s"]:
        raise ValueError("insufficient anchor/window capacity for this task profile")
    if any(config["geometry"][k]<v for k,v in (("max_reference_images",1),("max_masks",3),("max_tracks",2))):
        raise ValueError("geometry profile needs one reference, three masks and two tracks")
    return config


def sha256(path):
    value=hashlib.sha256()
    with Path(path).open("rb") as stream:
        while data:=stream.read(1024*1024):
            value.update(data)
    return value.hexdigest()

def make_vision(config):
    if config["backend"]=="fixed_supervised_fusion":
        from .fusion import TrainedFusionVision
        return TrainedFusionVision(config["geometry"],config["rules"],config["fusion"]["path"])
    if config["backend"]=="calibrated_rgb_geometry":
        return GeometryVision(config["geometry"],config["rules"])
    if config["backend"]=="multiview_effect_geometry":
        from .multiview import MultiViewGeometryVision
        return MultiViewGeometryVision(config["geometry"],config["rules"])
    raise ValueError("unknown production backend; Qwen remains an explicitly separate experiment")


def run(config, output=None):
    run_id=datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    run_dir=Path(output) if output else Path(config["output"])/run_id
    run_dir.mkdir(parents=True,exist_ok=False)
    (run_dir/"config_snapshot.json").write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8")
    (run_dir/"source_snapshot").mkdir()
    for path in BASE.glob("*.py"):
        shutil.copy2(path,run_dir/"source_snapshot"/path.name)
    root=Path(config["source"]["root"])
    vision=make_vision(config)
    monitor=Monitor(config,vision)
    summary={"run_id":run_id,"date_timezone":"Asia/Shanghai","python":sys.version,"backend":config["backend"],
        "actual_data":True,"synthetic":False,"execution_mode":"advisory_replay","clock_policy":"25 FPS source; configured sampling; synchronous backpressure, not wall-clock real-time guarantee",
        "episodes":[],"code_sha256":{p.name:sha256(p) for p in BASE.glob("*.py")},
        "test_sha256":{p.name:sha256(p) for p in (BASE/"tests").glob("test_*.py")}}
    start=time.monotonic()
    try:
        for episode in config["source"]["episodes"]:
            episode_start=time.monotonic()
            monitor.reset()
            instruction=instruction_for_episode(root,episode)
            directory=run_dir/f"trajectory_{episode:06d}"
            (directory/"frames").mkdir(parents=True)
            stream=LeRobotStream(root,episode,config["source"]["cameras"],config["source"]["sample_every"],
                config["source"]["max_age_s"],config["model"]["image_edge"],instruction)
            counts={"decisions":0,"notifications":0,"confirmed_failure_decisions":0,"uncertain_decisions":0,"model_invalid":0}
            maxima={}
            milestones=[]
            with (directory/"decisions.jsonl").open("x",encoding="utf-8") as out:
                for observation in stream:
                    tick=time.monotonic()
                    result=monitor.step(observation)
                    elapsed=time.monotonic()-tick
                    result["processing"]={"decision_wall_s":elapsed,"includes_decode_or_image_save":False}
                    for camera,im in observation.images.items():
                        im.save(directory/"frames"/f"{observation.ref}_{camera}.jpg",quality=90)
                    out.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+"\n");out.flush()
                    counts["decisions"]+=1
                    counts["notifications"]+=int(result["notify"])
                    counts["confirmed_failure_decisions"]+=int(result["execution_status"]=="confirmed_failure")
                    counts["uncertain_decisions"]+=int(result["execution_status"]=="information_insufficient")
                    counts["model_invalid"]+=int(not result["visual_facts"]["valid"])
                    for k,v in result["memory_usage"].items():
                        maxima[k]=max(maxima.get(k,0),v)
                    milestones.append({"timestamp":result["timestamp"],"stage":result["state"]["stage"],"status":result["execution_status"],
                        "held":result["state"]["preconditions"]["reliable_grasp"],"event":result["failure_event"]["event_id"] if result["failure_event"] else None,
                        "notify":result["notify"],"decision_latency_s":elapsed})
                    print(f'轨迹 {episode} t={result["timestamp"]:.2f}s {result["state"]["stage_zh"]} | {result["execution_status"]} | 提醒={result["notify"]}',flush=True)
            end=monitor.end_of_stream()
            (directory/"end.json").write_text(json.dumps(end,ensure_ascii=False,indent=2),encoding="utf-8")
            chunk=episode//1000
            files=[root/f'data/chunk-{chunk:03d}/episode_{episode:06d}.parquet']+[root/f'videos/chunk-{chunk:03d}/observation.images.{c}/episode_{episode:06d}.mp4' for c in config["source"]["cameras"]]
            summary["episodes"].append({"episode_for_audit_only":episode,"output_subdir":directory.name,"counts":counts,
                "elapsed_wall_s":time.monotonic()-episode_start,
                "decision_wall_total_s":sum(m["decision_latency_s"] for m in milestones),
                "decision_wall_max_s":max((m["decision_latency_s"] for m in milestones),default=0),
                "source_audit":stream.audit,"memory_maxima":maxima,"timeline":milestones,"end":end,
                "source_files":[{"path":str(p),"sha256":sha256(p)} for p in files if p.exists()]})
        summary["elapsed_wall_s"]=time.monotonic()-start
        (run_dir/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
        from .report import render_report
        render_report(run_dir)
        print("实际运行输出："+str(run_dir),flush=True)
        return run_dir
    finally:
        vision.close()


def main():
    parser=argparse.ArgumentParser(description="独立流式任务监测与分层纠错")
    parser.add_argument("command",choices=["run","validate"],nargs="?",default="run")
    parser.add_argument("--config")
    parser.add_argument("--output")
    parser.add_argument("--run",help="validate 复用已生成的真实回放，不重复六条推理")
    parser.add_argument("--manual-review",help="validate 使用的离线关键时刻核查 JSON")
    parser.add_argument("--require-complete",action="store_true",help="语义能力仍有缺口时也返回非零状态")
    args=parser.parse_args()
    config=load_config(args.config)
    if args.command=="run":
        run(config,args.output)
    else:
        from .validation import validate
        validate(config,args.output,args.run,args.manual_review,args.require_complete)


if __name__=="__main__":
    main()
