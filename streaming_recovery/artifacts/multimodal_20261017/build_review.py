"""离线审计及目视抽查记录整理；监测器不导入本文件，不将标签送回推理。"""
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parents[1]
WORKSPACE=BASE.parent

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def rows(directory,episode):
    return [json.loads(l) for l in (directory/f"trajectory_{episode:06d}/decisions.jsonl").read_text(encoding="utf-8").splitlines()]

# 以下是本次实际查看原始输入图片后的笔记，不是由 episode 标签生成的判断。
NOTES={
18:{"times":[0,4,5,9,14,19,20],"sheets":["18a.jpg","18b.jpg"],
    "observed":"初始无分离小块；5s 刀已离架、腕部可见刀与两指连接；9s 刀在香肠端部作用；14s 已见新圆片但算法当前夹持未知；19/20s 小片与刀具回架、夹爪分离清楚。",
    "assessment":"完整成功得到证据，无失败误报。4/5s 保守等待连续夹持；14/18s 仍漏掉部分当前关系，不能称所有阶段均正确。",
    "phase_gaps":["4/5s 抓取确认等待","14/18s 当前阶段未知"],"completion_missed":False},
19:{"times":[0,5,8,10,12,15,16],"sheets":["19a.jpg","19b.jpg"],
    "observed":"初始无小片；5/8s 腕图可见持刀邻接香肠；10s 已出现分离圆片并持刀离开砧板；12s 刀返回刀架；15/16s 圆片与归位清楚。",
    "assessment":"无失败误报，但整任务成功漏确认。1Hz 下 5s 有刀—食物框、6至8s 腕部刀刃轮廓未识别，未建立连续相对作用证据；不得借最终标签补齐。首次运行的回架/释放阶段映射错误已通过独立逻辑回归修复，切割识别缺口保留。",
    "phase_gaps":["3至7s 可靠夹持仍待确认","切割作用未确认，故新效果/完整成功不确认","11s 切割效果待确认","14s 当前阶段未知"],"completion_missed":True},
0:{"times":[7,9,10],"sheets":["0.jpg"],
   "observed":"7s 夹爪已闭但刀仍在架边；9/10s 空夹爪在香肠上方、刀留架，支持当前尝试未建立持刀。",
   "assessment":"抓取未建立提醒与目视证据一致；未误把闭爪当成功。原因仍未知。",
   "phase_gaps":["7s 抓取效果未知","刀柄/指尖精确对齐未识别"],"completion_missed":False},
2:{"times":[2,10,11],"sheets":["2.jpg"],
   "observed":"2s 顶视刀在架、腕图两指张开接近；10/11s 顶视刀仍在架、腕图空夹爪进入工作区。",
   "assessment":"前置条件未满足诊断正确；没有可靠记录目标附近闭合，attempt=0，因此未编造具体失败尝试次数。对齐/闭合阶段识别不足。",
   "phase_gaps":["0/1s 初始校准失败","接近/对齐/闭合仍多次输出 locate_tool","9s 场景证据不可用"],"completion_missed":False},
8:{"times":[4,7,8],"sheets":["8.jpg"],
   "observed":"4s 两指在刀旁、刀仍支撑于架；7/8s 空夹爪进入砧板上方，刀未被带走。",
   "assessment":"未见样例首次冻结测试正确提醒缺少持刀前置条件；未确认目标附近的完整开闭尝试（attempt=0）。4/5s 仍是 locate_tool，阶段粒度不足，即使 execution_status 没有 information_insufficient 也不能称全阶段正确。",
   "phase_gaps":["4/5s 对齐/尝试没有可靠定位","未确认实际抓取尝试次数"],"completion_missed":False},
29:{"times":[5,6,8,9],"sheets":["29.jpg"],
    "observed":"5/6s 两指向刀收拢但刀仍在架；8/9s 夹爪空手到砧板，刀留架。腕图仍能看见指间间隙；原始开度约0.0188被既有阈值归为closed，不能解释为完全闭合或成功夹持。",
    "assessment":"未见样例首次冻结测试正确提醒抓取未建立。算法未断言夹持力/摩擦原因，也没有精确诊断官方 gripper state 原因标签。",
    "phase_gaps":["0s 刀架初始化不确定","6s 抓取效果未知","closed 为粗阈值区间，完全闭合和接触仍未知"],"completion_missed":False},
}

def main():
    directory=ROOT/"final_run"
    summary=read(directory/"summary.json")
    selection=read(ROOT/"selection.json")
    annotations={r["episode_index"]:r for r in (json.loads(l) for l in
        (WORKSPACE/"data/vifailback_lerobot/meta/vifailback_annotations.jsonl").read_text(encoding="utf-8").splitlines())}
    result={"actual_real_run":True,"synthetic":False,"manual_review_kind":"assistant visually inspected 27 sampled timepoints (52 images); not independent human phase annotation",
            "run":"final_run","first_frozen_run":"run","sampling_hz":1,"per_episode":[],
            "elapsed_wall_s":summary["elapsed_wall_s"],"detection_latency":None,
            "latency_reason":"no first-detectable-failure manual timestamps; notification video times are not detection delays",
            "generalization_claim":False,"no_physical_execution":True}
    checks={"split_unchanged":all(sha(WORKSPACE/"configs/splits"/n)==v for n,v in selection["split_sha256"].items()),
            "current_code_matches_final_snapshot":all(sha(BASE/n)==v for n,v in summary["code_sha256"].items()),
            "current_tests_match_final_snapshot":all(sha(BASE/"tests"/n)==v for n,v in summary["test_sha256"].items()),
            "final_validation_passed":read(ROOT/"final_validation/validation.json")["passed"]}
    frozen={r["episode_index"] for r in (json.loads(l) for l in (WORKSPACE/"configs/splits/test.jsonl").read_text(encoding="utf-8").splitlines())}
    checks["no_frozen_test_episode"]=not (frozen & set(NOTES))
    totals={"native_rows":0,"decoded_images":0,"decisions":0,"wrist_consulted_decisions":0,"max_images_per_decision":0}
    for e in summary["episodes"]:
        ep=e["episode_for_audit_only"]
        records=rows(directory,ep)
        first=rows(ROOT/"run",ep)
        semantic=lambda rs:[{k:v for k,v in r.items() if k!="processing"} for r in rs]
        unchanged=semantic(records)==semantic(first)
        if ep in (8,29):
            checks[f"unseen_first_and_final_semantics_identical_{ep}"]=unchanged
        checks[f"input_files_verified_{ep}"]=all(sha(Path(p["path"]))==p["sha256"] for p in e["source_files"])
        checks[f"timestamps_valid_{ep}"]=e["source_audit"]["invalid_video_timestamps"]==0
        checks[f"all_selected_measured_fields_present_{ep}"]=all(r["modalities"][k]["status"]=="available" for r in records for k in ("observation.state","observation.velocity"))
        checks[f"all_advisory_only_{ep}"]=all(not r["execution"]["actually_executed"] and r["execution"]["execution_receipt"] is None and not r["execution"]["outcome_attributed_to_suggestion"] for r in records)
        checks[f"no_future_references_{ep}"]=all(ref["timestamp"]<=r["timestamp"] for r in records for ref in r["visual_facts"]["evidence_refs"]+r["visual_facts"]["measurements"]["joint_motion"]["evidence_refs"])
        checks[f"sample_count_{ep}"]=len(records)==(e["source_audit"]["rows_read"]+24)//25
        checks[f"complete_both_videos_{ep}"]=all(n==e["source_audit"]["rows_read"] for n in e["source_audit"]["frames_decoded"].values())
        checks[f"all_sensor_categories_consumed_{ep}"]=all(all(k in r["modalities_used"] for k in ["cam_high","observation.state[7:13]","observation.state[13]","observation.velocity[7:13]"]) for r in records)
        events=[{"timestamp":r["timestamp"],"event_id":r["failure_event"]["event_id"],"kind":r["failure_event"]["kind"],"state":r["failure_event"]["state"]} for r in records if r["notify"]]
        success=annotations[ep]["episode_success"]
        confirmed=any(r["execution_status"]=="confirmed_failure" for r in records)
        per={"episode_for_audit_only":ep,"official_outcome":"success" if success else "failure",**NOTES[ep],
             "independent_outcome_label_read_after_run":True,"unseen_at_first_run":ep in (8,29),
             "first_and_final_semantics_identical":unchanged,"full_replay":True,
             "confirmed_failure":confirmed,"failure_missed_at_episode_level":not confirmed if not success else None,
             "false_failure_alarm_on_normal":any(r["notify"] for r in records) if success else None,
             "false_alert_found_in_reviewed_moments":False,"final_task_success":e["end"]["task_success"],
             "alerts":events,"unknown_decision_timestamps":[r["timestamp"] for r in records if r["execution_status"]=="information_insufficient"],
             "counts":e["counts"],"elapsed_wall_s":e["elapsed_wall_s"],
             "decision_wall_total_s":e["decision_wall_total_s"],"decision_wall_mean_s":mean(r["processing"]["decision_wall_s"] for r in records),
             "decision_wall_max_s":e["decision_wall_max_s"],
             "wrist_consulted_decisions":sum("cam_right_wrist" in r["modalities_used"] for r in records)}
        result["per_episode"].append(per)
        totals["native_rows"]+=e["source_audit"]["rows_read"]
        totals["decoded_images"]+=sum(e["source_audit"]["frames_decoded"].values())
        totals["decisions"]+=len(records)
        totals["wrist_consulted_decisions"]+=per["wrist_consulted_decisions"]
        totals["max_images_per_decision"]=max(totals["max_images_per_decision"],e["memory_maxima"]["model_input_images"])
        for t in NOTES[ep]["times"]:
            cameras=["cam_high"] if t==0 and ep in (18,19) else ["cam_high","cam_right_wrist"]
            for c in cameras:
                name=f"trajectory_{ep:06d}/frames/obs-{t*25:06d}_{c}.jpg"
                assert sha(ROOT/"run"/name)==sha(directory/name),"reviewed picture changed"
    checks["at_most_3_images"]=totals["max_images_per_decision"]<=3
    assert all(checks.values()),checks
    result["totals"]=totals;result["checks"]=checks
    (ROOT/"review_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    sample=next(r for r in rows(directory,0) if r["timestamp"]==10.)
    (ROOT/"correction_example.json").write_text(json.dumps(sample,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"totals":totals,"checks_passed":len(checks),"wall_s":summary["elapsed_wall_s"]},ensure_ascii=False))

if __name__=="__main__":
    main()
