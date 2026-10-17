"""只读复核原项目边界，不修改旧实现。"""
import json
from pathlib import Path
from run_task_monitor import sampled_frame
from current_observation import validate_result
from diagnostic_tracking import DiagnosticTracker


def old_result(frame,kind=None,grasp="unknown",stage="grasping"):
    failure=kind is not None
    return {"stage":stage,"status":"failure" if failure else "in_progress","failure":failure,
        "evidence":["observed object still on support"],
        "high_level_advice":"Restore the grasp before transport" if failure else None,
        "low_level_advice":{"action":"Continue cutting: move -0.1 m in the robot base frame","reference_frame":"image","completion_check":"Check completion"} if failure else None,
        "key_states":{"grasp":grasp,"target_relation":"unknown","task_goal":"unknown"},
        "expected_effect":"object held","observed_effect":"object still on support","action_outcome":"not_achieved" if failure else "uncertain",
        "failure_type":kind or "none","evidence_frame_indices":[frame]}


def main():
    report={}
    monitor={"sample_every":25,"include_last_frame":True}
    report["sampling_depends_on_total_length"]={"frame_index":7,"length8":sampled_frame(7,8,monitor),"length100":sampled_frame(7,100,monitor)}
    fields=("stage","status","failure","evidence","high_level_advice","low_level_advice")
    row=old_result(0,"grasp_miss","not_held")
    try:
        validate_result({k:row[k] for k in fields})
        report["format_validator_accepts_semantically_inconsistent_advice"]=True
    except ValueError:
        report["format_validator_accepts_semantically_inconsistent_advice"]=False
    tracker=DiagnosticTracker(stage_confirmation=1,failure_confirmation=1,recovery_confirmation=2,cooldown_s=0)
    timeline=[]
    for i,kind,grasp in [(0,"grasp_miss","not_held"),(1,None,"not_held"),(2,None,"not_held"),(3,"grasp_miss","not_held")]:
        result=tracker.update(0,i,float(i),old_result(i,kind,grasp))
        timeline.append({"frame":i,"result":result})
    report["retry_observation_boundary"]={"note":"旧 tracker 没有尝试编号/夹爪重新打开输入；未恢复期间的同类失败仍归入原 incident。此探针不冒充实际新重试已执行。","timeline":timeline}
    path=Path(__file__).resolve().parent/"artifacts/existing_review.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="retry_observation_boundary"},ensure_ascii=False))


if __name__=="__main__":
    main()
