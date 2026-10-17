"""核查本轮实际交付文件。只读日志，绝不由在线推理调用。"""
from pathlib import Path
import hashlib
import json
import re

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
ROOT=BASE.parent
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    run=HERE/"delivery_run"
    summary=json.loads((run/"summary.json").read_text(encoding="utf8"))
    validation=json.loads((HERE/"delivery_validation/validation.json").read_text(encoding="utf8"))
    assert validation["passed"]
    for name,h in summary["code_sha256"].items():
        assert sha(BASE/name)==h,("current code mismatch",name)
        assert sha(run/"source_snapshot"/name)==h,("snapshot mismatch",name)
    for name,h in summary["test_sha256"].items():
        assert sha(BASE/"tests"/name)==h,("test mismatch",name)
    selection=json.loads((HERE/"additional_selection.json").read_text(encoding="utf8"))
    for name,h in selection["split_hashes"].items():
        assert sha(ROOT/"configs/splits"/name)==h,("split changed",name)
    records=[]
    for e in summary["episodes"]:
        rows=[json.loads(s) for s in (run/e["output_subdir"]/"decisions.jsonl").read_text(encoding="utf8").splitlines()]
        ep=e["episode_for_audit_only"]
        assert len(rows)==e["counts"]["decisions"]
        assert all(not r["execution"]["actually_executed"] and r["execution"]["execution_receipt"] is None for r in rows)
        assert all(all(i["timestamp"]<=r["timestamp"] for i in r["visual_facts"]["evidence_refs"]) for r in rows)
        for f in e["source_files"]:
            assert sha(Path(f["path"]))==f["sha256"]
        if ep in (18,19):
            assert rows[-1]["state"]["task_completed"] and not any(r["notify"] for r in rows)
        else:
            assert any(r["execution_status"]=="confirmed_failure" for r in rows)
            assert e["end"]["task_success"] is None
        records.append({"episode":ep,"decisions":len(rows),"current_complete":rows[-1]["state"]["task_completed"]})
    for p in (ROOT/"README.md",BASE/"README.md",BASE/"docs/增量修复与验收_20261017.md"):
        for target in re.findall(r"\]\(([^)]+)\)",p.read_text(encoding="utf8")):
            if target.startswith(("http:","https:","#")):
                continue
            assert (p.parent/target).resolve().exists(),("missing documentation link",p,target)
    frozen=json.loads((HERE/"rule_freeze_v2.json").read_text(encoding="utf8"))
    delta=[name for name,h in frozen["frozen_inference_sha256"].items() if sha(BASE/name)!=h]
    assert set(delta)=={"recovery.py","tracking.py"},delta
    result={"passed":True,"code_and_test_hashes_match":True,"source_snapshot_matches":True,
        "data_hashes_match":True,"split_hashes_unchanged":True,"documentation_links_exist":True,
        "final_visual_config_matches_second_freeze":True,"post_freeze_logic_changes":delta,
        "logic_change_reason":"synthetic regressions: reject undeclared parameters/unverified preconditions; end explicitly released connection",
        "total_decisions":sum(r["decisions"] for r in records),"records":records}
    (HERE/"delivery_audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps(result,ensure_ascii=False))
if __name__=="__main__":
    main()
