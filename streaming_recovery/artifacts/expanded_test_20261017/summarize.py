"""把真实补测、逐图核查与未通过项汇总，不把工程通过写成语义通过。"""
from pathlib import Path
from collections import Counter
import json
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from streaming_recovery.review_binding import load_run_rows, file_sha256
OUT=Path(__file__).resolve().parent
BASE=ROOT/'streaming_recovery'

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def write(p,value):
    with p.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)

def main():
    selection=read(OUT/'selection.json')
    summary=read(OUT/'run/summary.json')
    evaluation=read(OUT/'evaluation.json')
    rows=load_run_rows(OUT/'run')
    source_equal={name:file_sha256(BASE/name)==sha==summary['code_sha256'][name]
                  for name,sha in selection['source_sha256_before'].items()}
    assert all(source_equal.values()),'source was changed'
    frozen_equal=file_sha256(ROOT/'configs/splits/test.jsonl')==selection['frozen_test_sha256']
    assert frozen_equal
    old_files=read(OUT/'download_manifest_before.json')['files']
    preserved={f['path']:file_sha256(ROOT/'data/vifailback_lerobot'/f['path'])==f['sha256'] for f in old_files}
    assert all(preserved.values()),'existing data changed'
    details=[]
    for e in evaluation['per_episode']:
        ep=e['episode_for_audit_only']
        r=rows[ep]
        details.append({k:e[k] for k in ('episode_for_audit_only','official_outcome','semantic_status',
            'failure_missed_at_episode_level','false_failure_alarm_on_normal','false_task_success_on_failure',
            'unknown_decision_timestamps','unknown_stage_timestamps','elapsed_wall_s','action_counts','notification_counts')}|{
            'decisions':len(r),'final_task_success':e['final_task_success'],
            'first_completion_s':next((x['timestamp'] for x in r if x['state']['task_completed']),None),
            'first_confirmed_failure_s':next((x['timestamp'] for x in r if x['execution_status']=='confirmed_failure'),None),
            'stages':dict(Counter(x['state']['stage'] for x in r)),
            'modalities_used':dict(Counter(','.join(x['modalities_used']) for x in r)),
            'manual_gaps':[{'timestamp':c['timestamp'],'observed':c['observed'],'differences':c['differences']}
                           for c in e['manual_moments'] if c['status']!='passed'],
            'actual_execution_claims':sum(x['execution']['actually_executed'] is not False for x in r)})
    source_rows=sum(e['source_audit']['rows_read'] for e in summary['episodes'] if e['episode_for_audit_only'] in (1,6,11,14))
    new_decisions=sum(len(rows[ep]) for ep in (1,6,11,14))
    reviewed=read(OUT/'supplemental_visual_review.json')
    report={'scope':'additional four same-task dev/validation episodes plus two seen normal controls; frozen source and thresholds',
        'engineering_status':evaluation['engineering_status'],'evidence_status':evaluation['evidence_status'],
        'semantic_status':evaluation['semantic_status'],'require_complete_exit_code':int((OUT/'evaluation_exit_code.txt').read_text()),
        'totals':evaluation['totals'],'additional_only':{'episodes':[1,6,11,14],'native_rows':source_rows,
            'decoded_images':source_rows*2,'decisions':new_decisions,'failure_confirmed':[1,6],'failure_not_confirmed':[11,14]},
        'combined_with_previous_six':{'unique_episodes':[0,1,2,6,8,11,14,18,19,29],
            'normal_completion_observed':[18,19],'failure_confirmed':[0,1,2,6,8,29],'failure_not_confirmed':[11,14],
            'warning':'counts of these inspected development/validation examples; not generalization accuracy'},
        'elapsed_wall_s':summary['elapsed_wall_s'],'per_episode':details,'capability_gaps':evaluation['capability_gaps'],
        'manual_review':{'bound_checkpoints':sum(len(e['manual_moments']) for e in evaluation['per_episode']),
            'passed':sum(c['status']=='passed' for e in evaluation['per_episode'] for c in e['manual_moments']),
            'gaps':sum(c['status']=='gap' for e in evaluation['per_episode'] for c in e['manual_moments']),
            'new_images_actually_viewed':reviewed['actually_viewed_images'],'independent_human_temporal_labels':False},
        'issues_observed':[
            {'episode':11,'issue':'切割未产生可见新增小段，但只有效果未确认，没有失败事件；官方标注失败子任务4。',
             'evidence_times':[7,10,14,15],'cause_hypothesis':'当前失败类型偏向夹持/前置条件，没有操作后无切割效果的充分诊断规则。不能据此断言物理力或姿态原因。'},
            {'episode':14,'issue':'8秒实际持刀却判刀仍在架，生成错误prerequisite_unmet疑似；10秒retracted/recovered；整体未确认失败。',
             'evidence_times':[8,10,16,24],'implementation_evidence':'8秒rack_blade_bright_fraction=0.62397超过0.60门槛，腕图可用但consulted=false；当前腕图能看到刀刃与两指连接。',
             'additional_gap':'24秒顶视刀已回架但模型位置unknown；官方失败子任务3未获得有效诊断。'}],
        'no_code_or_threshold_changes':True,'frozen_test_split_unchanged':frozen_equal,'source_sha256_matches_frozen':source_equal,
        'prior_local_data_hashes_preserved':preserved,'physical_execution':False,'generalization_claim':False,'detection_latency':None,
        'validation_scope':'本轮未改业务代码，运行6条真实回放及严格离线容量/时间引用/完整读取/输入哈希/动作一致性/核查绑定检查；上轮108项和5组全链验收未重复运行。'}
    write(OUT/'test_summary.json',report)
    artifacts={p.relative_to(OUT).as_posix():file_sha256(p) for p in OUT.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.txt','.py')}
    write(OUT/'delivery_manifest.json',{'source_unchanged':all(source_equal.values()),'frozen_test_unchanged':frozen_equal,
        'source_sha256':summary['code_sha256'],'input_hashes_reference':'run/summary.json: episodes[].source_files',
        'artifacts_sha256':artifacts,'report_sha256':file_sha256(ROOT/'docs/补充小规模测试_20261017.md')})
    print(json.dumps({'engineering':report['engineering_status'],'semantic':report['semantic_status'],'additional':report['additional_only'],
                      'wall_s':report['elapsed_wall_s'],'manual_review':report['manual_review']},ensure_ascii=False))

if __name__=='__main__':
    main()
