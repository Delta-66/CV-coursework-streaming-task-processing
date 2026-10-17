"""离线整理本轮交付，不改变推理记录与历史报告。"""
from pathlib import Path
import difflib
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from streaming_recovery.evaluation import parse_outcome

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'streaming_recovery'
OUT=Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    with path.open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def main():
    run=OUT/'accepted_run';summary=read(run/'summary.json')
    validation=read(OUT/'validation/validation.json');acceptance=read(OUT/'validation/real_acceptance.json')
    assert validation['passed'] and acceptance['overall_status']=='passed'
    source_match={name:sha(BASE/name)==value for name,value in summary['code_sha256'].items()}
    test_match={name:sha(BASE/'tests'/name)==value for name,value in summary['test_sha256'].items()}
    assert all(source_match.values()) and all(test_match.values()),'final snapshot drift'
    before=read(OUT/'before/sha256.json')
    candidates={Path(p) for p in before}
    candidates.update(p.relative_to(ROOT) for p in BASE.glob('*.py'))
    candidates.update(p.relative_to(ROOT) for p in (BASE/'tests').glob('test_*.py'))
    candidates.add(Path('docs/增量修复与验收_20261017.md'))
    candidates.update(p.relative_to(ROOT) for p in (OUT/'build_delivery.py',OUT/'bind_manual_review.py'))
    changes=[];patch=[]
    for rel in sorted(candidates,key=str):
        old=OUT/'before'/rel;new=ROOT/rel
        if not new.is_file():continue
        old_text=old.read_text(encoding='utf-8') if old.is_file() else ''
        new_text=new.read_text(encoding='utf-8')
        if old_text==new_text:continue
        changes.append({'path':rel.as_posix(),'before_sha256':sha(old) if old.is_file() else None,'after_sha256':sha(new)})
        patch+=difflib.unified_diff(old_text.splitlines(True),new_text.splitlines(True),
                                   fromfile='a/'+rel.as_posix() if old.is_file() else '/dev/null',tofile='b/'+rel.as_posix())
    (OUT/'incremental.patch').write_text(''.join(patch),encoding='utf-8')
    examples=[]
    for ep,t in ((18,20),(19,8),(19,16),(0,10),(2,2),(2,7),(8,8),(29,9)):
        row=next(r for r in map(json.loads,(run/f'trajectory_{ep:06d}/decisions.jsonl').read_text(encoding='utf-8').splitlines()) if r['timestamp']==t)
        examples.append({'episode_for_audit_only':ep,'decision':row})
    write(OUT/'real_examples.json',{'real_decisions':examples,'conditional_direction_checks':acceptance['conditional_direction_checks'],
                                  'warning':'条件化方向不是自然失败事件后的执行；所有建议未执行。'})
    data=ROOT/'data/vifailback_lerobot';info=read(data/'meta/info.json')
    import pyarrow.parquet as pq
    fields={}
    for ep in (18,19,0,2,8,29):
        f=pq.ParquetFile(data/f'data/chunk-000/episode_{ep:06d}.parquet')
        row=next(f.iter_batches(batch_size=1,columns=['timestamp','observation.state','observation.velocity'])).to_pylist()[0]
        fields[str(ep)]={'parquet_rows':f.metadata.num_rows,'first_timestamp':row['timestamp'],
                         'state_length':len(row['observation.state']),'velocity_length':len(row['observation.velocity'])}
    write(OUT/'environment_and_data.json',{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
        'cpu_logical_cores':os.cpu_count(),'backend':'CPU OpenCV; no learned weights/API/GPU invoked',
        'packages':{name:importlib.metadata.version(name) for name in ('numpy','opencv-python','Pillow','pyarrow','av')},
        'data_root':str(data),'codebase_version':info['codebase_version'],'fps':info['fps'],
        'actual_fields':fields,'relationship':'qpos and gripper share observation.state; qvel measured, action/target EEF unused',
        'units':'source raw qpos/qvel/gripper units; physical calibration unavailable',
        'frames':acceptance['totals'],'frozen_split_sha256':acceptance['frozen_test_split_sha256']})
    old=read(OUT/'review_summary_v2_final.json')
    before_semantic={str(e['episode_for_audit_only']):e for e in old['per_episode']}
    comparisons=[]
    for e in acceptance['per_episode']:
        prior=before_semantic[str(e['episode_for_audit_only'])]
        comparisons.append({'episode':e['episode_for_audit_only'],'official_outcome':parse_outcome(e['official_outcome']).value,
            'previous_completion':prior['final_task_success'],'current_completion':e['final_task_success'],
            'previous_localized_decisions':prior['joint_localization_decisions'],'current_localized_decisions':e['joint_localization_decisions'],
            'previous_unknown_times':prior['unknown_decision_timestamps'],'current_unknown_times':e['unknown_decision_timestamps'],
            'previous_corrections':prior['action_counts'],'current_corrections':e['action_counts']})
    write(OUT/'before_after.json',{'kind':'real six-trajectory regression; no unseen/generalization claim','per_episode':comparisons})
    write(OUT/'delivery_manifest.json',{'modified_files':changes,'patch_sha256':sha(OUT/'incremental.patch'),
        'run_source_hash_matches_current':source_match,'run_test_hash_matches_current':test_match,
        'frozen_test_split_sha256':acceptance['frozen_test_split_sha256'],
        'frozen_split_unchanged_from_pre_repair_audit':old['frozen_test_split_sha256']==acceptance['frozen_test_split_sha256'],
        'engineering_status':validation['engineering_status'],'selected_semantic_acceptance':validation['semantic_status'],
        'unit_tests':validation['checks'][0]['detail'],'validation_groups':len(validation['checks']),
        'selected_samples_exposure':'all six seen regression; no frozen test samples',
        'run_summary_sha256':sha(run/'summary.json'),'manual_review_sha256':sha(OUT/'manual_review.json'),
        'acceptance_report_sha256':sha(OUT/'validation/real_acceptance.json'),'validation_report_sha256':sha(OUT/'validation/validation.json'),
        'historical_wrong_report_preserved_sha256':sha(BASE/'artifacts/multimodal_20261017/review_summary.json'),
        'actual_execution':False,'detection_latency_reported':False,'generalization_claim':False})
    print(json.dumps({'changed_files':len(changes),'source_snapshot_matches':all(source_match.values()),
                      'tests':validation['checks'][0]['detail']['tests_run'],'validation_groups':len(validation['checks'])}))


if __name__=='__main__':main()
