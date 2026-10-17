"""离线冻结清单与相对本轮起点的补丁；不修改旧结果或在线判断。"""
from pathlib import Path
import difflib
import hashlib
import importlib.metadata
import json
import platform
import sys
from inspect_final_run import semantic

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'streaming_recovery'
OUT=Path(__file__).resolve().parent

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(p,value):
    with p.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)

def rows(run,ep):
    return [json.loads(line) for line in (run/f'trajectory_{ep:06d}/decisions.jsonl').read_text(encoding='utf-8').splitlines()]

def main():
    run=OUT/'final_run'
    previous=BASE/'artifacts/repair_acceptance_20261017/accepted_run'
    summary=read(run/'summary.json')
    validation=read(OUT/'validation/validation.json')
    acceptance=read(OUT/'validation/real_acceptance.json')
    assert validation['passed'] and acceptance['overall_status']=='passed'
    source_match={name:sha(BASE/name)==value for name,value in summary['code_sha256'].items()}
    test_match={name:sha(BASE/'tests'/name)==value for name,value in summary['test_sha256'].items()}
    assert all(source_match.values()) and all(test_match.values()),'final snapshot drift'
    before=read(OUT/'before/sha256.json')
    original_tests={p:sha(ROOT/p)==digest for p,digest in before.items() if '/tests/test_' in p}
    assert all(original_tests.values()),'original tests modified'
    candidates={Path(p) for p in before}
    candidates.update(p.relative_to(ROOT) for p in BASE.glob('*.py'))
    candidates.update(p.relative_to(ROOT) for p in (BASE/'tests').glob('test_*.py'))
    candidates.add(Path('docs/冻结修复与验收_20261017.md'))
    candidates.update(p.relative_to(ROOT) for p in (OUT/'build_delivery.py',OUT/'inspect_final_run.py'))
    changes=[]
    patch=[]
    for rel in sorted(candidates,key=str):
        old=OUT/'before'/rel
        new=ROOT/rel
        if not new.is_file():
            continue
        a=old.read_text(encoding='utf-8') if old.is_file() else ''
        b=new.read_text(encoding='utf-8')
        if a==b:
            continue
        changes.append({'path':rel.as_posix(),'before_sha256':sha(old) if old.is_file() else None,'after_sha256':sha(new)})
        patch.extend(difflib.unified_diff(a.splitlines(True),b.splitlines(True),
                     fromfile='a/'+rel.as_posix() if old.is_file() else '/dev/null',tofile='b/'+rel.as_posix()))
    with (OUT/'incremental.patch').open('x',encoding='utf-8') as stream:
        stream.write(''.join(patch))
    examples=[]
    for ep,t in ((18,9),(18,14),(18,15),(18,20),(19,8),(19,10),(19,11),(19,16),(0,10),(2,2),(2,3),(2,11),(8,8),(29,9)):
        row=next(r for r in rows(run,ep) if r['timestamp']==t)
        examples.append({'episode_for_audit_only':ep,'decision':row})
    write(OUT/'real_examples.json',{'real_decisions':examples,'conditional_direction_checks':acceptance['conditional_direction_checks'],
        'warning':'已见固定场景；方向为条件化交接；没有执行任何恢复动作；首变时间是首次观察变化而非精确物理发生时刻。'})
    comparisons=[]
    for episode in summary['episodes']:
        ep=episode['episode_for_audit_only']
        a,b=rows(previous,ep),rows(run,ep)
        differences=[{'timestamp':new['timestamp'],'old':semantic(old),'new':semantic(new)} for old,new in zip(a,b) if semantic(old)!=semantic(new)]
        assert len(a)==len(b) and not differences,f'core semantic regression: {ep}'
        comparisons.append({'episode':ep,'decisions':len(b),'core_semantic_equal':True,'core_semantic_differences':differences,
            'new_attempt_timestamps':[r['timestamp'] for r in b if r['state']['new_attempt_observed']],
            'completion_timestamps':[r['timestamp'] for r in b if r['state']['task_completed']],
            'confirmation_timestamps':[r['timestamp'] for r in b if r['notify'] and r['failure_event'] and r['failure_event']['state']=='confirmed'],
            'old_decisions_sha256':sha(previous/f'trajectory_{ep:06d}/decisions.jsonl'),
            'new_decisions_sha256':sha(run/f'trajectory_{ep:06d}/decisions.jsonl')})
    write(OUT/'before_after.json',{'scope':'compare core state fields enumerated in final_run_inspection.json, status, event, notify and correction; timing/measurement/capacity additions and updated evidence wording excluded; all changed state keys retained in final_run_inspection.json',
                                'per_episode':comparisons,'old_run_preserved':str(previous)})
    write(OUT/'environment.json',{'executable':sys.executable,'python':sys.version,'platform':platform.platform(),
        'backend':'CPU OpenCV classical vision; no new learned model/weights/API/GPU',
        'packages':{name:importlib.metadata.version(name) for name in ('numpy','opencv-python','Pillow','pyarrow','av')},
        'input_facts_reference':'../repair_acceptance_20261017/environment_and_data.json',
        'real_sensor_relation':'qpos and gripper share observation.state; qvel measured; action/target EEF unused',
        'units':'source raw values; no physical gripper/coordinate calibration',
        'input_hashes_reference':'final_run/summary.json: episodes[].source_files',
        'config_sha256':sha(BASE/'config_multimodal.json'),'config_snapshot_sha256':sha(run/'config_snapshot.json')})
    before_split=read(OUT/'starting_facts.json')['frozen_test_split_sha256']
    assert before_split==acceptance['frozen_test_split_sha256']
    artifacts={p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*')
               if p.is_file() and p.suffix in ('.json','.jsonl','.txt','.patch') and 'before' not in p.relative_to(OUT).parts}
    write(OUT/'delivery_manifest.json',{'modified_files':changes,'patch_sha256':sha(OUT/'incremental.patch'),
        'run_source_hash_matches_current':source_match,'run_test_hash_matches_current':test_match,
        'original_test_files_unchanged':original_tests,'frozen_test_split_sha256':before_split,'frozen_test_split_unchanged':True,
        'engineering_status':validation['engineering_status'],'selected_semantic_acceptance':validation['semantic_status'],
        'unit_tests':validation['checks'][0]['detail'],'validation_groups':len(validation['checks']),
        'sample_exposure':'all six seen regression; no frozen test samples','actual_execution':False,
        'detection_latency_reported':False,'generalization_claim':False,'artifacts_sha256':artifacts})
    print(json.dumps({'changed_files':len(changes),'source_snapshot_matches':all(source_match.values()),
                      'tests':validation['checks'][0]['detail']['tests_run'],'validation_groups':len(validation['checks'])}))

if __name__=='__main__':
    main()
