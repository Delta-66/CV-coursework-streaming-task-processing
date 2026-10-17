"""比较同一六条真实回放的核心决策，另列新增测量/证据说明；仅离线审计。"""
from pathlib import Path
import json
from collections import Counter

OUT=Path(__file__).resolve().parent
BASE=OUT.parents[1]
FIELDS=('stage','preconditions','candidate','attempt','attempt_held','attempt_context','new_attempt_observed',
        'task_completed','historical_completion','observable','modality_conflict','recovered','unknowns','arm_activity')

def read_rows(run,ep):
    return [json.loads(line) for line in (run/f'trajectory_{ep:06d}/decisions.jsonl').read_text(encoding='utf-8').splitlines()]

def semantic(r):
    return {key:r[key] for key in ('timestamp','observation_ref','execution_status','failure_event','notify','correction')} | {
        'state':{key:r['state'][key] for key in FIELDS}}

def main():
    old=BASE/'artifacts/repair_acceptance_20261017/accepted_run'
    new=OUT/'final_run'
    summary=json.loads((new/'summary.json').read_text(encoding='utf-8'))
    data=[]
    for episode in summary['episodes']:
        ep=episode['episode_for_audit_only']
        a,b=read_rows(old,ep),read_rows(new,ep)
        diffs=[{'timestamp':r['timestamp'],'old':semantic(p),'new':semantic(r)} for p,r in zip(a,b) if semantic(p)!=semantic(r)]
        details=[{'timestamp':r['timestamp'],'fields':sorted(k for k in set(p['state'])|set(r['state']) if p['state'].get(k)!=r['state'].get(k))} for p,r in zip(a,b)]
        last=b[-1]
        data.append({'episode':ep,'same_decision_count':len(a)==len(b),'core_semantic_differences':diffs,'state_changed_fields':details,
            'wall_s':episode['elapsed_wall_s'],'decisions':len(b),'unknown_timestamps':[r['timestamp'] for r in b if r['state']['stage']=='unknown'],
            'completion_timestamps':[r['timestamp'] for r in b if r['state']['task_completed']],
            'notifications':[{'t':r['timestamp'],'event':r['failure_event']['event_id'],'state':r['failure_event']['state']} for r in b if r['notify']],
            'attempt_timestamps':[r['timestamp'] for r in b if r['state']['new_attempt_observed']],
            'final_effect_association':last['state']['cut_effect_order'],
            'cut_confirmation_timestamps':[r['timestamp'] for r in b if r['state']['preconditions']['cut_effect_confirmed']],
            'closed_decisions':sum(r['visual_facts']['gripper']=='closed' for r in b),
            'closed_joint_localizations':sum(r['visual_facts']['gripper']=='closed' and r['visual_facts']['knife_handle_xy'] is not None and r['visual_facts']['gripper_xy'] is not None for r in b),
            'modality_patterns':dict(Counter(','.join(r['modalities_used']) for r in b)),
            'memory_maxima':episode['memory_maxima']})
    report={'selected_core_state_fields':FIELDS,'total_wall_s':summary['elapsed_wall_s'],'per_episode':data}
    with (OUT/'final_run_inspection.json').open('x',encoding='utf-8') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'wall_s':report['total_wall_s'],'episodes':[{k:v for k,v in x.items() if k in ('episode','core_semantic_differences','wall_s','completion_timestamps','notifications','attempt_timestamps')} for x in data]},ensure_ascii=False))
    assert all(x['same_decision_count'] and not x['core_semantic_differences'] for x in data)

if __name__=='__main__':
    main()
