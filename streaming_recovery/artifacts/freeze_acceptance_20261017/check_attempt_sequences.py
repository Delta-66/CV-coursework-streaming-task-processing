"""旧六条真实日志的raw/当前目标关系快检；不是新六条端到端回放。"""
import hashlib,json
from pathlib import Path
from streaming_recovery.attempts import attempt_transition

OUT=Path(__file__).resolve().parent
RUN=OUT.parent/'repair_acceptance_20261017/accepted_run'
config=json.loads((RUN/'config_snapshot.json').read_text(encoding='utf8'))
results=[]
for episode in config['source']['episodes']:
    path=RUN/f'trajectory_{episode:06d}/decisions.jsonl'
    rows=[json.loads(line)for line in path.read_text(encoding='utf8').splitlines()]
    armed=False;last_time=None;attempt_times=[]
    for r in rows:
        f=r['visual_facts'];t=r['timestamp']
        good=f['valid'] and f['effect_usable'] and f['visibility'] not in ('occluded','unknown')
        near=good and (f['motion'] in ('at_holder','approaching_holder') or
             f['knife_location']in ('board','elsewhere') and f['knife_handle_xy']is not None)
        state=r['observation_inputs']['observation.state']
        raw=state[13] if state is not None and r['modalities']['observation.state']['status']=='available' else None
        armed,edge,new=attempt_transition(armed,raw,f['gripper'],f['gripper_source']=='independent_visual',
              good,last_time is not None and t-last_time>config['rules']['max_evidence_gap_s'],near,config['rules'])
        if new:attempt_times.append(t)
        last_time=t
    prior=[r['timestamp']for r in rows if r['state']['new_attempt_observed']]
    results.append({'episode_for_audit_only':episode,'source_decisions_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'old_attempt_timestamps':prior,'new_latch_attempt_timestamps':attempt_times,
                    'matches':prior==attempt_times})
result={'kind':'offline_replay_of_existing_real_raw_and_visual_facts_not_new_visual_inference',
        'source_run':str(RUN.resolve()),'per_episode':results,'all_match':all(r['matches']for r in results)}
(OUT/'attempt_real_raw_regression.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(result,ensure_ascii=False,indent=2))
