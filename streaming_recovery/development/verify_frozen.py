"""Offline acceptance harness; excluded from export. Failures never hidden."""
from pathlib import Path
from copy import deepcopy
from dataclasses import replace
import argparse,json,time,hashlib,traceback
from PIL import Image
from streaming_recovery.__main__ import load_config,make_vision
from streaming_recovery.engine import Monitor
from streaming_recovery.source import LeRobotStream,instruction_for_episode
from streaming_recovery.validation import build_prefix_fixture,real_decisions

def verify(directory,suffix=''):
    directory=Path(directory).resolve();config=load_config(directory/'selected_config.json')
    out=directory/('independent_checks'+suffix);out.mkdir(exist_ok=False);checks=[]
    root=Path(config['source']['root'])
    def source(ep):return LeRobotStream(root,ep,config['source']['cameras'],25,.12,512,instruction_for_episode(root,ep))
    def record(name,fn):
        tick=time.perf_counter()
        try:detail=fn();status='passed'
        except Exception:detail=traceback.format_exc();status='failed'
        checks.append({'name':name,'status':status,'detail':detail,'wall_s':time.perf_counter()-tick})
        (out/'results.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
        print(name,status,flush=True)
    def equality():
        counts={}
        for ep in config['source']['episodes']:
            def rows(name):return [{k:v for k,v in json.loads(s).items() if k!='processing'} for s in
                (directory/name/f'trajectory_{ep:06d}/decisions.jsonl').read_text(encoding='utf-8').splitlines()]
            a,b=rows('dev_selected'+suffix),rows('frozen_run'+suffix);assert a==b,f'export_mismatch:{ep}'
            counts[ep]=len(a)
        audit=json.loads((directory/('frozen_run'+suffix)/'independence_audit.json').read_text(encoding='utf-8'))
        assert audit['training_calls']==0 and not audit['imported_training_libraries'] and audit['parameters_and_config_unchanged']
        assert not audit['forbidden_file_attempts'] and not audit['forbidden_imports'] and audit['isolation_flag']==1
        return {'identical_semantic_rows':counts,'isolated_audit':'frozen_run'+suffix+'/independence_audit.json'}
    record('export_vs_development_inference_and_isolation',equality)
    def causal():
        cases=[]
        for seed,frames in ((11,375),(14,225)):
            a,b=out/f'prefix_{seed}_a',out/f'prefix_{seed}_b'
            ea,cut=build_prefix_fixture(config,a,False,seed,frames);eb,_=build_prefix_fixture(config,b,True,seed,frames)
            ra,aa,_=real_decisions(config,a,ea);rb,ab,_=real_decisions(config,b,eb)
            pa=[r for r in ra if r['timestamp']<cut];pb=[r for r in rb if r['timestamp']<cut]
            assert pa==pb and ra[-1]!=rb[-1]
            for label,rows in (('a',ra),('b',rb)):
                (out/f'prefix_{seed}_{label}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
            cases.append({'source':seed,'same_prefix_decisions':len(pa),'prefix_end_exclusive_s':cut,'reader_advanced_only_to_yield':True})
        return {'cases':cases,'fixture':'real pixels prefix plus artificial future/path/label/action changes; not additional real trajectories'}
    record('full_pipeline_future_suffix_and_label_independence',causal)
    def old_advice():
        a=Monitor(config,make_vision(config));b=Monitor(config,make_vision(config));count=0
        for obs in source(11):
            ra=a.step(deepcopy(obs));rb=b.step(deepcopy(obs));assert ra==rb
            b.recovery.observe_request({'low_level':{'recovery_phase':'unrelated_old_move_success_'+str(count)}})
            if rb['correction']:
                rb['correction']['high_level']['next_recovery_subgoal']='invented previous plan, must not feed back'
                rb['correction']['low_level']['instruction']='pretend successful execution'
            count+=1
        return {'same_decisions_with_irrelevant_old_advice':count,'advice_to_fact_writes':0}
    record('old_advice_does_not_affect_same_real_observations',old_advice)
    def long():
        cycle=list(source(0));monitor=Monitor(config,make_vision(config));maxima={}
        for i in range(2000):
            obs=cycle[i%len(cycle)];interval=deepcopy(obs.joint_interval)
            if interval:
                delta=float(i)-obs.timestamp
                interval['start_timestamp']+=delta;interval['end_timestamp']+=delta
                interval['start_ref']=f'stress-start-{i}';interval['end_ref']=f'stress-end-{i}'
            current=replace(obs,timestamp=float(i),ref=f'obs-{i:06d}',joint_interval=interval,
                modalities={k:dict(v,timestamp=float(i)) for k,v in obs.modalities.items()})
            r=monitor.step(current)
            for k,v in r['memory_usage'].items():maxima[k]=max(maxima.get(k,0),v)
        assert maxima['log_reads']==maxima['suggestion_history_items']==maxima['student_history_items']==maxima['training_updates']==0
        assert maxima['model_input_images']<=3 and maxima['student_parameter_values']==78
        return {'decisions':2000,'maxima':maxima,'source':'cycled real frames with synthetic monotonically increasing clock, not a natural long episode'}
    record('all_inference_memory_bounds_2000_dynamic_steps',long)
    def reset_faults():
        m=Monitor(config,make_vision(config))
        for obs in source(0):m.step(obs)
        m.reset();after=[m.step(o) for o in source(18)];fresh,_,_=real_decisions(config,root,18);assert after==fresh
        m.reset();initial=next(iter(source(0)));m.step(initial)
        missing=replace(initial,timestamp=1.,ref='missing',images={},joint_interval=None,
            modalities={k:dict(v,status='missing',timestamp=None) for k,v in initial.modalities.items()})
        assert m.step(missing)['execution_status']=='information_insufficient'
        black=replace(initial,timestamp=2.,ref='black',joint_interval=None,images={'cam_high':Image.new('RGB',initial.images['cam_high'].size)},
            modalities={k:dict(v,timestamp=2.) for k,v in initial.modalities.items()})
        assert m.step(black)['execution_status']=='information_insufficient'
        current=replace(initial,timestamp=3.,ref='resume',joint_interval=None,modalities={k:dict(v,timestamp=3.) for k,v in initial.modalities.items()})
        assert m.step(current)['visual_facts']['valid']
        m.reset();wrong=deepcopy(initial);wrong.modalities['observation.state']['timestamp']=1.
        try:m.step(wrong)
        except ValueError:pass
        else:raise AssertionError('future modality accepted')
        return {'cross_episode_equal_to_fresh':True,'missing_black_and_resume':True,'future_sample_rejected':True}
    record('reset_and_missing_invalid_future_modalities',reset_faults)
    def input_effect():
        # Measured modality perturbations; only engineering sensitivity, not real new trials.
        seq=list(source(14));obs=seq[8]
        def infer(mode):
            m=Monitor(config,make_vision(config))
            for item in seq[:8]:m.step(deepcopy(item))
            current=deepcopy(obs)
            if mode=='open':current.observation_state[13]=.06
            if mode=='no_wrist':current.images.pop('cam_right_wrist');current.modalities['cam_right_wrist']['status']='missing'
            if mode=='no_joint':
                current.observation_velocity=None;current.joint_interval=None;current.observation_state=None
                for name in ('observation.velocity','observation.state'):current.modalities[name]['status']='missing'
            return m.step(current)
        rows={s:infer(s) for s in ('actual','open','no_wrist','no_joint')}
        actual=rows['actual'];assert actual['visual_facts']['knife_location']=='gripper'
        assert rows['open']['visual_facts']['knife_location']!='gripper'
        assert rows['no_wrist']['visual_facts']['knife_location']!='gripper'
        assert rows['no_joint']['state']['arm_activity']=='unknown'
        return {'actual':actual['visual_facts']['knife_location'], 'changed':{k:{'relation':r['visual_facts']['knife_location'],'motion':r['state']['arm_activity']} for k,r in rows.items()},
            'note':'state13 gripper shares vector with positions; no independent third sensor claim'}
    record('actual_input_modalities_affect_decision',input_effect)
    if any(c['status']!='passed' for c in checks):raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--suffix',default='')
    args=p.parse_args();verify(args.directory,args.suffix)
