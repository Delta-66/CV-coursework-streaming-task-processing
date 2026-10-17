"""Offline report built from actual runs and explicitly reviewed image checkpoints."""
from pathlib import Path
import sys,json,hashlib,argparse
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from streaming_recovery.review_binding import row_fingerprint,file_sha256,BINDING_VERSION,FINGERPRINT_VERSION,FINGERPRINT_DEFINITION,validate_review_bindings,rebind_review
from streaming_recovery.evaluation import evaluate_run
from streaming_recovery.fusion import FixedFusion
import numpy as np
parser=argparse.ArgumentParser();parser.add_argument('--suffix',default='');suffix=parser.parse_args().suffix
P=Path(__file__).resolve().parent;run=P/('frozen_run'+suffix)
def output(name):
    name=Path(name);return P/(name.stem+suffix+name.suffix)
summary=json.loads((run/'summary.json').read_text(encoding='utf-8'))
rows={e['episode_for_audit_only']:list(map(json.loads,(run/e['output_subdir']/'decisions.jsonl').read_text(encoding='utf-8').splitlines())) for e in summary['episodes']}
review={'review_kind':'助手逐图审核；非独立人工首次检测时刻标注；保留真实缺口',
    'source_run':str(run),'binding_version':BINDING_VERSION,'row_fingerprint_version':FINGERPRINT_VERSION,
    'row_fingerprint_definition':FINGERPRINT_DEFINITION,'source_run_summary_sha256':file_sha256(run/'summary.json'),'checkpoints':[]}
def add(ep,t,observed,expected):
    r=next(r for r in rows[ep] if r['timestamp']==t)
    images=[f"trajectory_{ep:06d}/frames/{r['observation_ref']}_{cam}.jpg" for cam in ('cam_high','cam_right_wrist')]
    review['checkpoints'].append({'episode_for_audit_only':ep,'timestamp':t,'observation_ref':r['observation_ref'],
        'observed':observed,'expected_fields':expected,'image_refs':images,'image_sha256':{im:file_sha256(run/im) for im in images},
        'row_sha256':row_fingerprint(r),'row_fingerprint_version':FINGERPRINT_VERSION})
for ep,ts in {0:(9,10),1:(13,14),2:(10,11),6:(9,10),8:(7,8),29:(8,9)}.items():
    # ep6@9 wasn't directly displayed: only reviewed @10 is included.
    for i,t in enumerate(ts):
        if ep==6 and t==9:continue
        add(ep,t,'刀仍在架、机械手空手进入砧板；腕图无持刀连接',{'visual_facts.knife_location':'holder','execution_status':'suspicious' if i==0 else 'confirmed_failure'})
add(6,3,'接近刀具阶段，尚无抓取成败依据',{'failure_event':None})
for t in (7,8):add(11,t,'腕图刀在两指间且刀—香肠相对关系变化；尚未见新分离效果',{'visual_facts.knife_location':'gripper','state.task_completed':False})
for t,status in ((14,'suspicious'),(15,'confirmed_failure')):add(11,t,'机械手退出工作区，顶视香肠仍完整；与此前工作区观测构成效果偏差',{'execution_status':status,'visual_facts.cut_piece':'no'})
add(14,5,'持刀移向砧板，不能因靠近刀架就判断归位',{'state.stage':'transport_tool'})
add(14,8,'腕图刀刃位于两指之间；顶视刀架亮度不能否定持刀',{'visual_facts.knife_location':'gripper','state.preconditions.reliable_grasp':True,'failure_event':None})
add(14,16,'仍持刀在工作区；此时不要求已经完成切割',{'failure_event':None})
for t,status in ((17,'suspicious'),(18,'confirmed_failure')):add(14,t,'退出工作区后仍无新分离小段',{'execution_status':status,'visual_facts.cut_piece':'no'})
add(14,24,'刀可见于刀架，香肠仍完整；系统目前未恢复该归位关系',{'visual_facts.knife_location':'holder'})
add(18,5,'刀离架向砧板移动',{'visual_facts.knife_location':'gripper','state.stage':'transport_tool'})
add(18,10,'刀在香肠端部附近作用；未等同于切开',{'state.stage':'cut_operation','state.task_completed':False})
add(18,14,'顶视新增圆片，刀仍在夹爪',{'visual_facts.cut_piece':'yes'})
add(18,20,'新圆片仍可见，刀归架且两指退出',{'state.task_completed':True,'notify':False})
for t in (15,16):
    for cam in ('cam_high','cam_right_wrist'):
        name=f'obs-{25*t:06d}_{cam}.jpg'
        assert file_sha256(run/'trajectory_000019/frames'/name)==file_sha256(P/'baseline_fit/trajectory_000019/frames'/name)
add(19,16,'与训练审核时相同字节图像：新切片已出现，刀已归架',{'state.task_completed':True,'notify':False})
add(13,6,'持刀进入工作区，刀与香肠有作用关系，尚无新小块',{'visual_facts.knife_location':'gripper','failure_event':None})
add(13,12,'最后采样退出工作区，香肠完整；仅一个清晰退出后时刻，尚不足重复确认',{'execution_status':'suspicious','state.task_completed':False})
add(16,6,'持刀工作区阶段，尚未达到完成条件',{'visual_facts.knife_location':'gripper','failure_event':None})
for t,status in ((13,'suspicious'),(14,'confirmed_failure')):add(16,t,'工具退出而香肠仍完整；两次当前顶视证据',{'execution_status':status,'visual_facts.cut_piece':'no'})
old=json.loads((P.parent/'freeze_acceptance_20261017/manual_review.json').read_text(encoding='utf-8'))
old['checkpoints']=[]
rebound=rebind_review(old,old['source_run'],run,'root assistant explicit evidence rebinding',
    'Preserve two previously pixel-reviewed image-direction expectations; verify identical JPEG SHA and rebind full new decision rows; not natural retry.')
review['conditional_directions']=rebound['conditional_directions']
review['conditional_direction_rebind_audit']=rebound['binding_rebind']
output('manual_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8')
assert validate_review_bindings(review,rows,run)['valid']
cfg=json.loads((run/'config_snapshot.json').read_text(encoding='utf-8'))
report=evaluate_run(cfg,run,review)
output('evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
assert report['engineering_status']=='passed'
model=FixedFusion(P/('frozen_runtime'+suffix)/'fusion.json')
features=next(r for r in rows[14] if r['timestamp']==8)['visual_facts']['measurements']['learned_relation']['features']
label,score=model.predict(np.asarray(features));assert label=='gripper'
learning={'parameter_effect_at_reviewed14_at8':{'trained_label':label,'trained_score':score.tolist(),
    'zero_weight_softmax':[1/3]*3,'zero_weight_label_with_frozen_threshold':'unknown'},
    'parameters_readonly':all(not getattr(model,k).flags.writeable for k in ('mean','scale','coef','intercept')),
    'selected_model_sha256':model.sha256,'fit_bindings_checked':0}
training=json.loads((P/'reviewed_samples.json').read_text(encoding='utf-8'))
for sample in training['samples']:
    ep=sample['episode_for_grouping_only'];t=sample['timestamp']
    records=list(map(json.loads,(P/'baseline_fit'/f'trajectory_{ep:06d}'/'decisions.jsonl').read_text(encoding='utf-8').splitlines()))
    row=next(r for r in records if r['timestamp']==t)
    assert hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=False).encode()).hexdigest()==sample['prediction_source_sha256']
    for image in sample['images']:assert file_sha256(image['path'])==image['sha256']
    learning['fit_bindings_checked']+=1
output('learning_evidence.json').write_text(json.dumps(learning,ensure_ascii=False,indent=2),encoding='utf-8')
protocol=json.loads((P/'protocol.json').read_text(encoding='utf-8'))
for split,sha in protocol['split_sha256'].items():assert file_sha256(ROOT/f'configs/splits/{split}.jsonl')==sha
table=[]
for e in report['per_episode']:
    ep=e['episode_for_audit_only'];r=rows[ep];bname='baseline_audit' if ep in (13,16) else 'baseline_regression'
    bs=json.loads((P/bname/'summary.json').read_text(encoding='utf-8'));be=next(e for e in bs['episodes'] if e['episode_for_audit_only']==ep)
    br=list(map(json.loads,(P/bname/be['output_subdir']/'decisions.jsonl').read_text(encoding='utf-8').splitlines()))
    rr=next(e for e in summary['episodes'] if e['episode_for_audit_only']==ep)
    table.append({'episode':ep,'role':'reserved_train_audit_after_selection' if ep in (13,16) else 'fit_normal_control' if ep==19 else 'previously_exposed_regression',
        'official_outcome':e['official_outcome'],'baseline_confirmed':any(x['execution_status']=='confirmed_failure' for x in br),
        'final_confirmed':e['confirmed_failure'],'final_success':e['final_task_success'],'missed_failure':e['failure_missed_at_episode_level'],
        'normal_false_alarm':e['false_failure_alarm_on_normal'],'confirm_timestamp_not_delay':[x['timestamp'] for x in r if x['notification_reason']=='evidence_confirmed'],
        'alerts':e['notification_counts'],'unknown_stage_timestamps':e['unknown_stage_timestamps'],
        'unknown_information_timestamps':e['unknown_decision_timestamps'],
        'baseline_wall_s':be['elapsed_wall_s'],'final_wall_s':rr['elapsed_wall_s'],
        'decision_wall_total_s':rr['decision_wall_total_s'],'decision_wall_max_s':rr['decision_wall_max_s'],
        'baseline_wrist_calls':sum('cam_right_wrist' in x['modalities_used'] for x in br),
        'final_wrist_calls':sum('cam_right_wrist' in x['modalities_used'] for x in r),
        'student_forward_calls':sum('learned_relation' in x['visual_facts']['measurements'] for x in r),
        'semantic_gaps':e['capability_gaps']})
output('comparison.json').write_text(json.dumps({'episodes':table,'totals':report['totals'],'immutable_splits_verified':True,
    'frozen_test_used':False,'no_detection_delay_annotation':True,'no_generalization_claim':True,
    'budget':'same25Hzsource1Hzdecisions max3images; baselineadaptivewrist finalalwayscurrentwristwhengeometryvalid',
    'timing':'actual measured wall time; frozen run has audit profiler, not a controlled speed benchmark'},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'engineering':report['engineering_status'],'semantic':report['semantic_status'],'checkpoints':len(review['checkpoints']),'gaps':report['capability_gaps'],'totals':report['totals']},ensure_ascii=False))
