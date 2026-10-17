"""Offline observed feedback. Future observations audit predictions, never act as causal rewards."""
from pathlib import Path
import argparse,json
from ..review_binding import row_fingerprint,file_sha256

def collect(run,data_root,output):
    run,data_root,output=Path(run),Path(data_root),Path(output)
    summary=json.loads((run/'summary.json').read_text(encoding='utf-8'))
    annotations={r['episode_index']:r for r in map(json.loads,(data_root/'meta/vifailback_annotations.jsonl').read_text(encoding='utf-8').splitlines())}
    windows=[]
    for ep in summary['episodes']:
        episode=ep['episode_for_audit_only']
        rows=[json.loads(x) for x in (run/ep['output_subdir']/'decisions.jsonl').read_text(encoding='utf-8').splitlines()]
        final=annotations[episode]['episode_success']
        missed=final=='failure' and not any(r['execution_status']=='confirmed_failure' for r in rows)
        false=final=='success' and any(r['notify'] and r['execution_status'] in ('suspicious','confirmed_failure') for r in rows)
        for i,r in enumerate(rows):
            reasons=[]
            if r['execution_status']=='information_insufficient':reasons.append('uncertain_current_state')
            if missed and i>=len(rows)-3:reasons.append('episode_failure_without_confirmation_review_only')
            if false and r['notify']:reasons.append('normal_trajectory_alert_review_only')
            if r['correction']:reasons.append('audit_proposed_post_checks')
            if not reasons:continue
            refs=[f"{ep['output_subdir']}/frames/{r['observation_ref']}_{c}.jpg" for c in ('cam_high','cam_right_wrist')]
            windows.append({'episode_for_development_only':episode,'timestamp':r['timestamp'],'observation_ref':r['observation_ref'],
                'row_sha256':row_fingerprint(r),'image_sha256':{p:file_sha256(run/p) for p in refs if (run/p).exists()},
                'reasons':reasons,'current_stage':r['state']['stage'],'current_facts':r['visual_facts'],
                'current_diagnosis':r['diagnosis'],'current_suggestion':r['correction'],
                'requested_post_checks':r['correction']['low_level']['post_checks'] if r['correction'] else [],
                'subsequent_observations_for_offline_review':[{'timestamp':n['timestamp'],'ref':n['observation_ref'],
                    'knife_location':n['visual_facts']['knife_location'],'gripper':n['visual_facts']['gripper'],
                    'cut_piece':n['visual_facts']['cut_piece'],'task_completed':n['state']['task_completed']} for n in rows[i+1:i+3]],
                'feedback_status':'requires_independent_pixel_review_not_a_training_label',
                'actually_executed':False,'execution_receipt':None,'caused_by_suggestion':None,
                'final_label_for_audit_only':final})
    with output.open('x',encoding='utf-8') as f:json.dump({'source_run':str(run.resolve()),'offline_only':True,
        'future_observations_are_not_online_features_or_action_rewards':True,'windows':windows},f,ensure_ascii=False,indent=2)
    print(json.dumps({'review_windows':len(windows),'training_labels_automatically_created':0}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--data-root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();collect(a.run,a.data_root,a.output)
