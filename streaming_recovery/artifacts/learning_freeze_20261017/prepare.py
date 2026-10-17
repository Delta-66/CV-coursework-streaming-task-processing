"""Versioned development protocol; labels and this file never enter runtime."""
from pathlib import Path
import json, shutil, sys, hashlib
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from download_vifailback import HubDownload, LEROBOT_REPO, sha256
OUT=Path(__file__).resolve().parent
BASE=ROOT/'streaming_recovery'
FIT=[3,10,12,17,19]
VALIDATION=[2,6,8,11,14,29]
AUDIT=[13,16]
REGRESSION=[0,1,2,6,8,11,14,18,19,29]
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def main():
    mode=sys.argv[1]
    data=ROOT/'data/vifailback_lerobot'
    splits={name:{x['episode_index'] for x in map(json.loads,(ROOT/f'configs/splits/{name}.jsonl').read_text(encoding='utf-8').splitlines())} for name in ('train','validation','dev_seen','test')}
    assert set(FIT+AUDIT)<=splits['train'] and set(VALIDATION)<=splits['validation']
    assert not set(FIT)&set(AUDIT) and not set(FIT+VALIDATION+AUDIT+REGRESSION)&splits['test']
    if mode=='prepare':
        sources={p.name:sha256(p) for p in BASE.glob('*.py')}
        old=json.loads((BASE/'artifacts/freeze_acceptance_20261017/final_run/summary.json').read_text(encoding='utf-8'))
        assert sources==old['code_sha256']
        for p in list(BASE.glob('*.py'))+list((BASE/'tests').glob('*.py'))+[BASE/'config_multimodal.json',BASE/'README.md',ROOT/'README.md']:
            target=OUT/'before'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
        package=OUT/'baseline_package/streaming_recovery';package.mkdir(parents=True)
        for p in BASE.glob('*.py'):shutil.copy2(p,package/p.name)
        write(OUT/'protocol.json',{'version':'supervised_fusion_dev_v1','seed':17,'fit':FIT,'validation_for_selection':VALIDATION,
            'regression':REGRESSION,'additional_audit_reserved_from_train':AUDIT,
            'audit_policy':'existing train membership unchanged; these entire trajectories excluded from fit and candidate selection; inspect only after choice',
            'exposure':'all original ten previously inspected; 19 is fit; 18 is seen normal control; no independent success trajectory',
            'train_labels':'time-local reviewed RGB/proprioceptive relation labels; no final-outcome broadcasting',
            'candidate_limit':3,'max_feedback_rounds':2,'stop':'one fitted softmax family; choose on fixed validation, no normal false failure; holdouts cannot trigger tuning',
            'candidates_C':[0.1,1.0,10.0],'split_sha256':{name:sha256(ROOT/f'configs/splits/{name}.jsonl') for name in splits},
            'source_sha256_before':sources,'test_sha256_before':{p.name:sha256(p) for p in (BASE/'tests').glob('test_*.py')},
            'revision':json.loads((data/'download_manifest.json').read_text(encoding='utf-8'))['revision']})
        shutil.copy2(data/'download_manifest.json',OUT/'download_manifest_before.json')
        cfg=json.loads((BASE/'config_multimodal.json').read_text(encoding='utf-8'))
        cfg['source']['root']=str(data);cfg['model']['path']=str(ROOT/'models/Qwen3-VL-2B-Instruct');cfg['output']=str(OUT/'runs')
        for name,eps in [('fit',FIT),('validation',VALIDATION),('regression',REGRESSION),('audit',AUDIT)]:
            c=json.loads(json.dumps(cfg));c['source']['episodes']=eps;write(OUT/f'baseline_{name}_config.json',c)
        print('protocol and baseline snapshot saved')
    elif mode=='download':
        protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
        hub=HubDownload(LEROBOT_REPO,data,protocol['revision'])
        for ep in FIT+AUDIT:
            for path in [f'data/chunk-000/episode_{ep:06d}.parquet']+[f'videos/chunk-000/observation.images.{cam}/episode_{ep:06d}.mp4' for cam in ('cam_high','cam_right_wrist')]:hub.download(path)
        hub.save_manifest(FIT+AUDIT)
        write(OUT/'download_verified.json',{'repo':LEROBOT_REPO,'revision':hub.revision,'files':hub.files})
        print('verified fit and sealed audit source files; no audit inference/labels read')
if __name__=='__main__':main()
