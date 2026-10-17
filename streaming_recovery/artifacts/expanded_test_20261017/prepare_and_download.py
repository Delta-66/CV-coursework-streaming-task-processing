"""本轮补测的固定选样与官方数据下载；不修改监测源码/参数或冻结划分。"""
from pathlib import Path
import argparse
import json
import shutil
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from download_vifailback import HubDownload, LEROBOT_REPO, sha256

OUT=Path(__file__).resolve().parent
BASE=ROOT/'streaming_recovery'
DATA=ROOT/'data/vifailback_lerobot'
SELECTED=[18,19,1,6,11,14]

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def write(path,value):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('prepare','download'))
    args=parser.parse_args()
    splits={name:{r['episode_index']:r for r in map(json.loads,(ROOT/f'configs/splits/{name}.jsonl').read_text(encoding='utf-8').splitlines())}
            for name in ('dev_seen','validation','train','test')}
    assert not set(SELECTED)&set(splits['test'])
    if args.mode=='prepare':
        frozen=read(BASE/'artifacts/freeze_acceptance_20261017/final_run/summary.json')
        assert all(sha256(BASE/name)==digest for name,digest in frozen['code_sha256'].items())
        entries=[]
        for ep in SELECTED:
            split=next(name for name,items in splits.items() if ep in items)
            entry=splits[split][ep]
            assert entry['task_index']==0
            entries.append({'episode':ep,'split':split,'task_index':0,'frames_from_metadata':entry['num_frames'],
                            'purpose':'normal_control' if ep in (18,19) else 'additional_failure',
                            'previous_exposure':{18:'seen_development_and_regression',19:'seen_development_and_regression',
                                1:'annotation_seen; media_not_previously_local',6:'earlier_version_real_run_seen',
                                11:'no_prior_local_media_or_run_found',14:'no_prior_local_media_or_run_found'}[ep],
                            'local_before_download':(DATA/f'data/chunk-000/episode_{ep:06d}.parquet').exists()})
        manifest=read(DATA/'download_manifest.json')
        write(OUT/'selection.json',{'selected_before_download_or_new_inference':True,'episodes':entries,
            'selection_rule':'all remaining task0 dev_seen/validation outside previous six, plus the only two success controls; no train failures added',
            'original_six':[18,19,0,2,8,29],'no_threshold_or_model_changes':True,'no_frozen_test_episode':True,
            'frozen_test_sha256':sha256(ROOT/'configs/splits/test.jsonl'),'source_sha256_before':frozen['code_sha256'],
            'dataset_revision':manifest['revision'],'generalization_claim':False})
        shutil.copy2(DATA/'download_manifest.json',OUT/'download_manifest_before.json')
        config=read(BASE/'config_multimodal.json')
        config['source']['root']=str(DATA)
        config['source']['episodes']=SELECTED
        config['model']['path']=str(ROOT/'models/Qwen3-VL-2B-Instruct')
        config['output']=str(OUT/'runs')
        write(OUT/'config.json',config)
        print('prepared fixed selection',SELECTED,flush=True)
    else:
        selection=read(OUT/'selection.json')
        assert [e['episode'] for e in selection['episodes']]==SELECTED
        assert all(sha256(BASE/name)==digest for name,digest in selection['source_sha256_before'].items())
        hub=HubDownload(LEROBOT_REPO,DATA,selection['dataset_revision'])
        for ep in SELECTED:
            paths=[f'data/chunk-000/episode_{ep:06d}.parquet']+[
                f'videos/chunk-000/observation.images.{camera}/episode_{ep:06d}.mp4' for camera in ('cam_high','cam_right_wrist')]
            for path in paths:
                hub.download(path)
        hub.save_manifest(SELECTED)
        write(OUT/'download_verified.json',{'repo':LEROBOT_REPO,'revision':hub.revision,'files':hub.files})
        print('downloaded/verified',SELECTED,flush=True)

if __name__=='__main__':
    main()
