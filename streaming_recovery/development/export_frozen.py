"""Copies only audited inference modules and fixed parameters, no development files."""
import argparse,json,shutil,hashlib,importlib.metadata
from pathlib import Path

MODULES=('__init__','attempts','contracts','dynamics','effects','engine','events','fusion','geometry',
         'localization','memory','multiview','recovery','report','source','task_spec','tracking')

def export(directory,bundle_name='frozen_runtime'):
    directory=Path(directory).resolve();source=Path(__file__).resolve().parents[1]
    selection=json.loads((directory/'selection.json').read_text(encoding='utf-8'))
    bundle=directory/bundle_name;(bundle/'streaming_recovery').mkdir(parents=True,exist_ok=False)
    for name in MODULES:shutil.copy2(source/f'{name}.py',bundle/'streaming_recovery'/f'{name}.py')
    text=(source/'__main__.py').read_text(encoding='utf-8').split('\ndef main():')[0]
    (bundle/'streaming_recovery/__main__.py').write_text(text,encoding='utf-8')
    shutil.copy2(Path(__file__).with_name('frozen_entry_template.py'),bundle/'run_frozen.py')
    shutil.copy2(directory/f"fusion_C{selection['selected_C']:g}.json",bundle/'fusion.json')
    config=json.loads((directory/'selected_config.json').read_text(encoding='utf-8'))
    config['fusion']['path']='fusion.json';config['model']['path']='.';config['output']='runs'
    (bundle/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    packages=('numpy','opencv-python','Pillow','av','pyarrow')
    (bundle/'requirements.txt').write_text('\n'.join(n+'=='+importlib.metadata.version(n) for n in packages)+'\n',encoding='utf-8')
    files={str(p.relative_to(bundle)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.rglob('*') if p.is_file()}
    manifest={'schema':'frozen_standalone_v1','files':files,'selection_sha256':hashlib.sha256((directory/'selection.json').read_bytes()).hexdigest(),
        'dev_files_included':False,'model_inference':'fixed_numpy_linear_softmax_no_fit',
        'run':'python -I run_frozen.py --output ABSOLUTE_NEW_DIRECTORY'}
    (bundle/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(bundle)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--bundle-name',default='frozen_runtime')
    args=p.parse_args();export(args.directory,args.bundle_name)
