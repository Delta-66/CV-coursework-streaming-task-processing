"""Standalone entry: isolated imports, allowlisted input files, no training or advice input."""
import argparse, hashlib, json, os, sys
from pathlib import Path

def main():
    bundle=Path(__file__).resolve().parent
    # python -I excludes cwd/PYTHONPATH; only this exported package is inserted.
    sys.path.insert(0,str(bundle));sys.dont_write_bytecode=True
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    args=parser.parse_args();output=Path(args.output).resolve()
    config_path=bundle/'config.json';model_path=bundle/'fusion.json'
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    before={p.name:digest(p) for p in (config_path,model_path)}
    manifest=json.loads((bundle/'manifest.json').read_text(encoding='utf-8'))
    for name,value in manifest['files'].items():
        if digest(bundle/name)!=value:raise RuntimeError('frozen_file_changed:'+name)
    config=json.loads(config_path.read_text(encoding='utf-8'))
    root=Path(config['source']['root']).resolve()
    allowed={root/'meta/info.json',root/'meta/tasks.jsonl'}
    for ep in config['source']['episodes']:
        chunk=ep//1000
        allowed.add(root/f'data/chunk-{chunk:03d}/episode_{ep:06d}.parquet')
        for cam in config['source']['cameras']:
            allowed.add(root/f'videos/chunk-{chunk:03d}/observation.images.{cam}/episode_{ep:06d}.mp4')
    runtime_roots=(Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve())
    evidence={'training_calls':0,'forbidden_imports':[],'forbidden_file_attempts':[],
        'parquet_columns':[],'input_files':[],'guard':'python_audit_plus_native_decoder_and_parquet_wrappers',
        'teacher_feedback_training_labels_old_suggestions_allowed':False,'isolation_flag':sys.flags.isolated,
        'note':'application input audit, not an operating-system security sandbox'}
    blocked=('sklearn','torch','transformers','streaming_recovery.development','streaming_recovery.validation',
             'streaming_recovery.evaluation','streaming_recovery.review_existing','streaming_recovery.vision')
    def permitted(path,write=False):
        if isinstance(path,int):return
        p=Path(os.fsdecode(path)).resolve()
        if p.is_relative_to(output):
            if write:return
            frame=sys._getframe(1)
            while frame:
                if frame.f_globals.get('__name__')=='streaming_recovery.report':return
                frame=frame.f_back
            raise PermissionError('inference_cannot_read_output_history')
        if p in allowed or p.is_relative_to(bundle) or any(p.is_relative_to(x) for x in runtime_roots):
            if write and not p.is_relative_to(output):raise PermissionError('frozen_input_write')
            return
        if str(p).lower() in ('nul','/dev/null'):return
        evidence['forbidden_file_attempts'].append(str(p));raise PermissionError('non_runtime_input:'+str(p))
    def audit(event,args):
        if event=='import' and any(args[0]==x or args[0].startswith(x+'.') for x in blocked):
            evidence['forbidden_imports'].append(args[0]);raise ImportError('development_dependency_forbidden')
        if event=='open':
            mode=args[1];flags=args[2]
            write=bool(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
            permitted(args[0],write)
    sys.addaudithook(audit)
    def profile(frame,event,arg):
        if event=='call' and frame.f_code.co_name in ('fit','partial_fit','backward','optimizer_step'):
            evidence['training_calls']+=1;raise RuntimeError('training_call_in_frozen_runtime')
    sys.setprofile(profile)
    import av
    import pyarrow.parquet as pq
    original_av=av.open;original_parquet=pq.ParquetFile
    def av_open(path,*a,**kw):
        permitted(path);evidence['input_files'].append(str(path));return original_av(path,*a,**kw)
    class ReadOnlyParquet:
        def __init__(self,path,*a,**kw):
            permitted(path);self.table=original_parquet(path,*a,**kw)
            self.schema_arrow=self.table.schema_arrow;evidence['input_files'].append(str(path))
        def iter_batches(self,*a,**kw):
            columns=kw.get('columns')
            if columns is None or not set(columns)<={'timestamp','frame_index','task_index','observation.state','observation.velocity'}:
                raise PermissionError('non_observation_parquet_columns')
            evidence['parquet_columns'].append(columns)
            return self.table.iter_batches(*a,**kw)
    av.open=av_open;pq.ParquetFile=ReadOnlyParquet
    from streaming_recovery.__main__ import load_config,run
    run(load_config(config_path),output)
    sys.setprofile(None)
    after={p.name:digest(p) for p in (config_path,model_path)}
    assert before==after and evidence['training_calls']==0
    evidence.update(before_sha256=before,after_sha256=after,parameters_and_config_unchanged=True,
        imported_project_modules=sorted(n for n in sys.modules if n.startswith('streaming_recovery')),
        imported_training_libraries=sorted(n for n in sys.modules if n.split('.')[0] in ('torch','sklearn','transformers')))
    (output/'independence_audit.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
