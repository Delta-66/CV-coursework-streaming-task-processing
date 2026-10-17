"""Use exactly the runtime causal feature function, without fitting/using outcome labels."""
from pathlib import Path
import argparse,json
from ..__main__ import load_config
from ..source import LeRobotStream,instruction_for_episode
from ..engine import Monitor
from ..multiview import MultiViewGeometryVision
from ..fusion import fusion_features,FEATURE_NAMES

class FeatureVision(MultiViewGeometryVision):
    def query_wrist(self,rack_present,rack_clear,opened):return True
    def relation_prediction(self,observation,measurements,wrist,bright,dark):
        measurements['student_features']=fusion_features(observation,measurements,wrist,bright,dark,self.rules).tolist()
        return None

def collect(config,output):
    config=load_config(config);root=Path(config['source']['root'])
    vision=FeatureVision(config['geometry'],config['rules']);monitor=Monitor(config,vision)
    records=[]
    for ep in config['source']['episodes']:
        monitor.reset()
        stream=LeRobotStream(root,ep,config['source']['cameras'],config['source']['sample_every'],config['source']['max_age_s'],config['model']['image_edge'],instruction_for_episode(root,ep))
        for obs in stream:
            r=monitor.step(obs)
            records.append({'episode_for_grouping_only':ep,'timestamp':obs.timestamp,'ref':obs.ref,
                'features':r['visual_facts']['measurements'].get('student_features'),
                'valid':r['visual_facts']['valid'],'error':r['visual_facts']['error'],
                'current_wrist':r['visual_facts']['measurements'].get('wrist'),
                'no_outcome_stage_time_or_episode_input':True})
    with Path(output).open('x',encoding='utf-8') as f:json.dump({'feature_names':FEATURE_NAMES,'records':records,
        'sampling':'1Hz causal source; always both current views and one reference, same as student runtime'},f,ensure_ascii=False,indent=2)
    print(json.dumps({'records':len(records),'valid_features':sum(r['features'] is not None for r in records)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args();collect(a.config,a.output)
