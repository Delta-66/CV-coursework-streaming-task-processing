"""Offline supervised fitting only; never imported by the frozen runtime."""
import argparse, hashlib, json, time
from pathlib import Path
from collections import Counter
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from streaming_recovery.fusion import FEATURE_NAMES, CLASSES, FixedFusion

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def train(directory):
    directory=Path(directory)
    protocol=json.loads((directory/'protocol.json').read_text(encoding='utf-8'))
    review=json.loads((directory/'reviewed_samples.json').read_text(encoding='utf-8'))
    features=json.loads((directory/'fit_features.json').read_text(encoding='utf-8'))
    root=Path(__file__).resolve().parents[2]
    members={json.loads(x)['episode_index'] for x in (root/'configs/splits/train.jsonl').read_text(encoding='utf-8').splitlines()}
    lookup={(r['episode_for_grouping_only'],r['timestamp']):r for r in features['records']}
    X,y,groups=[],[],[]
    for r in review['samples']:
        ep=r['episode_for_grouping_only'];t=r['timestamp']
        assert ep in protocol['fit'] and ep in members and ep not in protocol['additional_audit_reserved_from_train']
        assert r['review_status']=='pixel_reviewed' and r['actually_executed'] is False
        for im in r['images']: assert digest(im['path'])==im['sha256']
        rows=[json.loads(s) for s in (directory/'baseline_fit'/f'trajectory_{ep:06d}'/'decisions.jsonl').read_text(encoding='utf-8').splitlines()]
        matches=[row for row in rows if row['timestamp']==t]
        assert len(matches)==1
        fingerprint=hashlib.sha256(json.dumps(matches[0],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        assert fingerprint==r['prediction_source_sha256']
        f=lookup[(ep,t)]
        assert f['valid'] and f['features'] is not None
        X.append(f['features']);y.append(r['relation']);groups.append(ep)
    X=np.asarray(X);y=np.asarray(y)
    counts=Counter(groups)
    weights=np.asarray([len(groups)/len(counts)/counts[g] for g in groups])
    scaler=StandardScaler().fit(X,sample_weight=weights)
    z=scaler.transform(X)
    report={'schema':'offline_fit_v1','review_sha256':digest(directory/'reviewed_samples.json'),
        'features_sha256':digest(directory/'fit_features.json'),'seed':protocol['seed'],'sklearn':sklearn.__version__,
        'samples':len(y),'group_counts':dict(counts),'class_counts':dict(Counter(y)),
        'excluded':review['excluded'],'standardizer_fit_groups':sorted(counts),
        'validation_used_in_fit':False,'labels_are_current_relation_not_episode_outcome':True,
        'uniform_initial_log_loss':float(np.log(len(CLASSES))),'candidates':[]}
    for C in protocol['candidates_C']:
        start=time.perf_counter()
        model=LogisticRegression(C=C,max_iter=1000,random_state=protocol['seed'],class_weight='balanced',solver='lbfgs')
        model.fit(z,y,sample_weight=weights)
        order=[list(model.classes_).index(c) for c in CLASSES]
        data={'schema':'fixed_softmax_fusion_v1','features':list(FEATURE_NAMES),'classes':list(CLASSES),
            'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),
            'coef':model.coef_[order].tolist(),'intercept':model.intercept_[order].tolist(),
            'minimum_score':.55,'training_metadata':{'C':C,'seed':protocol['seed'],'groups':sorted(counts),
                'sample_count':len(y),'review_sha256':report['review_sha256'],'online_updates_allowed':False}}
        path=directory/f'fusion_C{C:g}.json'
        path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
        exported=FixedFusion(path)
        expected=model.predict_proba(z)[:,order]
        error=max(float(np.max(abs(exported.predict(x)[1]-p))) for x,p in zip(X,expected))
        assert error<1e-12
        report['candidates'].append({'C':C,'path':str(path.resolve()),'sha256':digest(path),
            'fit_wall_s':time.perf_counter()-start,'iterations':model.n_iter_.tolist(),
            'log_loss':float(log_loss(y,model.predict_proba(z),labels=model.classes_,sample_weight=weights)),
            'training_relation_accuracy':float(np.average(model.predict(z)==y,weights=weights)),
            'parameter_l2_from_zero':float(np.linalg.norm(model.coef_)),
            'export_probability_max_abs_error':error})
    (directory/'training_log.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);train(p.parse_args().directory)
