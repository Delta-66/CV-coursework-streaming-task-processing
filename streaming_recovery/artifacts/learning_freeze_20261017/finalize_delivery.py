"""Verify immutable evidence and point to the second, final engineering export."""
from pathlib import Path
import json,hashlib,sys,platform
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
checks=read(P/'independent_checks_v2/results.json')
assert len(checks)==6 and all(c['status']=='passed' for c in checks)
text=(P/'final_regression_tests_v2.txt').read_text(encoding='utf-8')
assert 'Ran 117 tests' in text and text.rstrip().endswith('OK')
audit=read(P/'frozen_run_v2/independence_audit.json')
assert audit['parameters_and_config_unchanged'] and audit['training_calls']==0 and not audit['forbidden_file_attempts']
bundle=P/'frozen_runtime_v2';manifest=read(bundle/'manifest.json')
for name,digest in manifest['files'].items():assert sha(bundle/name)==digest
protocol=read(P/'protocol.json')
for name,digest in protocol['source_sha256_before'].items():assert sha(P/'baseline_package/streaming_recovery'/name)==digest
for name,digest in protocol['split_sha256'].items():assert sha(ROOT/f'configs/splits/{name}.jsonl')==digest
assert sha(bundle/'fusion.json')==sha(P/'frozen_runtime/fusion.json')
examples=[]
for ep,t in ((11,15),(14,8),(18,20),(0,10),(13,12),(16,14)):
    rows=list(map(json.loads,(P/'frozen_run_v2'/f'trajectory_{ep:06d}'/'decisions.jsonl').read_text(encoding='utf-8').splitlines()))
    examples.append({'episode_for_audit_only':ep,'result':next(r for r in rows if r['timestamp']==t)})
(P/'real_examples_v2.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2),encoding='utf-8')
files=[P/name for name in ('protocol.json','selection.json','training_log.json','reviewed_samples.json','manual_review_v2.json',
    'evaluation_v2.json','comparison_v2.json','learning_evidence_v2.json','exposure_ledger.json','real_examples_v2.json',
    'independent_checks_v2/results.json','frozen_run_v2/summary.json','frozen_run_v2/independence_audit.json',
    'final_regression_tests_v2.txt','retry_guard_revalidation.json','work_retry_before.txt','work_retry_after.txt')]
files += [ROOT/'docs/监督反馈学习与独立冻结验收_20261017.md',ROOT/'streaming_recovery/README.md',ROOT/'README.md']
delivery={'active_bundle':'frozen_runtime_v2','active_run':'frozen_run_v2','engineering_status':'passed','semantic_status':'partial',
    'unit_tests':117,'independent_groups':6,'baseline_preserved':True,'original_splits_unchanged':True,
    'training_rounds':2,'post_selection_engineering_fix':'observed_new_attempt_resets_previous_work_cycle; no parameter tuning',
    'python':sys.version,'platform':platform.platform(),'source_sha256':{f.name:sha(f) for f in (ROOT/'streaming_recovery').glob('*.py')},
    'artifacts':{str(f.relative_to(ROOT)).replace('\\','/'):sha(f) for f in files},'bundle_manifest_sha256':sha(bundle/'manifest.json'),
    'remaining':['13 only suspected, failure unconfirmed','14 final holder relation unknown','12 initial calibration failure',
        'no natural retry/depth/contact/physical execution/generalization validation']}
(P/'delivery_manifest_v2.json').write_text(json.dumps(delivery,ensure_ascii=False,indent=2),encoding='utf-8')
(P/'CURRENT_DELIVERY.json').write_text(json.dumps({'bundle':'frozen_runtime_v2','results':'frozen_run_v2',
    'manifest':'delivery_manifest_v2.json','evaluation':'evaluation_v2.json','report':'../../../docs/监督反馈学习与独立冻结验收_20261017.md'},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'verified':True,'unit_tests':117,'independent_groups':6,'baseline_preserved':True,'active_bundle':'frozen_runtime_v2'},ensure_ascii=False))
