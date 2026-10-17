"""合成语义渐闭与事件转换示例；不声称真实像素/自然重试。"""
import json
from pathlib import Path
from streaming_recovery.tests.test_smooth_attempt import SmoothAttemptTests

x=SmoothAttemptTests();x.setUp()
lost=x.held_then_lost()
rows=[x.feed(t,raw)for t,raw in enumerate((.06,.045,.035,.025,.015,.005,.005),4)]
result={'evidence_kind':'synthetic_semantic_logic_not_real_visual_or_natural_retry',
        'previous_lost_event':lost,'smooth_close_rows':rows,
        'new_attempt_timestamps':[r['timestamp']for r in rows if r['state']['new_attempt_observed']],
        'new_state_slots':0,'latch':'existing TaskTracker.armed boolean',
        'not_supported_by_current_real_backend':'closed gripper board/elsewhere joint localization; actual backend localizes only opened holder view'}
out=Path(__file__).resolve().parent/'attempt_synthetic_trace.json'
out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps([{'timestamp':r['timestamp'],'raw':r['observation_inputs']['observation.state'][13],
                  'attempt':r['state']['attempt'],'new_attempt':r['state']['new_attempt_observed'],
                  'event':{k:r['failure_event'][k]for k in ('event_id','kind','state')} if r['failure_event']else None}
                 for r in rows],ensure_ascii=False,indent=2))
