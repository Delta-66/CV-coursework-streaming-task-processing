"""离线补测核查；预期来自实际查看的图片，保留不通过点，不改变监测器。"""
from pathlib import Path
from copy import deepcopy
import json
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from streaming_recovery.__main__ import load_config
from streaming_recovery.review_binding import (load_run_rows, row_fingerprint, file_sha256,
    rebind_review, validate_review_bindings, BINDING_VERSION, FINGERPRINT_VERSION, FINGERPRINT_DEFINITION)

OUT=Path(__file__).resolve().parent
RUN=OUT/'run'
PREVIOUS=ROOT/'streaming_recovery/artifacts/freeze_acceptance_20261017'

def write(path,value):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)

def main():
    rows=load_run_rows(RUN)
    old=json.loads((PREVIOUS/'manual_review.json').read_text(encoding='utf-8'))
    old['checkpoints']=[c for c in old['checkpoints'] if c['episode_for_audit_only'] in (18,19)]
    old['conditional_directions']=[]
    # 正常对照沿用原目视预期，必须严格同图字节且显式重绑定新行。
    review=rebind_review(old,PREVIOUS/'final_run',RUN,'root assistant',
                         '正常18/19复跑；同图SHA核验后重绑定，原预期不变。追加样例只作离线目视核查。')
    review['review_kind']='助手目视复核20张新样例图；正常8点沿用已核同字节图；非独立人工时序标注'
    notes=[
        (1,0,['cam_high'],'刀在右侧刀架，香肠完整，机械手尚未取刀。',{'state.task_completed':False,'visual_facts.knife_location':'holder'}),
        (6,0,['cam_high'],'刀在刀架，砧板上只有完整长条香肠。',{'state.task_completed':False,'visual_facts.knife_location':'holder'}),
        (11,0,['cam_high'],'刀在刀架，砧板香肠为一整根。',{'state.task_completed':False,'visual_facts.knife_location':'holder'}),
        (14,0,['cam_high'],'刀在刀架，未见分离小段。',{'state.task_completed':False,'visual_facts.knife_location':'holder'}),
        (1,14,['cam_high','cam_right_wrist'],'刀仍在刀架而末端已到砧板；腕图闭合两指前未见刀，支持抓取未建立。',
         {'state.preconditions.reliable_grasp':False,'state.candidate':'grasp_not_established','execution_status':'confirmed_failure'}),
        (6,10,['cam_high','cam_right_wrist'],'刀仍在架，末端已到香肠处；腕图两指前未见刀，支持抓取失败。',
         {'state.preconditions.reliable_grasp':False,'state.candidate':'grasp_not_established','execution_status':'confirmed_failure'}),
        (11,7,['cam_right_wrist'],'刀刃可见且连接两指前方；刀与香肠的二维投影相交，但这不证明切开或真实接触。',
         {'visual_facts.knife_location':'gripper','state.preconditions.cut_effect_confirmed':False}),
        (11,14,['cam_high'],'机械手携刀回到刀架附近，香肠仍为长条，没有可见新增分离小段。',
         {'visual_facts.cut_piece':'no','state.task_completed':False}),
        (14,8,['cam_high','cam_right_wrist'],'当前刀在末端，腕图刀刃与两指连接，香肠位于刀右侧。模型未查询此腕图却称刀在架/分离，这是明确图像判断错误。',
         {'visual_facts.knife_location':'gripper','visual_facts.contact':'between_fingers','state.preconditions.reliable_grasp':True,'state.candidate':None}),
        (14,16,['cam_right_wrist'],'刀刃仍在两指前，香肠在图像右下，未见分离小段；不能仅凭刀和香肠都在砧板区域认定切割成功。',
         {'visual_facts.knife_location':'gripper','state.preconditions.cut_effect_confirmed':False}),
        (14,24,['cam_high'],'刀已在右侧刀架，机械手退出顶视范围，香肠仍完整；模型将刀位置标未知，归位感知不足。',
         {'visual_facts.knife_location':'holder','state.task_completed':False})]
    images=[]
    for ep,t,cameras,observed,expected in notes:
        row=next(r for r in rows[ep] if r['timestamp']==t)
        refs=[f'trajectory_{ep:06d}/frames/{row["observation_ref"]}_{cam}.jpg' for cam in cameras]
        hashes={p:file_sha256(RUN/p) for p in refs}
        record={'episode_for_audit_only':ep,'timestamp':t,'observation_ref':row['observation_ref'],
            'row_sha256':row_fingerprint(row),'row_fingerprint_version':FINGERPRINT_VERSION,
            'image_refs':refs,'image_sha256':hashes,'observed':observed,'expected_fields':expected}
        review['checkpoints'].append(record)
        images.append(deepcopy(record))
    # 补充实际看过但不强行设定阶段真值的图片。与上面合计20张。
    extra=[(11,4,'cam_right_wrist','刀具和刀架与两指相邻，单张图不保证已经建立可靠夹持。'),
           (11,10,'cam_right_wrist','刀刃投影横跨香肠，香肠仍连在一起；不由二维投影断言物理接触或切入深度。'),
           (11,15,'cam_high','机械手在刀架附近，香肠仍完整。'),
           (14,5,'cam_high','刀已离架，机械手在砧板/刀架边缘；单图不足以确定运动方向，暂不强制阶段真值。'),
           (14,10,'cam_high','仍持刀在砧板，香肠完整；不将旧疑似撤回当成建议导致的恢复。'),
           (14,10,'cam_right_wrist','刀刃与两指连接，香肠在刀右侧，与8秒图关系相近。')]
    for ep,t,cam,note in extra:
        row=next(r for r in rows[ep] if r['timestamp']==t)
        path=f'trajectory_{ep:06d}/frames/{row["observation_ref"]}_{cam}.jpg'
        images.append({'episode_for_audit_only':ep,'timestamp':t,'observation_ref':row['observation_ref'],
                       'row_sha256':row_fingerprint(row),'image_refs':[path],'image_sha256':{path:file_sha256(RUN/path)},'observed':note})
    binding=validate_review_bindings(review,rows,RUN)
    assert binding['valid'],binding
    write(OUT/'manual_review.json',review)
    write(OUT/'supplemental_visual_review.json',{'source_run':str(RUN),'reviewer':'root assistant via view_image',
        'actually_viewed_images':sum(len(x['image_refs']) for x in images),'records':images,
        'not_independent_human_temporal_ground_truth':True,'detection_latency_reported':False})
    write(OUT/'binding_validation.json',binding)
    previous_rows=load_run_rows(PREVIOUS/'final_run')
    controls={}
    for ep in (18,19):
        controls[str(ep)]=[{k:v for k,v in r.items() if k!='processing'} for r in rows[ep]]==[
            {k:v for k,v in r.items() if k!='processing'} for r in previous_rows[ep]]
    assert all(controls.values())
    write(OUT/'normal_control_comparison.json',{'semantic_equal_excluding_processing':controls,'images_rebound_with_identical_sha256':True})
    print(json.dumps({'valid_bound_checkpoints':len(review['checkpoints']),
                      'new_images_actually_viewed':sum(len(x['image_refs']) for x in images),'normal_controls_equal':controls}))

if __name__=='__main__':
    main()
