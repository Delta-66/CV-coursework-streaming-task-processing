"""Actual-image overlays and offline review, never read by online inference."""
import json, hashlib
from pathlib import Path
from PIL import Image,ImageDraw
from streaming_recovery.localization import image_alignment_directions

OUT=Path(__file__).resolve().parent
RUN=OUT/'visual_probe_run_v1'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def rows(ep):
    return {r['timestamp']:r for r in map(json.loads,(RUN/f'trajectory_{ep:06d}/decisions.jsonl').read_text(encoding='utf8').splitlines())}

def localization(ep,t):
    r=rows(ep)[float(t)];f=r['visual_facts'];loc=f['measurements']['localization']
    p=RUN/f'trajectory_{ep:06d}/frames/{r["observation_ref"]}_cam_right_wrist.jpg'
    im=Image.open(p).convert('RGB');draw=ImageDraw.Draw(im)
    points=loc['finger_inner_points_pixels'];neck=loc['handle_neck_pixels']
    rack=[tuple(q)for q in loc['rack_contour_pixels']]
    if rack:draw.line(rack+[rack[0]],fill='yellow',width=2)
    for x,y in points:
        draw.ellipse((x-5,y-5,x+5,y+5),outline='cyan',width=2)
    draw.line([tuple(v)for v in points],fill='cyan',width=2)
    for a,b,c,d in neck['edge_segments']:draw.line((a,b,c,d),fill='red',width=2)
    handle=neck['center'];mid=[sum(p[i]for p in points)/2 for i in (0,1)]
    draw.line((*mid,*handle),fill='red',width=3)
    draw.text((8,8),f'ep={ep} t={t}s; cyan: finger inner edges; red: handle neck',fill='red')
    target=OUT/f'visual_localization_{ep}_{t}s.png';im.save(target)
    directions=image_alignment_directions(f['knife_handle_xy'],f['gripper_xy'],r['decision_rules']['direction_deadband_normalized'])
    return {'episode':ep,'timestamp':r['timestamp'],'observation_ref':r['observation_ref'],
            'source_image':str(p.resolve()),'source_image_sha256':sha(p),'overlay':str(target.resolve()),
            'localization_frame':f['localization_frame'],'knife_handle_xy':f['knife_handle_xy'],
            'gripper_xy':f['gripper_xy'],'units':'normalized_image_0_1000','localization':loc,
            'directions':directions,'deadband_normalized':r['decision_rules']['direction_deadband_normalized'],
            'inspection':'实际刀柄从细刃变宽的可见颈部横截面，黄色为独立刀架轮廓；青色两个端点为手指内缘，青线中点为抓取参考点。点不表示深度或接触。',
            'conditional_handoff':{'object':'knife','actuator':'right_gripper',
               'action_type':'align_in_image','coordinate_frame':f['localization_frame'],
               'direction':directions,'numeric_command':None,'physical_units':None,
               'preconditions':['当前实测右夹爪打开','当前刀柄颈部与两指内缘在同一腕图可见',
                                '深度、姿态、路径和机器人坐标映射未满足，须由执行器独立验证'],
               'stop_condition':'当前两指中点与柄颈参考点在新图像中进入二维死区，或任一目标丢失',
               'post_checks':['重新核对图像二维对齐','独立验证深度、姿态、接触和安全路径'],
               'mode':'current_real_pixel_conditional_handoff_without_failure_event',
               'actually_executed':False,'execution_receipt':None,'outcome_attributed_to_suggestion':False},
            'natural_failure_event':r['failure_event'],'natural_correction':r['correction']}

def blade_sheet():
    result=Image.new('RGB',(1024,828),'white');d=ImageDraw.Draw(result)
    evidence=[]
    for i,t in enumerate((5,6,7,8)):
        r=rows(19)[float(t)];m=r['visual_facts']['measurements'];wrist=m['wrist']
        p=RUN/f'trajectory_000019/frames/{r["observation_ref"]}_cam_right_wrist.jpg'
        im=Image.open(p).convert('RGB');draw=ImageDraw.Draw(im)
        for key,color in (('blade','cyan'),('food','red')):
            b=wrist.get(key)
            if b:
                x,y,w,h=b;draw.rectangle((x,y,x+w,y+h),outline=color,width=2)
        x,y=(i%2)*512,(i//2)*414
        result.paste(im,(x,y+30));d.text((x+8,y+8),f't={t}s blade/finger={wrist.get("joined")}; operation={m["cut_operation"]}',fill='black')
        evidence.append({'episode':19,'timestamp':t,'image':str(p.resolve()),'sha256':sha(p),
                         'wrist':wrist,'cut_operation':m['cut_operation'],'stage':r['state']['stage'],
                         'new_cut_effect':m['new_cut_effect']})
    result.save(OUT/'visual_19_actual_blade_operation.png')
    return evidence

def phase_sheet():
    selected=[(2,3),(2,7),(2,11),(8,2),(8,5),(8,8),(19,8),(19,11),(19,16)]
    result=Image.new('RGB',(1536,414*3),'white');d=ImageDraw.Draw(result)
    for i,(ep,t) in enumerate(selected):
        r=rows(ep)[float(t)];p=RUN/f'trajectory_{ep:06d}/frames/{r["observation_ref"]}_cam_high.jpg'
        x,y=(i%3)*512,(i//3)*414
        result.paste(Image.open(p),(x,y+30))
        d.text((x+8,y+8),f'ep={ep} t={t}s; stage={r["state"]["stage"]}',fill='black')
    result.save(OUT/'visual_real_phase_check.png')

def phase_review():
    notes=[
      (2,2,'align_grasp','右夹爪打开，腕图刀柄颈部位于两指中点左方；只支持二维对齐准备，不证明深度接触。','旧locate_tool漏掉准备阶段，当前align_grasp与像素及开爪证据一致。'),
      (2,3,'align_grasp','刀柄颈部位于两指中点下方，夹爪未夹住刀；顶视机械手在架上方。','当前识别二维准备，仍不能声称已实际对齐。'),
      (2,7,'verify_grasp','实测闭合；顶视刀仍留架，闭合夹爪位于架上方，尚未观察成功搬运。','新尝试来自观测开闭变化与近架投影，进入效果确认；闭爪不被当作成功。'),
      (2,11,'verify_grasp','空夹爪进入香肠工作区，白柄刀仍在右侧架上。','已有重复明确分离证据，grasp_not_established确认；具体物理失误原因仍未知。'),
      (8,2,'approach_tool','开爪机械手靠近架侧；腕图可见刀，但柄颈/手指不满足完整联合定位条件。','当前approach_tool合理；不能把阶段名或粗轮廓代理当已对齐，点仍未知。'),
      (8,5,'verify_grasp','右夹爪实际闭合，刀仍留架；机械手与刀柄在图像中分离。','观测到开闭尝试后的效果确认；尚不只凭闭合立即失败。'),
      (8,8,'verify_grasp','机械手空手进入砧板，刀仍清晰留在刀架。','重复分离证据确认grasp_not_established；无可靠夹持。'),
      (19,8,'cut_operation','腕图中央刀刃在两指之间，邻接香肠端部颜色区域较上一可用腕图移动；顶视已进入砧板。','旧候选因亮边连成宽轮廓拒识；当前窄纵向可见刀刃及操作证据保留，不直接等于切开。'),
      (19,11,'return_tool','顶视可见原香肠之外的圆形分离小段；携刀返回架边。','已存在早先参考、有效操作，当前新增分离效果支持切割确认，尚未最终释放完成。'),
      (19,16,'completed','机械手退出刀架，刀位于架上；砧板新分离圆形小段仍可见。','连续归位释放和既有操作—新效果链完整；建议未执行，不能归因于建议。')]
    result=[]
    for ep,t,expected,observed,assessment in notes:
        r=rows(ep)[float(t)]
        result.append({'episode_for_audit_only':ep,'timestamp':float(t),'observation_ref':r['observation_ref'],
            'expected_stage':expected,'observed_features':observed,'program_stage':r['state']['stage'],
            'assessment':assessment,'reviewer':'assistant_visual_inspection_using_view_image',
            'evidence':r['state']['evidence'],'failure_event':r['failure_event'],
            'frame_images':[{ 'camera':cam,'path':str((RUN/f'trajectory_{ep:06d}/frames/{r["observation_ref"]}_{cam}.jpg').resolve()),
                              'sha256':sha(RUN/f'trajectory_{ep:06d}/frames/{r["observation_ref"]}_{cam}.jpg')}
                              for cam in ('cam_high','cam_right_wrist')]})
    return result

if __name__=='__main__':
    records={'source_run':str(RUN.resolve()),'classification':'offline_real_pixel_review_not_independent_unseen_test',
             'conditional_directions':[localization(2,2),localization(2,3)],'blade_evidence':blade_sheet(),
             'key_stage_reviews':phase_review(),
             'limitations':['定位只在2的两个实际帧可靠；8的混连/遮挡主动拒识',
                 '真实失败确认后没有再次打开夹爪，所以无自然事件后align_in_image，不用合成事件补齐',
                 '未观测独立深度、姿态、接触准备；无真实闭合建议验证，无自然重试过程验证',
                 '6条均为已暴露开发/回归样例，不是新未见成绩']}
    phase_sheet()
    (OUT/'visual_real_evidence.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
