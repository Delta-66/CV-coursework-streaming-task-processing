"""保存本轮助手实际逐图核查记录；不进行推理、不制造事件或独立人工标注。"""
import argparse
import hashlib
import json
from pathlib import Path
from streaming_recovery.evaluation import row_fingerprint


def build(run):
    run=Path(run)
    notes=[
        (18,0,"顶视初始只有长条香肠，刀在架；未出现新分离小段。",None,{"state.task_completed":False}),
        (18,9,"腕图刀与两指连接，刀刃位于香肠端部；顶视机械手在砧板。",["cut_operation"],{"state.preconditions.reliable_grasp":True,"visual_facts.measurements.cut_operation":"yes"}),
        (18,20,"与初始相比新增独立圆片；刀已回架且机械手退出。",["completed"],{"state.task_completed":True}),
        (19,8,"连续5至8秒腕图显示刀刃在两指之间、与香肠端部相对位置变化。",["cut_operation"],{"state.preconditions.reliable_grasp":True,"visual_facts.measurements.cut_operation":"yes"}),
        (19,10,"与0秒相比新增明显圆片；仍持刀开始回架。单帧新增物需后续稳定确认。",["return_tool"],{"visual_facts.measurements.new_cut_effect":"yes"}),
        (19,16,"新圆片仍在砧板，刀回架且右手退出；切割参考、作用和效果链可核查。",["completed"],{"state.task_completed":True}),
        (0,7,"腕图两指已收拢但刀柄在侧旁/原架，接触关系仍不清；不应由闭爪报告成功。",["verify_grasp"],{"failure_event":None}),
        (0,10,"顶视刀留架，腕图空夹爪在香肠上方，支持抓取未建立。",["verify_grasp"],{"state.preconditions.reliable_grasp":False,"state.candidate":"grasp_not_established","execution_status":"confirmed_failure"}),
        (2,2,"腕图可见独立刀柄颈部和两指内缘；两指张开，正在准备对齐。",["align_grasp"],{"visual_facts.gripper":"open","visual_facts.measurements.localization.status":"localized","visual_facts.grasp_readiness":{}}),
        (2,3,"刀柄颈部移至两指中点下方，仍为准备对齐，不表示已经接触。",["align_grasp"],{"visual_facts.measurements.localization.status":"localized"}),
        (2,7,"相对6秒可见两指明显收拢，刀仍未带离架；实际闭合应建立一次尝试而非成功。",["verify_grasp"],{"state.attempt":1}),
        (2,11,"刀留架而空夹爪已进入工作区；有前面的实际闭合，诊断该次抓取未建立。",["verify_grasp"],{"state.preconditions.reliable_grasp":False,"execution_status":"confirmed_failure","state.candidate":"grasp_not_established"}),
        (8,4,"两指在刀架旁，刀架和手指混连；应识别接近但不能伪造精确对齐坐标。",["approach_tool"],{"visual_facts.knife_handle_xy":None,"visual_facts.gripper_xy":None}),
        (8,5,"相对4秒两指收拢，刀仍在原架；进入抓取效果检查。",["verify_grasp"],{"state.attempt":1}),
        (8,8,"空夹爪处于砧板上，刀仍留在架；支持当前抓取失败。",["verify_grasp"],{"state.preconditions.reliable_grasp":False,"execution_status":"confirmed_failure","state.candidate":"grasp_not_established"}),
        (29,6,"两指已收拢但仍有可见间隙，刀在架旁；closed只表示原始开度阈值，效果仍需检查。",["verify_grasp"],{"state.attempt":1}),
        (29,9,"夹爪为空、刀在架，未带走刀便进入工作区；不推断夹持力或摩擦原因。",["verify_grasp"],{"state.preconditions.reliable_grasp":False,"execution_status":"confirmed_failure"}),
        (18,18,"归位接触交接被右臂与刀架遮挡，允许当前关系未知；后续清楚释放才能确认完成。",["unknown"],{"execution_status":"information_insufficient","failure_event":None}),
        (19,14,"回架时手指/刀架投影混叠，当前连接或释放尚不可靠；保留未知并继续观察。",["unknown"],{"execution_status":"information_insufficient","failure_event":None}),
    ]
    rows={ep:{r['timestamp']:r for r in map(json.loads,(run/f'trajectory_{ep:06d}/decisions.jsonl').read_text(encoding='utf-8').splitlines())} for ep in (18,19,0,2,8,29)}
    checkpoints=[]
    for ep,t,observed,stages,fields in notes:
        refs=[f'trajectory_{ep:06d}/frames/obs-{t*25:06d}_{c}.jpg' for c in ('cam_high','cam_right_wrist')]
        item={'episode_for_audit_only':ep,'timestamp':float(t),'observed':observed,'expected_fields':fields,
              'image_refs':refs,'image_sha256':{p:hashlib.sha256((run/p).read_bytes()).hexdigest() for p in refs}}
        if stages:item['expected_stages']=stages
        checkpoints.append(item)
    directions=[]
    for t,direction,observed in ((2,'图像左侧','可见刀柄颈部宽度过渡在两指内缘中点左侧，纵向在死区内。'),
                                 (3,'图像下方','可见刀柄颈部位于两指内缘中点下方，横向差未超过50/1000死区。')):
        row=rows[2][t];facts=row['visual_facts']
        image_path=f'trajectory_000002/frames/{row["observation_ref"]}_cam_right_wrist.jpg'
        directions.append({'episode_for_audit_only':2,'timestamp':float(t),'observation_ref':row['observation_ref'],
            'row_sha256':row_fingerprint(row),'image_path':image_path,'image_sha256':hashlib.sha256((run/image_path).read_bytes()).hexdigest(),
            'camera':'cam_right_wrist','localization_frame':facts['localization_frame'],
            'knife_handle_xy':facts['knife_handle_xy'],'gripper_xy':facts['gripper_xy'],
            'localization_method':facts['measurements']['localization']['method'],
            'expected_directions':[direction],'reviewer':'root assistant visual inspection',
            'observed_features':observed,'reviewed_handle_neck':True,'reviewed_two_finger_edges':True,
            'coordinate_claim':'image_plane_only_no_robot_mapping'})
    return {'review_kind':'本轮助手逐图目视核查；非独立人工阶段标注，无首次可检测时刻标注',
            'source_run':str(run.resolve()),'checkpoints':checkpoints,'conditional_directions':directions,
            'scope':'六条已见回归；少量关键时刻约束，不要求每帧确定，也不证明全部时刻正确。',
            'remaining_gaps':['2:0/1秒初始化等待、9秒遮挡','8精确柄/两指对齐仍拒识',
                              '18:18秒与19:14秒回架交接未知','没有独立深度/姿态/接触准备',
                              '六条中没有自然失败后重新打开/重抓恢复过程']}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    with Path(args.output).open('x',encoding='utf-8') as f:
        json.dump(build(args.run),f,ensure_ascii=False,indent=2)
