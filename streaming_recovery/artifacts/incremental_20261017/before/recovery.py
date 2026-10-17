"""同一事件生成两层建议；仅输出一个下一动作，不把建议写回任务状态。"""
from __future__ import annotations


PROBLEMS = {
    "grasp_not_established": ("抓取尝试后机器人已进入工作区，但刀仍留在刀架", "刀具未建立可靠夹持，携刀进入切割区的预期效果未实现"),
    "prerequisite_unmet": ("机器人进入工作区时刀仍在刀架，缺少取刀证据", "执行后续操作所需的刀具夹持前置条件未满足"),
    "lost_grasp": ("此前存在夹持证据，随后刀与夹爪分离", "刀具连接状态失效；是否为意外滑落仍需排除有意放下"),
}


def diagnose(event, state, facts):
    if event is None or event["state"] in ("resolved", "retracted", "closed_for_retry", "superseded"):
        return None
    problem, mismatch = PROBLEMS[event["kind"]]
    return {"event_id":event["event_id"],"type":event["kind"],"problem":problem,
            "expected_effect":state["expected_effect"],"observed_deviation":mismatch,
            "observed_facts":event["evidence"],"current_evidence_status":event["observation_status"],
            "established_facts":["本诊断依据已到达图像的物体位置/运动与夹爪观测；不使用控制意图或轨迹结果标签"],
            "cause_hypotheses":[{"hypothesis":"刀柄对齐或接触条件可能未建立" if event["kind"] != "lost_grasp" else "夹持连接可能丢失，也可能是有意放下",
                                  "certainty":"unconfirmed","missing_evidence":"清晰的刀柄—两指接触图像；实测接触/夹持反馈"}],
            "uncertainty":"证据支持当前效果偏差，不能据此确定摩擦、夹持力或三维姿态原因；经典视觉模型受固定场景假设限制",
            "need_more_observation":True,
            "requested_evidence":["当前刀架与刀具位置", "刀柄与夹爪两指关系", "下一次尝试后刀是否随夹爪离架运动"],
            "confidence":None,"confidence_explanation":"未输出概率；疑似/确认来自配置的连续观测条数和时间跨度"}


def propose(event, diagnosis, state, facts, rules):
    if diagnosis is None:
        return None
    goal_id = event["event_id"] + ":restore_reliable_grasp"
    high = {"event_id":event["event_id"],"recovery_goal_id":goal_id,"object":"knife",
            "blocked_subtask":"携刀进入工作区并执行切割", "failed_precondition":"reliable_grasp",
            "impact":"继续移动空夹爪不能完成切割；必须先恢复刀具夹持条件",
            "strategy":"return_to_grasp_preparation", "next_recovery_subgoal":"重新建立刀柄定位、两指对齐及可靠夹持条件",
            "reason":diagnosis["observed_deviation"],
            "resume_when":"连续新观测支持刀离开刀架且随夹爪运动，再恢复携刀转移"}
    low = {"event_id":event["event_id"],"recovery_goal_id":goal_id,"object":"knife","actuator":"right_gripper",
           "action_type":None,"instruction":None,"coordinate_frame":None,"target_mode":"qualitative",
           "translation":None,"rotation":None,"numeric_command":None,"units":None,"numeric_source":None,
           "preconditions":[],"stop_condition":None,"post_checks":[],
           "directly_executable":False,"handoff_requirements":["接入经过验证的执行器接口", "由执行器检查碰撞与动作可行性"],
           "missing_information":["相机到机器人坐标变换", "深度与机器人运动接口"]}
    reliable_empty = facts.valid and facts.knife_location=="holder" and facts.contact=="separate" and not state["modality_conflict"]
    if event["state"] == "suspected" or not reliable_empty:
        low.update(action_type="hold_and_observe",coordinate_frame="not_applicable_no_spatial_move",
            instruction="暂停继续进入切割步骤，保持当前末端与夹爪设定，获取刀架和右夹爪的清晰新观测；不自动打开可能持物的夹爪。",
            preconditions=[{"condition":"执行器能够安全保持当前状态","status":"requires_executor_check"}],
            stop_condition="取得可判定刀位置与两指接触关系的新观测，或交由操作人员处理",
            post_checks=["刀是否仍在刀架", "夹爪是否与刀分离", "是否出现可靠持刀证据"])
    elif facts.gripper == "closed":
        low.update(action_type="open_gripper",coordinate_frame="right_gripper_actuator_no_translation",
            instruction="确认夹爪内无刀且当前位姿可安全保持后，保持末端位姿并打开右夹爪，为重新对齐刀柄建立条件；本次不同时发出移动或再次闭合动作。",
            preconditions=[{"condition":"刀仍在刀架且与右夹爪分离","status":"supported_by_current_observation"},
                           {"condition":"夹爪内无其他物体且可安全打开","status":"requires_executor_check"}],
            stop_condition="夹爪反馈与图像均表明已经打开；出现接触异常则停止并交回执行器",
            post_checks=["右夹爪确实打开", "刀仍在刀架且未被拖动", "重新观测刀柄与两指的相对位置，再决定下一动作"])
    elif facts.gripper == "open" and facts.knife_handle_xy is not None and facts.gripper_xy is not None:
        delta = [a-b for a,b in zip(facts.knife_handle_xy,facts.gripper_xy)]
        directions = []
        deadband = rules["direction_deadband_normalized"]
        if abs(delta[0]) > deadband:
            directions.append("图像右侧" if delta[0]>0 else "图像左侧")
        if abs(delta[1]) > deadband:
            directions.append("图像下方" if delta[1]>0 else "图像上方")
        low.update(action_type="align_in_image",coordinate_frame="cam_high_image_x_right_y_down",
            target_mode="relative_image_alignment",
            instruction="在完成标定与可达性检查后，使夹爪两指中点朝" + ("、".join(directions) if directions else "刀柄中心") + "对齐；保持夹爪打开，姿态需由深度/接触观测确定，本次不闭合。",
            translation={"kind":"qualitative_image_plane","direction":directions or ["对齐刀柄中心"],"magnitude":None},
            preconditions=[{"condition":"刀柄和两指位置在当前顶视图中可定位","status":"supported_by_current_observation"},
                           {"condition":"图像到执行器的映射、深度和路径可行性已验证","status":"not_satisfied"}],
            stop_condition="新图像表明刀柄进入两指间且执行器确认无碰撞；失去目标则停止",
            post_checks=["刀柄位于两指间", "深度与姿态满足接触条件", "保持打开，检查通过后另行生成闭合建议"])
    else:
        low.update(action_type="relocalize",coordinate_frame="cam_high_and_cam_right_wrist_images",
            instruction="保持已打开的夹爪和当前位姿，重新获取刀柄及两指的清晰顶视/腕部图像，交接刀柄对齐目标；未定位前不发出空间移动。",
            preconditions=[{"condition":"执行器能够安全保持当前位姿","status":"requires_executor_check"}],
            stop_condition="刀柄与两指可定位且具备映射依据；否则继续获取新观测或人工定位",
            post_checks=["刀柄与两指均可见", "取得图像到机器人映射后才生成物理位移"])
    return {"high_level":high,"low_level":low,
            "execution":{"mode":"advisory_replay","suggestion_generated":True,"actually_executed":False,
                         "execution_receipt":None,"outcome_attributed_to_suggestion":False,
                         "post_check_result":"pending_no_execution_receipt"}}


def validate_consistency(result):
    """运行时强约束；失败即终止该结果写出，不静默输出不一致控制建议。"""
    suggestion = result["correction"]
    if suggestion is None:
        return
    high,low = suggestion["high_level"],suggestion["low_level"]
    if high["event_id"] != result["failure_event"]["event_id"] or high["event_id"] != low["event_id"]:
        raise ValueError("correction_event_mismatch")
    if high["recovery_goal_id"] != low["recovery_goal_id"] or high["object"] != low["object"]:
        raise ValueError("correction_goal_object_mismatch")
    if low["action_type"] not in ("hold_and_observe","open_gripper","align_in_image","relocalize"):
        raise ValueError("action_does_not_serve_restore_grasp")
    if not low["coordinate_frame"] or not low["preconditions"] or not low["post_checks"] or not low["stop_condition"]:
        raise ValueError("missing_action_contract")
    if low["numeric_command"] is not None or low["directly_executable"]:
        raise ValueError("uncalibrated_numeric_or_executable_command")
    if low["action_type"] == "align_in_image" and low["coordinate_frame"] != "cam_high_image_x_right_y_down":
        raise ValueError("image_robot_coordinate_confusion")
    if low["action_type"] == "open_gripper":
        facts = result["visual_facts"]
        if facts["knife_location"] != "holder" or facts["contact"] != "separate":
            raise ValueError("unsafe_open_precondition")
    execution = suggestion["execution"]
    if execution["actually_executed"] or execution["execution_receipt"] is not None or execution["outcome_attributed_to_suggestion"]:
        raise ValueError("replay_must_not_claim_execution")
