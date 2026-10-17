"""按当前证据推进单步建议。恢复上下文有界，输出建议不修改物理状态。"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
from .localization import image_alignment_directions

PROBLEMS = {
    "grasp_not_established": ("实际抓取尝试后观察到空手推进，或刀仍在原支撑面且与闭合两指明确分离", "当前尝试未建立可靠夹持"),
    "prerequisite_unmet": ("进入工作区时没有可靠持刀证据", "后续切割所需的夹持前置条件未满足"),
    "lost_grasp": ("近期可靠夹持后观察到刀与夹爪分离", "夹持连接失效；意外滑落与有意放下的原因尚未区分"),
}

class RecoveryContext:
    """仅保留一个事件（证据最多4项）、一个等待类型和饱和计数。"""
    def __init__(self):
        self.event = None
        self.wait_key = None
        self.wait_count = 0

    def update(self, event, state):
        if state["recovered"] or (event and event["state"] in ("resolved", "retracted")):
            self.event = None
        elif event:
            self.event = deepcopy(event)
        if self.event and not state["observable"]:
            self.event["observation_status"] = "unobservable"
        elif self.event and not state["candidate"]:
            self.event["observation_status"] = "not_currently_supported"
        return deepcopy(self.event)

    def observe_request(self, correction):
        key = correction["low_level"]["recovery_phase"] if correction else None
        self.wait_count = min(3, self.wait_count+1) if key == self.wait_key else int(key is not None)
        self.wait_key = key
        if correction:
            low=correction["low_level"]
            low["same_requirement_observations"]=self.wait_count
            low["escalation"]="连续3次仍缺相同证据：交接操作者/传感器适配器补齐所列观测；不自动重复动作" if self.wait_count>=3 else None

    def usage(self):
        return {"recovery_contexts":int(self.event is not None),
                "recovery_sensor_refs":sum(len(item.get("sensor_refs",[])) for item in self.event["evidence"]) if self.event else 0,
                "recovery_event_evidence":len(self.event["evidence"]) if self.event else 0,
                "recovery_state_slots":3}

def usable(facts, state, now):
    return (facts.get("valid") is True and facts.get("effect_usable") is True and
            facts.get("visibility") in ("clear","partial") and facts.get("timestamp")==now and
            state.get("observable") is True and not state.get("modality_conflict") and
            any(e.get("timestamp")==now for e in facts.get("evidence_refs",[])))

def readiness(facts, now):
    r=facts.get("grasp_readiness") or {}
    missing=[k for k in ("depth","orientation","contact_region") if r.get(k)!="confirmed"]
    source_ok = r.get("source") in ("measured_depth_and_contact","validated_controller_observation")
    if facts.get("provenance")=="synthetic_test_double":
        source_ok |= r.get("source")=="synthetic_independent_depth_contact"
    if not source_ok or r.get("timestamp")!=now or not isinstance(r.get("ref"),str) or not r["ref"]:
        missing.append("fresh_independent_readiness_source")
    return missing

def diagnose(event, state, facts):
    if event is None or event["state"] in ("resolved","retracted","superseded"):
        return None
    problem,mismatch=PROBLEMS[event["kind"]]
    return {"event_id":event["event_id"],"type":event["kind"],"problem":problem,
        "expected_effect":state["expected_effect"],"observed_deviation":mismatch,
        "observed_facts":event["evidence"],"current_evidence_status":event["observation_status"],
        "established_facts":["事件记录的是已观察效果偏差；当前是否仍成立另由本时刻证据判断"],
        "cause_hypotheses":[{"hypothesis":"对齐/接触未建立" if event["kind"]!="lost_grasp" else "意外丢失连接或有意放下",
            "certainty":"unconfirmed","missing_evidence":"独立深度、接触及实际执行反馈"}],
        "uncertainty":"不从闭爪、历史失败或图像平面位置推断夹持力、三维姿态及确定原因",
        "need_more_observation":True,"requested_evidence":["当前刀位置与两指关系","新尝试后随动或分离证据"],
        "confidence":None,"confidence_explanation":"规则满足程度；不是经校准的概率"}

def propose(event, diagnosis, state, facts, rules):
    if diagnosis is None:
        return None
    f=asdict(facts)
    now=facts.timestamp
    current=usable(f,state,now)
    location=facts.knife_location if current else "unknown"
    location_zh={"holder":"刀架","board":"砧板","elsewhere":"当前可见支撑位置","unknown":"待重新定位位置","gripper":"夹爪内"}[location]
    context = state.get("attempt_context") or {}
    support = context.get("support_location") if location == "gripper" else location
    support_zh = {"holder":"刀架", "board":"砧板", "elsewhere":"原观测支撑位置"}.get(support)
    lost=event["kind"]=="lost_grasp"
    strategy="reacquire_from_observed_support" if lost else ("restore_missing_precondition" if event["kind"]=="prerequisite_unmet" else "retry_tool_acquisition")
    goal_id=event["event_id"]+(":reacquire_detached_knife" if lost else ":restore_reliable_grasp")
    high={"event_id":event["event_id"],"recovery_goal_id":goal_id,"object":"knife",
        "blocked_subtask":"刀具归位" if state["preconditions"]["cut_effect_confirmed"] else "携刀转移与切割",
        "failed_precondition":"reliable_grasp","impact":"缺少可靠夹持时不能继续携刀操作",
        "strategy":strategy,"next_recovery_subgoal":f"在{location_zh}重新定位刀柄，建立接触准备并确认可靠夹持",
        "reason":diagnosis["observed_deviation"],
        "resume_when":"新观测连续支持刀与两指保持连接并连续随动" +
            (f"，移动时已离开{support_zh}" if support_zh else "") +
            "，再继续" + ("归位" if state["preconditions"]["cut_effect_confirmed"] else "携刀转移")}
    if current and location == "gripper":
        high["next_recovery_subgoal"] = "核查当前刀与两指连接，取得连续随动证据后再恢复可靠夹持前置条件"
    low={"event_id":event["event_id"],"recovery_goal_id":goal_id,"object":"knife","actuator":"right_gripper",
        "action_type":"hold_and_observe","recovery_phase":"observe_scene",
        "instruction":f"保持当前末端与夹爪设定，暂停后续携刀操作；取得{location_zh}的刀具及两指清晰新观测。",
        "coordinate_frame":"not_applicable_no_spatial_move","target_mode":"qualitative",
        "translation":None,"rotation":None,"numeric_command":None,"units":None,"numeric_source":None,
        "preconditions":[{"condition":"safe_hold","status":"requires_executor_check"}],
        "stop_condition":"取得当前刀具位置、两指关系及开闭状态，或交接操作者补观测",
        "post_checks":["刀当前在哪里","刀是否与夹爪分离","当前开闭状态"],
        "directly_executable":False,"handoff_requirements":["执行器检查保持/动作可行性与碰撞","需要物理移动时先提供标定映射"],
        "missing_information":["当前刀位置、两指接触关系和开闭状态"] if not current else []}
    empty=current and location in ("holder","board","elsewhere") and facts.contact=="separate"
    # 明确又打开且刀仍分离，表示这次接触检查已被实际动作中断，可以重新准备。
    # 不由建议增加 attempt；未知/遮挡仍只等待新证据。
    retry_pending=(event["state"]=="closed_for_retry" and state["attempt"]>event["attempt"] and
                   not (empty and facts.gripper=="open"))
    points=facts.knife_handle_xy is not None and facts.gripper_xy is not None
    if not current or event["state"]=="suspected":
        if current:
            low["missing_information"]=["连续新观测能否再次支持相同异常"]
    elif retry_pending or (current and location=="gripper"):
        low.update(recovery_phase="verify_grasp_effect",
            instruction="已观察到新的闭合尝试或接触，等待刀与两指连接及相对支撑面的随动证据；不把闭合当成成功。",
            post_checks=["刀是否离开原支撑位置","刀与夹爪是否连续保持连接"],
            missing_information=["连续接触/随动证据"])
    elif empty and facts.gripper=="closed":
        low.update(action_type="open_gripper",recovery_phase="prepare_open",
            coordinate_frame="right_gripper_actuator_no_translation",
            instruction=f"当前刀在{location_zh}且与夹爪分离；由执行器确认夹爪内无其他物体并可安全打开后，保持末端位姿，单独打开右夹爪。",
            preconditions=[{"condition":"current_knife_separate","status":"supported_by_current_observation"},
                           {"condition":"empty_of_other_objects_and_safe_open","status":"requires_executor_check"}],
            stop_condition="收到实际打开反馈；出现异常交回执行器",
            post_checks=["右夹爪实际打开",f"刀仍在{location_zh}","重新定位刀柄与两指"])
    elif empty and facts.gripper=="open" and points:
        deadband=rules["direction_deadband_normalized"]
        directions=image_alignment_directions(facts.knife_handle_xy,facts.gripper_xy,deadband)
        if directions:
            low.update(action_type="align_in_image",recovery_phase="align_image",coordinate_frame=facts.localization_frame,
                target_mode="relative_image_alignment",
                instruction="交接图像对齐目标："+"、".join(directions)+"；保持已观测打开的夹爪，深度、姿态和路径需独立验证后才转换为物理移动。",
                translation={"kind":"qualitative_image_plane","direction":directions,"magnitude":None},
                preconditions=[{"condition":"current_handle_and_finger_midpoint","status":"supported_by_current_observation"},
                               {"condition":"calibrated_mapping_depth_safe_path","status":"not_satisfied"}],
                stop_condition="新图像显示两指中点与刀柄二维对齐；丢失目标则停止",
                post_checks=["图像内对齐","独立深度/姿态/接触准备"],missing_information=["机器人坐标映射","深度与碰撞检查"])
        else:
            missing=readiness(f,now)
            low.update(action_type="verify_contact_readiness",recovery_phase="contact_preparation",
                instruction="二维对齐已满足；保持当前设定，取得独立的深度、刀柄姿态与可闭合接触区域证据。",
                post_checks=["深度已确认","姿态已确认","接触区域已确认"],missing_information=missing,
                stop_condition="三项准备条件均由新鲜独立观测确认")
            if not missing:
                low.update(action_type="close_gripper",recovery_phase="close_ready",
                    coordinate_frame="right_gripper_actuator_no_translation",
                    instruction="二维对齐与独立深度/姿态/接触准备均已确认；执行器复核安全后，保持末端位姿，单独闭合右夹爪。",
                    preconditions=[{"condition":"current_open_aligned_and_contact_ready","status":"supported_by_current_observation"},
                                   {"condition":"safe_close","status":"requires_executor_check"}],
                    stop_condition="收到实际闭合反馈或接触异常；闭合反馈不代表抓取成功",
                    post_checks=["实际闭合","刀与两指连接","随后连续随动证据"])
    elif current:
        low.update(action_type="relocalize",recovery_phase="localize",
            coordinate_frame="cam_high_and_cam_right_wrist_images",
            instruction=f"保持当前夹爪设定（开闭={facts.gripper}）与末端位姿；获取{location_zh}刀柄及两指的可定位图像，未定位前不发出移动。",
            post_checks=["当前刀柄与两指能否同时定位","当前夹爪开闭状态"],
            stop_condition="得到当前刀柄和两指定位及开闭状态",
            missing_information=["刀柄/两指的可靠图像定位"]+(["开闭状态"] if facts.gripper=="unknown" else []))
    return {"high_level":high,"low_level":low,
        "execution":{"mode":"advisory_replay","suggestion_generated":True,"actually_executed":False,
            "execution_receipt":None,"outcome_attributed_to_suggestion":False,"post_check_result":"pending_no_execution_receipt"}}

def validate_consistency(result):
    """独立执行门控与逐动作参数约束；任一不满足时拒绝写出。"""
    for execution in (result["execution"], (result.get("correction") or {}).get("execution",{})):
        if execution.get("actually_executed") or execution.get("execution_receipt") is not None or execution.get("outcome_attributed_to_suggestion"):
            raise ValueError("replay_must_not_claim_execution")
    suggestion=result["correction"]
    if suggestion is None:
        return
    high,low=suggestion["high_level"],suggestion["low_level"]
    allowed={"event_id","recovery_goal_id","object","actuator","action_type","recovery_phase","instruction",
             "coordinate_frame","target_mode","translation","rotation","numeric_command","units","numeric_source",
             "preconditions","stop_condition","post_checks","directly_executable","handoff_requirements",
             "missing_information","same_requirement_observations","escalation"}
    if set(low)-allowed:
        raise ValueError("undeclared_action_parameter")
    event=result.get("recovery_event") or result["failure_event"]
    if high["event_id"]!=event["event_id"] or high["event_id"]!=low["event_id"]:
        raise ValueError("correction_event_mismatch")
    if high["recovery_goal_id"]!=low["recovery_goal_id"] or high["object"]!=low["object"] or low["object"]!="knife" or low["actuator"]!="right_gripper":
        raise ValueError("correction_goal_object_mismatch")
    expected_goal=event["event_id"]+(":reacquire_detached_knife" if event["kind"]=="lost_grasp" else ":restore_reliable_grasp")
    expected_strategy={"lost_grasp":"reacquire_from_observed_support","prerequisite_unmet":"restore_missing_precondition","grasp_not_established":"retry_tool_acquisition"}[event["kind"]]
    if high["recovery_goal_id"]!=expected_goal or high["strategy"]!=expected_strategy or high["failed_precondition"]!="reliable_grasp":
        raise ValueError("strategy_goal_mismatch")
    frames={"hold_and_observe":"not_applicable_no_spatial_move","verify_contact_readiness":"not_applicable_no_spatial_move",
        "open_gripper":"right_gripper_actuator_no_translation","close_gripper":"right_gripper_actuator_no_translation",
        "relocalize":"cam_high_and_cam_right_wrist_images"}
    action=low["action_type"]
    if action not in frames and action!="align_in_image":
        raise ValueError("action_does_not_serve_restore_grasp")
    if action in frames and low["coordinate_frame"]!=frames[action]:
        raise ValueError("action_frame_mismatch")
    if not low["preconditions"] or not low["post_checks"] or not low["stop_condition"]:
        raise ValueError("missing_action_contract")
    if any(low[k] is not None for k in ("rotation","numeric_command","units","numeric_source")) or low["directly_executable"]:
        raise ValueError("unsupported_spatial_or_numeric_command")
    if action!="align_in_image" and low["translation"] is not None:
        raise ValueError("translation_for_non_spatial_action")
    f,state,now=result["visual_facts"],result["state"],result["timestamp"]
    current=usable(f,state,now)
    if action in ("open_gripper","close_gripper","align_in_image","verify_contact_readiness"):
        if not current or event["state"]=="suspected" or event["observation_status"]=="unobservable":
            raise ValueError("unusable_current_action_evidence")
        if f["knife_location"] not in ("holder","board","elsewhere") or f["contact"]!="separate":
            raise ValueError("knife_not_separate")
    if action=="open_gripper" and f["gripper"]!="closed":
        raise ValueError("open_requires_observed_closed")
    if action in ("close_gripper","align_in_image","verify_contact_readiness"):
        if f["gripper"]!="open" or f["knife_handle_xy"] is None or f["gripper_xy"] is None:
            raise ValueError("missing_open_or_localization")
    if action=="align_in_image":
        tr=low["translation"]
        if low["coordinate_frame"]!=f["localization_frame"] or low["coordinate_frame"] not in ("cam_high_image_x_right_y_down","cam_right_wrist_image_x_right_y_down"):
            raise ValueError("image_robot_coordinate_confusion")
        if not isinstance(tr,dict) or set(tr)!={"kind","direction","magnitude"} or tr["kind"]!="qualitative_image_plane" or tr["magnitude"] is not None:
            raise ValueError("unfounded_translation")
        if not tr["direction"] or not set(tr["direction"])<=set(("图像右侧","图像左侧","图像下方","图像上方")):
            raise ValueError("invalid_direction")
        deadband=result.get("decision_rules",{}).get("direction_deadband_normalized",50)
        expected=image_alignment_directions(f["knife_handle_xy"],f["gripper_xy"],deadband)
        if tr["direction"]!=expected:
            raise ValueError("direction_not_supported_by_points")
    if action=="close_gripper":
        deadband=result.get("decision_rules",{}).get("direction_deadband_normalized",50)
        if readiness(f,now) or any(abs(a-b)>deadband for a,b in zip(f["knife_handle_xy"],f["gripper_xy"])):
            raise ValueError("close_requires_independent_readiness_and_alignment")
    required={"open_gripper":{"current_knife_separate","empty_of_other_objects_and_safe_open"},
              "close_gripper":{"current_open_aligned_and_contact_ready","safe_close"},
              "align_in_image":{"current_handle_and_finger_midpoint","calibrated_mapping_depth_safe_path"}}
    if action in required and {p["condition"] for p in low["preconditions"]}!=required[action]:
        raise ValueError("missing_action_specific_precondition")
    statuses={"current_knife_separate":"supported_by_current_observation",
              "current_open_aligned_and_contact_ready":"supported_by_current_observation",
              "current_handle_and_finger_midpoint":"supported_by_current_observation",
              "empty_of_other_objects_and_safe_open":"requires_executor_check","safe_close":"requires_executor_check",
              "calibrated_mapping_depth_safe_path":"not_satisfied","safe_hold":"requires_executor_check"}
    if any(p.get("condition") not in statuses or p.get("status")!=statuses[p["condition"]] for p in low["preconditions"]):
        raise ValueError("unverified_precondition_claim")
    if action in ("open_gripper","close_gripper","align_in_image"):
        for name in (("cam_right_wrist",) if f["localization_frame"].startswith("cam_right") and action!="open_gripper" else ("cam_high",)):
            m=result.get("modalities",{}).get(name,{})
            if m.get("status")!="available" or m.get("timestamp") is None or not 0<=now-m["timestamp"]<=.12:
                raise ValueError("stale_or_missing_action_modality")

