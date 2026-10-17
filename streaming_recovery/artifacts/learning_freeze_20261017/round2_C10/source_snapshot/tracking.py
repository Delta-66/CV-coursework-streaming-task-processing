"""任务效果状态：历史成就、当前谓词、当前尝试分别存储，全部容量有界。"""
from __future__ import annotations
from copy import deepcopy
from .contracts import evidence
from .task_spec import supports, SPEC
from .effects import CutEffectTracker
from .attempts import attempt_transition

STAGES = {
    "locate_tool": "寻找与接近刀具", "align_grasp": "刀柄对齐与抓取准备",
    "verify_grasp": "抓取效果确认", "transport_tool": "携刀转移",
    "cutting": "已携刀进入工作区，切割效果待确认", "cut_operation": "观察到刀具作用于香肠",
    "verify_cut": "新切割效果确认", "return_tool": "刀具归位",
    "verify_completion": "释放与完成确认", "completed": "当前完成条件满足",
    "unknown": "当前阶段暂不可判断",
    "blocked_precondition": "后续操作被夹持前置条件阻断",
    "approach_tool": "右臂移动接近刀具，抓取准备尚在进行",
    "holding_tool": "持刀停留，等待后续移动或效果确认",
    "blocked_cut_effect": "工作区操作后切割效果未达成",
}

class TaskTracker:
    def __init__(self, rules, memory):
        self.rules, self.memory = rules, memory
        self.stage = "locate_tool"
        self.attempt = 0
        self.armed = False
        self.last_gripper = None
        self.last_visual_gripper = "unknown"
        self.last_time = None
        self.held = None
        self.ever_held = False
        self.cut_confirmed = False
        self.returned = None
        self.completed = False  # 历史完成；task_completed 输出另检查当前谓词。
        self.previous_holder = False
        self.attempt_held = False
        self.attempt_started = None
        self.cut_reference = None
        self.operation_at = None
        # 仅保存最新一次实际尝试：6 个固定字段，无累计尝试历史。
        self.attempt_context = None
        self.effect_state = CutEffectTracker(rules)
        self.work_cycle = None  # one bounded observed cycle, never a recovery plan

    supports = staticmethod(supports)

    def _stable(self, predicate, now):
        items = []
        for f in reversed(self.memory.features):
            if not f.valid or not f.effect_usable or f.visibility in ("occluded", "unknown") or not predicate(f):
                break
            if self.attempt_started is not None and f.timestamp < self.attempt_started:
                break
            if items and items[-1].timestamp-f.timestamp > self.rules["max_evidence_gap_s"]:
                break
            items.append(f)
        return (len(items) >= self.rules["confirm_samples"] and now-items[-1].timestamp+1e-6 >= self.rules["confirm_span_s"]), items

    def update(self, observation, facts):
        now, old_held = observation.timestamp, self.held
        good = facts.valid and facts.effect_usable and facts.visibility not in ("occluded", "unknown")
        gap = self.last_time is not None and now-self.last_time > self.rules["max_evidence_gap_s"]
        raw = observation.observation_state[13] if observation.observation_state else None
        raw_open = raw is not None and raw >= self.rules["gripper_open_raw"]
        raw_closed = raw is not None and raw <= self.rules["gripper_closed_raw"]
        independent = facts.gripper_source == "independent_visual"
        visually_open = good and facts.gripper == "open"
        visually_closed = good and facts.gripper == "closed"
        conflict = good and independent and ((raw_open and visually_closed) or (raw_closed and visually_open))
        good = good and not conflict
        facts.effect_usable = good
        near_holder = good and facts.motion in ("at_holder", "approaching_holder")
        near_object = near_holder or (good and facts.knife_location in ("board", "elsewhere") and facts.knife_handle_xy is not None)
        if gap:
            self.work_cycle=None
            self.armed, self.last_gripper, self.last_visual_gripper, self.previous_holder = False, None, "unknown", False
        self.armed, closing_edge, new_attempt = attempt_transition(
            self.armed, raw, facts.gripper, independent, good, gap, near_object, self.rules)
        if new_attempt:
            self.attempt += 1
            self.armed, self.attempt_held, self.attempt_started = False, False, now
            self.attempt_context = {
                "object":"knife", "support_location":facts.knife_location if facts.knife_location in ("holder","board","elsewhere") else "unknown",
                "started_at":now, "ref":facts.ref,
                "expected_effect":"刀与两指建立连接并在移动时离开原支撑面；仅闭合或等待不证明成败",
                "localization_frame":facts.localization_frame,
            }
            self.held = old_held = None
            for key in ("grasp", "lost_grasp"):
                self.memory.anchors.pop(key, None)
            self.memory.anchor("attempt", evidence(facts, "观测到目标附近夹爪实际从开转闭；仅证明新抓取尝试"))
        self.last_gripper, self.last_visual_gripper = raw if good else None, facts.gripper if good else "unknown"
        self.last_time, self.previous_holder = now, near_object
        known, unknown, current_evidence, evidence_conflicts = [], [], [], []
        joint = facts.measurements.get("joint_motion", {"status":"unknown", "evidence_refs":[]})
        active = joint["status"] in ("moving", "recent_motion")
        if joint["status"] == "unknown":
            unknown.append("右臂运动状态信息不足；需要相邻实测关节位置和当前速度")
        else:
            known.append("实测右臂六关节（排除夹爪维）运动状态=" + joint["status"])
            current_evidence.append({"timestamp":now,"ref":observation.ref,
                "claim":"right_arm_motion="+joint["status"],"source":"measured_qpos_qvel",
                "images":[],"sensor_refs":joint["evidence_refs"]})
        if raw is not None:
            known.append(f"右夹爪观测原始值 {raw:.5f}（数据集原始单位；不代表夹持成功）")
        else:
            unknown.append("右夹爪状态未到达或无效")
        if not independent:
            known.append("夹爪开闭来自 observation.state[13]，不属于独立视觉互证")
        if conflict:
            unknown.append("独立视觉与夹爪原始观测冲突")
        if not good:
            if self.work_cycle and self.work_cycle["exit_at"] is None:
                self.work_cycle=None
            if not self.cut_confirmed:
                self.effect_state.interrupt(now)
                self.operation_at = None
            else:
                self.effect_state.last_time = now
            facts.measurements["cut_effect_order"] = self.effect_state.summary(reason=
                "historical_confirmed_effect_current_observation_unavailable" if self.cut_confirmed else "current_observation_unavailable")
            self.held, self.returned, self.stage = None, None, "unknown"
            unknown.append("当前视觉证据不可用：" + (facts.error or facts.visibility))
            return self._result(known, unknown, current_evidence, None, False, False, conflict, new_attempt, evidence_conflicts)
        known.append(f"视觉观察：{facts.description}")
        current_evidence.append(evidence(facts, f"knife_location={facts.knife_location}; contact={facts.contact}; motion={facts.motion}"))
        if self.cut_reference is None and not self.ever_held and facts.cut_piece in ("yes", "no"):
            self.cut_reference = {"timestamp": now, "ref": facts.ref, "cut_piece": facts.cut_piece}
            self.memory.anchor("cut_reference", evidence(facts, "切割前布局；已有小块=" + facts.cut_piece))
        attached = lambda f: f.knife_location == "gripper" and f.contact in ("between_fingers", "co_moving") and f.released != "yes"
        held_stable, _ = self._stable(attached, now)
        grasp = self.memory.anchors.get("grasp")
        recent_grasp = grasp is not None and 0 <= now-grasp["timestamp"] <= self.rules.get("grasp_evidence_ttl_s", 4.)
        explicit_separation = facts.contact == "separate" or facts.released == "yes"
        if explicit_separation:
            # 清晰分离/释放是当前反证，优先于旧夹持与粗区域存在代理。
            self.held = False
            self.memory.anchors.pop("grasp", None)
            if facts.measurements.get("tool_on_board") and facts.measurements.get("robot_at_board"):
                evidence_conflicts.append({"timestamp":now, "ref":facts.ref,
                    "type":"coarse_presence_vs_explicit_separation",
                    "resolution":"优先采用当前刀与夹爪分离/释放证据；区域共现不能维持夹持"})
                known.append("粗区域共现与明确分离反证冲突：当前可靠夹持为否，不继续沿用夹持锚点")
            if facts.knife_location == "holder" and facts.released == "yes":
                # 正常回架释放明确结束连接，不可随后误用为失抓前件。
                recent_grasp = False
                self.memory.anchors.pop("lost_grasp", None)
        elif held_stable:
            self.held, self.ever_held, self.attempt_held = True, True, True
            self.memory.anchor("grasp", evidence(facts, "连续图像支持刀具与夹爪连接；闭爪值本身不是成功依据"))
            self.memory.anchors.pop("lost_grasp", None)
            known.append("连续视觉接触/随动证据支持当前持刀")
        elif (old_held is True and recent_grasp and facts.measurements.get("tool_on_board") and
              facts.measurements.get("robot_at_board") and raw_closed and facts.knife_location != "board" and
              facts.contact != "separate" and facts.released != "yes"):
            self.held = True
            known.append("近期夹持证据有效且当前未出现分离反证，维持有条件的夹持假设")
        else:
            self.held = None
            unknown.append("当前可靠夹持尚未确认；历史夹持不等于当前持有")
        operation = self.held is True and facts.motion == "at_board" and facts.measurements.get("cut_operation") == "yes"
        facts.measurements["cut_effect_order"] = self.effect_state.observe(facts,operation,self.cut_confirmed)
        if self.effect_state.operation is not None:
            if self.operation_at != self.effect_state.operation["timestamp"]:
                self.operation_at = self.effect_state.operation["timestamp"]
                self.memory.anchor("cut_operation", evidence(facts, "观察到持刀作用；与此前最近可用布局关联，尚不等于切开"))
        elif not self.cut_confirmed:
            self.operation_at = None
        if self.held is True and facts.motion == "at_board":
            self.memory.anchor("work_area", evidence(facts, "已观察到持刀进入砧板工作区；不等于完成切割"))
            if self.rules.get("observe_work_exit",False):
                self.work_cycle={"work_at":now,"work_ref":facts.ref,"exit_at":None,"exit_ref":None}
        elif (self.work_cycle is not None and self.work_cycle["exit_at"] is None and
              not facts.measurements.get("robot_at_board",True) and active and
              facts.cut_piece=="no" and facts.measurements.get("cut_layout",{}).get("clear") is True):
            self.work_cycle.update(exit_at=now,exit_ref=facts.ref)
        def new_effect(f):
            ordered=f.measurements.get("cut_effect_order",{})
            reference=ordered.get("operation_ref")
            return (f.cut_piece=="yes" and ordered.get("supports_current_operation") is True and
                    reference is not None and reference["timestamp"]==self.operation_at)
        piece_stable, _ = self._stable(new_effect, now)
        if self.ever_held and piece_stable and self.operation_at is not None:
            self.cut_confirmed = True
            self.memory.anchor("cut", evidence(facts, "切割前参考→刀作用→新增分离小段，连续新图像确认"))
        release_stable, _ = self._stable(lambda f: f.knife_location == "holder" and f.released == "yes" and f.contact == "separate", now)
        if facts.knife_location == "holder" and facts.released == "yes" and facts.contact == "separate":
            self.returned = bool(self.ever_held and release_stable)
        elif facts.knife_location in ("gripper", "board", "elsewhere"):
            self.returned = False
        else:
            self.returned = None
        if self.returned:
            self.memory.anchor("return", evidence(facts, "当前刀具在刀架且夹爪分离"))
        current_complete = self.cut_confirmed and self.returned is True
        if current_complete and not self.completed:
            self.completed = True
            self.memory.anchor("completion", evidence(facts, "历史首次同时满足新切割效果、归位与释放"))
        candidate = None
        cycle=self.work_cycle
        if (cycle and cycle["exit_at"] is not None and not self.cut_confirmed and
            facts.cut_piece=="no" and facts.measurements.get("cut_layout",{}).get("clear") is True and
            not facts.measurements.get("robot_at_board",True)):
            candidate="work_effect_not_achieved"
            current_evidence.append({"timestamp":now,"ref":facts.ref,"claim":"observed_work_exit_then_clear_no_cut_piece",
                "work_at":cycle["work_at"],"work_ref":cycle["work_ref"],"exit_at":cycle["exit_at"],"exit_ref":cycle["exit_ref"],
                "images":[],"sensor_refs":[]})
        departed_empty = (facts.knife_location == "holder" and facts.contact == "separate" and
                          facts.motion in ("at_board", "to_board", "away") and
                          (self.attempt > 0 or visually_closed or (facts.motion=="at_board" and active)))
        if not current_complete and departed_empty and not self.attempt_held:
            candidate = "grasp_not_established" if self.attempt > 0 else "prerequisite_unmet"
        # 曾持刀并不等于现在持刀：提前回架后，明确空手继续切割需要重新建立前置条件。
        # 回架/释放本身、驻留或视觉不明均不足以触发此分支。
        if (not self.cut_confirmed and self.held is False and self.attempt_held and
                facts.knife_location == "holder" and facts.contact == "separate" and
                facts.motion in ("to_board", "at_board") and active):
            candidate = "prerequisite_unmet"
        def retry_miss(f):
            context = self.attempt_context
            return bool(context and context["support_location"] in ("board","elsewhere") and
                f.timestamp > context["started_at"] and f.knife_location == context["support_location"] and
                f.contact == "separate" and f.gripper == "closed" and
                f.localization_frame == context["localization_frame"] and
                f.knife_handle_xy is not None and f.gripper_xy is not None and
                any(abs(a-b) > self.rules["direction_deadband_normalized"]
                    for a,b in zip(f.knife_handle_xy, f.gripper_xy)))
        retry_missed, _ = self._stable(retry_miss, now)
        if not current_complete and not self.attempt_held and self.held is False and retry_missed:
            candidate = "grasp_not_established"
            current_evidence.append(evidence(facts,
                "实际重抓后连续清晰观测：刀仍在原支撑面且与闭合两指显著分离；不是仅由等待或尚未抬起判断失败"))
        detached = facts.knife_location in ("board", "elsewhere") and facts.contact == "separate"
        if detached and (recent_grasp or "lost_grasp" in self.memory.anchors):
            candidate = "lost_grasp"
            if "lost_grasp" not in self.memory.anchors:
                self.memory.anchor("lost_grasp", evidence(facts, "最近可靠夹持之后观察到分离；原因未确定"))
                current_evidence.append(grasp)
        if current_complete:
            self.stage = "completed"
        elif candidate == "work_effect_not_achieved":
            self.stage = "blocked_cut_effect"
        elif candidate == "prerequisite_unmet":
            self.stage = "blocked_precondition"
        elif facts.knife_location == "gripper" and facts.motion == "to_board":
            self.stage = ("return_tool" if self.cut_confirmed else
                          "verify_cut" if "work_area" in self.memory.anchors else "transport_tool")
        elif self.held is True:
            if self.cut_confirmed or (facts.motion == "returning_holder" and "work_area" in self.memory.anchors):
                self.stage = "return_tool"
            elif facts.measurements.get("cut_operation") == "yes":
                self.stage = "cut_operation"
            else:
                self.stage = "cutting" if facts.motion == "at_board" else "transport_tool"
        elif self.returned or (self.ever_held and facts.knife_location == "holder" and facts.released == "yes" and facts.contact == "separate"):
            self.stage = "verify_completion"
        elif self.cut_confirmed and facts.motion == "returning_holder":
            self.stage = "return_tool"
        elif facts.knife_location=="gripper" and self.held is None:
            self.stage="verify_grasp"
        elif self.ever_held and self.held is None:
            self.stage = "unknown"
        elif self.attempt > 0:
            self.stage = "align_grasp" if near_object and (visually_open or raw_open) else "verify_grasp"
        else:
            self.stage = "align_grasp" if near_object else "locate_tool"
        # 关节只限定动作是否正在发生，绝不据此建立物体夹持或切开事实。
        if self.stage == "transport_tool" and joint["status"] == "stationary":
            self.stage = "holding_tool" if self.held is True else "verify_grasp"
        elif (self.stage == "align_grasp" and active and not new_attempt and
              (facts.knife_handle_xy is None or facts.gripper_xy is None)):
            self.stage = "approach_tool"
        if not self.cut_confirmed:
            unknown.append("本次新切割效果尚未确认；已有小块不能补齐")
        if self.returned is not True:
            unknown.append("当前刀具归位与释放尚未确认")
        return self._result(known, unknown, current_evidence, candidate, held_stable and not explicit_separation, True, conflict, new_attempt, evidence_conflicts)

    def _result(self, known, unknown, current_evidence, candidate, recovered, observable, conflict, new_attempt, evidence_conflicts):
        goals = []
        if self.ever_held:
            goals.append("曾建立刀具夹持")
        if self.cut_confirmed:
            goals.append("本次操作的新切割效果曾获确认")
        if self.returned:
            goals.append("当前刀具已归位且释放")
        expectations = {
            "approach_tool": "右臂在移动；观察刀柄与两指关系，等待实际抓取尝试",
            "holding_tool": "当前关节低于运动阈值；继续观察连接，不把停留判成失败或搬运进展",
            "locate_tool": "定位刀柄和两指，不要求刀已离架",
            "align_grasp": "确认二维对齐、深度、姿态与接触准备，然后尝试闭合",
            "verify_grasp": "刀应与夹爪连接并离开支撑位置；闭爪不等于成功",
            "transport_tool": "刀与夹爪保持连接并向砧板转移",
            "cutting": "观察实际刀具作用与新增分离效果",
            "cut_operation": "刀作用之后出现相对切割前参考新增的小段",
            "verify_cut": "连续清晰观测确认新增小段",
            "return_tool": "携刀回到刀架，再观察释放",
            "verify_completion": "补足本次切割效果、当前归位与释放",
            "completed": "当前完成条件满足；继续观测时重新核查当前位置",
            "unknown": "补充当前刀具、两指关系与香肠布局的清晰观测",
            "blocked_precondition": "停止空手进行后续操作，恢复取刀前置条件",
            "blocked_cut_effect": "工作区操作后应形成新分离小段；复核当前刀具与切割目标关系",
        }
        return {"task_spec": SPEC, "stage": self.stage, "stage_zh": STAGES[self.stage], "subtask": STAGES[self.stage],
                "confirmed_subgoals": goals, "preconditions": {"reliable_grasp": self.held,
                    "cut_effect_confirmed": self.cut_confirmed, "tool_returned": self.returned},
                "known_facts": known, "unknowns": unknown, "evidence": current_evidence,
                "expected_effect": expectations[self.stage], "candidate": candidate, "recovered": recovered,
                "observable": observable, "modality_conflict": conflict, "new_attempt_observed": new_attempt,
                "attempt": self.attempt, "attempt_held": self.attempt_held,
                "attempt_context": dict(self.attempt_context) if self.attempt_context is not None else None,
                "evidence_conflicts": evidence_conflicts,
                "cut_effect_order": deepcopy(self.memory.features[-1].measurements.get("cut_effect_order",self.effect_state.summary())),
                "task_completed": self.cut_confirmed and self.returned is True,
                "historical_completion": self.completed, "cut_reference": self.cut_reference,
                "arm_activity": (self.memory.features[-1].measurements.get("joint_motion",{}).get("status","unknown") if self.memory.features else "unknown"),
                "work_cycle":deepcopy(self.work_cycle),
                "anchors": dict(self.memory.anchors)}

    def usage(self):
        return {"task_state_slots": len([k for k in vars(self) if k not in ("rules", "memory")]),
                "task_reference_entries": int(self.cut_reference is not None),
                "attempt_context_entries":int(self.attempt_context is not None),
                "work_cycle_entries":int(self.work_cycle is not None),
                "work_cycle_fields":len(self.work_cycle) if self.work_cycle else 0,
                "attempt_context_fields":len(self.attempt_context) if self.attempt_context else 0} | self.effect_state.usage()

