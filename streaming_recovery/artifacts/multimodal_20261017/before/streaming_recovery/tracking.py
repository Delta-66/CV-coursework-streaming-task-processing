"""任务效果状态：历史成就、当前谓词、当前尝试分别存储，全部容量有界。"""
from __future__ import annotations
from .contracts import evidence
from .task_spec import supports, SPEC

STAGES = {
    "locate_tool": "寻找与接近刀具", "align_grasp": "刀柄对齐与抓取准备",
    "verify_grasp": "抓取效果确认", "transport_tool": "携刀转移",
    "cutting": "已携刀进入工作区，切割效果待确认", "cut_operation": "观察到刀具作用于香肠",
    "verify_cut": "新切割效果确认", "return_tool": "刀具归位",
    "verify_completion": "释放与完成确认", "completed": "当前完成条件满足",
    "unknown": "当前阶段暂不可判断",
    "blocked_precondition": "后续操作被夹持前置条件阻断",
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
            self.armed, self.last_gripper, self.last_visual_gripper, self.previous_holder = False, None, "unknown", False
        if good and (visually_open or raw_open):
            self.armed = True
        closing_edge = ((raw is not None and self.last_gripper is not None and
                         self.last_gripper-raw >= self.rules["close_delta_raw"] and raw_closed) or
                        (independent and visually_closed and self.last_visual_gripper == "open"))
        new_attempt = bool(good and self.armed and closing_edge and near_object)
        if new_attempt:
            self.attempt += 1
            self.armed, self.attempt_held, self.attempt_started = False, False, now
            self.held = old_held = None
            for key in ("grasp", "lost_grasp"):
                self.memory.anchors.pop(key, None)
            self.memory.anchor("attempt", evidence(facts, "观测到目标附近夹爪实际从开转闭；仅证明新抓取尝试"))
        self.last_gripper, self.last_visual_gripper = raw if good else None, facts.gripper if good else "unknown"
        self.last_time, self.previous_holder = now, near_object
        known, unknown, current_evidence = [], [], []
        if raw is not None:
            known.append(f"右夹爪观测原始值 {raw:.5f}（数据集原始单位；不代表夹持成功）")
        else:
            unknown.append("右夹爪状态未到达或无效")
        if not independent:
            known.append("夹爪开闭来自 observation.state[13]，不属于独立视觉互证")
        if conflict:
            unknown.append("独立视觉与夹爪原始观测冲突")
        if not good:
            self.held, self.returned, self.stage = None, None, "unknown"
            unknown.append("当前视觉证据不可用：" + (facts.error or facts.visibility))
            return self._result(known, unknown, current_evidence, None, False, False, conflict, new_attempt)
        known.append(f"视觉观察：{facts.description}")
        current_evidence.append(evidence(facts, f"knife_location={facts.knife_location}; contact={facts.contact}; motion={facts.motion}"))
        if self.cut_reference is None and not self.ever_held and facts.cut_piece in ("yes", "no"):
            self.cut_reference = {"timestamp": now, "ref": facts.ref, "cut_piece": facts.cut_piece}
            self.memory.anchor("cut_reference", evidence(facts, "切割前布局；已有小块=" + facts.cut_piece))
        attached = lambda f: f.knife_location == "gripper" and f.contact in ("between_fingers", "co_moving") and f.released != "yes"
        held_stable, _ = self._stable(attached, now)
        grasp = self.memory.anchors.get("grasp")
        recent_grasp = grasp is not None and 0 <= now-grasp["timestamp"] <= self.rules.get("grasp_evidence_ttl_s", 4.)
        if held_stable:
            self.held, self.ever_held, self.attempt_held = True, True, True
            self.memory.anchor("grasp", evidence(facts, "连续图像支持刀具与夹爪连接；闭爪值本身不是成功依据"))
            self.memory.anchors.pop("lost_grasp", None)
            known.append("连续视觉接触/随动证据支持当前持刀")
        elif (old_held is True and recent_grasp and facts.measurements.get("tool_on_board") and
              facts.measurements.get("robot_at_board") and raw_closed and facts.knife_location != "board"):
            self.held = True
            known.append("近期夹持证据有效且当前未出现分离反证，维持有条件的夹持假设")
        elif facts.knife_location in ("holder", "board", "elsewhere") and facts.contact == "separate":
            self.held = False
            if facts.knife_location == "holder" and facts.released == "yes":
                # 已明确回架释放，旧连接已经结束；它与短暂不可见不同。
                self.memory.anchors.pop("grasp", None)
                recent_grasp = False
        else:
            self.held = None
            unknown.append("当前可靠夹持尚未确认；历史夹持不等于当前持有")
        if self.held is True and facts.motion == "at_board" and facts.measurements.get("cut_operation") == "yes":
            if self.operation_at is None:
                self.operation_at = now
                self.memory.anchor("cut_operation", evidence(facts, "观察到持刀在香肠位置作用；尚不等于切开"))
        def new_effect(f):
            reference = f.measurements.get("cut_reference_timestamp")
            explicit = (f.measurements.get("new_cut_effect") == "yes" and isinstance(reference, (int, float)) and
                        self.operation_at is not None and reference < self.operation_at <= f.timestamp)
            transition = (self.cut_reference is not None and self.cut_reference["cut_piece"] == "no" and
                          self.operation_at is not None and self.cut_reference["timestamp"] < self.operation_at <= f.timestamp)
            return f.cut_piece == "yes" and (explicit or transition)
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
        departed_empty = (facts.knife_location == "holder" and facts.contact == "separate" and
                          facts.motion in ("at_board", "to_board", "away") and (self.attempt > 0 or visually_closed))
        if not current_complete and departed_empty and not self.attempt_held:
            candidate = "grasp_not_established" if self.attempt > 0 else "prerequisite_unmet"
        detached = facts.knife_location in ("board", "elsewhere") and facts.contact == "separate"
        if detached and (recent_grasp or "lost_grasp" in self.memory.anchors):
            candidate = "lost_grasp"
            if "lost_grasp" not in self.memory.anchors:
                self.memory.anchor("lost_grasp", evidence(facts, "最近可靠夹持之后观察到分离；原因未确定"))
                current_evidence.append(grasp)
        if current_complete:
            self.stage = "completed"
        elif candidate == "prerequisite_unmet":
            self.stage = "blocked_precondition"
        elif facts.knife_location == "gripper" and facts.motion == "to_board":
            self.stage = "transport_tool"
        elif self.held is True:
            if self.cut_confirmed:
                self.stage = "return_tool"
            elif facts.measurements.get("cut_operation") == "yes":
                self.stage = "cut_operation"
            else:
                self.stage = "cutting" if facts.motion == "at_board" else "transport_tool"
        elif self.returned or (self.cut_confirmed and facts.knife_location == "holder" and facts.released == "yes"):
            self.stage = "verify_completion"
        elif self.cut_confirmed and facts.motion == "returning_holder":
            self.stage = "return_tool"
        elif self.ever_held and self.held is None:
            self.stage = "unknown"
        elif self.attempt > 0:
            self.stage = "align_grasp" if near_object and (visually_open or raw_open) else "verify_grasp"
        else:
            self.stage = "align_grasp" if near_object else "locate_tool"
        if not self.cut_confirmed:
            unknown.append("本次新切割效果尚未确认；已有小块不能补齐")
        if self.returned is not True:
            unknown.append("当前刀具归位与释放尚未确认")
        return self._result(known, unknown, current_evidence, candidate, held_stable, True, conflict, new_attempt)

    def _result(self, known, unknown, current_evidence, candidate, recovered, observable, conflict, new_attempt):
        goals = []
        if self.ever_held:
            goals.append("曾建立刀具夹持")
        if self.cut_confirmed:
            goals.append("本次操作的新切割效果曾获确认")
        if self.returned:
            goals.append("当前刀具已归位且释放")
        expectations = {
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
        }
        return {"task_spec": SPEC, "stage": self.stage, "stage_zh": STAGES[self.stage], "subtask": STAGES[self.stage],
                "confirmed_subgoals": goals, "preconditions": {"reliable_grasp": self.held,
                    "cut_effect_confirmed": self.cut_confirmed, "tool_returned": self.returned},
                "known_facts": known, "unknowns": unknown, "evidence": current_evidence,
                "expected_effect": expectations[self.stage], "candidate": candidate, "recovered": recovered,
                "observable": observable, "modality_conflict": conflict, "new_attempt_observed": new_attempt,
                "attempt": self.attempt, "attempt_held": self.attempt_held,
                "task_completed": self.cut_confirmed and self.returned is True,
                "historical_completion": self.completed, "cut_reference": self.cut_reference,
                "anchors": dict(self.memory.anchors)}

    def usage(self):
        return {"task_state_slots": len([k for k in vars(self) if k not in ("rules", "memory")]),
                "task_reference_entries": int(self.cut_reference is not None)}

