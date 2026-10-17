"""任务专用、可解释的效果跟踪；既不把闭爪当成功，也不按视频长度推进阶段。"""
from __future__ import annotations

from .contracts import evidence


STAGES = {
    "locate_tool": "寻找与接近刀具", "align_grasp": "刀柄对齐与抓取准备",
    "verify_grasp": "抓取效果确认", "transport_tool": "携刀转移",
    "cutting": "已携刀进入工作区，切割效果待确认", "return_tool": "刀具归位",
    "verify_completion": "释放与完成确认", "completed": "任务完成",
    "unknown": "当前阶段暂不可判断",
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
        self.returned = False
        self.completed = False
        self.previous_holder = False

    @staticmethod
    def supports(instruction):
        text = instruction.lower()
        return all(word in text for word in ("right", "knife", "sausage", "holder"))

    def _stable(self, predicate, now):
        items = []
        for f in reversed(self.memory.features):
            if not f.valid or not f.effect_usable or f.visibility in ("occluded", "unknown") or not predicate(f):
                break
            if items and items[-1].timestamp - f.timestamp > self.rules["max_evidence_gap_s"]:
                break
            items.append(f)
        return (len(items) >= self.rules["confirm_samples"] and
                now - items[-1].timestamp >= self.rules["confirm_span_s"]), items

    def update(self, observation, facts):
        now, old_held = observation.timestamp, self.held
        good = facts.valid and facts.visibility not in ("occluded", "unknown")
        gap = self.last_time is not None and now - self.last_time > self.rules["max_evidence_gap_s"]
        raw = observation.observation_state[13] if observation.observation_state else None
        visually_open = good and facts.gripper == "open"
        visually_closed = good and facts.gripper == "closed"
        raw_open = raw is not None and raw >= self.rules["gripper_open_raw"]
        raw_closed = raw is not None and raw <= self.rules["gripper_closed_raw"]
        conflict = good and ((raw_open and visually_closed) or (raw_closed and visually_open))
        facts.effect_usable = good and not conflict
        near_holder = good and facts.motion in ("at_holder", "approaching_holder")
        if gap:
            self.armed = False
            self.last_gripper = None
            self.last_visual_gripper = "unknown"
            self.previous_holder = False
        if (visually_open or raw_open) and not conflict:
            self.armed = True
        closing_edge = ((raw is not None and self.last_gripper is not None and
                         self.last_gripper - raw >= self.rules["close_delta_raw"] and raw_closed) or
                        (visually_closed and self.last_visual_gripper == "open"))
        new_attempt = self.armed and closing_edge and (near_holder or self.previous_holder) and not conflict
        if new_attempt:
            self.attempt += 1
            self.armed = False
            self.memory.anchor("attempt", evidence(facts, "观测到刀架附近夹爪从打开转为闭合；仅证明抓取尝试"))
        self.last_gripper, self.last_visual_gripper = raw, facts.gripper if good else "unknown"
        self.last_time, self.previous_holder = now, near_holder
        known, unknown, current_evidence = [], [], []
        if raw is not None:
            known.append(f"右夹爪观测原始值 {raw:.5f}（数据集原始单位；不代表夹持成功）")
        else:
            unknown.append("右夹爪状态未到达或无效")
        if conflict:
            unknown.append("图像与夹爪原始阈值状态冲突；不据此确认抓取或失败")
        if not good:
            unknown.append("视觉证据不足：" + (facts.error or facts.visibility))
            self.held = None
            if not self.completed:
                self.stage = "unknown"
            return self._result(known, unknown, current_evidence, None, False, False, conflict, new_attempt)

        known.append(f"视觉模型观察：{facts.description}")
        current_evidence.append(evidence(facts, f"knife_location={facts.knife_location}; contact={facts.contact}; motion={facts.motion}"))
        attached = lambda f: f.knife_location == "gripper" and f.contact in ("between_fingers", "co_moving") and f.released != "yes"
        held_stable, held_items = self._stable(attached, now)
        # 可见刀已离架且连续位于两指间，与闭合值本身分开判断。
        if held_stable and not conflict:
            self.held, self.ever_held = True, True
            self.memory.anchor("grasp", evidence(facts, "连续图像支持刀已离架，并通过两指内接触或刀/机器人共同运动确认夹持"))
            known.append("刀具夹持已获连续视觉证据支持（未测量夹持力）")
        elif (old_held is True and facts.measurements.get("tool_on_board") and
              facts.measurements.get("robot_at_board") and raw_closed and not conflict and
              "grasp" in self.memory.anchors and now-self.memory.anchors["grasp"]["timestamp"]<=self.memory.config["window_s"]):
            self.held = True
            known.append("近期共同运动证据仍在容量窗口内，当前仍见工作区刀具与闭合夹爪；维持夹持假设")
        elif facts.knife_location in ("holder", "board", "elsewhere") and facts.contact == "separate":
            self.held = False
        else:
            self.held = None
            unknown.append("当前可靠夹持尚未确认")

        piece_stable, _ = self._stable(lambda f: f.cut_piece == "yes", now)
        if self.ever_held and piece_stable:
            self.cut_confirmed = True
            self.memory.anchor("cut", evidence(facts, "在已取刀之后，连续图像出现分离的小段香肠"))
        release_stable, _ = self._stable(lambda f: f.knife_location == "holder" and f.released == "yes" and f.contact == "separate", now)
        if self.ever_held and release_stable:
            self.returned = True
            self.memory.anchor("return", evidence(facts, "刀具重新位于刀架且夹爪与其分离"))
        if self.cut_confirmed and self.returned and release_stable:
            self.completed = True
            self.memory.anchor("completion", evidence(facts, "切割效果、刀归位、释放条件同时已获证据"))

        candidate = None
        if not conflict and not self.completed:
            # 仅在机器人已经进入工作区、但刀仍留在刀架时判定效果偏差；单纯等待不失败。
            departed_empty = (facts.knife_location == "holder" and facts.contact == "separate" and
                              facts.motion in ("at_board", "to_board", "away") and
                              (self.attempt > 0 or visually_closed))
            if departed_empty and not self.ever_held:
                candidate = "grasp_not_established" if self.attempt > 0 else "prerequisite_unmet"
            if old_held is True and facts.knife_location in ("board", "elsewhere") and facts.contact == "separate" and not self.cut_confirmed:
                candidate = "lost_grasp"
            if self.memory.anchors.get("lost_grasp") and self.held is False and facts.knife_location in ("board", "elsewhere"):
                candidate = "lost_grasp"
            if candidate == "lost_grasp":
                self.memory.anchor("lost_grasp", evidence(facts, "此前有夹持证据，现在刀与夹爪分离"))
            if held_stable and "lost_grasp" in self.memory.anchors:
                del self.memory.anchors["lost_grasp"]

        if self.completed:
            self.stage = "completed"
        elif self.returned:
            self.stage = "verify_completion"
        elif self.cut_confirmed:
            self.stage = "return_tool"
        elif self.held is True:
            self.stage = "cutting" if facts.motion == "at_board" else "transport_tool"
        elif self.ever_held and self.held is None:
            # 已经持刀进入过工作区，随后移出可观测区域不等于重新开始抓取。
            self.stage = "unknown"
        elif self.attempt > 0:
            self.stage = "align_grasp" if (near_holder and visually_open) else "verify_grasp"
        elif near_holder:
            self.stage = "align_grasp"
        else:
            self.stage = "locate_tool"
        if not self.cut_confirmed:
            unknown.append("切割产生独立小段的效果尚未确认")
        if not self.returned:
            unknown.append("完成任务后的刀具归位与释放尚未确认")
        return self._result(known, unknown, current_evidence, candidate, held_stable and not conflict, True, conflict, new_attempt)

    def _result(self, known, unknown, current_evidence, candidate, recovered, observable, conflict, new_attempt):
        goals = []
        if self.ever_held:
            goals.append("曾建立刀具夹持")
        if self.cut_confirmed:
            goals.append("已观察到分离的小段香肠")
        if self.returned:
            goals.append("刀具已归位且释放")
        expectations = {
            "locate_tool": "定位刀柄并使夹爪接近；此时不要求刀离开刀架",
            "align_grasp": "建立刀柄在夹爪两指间的对齐条件，然后尝试闭合",
            "verify_grasp": "刀应离开刀架并与夹爪保持连接；闭合状态不足以证明成功",
            "transport_tool": "刀与夹爪保持连接，进入砧板工作区",
            "cutting": "刀作用后出现与主体分离的小段香肠",
            "return_tool": "刀回到刀架并与夹爪分离",
            "verify_completion": "补足切割效果、归位及释放证据",
            "completed": "完成证据已获得；后续观测仍可揭示变化",
            "unknown": "取得新的清晰图像，恢复当前刀与夹爪关系的判断",
        }
        return {"stage": self.stage, "stage_zh": STAGES[self.stage], "subtask": STAGES[self.stage],
                "confirmed_subgoals": goals, "preconditions": {"reliable_grasp": self.held,
                    "cut_effect_confirmed": self.cut_confirmed, "tool_returned": self.returned},
                "known_facts": known, "unknowns": unknown, "evidence": current_evidence,
                "expected_effect": expectations[self.stage], "candidate": candidate, "recovered": recovered,
                "observable": observable, "modality_conflict": conflict, "new_attempt_observed": new_attempt,
                "attempt": self.attempt, "task_completed": self.completed,
                "anchors": dict(self.memory.anchors)}
