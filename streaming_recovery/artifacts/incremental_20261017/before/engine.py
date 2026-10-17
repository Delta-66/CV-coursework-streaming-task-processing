from __future__ import annotations

from dataclasses import asdict
import math
from .memory import BoundedMemory
from .events import EventManager
from .tracking import TaskTracker
from .recovery import diagnose, propose, validate_consistency
from .contracts import VisualFacts, evidence


class Monitor:
    def __init__(self, config, vision):
        # 源路径、episode 清单、输出目录不进入推理实例。
        self.config={k:config[k] for k in ("method","backend","memory","model","rules","geometry")}
        self.vision = vision
        self.reset()

    def reset(self):
        self.memory = BoundedMemory(self.config["memory"])
        self.tracker = TaskTracker(self.config["rules"],self.memory)
        self.events = EventManager(self.config["memory"],self.config["rules"])
        self.last_timestamp = -math.inf
        self.instruction = None
        if hasattr(self.vision,"reset"):
            self.vision.reset()

    def step(self, observation):
        if not math.isfinite(observation.timestamp) or observation.timestamp <= self.last_timestamp:
            raise ValueError("decision timestamps must strictly increase")
        if len(observation.instruction) > 1000 or not TaskTracker.supports(observation.instruction):
            raise ValueError("unsupported task: this profile requires right-arm knife/sausage/holder instruction")
        if self.instruction is not None and observation.instruction != self.instruction:
            raise ValueError("instruction_changed_without_reset")
        self.instruction = observation.instruction
        for modality in observation.modalities.values():
            t = modality.get("timestamp")
            if t is not None and t > observation.timestamp+1e-6:
                raise ValueError("future modality exposed to inference")
        if len(observation.images)>2 or any(max(im.size)>self.config["model"]["image_edge"] for im in observation.images.values()):
            raise ValueError("image memory capacity exceeded")
        self.memory.expire(observation.timestamp)
        previous = self.memory.frames[-1] if self.memory.frames else None
        facts = self.vision.observe(observation,previous)
        if facts.timestamp != observation.timestamp or facts.ref != observation.ref:
            raise ValueError("visual output clock mismatch")
        if len(facts.description)>400 or len(facts.error or "")>500 or len(facts.evidence_refs)>3:
            raise ValueError("visual output capacity exceeded")
        if any(e["timestamp"]>observation.timestamp for e in facts.evidence_refs):
            raise ValueError("future evidence")
        self.memory.add(observation,facts)
        state = self.tracker.update(observation,facts)
        item = evidence(facts,"当前支持 " + state["candidate"] if state["candidate"] else "当前夹持恢复证据")
        event,notify,notify_reason = self.events.update(state["candidate"],state["attempt"],observation.timestamp,item,
                                                       state["recovered"],state["observable"] and not state["modality_conflict"])
        if not state["observable"] or state["modality_conflict"]:
            status = "information_insufficient"
        elif event and event["state"] in ("resolved","retracted"):
            status = "recovered"
        elif event and event["state"] in ("confirmed","persistent"):
            status = "confirmed_failure" if event["observation_status"]=="supported" else "information_insufficient"
        elif event and event["state"]=="suspected":
            status = "suspicious" if event["observation_status"]=="supported" else "information_insufficient"
        elif state["task_completed"]:
            status = "completed"
        elif state["stage"] == "unknown" or (state["stage"] == "verify_grasp" and state["preconditions"]["reliable_grasp"] is None):
            status = "information_insufficient"
        else:
            status = "executing"
        diagnosis = diagnose(event,state,facts)
        correction = propose(event,diagnosis,state,facts,self.config["rules"])
        usage = self.memory.usage() | self.events.usage()
        if hasattr(self.vision,"usage"):
            usage.update(self.vision.usage())
        usage.update({"task_state_slots":12,"object_states":3,"aligner_latest_samples":len(observation.modalities),
                      "task_instruction_chars":len(observation.instruction),
                      "model_last_response_chars":len(getattr(self.vision,"last_metrics",{}).get("raw","")),
                      "anchor_image_refs":sum(len(x.get("images",[])) for x in self.memory.anchors.values()),
                      "model_request_queue_capacity":1,"model_response_queue_capacity":1,
                      "model_input_images":getattr(self.vision,"last_metrics",{}).get("input_images",0),
                      "model_input_tokens":getattr(self.vision,"last_metrics",{}).get("input_tokens",0)})
        result = {"schema_version":"1.0","timestamp":observation.timestamp,"observation_ref":observation.ref,
                  "method":self.config["method"],"instruction":observation.instruction,
                  "scope":"刀具获取、转移与后续操作前置条件；真实视觉基线不确认切割完成/刀具归位",
                  "state":state,"execution_status":status,"visual_facts":asdict(facts),
                  "failure_event":event,"notify":notify,"notification_reason":notify_reason,
                  "diagnosis":diagnosis,"correction":correction,"modalities":observation.modalities,
                  "modalities_used":["cam_high","observation.state[13]"] if self.config["backend"]=="calibrated_rgb_geometry" else list(observation.images),
                  "modalities_for_audit_only":["cam_right_wrist"] if self.config["backend"]=="calibrated_rgb_geometry" else [],
                  "memory_usage":usage,"memory_limits":self.config["memory"],
                  "execution":{"mode":"advisory_replay","actually_executed":False,"execution_receipt":None,
                               "outcome_attributed_to_suggestion":False}}
        validate_consistency(result)
        self.check_capacity(result)
        self.last_timestamp = observation.timestamp
        return result

    def check_capacity(self,result):
        u,m=result["memory_usage"],self.config["memory"]
        limits={"observations":m["frames"],"images":2*m["frames"],"features":m["features"],"anchors":m["anchors"],
                "events":m["events"],"event_evidence":m["events"]*m["event_evidence"],"object_states":m["objects"],
                "feature_text_chars":m["features"]*900,"visual_reference_images":self.config["geometry"]["max_reference_images"],
                "visual_masks":self.config["geometry"]["max_masks"],"visual_tracks":self.config["geometry"]["max_tracks"],
                "task_instruction_chars":1000,"model_last_response_chars":6000,"anchor_image_refs":3*m["anchors"],
                "model_input_images":self.config["model"]["max_images"],"model_input_tokens":self.config["model"]["max_input_tokens"]}
        for key,limit in limits.items():
            if u.get(key,0)>limit:
                raise ValueError(f"memory_capacity_exceeded:{key}")
        if any(u[k] for k in ("conversation_turns","persistent_model_kv_tokens","log_reads")):
            raise ValueError("undeclared_history")

    def end_of_stream(self):
        return {"stream_ended":True,"last_decision_timestamp":self.last_timestamp if math.isfinite(self.last_timestamp) else None,
                "task_success":True if self.tracker.completed else None,
                "reason":"task_completion_evidence" if self.tracker.completed else "stream_end_is_not_success",
                "actually_executed":False}
