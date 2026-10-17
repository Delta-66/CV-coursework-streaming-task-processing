from __future__ import annotations

from dataclasses import asdict, replace
import math
import json
from .memory import BoundedMemory
from .events import EventManager
from .tracking import TaskTracker
from .recovery import diagnose, propose, validate_consistency, RecoveryContext
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
        self.recovery = RecoveryContext()
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
        def fresh(name):
            m=observation.modalities.get(name,{})
            t=m.get("timestamp")
            return m.get("status")=="available" and t is not None and 0<=observation.timestamp-t<=.12
        observation=replace(observation,images={k:v for k,v in observation.images.items() if fresh(k)},
                            observation_state=observation.observation_state if fresh("observation.state") else None)
        if len(observation.images)>2 or any(max(im.size)>self.config["model"]["image_edge"] for im in observation.images.values()):
            raise ValueError("image memory capacity exceeded")
        self.memory.expire(observation.timestamp)
        previous = self.memory.frames[-1] if self.memory.frames else None
        facts = self.vision.observe(observation,previous)
        if not fresh("cam_high"):
            facts.effect_usable=False
        if observation.grasp_readiness is not None:
            facts.grasp_readiness=dict(observation.grasp_readiness)
        facts.validate_bounded_extensions()
        if facts.timestamp != observation.timestamp or facts.ref != observation.ref:
            raise ValueError("visual output clock mismatch")
        if len(facts.description)>400 or len(facts.error or "")>500 or len(facts.evidence_refs)>4:
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
        elif state["stage"] == "unknown" or (state["stage"] in ("verify_grasp","transport_tool") and state["preconditions"]["reliable_grasp"] is None):
            status = "information_insufficient"
        else:
            status = "executing"
        recovery_event = self.recovery.update(event,state)
        diagnosis = diagnose(recovery_event,state,facts)
        correction = propose(recovery_event,diagnosis,state,facts,self.config["rules"])
        self.recovery.observe_request(correction)
        usage = self.memory.usage() | self.events.usage()
        if hasattr(self.vision,"usage"):
            usage.update(self.vision.usage())
        usage.update(self.tracker.usage() | self.recovery.usage())
        usage.update({"object_states":3,"aligner_latest_samples":len(observation.modalities),
                      "task_instruction_chars":len(observation.instruction),
                      "model_last_response_chars":len(getattr(self.vision,"last_metrics",{}).get("raw","")),
                      "anchor_image_refs":sum(len(x.get("images",[])) for x in self.memory.anchors.values()),
                      "model_request_queue_capacity":1,"model_response_queue_capacity":1,
                      "model_input_images":getattr(self.vision,"last_metrics",{}).get("input_images",0),
                      "model_input_tokens":getattr(self.vision,"last_metrics",{}).get("input_tokens",0)})
        result = {"schema_version":"1.1","timestamp":observation.timestamp,"observation_ref":observation.ref,
                  "method":self.config["method"],"instruction":observation.instruction,
                  "scope":("右臂取刀、切下新小段并归位；固定场景双视角视觉，空间/接触准备须独立验证" if self.config["backend"]=="multiview_effect_geometry"
                           else "经典单视角对照：只支持取刀效果，切割完成/归位未获该后端支持"),
                  "state":state,"execution_status":status,"visual_facts":asdict(facts),
                  "failure_event":event,"recovery_event":recovery_event,"notify":notify,"notification_reason":notify_reason,
                  "decision_rules":{"direction_deadband_normalized":self.config["rules"]["direction_deadband_normalized"]},
                  "diagnosis":diagnosis,"correction":correction,"modalities":observation.modalities,
                  "modalities_used":["cam_high","observation.state[13]"] if self.config["backend"]=="calibrated_rgb_geometry" else list(observation.images)+(["observation.state[13]"] if observation.observation_state else []),
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
                "task_instruction_chars":1000,"model_last_response_chars":6000,"anchor_image_refs":4*m["anchors"],
                "model_input_images":self.config["model"]["max_images"],"model_input_tokens":self.config["model"]["max_input_tokens"]}
        limits.update(task_state_slots=18,task_reference_entries=1,recovery_contexts=1,recovery_event_evidence=m["event_evidence"],recovery_state_slots=3)
        limits.update(wrist_track_entries=3,food_reference_entries=5,visual_scalar_states=1)
        limits.update(feature_payload_chars=12000*m["features"])
        for key,limit in limits.items():
            if u.get(key,0)>limit:
                raise ValueError(f"memory_capacity_exceeded:{key}")
        if any(u[k] for k in ("conversation_turns","persistent_model_kv_tokens","log_reads")):
            raise ValueError("undeclared_history")

    def end_of_stream(self):
        return {"stream_ended":True,"last_decision_timestamp":self.last_timestamp if math.isfinite(self.last_timestamp) else None,
                "task_success":True if self.tracker.cut_confirmed and self.tracker.returned is True else None,
                "historical_completion":self.tracker.completed,
                "reason":"task_completion_evidence" if self.tracker.cut_confirmed and self.tracker.returned is True else "stream_end_is_not_success",
                "actually_executed":False}
