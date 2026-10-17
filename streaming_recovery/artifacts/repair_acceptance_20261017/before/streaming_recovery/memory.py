"""所有跨决策的推理历史都集中于明确有界的容器。"""
from collections import deque
from dataclasses import asdict
import json


class BoundedMemory:
    def __init__(self, config):
        self.config = config
        self.frames = deque(maxlen=config["frames"])
        self.features = deque(maxlen=config["features"])
        self.anchors = {}

    def expire(self, now):
        for queue in (self.frames, self.features):
            while queue and now - queue[0].timestamp > self.config["window_s"]:
                queue.popleft()

    def add(self, observation, facts):
        self.expire(observation.timestamp)
        self.frames.append(observation)
        self.features.append(facts)

    def anchor(self, key, value):
        if key not in self.anchors and len(self.anchors) >= self.config["anchors"]:
            del self.anchors[next(iter(self.anchors))]
        self.anchors[key] = value

    def usage(self):
        return {"observations": len(self.frames), "images": sum(len(o.images) for o in self.frames),
                "image_bytes": sum(im.width * im.height * 3 for o in self.frames for im in o.images.values()),
                "features": len(self.features), "anchors": len(self.anchors),
                "joint_vector_values": sum(len(o.observation_state or [])+len(o.observation_velocity or []) for o in self.frames),
                "joint_feature_refs": sum(len(f.measurements.get("joint_motion",{}).get("evidence_refs",[])) for f in self.features),
                "feature_text_chars": sum(len(f.description) + len(f.error or "") for f in self.features),
                "feature_payload_chars":sum(len(json.dumps(asdict(f),ensure_ascii=False)) for f in self.features),
                "conversation_turns": 0, "persistent_model_kv_tokens": 0, "log_reads": 0}

