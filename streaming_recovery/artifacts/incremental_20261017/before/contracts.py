"""推理边界：只允许已到达的观测，路径/episode/标签不在此契约内。"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any


@dataclass
class Sample:
    modality: str
    timestamp: float
    arrived_at: float
    value: Any
    valid: bool = True
    reason: str | None = None


@dataclass
class Observation:
    timestamp: float
    instruction: str
    images: dict[str, Any]
    observation_state: list[float] | None
    modalities: dict[str, dict]
    ref: str


ENUMS = {
    "visibility": {"clear", "partial", "occluded", "unknown"},
    "knife_location": {"holder", "gripper", "board", "elsewhere", "unknown"},
    "gripper": {"open", "closed", "unknown"},
    "motion": {"approaching_holder", "at_holder", "to_board", "at_board", "returning_holder", "away", "unknown"},
    "contact": {"between_fingers", "co_moving", "separate", "occluded", "unknown"},
    "co_motion": {"yes", "no", "unknown"},
    "cut_piece": {"yes", "no", "unknown"},
    "released": {"yes", "no", "unknown"},
}


@dataclass
class VisualFacts:
    timestamp: float
    ref: str
    visibility: str = "unknown"
    knife_location: str = "unknown"
    gripper: str = "unknown"
    motion: str = "unknown"
    contact: str = "unknown"
    co_motion: str = "unknown"
    cut_piece: str = "unknown"
    released: str = "unknown"
    knife_handle_xy: list[float] | None = None
    gripper_xy: list[float] | None = None
    description: str = ""
    valid: bool = False
    error: str | None = None
    provenance: str = "unknown"
    # 当前/历史图像引用，最多 3 个；绝不携带整个视频或路径。
    evidence_refs: list[dict] = field(default_factory=list)
    measurements: dict = field(default_factory=dict)
    effect_usable: bool = True

    @classmethod
    def parse(cls, data: dict, timestamp: float, ref: str, provenance: str, evidence_refs: list[dict]):
        expected = set(ENUMS) | {"knife_handle_xy", "gripper_xy", "description"}
        if not isinstance(data, dict) or set(data) != expected:
            raise ValueError("visual_schema: fields must exactly match the contract")
        for key, values in ENUMS.items():
            if not isinstance(data[key], str) or data[key] not in values:
                raise ValueError(f"visual_schema: invalid {key}")
        for key in ("knife_handle_xy", "gripper_xy"):
            point = data[key]
            if point is not None and (not isinstance(point, list) or len(point) != 2 or
                any(isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) or not 0 <= v <= 1000 for v in point)):
                raise ValueError(f"visual_schema: invalid image point {key}")
        if not isinstance(data["description"], str) or len(data["description"]) > 400:
            raise ValueError("visual_schema: description must be <= 400 characters")
        if len(evidence_refs) > 3 or any(r["timestamp"] > timestamp + 1e-6 for r in evidence_refs):
            raise ValueError("future or excessive visual evidence")
        # 矛盾不能静默变成正常事实。
        if data["knife_location"] == "holder" and data["co_motion"] == "yes":
            raise ValueError("visual_conflict: knife remains in holder but co_motion=yes")
        if data["knife_location"] == "gripper" and data["released"] == "yes":
            raise ValueError("visual_conflict: held and released simultaneously")
        return cls(timestamp, ref, **data, valid=True, provenance=provenance, evidence_refs=evidence_refs)


def evidence(facts: VisualFacts, claim: str) -> dict:
    return {"timestamp": facts.timestamp, "ref": facts.ref, "claim": claim,
            "source": facts.provenance, "images": facts.evidence_refs}

