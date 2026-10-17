from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, asdict


@dataclass
class FailureEvent:
    event_id: str
    kind: str
    attempt: int
    opened_at: float
    updated_at: float
    state: str = "suspected"
    support_count: int = 0
    evidence: list[dict] = field(default_factory=list)
    last_support_at: float | None = None
    support_started_at: float | None = None
    observation_status: str = "supported"
    closure_reason: str | None = None


class EventManager:
    def __init__(self, memory_config, rules):
        self.events = deque(maxlen=memory_config["events"])
        self.evidence_limit = memory_config["event_evidence"]
        self.rules, self.counter, self.active = rules, 0, None

    def update(self, kind, attempt, now, item=None, recovered=False, observable=True):
        notification, reason = False, None
        event = self.active
        closed_event = None
        if event is not None and attempt > event.attempt:
            event.state, event.closure_reason = "closed_for_retry", "observed_new_attempt_not_a_suggestion"
            event.updated_at = now
            closed_event = event
            self.active = event = None
            notification, reason = True, "new_attempt_observed"
        if event is not None and recovered:
            event.state = "resolved" if event.state in ("confirmed", "persistent") else "retracted"
            event.closure_reason = "positive_recovery_evidence"
            event.updated_at = now
            if item:
                event.evidence = (event.evidence + [item])[-self.evidence_limit:]
            self.active = None
            return asdict(event), True, "recovery_observed_without_execution_attribution"
        if kind:
            if event is None or event.kind != kind:
                if event is not None:
                    event.state, event.closure_reason = "superseded", "different_observed_problem"
                self.counter += 1
                event = FailureEvent(f"event-{self.counter:04d}", kind, attempt, now, now)
                self.events.append(event)
                self.active = event
                notification, reason = True, "new_suspicion"
            if event.last_support_at is None or now > event.last_support_at:
                gap = None if event.last_support_at is None else now - event.last_support_at
                if gap is None or gap > self.rules["max_evidence_gap_s"]:
                    event.support_count, event.support_started_at = 0, now
                event.support_count = min(event.support_count + 1, self.rules["confirm_samples"])
                event.last_support_at = now
                event.updated_at = now
                if item:
                    event.evidence = (event.evidence + [item])[-self.evidence_limit:]
            event.observation_status = "supported"
            sufficient = (event.support_count >= self.rules["confirm_samples"] and
                          now - event.support_started_at >= self.rules["confirm_span_s"])
            if sufficient and event.state == "suspected":
                event.state = "confirmed"
                notification, reason = True, "evidence_confirmed"
            elif event.state == "confirmed":
                event.state = "persistent"
        elif event is not None:
            event.observation_status = "not_currently_supported" if observable else "unobservable"
            # 不因遮挡或时间流逝宣称故障解除；疑似证据连续性也不能跨缺失累加。
            if event.state == "suspected":
                event.support_count, event.support_started_at = 0, None
                event.last_support_at = None
        return asdict(event) if event else (asdict(closed_event) if closed_event else None), notification, reason

    def usage(self):
        return {"events": len(self.events), "event_evidence": sum(len(e.evidence) for e in self.events),
                "active_events": int(self.active is not None)}
