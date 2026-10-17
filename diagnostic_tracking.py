"""Bounded causal stage tracking and conservative, uncalibrated alert heuristics.

The model supplies validated observations; repeated assertions are not ground truth.
Only source timestamps govern cooldowns, so slow GPU inference cannot change them.
"""
import math


FAILURE_TYPES = frozenset({"grasp_miss", "object_dropped", "placement_miss",
                           "obstructed", "task_deviation", "other"})
KEY_STATE_VALUES = {
    "grasp": frozenset({"held", "not_held", "unknown"}),
    "target_relation": frozenset({"at_target", "away_from_target", "unknown"}),
    "task_goal": frozenset({"met", "not_met", "unknown"}),
}


class DiagnosticTracker:
    """Track consecutive support without retaining observations or model responses.

    ``stage_stale`` means the returned ``tracked_stage`` lacks a currently
    confirmed run of observations. Unknown/invalid observations break all pending
    confirmations, but do not erase an unresolved incident. At most one incident
    record per concrete failure type is retained, including its alert flag.
    Key states use the same confirmation length as stages. Recovery requires
    current key-state assertions relevant to every unresolved failure type; this
    is an explicit uncalibrated heuristic, not a physical recovery guarantee.
    """

    def __init__(self, stage_confirmation=2, failure_confirmation=2,
                 recovery_confirmation=2, cooldown_s=8.0):
        for name, value in (("stage_confirmation", stage_confirmation),
                            ("failure_confirmation", failure_confirmation),
                            ("recovery_confirmation", recovery_confirmation)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (type(cooldown_s) not in (int, float) or not math.isfinite(cooldown_s)
                or cooldown_s < 0):
            raise ValueError("cooldown_s must be a finite nonnegative number")
        self.stage_confirmation = stage_confirmation
        self.failure_confirmation = failure_confirmation
        self.recovery_confirmation = recovery_confirmation
        self.cooldown_s = float(cooldown_s)
        self._reset(None)

    def _reset(self, episode_index):
        self.episode_index = episode_index
        self.frame_index = None
        self.timestamp = None
        self.tracked_stage = "unknown"
        self.stage_candidate = None
        self.stage_candidate_count = 0
        self.tracked_key_states = {name: "unknown" for name in KEY_STATE_VALUES}
        self.key_state_candidates = {name: None for name in KEY_STATE_VALUES}
        self.key_state_candidate_counts = {name: 0 for name in KEY_STATE_VALUES}
        self.candidate_failure_type = None
        self.candidate_count = 0
        self.recovery_count = 0
        self.active_failure_type = None
        self._incidents = {}
        self._incident_sequence = 0
        self._last_alert_timestamp = None

    @staticmethod
    def _has_current_evidence(result, frame_index):
        evidence = result.get("evidence")
        indices = result.get("evidence_frame_indices")
        return (isinstance(evidence, list) and bool(evidence)
                and all(isinstance(item, str) and item.strip() for item in evidence)
                and isinstance(indices, list) and frame_index in indices
                and all(type(item) is int and 0 <= item <= frame_index for item in indices))

    @staticmethod
    def _recovery_key_state_matches(failure_type, key_states):
        # These model assertions are gates for a heuristic, not verified facts.
        if key_states["task_goal"] == "met":
            return True
        if failure_type in ("grasp_miss", "object_dropped"):
            return key_states["grasp"] == "held"
        if failure_type == "placement_miss":
            return key_states["target_relation"] == "at_target"
        # Generic obstruction/deviation/other have no universally valid subgoal.
        return False

    def update(self, episode_index, frame_index, timestamp, result):
        """Consume one validated result (or None) at a strictly later source time.

        Episode changes reset all state, including cooldown. A recovered event
        may alert again only after fresh confirmation and the source cooldown.
        ``incident_id`` is an integer local to its episode; pair with episode ID.
        """
        if type(episode_index) is not int or episode_index < 0:
            raise ValueError("episode_index must be a nonnegative integer")
        if type(frame_index) is not int or frame_index < 0:
            raise ValueError("frame_index must be a nonnegative integer")
        if type(timestamp) not in (int, float) or not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("timestamp must be a finite nonnegative number")
        if result is not None and not isinstance(result, dict):
            raise ValueError("result must be a validated dictionary or None")
        if self.episode_index == episode_index and self.frame_index is not None:
            if frame_index <= self.frame_index or timestamp <= self.timestamp:
                raise ValueError("frame_index and timestamp must both strictly increase within an episode")
        if self.episode_index != episode_index:
            self._reset(episode_index)
        self.frame_index = frame_index
        self.timestamp = float(timestamp)

        observed_stage = result["stage"] if result is not None else "unknown"
        if observed_stage == "unknown":
            self.stage_candidate = None
            self.stage_candidate_count = 0
        elif observed_stage == self.stage_candidate:
            self.stage_candidate_count = min(self.stage_confirmation, self.stage_candidate_count + 1)
        else:
            self.stage_candidate = observed_stage
            self.stage_candidate_count = 1
        stage_confirmed = self.stage_candidate_count >= self.stage_confirmation
        if stage_confirmed:
            self.tracked_stage = observed_stage

        observed_key_states = {name: result["key_states"][name] if result is not None else "unknown"
                               for name in KEY_STATE_VALUES}
        key_state_confirmed = {}
        for name, value in observed_key_states.items():
            if value == "unknown":
                self.key_state_candidates[name] = None
                self.key_state_candidate_counts[name] = 0
            elif value == self.key_state_candidates[name]:
                self.key_state_candidate_counts[name] = min(
                    self.stage_confirmation, self.key_state_candidate_counts[name] + 1)
            else:
                self.key_state_candidates[name] = value
                self.key_state_candidate_counts[name] = 1
            key_state_confirmed[name] = self.key_state_candidate_counts[name] >= self.stage_confirmation
            if key_state_confirmed[name]:
                self.tracked_key_states[name] = value

        alert = False
        recovery_confirmed = False
        recovered_incident_ids = []
        current_evidence = result is not None and self._has_current_evidence(result, frame_index)
        failure_supported = (result is not None and result["status"] == "failure"
                             and result["failure"] is True
                             and result["failure_type"] in FAILURE_TYPES
                             and result["action_outcome"] == "not_achieved" and current_evidence)
        recovery_action_supported = (result is not None and result["status"] in ("in_progress", "completed")
                                     and result["failure"] is False
                                     and result["action_outcome"] == "achieved" and current_evidence)
        recovery_key_state_matches = {
            failure_type: self._recovery_key_state_matches(failure_type, observed_key_states)
            for failure_type in self._incidents
        }
        recovery_supported = (recovery_action_supported and bool(recovery_key_state_matches)
                              and all(recovery_key_state_matches.values()))

        if failure_supported:
            self.recovery_count = 0
            if self.candidate_failure_type == result["failure_type"]:
                self.candidate_count = min(self.failure_confirmation, self.candidate_count + 1)
            else:
                self.candidate_failure_type = result["failure_type"]
                self.candidate_count = 1
            suppression_reason = "awaiting_failure_confirmation"
            if self.candidate_count >= self.failure_confirmation:
                failure_type = self.candidate_failure_type
                if failure_type not in self._incidents:
                    self._incident_sequence += 1
                    self._incidents[failure_type] = {"incident_id": self._incident_sequence,
                                                     "alert_emitted": False}
                self.active_failure_type = failure_type
                incident = self._incidents[failure_type]
                if incident["alert_emitted"]:
                    suppression_reason = "duplicate_unresolved_incident"
                elif self._cooldown_remaining(timestamp) > 0:
                    suppression_reason = "cooldown"
                else:
                    alert = True
                    incident["alert_emitted"] = True
                    self._last_alert_timestamp = float(timestamp)
                    suppression_reason = None
        else:
            self.candidate_failure_type = None
            self.candidate_count = 0
            if self._incidents and recovery_supported:
                self.recovery_count = min(self.recovery_confirmation, self.recovery_count + 1)
                suppression_reason = "awaiting_recovery_confirmation"
                if self.recovery_count >= self.recovery_confirmation:
                    recovery_confirmed = True
                    recovered_incident_ids = sorted(item["incident_id"] for item in self._incidents.values())
                    self._incidents.clear()
                    self.active_failure_type = None
                    suppression_reason = "recovery_confirmed"
            else:
                self.recovery_count = 0
                if result is None:
                    suppression_reason = "invalid_model_output"
                elif result["status"] == "failure":
                    suppression_reason = "insufficient_current_failure_evidence"
                elif self._incidents:
                    suppression_reason = "awaiting_recovery_evidence"
                elif result["status"] == "insufficient_evidence":
                    suppression_reason = "insufficient_evidence"
                else:
                    suppression_reason = "no_failure"

        if recovery_confirmed:
            failure_state = "recovered"
        elif self._incidents:
            failure_state = "recovery_pending" if self.recovery_count else "confirmed"
        elif (result is None or result["status"] == "insufficient_evidence"
              or (result["status"] == "failure" and not failure_supported)):
            failure_state = "unknown"
        else:
            failure_state = "suspected" if self.candidate_count else "clear"
        incident = self._incidents.get(self.active_failure_type)
        return {
            "tracked_stage": self.tracked_stage,
            "observed_stage": observed_stage,
            "stage_confirmed": stage_confirmed,
            "stage_stale": not stage_confirmed,
            "stage_candidate": self.stage_candidate,
            "stage_candidate_count": self.stage_candidate_count,
            "observed_key_states": observed_key_states,
            "tracked_key_states": self.tracked_key_states.copy(),
            "key_state_stale": {name: not confirmed for name, confirmed in key_state_confirmed.items()},
            "key_state_confirmed": key_state_confirmed,
            "key_state_candidates": self.key_state_candidates.copy(),
            "key_state_candidate_counts": self.key_state_candidate_counts.copy(),
            "failure_state": failure_state,
            "active_failure_type": self.active_failure_type,
            "candidate_failure_type": self.candidate_failure_type,
            "candidate_count": self.candidate_count,
            "alert": alert,
            "incident_id": incident["incident_id"] if incident else None,
            "suppression_reason": suppression_reason,
            "current_evidence_present": current_evidence,
            "failure_evidence_supported": failure_supported,
            "recovery_action_supported": recovery_action_supported,
            "recovery_key_state_matches": recovery_key_state_matches,
            "recovery_evidence_supported": recovery_supported,
            "recovery_policy": "uncalibrated_key_state_heuristic",
            "recovery_count": self.recovery_count,
            "recovery_confirmed": recovery_confirmed,
            "recovered_incident_ids": recovered_incident_ids,
            "unresolved_failure_types": sorted(self._incidents),
            "cooldown_remaining_s": self._cooldown_remaining(timestamp),
            "policy_calibrated": False,
        }

    def _cooldown_remaining(self, timestamp):
        if self._last_alert_timestamp is None:
            return 0.0
        return max(0.0, self.cooldown_s - (timestamp - self._last_alert_timestamp))
