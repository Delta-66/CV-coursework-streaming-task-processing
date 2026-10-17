import importlib.util
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from temporal_diagnosis import (CausalWindow, QwenTemporalDiagnosis, build_messages,
                                output_schema, parse_result)


class Packet(SimpleNamespace):
    def __getattr__(self, name):
        raise AssertionError(f"Forbidden input field accessed: {name}")


def packet(frame_index=0, episode_index=0, timestamp=None):
    return Packet(episode_index=episode_index, frame_index=frame_index,
                  timestamp=frame_index / 25 if timestamp is None else timestamp,
                  timestamp_source="fixture", task_instruction="Place the block in the tray.",
                  images={"cam": np.full((80, 160, 3), frame_index % 256, np.uint8)},
                  observations={"observation.state": np.arange(14, dtype=np.float32),
                                "observation.velocity": np.zeros(14),
                                "observation.effort": np.ones(14),
                                "labels.secret": [999], "action": [999]},
                  source_references={"cam": {"frame_index": frame_index}})


def diagnosis(status="insufficient_evidence", indices=None):
    result = {"stage": "completed" if status == "completed" else "grasping",
              "status": status,
              "failure": {"failure": True, "in_progress": False, "completed": False,
                          "insufficient_evidence": None}[status],
              "evidence": ["The current gripper is visible."] if indices else [],
              "high_level_advice": None, "low_level_advice": None,
              "key_states": {"grasp": "unknown", "target_relation": "unknown",
                             "task_goal": "met" if status == "completed" else "unknown"},
              "expected_effect": "Hold the block.", "observed_effect": "The gripper is near the block.",
              "action_outcome": "not_achieved" if status == "failure" else
                                "achieved" if status == "completed" else "uncertain",
              "failure_type": "grasp_miss" if status == "failure" else
                              "unknown" if status == "insufficient_evidence" else "none",
              "evidence_frame_indices": indices or []}
    if status == "failure":
        result["high_level_advice"] = "Realign the gripper with the block."
        result["low_level_advice"] = {"action": "Move toward the block in the image.",
                                      "reference_frame": "image", "completion_check": "Check alignment."}
    return result


class TemporalTests(unittest.TestCase):
    def test_bounded_chronological_window_and_episode_reset(self):
        window = CausalWindow(2, 8, 64)
        for frame_index in (0, 100, 200, 300):
            request = window.append(packet(frame_index), "cam")
        self.assertEqual([item.frame_index for item in request.frames], [100, 200, 300])
        request = window.append(packet(600), "cam")
        self.assertEqual([item.frame_index for item in request.frames], [600])
        request = window.append(packet(0, episode_index=18), "cam")
        self.assertEqual([item.frame_index for item in request.frames], [0])

    def test_window_only_copies_measured_allowlist_and_gripper_raw_values(self):
        source = packet()
        window = CausalWindow(2, 8, 64)
        request = window.append(source, "cam")
        frame = request.frames[0]
        self.assertEqual(set(frame.observations), {"observation.state", "observation.velocity", "observation.effort"})
        self.assertEqual(frame.gripper_state["left"], 6.0)
        self.assertEqual(frame.gripper_state["right"], 13.0)
        self.assertEqual(frame.image.shape, (32, 64, 3))
        self.assertFalse(frame.image.flags.writeable)
        source.observations["observation.state"][:] = -1
        frame.observations["observation.state"][0] = 999
        source.images["cam"][:] = 255
        next_request = window.append(packet(100), "cam")
        self.assertEqual(next_request.frames[0].observations["observation.state"][0], 0.0)
        self.assertTrue((next_request.frames[0].image == 0).all())

    def test_zero_history_and_missing_optional_states(self):
        window = CausalWindow(0)
        source = packet()
        source.observations = {}
        request = window.append(source, "cam")
        self.assertEqual(request.frames[0].observations, {})
        self.assertIsNone(request.frames[0].gripper_state)
        self.assertEqual(len(window.append(packet(100), "cam").frames), 1)

    def test_nonmonotonic_or_changing_identity_is_rejected_before_mutation(self):
        window = CausalWindow()
        window.append(packet(100), "cam")
        invalid = [packet(100), packet(101, timestamp=3), packet(101, timestamp=float("nan"))]
        changed = packet(101)
        changed.task_instruction = "Another task"
        invalid.append(changed)
        for source in invalid:
            with self.subTest(source=source.frame_index), self.assertRaises(ValueError):
                window.append(source, "cam")
        self.assertEqual([item.frame_index for item in window.frames], [100])

    def test_invalid_measured_vectors_and_budgets_are_rejected(self):
        for values in ([0] * 13, [float("nan")] * 14, ["x"] * 14):
            source = packet()
            source.observations["observation.state"] = values
            with self.assertRaises(ValueError):
                CausalWindow().append(source, "cam")
        for kwargs in ({"history_size": -1}, {"history_size": True}, {"history_size": 33},
                       {"max_history_s": float("inf")}, {"max_history_s": 0}, {"max_image_edge": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CausalWindow(**kwargs)

    def request(self):
        window = CausalWindow()
        window.append(packet(0), "cam")
        return window.append(packet(100), "cam")

    def test_all_statuses_parse_with_explicit_causal_evidence(self):
        request = self.request()
        for status in ("in_progress", "failure", "completed", "insufficient_evidence"):
            result = diagnosis(status, [0, 100])
            self.assertEqual(parse_result(json.dumps(result), request), result)
        text = "```json\n" + json.dumps(diagnosis()) + "\n```"
        self.assertEqual(parse_result(text, request), diagnosis())

    def test_future_missing_duplicate_or_only_past_evidence_is_rejected(self):
        request = self.request()
        for indices in ([101], [50, 100], [100, 100], [0], [True, 100], "100"):
            result = diagnosis("failure", indices)
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                parse_result(json.dumps(result), request)
        result = diagnosis("failure", [100])
        with self.assertRaisesRegex(ValueError, "past AND current"):
            parse_result(json.dumps(result), request)

    def test_contradictory_outcomes_and_false_completion_are_rejected(self):
        request = self.request()
        invalid = []
        for status, name, value in (("failure", "failure_type", "none"),
                                    ("failure", "action_outcome", "uncertain"),
                                    ("completed", "action_outcome", "uncertain"),
                                    ("insufficient_evidence", "action_outcome", "achieved"),
                                    ("in_progress", "action_outcome", "not_achieved"),
                                    ("in_progress", "failure_type", "object_dropped"),
                                    ("in_progress", "expected_effect", " ")):
            result = diagnosis(status, [0, 100])
            result[name] = value
            invalid.append(result)
        result = diagnosis("completed", [100])
        result["key_states"]["task_goal"] = "not_met"
        invalid.append(result)
        result = diagnosis("in_progress", [100])
        result["evidence"] = []
        invalid.append(result)
        for result in invalid:
            with self.subTest(result=result), self.assertRaises(ValueError):
                parse_result(json.dumps(result), request)

    def test_duplicate_keys_and_non_json_constants_are_rejected(self):
        request = self.request()
        text = json.dumps(diagnosis()).replace('"failure": null', '"failure": null, "failure": null')
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            parse_result(text, request)
        with self.assertRaises(ValueError):
            parse_result(json.dumps(diagnosis()).replace('"failure": null', '"failure": NaN'), request)

    def test_messages_use_exact_chronological_images_and_no_previous_answers(self):
        request = self.request()
        messages = build_messages(request)
        self.assertEqual([item["role"] for item in messages], ["system", "user"])
        content = messages[1]["content"]
        images = [item["image"] for item in content if item["type"] == "image"]
        self.assertEqual(len(images), 2)
        self.assertTrue(np.array_equal(np.array(images[0]), request.frames[0].image))
        self.assertTrue(np.array_equal(np.array(images[1]), request.frames[1].image))
        states = [json.loads(item["text"]) for item in content[1:] if item["type"] == "text"]
        self.assertEqual([item["frame_index"] for item in states], [0, 100])
        self.assertEqual([item["current"] for item in states], [False, True])
        self.assertNotIn("labels.secret", json.dumps(states))
        self.assertNotIn("action", states[0]["observations"])
        backend = QwenTemporalDiagnosis.__new__(QwenTemporalDiagnosis)
        with patch.object(backend, "predict_messages", return_value=("response", {})) as generate:
            self.assertEqual(backend.predict(request), ("response", {}))
        self.assertEqual(len(generate.call_args.args[0]), 2)

    @unittest.skipUnless(importlib.util.find_spec("lmformatenforcer"), "Optional model dependency")
    def test_constrained_schema_accepts_every_status(self):
        from lmformatenforcer import JsonSchemaParser
        for status in ("in_progress", "failure", "completed", "insufficient_evidence"):
            parser = JsonSchemaParser(output_schema())
            for character in json.dumps(diagnosis(status, [0, 100])):
                self.assertIn(character, parser.get_allowed_characters())
                parser = parser.add_character(character)
            self.assertTrue(parser.can_end())

    @unittest.skipUnless(importlib.util.find_spec("lmformatenforcer"), "Optional model dependency")
    def test_decoder_prevents_subgoal_success_being_a_met_full_task_goal(self):
        from lmformatenforcer import JsonSchemaParser
        for status in ("in_progress", "failure", "insufficient_evidence", "completed"):
            result = diagnosis(status, [0, 100])
            result["key_states"]["task_goal"] = "not_met" if status == "completed" else "met"
            parser = JsonSchemaParser(output_schema())
            rejected = False
            for character in json.dumps(result):
                if character not in parser.get_allowed_characters():
                    rejected = True
                    break
                parser = parser.add_character(character)
            self.assertTrue(rejected, f"Decoder allowed contradictory full goal for {status}")


if __name__ == "__main__":
    unittest.main()
