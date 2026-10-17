"""显式任务规格，不做四个关键词匹配，也不声称任意语言任务泛化。"""
import re

INSTRUCTION = "Use the right arm to pick up the knife from the knife holder, cut a small piece of sausage and put it back on the knife holder."
SPEC = {
    "id": "right_knife_cut_sausage_return_v1", "arm": "right", "tool": "knife", "workpiece": "sausage",
    "ordered_subtasks": ["acquire_knife", "transport_to_board", "cut_new_piece", "return_knife", "release_verify"],
    "preconditions": {"cut_new_piece": ["reliable_grasp", "tool_at_board"], "release_verify": ["knife_at_holder"]},
    "completion": ["new_cut_effect_after_observed_operation", "knife_currently_at_holder", "released"],
    "binding": "explicit_instruction_allowlist_normalized_whitespace_case_punctuation",
}

def normalize(text):
    return re.sub(r"\s+", " ", text.strip().lower()).rstrip(".。")

def supports(instruction):
    return normalize(instruction) in {normalize(INSTRUCTION), normalize("用右臂从刀架上拿起刀，切下一小段香肠，然后将刀放回刀架。")}
