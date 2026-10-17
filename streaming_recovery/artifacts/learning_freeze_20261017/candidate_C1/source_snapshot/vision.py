"""真实本地 VLM：不读文件名、标签、总时长、历史日志或任务状态摘要。"""
from __future__ import annotations

import json
import time
import multiprocessing as mp
from queue import Empty
from .contracts import VisualFacts


def strict_json(raw):
    if len(raw)>6000:
        raise ValueError("model_response_too_long")
    if raw.startswith("```json") and raw.endswith("```"):
        raw=raw[7:-3].strip()
    def unique_keys(pairs):
        result={}
        for key,value in pairs:
            if key in result:
                raise ValueError("duplicate_json_key")
            result[key]=value
        return result
    def reject_constant(value):
        raise ValueError("nonfinite_json_constant")
    return json.loads(raw,object_pairs_hook=unique_keys,parse_constant=reject_constant)


PROMPT = '''Look at the CURRENT images. Describe only visible spatial relations of the white-handled KNIFE, the robot's right GRIPPER, the narrow black RACK on the table, and the sausage on the brown BOARD. Do not invent a task narrative. Earlier HIGH is a previous view of the same camera. CURRENT WRIST is a different camera at the CURRENT time.
Output JSON with these exact fields:
visibility: clear, partial, occluded, or unknown.
knife_location: holder (knife resting in the narrow black rack on the right side of the table), gripper (knife visibly removed from rack and carried by robot), board (knife lying on brown board), elsewhere, or unknown. The robot gripper is NOT the rack.
gripper: open, closed, or unknown.
motion: at_holder (gripper at black rack), at_board (gripper over brown board), away, or unknown. Describe CURRENT position only.
contact: between_fingers (knife handle visibly in robot fingers), separate, occluded, or unknown.
co_motion: unknown. Motion will be checked separately from consecutive detections.
cut_piece: yes if a small sausage piece is visibly separated, no if one intact sausage, otherwise unknown.
released: yes if robot fingers are visibly separate from knife, no if gripping it, otherwise unknown.
knife_handle_xy and gripper_xy: [x,y] in CURRENT HIGH image normalized 0..1000 (x right, y down), or null if uncertain. Knife point must be the handle center, gripper point the finger midpoint, not arm base.
description: one short sentence about visible facts, maximum 250 characters.
Return only JSON. Do not infer pickup, cutting or return from an instruction. When knife_location=holder, co_motion must be unknown.'''


def make_request(observation, previous, max_images=3):
    frames = []
    if previous is not None and "cam_high" in previous.images:
        frames.append({"image": previous.images["cam_high"], "camera": "cam_high", "timestamp": previous.timestamp,
                       "ref": previous.ref, "role": "EARLIER HIGH"})
    for camera, role in (("cam_high", "CURRENT HIGH"), ("cam_right_wrist", "CURRENT WRIST")):
        if camera in observation.images:
            frames.append({"image": observation.images[camera], "camera": camera, "timestamp": observation.timestamp,
                           "ref": observation.ref, "role": role})
    if len(frames) > max_images or any(f["timestamp"] > observation.timestamp for f in frames):
        raise ValueError("visual request capacity/causality violated")
    return {"timestamp": observation.timestamp, "ref": observation.ref, "instruction": observation.instruction, "frames": frames}


class LocalQwen:
    def __init__(self, config):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.torch, self.config = torch, config
        torch.manual_seed(config["seed"])
        torch.set_num_threads(4)
        if not torch.cuda.is_available() or not config["device"].startswith("cuda"):
            raise RuntimeError("configured real backend requires CUDA; no silent mock/CPU fallback")
        free, _ = torch.cuda.mem_get_info(config["device"])
        if free / 2**20 < config["min_free_mib"]:
            raise RuntimeError("insufficient free GPU memory; other workloads were not stopped")
        self.processor = AutoProcessor.from_pretrained(config["path"], local_files_only=True,
                                                       min_pixels=64*32*32, max_pixels=196608)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(config["path"], local_files_only=True,
            dtype=torch.bfloat16, device_map=config["device"], attn_implementation="sdpa").eval()

    def infer(self, request):
        prompt = PROMPT
        if len(prompt) > self.config["max_text_chars"] or len(request["frames"]) > self.config["max_images"]:
            raise ValueError("model input budget exceeded")
        content = []
        for f in request["frames"]:
            content.extend([{"type": "text", "text": f'{f["role"]} at t={f["timestamp"]:.3f}s'},
                            {"type": "image", "image": f["image"]}])
        content.append({"type": "text", "text": prompt})
        inputs = self.processor.apply_chat_template([{"role":"user", "content":content}], tokenize=True,
            add_generation_prompt=True, return_dict=True, return_tensors="pt").to(self.config["device"])
        tokens = inputs["input_ids"].shape[-1]
        if tokens > self.config["max_input_tokens"]:
            raise ValueError(f"model token budget exceeded: {tokens}")
        start = time.monotonic()
        with self.torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=self.config["max_new_tokens"],
                do_sample=False, use_cache=True, max_time=self.config["timeout_s"] - 5)
        raw = self.processor.decode(outputs[0, tokens:], skip_special_tokens=True).strip()
        elapsed = time.monotonic() - start
        generated = outputs.shape[-1] - tokens
        # KV cache 的生存期仅为本次调用；不创建对话 session 或累计特征缓存。
        del inputs, outputs
        return {"raw": raw, "input_tokens": tokens, "output_tokens": generated, "latency_s": elapsed,
                "gpu_allocated_mib": self.torch.cuda.memory_allocated() / 2**20}


def _worker(config, incoming, outgoing):
    try:
        backend = LocalQwen(config)
        outgoing.put({"ready": True})
    except Exception as e:
        outgoing.put({"ready": False, "error": f"{type(e).__name__}: {e}"[:500]})
        return
    while True:
        request = incoming.get()
        if request is None:
            break
        try:
            outgoing.put(backend.infer(request))
        except Exception as e:
            outgoing.put({"error": f"{type(e).__name__}: {e}"[:500]})


class QwenProcess:
    provenance = "real_qwen3_vl_2b"

    def __init__(self, config):
        self.config = config
        self.incoming = self.outgoing = self.process = None
        self.error = None
        self.calls = 0
        self.last_metrics = {}
        ctx = mp.get_context("spawn")
        self.incoming, self.outgoing = ctx.Queue(maxsize=1), ctx.Queue(maxsize=1)
        self.process = ctx.Process(target=_worker, args=(config, self.incoming, self.outgoing), daemon=True)
        self.process.start()
        try:
            result = self.outgoing.get(timeout=config["load_timeout_s"])
            if not result.get("ready"):
                self.error = result.get("error", "model_load_failed")
                self.close()
        except Empty:
            self.error = "model_load_timeout"
            self.close()

    def observe(self, observation, previous):
        request = make_request(observation, previous, self.config["max_images"])
        refs = [{k:f[k] for k in ("ref", "camera", "timestamp")} for f in request["frames"]]
        error, result = self.error, {}
        if "cam_high" not in observation.images:
            error = "current_high_image_missing"
        if not error:
            self.calls += 1
            self.incoming.put(request, timeout=2)
            try:
                result = self.outgoing.get(timeout=self.config["timeout_s"])
                error = result.get("error")
            except Empty:
                error = self.error = "model_timeout_worker_terminated"
                self.close()
        self.last_metrics = {k:v for k,v in result.items() if k != "raw"}
        self.last_metrics.update({"calls": self.calls, "raw": result.get("raw", "")[:6000],
                                  "input_images": len(refs), "provenance": self.provenance})
        if error:
            return VisualFacts(observation.timestamp, observation.ref, error=error, provenance=self.provenance,
                               evidence_refs=refs)
        try:
            raw = result["raw"]
            data = strict_json(raw)
            if not isinstance(data,dict):
                raise ValueError("model_response_not_object")
            if data.get("knife_location") == "rack":
                data["knife_location"] = "holder"
            return VisualFacts.parse(data, observation.timestamp, observation.ref, self.provenance, refs)
        except (ValueError, KeyError, TypeError) as e:
            return VisualFacts(observation.timestamp, observation.ref, error=str(e)[:500], provenance=self.provenance,
                               evidence_refs=refs)

    def close(self):
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=5)
            self.process = None
        for queue in (self.incoming, self.outgoing):
            if queue is not None:
                queue.cancel_join_thread()
                queue.close()
        self.incoming = self.outgoing = None
