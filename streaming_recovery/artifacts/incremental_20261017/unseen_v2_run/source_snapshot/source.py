"""严格允许字段的逐帧源；对齐器只能接纳已到达且非未来的样本。"""
from __future__ import annotations

import json
import math
from pathlib import Path
from .contracts import Observation, Sample


class CausalAligner:
    def __init__(self, modalities: list[str], max_age_s: float):
        self.modalities = tuple(modalities)
        self.max_age_s = max_age_s
        self.latest: dict[str, Sample] = {}
        self.clock = -math.inf

    def push(self, sample: Sample, now: float):
        if sample.modality not in self.modalities:
            raise ValueError("unregistered modality")
        if not all(math.isfinite(v) for v in (sample.timestamp, sample.arrived_at, now)):
            raise ValueError("non-finite timestamp")
        if sample.timestamp > sample.arrived_at + 1e-6 or sample.arrived_at > now + 1e-6 or now < self.clock:
            raise ValueError("future sample or clock reversal")
        self.clock = now
        old = self.latest.get(sample.modality)
        if old is None or sample.timestamp >= old.timestamp:
            self.latest[sample.modality] = sample

    def snapshot(self, now: float):
        if now < self.clock:
            raise ValueError("clock reversal")
        self.clock = now
        values, status = {}, {}
        for key in self.modalities:
            s = self.latest.get(key)
            state = "missing" if s is None else ("invalid" if not s.valid else ("stale" if now - s.timestamp > self.max_age_s else "available"))
            status[key] = {"status": state, "timestamp": s.timestamp if s else None,
                           "age_s": round(now - s.timestamp, 6) if s else None,
                           "reason": s.reason if s else "not_arrived"}
            if state == "available":
                values[key] = s.value
        return values, status

    def reset(self):
        self.latest.clear()
        self.clock = -math.inf


class LeRobotStream:
    # 不读取 task_index/episode_index/index/flags/actions；任务指令由调用者提供。
    COLUMNS = ["timestamp", "frame_index", "observation.state"]

    def __init__(self, root: Path, episode: int, cameras: list[str], sample_every: int = 25,
                 max_age_s: float = .12, image_edge: int = 512, instruction: str = ""):
        if sample_every < 1 or image_edge < 64 or len(cameras) > 2:
            raise ValueError("invalid source capacity")
        self.root, self.episode, self.cameras = Path(root), episode, cameras
        self.sample_every, self.image_edge, self.instruction = sample_every, image_edge, instruction
        info = json.loads((self.root / "meta/info.json").read_text(encoding="utf-8"))
        if info["codebase_version"] != "v2.1":
            raise ValueError("only verified LeRobot v2.1 is supported")
        self.fps = float(info["fps"])
        self.chunk = episode // int(info["chunks_size"])
        self.aligner = CausalAligner(["observation.state"] + cameras, max_age_s)
        self.audit = {"rows_read": 0, "frames_decoded": {c: 0 for c in cameras},
                      "columns_read": self.COLUMNS, "future_frames_exposed": 0,
                      "invalid_video_timestamps": 0, "source_mode": "frame_order_backpressure"}

    def __iter__(self):
        import av
        import pyarrow.parquet as pq
        self.aligner.reset()
        path = self.root / f"data/chunk-{self.chunk:03d}/episode_{self.episode:06d}.parquet"
        table = pq.ParquetFile(path)
        videos, decoders, errors = {}, {}, {}
        for camera in self.cameras:
            p = self.root / f"videos/chunk-{self.chunk:03d}/observation.images.{camera}/episode_{self.episode:06d}.mp4"
            if p.exists():
                try:
                    videos[camera] = av.open(str(p))
                    decoders[camera] = videos[camera].decode(video=0)
                except Exception as e:
                    errors[camera] = "video_open_error:" + type(e).__name__
            else:
                errors[camera] = "video_file_missing"
        try:
            # Arrow 内部压缩页/AV1 解码器是运输缓冲，不暴露给推理；每次最多返回一行/一帧。
            for expected, batch in enumerate(table.iter_batches(batch_size=1, columns=self.COLUMNS, use_threads=False)):
                row = batch.to_pylist()[0]
                if row["frame_index"] != expected:
                    raise ValueError("non-contiguous frame index; cannot assume synchronized cameras")
                now = expected / self.fps
                if not math.isfinite(row["timestamp"]) or abs(row["timestamp"] - now) > 1e-5:
                    raise ValueError("table timestamp is not aligned to the declared frame clock")
                self.audit["rows_read"] += 1
                state = row["observation.state"]
                valid = isinstance(state, list) and len(state) == 14 and all(isinstance(v, (float, int)) and math.isfinite(v) for v in state)
                self.aligner.push(Sample("observation.state", now, now, state if valid else None,
                                         valid, None if valid else "invalid_14d_state"), now)
                for camera in self.cameras:
                    if camera not in decoders:
                        self.aligner.push(Sample(camera, now, now, None, False, errors.get(camera, "video_exhausted")), now)
                        continue
                    try:
                        frame = next(decoders[camera])
                        self.audit["frames_decoded"][camera] += 1
                    except StopIteration:
                        del decoders[camera]
                        errors[camera] = "video_exhausted"
                        self.aligner.push(Sample(camera, now, now, None, False, "video_exhausted"), now)
                        continue
                    except Exception as e:
                        # 损坏的可选视频成为明确无效模态，其余传感器流继续推进。
                        errors[camera] = "video_decode_error:" + type(e).__name__
                        del decoders[camera]
                        self.aligner.push(Sample(camera, now, now, None, False, errors[camera]), now)
                        continue
                    frame_time = float(frame.time) if frame.time is not None else math.nan
                    aligned = math.isfinite(frame_time) and abs(frame_time - now) <= 1e-6
                    if not aligned:
                        self.audit["invalid_video_timestamps"] += 1
                        self.aligner.push(Sample(camera, now, now, None, False, "video_timestamp_mismatch"), now)
                    else:
                        # 非决策帧也按序解码，不进行未来关键帧选择。
                        im = frame.to_image()
                        im.thumbnail((self.image_edge, self.image_edge))
                        self.aligner.push(Sample(camera, frame_time, now, im), now)
                if expected % self.sample_every == 0:
                    values, status = self.aligner.snapshot(now)
                    yield Observation(now, self.instruction,
                        {c: values[c] for c in self.cameras if c in values}, values.get("observation.state"),
                        status, f"obs-{expected:06d}")
        finally:
            for video in videos.values():
                video.close()


def instruction_for_episode(root: Path, episode: int) -> str:
    """适配器控制面：只从 task_index 取文本，不传标签/编号/路径到模型。"""
    import pyarrow.parquet as pq
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    path = root / f'data/chunk-{episode // info["chunks_size"]:03d}/episode_{episode:06d}.parquet'
    batch = next(pq.ParquetFile(path).iter_batches(batch_size=1, columns=["task_index"]))
    task_index = batch.column(0)[0].as_py()
    with (root / "meta/tasks.jsonl").open(encoding="utf-8") as f:
        for line in f:
            task = json.loads(line)
            if task["task_index"] == task_index:
                instruction = task["task"]
                if not isinstance(instruction, str) or len(instruction) > 1000:
                    raise ValueError("invalid instruction")
                return instruction
    raise ValueError("task instruction missing")
