"""功能验收，不计算识别精度或泛化指标；合成与真实像素测试明确分开。"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone, timedelta
import io
import json
from pathlib import Path
import time
import unittest
from PIL import Image
from .__main__ import BASE, make_vision, sha256
from .engine import Monitor
from .geometry import GeometryVision
from .source import LeRobotStream, instruction_for_episode


def real_decisions(config, root, episode):
    vision=make_vision(config)
    monitor=Monitor(config,vision)
    stream=LeRobotStream(Path(root),episode,config["source"]["cameras"],config["source"]["sample_every"],
                        config["source"]["max_age_s"],config["model"]["image_edge"],instruction_for_episode(Path(root),episode))
    results=[]
    for obs in stream:
        expected=round(obs.timestamp*stream.fps)+1
        assert stream.audit["rows_read"]==expected,"reader consumed future rows before yielding"
        assert all(n<=expected for n in stream.audit["frames_decoded"].values()),"decoder exposed future transport steps"
        results.append(monitor.step(obs))
    return results,stream.audit,monitor


def build_prefix_fixture(config, directory, suffix_black=False):
    """离线测试数据制作者：真实像素公共前缀＋显式修改的测试后缀，不是新增真实轨迹。"""
    import av
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from fractions import Fraction
    root=Path(config["source"]["root"])
    directory=Path(directory)
    (directory/"meta").mkdir(parents=True)
    (directory/"data/chunk-000").mkdir(parents=True)
    info={"codebase_version":"v2.1","fps":25,"chunks_size":1000}
    (directory/"meta/info.json").write_text(json.dumps(info),encoding="utf-8")
    instruction=instruction_for_episode(root,0)
    (directory/"meta/tasks.jsonl").write_text(json.dumps({"task_index":0,"task":instruction})+"\n",encoding="utf-8")
    # 故意改变路径编号和全部事后标签，公共前缀诊断不能随之变化。
    episode=901 if suffix_black else 101
    rows=pq.read_table(root/"data/chunk-000/episode_000000.parquet").to_pylist()
    prefix_frames=275
    generated=[]
    for i,row in enumerate(rows):
        state=row["observation.state"]
        if suffix_black and i>=prefix_frames:
            state=[0.0]*14
        generated.append({"timestamp":i/25,"frame_index":i,"task_index":0,"observation.state":state,
            "episode_index":episode,"complementary_info.vifailback_flags":[suffix_black]*3,
            "action":[999.0 if suffix_black else -999.0]*14,
            "complementary_info.action_eef":[float("nan")]*16})
    pq.write_table(pa.Table.from_pylist(generated),directory/f"data/chunk-000/episode_{episode:06d}.parquet")
    for camera in config["source"]["cameras"]:
        folder=directory/f"videos/chunk-000/observation.images.{camera}"
        folder.mkdir(parents=True)
        source=av.open(str(root/f"videos/chunk-000/observation.images.{camera}/episode_000000.mp4"))
        # FFV1 帧内无损容器，避免压缩器的未来参考改变公共前缀像素。扩展名沿用适配器布局。
        target=av.open(str(folder/f"episode_{episode:06d}.mp4"),mode="w",format="matroska")
        stream=target.add_stream("ffv1",rate=25)
        stream.width,stream.height,stream.pix_fmt=512,384,"bgr0"
        for i,frame in enumerate(source.decode(video=0)):
            im=frame.to_image();im.thumbnail((512,512))
            rgb=np.asarray(im)
            if suffix_black and i>=prefix_frames:
                rgb=np.zeros_like(rgb)
            out=av.VideoFrame.from_ndarray(rgb,format="rgb24")
            out.pts=i;out.time_base=Fraction(1,25)
            for packet in stream.encode(out):
                target.mux(packet)
        for packet in stream.encode():
            target.mux(packet)
        source.close();target.close()
    return episode,prefix_frames/25


def validate(config,output=None):
    name=datetime.now(timezone(timedelta(hours=8))).strftime("validation_%Y%m%d_%H%M%S_%f")
    directory=Path(output) if output else BASE/"artifacts"/name
    directory.mkdir(parents=True,exist_ok=False)
    checks=[]

    def record(name,fn,kind):
        start=time.monotonic()
        try:
            detail=fn()
            checks.append({"name":name,"status":"passed","kind":kind,"detail":detail,"elapsed_s":time.monotonic()-start})
        except Exception as e:
            import traceback
            checks.append({"name":name,"status":"failed","kind":kind,"error":f"{type(e).__name__}: {e}",
                           "traceback":traceback.format_exc(),"elapsed_s":time.monotonic()-start})
        print(name+": "+checks[-1]["status"],flush=True)

    def logic():
        suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in
            ("streaming_recovery.tests.test_system","streaming_recovery.tests.test_incremental_regressions.IncrementalRegressions"))
        stream=io.StringIO()
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        (directory/"unit_tests.txt").write_text(stream.getvalue(),encoding="utf-8")
        assert result.wasSuccessful(),stream.getvalue()
        return {"tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),
                "long_synthetic_decisions":10000,"warning":"合成语义替身；不证明真实视觉准确率"}
    record("合成状态机、事件、坐标、回放、重置与 10000 步全历史容量",logic,"synthetic_semantic_inputs")

    real_cache={}
    def real_runs():
        for episode in (0,18):
            rows,audit,monitor=real_decisions(config,config["source"]["root"],episode)
            real_cache[episode]=rows
            (directory/f"real_{episode}_decisions.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False,allow_nan=False)+"\n" for r in rows),encoding="utf-8")
            assert audit["invalid_video_timestamps"]==0
            assert all(not r["execution"]["actually_executed"] for r in rows)
            if episode==0:
                assert monitor.end_of_stream()["task_success"] is None
        assert not any(r["notify"] for r in real_cache[18]),"正常片段误提醒"
        assert any(r["state"]["preconditions"]["reliable_grasp"] is True for r in real_cache[18]),"正常持刀未获得证据"
        if config["backend"]=="multiview_effect_geometry":
            stages={r["state"]["stage"] for r in real_cache[18]}
            assert {"transport_tool","cut_operation","return_tool","verify_completion","completed"}<=stages,stages
            assert any(r["visual_facts"]["measurements"].get("new_cut_effect")=="yes" for r in real_cache[18])
            assert real_cache[18][-1]["state"]["task_completed"],"没有获得全部任务效果证据"
            assert not real_cache[0][-1]["state"]["preconditions"]["cut_effect_confirmed"]
        failures=[r for r in real_cache[0] if r["execution_status"]=="confirmed_failure"]
        assert failures,"真实失败未确认"
        assert len({r["failure_event"]["event_id"] for r in failures})==1
        assert failures[0]["correction"]["low_level"]["action_type"]=="open_gripper"
        assert not failures[-1]["notify"],"持续失败重复提醒"
        # 标签只在全部推理结束后单独打开，不输入监测器。
        labels={}
        with (Path(config["source"]["root"])/"meta/vifailback_annotations.jsonl").open(encoding="utf-8") as file:
            for line in file:
                item=json.loads(line)
                if item["episode_index"] in (0,18):
                    labels[item["episode_index"]]={"episode_success":item["episode_success"],"avoid_keyframe_indices":item["avoid_keyframe_indices"]}
        return {"decisions":{str(k):len(v) for k,v in real_cache.items()},"independent_labels":labels,
                "first_suspicion_s":next(r["timestamp"] for r in real_cache[0] if r["execution_status"]=="suspicious"),
                "first_confirmation_s":failures[0]["timestamp"],
                "normal_grasp_confirmed_times":[r["timestamp"] for r in real_cache[18] if r["state"]["preconditions"]["reliable_grasp"] is True],
                "normal_stages":sorted({r["state"]["stage"] for r in real_cache[18]}),
                "normal_unknown_times":[r["timestamp"] for r in real_cache[18] if r["execution_status"]=="information_insufficient"],
                "normal_current_completion":real_cache[18][-1]["state"]["task_completed"],
                "scope":"仅演示；阈值用这两个可见场景开发，不能报告无偏识别精度或泛化性能",
                "first_detectable_failure_time":"not_annotated; avoid keyframe is not assumed to be first-detectable"}
    record("真实正常/失败完整回放与两层建议",real_runs,"real_robot_pixels_and_proprioception")

    def causal():
        a=directory/"prefix_fixture_A"
        b=directory/"prefix_fixture_B"
        ep_a,cut=build_prefix_fixture(config,a,False)
        ep_b,_=build_prefix_fixture(config,b,True)
        compared={}
        for backend in ("calibrated_rgb_geometry","multiview_effect_geometry"):
            cfg=deepcopy(config);cfg["backend"]=backend
            ra,aa,_=real_decisions(cfg,a,ep_a)
            rb,ab,_=real_decisions(cfg,b,ep_b)
            pa=[r for r in ra if r["timestamp"]<cut]
            pb=[r for r in rb if r["timestamp"]<cut]
            assert pa==pb,"公共前缀结果受到后缀/标签/路径改变影响:"+backend
            assert ra[-1]!=rb[-1],"后缀没有真正改变"
            compared[backend]=len(pa)
            (directory/f"prefix_A_{backend}.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in ra),encoding="utf-8")
            (directory/f"prefix_B_{backend}.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rb),encoding="utf-8")
        return {"common_prefix_native_frames":275,"identical_decisions":len(pa),"prefix_end_exclusive_s":cut,
                "suffix_modified":True,"path_episode_labels_actions_modified":True,
                "backends_compared":compared,"reader_progress_checked_at_every_yield":True,
                "coverage":"Parquet读取、逐帧解码、对齐、初始图像校准、视觉特征、记忆、状态、事件、两层纠错、结构化输出",
                "fixture_kind":"真实图像公共前缀＋人工修改后缀；不是新增真实机器人轨迹"}
    record("完整链路公共前缀因果性与标签路径隔离",causal,"real_pixels_counterfactual_suffix_fixture")

    def long_vision():
        root=Path(config["source"]["root"])
        source=LeRobotStream(root,0,["cam_high","cam_right_wrist"],10,.12,512,instruction_for_episode(root,0))
        cycle=list(source)
        model=make_vision(config)
        monitor=Monitor(config,model)
        maxima={}
        for i in range(2000):
            obs=cycle[i%len(cycle)]
            current=replace(obs,timestamp=float(i),ref=f"obs-{i:06d}",modalities={k:dict(v,timestamp=float(i)) for k,v in obs.modalities.items()})
            r=monitor.step(current)
            for k,v in r["memory_usage"].items():
                maxima[k]=max(maxima.get(k,0),v)
        assert maxima["visual_reference_images"]==1 and maxima["visual_masks"]<=config["geometry"]["max_masks"]
        assert maxima["visual_tracks"]<=2 and maxima["log_reads"]==0
        assert monitor.tracker.attempt>50 and maxima["events"]==config["memory"]["events"]
        return {"decisions":2000,"maxima":maxima,"observed_attempts":monitor.tracker.attempt,
                "kind":"循环真实失败片段制造动态长流，含位置/遮挡/打开闭合变化；非真实连续2000步轨迹"}
    record("实际双视角后端 2000 步动态全缓存容量",long_vision,"real_pixels_dynamic_engineering_stress")

    def missing_and_reset():
        import av
        from unittest.mock import patch
        root=Path(config["source"]["root"])
        vision=make_vision(config)
        monitor=Monitor(config,vision)
        source=lambda episode:LeRobotStream(root,episode,["cam_high","cam_right_wrist"],config["source"]["sample_every"],.12,512,instruction_for_episode(root,episode))
        for obs in source(0):
            monitor.step(obs)
        monitor.reset()
        reset_rows=[monitor.step(obs) for obs in source(18)]
        assert reset_rows==real_cache[18],"跨轨迹状态泄漏"
        monitor.reset()
        initial=next(iter(source(0)))
        monitor.step(initial)
        missing=replace(initial,timestamp=1.,ref="missing",images={},modalities={k:dict(v,status="missing",timestamp=None) for k,v in initial.modalities.items()})
        result=monitor.step(missing)
        assert result["execution_status"]=="information_insufficient"
        assert result["visual_facts"]["error"]=="current_high_image_missing"
        black=replace(initial,timestamp=2.,ref="black",images={"cam_high":Image.new("RGB",initial.images["cam_high"].size)},
                      modalities={k:dict(v,timestamp=2.) for k,v in initial.modalities.items()})
        result=monitor.step(black)
        assert result["execution_status"]=="information_insufficient"
        resumed=replace(initial,timestamp=3.,ref="resumed",modalities={k:dict(v,timestamp=3.) for k,v in initial.modalities.items()})
        assert monitor.step(resumed)["execution_status"]=="executing"
        monitor.reset()
        with patch("av.open",side_effect=ValueError("injected_open_failure")):
            failed_open=[monitor.step(obs) for obs in source(0)]
        assert len(failed_open)==len(real_cache[0]) and all(r["execution_status"]=="information_insufficient" for r in failed_open)
        assert failed_open[0]["modalities"]["cam_high"]["reason"].startswith("video_open_error:")
        original_open=av.open
        class BrokenDecoder:
            def __init__(self,path):
                self.container=original_open(path)
            def decode(self,video=0):
                yield next(self.container.decode(video=video))
                raise ValueError("injected_corrupt_packet_after_first_frame")
            def close(self):
                self.container.close()
        monitor.reset()
        with patch("av.open",side_effect=BrokenDecoder):
            failed_decode=[monitor.step(obs) for obs in source(0)]
        assert failed_decode[0]["execution_status"]=="executing"
        assert all(r["execution_status"]=="information_insufficient" for r in failed_decode[1:])
        assert failed_decode[1]["modalities"]["cam_high"]["reason"].startswith("video_decode_error:")
        monitor.reset()
        monitor.step(replace(missing,timestamp=0.,ref="initial-missing"))
        delayed=replace(initial,timestamp=1.,ref="delayed-clear",modalities={k:dict(v,timestamp=1.) for k,v in initial.modalities.items()})
        result=monitor.step(delayed)
        assert result["visual_facts"]["valid"],"缺失首帧后不能恢复校准"
        assert vision.reference_time==1.,"校准不得倒填到未看到的t=0"
        stale=replace(initial,timestamp=2.,ref="stale",modalities={k:dict(v,status="stale") for k,v in initial.modalities.items()})
        assert monitor.step(stale)["execution_status"]=="information_insufficient"
        return {"normal_after_failed_equals_fresh":True,"missing_high_continued":True,"black_frame_continued":True,
                "new_clear_frame_resumed":True,"video_open_failure_continued":True,"video_decode_failure_continued":True,
                "missing_initial_then_causal_calibration":True,"stale_images_filtered":True,
                "occlusion_kind":"人工缺帧/黑帧/解码故障注入，不是真实遮挡标注集"}
    record("真实后端缺失/遮挡恢复与跨轨迹重置",missing_and_reset,"real_pipeline_with_injected_faults")

    report={"checks":checks,"passed":all(c["status"]=="passed" for c in checks),
            "code_sha256":{p.name:sha256(p) for p in BASE.glob("*.py")},"config":config,
            "not_validated":["识别精度与跨场景泛化", "真实机器人重试和滑落样本", "真实坐标变换和物理动作执行",
                             "真实刀柄/两指联合定位及闭合准备", "Qwen可靠结构化诊断", "目标4完整实验"],
            "no_execution_claim":True,"synthetic_not_used_as_visual_accuracy":True}
    (directory/"validation.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print("验证记录："+str(directory),flush=True)
    if not report["passed"]:
        raise RuntimeError("one or more checks failed; inspect validation.json")
    return directory
