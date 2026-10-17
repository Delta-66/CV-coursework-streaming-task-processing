"""仅离线读审计日志生成查看页；本模块不被推理模块导入。"""
import html
import json
from pathlib import Path


def render_report(run_dir):
    run_dir=Path(run_dir)
    summary=json.loads((run_dir/"summary.json").read_text(encoding="utf-8"))
    cards=[]
    for episode in summary["episodes"]:
        directory=run_dir/episode["output_subdir"]
        for line in (directory/"decisions.jsonl").read_text(encoding="utf-8").splitlines():
            r=json.loads(line)
            state=r["state"]
            images="".join(f'<img src="{html.escape(episode["output_subdir"])}/frames/{r["observation_ref"]}_{camera}.jpg" alt="{camera} t={r["timestamp"]}">' for camera in ("cam_high","cam_right_wrist") if (directory/"frames"/f'{r["observation_ref"]}_{camera}.jpg').exists())
            advice=""
            if r["correction"]:
                high,low=r["correction"]["high_level"],r["correction"]["low_level"]
                advice=f'<p><b>恢复子目标：</b>{html.escape(high["next_recovery_subgoal"])}</p><p><b>下一动作：</b>{html.escape(low["instruction"])}</p><p><b>执行后检查：</b>{html.escape("；".join(low["post_checks"]))}</p>'
            failure=r["failure_event"]
            badge=f'{failure["event_id"]} / {failure["state"]}' if failure else '无失败事件'
            cards.append(f'<article data-episode="{episode["episode_for_audit_only"]}" data-alert="{str(r["notify"]).lower()}"><h2>轨迹 {episode["episode_for_audit_only"]} · {r["timestamp"]:.2f} s · {html.escape(state["stage_zh"])}</h2><p class="status">{r["execution_status"]} · {badge} · 提醒={r["notify"]}</p><div class="images">{images}</div><p>{html.escape(r["visual_facts"]["description"] or r["visual_facts"]["error"] or "未知")}</p><p><b>预期效果：</b>{html.escape(state["expected_effect"])}</p><p><b>待确认：</b>{html.escape("；".join(state["unknowns"]))}</p>{advice}<details><summary>完整结构化结果与容量记录</summary><pre>{html.escape(json.dumps(r,ensure_ascii=False,indent=2))}</pre></details></article>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>流式任务监测 · 实际运行</title><style>body{font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif;background:#eef2f5;color:#172b3a;margin:0}main{max-width:1120px;margin:30px auto;padding:0 20px}header,article{background:white;padding:24px;margin:16px 0;border-radius:10px;border:1px solid #d8e0e5}h1{font-size:26px}h2{font-size:20px}.status{color:#8b3b14}img{width:48%;height:auto}.images{display:flex;gap:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7f9;padding:12px;font-size:13px}select,label{margin-right:18px}article[data-alert=true]{border-left:5px solid #c66b30}</style><main><header><h1>机器人流式任务监测与分层纠错</h1><p>真实输入逐步回放。主方法：固定场景经典视觉模型＋有界状态跟踪。所有建议均未执行；未将原轨迹后续变化归因于建议。</p><p>支持取刀、转移及夹持前置条件。没有切割完成或归位证据时，整任务成功保持未知。</p><label>轨迹 <select id="episode"><option value="all">全部</option>'''
    page+=''.join(f'<option value="{e["episode_for_audit_only"]}">{e["episode_for_audit_only"]}</option>' for e in summary["episodes"])
    page+='''</select></label><label><input id="alerts" type="checkbox">仅显示提醒时刻</label><p><a href="summary.json">运行摘要</a> · <a href="config_snapshot.json">配置快照</a></p></header>'''+''.join(cards)+'''</main><script>function filter(){let e=document.querySelector('#episode').value,a=document.querySelector('#alerts').checked;for(let c of document.querySelectorAll('article'))c.hidden=(e!=='all'&&c.dataset.episode!==e)||(a&&c.dataset.alert!=='true');}document.querySelector('#episode').onchange=filter;document.querySelector('#alerts').onchange=filter;</script></html>'''
    (run_dir/"index.html").write_text(page,encoding="utf-8")

