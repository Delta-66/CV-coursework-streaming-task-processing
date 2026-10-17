"""离线审计图；仅读取已完成的运行日志，不参与推理。"""
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont

base=Path(__file__).resolve().parent
run=base/"verified_run"
font=ImageFont.truetype("C:/Windows/Fonts/msyh.ttc",23)
small=ImageFont.truetype("C:/Windows/Fonts/msyh.ttc",18)
title=ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc",30)
canvas=Image.new("RGB",(1100,1190),"#eff3f6")
d=ImageDraw.Draw(canvas)
d.text((26,18),"真实轨迹运行证据 · 取刀前置条件监测",font=title,fill="#173047")
d.text((26,65),"经典视觉基线｜当前输入与历史有界｜两层建议均未实际执行",font=small,fill="#42566a")

examples=[]
for y,ep,t,label,color in [(112,0,10,"失败片段：刀仍在刀架，空夹爪已进入工作区","#a94023"),
                            (650,18,11,"正常片段：刀具与机器人运动关联支持持刀","#176b61")]:
    folder=run/f"trajectory_{ep:06d}"
    rows=[json.loads(line) for line in (folder/"decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    row=next(r for r in rows if r["timestamp"]==t)
    examples.append(row)
    d.rounded_rectangle((18,y-4,1082,y+512),radius=12,fill="white")
    d.text((32,y+8),f"{label}  ·  t = {t:.0f} s",font=font,fill=color)
    for x,camera,name in [(32,"cam_high","顶视图：用于判定"),(568,"cam_right_wrist","右腕图：仅供人工核查")]:
        im=Image.open(folder/"frames"/f'{row["observation_ref"]}_{camera}.jpg')
        im.thumbnail((500,375))
        canvas.paste(im,(x,y+52))
        d.text((x,y+430),name,font=small,fill="#42566a")
    if ep==0:
        d.text((32,y+463),"event-0001：确认失败 → 恢复抓取条件 → 先安全打开右夹爪，再检查",font=small,fill=color)
        d.text((32,y+487),"9 s 疑似；10 s 确认；11 s 沿用事件，不重复提醒。物理移动量未生成。",font=small,fill="#42566a")
    else:
        d.text((32,y+463),"11–13 s 获得持刀证据；未触发失败提醒；切割完成与归位仍未确认。",font=small,fill=color)
        d.text((32,y+487),"两个已阅开发样例的功能展示，不代表独立测试集精度或跨场景泛化。",font=small,fill="#42566a")
canvas.save(base/"实际运行证据.png")
(base/"correction_example.json").write_text(json.dumps(examples[0],ensure_ascii=False,indent=2),encoding="utf-8")
(base/"normal_example.json").write_text(json.dumps(examples[1],ensure_ascii=False,indent=2),encoding="utf-8")
print(base/"实际运行证据.png")
