"""Offline actual-image inspection; episode/time selectors are never inference inputs."""
import json
from pathlib import Path
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[3]
OLD=ROOT/'streaming_recovery/artifacts/multimodal_20261017/final_run'
OUT=Path(__file__).resolve().parent

def sheet(ep, seconds, camera, name):
    cols=3
    result=Image.new('RGB',(512*cols,414*((len(seconds)+cols-1)//cols)),'white')
    draw=ImageDraw.Draw(result)
    for i,t in enumerate(seconds):
        p=OLD/f'trajectory_{ep:06d}/frames/obs-{round(t*25):06d}_{camera}.jpg'
        im=Image.open(p)
        x,y=(i%cols)*512,(i//cols)*414
        result.paste(im,(x,y+30))
        draw.text((x+8,y+8),f'episode {ep}; t={t}s; {camera}',fill='black')
    result.save(OUT/name)

if __name__=='__main__':
    sheet(19,[3,4,5,6,7,8],'cam_right_wrist','visual_19_wrist_before.png')
    for ep in (2,8):
        sheet(ep,[1,2,3,4,5,6],'cam_right_wrist',f'visual_{ep}_wrist_before.png')
        sheet(ep,[1,2,3,4,5,6],'cam_high',f'visual_{ep}_high_before.png')
    rows=[]
    for ep in (2,8,19):
        for line in (OLD/f'trajectory_{ep:06d}/decisions.jsonl').read_text(encoding='utf8').splitlines():
            r=json.loads(line);f=r['visual_facts'];m=f['measurements']
            rows.append({'episode':ep,'timestamp':r['timestamp'],'stage':r['state']['stage'],
                         'location':f['knife_location'],'motion':f['motion'],'gripper':f['gripper'],
                         'wrist':m.get('wrist'),'cut_operation':m.get('cut_operation'),
                         'arm_activity':r['state']['arm_activity'],'new_cut_effect':m.get('new_cut_effect')})
    (OUT/'visual_before_observations.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
