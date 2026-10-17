import cv2, json
import numpy as np
from pathlib import Path
from PIL import Image,ImageDraw

BASE=Path(__file__).resolve().parents[1]/'multimodal_20261017/final_run'
OUT=Path(__file__).resolve().parent

def probe(ep,t):
    p=BASE/f'trajectory_{ep:06d}/frames/obs-{round(t*25):06d}_cam_right_wrist.jpg'
    im=np.asarray(Image.open(p));h,w=im.shape[:2]
    hsv=cv2.cvtColor(im,cv2.COLOR_RGB2HSV)
    result=Image.fromarray(im);d=ImageDraw.Draw(result)
    cs=cv2.findContours(((hsv[:,:,2]<85)&(hsv[:,:,1]<150)).astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    rows=[]
    for c in cs:
        if cv2.contourArea(c)<400:continue
        x,y,bw,bh=cv2.boundingRect(c)
        rows.append({'bbox':[x,y,bw,bh],'area':cv2.contourArea(c),'contour':c.reshape(-1,2).tolist()})
        d.line([tuple(v) for v in c.reshape(-1,2)],fill='red',width=2)
        d.text((x,y),str(len(rows)),fill='red')
    result.save(OUT/f'visual_probe_{ep}_{t}.png')
    return {'episode':ep,'t':t,'components':rows}

if __name__=='__main__':
    rows=[probe(ep,t) for ep,t in [(2,2),(2,3),(8,2),(8,4),(0,3),(19,6)]]
    (OUT/'visual_probe_contours.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
    print([{**{k:v for k,v in r.items() if k!='components'},'boxes':[{k:v for k,v in c.items() if k!='contour'}for c in r['components']]}for r in rows])
