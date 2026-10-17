"""受限腕部图像定位：真实柄颈宽度过渡与两指内缘，不用对象区域质心。

输出是刀柄可见颈部的抓取参考点，不是完整刀具的质心，也不是三维抓取位姿。
必须同时看到细刃、宽柄颈和两指轮廓；混连、截断或宽度关系不清时拒识。
所有阈值均是本固定腕部安装的像素比例配置，无学习权重、轨迹编号或标签。
"""
from __future__ import annotations
import cv2
import numpy as np


def image_alignment_directions(knife_handle_xy, gripper_xy, deadband):
    """图像 x 向右、y 向下；不转换到机器人坐标，不产生物理数值。"""
    directions=[]
    for delta,negative,positive in zip(np.subtract(knife_handle_xy,gripper_xy),
                                     ("图像左侧","图像上方"),("图像右侧","图像下方")):
        if abs(float(delta))>deadband:
            directions.append(positive if delta>0 else negative)
    return directions


def localize_wrist_handle_and_fingers(image):
    im=np.asarray(image.convert("RGB")) if hasattr(image,"convert") else np.asarray(image)
    h,w=im.shape[:2]
    result={"status":"unknown","reason":None,"frame":"cam_right_wrist_image_x_right_y_down",
            "unit":"normalized_image_0_1000","image_size_pixels":[w,h],
            "knife_handle_xy":None,"gripper_xy":None,"finger_inner_points_pixels":[],
            "handle_neck_pixels":None,"rack_contour_pixels":[],
            "method":"white_blade_to_wider_handle_neck_and_two_finger_inner_edges"}
    if im.std()<8:
        result["reason"]="black_or_textureless_image";return result
    hsv=cv2.cvtColor(im,cv2.COLOR_RGB2HSV)
    dark=(hsv[:,:,2]<85)&(hsv[:,:,1]<150)
    contours=cv2.findContours(dark.astype("uint8"),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    fingers=[]
    for left in (True,False):
        candidates=[]
        for c in contours:
            x,y,bw,bh=cv2.boundingRect(c)
            if y+bh<h-2 or cv2.contourArea(c)<h*w*.02:
                continue
            p=c.reshape(-1,2)
            p=p[(p[:,1]>.60*h)&(p[:,1]<.75*h)]
            p=p[(p[:,0]>.18*w)&(p[:,0]<.48*w)] if left else p[(p[:,0]>.52*w)&(p[:,0]<.85*w)]
            if not len(p):continue
            edge=int(p[:,0].max() if left else p[:,0].min())
            edge_points=p[abs(p[:,0]-edge)<=2]
            # 内缘最上端角点；上方须是背景，否则可能把混连刀架当作手指。
            tip_y=int(edge_points[:,1].min())
            above=hsv[max(0,tip_y-18):max(1,tip_y-6),max(0,edge-2):min(w,edge+3),2]
            if not above.size or float(np.mean(above>95))<.6:
                continue
            candidates.append([float(edge),float(tip_y)])
        if len(candidates)!=1:
            result["reason"]="finger_inner_edge_missing_or_merged";return result
        fingers.append(candidates[0])
    if not (.12*w<fingers[1][0]-fingers[0][0]<.72*w and abs(fingers[0][1]-fingers[1][1])<.08*h):
        result["reason"]="two_finger_geometry_ambiguous";return result
    result["finger_inner_points_pixels"]=fingers
    racks=[]
    for c in contours:
        x,y,bw,bh=cv2.boundingRect(c)
        if (.12*w<x+bw/2<.80*w and 2<bh/max(bw,1)<8 and bh>.40*h and bw<.27*w
                and y+bh<h-5 and cv2.contourArea(c)>h*w*.015):
            racks.append(c)
    if len(racks)!=1:
        result["reason"]="separate_rack_contour_unavailable";return result
    c=racks[0]
    hull=cv2.convexHull(c)
    mask=np.zeros((h,w),np.uint8);cv2.drawContours(mask,[hull],0,1,-1)
    white=(hsv[:,:,1]<75)&(hsv[:,:,2]>110)&mask.astype(bool)
    x,y,bw,bh=cv2.boundingRect(c)
    rows=[]
    for yy in range(y+3,min(h-2,y+bh-3)):
        xx=np.flatnonzero(white[yy])
        if len(xx):
            # 只允许一个主要连续白色条段；多条反光不能拼成刀柄。
            groups=np.split(xx,np.flatnonzero(np.diff(xx)>1)+1)
            group=max(groups,key=len)
            if len(group)>=2:
                rows.append((yy,int(group[0]),int(group[-1]),len(group)))
    shaft=[r for r in rows if y+.08*bh<r[0]<y+.65*bh and 2<=r[3]<.025*w]
    if len(shaft)<.28*bh:
        result["reason"]="thin_blade_shaft_not_observed";return result
    shaft_width=float(np.median([r[3] for r in shaft]))
    slope,intercept=np.polyfit([r[0] for r in shaft],[(r[1]+r[2])/2 for r in shaft],1)
    fit_error=float(np.median([abs((r[1]+r[2])/2-(slope*r[0]+intercept)) for r in shaft]))
    if fit_error>3:
        result["reason"]="blade_shaft_not_straight";return result
    neck=[r for r in rows if r[0]>y+.62*bh and r[3]>=max(2.2*shaft_width,.022*w)
          and r[3]<.09*w and abs((r[1]+r[2])/2-(slope*r[0]+intercept))<.025*w]
    if len(neck)<6 or neck[5][0]-neck[0][0]>8 or neck[0][0]>h-12:
        result["reason"]="visible_handle_width_transition_unavailable";return result
    band=neck[1:6]
    center=[float(np.mean([(r[1]+r[2])/2 for r in band])),float(np.mean([r[0] for r in band]))]
    midpoint=[(fingers[0][0]+fingers[1][0])/2,(fingers[0][1]+fingers[1][1])/2]
    # 明确输出局部柄颈横截面、两指内缘；不输出深度、姿态或接触 readiness。
    result.update(status="localized",reason=None,
        knife_handle_xy=[round(center[0]/w*1000,3),round(center[1]/h*1000,3)],
        gripper_xy=[round(midpoint[0]/w*1000,3),round(midpoint[1]/h*1000,3)],
        handle_neck_pixels={"center":center,"edge_segments":[[r[1],r[0],r[2],r[0]]for r in band],
                            "thin_blade_width_pixels":shaft_width,"shaft_fit_error_pixels":fit_error},
        rack_contour_pixels=cv2.approxPolyDP(hull,2,True).reshape(-1,2).tolist()[:16])
    return result
