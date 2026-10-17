"""固定相机/白柄刀/橙色香肠场景的双视角经典视觉；无轨迹编号或标签输入。
腕部刀刃连接、顶视刀架外观及新增分离区域分别给出证据。未提供深度/物理控制。
"""
from collections import deque
import cv2
import numpy as np
from .geometry import GeometryVision
from .localization import localize_wrist_handle_and_fingers

class MultiViewGeometryVision(GeometryVision):
    provenance = "real_multiview_effect_geometry"

    def reset(self):
        super().reset()
        self.rack_blade_mask = None
        self.food_reference = None
        self.wrist_tracks = deque(maxlen=3)
        self.tool_departed = False

    def food_parts(self, hsv, mask):
        food = mask & (hsv[:,:,0]>2) & (hsv[:,:,0]<20) & (hsv[:,:,1]>90) & (hsv[:,:,2]>125)
        food = cv2.morphologyEx(food.astype("uint8"),cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        parts=[]
        for c in self.contours(food):
            area=float(cv2.contourArea(c))
            if area<15:
                continue
            x,y,w,h=cv2.boundingRect(c)
            perimeter=cv2.arcLength(c,True)
            parts.append({"bbox":[x,y,w,h],"area":area,"roundness":4*np.pi*area/max(1,perimeter**2)})
        return sorted(parts,key=lambda p:p["area"],reverse=True)[:9]

    def calibrate(self, image, observation):
        # 允许在起始有限等待期重新尝试。缺帧期间不推断完成了什么。
        super().calibrate(image,observation)
        hsv=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        contours=[c for c in self.contours(hsv[:,:,2]<75) if cv2.boundingRect(c)==self.rack_box]
        if len(contours)!=1:
            self.reference=None
            raise ValueError("rack_blade_reference_unavailable")
        mask=np.zeros(image.shape[:2],dtype="uint8")
        cv2.drawContours(mask,[cv2.convexHull(contours[0])],0,1,-1)
        self.rack_blade_mask=mask.astype(bool)&(hsv[:,:,2]>90)&(hsv[:,:,1]<80)
        if self.rack_blade_mask.sum()<40:
            self.reference=None
            raise ValueError("rack_blade_reference_too_small")
        parts=self.food_parts(hsv,self.board_mask)
        main=next((p for p in parts if max(p["bbox"][2:])/min(p["bbox"][2:])>3 and p["area"]>150),None)
        if main is None:
            self.reference=None
            raise ValueError("initial_single_sausage_not_localized")
        small=[p for p in parts if p is not main and .06*main["area"]<p["area"]<.5*main["area"] and p["roundness"]>.45]
        self.food_reference={"timestamp":observation.timestamp,"ref":observation.ref,"main":main,"small":small[:4]}
        self.calibration_error=None

    def wrist_observation(self, observation):
        if "cam_right_wrist" not in observation.images:
            self.wrist_tracks.clear()
            return {"available":False,"blade":None,"cut_operation":False,"reason":"wrist_missing"}
        im=np.asarray(observation.images["cam_right_wrist"].convert("RGB"))
        h,w=im.shape[:2]
        if im.std()<8:
            self.wrist_tracks.clear()
            return {"available":False,"blade":None,"cut_operation":False,"reason":"wrist_black_or_occluded"}
        hsv=cv2.cvtColor(im,cv2.COLOR_RGB2HSV)
        candidates=[]
        for threshold in (120,160,180,200):
            white=(hsv[:,:,1]<75)&(hsv[:,:,2]>threshold)
            # 只检测两指尖端上方可见的刀刃段。刃下部镜面反光可延伸至画面底端，
            # 不能因此把同一把真实刀拒绝；仍要求中央、细长、足够长度以及两指夹持证据。
            white[int(h*.70):,:]=False
            # 香肠亮边可能与刀刃相连；纵向开运算保留可见长刃，去掉短横向亮桥。
            # 不改变颜色门槛、不用当前位置以外的图像，也不把宽背景裁成细刃。
            white=cv2.morphologyEx(white.astype("uint8"),cv2.MORPH_OPEN,
                                   np.ones((max(3,int(h*.28)),1),np.uint8)).astype(bool)
            for c in self.contours(white):
                x,y,bw,bh=cv2.boundingRect(c)
                if (.42*w < x+bw/2 < .62*w and bh>.28*h and bw<.13*w and bh/max(bw,1)>5 and
                    .60*h<y+bh<.79*h and cv2.contourArea(c)>h*w*.002):
                    candidates.append((x,y,bw,bh))
        blade=max(candidates,key=lambda b:b[3]) if candidates else None
        # 腕部条形白刃必须止于可见两指的尖端区域；与单纯闭爪原始值分离。
        joined=False
        if blade:
            x,y,bw,bh=blade
            edge_y=int(h*.68)  # 已声明腕部相机安装下的两指尖端带；不输出机器人坐标。
            x0,x1=int(x+bw/2-w*.06),int(x+bw/2+w*.06)
            left=hsv[edge_y:min(h,edge_y+15),max(0,x0-12):max(1,x0),2]
            right=hsv[edge_y:min(h,edge_y+15),x1:min(w,x1+12),2]
            joined=bool(left.size and right.size and np.mean(left<85)>.5 and np.mean(right<85)>.5)
        food=(hsv[:,:,0]>2)&(hsv[:,:,0]<25)&(hsv[:,:,1]>65)&(hsv[:,:,2]>140)
        band=np.zeros((h,w),bool);band[:int(h*.70),:]=True
        food_parts=self.contours(food & band)
        sausage=[]
        if blade:
            x,y,bw,bh=blade
            for c in food_parts:
                xx,yy,ww,hh=cv2.boundingRect(c)
                adjacent=min(abs(xx-(x+bw)),abs(xx+ww-x))<w*.07
                if ww>hh*2 and ww>w*.15 and hh>h*.025 and cv2.contourArea(c)>w*h*.004 and adjacent:
                    sausage.append((xx,yy,ww,hh))
        food_box=max(sausage,key=lambda b:b[2]) if sausage else None
        operation=False
        old=self.wrist_tracks[-1] if self.wrist_tracks else None
        if joined and food_box and old and old["food"] and observation.timestamp-old["timestamp"]<=self.rules["max_evidence_gap_s"]:
            # 刀刃邻接香肠端部，端部在刀刃参考系发生相对运动：操作证据，仍非切开效果。
            operation=abs(food_box[1]-old["food"][1])>=self.config.get("cut_stroke_min_pixels",2)
        row={"timestamp":observation.timestamp,"ref":observation.ref,"blade":list(blade) if blade else None,
             "joined":joined,"food":list(food_box) if food_box else None}
        self.wrist_tracks.append(row)
        return {"available":True,"blade":row["blade"],"joined":joined,"food":row["food"],
                "cut_operation":bool(operation),"previous_ref":old["ref"] if old else None,
                "previous_timestamp":old["timestamp"] if old else None,
                "blade_region":"visible_segment_above_finger_tip_band"}

    def observe(self, observation, previous=None):
        facts=super().observe(observation,previous)
        if not facts.valid:
            self.wrist_tracks.clear()
            return facts
        image=np.asarray(observation.images["cam_high"].convert("RGB"))
        hsv=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        m=facts.measurements
        m["gripper_evidence_source"]="observation.state[13] only; not independent visual corroboration"
        bright=float((hsv[:,:,2][self.rack_blade_mask]>90).mean())
        dark=float((hsv[:,:,2][self.rack_blade_mask]<60).mean())
        rack_clear=not m["source_occluded"]
        rack_present=bright>=self.config.get("rack_blade_present_fraction",.60) and dark<.35
        # 顶视图中机械手可在架的上方/侧方接近而尚未遮住刀。该粗区域变化仅用于
        # 接近阶段，不作为两指定位、接触或三维距离；区域来自起始已观测刀架。
        h,w=image.shape[:2]
        rx,ry,rw,rh=self.rack_box
        xa,xb=max(0,int(rx-.12*w)),min(w,int(rx+rw+.20*w))
        ya,yb=max(0,int(ry-.22*h)),min(h,int(ry+.35*rh))
        prior=cv2.cvtColor(self.reference,cv2.COLOR_RGB2HSV)
        near=(hsv[ya:yb,xa:xb,2]<70)&(prior[ya:yb,xa:xb,2]>100)
        near_fraction=float(near.mean()) if near.size else 0.
        robot_near_holder=near_fraction>.035
        m["robot_near_holder_projection"]={"present":bool(robot_near_holder),
            "new_dark_fraction":round(near_fraction,5),"region_pixels":[xa,ya,xb, yb],
            "meaning":"coarse_current_robot_projection_near_holder_not_contact_or_finger_localization"}
        # 先看当前顶视；刀架被遮挡/刀已离架时，才补看当前腕部确认刀—两指关系。
        # 不读未来腕部，不将未查询的腕部写作实际使用；旧腕部只保留有界几何特征。
        raw=observation.observation_state[13] if observation.observation_state else None
        opened=raw is not None and raw>=self.rules["gripper_open_raw"]
        # 开爪准备取刀时顶视仍不能解释柄—两指的投影关系，也查询当前腕图。
        need_wrist = not (rack_present and rack_clear) or opened
        if need_wrist:
            wrist=self.wrist_observation(observation)
            wrist["consulted"]="cam_right_wrist" in observation.images
            wrist["selection_reason"]="open_gripper_handle_alignment_unclear" if opened else "top_view_grasp_relation_ambiguous"
        else:
            self.wrist_tracks.clear()
            wrist={"available":False,"consulted":False,"blade":None,"cut_operation":False,
                   "reason":"top_view_tool_still_on_holder_and_separate"}
        m["wrist"]=wrist
        closed=raw is not None and raw<=self.rules["gripper_closed_raw"]
        # 腕部刀刃在两指之间，同时原刀架刃区已改变。两个视角均进入判定。
        connected=bool(wrist.get("joined") and closed and bright<.6)
        if connected:
            self.tool_departed=True
            facts.knife_location="gripper"
            facts.contact="between_fingers"
            facts.released="no"
            facts.motion="at_board" if m["robot_at_board"] else ("returning_holder" if m["source_occluded"] else "to_board")
            # 离架、接触由腕部解释；不虚构全局三维轨迹。
        elif rack_present and rack_clear:
            facts.knife_location="holder"
            facts.contact="separate"
            facts.released="yes"
            facts.motion="at_board" if m["robot_at_board"] else ("at_holder" if robot_near_holder else ("away" if self.tool_departed else "unknown"))
        elif m["tool_on_board"] and not m["robot_at_board"] and wrist.get("available") and not wrist.get("blade"):
            facts.knife_location="board"
            facts.contact="separate"
            facts.released="yes"
            facts.motion="unknown"
        elif facts.knife_location=="holder" and not rack_present:
            facts.knife_location="unknown"
            facts.contact="unknown"
            facts.released="unknown"
        facts.visibility="clear" if rack_clear else "partial"
        # 切割前只保存一个主物体及最多4个已有小块的几何参考。
        parts=self.food_parts(hsv,self.board_mask)
        ref=self.food_reference
        main=next((p for p in parts if p["area"]>.35*ref["main"]["area"] and max(p["bbox"][2:])/min(p["bbox"][2:])>3),None)
        small=[p for p in parts if p is not main and .06*ref["main"]["area"]<p["area"]<.5*ref["main"]["area"] and p["roundness"]>.45]
        rx,ry,rw,rh=ref["main"]["bbox"]
        def novel(p):
            x,y,w,h=p["bbox"];cx,cy=x+w/2,y+h/2
            outside=not (rx-3<=cx<=rx+rw+3 and ry-3<=cy<=ry+rh+3)
            old=any(abs(cx-(q["bbox"][0]+q["bbox"][2]/2))<12 and abs(cy-(q["bbox"][1]+q["bbox"][3]/2))<12 for q in ref["small"])
            return outside and not old
        clear_food = main is not None and not m["robot_at_board"] and len(parts)<=8
        facts.cut_piece=("yes" if small else "no") if clear_food else "unknown"
        m.update(rack_blade_bright_fraction=round(bright,5),rack_blade_dark_fraction=round(dark,5),
                 rack_present=bool(rack_present),wrist_contributes_connection=connected,
                 food_components=parts[:8],initial_food_reference=ref,
                 new_cut_effect="yes" if clear_food and any(novel(p) for p in small) else "unknown",
                 cut_reference_timestamp=ref["timestamp"],
                 cut_operation="yes" if wrist.get("cut_operation") and connected and m["robot_at_board"] else "unknown")
        # 此数据无经校准深度/接触反馈，绝不把腕部投影或区域质心当接触准备。
        facts.grasp_readiness={}
        localization=localize_wrist_handle_and_fingers(observation.images["cam_right_wrist"]) if wrist.get("consulted") and opened else {
            "status":"unknown","reason":"current_open_wrist_localization_not_requested"}
        m["localization"]=localization
        if localization["status"]=="localized":
            facts.knife_handle_xy=localization["knife_handle_xy"]
            facts.gripper_xy=localization["gripper_xy"]
            facts.localization_frame=localization["frame"]
            if facts.knife_location=="holder" and not m["robot_at_board"]:
                facts.motion="at_holder"
        facts.description=(f"刀架刃区亮像素比例={bright:.3f}；腕部刀刃—两指连接={wrist.get('joined', '未观测')}；"
                           f"刀架/腕部联合夹持证据={connected}；新分离小段={m['new_cut_effect']}；"
                           f"腕部刀—香肠相对作用={m['cut_operation']}")
        facts.evidence_refs=[{"ref":observation.ref,"timestamp":observation.timestamp,"camera":"cam_high"},
                             {"ref":observation.ref,"timestamp":observation.timestamp,"camera":"cam_right_wrist"}] if wrist["available"] else facts.evidence_refs
        facts.evidence_refs.append({"ref":self.reference_ref,"timestamp":self.reference_time,"camera":"cam_high"})
        if wrist.get("previous_ref"):
            facts.evidence_refs.append({"ref":wrist["previous_ref"],"timestamp":wrist["previous_timestamp"],"camera":"cam_right_wrist"})
        facts.evidence_refs=facts.evidence_refs[-4:]
        self.last_metrics={"calls":self.calls,"input_images":2+int(wrist["consulted"]),
                           "images_used":["cam_high"]+(["cam_right_wrist"] if wrist["consulted"] else []),
                           "input_tokens":0,"output_tokens":0,
                           "raw":"","provenance":self.provenance,"measurements":m}
        return facts

    def usage(self):
        u=super().usage()
        u["visual_masks"]+=int(self.rack_blade_mask is not None)
        u["visual_mask_bytes"]+=self.rack_blade_mask.nbytes if self.rack_blade_mask is not None else 0
        u.update(wrist_track_entries=len(self.wrist_tracks),
                 food_reference_entries=1+len(self.food_reference["small"]) if self.food_reference else 0,
                 visual_scalar_states=1)
        return u
