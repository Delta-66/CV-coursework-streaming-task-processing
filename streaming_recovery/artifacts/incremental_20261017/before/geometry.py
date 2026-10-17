"""场景受限的经典视觉模型，不是训练过的通用检测器。

假设：固定顶视相机、首帧棕色砧板与右侧黑刀架可见、白柄刀是进入板面的
唯一大面积浅色细长物体。所有参考均从首个到达图像建立；不读轨迹标识。
"""
from __future__ import annotations

import numpy as np
import cv2
from .contracts import VisualFacts


class GeometryVision:
    provenance = "real_rgb_calibrated_geometry"

    def __init__(self, config, rules):
        self.config, self.rules = config, rules
        self.reset()

    def reset(self):
        self.reference = self.board_mask = self.source_mask = self.watch_mask = None
        self.reference_ref = None
        self.reference_time = None
        self.rack_box = None
        self.previous_tracks = None
        self.last_metrics = {}
        self.calls = 0
        self.calibration_error = None

    @staticmethod
    def contours(mask):
        return cv2.findContours(mask.astype("uint8") * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    def calibrate(self, image, observation):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        brown = (hsv[:,:,0] < 25) & (hsv[:,:,1] > 65) & (hsv[:,:,2] > 30) & (hsv[:,:,2] < 180)
        contours = self.contours(brown)
        if not contours:
            raise ValueError("initial_board_not_found")
        board = max(contours, key=cv2.contourArea)
        height, width = image.shape[:2]
        if not .06 < cv2.contourArea(board) / (height*width) < .35:
            raise ValueError("initial_board_area_out_of_profile")
        bx, by, bw, bh = cv2.boundingRect(board)
        candidates = []
        for contour in self.contours(hsv[:,:,2] < 75):
            x,y,w,h = cv2.boundingRect(contour)
            if x > bx+bw*.85 and by - bh*.3 < y < by+bh*.5 and 1.5 < h/max(w,1) < 6 and h > bh*.5 and cv2.contourArea(contour) > height*width*.003:
                candidates.append((x, y, w, h))
        if len(candidates) != 1:
            raise ValueError("initial_rack_ambiguous_or_missing")
        x,y,w,h = self.rack_box = candidates[0]
        mask = np.zeros((height,width), dtype="uint8")
        cv2.drawContours(mask,[board],-1,1,-1)
        radius = max(3, int(width/80) | 1)
        self.board_mask = cv2.erode(mask,np.ones((radius,radius),dtype="uint8")).astype(bool)
        self.source_mask = np.zeros((height,width),dtype=bool)
        self.source_mask[y:y+h,x+int(w*.45):x+w] = True
        self.source_mask &= (hsv[:,:,2] > 155) & (hsv[:,:,1] < 70)
        if self.source_mask.sum() < height*width*.002:
            raise ValueError("initial_source_appearance_insufficient")
        # 固定工作区之外的墙面：检测相机大幅移动或整帧黑屏，不用未来稳像。
        self.watch_mask = np.zeros((height,width),dtype=bool)
        self.watch_mask[int(height*.03):int(height*.22),int(width*.1):int(width*.55)] = True
        self.reference = image.copy()
        self.reference_ref, self.reference_time = observation.ref, observation.timestamp

    def observe(self, observation, previous=None):
        self.calls += 1
        refs = [{"ref": observation.ref,"camera":"cam_high","timestamp":observation.timestamp}]
        if "cam_high" not in observation.images:
            self.previous_tracks = None
            return VisualFacts(observation.timestamp, observation.ref, error="current_high_image_missing",
                               provenance=self.provenance, evidence_refs=refs)
        image = np.asarray(observation.images["cam_high"].convert("RGB"))
        if self.reference is None:
            if self.calibration_error is not None or observation.timestamp != 0:
                return VisualFacts(observation.timestamp,observation.ref,error="initial_calibration_failed_restart_from_clear_t0",
                                   provenance=self.provenance,evidence_refs=refs)
            try:
                self.calibrate(image, observation)
            except ValueError as e:
                self.calibration_error = str(e)
                return VisualFacts(observation.timestamp, observation.ref, error=str(e),provenance=self.provenance,evidence_refs=refs)
        if image.shape != self.reference.shape:
            return VisualFacts(observation.timestamp, observation.ref, error="camera_shape_changed", provenance=self.provenance,evidence_refs=refs)
        hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        initial = cv2.cvtColor(self.reference,cv2.COLOR_RGB2HSV)
        difference = np.max(np.abs(image.astype("int16") - self.reference.astype("int16")),axis=2)
        watch_change = float((difference[self.watch_mask] > 45).mean())
        if watch_change > .35 or image.std() < 8:
            self.previous_tracks = None
            return VisualFacts(observation.timestamp,observation.ref,error="scene_changed_or_image_occluded",
                               provenance=self.provenance,evidence_refs=refs)
        changed = float((difference[self.source_mask] > self.config["pixel_difference"]).mean())
        dark_source = float((hsv[:,:,2][self.source_mask] < 90).mean())
        source_unchanged = changed < self.config["source_unchanged_fraction"] and dark_source < .1
        source_occluded = dark_source > self.config["source_occluded_fraction"]
        white = (self.board_mask & (hsv[:,:,2] > 160) & (hsv[:,:,1] < 65) &
                 (hsv[:,:,2].astype("int16") - initial[:,:,2].astype("int16") > 45))
        robot = (self.board_mask & (hsv[:,:,2] < 65) &
                 (initial[:,:,2].astype("int16") - hsv[:,:,2].astype("int16") > 15))
        board_area = int(self.board_mask.sum())
        white_fraction = float(white.sum()/board_area)
        robot_fraction = float(robot.sum()/board_area)
        parts = sorted(self.contours(white),key=cv2.contourArea,reverse=True)
        elongated = False
        if parts:
            x,y,w,h = cv2.boundingRect(parts[0])
            elongated = max(w,h)/max(1,min(w,h)) > 1.6 and cv2.contourArea(parts[0]) > board_area*.012
        tool_on_board = white_fraction > self.config["tool_board_fraction"] and elongated
        robot_at_board = robot_fraction > self.config["robot_board_fraction"]
        height,width = image.shape[:2]
        center = lambda mask: [float(np.nonzero(mask)[1].mean()/width*1000),float(np.nonzero(mask)[0].mean()/height*1000)] if mask.any() else None
        knife_point = center(white) if tool_on_board else None
        robot_point = center(robot) if robot_at_board else None
        raw = observation.observation_state[13] if observation.observation_state else None
        closed = raw is not None and raw <= self.rules["gripper_closed_raw"]
        opened = raw is not None and raw >= self.rules["gripper_open_raw"]
        common_motion = False
        motion_metrics = None
        if tool_on_board and robot_at_board and self.previous_tracks is not None:
            old = self.previous_tracks
            if observation.timestamp-old["timestamp"] <= self.rules["max_evidence_gap_s"]:
                a = np.array(knife_point)-old["knife"]
                b = np.array(robot_point)-old["robot"]
                motion_metrics = {"tool_motion_norm":float(np.linalg.norm(a)),"robot_motion_norm":float(np.linalg.norm(b)),
                                  "relative_motion_norm":float(np.linalg.norm(a-b)),"dot":float(a@b)}
                common_motion = (min(np.linalg.norm(a),np.linalg.norm(b)) >= self.config["min_motion_normalized"] and
                                 np.linalg.norm(a-b) <= self.config["max_relative_motion_normalized"] and a@b > 0)
                refs.append({"ref":old["ref"],"camera":"cam_high","timestamp":old["timestamp"]})
        self.previous_tracks = {"knife":knife_point,"robot":robot_point,"timestamp":observation.timestamp,"ref":observation.ref} if tool_on_board and robot_at_board else None
        held_evidence = tool_on_board and robot_at_board and common_motion and closed and not source_unchanged
        # 刀仍在原架上 + 工作区只有机器人、没有刀：支持未建立搬运前置条件。
        location = "gripper" if held_evidence else ("holder" if source_unchanged else "unknown")
        contact = "co_moving" if held_evidence else ("separate" if source_unchanged and robot_at_board and not tool_on_board else "unknown")
        motion = "at_board" if robot_at_board else ("at_holder" if source_occluded else "unknown")
        refs.append({"ref":self.reference_ref,"camera":"cam_high","timestamp":self.reference_time})
        measurements = {"source_changed_fraction":round(changed,5),"source_dark_fraction":round(dark_source,5),
            "tool_board_fraction":round(white_fraction,5),"robot_board_fraction":round(robot_fraction,5),
            "source_unchanged":source_unchanged,"source_occluded":source_occluded,
            "tool_on_board":bool(tool_on_board),"robot_at_board":bool(robot_at_board),
            "common_motion":bool(common_motion),"motion":motion_metrics,"reference_origin":"first_arrived_frame",
            "point_unit":"normalized_image_0_1000","reference_rack_bbox_pixels":list(self.rack_box),
            "scene_assumptions":"fixed_camera_brown_board_black_rack_white_knife_only"}
        self.last_metrics = {"calls":self.calls,"input_images":1,"input_tokens":0,"output_tokens":0,
                             "raw":"","provenance":self.provenance,"measurements":measurements}
        description = (f"刀架参考区域变化比例={changed:.3f}；砧板浅色刀具比例={white_fraction:.3f}；"
                       f"机器人遮挡比例={robot_fraction:.3f}；共同运动={bool(common_motion)}")
        return VisualFacts(observation.timestamp,observation.ref,visibility="partial" if source_occluded else "clear",
            knife_location=location,gripper="closed" if closed else ("open" if opened else "unknown"),motion=motion,
            contact=contact,co_motion="yes" if common_motion else "unknown",cut_piece="unknown",
            released="yes" if contact=="separate" else ("no" if held_evidence else "unknown"),
            # 不将机器人像素质心冒充夹爪中点，不把刀像素质心冒充刀柄。
            knife_handle_xy=None,gripper_xy=None,description=description,valid=True,
            provenance=self.provenance,evidence_refs=refs[-3:],measurements=measurements)

    def usage(self):
        return {"visual_reference_images":int(self.reference is not None),
                "visual_reference_bytes":int(self.reference.nbytes) if self.reference is not None else 0,
                "visual_masks":sum(x is not None for x in (self.board_mask,self.source_mask,self.watch_mask)),
                "visual_mask_bytes":sum(x.nbytes for x in (self.board_mask,self.source_mask,self.watch_mask) if x is not None),
                "visual_tracks":2 if self.previous_tracks is not None else 0,
                "visual_calibration_entries":2 if self.reference is not None else 0}

    def close(self):
        pass
