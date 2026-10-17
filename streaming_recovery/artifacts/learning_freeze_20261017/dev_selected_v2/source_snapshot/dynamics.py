"""仅用已到达实测 qpos/qvel 的固定容量区间汇总和有限窗口；不读取 action/EEF/全轨迹统计量。

位置与夹爪来自同一向量。右臂任务只使用右侧六关节，排除夹爪维，避免把闭爪当作手臂移动。
阈值是显式原始单位工程阈值，不是校准后的物理速度或成功概率。
"""
from math import isfinite


class JointInterval:
    """25 FPS 已到达数据的区间聚合：前一位置6值＋路径和6值＋固定标量。"""
    def __init__(self, fps=25.):
        self.fps=fps
        self.reset()

    def reset(self):
        self.first_time=self.last_time=None
        self.first_ref=self.last_ref=None
        self.last_position=None
        self.path_length=[0.]*6
        self.peak_velocity=None
        self.count=self.position_count=self.velocity_count=0
        self.contiguous=True

    def push(self, timestamp, qpos, qvel, ref):
        if not isfinite(timestamp) or (self.last_time is not None and timestamp<=self.last_time):
            raise ValueError("native_joint_clock_reversal")
        if self.last_time is not None and abs(timestamp-self.last_time-1/self.fps)>1e-5:
            self.contiguous=False
        if self.first_time is None:
            self.first_time,self.first_ref=timestamp,ref
        self.last_time,self.last_ref=timestamp,ref
        self.count+=1
        if valid_vector(qpos):
            values=qpos[7:13]
            if self.last_position is not None:
                self.path_length=[a+abs(b-c) for a,b,c in zip(self.path_length,values,self.last_position)]
            self.last_position=values
            self.position_count+=1
        else:
            self.last_position=None
        if valid_vector(qvel):
            peak=max(abs(v) for v in qvel[7:13])
            self.peak_velocity=peak if self.peak_velocity is None else max(self.peak_velocity,peak)
            self.velocity_count+=1

    def snapshot(self):
        if self.last_time is None:
            return None
        span=self.last_time-self.first_time
        expected=round(span*self.fps)+1
        return {"source":"native_measured_joint_interval","start_timestamp":self.first_time,
                "end_timestamp":self.last_time,"start_ref":self.first_ref,"end_ref":self.last_ref,
                "span_s":span,"sample_count":self.count,"expected_sample_count":expected,
                "valid_position_samples":self.position_count,"valid_velocity_samples":self.velocity_count,
                "contiguous":self.contiguous,
                "complete":bool(span>0 and self.contiguous and self.count==expected==self.position_count==self.velocity_count),
                "max_abs_velocity_raw":self.peak_velocity,
                "max_joint_path_raw":max(self.path_length) if self.position_count else None}

    def usage(self):
        return {"native_joint_previous_values":len(self.last_position or []),"native_joint_path_values":6,
                "native_joint_scalar_slots":10,"native_joint_history_samples":0}


def valid_interval(value, now):
    keys={"source","start_timestamp","end_timestamp","start_ref","end_ref","span_s","sample_count",
          "expected_sample_count","valid_position_samples","valid_velocity_samples","contiguous","complete",
          "max_abs_velocity_raw","max_joint_path_raw"}
    if not isinstance(value,dict) or set(value)!=keys or value["source"]!="native_measured_joint_interval":
        return False
    for k in ("start_timestamp","end_timestamp","span_s"):
        if isinstance(value[k],bool) or not isinstance(value[k],(int,float)) or not isfinite(value[k]):
            return False
    if value["end_timestamp"]!=now or not 0<=value["start_timestamp"]<=now or abs(value["span_s"]-(now-value["start_timestamp"]))>1e-6:
        return False
    if any(not isinstance(value[k],str) or not 0<len(value[k])<=80 for k in ("start_ref","end_ref")):
        return False
    for k in ("sample_count","expected_sample_count","valid_position_samples","valid_velocity_samples"):
        if type(value[k]) is not int or not 0<=value[k]<=1_000_000:
            return False
    if not 0<value["sample_count"] or any(value[k]>value["sample_count"] for k in ("valid_position_samples","valid_velocity_samples")):
        return False
    if any(type(value[k]) is not bool for k in ("complete","contiguous")):
        return False
    for k in ("max_abs_velocity_raw","max_joint_path_raw"):
        if value[k] is not None and (isinstance(value[k],bool) or not isinstance(value[k],(int,float)) or not isfinite(value[k]) or value[k]<0):
            return False
    if value["complete"] and not (value["span_s"]>0 and value["sample_count"]>=2 and value["contiguous"] and
        value["sample_count"]==value["expected_sample_count"]==value["valid_position_samples"]==value["valid_velocity_samples"] and
        value["max_abs_velocity_raw"] is not None and value["max_joint_path_raw"] is not None):
        return False
    return True


def valid_vector(value):
    return (isinstance(value, list) and len(value) == 14 and
            all(not isinstance(v, bool) and isinstance(v, (float, int)) and isfinite(v) for v in value))


def joint_motion(observation, previous, rules):
    qpos = observation.observation_state
    qvel = observation.observation_velocity
    qpos = qpos if valid_vector(qpos) else None
    qvel = qvel if valid_vector(qvel) else None
    speed = max(abs(v) for v in qvel[7:13]) if qvel is not None else None
    delta = None
    refs = []
    if qpos is not None:
        refs.append({"ref": observation.ref, "timestamp": observation.timestamp, "field": "observation.state[7:13]"})
    if qvel is not None:
        refs.append({"ref": observation.ref, "timestamp": observation.timestamp, "field": "observation.velocity[7:13]"})
    dt = observation.timestamp - previous.timestamp if previous is not None else None
    if (qpos is not None and previous is not None and valid_vector(previous.observation_state) and
            0 < dt <= rules["max_evidence_gap_s"]):
        delta = max(abs(a-b) for a, b in zip(qpos[7:13], previous.observation_state[7:13]))
        refs.append({"ref": previous.ref, "timestamp": previous.timestamp, "field": "observation.state[7:13]"})
    velocity_threshold = rules.get("joint_velocity_moving_raw", .20)
    position_threshold = rules.get("joint_position_change_raw", .02)
    instantaneous = "unknown" if speed is None else ("moving" if speed >= velocity_threshold else "below_motion_threshold")
    interval = "unknown" if delta is None else ("moved" if delta >= position_threshold else "no_resolved_change")
    native=observation.joint_interval
    native=native if valid_interval(native,observation.timestamp) and native["span_s"]<=rules["max_evidence_gap_s"] else None
    coverage="endpoints_only"
    if native is not None:
        covers_decision_interval=(previous is not None and 0<dt<=rules["max_evidence_gap_s"] and
                                  native["start_timestamp"]<=previous.timestamp+1e-6)
        coverage=("native_complete" if native["complete"] and covers_decision_interval else
                  "native_partial_window" if native["complete"] else "native_partial")
        # 决策/事件引用实际参与判断的原生摘要区间；替换第3个端点引用，数量仍至多3。
        refs=[ref for ref in refs if not (ref["field"]=="observation.state[7:13]" and
                                          ref["timestamp"]!=observation.timestamp)]
        refs.append({"ref":native["end_ref"],"timestamp":native["end_timestamp"],
                     "field":"native_measured_joint_interval","start_ref":native["start_ref"],
                     "start_timestamp":native["start_timestamp"],"sample_count":native["sample_count"],
                     "complete":native["complete"]})
        peak,path=native["max_abs_velocity_raw"],native["max_joint_path_raw"]
        if (peak is not None and peak>=velocity_threshold) or (path is not None and path>=position_threshold):
            interval="moved"
        elif native["complete"] and covers_decision_interval and interval!="moved":
            interval="no_resolved_native_motion"
    if instantaneous == "moving":
        status = "moving"
    elif interval == "moved":
        status = "recent_motion"  # 过去采样间隔发生移动，不宣称当前瞬间仍在运动。
    elif interval=="no_resolved_native_motion" and native["complete"]:
        status = "stationary"
    elif speed is not None and delta is not None:
        status = "low_motion_support"  # 端点低差不能证明区间内没有往复。
    else:
        status = "unknown"
    return {"status": status, "instantaneous": instantaneous, "interval": interval,
            "arm": "right", "joint_indices": [7, 8, 9, 10, 11, 12],
            "max_abs_velocity_raw": speed, "max_abs_position_change_raw": delta, "interval_s": dt,
            "position_threshold_raw": position_threshold, "velocity_threshold_raw": velocity_threshold,
            "units": "source qpos/qvel units; physical units not verified",
            "position_available": qpos is not None, "velocity_available": qvel is not None,
            "interval_coverage":coverage,"native_interval":native,
            "evidence_refs": refs, "source": "measured_joint_state; gripper excluded; no action or EEF"}
