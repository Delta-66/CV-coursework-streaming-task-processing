"""仅用当前实测 qvel 和有限窗口中的前一个 qpos；不读取 action/EEF/轨迹统计量。

位置与夹爪来自同一向量。右臂任务只使用右侧六关节，排除夹爪维，避免把闭爪当作手臂移动。
阈值是显式原始单位工程阈值，不是校准后的物理速度或成功概率。
"""
from math import isfinite


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
    if instantaneous == "moving":
        status = "moving"
    elif interval == "moved":
        status = "recent_motion"  # 过去采样间隔发生移动，不宣称当前瞬间仍在运动。
    elif speed is not None and delta is not None:
        status = "stationary"
    else:
        status = "unknown"
    return {"status": status, "instantaneous": instantaneous, "interval": interval,
            "arm": "right", "joint_indices": [7, 8, 9, 10, 11, 12],
            "max_abs_velocity_raw": speed, "max_abs_position_change_raw": delta, "interval_s": dt,
            "position_threshold_raw": position_threshold, "velocity_threshold_raw": velocity_threshold,
            "units": "source qpos/qvel units; physical units not verified",
            "position_available": qpos is not None, "velocity_available": qvel is not None,
            "evidence_refs": refs, "source": "measured_joint_state; gripper excluded; no action or EEF"}
