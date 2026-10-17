"""实际开→闭转换：复用 armed 一个布尔锁存，支持平滑闭爪与阈值滞回。

纯函数，不保留图像、坐标、原始历史或建议。位置关系由当前视觉事实提供。
闭合消耗打开证据，即使当时不在目标附近也不能日后把同一次闭合补记成尝试。
"""
from math import isfinite


def attempt_transition(armed,raw,visual_gripper,independent,good,gap,near_object,rules):
    """返回 (新的 armed, closing_edge, new_attempt)，不改变现有变量名。

    raw 已由引擎过滤为当前实测值或 None。派生自 raw 的 visual_gripper 不是另一
    个传感器，raw 缺失/过期时不能借此继续使用打开历史。只有独立视觉来源允许补证。
    开阈值与闭阈值之间是滞回带，连续有效中间值保持锁存，而非要求相邻大幅变化。
    """
    if gap:
        armed=False
    raw_available=(raw is not None and not isinstance(raw,bool) and
                   isinstance(raw,(int,float)) and isfinite(raw))
    visual_available=independent and visual_gripper in ("open","closed")
    if not good or not (raw_available or visual_available):
        return False,False,False
    raw_open=raw_available and raw>=rules["gripper_open_raw"]
    raw_closed=raw_available and raw<=rules["gripper_closed_raw"]
    visual_open=visual_available and visual_gripper=="open"
    visual_closed=visual_available and visual_gripper=="closed"
    if (raw_open and visual_closed) or (raw_closed and visual_open):
        return False,False,False
    if raw_open or visual_open:
        return True,False,False
    if raw_closed or visual_closed:
        closing_edge=bool(armed)
        return False,closing_edge,bool(closing_edge and near_object)
    return bool(armed),False,False
