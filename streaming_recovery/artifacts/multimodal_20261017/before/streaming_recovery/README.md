# 机器人操作的流式理解、失败诊断与分层纠错

这是对现有 `streaming_recovery/` 的增量修复，范围为课题目标 1—3。当前主方法是 **`bounded_visual_state_tracker`＋`multiview_effect_geometry`**：真实顶视/右腕图像、夹爪原始状态、有限记忆与透明规则。没有训练新模型，没有接入真机。

已实际回放 5 条取刀—切香肠—归位轨迹，共 1,790 个原生时刻、3,580 张解码图、180 次判断。正常 episode 18、19 分别在 19.6 s、15.6 s 获得完整任务效果证据；失败 episode 0、2、6 分别在 9.6 s、10.4 s、9.6 s 确认异常。**这些是功能演示，不是识别准确率或泛化成绩。** episode 19 首次冻结验证漏掉切割操作，修复后已归入本轮开发，原始漏检记录保留。

真实刀柄/两指联合定位、独立接触准备、完整真实重试建议过程仍未验证。当前实际纠错演示为“恢复夹持前置条件→条件允许时打开空夹爪”；二维对齐、准备检查、闭合建议和效果确认的推进链已用明确标记的合成输入验证。不会把这两类结果混写。

## 运行

在项目根目录的 PowerShell 中：

```powershell
& 'C:\Users\lenovo\.conda\envs\cv-streaming\python.exe' -m streaming_recovery run
```

默认完整回放 episode 0、18，每 10 个原生帧判断一次（数据时间 2.5 Hz），输出 82 次判断。CPU 运行，不需要 GPU、模型权重或 API 凭据；每次新建输出目录，不覆盖旧记录。

复现本次五条完整回放：

```powershell
& 'C:\Users\lenovo\.conda\envs\cv-streaming\python.exe' -m streaming_recovery run --config streaming_recovery/artifacts/incremental_20261017/release_config.json
```

功能验收与合成逻辑回归：

```powershell
& 'C:\Users\lenovo\.conda\envs\cv-streaming\python.exe' -m streaming_recovery validate
& 'C:\Users\lenovo\.conda\envs\cv-streaming\python.exe' -m unittest discover -s streaming_recovery/tests -v
```

依赖版本见 [requirements.txt](requirements.txt)。本轮沿用 `cv-streaming`，没有重装环境。迁移时使用 Python 3.11，并修改 `config.json` 中的数据位置；配置相对路径以配置文件所在目录为基准。

## 可核查结果

- [本轮交付与修复说明](docs/增量修复与验收_20261017.md)：问题、补丁、验证等级和未完成项。
- [真实正常/失败查看页](artifacts/incremental_20261017/delivery_run/index.html)：每个时刻的双视角、阶段、依据、事件与两层建议。
- [冻结后新增 episode 6](artifacts/incremental_20261017/unseen_v2_run/index.html)。
- [功能验收](artifacts/incremental_20261017/delivery_validation/validation.json)、[补充验证与暴露记录](artifacts/incremental_20261017/additional_validation.json)。
- [修改前失败回归](artifacts/incremental_20261017/before/regressions.txt)、[修改后 29 项回归](artifacts/incremental_20261017/delivery_validation/unit_tests.txt)。
- [真实输出示例](artifacts/incremental_20261017/real_examples.json)、[合成恢复推进序列](artifacts/incremental_20261017/synthetic_recovery_trace.jsonl)。
- [关键帧对照](artifacts/incremental_20261017/真实运行关键帧.png)、[首次冻结运行，含 episode 19 漏检](artifacts/incremental_20261017/additional_run/index.html)。

每个新运行保存配置、源码副本、代码与输入文件 SHA256、完整 JSONL 和结束状态。根目录 Qwen 路线及其 500 条基准是另一条路线，本轮未运行或混用其指标。

## 数据、任务与模块

只绑定规格 `right_knife_cut_sausage_return_v1`：右臂取刀、携刀到砧板、产生本次新小段、刀回架、释放。支持规范英文指令及一个明确中文表述，允许大小写、空白与末尾句号规范化；否定、不同对象/顺序或其他任务明确拒绝，不用关键词凑流程。

`source.py → contracts.py → memory.py → multiview.py → tracking.py → events.py → recovery.py → engine.py` 构成完整链路；`__main__.py` 是入口；`report.py` 仅离线读取输出生成查看页。旧 `geometry.py` 后端保留，可把配置 `backend` 设为 `calibrated_rgb_geometry` 运行其受限对照；它不具备新双视角后端的切割/归位能力。

数据为官方固定版本 LeRobot v2.1、25 FPS、640×480 AV1，推理图像长边缩至 512。保留源变量名，只读取 `timestamp`、`frame_index`、`observation.state`；指令由控制面查任务表。状态第 13 项是右夹爪原始值，单位未被解释为米或力。`action`、目标 EEF、最终标签、样本目录名称与总长度不进入推理。

顶视图用于刀架刃区、砧板和香肠布局；腕部图像实际参与刀刃与两指连接、刀刃与香肠端部的相对运动判断，贡献与引用写入 `measurements.wrist`。开闭值来自同一原始状态，`gripper_source` 明确标识它不是独立视觉互证。

## 方法边界与记忆

这是固定相机、棕色砧板、黑刀架、白柄刀、橙色香肠的经典视觉方法，没有训练或通用识别保证。所有参考只来自已到达图像。首帧无效时允许在起始 3 s 内等待可校准观测；记录实际参考时间，不倒填历史；超过该范围明确要求从可验证初始状态重启。颜色、遮挡和镜头变化可造成未知或误识别。

历史容量同时约束：观测 4 条/4 s、图像至多 8 张、特征 6 条/4 s、任务锚点 8 条、事件 4 个且每事件 4 条证据。双视角后端额外保留 1 张参考、4 个掩码、2 个区域运动点、3 条腕部几何记录、最多 5 个初始食物几何条目。恢复上下文至多 1 个事件和 3 个状态槽。特征扩展字典单条最多 9,000 字符，整体 payload 也受检查；没有无限对话、模型 KV 或日志回读。详细容量见交付说明和每个输出的 `memory_usage`。

同步背压只保证因果次序，不承诺真机墙钟实时控制。两后端均通过完整输入链公共前缀对照；对齐只取已到达且未过期的样本，默认 0.12 s，无未来插值。流结束不触发成功。

## 状态与纠错

历史 `ever_held`、`completed` 与当前 `held`、`returned` 分开；每次实际在目标附近开→闭才建立新尝试。短暂未知不抹去最近 4 s 的夹持证据，也不凭未知宣称失抓。切割确认需要切割前参考、持刀操作与新增分离效果的时间关系；初始已有小块不能完成本次任务。完成后继续输入会更新当前谓词，同时保留历史完成事实。

事件经过疑似、确认、持续与结束；确认需不同观测、至少 0.8 s 连续支持，同一持续事件不逐帧提醒。阈值为工程门槛，不是校准概率。

高低层共享事件、对象和恢复子目标，失抓后的支撑位置会改变恢复策略。下一动作由当前可用、可见、无冲突、未过期的证据门控，不能靠历史失败建议开爪。二维对齐之后必须另有新鲜独立深度/姿态/接触准备，才允许生成闭合建议；闭合之后仍等待抓取效果。反复缺证据会具体列出补观测要求并交接操作者，不会假定建议已执行。

所有建议 `directly_executable=false`、`actually_executed=false`、`execution_receipt=null`。图像方向不映射为机器人基座方向；不开具无依据的位移、旋转、速度或夹持力数值。

旧版说明及第一次交付记录保留在 [修改前快照](artifacts/incremental_20261017/before/README.md) 和 [历史交付文档](docs/验证与交付.md)。

