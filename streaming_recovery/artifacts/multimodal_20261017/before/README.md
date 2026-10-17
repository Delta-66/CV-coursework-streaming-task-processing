# CV大作业—流式任务处理

目前有两条独立路线，不能混用运行次数或效果结论。2026-10-17 对现有 `streaming_recovery/` 完成增量修复：真实双视角图像支持右臂取刀、切下新小段和刀具归位；两条正常、三条失败轨迹完成了功能回放。真实刀柄/两指联合定位、接触准备与完整重试纠错仍有缺口，不能称为通用机器人操作系统。

| 路线 | 目标 1—3 当前证据 | 尚未验证 |
|---|---|---|
| `streaming_recovery`，CPU 双视角经典视觉＋有限状态 | 5 条真实轨迹、180 次判断；正常 18/19 有完整任务效果证据；失败 0/2/6 主动提醒；真实恢复建议可检查 | 可靠联合定位、实际深度/接触准备、自然滑落/重试和物理执行；完整建议推进仅合成验证 |
| 根目录 Qwen 当前观测/时序路线 | 保留历史输入、启动、资源与工程测试记录 | 500 条正式基准仍为 0 条完成；未获该路线完整诊断/分层纠错效果结论 |

本次主入口（项目根目录 PowerShell）：

```powershell
& 'C:\Users\lenovo\.conda\envs\cv-streaming\python.exe' -m streaming_recovery run
```

详见 [运行说明](streaming_recovery/README.md)、[增量修复与验收](streaming_recovery/docs/增量修复与验收_20261017.md) 和 [真实逐时刻查看页](streaming_recovery/artifacts/incremental_20261017/delivery_run/index.html)。episode 19 首次冻结检查漏检，修复后归入本轮开发；保留首轮输出，未把它作为未调参成绩。冻结划分和保留测试集均未改动。

以下保留根目录 Qwen 路线的历史进展与原入口。原“第2步尚未完成、第3步暂缓”的判断适用于该路线及当时版本，不代表上面的 CPU 增量实现。

历史状态：第1步的真实 ViFailback 流式输入和当前观测基线已完成工程验证。第2步尚未完成：已有有限历史监测原型，但此前9次判断只是启动检查，不能证明任务进度理解和失败诊断有效。改为完整轨迹的批量时序评测，覆盖率和效果应分别验收。

## 根目录 Qwen 路线：历史进展与入口

使用现有 `cv-baseline` 环境，无需重新安装依赖：

```bat
conda activate cv-baseline
python run_temporal_benchmark.py --download
```

VS Code 中按 `Ctrl+Shift+D`，选择“完整时序评测：500条轨迹（GPU，可续跑）”，再按 `Ctrl+F5`。本机解释器选择 `cv-baseline`。该入口按冻结清单运行全部500条检测轨迹，共128,611帧；每25帧判断一次并补上末帧，预计5,850次模型调用，没有9次总上限。按此前速度估计推理约19小时，另有下载和显存等待。逐轨迹按需下载cam_high视频与状态文件，总计约1.1 GB；显存不足暂停等待，保持GPU推理。

每条轨迹从头到尾顺序回放；模型每次最多读取两张历史图和当前图，通常约2秒历史，尚不保证跟上真实视频速度。完成轨迹保存检查点；恢复时跳过已核验的完整轨迹，未完成轨迹从头重跑，避免丢失历史与提醒状态。程序和参数冻结后再测试，不根据测试输出边运行边调参。

`python run_task_monitor.py` 仍保留为两个已阅开发样例的运行入口，现为35次判断；它不代表完整测试集。原9次GPU记录作为历史启动检查保留。

本轮程序检查：113项测试通过。2026-10-16已冻结完整评测计划，目录为 `output/temporal_benchmark/canonical_detection_v1/`。按照“约5 GiB显存、预计5小时内完成”的启动条件，**已停止此前等待显存的后台任务，当前仍为0/500条**。原配置在5 GiB分配上限下加入第二张图即OOM，原速度估算也超过5小时；省显存实验与正式入口分开记录。空闲时如何启动、检查资源和断点续跑，见 [空闲时运行完整测试](docs/空闲时运行完整测试.md)。进度用 `python run_temporal_benchmark.py --status` 查看。`evaluation.json` 区分中间结果和完整结果，运行结束自动重新汇总；`source_snapshot/` 保留与协议哈希一致的程序副本。

省显存资源实验已完成两个真实开发轨迹的35次判断，本进程专用显存采样峰值约4.89 GiB，平均10.90秒/次，按窗口比例估算全量纯推理约17.6小时，因此仍不满足5小时限制。这只证明已测样例可运行，未作为正式入口替换，也不是测试集效果成绩；详见 [资源检查记录](docs/temporal-resource-check.json)。

运行记录位于 `output/monitor_runs/`，`predictions.jsonl` 保留每次原始回答、证据和跟踪状态，`alerts.jsonl` 只保留实际发出的提醒。完整字段、规则、验证范围及限制见 [任务进度与失败诊断说明](docs/任务进度与失败诊断.md)。

2026-10-15 验证：72项测试通过，真实 GPU 9次调用均通过输出校验，源图像、状态与历史窗口复核一致。该样例未触发失败提醒，不能据此宣称诊断准确或漏报已解决；详见 [运行验证报告](docs/task-monitor-validation.json)。

2026-10-16 复审：真实运行仍只覆盖两个开发样例，未训练模型或完成独立效果测试。已冻结真实数据轨迹划分：训练预留3,754条、验证939条、已阅开发样例3条、测试保留506条，见 [划分摘要](docs/data-split-summary.json)。测试侧按官方benchmark实际引用保留，500条检测子集与额外6条分开标记；同一轨迹不按帧拆分。复现命令为 `python prepare_dataset_splits.py --download-benchmark`，不会执行训练或测试推理。第2步的子任务、效果一致性和采样门槛仍需完善，[复审说明](docs/任务进度与失败诊断.md)记录了具体问题；第3步暂缓实现。

## 当前观测基线

本机已有独立 `cv-baseline` 环境与下载的模型，可运行：

```bat
conda activate cv-baseline
python run_current_observation.py
```

默认使用GPU；显存不足时暂停等待，恢复后继续当前输入，不自动切换CPU。其他电脑的环境、模型下载及预算设置见 [当前观测基线说明](docs/当前观测基线.md)。模型权重不在仓库中。CLI保留原始响应、严格校验输出，不将不确定或格式错误结果当成成功/失败。

## 获取与安装

```bat
git clone https://github.com/Delta-66/CV-coursework-streaming-task-processing.git
cd CV-coursework-streaming-task-processing
conda env create -f environment.yml
conda activate cv-streaming
python -m pip install -r requirements-input.txt
```

如果本机已有 `cv-streaming`，直接激活即可。上述最小依赖支持输入回放和接口测试；完整 GPU 环境按 `环境说明.md` 和 `requirements-lock.txt` 重建。仓库不包含虚拟环境、下载的数据集或本机生成的运行结果。

下载本次验证所用的数据版本，再运行回放：

```bat
python download_vifailback.py --episodes 0 18 --revision 619791b94f61667333fe179d471051e0bca0e06f
python download_vifailback.py --episodes 0 --revision 619791b94f61667333fe179d471051e0bca0e06f --raw-root data/vifailback_raw --raw-revision 21df46c2585c52e5a62df5f4b946bae6cbc6fed7
```

网络需要代理时，显式添加适用于你电脑的 `--proxy` 参数；不需要修改其他项目环境。数据版本和SHA256清单位于 `docs/data-manifests/`。

## 真实数据输入（当前入口）

在 Anaconda Prompt 中：

```bat
conda activate cv-streaming
python stream_dataset.py --config configs/vifailback.yaml
python stream_dataset.py --config configs/vifailback_hdf5.yaml
python verify_inputs.py
```

前两条命令分别回放已下载的真实 LeRobot 和 HDF5 样例。第三条验证多相机同步、数值字段、深度保存与标签隔离。数据来源、字段含义、Python接口、记录文件和下载方法见 [ViFailback输入接口](docs/ViFailback输入接口.md)。

当前 LeRobot 样例为 episode 0 与18，共819帧；原始 HDF5 样例为同一 episode 0 的300帧，用于补充验证深度接口。所有输入记录保留源字段名。

结果保存到 `output/input_runs/`。`inputs.jsonl` 保存当前可用观测，`recorded_targets.jsonl` 保存动作目标，`offline_annotations/` 保存离线真值；它们不混用。本机运行会生成 `output/input_validation.json`；仓库保存的验证摘要见 [真实输入验证记录](docs/input-validation.json)。

需要重建环境时，按 `环境说明.md` 安装锁定依赖。输入读取新增 PyArrow/AV1 解码依赖，仅装入 `cv-streaming`。

## 原来的合成视频练习

```bat
python create_demo.py
python replay.py
```

原入口仍可使用：6秒合成画面仅用于理解回放与有限历史，不含真实机器人操作或失败标注。其日志中的诊断字段仍为占位值。

## 合成练习的输出

运行结果保存到 `output/runs/` 下新建的时间目录：

| 文件 | 内容 |
| --- | --- |
| `events.jsonl` | 每次观测的时间戳、历史帧编号、预留输出字段与处理耗时 |
| `frames/` | 本次处理的采样画面 |
| `summary.json` | 总帧数、观测次数、历史图像存储量与总耗时 |
| `config.json` | 本次运行配置，便于复现 |

`failure: null`、`alert: null` 表示尚未接入模型，不代表没有失败；当前不能用于评价诊断效果。耗时包含占位接口和图像保存，不代表模型推理延迟。存储统计只计历史图像数组，不计解码器缓冲、Python对象、当前帧或模型内存。

你先看 `configs/demo.yaml` 的3个配置：`video_path` 指向视频；`sample_every` 控制采样间隔；`history_size` 控制历史容量。改成0即只提供当前帧，改成4即最多提供4张过去的采样帧，当前帧另计。

现有时间戳通过帧编号除以FPS计算，只适合固定帧率视频。真实轨迹应优先使用数据集时间戳，尤其是变帧率视频或多模态同步数据。

## 后续工作

1. 按需增加真实轨迹，核对新任务的字段、时基和缺失模态。
2. 人工核对当前观测基线，补充小规模评测集的阶段/证据与首次可检测时刻标注。
3. 对已有有限历史与提醒策略进行校准。当前观测只用图像，新方法还使用状态向量；开展消融实验时需固定模态、分辨率、模型和预算，不能直接将差异全部归因于历史。

模型和历史帧的输入必须遵守时间顺序：时刻t仅能使用时间不晚于t的观测。失败关键帧标注需要核对，不能直接当成首次可检测时刻。

本机8GB显存使用已下载的 Qwen3-VL-2B-Instruct。官方 [ViFailback 仓库](https://github.com/x1nyuzhou/ViFailback) 的模型复现与本项目的小模型监测实现分别配置；需要复现官方模型时另建匹配环境。

完整环境使用与重建方法见 `环境说明.md`。既有 `base` 和 `d2l` 环境未修改。

## 项目资料

- [课题要求](大作业课题/机器人操作的流式任务理解、失败诊断与分层纠错.pdf)
- [10周进度安排](docs/plans/流式任务_10周进度安排.md)
- [前四周学习与协作安排](docs/plans/流式任务_三人写作与时间安排.md)

接口测试无需下载真实数据：`python -m unittest discover -s tests -v`。`verify_inputs.py` 检查已下载的真实样例，需先按上面的命令完成两种格式的回放。
