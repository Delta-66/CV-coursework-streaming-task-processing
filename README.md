# CV大作业—流式任务处理

目前完成基础环境和真实 ViFailback 数据输入：支持官方 LeRobot v2.1 与原始 HDF5，按时间回放图像、机器人状态和任务指令，并分别记录动作目标和离线标注。按当前安排，基线构建暂不进行。

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

## 后续工作（基线暂缓）

1. 按需增加真实轨迹，核对新任务的字段、时基和缺失模态。
2. 当前只维护数据输入与记录；模型、判定标准、提醒策略、分层纠错和基线实验均留待后续。

模型和历史帧的输入必须遵守时间顺序：时刻t仅能使用时间不晚于t的观测。失败关键帧标注需要核对，不能直接当成首次可检测时刻。

官方 [ViFailback 仓库](https://github.com/x1nyuzhou/ViFailback) 推荐 Python 3.10，注明8B模型常规精度推理约需20–24GB显存。本机8GB显存先用于数据开发；具体模型后续选择较小模型、评估量化可行性，或使用服务器。复现官方模型时另建匹配环境。

完整环境使用与重建方法见 `环境说明.md`。既有 `base` 和 `d2l` 环境未修改。

## 项目资料

- [课题要求](大作业课题/机器人操作的流式任务理解、失败诊断与分层纠错.pdf)
- [10周进度安排](docs/plans/流式任务_10周进度安排.md)
- [前四周学习与协作安排](docs/plans/流式任务_三人写作与时间安排.md)

接口测试无需下载真实数据：`python -m unittest discover -s tests -v`。`verify_inputs.py` 检查已下载的真实样例，需先按上面的命令完成两种格式的回放。
