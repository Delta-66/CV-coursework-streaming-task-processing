# ViFailback 数据要求与输入接口

本文仅描述独立的数据读取、按时间回放、数据校验与记录模块。模型与判断使用单独入口，见 [当前观测基线](当前观测基线.md)。

## 已查询的数据版本

- 官方原始数据：[ViFailback-Dataset](https://huggingface.co/datasets/sii-rhos-ai/ViFailback-Dataset)，本地样例版本固定为 `21df46c2585c52e5a62df5f4b946bae6cbc6fed7`。
- 官方转换：[vifailback-dataset-lerobot](https://huggingface.co/datasets/sii-rhos-ai/vifailback-dataset-lerobot)，本地版本固定为 `619791b94f61667333fe179d471051e0bca0e06f`，格式为 **LeRobot v2.1**。

转换版声明25 FPS、3路640×480 RGB，视频编码AV1，状态/动作存储为Parquet，任务指令由 `task_index` 关联 `meta/tasks.jsonl`。因此读取需要Parquet和AV1解码支持。本项目使用已验证的 PyArrow 和 PyAV，未安装训练或机器人控制SDK；不要求GPU，也不需要为了读取数据另装官方8B模型环境。

当前下载了同一个任务的episode 0和18（300和519帧），分别对应官方失败、成功轨迹，另下载episode 0原始HDF5验证深度。没有下载全部数据集。文件校验值与仓库版本保存在各数据目录的 `download_manifest.json` 中。

## 字段对应关系

源字段保持原名；下表只说明两个官方存储格式之间的对应关系，程序不会把源字段改名。

| 记录 | LeRobot v2.1字段 | 原始HDF5字段 | 接口处理 |
| --- | --- | --- | --- |
| 任务指令 | `task_index` + `meta/tasks.jsonl` | 外部任务映射/显式配置 | 作为任务输入 |
| 当前关节位置 | `observation.state`，14维 | `observations/qpos` | 当前观测，包含夹爪值 |
| 关节速度 | `observation.velocity`，14维 | `observations/qvel` | 当前观测 |
| 关节effort | `observation.effort`，14维 | `observations/effort` | 当前观测，保留源单位 |
| 三路RGB | `observation.images.cam_high/cam_left_wrist/cam_right_wrist` | `observations/images/cam_high/cam_left_wrist/cam_right_wrist` | 同帧解码为HWC、RGB、uint8 |
| 深度 | 转换版未包含 | `observations/images_depth/<camera>` | 保留原生单通道数组，已验证uint16 |
| 关节目标 | `action`，14维 | `action` | 单独的记录目标接口 |
| EEF目标 | `complementary_info.action_eef`，16维 | `action_eef` | 单独的记录目标接口 |
| 底座目标 | `complementary_info.base_action`，2维 | `base_action` | 单独的记录目标接口 |
| 主臂记录 | 转换版未包含 | `action_leader`（若实际存在） | 可读为记录目标，本次原始样例缺失 |
| 时间与索引 | `timestamp/frame_index/episode_index/index/task_index` | 显式时间字段若存在，否则配置FPS | 保存原索引，注明时间来源 |
| 失败/关键帧标签 | `complementary_info.vifailback_flags` | 外部标注 | 单独离线接口，排除在线输入 |
| 诊断与纠错标注 | `meta/vifailback_annotations.jsonl` | 原始标注文件 | 保留完整离线标注，不作为观测 |

关节向量中，左臂为0:7，右臂为7:14；各臂前6项为关节、最后1项为夹爪。辅助的 `gripper_state` 引用原字段的索引6和13，不做归一化，也不修改原向量。

## 语义和时间约束

1. **EEF目标不等于当前EEF实测位姿。** 当前样例没有独立的当前EEF位姿字段；接口报告缺失，不自动用目标冒充观测。
2. **effort不等于六维末端力/力矩。** 当前样例没有已校准六维wrench；保留源关节effort，不假定单位或坐标系。
3. 官方将 `action` 描述为下一步关节目标。其在时刻t的在线可用性需要进一步确认，所有目标默认排除在 `online_input()` 外，但可从 `recorded_targets` 读取和审计。
4. 失败标志是整段轨迹标签，在失败轨迹的每帧上重复；不能把每一帧当作当时已经发生失败。avoid/correct关键帧也不能直接当成首次可检测时刻。
5. 原始样例没有传感器时间戳，配置显式采用25 FPS推算时间。转换版时间戳属于转换时基，不代表原始硬件时钟。接口保证源索引和转换时基下的顺序/对齐，不宣称已校准多个传感器的实际采集延迟。
6. 官方说明Dabai相机RGB与深度未保证空间对齐。接口独立读取两者，深度以 `.npy` 原样存储，不执行RGB-D融合、不转点云、不假定深度单位。
7. 原始样例的RGB/深度实际为压缩字节序列，解码依据数据形状/类型，而不是仅依赖文件 `compress` 属性。原始RGB按官方存储说明解释为RGB；PNG输出时才转换给OpenCV的BGR写入接口。

## Python接口

```python
from contextlib import closing
from dataset_inputs import LeRobotEpisode

source = LeRobotEpisode("data/vifailback_lerobot", 0)
print(source.describe())  # 字段、维度、数据类型、可用/缺失模态

with closing(source.iter_frames()) as frames:
    for packet in frames:
        inputs = packet.online_input()
        task_instruction = inputs["task_instruction"]
        timestamp = inputs["timestamp"]
        state = inputs["observations"]["observation.state"]
        image = inputs["images"]["observation.images.cam_high"]
        targets = packet.recorded_targets  # 审计用途；未混入在线观测

# 离线标签通过独立接口读取。
annotation = source.offline_annotations()
flags = source.iter_offline_flags()
```

读取原始HDF5：

```python
from dataset_inputs import HDF5Episode

source = HDF5Episode(
    "data/vifailback_raw/raw_data/cut_sausage/episode_0.hdf5",
    task_instruction="Use the right arm to pick up the knife from the knife holder, cut a small piece of sausage and put it back on the knife holder.",
    fps=25,
    include_depth=True,
)
```

有原始时间字段时优先使用；有多个候选时间字段时必须明确 `timestamp_path`。缺少时间字段且未显式指定FPS会报错。只按顺序输出当前记录，保持图像与状态属于同一帧，不排序、补帧或偷偷改变采样率。视频PTS与状态时间不一致、相机帧超过当前时间（除浮点舍入容差）、时间倒退、字段长度/维度不一致、未知任务索引、坏图像或缺少指定相机，都会报错。

## 运行与输出

```bat
conda activate cv-streaming
cd /d C:\Users\lenovo\Desktop\CV
python stream_dataset.py --config configs/vifailback.yaml
python stream_dataset.py --config configs/vifailback_hdf5.yaml
python verify_inputs.py
```

每次生成独立运行目录，默认记录每一帧；每25帧保存一次画面/深度预览。未保存数组的记录仍含源文件、字段和帧编号，可按引用读取，不把整段图像塞入JSON。

| 文件 | 内容 |
| --- | --- |
| `inputs.jsonl` | 在线输入的任务、时间、原字段状态、图像/深度引用 |
| `recorded_targets.jsonl` | 原字段动作目标，逐帧对应，注明在线可用性尚未确认 |
| `extra_records.jsonl` | 源中的额外逐帧记录（若存在） |
| `arrays/` | RGB PNG与保留原生dtype的深度NPY |
| `offline_annotations/` | 完整轨迹标注、离线帧标签 |
| `schema.json` | 字段类型、维度、文件属性、实际可用与缺失模态 |
| `config.json`、`provenance.json` | 本次配置、数据来源、版本和下载清单路径 |
| `summary.json` | 解码/记录数量、运行状态、模型调用0、基线未构建 |

`realtime: true` 会按源时间戳节奏回放，`playback_speed` 控制倍速；`false` 则尽快顺序回放。每条轨迹独立重置时间，不把不同轨迹的时间接成一条。

## 下载其他样例与复现

```bat
python download_vifailback.py --episodes 0 18
python download_vifailback.py --episodes 0 --raw-root data/vifailback_raw
```

网络需要代理时，可显式添加 `--proxy http://127.0.0.1:7897`；此参数只作用于本次下载，不修改系统代理。首次下载固定当前仓库提交，以后沿用清单中的提交。同一数据目录拒绝混入其他版本。大型文件对照官方SHA256验证，所有文件保存本地SHA256与大小；已有可验证文件不会重复下载。

LeRobot转换版将5,202条轨迹统一放入 `train`，没有直接给出流式监测评测划分；输入模块的0、18仅用于开发验证，不能当独立测试。官方原始仓库另有 [VQA训练和benchmark文件](https://huggingface.co/datasets/sii-rhos-ai/ViFailback-Dataset/tree/21df46c2585c52e5a62df5f4b946bae6cbc6fed7/VQA)，不能把“转换版未指定划分”理解为“官方没有基准集”。官方数据许可标为MIT，研究使用应引用原论文和数据来源。

2026-10-16 核对了官方 `ViFailback_VQA_benchmark.json` 的图像来源路径，仅用轨迹身份建立保留集合，不把其答案送入在线监测。文件固定为原始仓库上述提交，SHA256为 `211c7047577ee4327d232727bd92fddc6456383ed3751b4b940267b0a1058218`。5,710条VQA共引用506条原始轨迹，其中成功/失败检测条目涉及500条，另6条只出现在部分高层问题中。发布文件的实际引用并集应全部隔离，不能依据论文的500条描述将额外6条重新放入训练侧。

完整轨迹是划分的最小单位：同轨迹的相机、原始/转换格式、窗口、关键帧和VQA必须同组。元数据有109个原始任务目录，但只有100个不同任务指令，`ep1/ep2`目录不能充当独立任务。当前没有可靠的跨轨迹场景/采集session/配对ID，因此只能保证已知轨迹身份隔离，不能宣称消除了所有采集相关性。逐时刻阶段和首次可检测失败仍需要补充标注；官方VQA评测不直接等同于流式延迟评测。

## 验证范围

已完整验证episode 0与18的819帧三相机输入，以及episode 0原始HDF5的300帧三相机RGB/深度。六类数值字段在原始和转换版之间逐帧完全一致，RGB的AV1转换差异在验证报告中记录；RGB与uint16深度的保存/读回无损。离线标签与目标不会进入在线输入。

8项接口测试覆盖压缩图像、原生深度、显式/推算时间、同步、未来帧拒绝、错误长度、文件释放、标签隔离和记录读回。`output/input_validation.json` 保存真实样例验证结果。

当前验证覆盖所下载样例，不代表已遍历全部5,202条轨迹。新增数据仍需经过接口检查；格式或模态不匹配时明确报错，不填造缺失数据。
