# Wuji MCL 第一轮：7 × 50 有效 episode

用户确定七个任务各采集 50 条，总计 350 条。数量是本轮实验预算，不是 RoboDojo 官方统一标准。

## 机器与目录

目标 PRO5000：`ssh -p 1025 bowenjing@8.130.80.101`。

当前 MacBook 直连超时，已经验证的 Mac Studio 中转：

```sh
ssh -i /Users/jingbowen/.ssh/id_ed25519_macstudio arthur12137@100.103.74.92 \
  'ssh -p 1025 bowenjing@8.130.80.101'
```

项目：`/media/external/bowenjing/projects/dexterity_living`

- 代码：`code/mcl350/`
- 参数：`configs/mcl350_collection.json`
- 合格数据：`datasets/wuji_mcl350_v1/<task>/episode_<seed>/`
- 总进度：`reports/mcl350/status.json`
- 数据清单：`datasets/wuji_mcl350_v1/manifest.json`
- 原始候选：`runs/physical_v1/mcl350_v1_*`
- 每条日志、失败原因：`reports/mcl350/`

有效数量以每个 episode 的 `acceptance.json` 为准；原始候选数、渲染帧数和 MP4 数都不代表通过数量。失败不计数，保留证据。每任务最多 100 次新尝试，连续 5 次失败暂停该任务供排查，不降低验收门槛凑 50 条。

## 任务与范围

| task | 任务 | 目标 |
|---|---|---:|
| pick_place | 单手抓放方块到左侧圆形区域、松手 | 50 |
| stack | 方块堆叠、松手稳定 | 50 |
| piano | 钢琴固定八音序列 | 50 |
| gesture_one | 手势一 | 50 |
| gesture_two | 手势二 | 50 |
| produce_sort | 橙子和甜椒分类入对应碗 | 50 |
| microwave_open | 左手四指抓把手、解锁并打开原生弹簧微波炉 | 50 |

沿用固定客厅、黑色 Piper＋Wuji、白桌、原 EGO 机位。当前接触控制器使用 PhysX CPU；GPU 用于 Isaac 应用和 RTX 图像采集。八张 GPU 分配独立 episode，不是 IsaacLab 向量化环境。

微波炉范围：四指共同受力用于抓持和解锁，后段含门的被动惯性运动；没有声称四指全程受力。手势为既有动作，不是规范手语认证。

## 数据与对齐

每条导出 `episode.hdf5`、完整 `ego.mp4`、验收、动力学、几何、接触、来源报告。

- EGO RGB：640×480、30 Hz、JPEG quality 95 原始帧逐帧存入 HDF5；另有预览 MP4。
- 状态：双侧共 52 个关节位置、实际速度；左右手掌世界位姿。
- 动作：实际发送的绝对关节目标，52 维；保留每个 480/960 Hz 子步目标、重力前馈及 PD 增益。
- 保存任务指令、随机种子、关节名称和顺序、相机内外参、初始化变化、控制器与代码哈希。
- T 个动作对应 T+1 个观测。`obs[t]` 在 `action[t]` 前，`obs[T]` 是终止状态，无伪造终止动作。
- 果蔬有 360 帧预沉降，随后各任务有 30 帧初始稳定；这部分留在原始数据，但从训练片段和视频排除。
- 原脚本的接触锁定可能在最后一个子步后更新下次目标；导出动作保留实际已发送命令，不能拿事后更新值覆盖。

HDF5 主要字段：

```text
observations/qpos                 [T+1,52]
observations/qvel                 [T+1,52]
observations/palm_pose_world_xyzw  [T+1,2,7]
observations/images/ego           [T+1] JPEG bytes, RGB
observations/time                 [T+1]
action                            [T,52]
action_substeps                   [T,16 or 32,52]
gravity_feedforward_substeps      [T,16 or 32,52]
controller/gains_kp_kd             [T,2,52]
metadata/joint_names
camera/intrinsics
camera/camera_to_world
source_frame
terminated                        [T]
success                           scalar
```

## 多样性与验收

首轮是窄分布最小闭环：物体 XY ±3 mm、活动臂初始关节 ±0.012 rad、手指 ±0.018 rad；手势抬手位置 ±8 mm，钢琴各移动/按键/松开段时长 ±0.025 s，微波炉接近时长 1.76–1.84 s。不改变家具、基座、相机、资产尺寸或摩擦。每条独立重新仿真，不复制旧视频；重复状态/动作轨迹拒收。

验收包括原动力学成功条件、物体无位姿命令/无吸附绑定、接触前漂移、任务终态与松手、钢琴音序、四指受力条件、逐记录帧可见网格碰撞审计、图像全帧解码、T/T+1 对齐及真实控制命令。几何审计覆盖记录帧，排除机械相邻关节接口，不能解释成连续时间的数学零穿透证明。

原始仿真过程的接触报告和图像渲染分别保留证据；离线渲染自身不产生物理验收结论。

参考 RoboDojo 的按任务组织、随机种子和观测/动作同步方式，当前不是官方格式认证或人工遥操数据。官方资料：https://github.com/RoboDojo-Benchmark/RoboDojo 。π0.5 训练、在线策略执行和成功率评测尚未包含在本次采集交付中。

## 启动或续跑

先确认 `reports/mcl350/status.json` 和进程，避免重复启动。协调器有单实例锁和逐 GPU 锁，已通过的数据不会覆盖。

```sh
cd /media/external/bowenjing/projects/dexterity_living
source scripts/pathtrace_env.sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/bowenjing/envs/isaaclab3/bin/python code/mcl350/collect_dataset.py
```

读取 `reports/mcl350/status.json` 才能确认实时接受数量。本地本说明不代表 350 条已经完成。
