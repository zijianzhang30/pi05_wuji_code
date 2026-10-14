# PRO5000 上的 π0.5 + Wuji 最小闭环部署

日期：2026-10-12。状态：独立环境 CPU 导入与 JAX/Blackwell GPU 算子验证已通过；官方权重仍在下载，尚未完成模型前向、训练或自主闭环验证。

## 已核实的机器与隔离边界

- 目标：`ssh -p 1025 bowenjing@8.130.80.101`。本次通过 Mac Studio `arthur12137@100.103.74.92` 中转。
- 8 × NVIDIA RTX PRO 5000 72GB Blackwell，驱动 580.126.09；Ubuntu 24.04.3，约 1.4 TiB RAM。
- 主目录盘只剩约 24 GB；环境、下载缓存、权重、日志均放 `/media/external/bowenjing/projects/pi05_wuji`，这是 NAS，不是本地 SSD。
- 仿真环境 `/home/bowenjing/envs/isaaclab3` 保持独立。π0.5 不继承 Isaac 的 LD_PRELOAD / LD_LIBRARY_PATH / PYTHONPATH。
- 官方源码：`openpi/`，本次 SHA `215abfb217dbac7d5f1273282331b9b1866c0479`。
- 新 Python 3.11 环境：`env/`。锁定依赖安装：`bootstrap_environment.py`；状态：`environment_status.json`；日志：`bootstrap_environment.log`。
- 安装过程显式隐藏 GPU、使用 JAX CPU 后端。采集优先，GPU 测试须等采集完成并取得现有 GPU 锁。

## 推荐路线

官方 OpenPI，JAX π0.5 + LoRA，从 `gs://openpi-assets/checkpoints/pi05_base` 初始化。官方支持 LoRA 的实现目前是 JAX；不能把官方 PyTorch 分支称作已支持 LoRA。

先验证 1 GPU 的权重加载、前向和短训练。正式首轮建议 1 卡训练 + 1 卡 Isaac 闭环评估，资源需求与吞吐测完再增加训练卡；不是为了用满 8 卡而先引入分布式复杂度。官方给出的 LoRA 显存需求约 >22.5 GB、全参数 >70 GB，后者在本机单卡上余量过小；这些是参考值，不是我们这个配置的实测值。

当前官方锁定 JAX 0.5.3 / CUDA 12 依赖。驱动满足通用 CUDA 12/13 驱动要求，JAX 0.5.3 已在此 Blackwell 卡通过 BF16 GEMM、float32 卷积与 JIT 梯度实测；完整模型加载和前向仍待权重下载与校验完成。不能因为 nvidia-smi 显示 CUDA 13 就直接替换全部依赖。先保留官方锁文件；若 GPU 编译失败，再在独立环境升级并记录版本差异。

## 数据接入：先保持单手任务范围

原始数据：`/media/external/bowenjing/projects/dexterity_living/datasets/wuji_mcl350_v1`。

七任务各 50 条，共 350 条，采集已完成且 worker 已退出；验收报告与必要产物审计见 collection_gate.json。每条包含 EGO、52 维双臂双手状态、52 维实际关节目标、30 Hz 时间戳、子步命令、接触与几何验收、language instruction。它们是脚本反馈专家生成的窄分布示范，不是人工遥操作数据。

推荐首版：所有七任务共享一个条件策略，当前操作侧 26 维（6 臂 + 20 手）补齐到官方 32 维；微波炉使用左侧，其余六任务使用右侧。由任务配置明确路由至对应关节，另一侧保持固定停放姿态。完整 52 维原始数据保留，不做覆盖或无条件截断。此方案仅适合当前单手任务；未来双手协同需重新设计 >32 维接口、权重加载与训练。

关节顺序必须使用 HDF5 `metadata/joint_names`；当前左侧为 0:26，右侧为 26:52。手指关节按关节级别分组，不是按手指连续排列。动作是弧度制绝对关节目标，不套用 ALOHA 夹爪二值映射或 LIBERO 末端增量变换。

训练输入：EGO RGB + 当前操作侧关节位置 + 明确的语言指令。缺失腕部相机使用官方图像 mask，不能复制 EGO 冒充腕部视角。原始图像 640×480；官方模型前处理至 224×224，需要检查小物体、指尖与把手仍可辨认。

HDF5 转官方所锁定版本的 LeRobot，按 `obs[t] -> action[t] -> obs[t+1]` 对齐。保留终态作为评估记录，不伪造最后一个动作。只导入已原子发布 acceptance 的 episode。

每任务建议 45 条训练 + 5 条验证，以完整 episode/seed 分组，不能随机切帧造成泄漏。归一化仅使用训练集。350 条仅用于首轮闭环，不代表跨场景泛化已经足够。

## Language prompt

原始 instruction 已在七任务样本 HDF5 中核实。训练派生版本需要保留原文并补足指代，同时可增加准确英文对照：

- 抓放：右手抓方块，放至左侧目标区域并松手；先确认视觉目标标记。
- 堆叠：右手把指定方块放到另一个方块上并松手。
- 钢琴：明确八次按键序列；日志 key indices 为 [5, 5, 6, 6, 7, 7, 6, 4]，未核实音高前不杜撰音名。
- 手势一/二：明确右手、对应数字和保持动作。
- 果蔬：右手把橙子放入黄色碗，把甜椒放入绿色碗并松手。
- 微波炉：左手四指勾住把手并拉开门。

语言版本更改要写入派生数据 manifest，不修改原始采集证据。仅有不同 prompt 不足以证明语言理解；固定场景与任务可能相关，后续需要同场景不同指令评估。

## 真正闭环与验收

模型动作进入同一有限力矩 PD / 重力补偿控制器，保持原始摩擦、碰撞、480/960 Hz 物理子步及 30 Hz 控制语义。微波炉等任务原控制器存在增益调度，应明确重现或重新验证，不能只取视频轨迹。

先测推理延迟，初始考虑每执行 3–5 个控制步重新观察；action horizon 与实际执行长度区分。不要把整段 action chunk 一次播完当作闭环，也不要宣称尚未实测的 30 Hz 模型推理速度。

从独立 reset 开始：EGO/关节状态 -> 模型 -> 关节目标 -> 真实仿真反馈。策略评估禁用专家路径/物体目标位姿驱动；接触日志用于验收，不能偷传目标真值到策略。记录任务成功、接触前漂移、穿模、掉落、动作速度/加速度突变、终态和推理延迟。动作平滑处理若改变命令，必须重新物理验收。

当前数据门槛包含物理接触与记录帧网格检查；尚未有完整数据的独立丝滑度验收，也不保证连续时间数学意义的零穿透。微波炉目前为四指勾握解锁后门体惯性继续运动，不能描述成全程四指持续接触。

部署里程碑分别记录：依赖安装；CPU 导入；GPU 算子；权重完整性与加载；真实样本前向；短训练；独立 reset 自主任务成功。前几项不等于最终闭环。

## 官方依据

- https://github.com/Physical-Intelligence/openpi
- https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0_config.py
- https://docs.jax.dev/en/latest/installation.html
