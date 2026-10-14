# π0.5 / Wuji 训练与评测代码

PRO5000 项目的代码、配置与审计报告快照，供检查 52 维双臂灵巧手适配和七任务训练/闭环评测。

**北京时间快照：2026-10-14T11:21:12.213078+08:00。** 训练、模型、数据与视频保存在 PRO5000；本仓库用于代码审查，原脚本中的服务器绝对路径保持原样。

## 推荐检查顺序

| 顺序 | 入口 | 检查内容 |
|---|---|---|
| 1 | [当前七任务训练](projects/pi05_wuji/runs/pi05_seven_resume10e_b32_20261014_001/code/train_full_8gpu.py) · [配置](projects/pi05_wuji/runs/pi05_seven_resume10e_b32_20261014_001/config.json) | 数据读取、采样、flow loss、梯度、checkpoint 与优化器恢复 |
| 2 | [单轨迹训练](projects/pi05_wuji/runs/pi05_single_episode_pick_overfit_20261014_102705/code/train_full_8gpu.py) · [配置](projects/pi05_wuji/runs/pi05_single_episode_pick_overfit_20261014_102705/config.json) | 固定抓放训练 episode0，331 帧，追加1000更新 |
| 3 | [推理适配](projects/pi05_wuji/runs/pi05_single_episode_pick_overfit_20261014_102705/evaluation/code/adapter.py) | JPEG、224图像、state/token、52维归一化与反归一化 |
| 4 | [策略服务](projects/pi05_wuji/runs/pi05_single_episode_pick_overfit_20261014_102705/evaluation/code/policy_server.py) | 模型恢复、噪声采样、动作chunk与在线时间条件 |
| 5 | [仿真入口](projects/pi05_wuji/runs/pi05_single_episode_pick_overfit_20261014_102705/evaluation/code/run_eval.py) · [执行与任务验收](projects/pi05_wuji/runs/pi05_single_episode_pick_overfit_20261014_102705/evaluation/code/eval_environment.py) | 初始状态、关节控制、任务/接触判定 |
| 6 | [几何验收](projects/pi05_wuji/runs/pi05_single_episode_pick_overfit_20261014_102705/evaluation/code/audit_geometry.py) | 逐记录帧网格相交检查 |
| 7 | [失败分析](projects/pi05_wuji/runs/pi05_single_episode_pick_overfit_20261014_102705/evaluation/REPORT.md) · [实验前后对比](projects/pi05_wuji/runs/pi05_single_episode_pick_overfit_20261014_102705/COMPARISON.json) | 模型预测误差与闭环偏差 |

## 这次单轨迹诊断的结果

从共享七任务模型第37000步恢复全部模型和AdamW状态，只用一条已验收抓放示范追加1000更新，global batch32、恒定学习率1e-6、52维绝对关节目标、action horizon30。其他输入/输出约定保持一致。

- 固定训练集 flow loss 从0.00138369降至0.00130461，下降约5.7%。
- 同一训练初始场景闭环复测0/1成功：450步内没有物体接触。
- 使用示范动作的同场景对照成功，任务、接触、逐记录帧几何和视频检查通过；这是控制对照，不能计为模型成功。
- 该实验没有留出验证集；fit_checks是训练集检查，不能用于泛化或正式七任务成功率结论。
- 这些结果尚未定位到具体代码bug，也不能据此认定π0.5架构不能完成任务。

## 数据与动作约定

七任务为 gesture_one、gesture_two、microwave_open、piano、pick_place、produce_sort、stack。每任务50条已验收窄分布仿真示范；原七任务训练按完整episode分为315训练、35验证。

state/action为52维，左26维在前、右26维在后，每侧6臂+20手；action是弧度制绝对关节目标。原始HDF5有T动作和T+1观测，转换后的JPEG、状态、动作与source_frame对齐。以split_manifest.json为实际划分，meta/info.json的splits.train容器字段不代表全部50条进入训练。两类手势的hold修复只在训练读取时裁剪，原始数据未覆盖。

## 官方实现与项目代码

- [实际使用的OpenPI源码](projects/pi05_wuji/openpi/)：commit `215abfb217dbac7d5f1273282331b9b1866c0479`。来源：[Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)。
- [官方模型与采样](projects/pi05_wuji/openpi/src/openpi/models/pi0.py) · [配置](projects/pi05_wuji/openpi/src/openpi/models/pi0_config.py)。
- [最初预训练加载代码](projects/pi05_wuji/runs/pi05_full52_en_b8_20261013_001/code/train_full.py) · [加载审计](projects/pi05_wuji/runs/pi05_full52_en_b8_20261013_001/weight_load_audit.json)：视觉、语言、动作专家来自pi05_base；52维动作输入/输出投影最初重新初始化。
- [采集与数据导出](projects/dexterity_living/code/mcl350/) · [物理专家控制器](projects/dexterity_living/code/physical_v11/)。
- [任务控制器配置](projects/dexterity_living/configs/mcl350_collection.json) · [关节运动学](projects/dexterity_living/code/physical_v10/model.py)。
- Being-H参考对照位于[代码审计目录](projects/pi05_wuji/runs/pi05_code_audit_20261013_211237/)；这是源码比较，没有在Wuji上运行Being-H实验。

## 文件范围与快照说明

包含源码、配置、归一化统计、数据元信息/划分和小型结果报告。模型权重、原始示范样本、视频、图像、机器人网格/URDF资产、Python环境和凭据不在本仓库。因此下载源码后还需服务器运行环境与外部资产才能运行完整实验。

[详细目录说明](docs/LOCAL_SNAPSHOT_GUIDE.md) · [状态快照](SNAPSHOT_STATE.json) · [OpenPI版本信息](OPENPI_SOURCE.json) · [OpenPI工作目录差异](OPENPI_WORKING_TREE.diff)。

SOURCE_SNAPSHOT_MANIFEST.json记录下载时的原始代码包；PUBLICATION_SCOPE.json说明发布时省略的文件；PUBLISH_MANIFEST.json记录本仓库发布文件的SHA256。历史文档中的旧32维/LoRA计划、旧PID和旧进度应结合对应运行目录读取。实时状态以PRO5000的JSON与实际进程为准。

## 第三方来源

第三方源码保留各自的版权与许可证；参见[来源说明](THIRD_PARTY_NOTICES.md)。本次上传不为项目自有代码另行指定许可证。
