> 本文件说明原始Mac下载快照；GitHub发布时省略了URDF机器人资产和本地下载校验回执，参见根目录PUBLICATION_SCOPE.json。仓库入口以README.md为准。

# π0.5 / Wuji 本地代码检查快照

北京时间快照：2026-10-14T11:21:12.213078+08:00
远端项目：/media/external/bowenjing/projects/pi05_wuji
OpenPI 当前提交：215abfb217dbac7d5f1273282331b9b1866c0479

## 先看这几个入口

| 用途 | 本地相对路径 |
|---|---|
| 当前七任务训练入口 | projects/pi05_wuji/runs/pi05_seven_resume10e_b32_20261014_001/code/train_full_8gpu.py |
| 当前训练实际配置 | projects/pi05_wuji/runs/pi05_seven_resume10e_b32_20261014_001/config.json |
| 单示范训练入口 | projects/pi05_wuji/runs/pi05_single_episode_pick_overfit_20261014_102705/code/train_full_8gpu.py |
| 单示范实验配置与对比结果 | projects/pi05_wuji/runs/pi05_single_episode_pick_overfit_20261014_102705/config.json、COMPARISON.json |
| 单示范训练及评测自动驱动 | projects/pi05_wuji/runs/pi05_single_episode_pick_overfit_20261014_102705/driver.py |
| 数据字段、JPEG、52维归一化与时间对齐 | 上述训练脚本的 load_episode、Dataset.normalize、Dataset.sample、Dataset.batch |
| 推理预处理与动作反归一化 | projects/pi05_wuji/runs/pi05_single_episode_pick_overfit_20261014_102705/evaluation/code/adapter.py |
| 模型加载和流匹配采样 | projects/pi05_wuji/runs/pi05_single_episode_pick_overfit_20261014_102705/evaluation/code/policy_server.py |
| 仿真初始化、关节执行、任务接触验收 | projects/pi05_wuji/runs/pi05_single_episode_pick_overfit_20261014_102705/evaluation/code/run_eval.py、eval_environment.py |
| 几何验收 | projects/pi05_wuji/runs/pi05_single_episode_pick_overfit_20261014_102705/evaluation/code/audit_geometry.py |
| 固定训练图像的预测误差诊断 | projects/pi05_wuji/runs/pi05_single_episode_pick_overfit_20261014_102705/evaluation/offline_teacher.py |
| 单示范闭环失败分析 | projects/pi05_wuji/runs/pi05_single_episode_pick_overfit_20261014_102705/evaluation/REPORT.md、ANALYSIS.json |
| 最初官方预训练权重加载与52维投影初始化 | projects/pi05_wuji/runs/pi05_full52_en_b8_20261013_001/code/train_full.py、weight_load_audit.json |
| 官方模型、flow loss和采样逻辑 | projects/pi05_wuji/openpi/src/openpi/models/pi0.py、pi0_config.py |
| 官方 LeRobot 数据接入与训练配置 | projects/pi05_wuji/openpi/src/openpi/training/data_loader.py、config.py |
| 原始采集、字段导出与图像渲染 | projects/dexterity_living/code/mcl350/ |
| 七任务物理专家控制器 | projects/dexterity_living/code/physical_v11/，具体入口见 configs/mcl350_collection.json |
| 关节定义与运动学 | projects/dexterity_living/code/physical_v10/model.py；sources/20261011/robots/wuji/assembled.urdf |
| 数据容器说明与实际episode划分 | projects/act_wuji/datasets_v21/<task>/meta/info.json、split_manifest.json |

## 快照范围与使用方式

这是实际运行代码、配置、归一化统计和小型审计/结果报告的副本。源码保留服务器目录结构，未把绝对服务器路径改成本地路径。
OpenPI 文件取自远端工作目录；OPENPI_SOURCE.json 和 OPENPI_WORKING_TREE.diff 记录提交及修改。
未包含模型/checkpoint、Parquet/HDF5示范样本、NPZ数组、视频、图像、网格资产、Python环境和凭据，因此本地用途为阅读、审查与编辑，不能直接在Mac运行原GPU训练/Isaac仿真。
在本地修改文件不会自动修改远端训练。要部署改动应另建远端实验版本，保留本次快照与历史结果。
SNAPSHOT_STATE.json 是本次抓取时的状态；实时状态以远端JSON和实际进程为准。
DEPLOYMENT_PLAN.md、NEXT_STEPS.md 等历史文档可能包含旧32维/LoRA方案或旧PID，审查当前配置时以对应运行目录的代码与config.json为准。
单示范实验没有留出验证集；fit_checks 是训练集检查，不能当作泛化结果。
manifest.json 记录每个快照文件的SHA256，下载后将逐个核验。
