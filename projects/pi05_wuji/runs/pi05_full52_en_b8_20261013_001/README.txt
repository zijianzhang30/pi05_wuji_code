当前运行：八卡 GPU0-7，FSDP8，总batch8，每卡1；从第3000步checkpoint恢复全部参数和优化器状态。入口 code/train_full_8gpu.py，启动器 launch_8gpu.py，日志 train_8gpu.log。原四卡记录保留在下方；当前状态以status.json为准。

π0.5 / Wuji 七任务联合全量微调，2026-10-13。
训练入口 code/train_full.py；可重入启动器 launch.py 检查本任务是否仍运行，再从已有checkpoint恢复。
全局batch=8，四卡FSDP，各卡2样本，无梯度累积。模型52维绝对关节目标、30步动作块、单EGO、英文指令。
train_frames=111310；10000步=80000个采样起点=0.718714等效帧epoch。不是不重复遍历；按任务/轨迹/起点三级均衡随机采样。
每个采样起点监督最长30步动作块；边界补齐不计入损失。不能用chunk长度乘出的目标帧曝光数冒充epoch。
模型全部参数可训练；低学习率1e-5、500步warmup；EMA关闭以节省显存；无LoRA。
max_token_len=320；全部训练/验证观测预先验证没有截断。
原数据、拆分及中文采集证据均不修改；英文prompt快照见prompts.json。
参数加载：保留所有兼容预训练权重，重新初始化52维动作输入/输出投影层，审计见weight_load_audit.json。
状态 status.json；逐步指标 metrics.jsonl；验证 validation/；checkpoint checkpoints/。
先保存第20步checkpoint并执行恢复核对，第200步再次保存，此后每1000步保存。
验证指标是固定留出样本的flow-matching损失，不是闭环成功率。任务成功率需另行冻结权重后在统一物理评测器中统计。
整个进程持有GPU0-3现有共享锁；GPU4-7未分配给本任务。

启动兼容性修复：原环境 NCCL 2.26.2 四卡大张量 all-reduce 复现非法内存访问；本任务独立预加载 runtime_overrides/nccl_2289 中的 NCCL 2.28.9 后测试通过，且完成训练参数更新。未修改原 ACT 环境。
原失败日志 attempt2_train.log / attempt3_train.log；对照测试 probe_nccl2289.log；运行库校验 runtime_compatibility.json。

八卡迁移审计：migration_8gpu.json、resume_8gpu.json、full_gradient_check_8gpu.json。原四卡最后记录3210步，未落盘部分从3000步重新计算；metrics.jsonl有重算步数，按migration中的分段行数区分，八卡独立日志无此歧义。
