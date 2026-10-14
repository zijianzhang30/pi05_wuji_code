> Current acceptance update 2026-10-11: microwave v29 gpu2 now passes full opening plus recorded-frame geometry checks. Seven presentation demos are available (five contact tasks and two gestures). See MICROWAVE_V29.md and reports/accepted_microwave_v29.json. Earlier microwave failure notes below are historical.

> Latest checkpoint: [CONTACT_SUITE_STATUS.md](CONTACT_SUITE_STATUS.md). Older revision details below are retained for provenance.

# 当前单手抓放与数据状态 — 2026-10-11

当前用户要求：单手抓放，目标画圈并向左移动，动作提速。双手交接已取消，不自动重启交接管线。

主机：ssh -p 1025 bowenjing@8.130.80.101
根目录：/media/external/bowenjing/projects/dexterity_living
Python：/home/bowenjing/envs/isaaclab3/bin/python
环境：先 source scripts/pathtrace_env.sh，OPENBLAS_NUM_THREADS=1。

## 最新单手版本

运行数据：runs/physical_v1/pick_left_fast_gpu0/
代码入口：code/physical_v10/run_pick_left_fast.py
参数：--candidate 1 --contact-stop .2 --pinch 0 --open-gap .012 --hold-force 3 --move-seconds 3 --motion-scale .60
动作：右手抓起方块、向左移、放下松手。左手停放。
目标圈中心 XY = [-2.17,-3.505] m，外径 130 mm、内径 116 mm。比旧目标向左移动160 mm。
目标圈仅绘制材质和几何，无碰撞，不改摩擦。
完整物理流程702帧/30Hz=23.4秒，旧左移版1109帧=36.97秒。
输出视频从source frame30开始，去除1秒初始静置；672帧=22.4秒，原速播放。
视频：runs/single_hand_fast_20261011/pick_place/ego_single_hand_fast.mp4
完成管线：code/collection/finish_single_hand.py，日志reports/single_hand_fast_pipeline.log。
视频必须等COMPLETED记录和解码检查后视为交付。

物体没有任务期间位姿命令、附着或外加搬运动力。只驱动机器人的有限关节电机，机器人加重力补偿。
PhysX CPU 480Hz，控制/记录30Hz；物体质量0.04kg。具体材料、驱动参数保存在initial_scene.usda和build.py。
最新运行：接触前最大位移5.96e-8m，最大求解接触穿入约17.2微米。
702个记录帧的表面采样和可见三角网格相交检查均为0。
此结论不等价于数学上的连续时间零穿透。碰撞几何为经烹制验证的保守外包络，未进行实物参数标定。

## 已保存数据

data/contact_pilot_20261011/episode_000000 至 episode_000003：4条原目标抓放，已通过动力学及逐帧几何检查，有完整同步RGB。
episode_000000为1920x1440；episode_000001..3为640x480。
episode_000004：本次提速左侧目标；由finish_single_hand.py在完成渲染后导出，1920x1440。
每条episode.npz含实际右手关节状态、下一步目标命令、物体和所有刚体位姿、接触力等，manifest说明时间对齐。
左臂关节测量未在这批standalone pilot中保存；只有左臂实际刚体位姿，不能伪造52维双臂关节观测。
episode0有移除手碰撞的因果对照，物体保持静止；不要把该独立对照声称为所有轨迹逐条做过。
这是pilot格式，不是已经完成pi05/LeRobot训练格式及策略闭环。

## Isaac Lab 并行

code/collection/lab_collect_candidates.py：GPU PhysX，64个复制环境、1199帧、初始XY扰动±1mm。
runs/lab_collect_64_v1/：342.3秒完成64条候选，55条通过接触、静止、抬起和落点初筛。
9条接触前漂移超阈值排除。55条仍须几何及接触细节检查；accepted_episode_count保持0。
geometry_summary.json由code/collection/audit_lab_batch.py异步写入，勿把候选数当正式验收数。
当前安装IsaacLab3位姿四元数为XYZW，已查安装源代码QUAT_XYZW_ELEMENT_NAMES。每个环境审查前减去env_origin。

## 进程与历史边界

code/collection/process_manager.py：每GPU项目锁和独立队列状态文件，不停止其他项目任务。
双手交接静态规划失败且用户取消，无成功交接示范，不自动排队。
微波炉仍未修复；其余原七任务的运动学回放不是新增物理验收。
历史视频、releases与源资产保持独立。不要将原七任务页面改标为真实接触通过。
