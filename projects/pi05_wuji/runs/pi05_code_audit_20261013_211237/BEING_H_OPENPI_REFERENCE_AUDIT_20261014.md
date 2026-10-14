# Being-H 与 OpenPI 对照审查

快照：2026-10-14T00:04:59.285432+08:00（北京时间）。本报告记录新增审查，不覆盖历史结果；当前350次评测及钢琴训练恢复队列未修改。

已阅读 Being-H 的统一动作空间、推理映射与归一化源码，以及 OpenPI 官方动作转换与训练配置。Being-H 的200维是语义槽位，包含预留、末端位姿、人手MANO等；不是200个真实关节。现有灵巧手映射不能直接容纳 Wuji 单手20维。可借鉴显式映射、维度断言、有效槽位 mask 和成对归一化，不能直接把52改成200。

OpenPI ALOHA默认对机械臂使用相对当前观测的动作，但仿真ALOHA配置明确支持绝对角。当前绝对52维训练和推理一致，尚未证明这种动作表示是bug。候选机械臂delta只修改0:6、26:32，其余40维手部仍为绝对目标。CPU执行实际官方DeltaActions/AbsoluteActions，在五任务真实训练样本30×52上往返误差小于3e-8rad，手部不变。转换依据chunk起始观测，不能逐帧累加。若训练该方案，必须在转换后仅用训练集重新计算动作统计，并部署相同反变换；不能把新解码器套到旧绝对动作checkpoint。

场景核对使用各任务 split_manifest.json 中45条实际训练轨迹；旧 acceptance.split 有部分历史不一致，已在v2诊断改为权威划分，保留初版诊断。物体初始位置基本落于训练范围，最大越界：抓放0.115mm、堆叠上块0.858mm、果蔬0.116mm以内，钢琴键起始位置一致。关节初始范围最大越界约0.0057rad。此项只核对位置范围，评测姿态是首个控制后记录；不能证明像素、相机、材质、完整动力学一致，也不能证明毫米误差无影响。

当前证据：已核对14条训练/留出样本的52维顺序、预处理、归一化往返和末段padding mask，未发现32维截断或任务one-hot错位。独立钢琴step1000开发评测0/5；错误顺序4/5、超过5N接触3/5，存在重叠，最终reason字段可能覆盖其他症状。专家钢琴控制器在检测到按键力/行程后冻结实际关节姿态；学习策略目前没有同类反馈，这属于需验证的控制/观测差异，不能宣称已定位唯一原因。

建议后续先用单条完整验收示范做真正过拟合：匹配预算对比绝对关节和机械臂delta，记录物理单位分模块误差，再在相同初始化下闭环测试，保持原成功/接触/几何标准。通过后再扩展45条及七任务。当前45条钢琴训练不是单条过拟合证明。本报告未启动新训练，不替换原恢复队列。

证据文件：
- evidence/official_delta_candidate_contract_20261014.json
- evidence/initial_scene_distribution_20261014_v2.json
- evidence/being_h_openpi_reference_audit_20261014.json

参考源码：
- Being-H unified action space: https://github.com/BeingBeyond/Being-H/blob/27ec67b84ac6c8e55f95408205e1ad7ba8bec253/Being-H05/docs/unified_action_space.md
- Being-H inference adapter: https://github.com/BeingBeyond/Being-H/blob/27ec67b84ac6c8e55f95408205e1ad7ba8bec253/Being-H05/BeingH/inference/beingh_policy.py
- Being-H state/action normalization: https://github.com/BeingBeyond/Being-H/blob/27ec67b84ac6c8e55f95408205e1ad7ba8bec253/Being-H05/BeingH/dataset/transform/state_action.py
- OpenPI training configs: https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/config.py
- OpenPI delta transforms: https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/transforms.py
