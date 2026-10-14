# Dexterity Living — PRO5000 主工作区

最新单手物理任务与数采状态：[CURRENT_SINGLE_HAND.md](CURRENT_SINGLE_HAND.md)。历史渲染条目不代表当前物理验收。

迁移日期：2026-10-11。目标 SSH：`ssh -p 1025 bowenjing@8.130.80.101`。
主目录：`/media/external/bowenjing/projects/dexterity_living`。
已通过 Mac Studio 从 L20 迁入；原机器文件保留。主包和源代码补充包均进行了 SHA-256 校验。

## 目录管理

- `releases/20261011/`：两版各 7 项场景、视频、共享贴图、机位参数，作为保留快照。
- `sources/20261011/robots/{wuji,sharpa}/assembled.urdf`：可编辑的完整双臂装配，网格引用为相对路径。
- `sources/20261011/source_runs/`：任务的原始动作数据、配置与脚本。
- `sources/20261011/microwave006_internal_spring_latch.zip`：用户指定微波炉原包。
- `sources/20261011/legacy_code/`：旧脚本用于溯源，含 L20 绝对路径，不是新机器入口。
- `code/`：新工作区维护脚本；`scripts/`：命令入口；`configs/`：环境与版本。
- `runs/`：后续新输出；`reports/`：迁移和加载检查；`imports/`：校验后的原始压缩包。

代码、配置、说明纳入本地 Git；大资产、视频和运行结果不放入 Git。运行实验前复制快照，不覆盖 release。

## 预览

`./scripts/serve.sh 8793` 仅监听本机回环地址。页面入口 `http://127.0.0.1:8793/`，可切换 Wuji / Sharpa。
通过 Mac Studio 的示例：先在 Studio 执行 `ssh -N -L 18794:127.0.0.1:8793 -p 1025 bowenjing@8.130.80.101`，本机再执行 `ssh -N -L 8794:127.0.0.1:18794 macstudio`，浏览器打开 `http://127.0.0.1:8794/`。如本机没有 macstudio alias，使用已配置的 Mac Studio SSH 命令。

## 下一次渲染

原版 Isaac Sim 5.1；目标现有环境是 Isaac Sim 6.1 / Python 3.12：`/home/bowenjing/envs/isaaclab3/bin/python`。兼容验证以 `reports/runtime_smoke.json` 为准。

```bash
cd /media/external/bowenjing/projects/dexterity_living
./scripts/prepare_run.sh experiment_001
export ISAAC_PYTHON=/home/bowenjing/envs/isaaclab3/bin/python
export DEX_RENDER_ROOT="$PWD/runs/experiment_001/living_mcl_wuji_v1_20261011"
./scripts/render.sh pick_place
# 确认抽帧后，再添加 --full
```

`code/retarget.py` 从保留的 Sharpa 场景重新生成 `runs/wuji_retarget`；需要 numpy/scipy/USD。`code/audit.py` 检查该输出的微波炉网格相交，需要 python-fcl。缺少依赖时不要修改其他项目环境，另建环境或补齐项目专用依赖。

## 已知边界

全部 14 段为展示/动作回放。Wuji 指尖存在厘米级误差，物体轨迹沿用原版，尚未完成换手后真实接触验收。微波炉穿模用户要求暂不处理。弹簧在单独 PhysX 对照中起作用，但没有合入当前展示视频。严禁把文件迁移、渲染成功或旧资产伺服测试解释为任务物理成功。

## 本次目标机验证

14 个场景依赖解析完成，缺失引用为 0；两套 URDF 网格引用通过；14 段视频 HTTP 检查通过。图形运行库独立存于 runtime/，NVIDIA Vulkan 使用 EGL ICD。重建依赖运行 scripts/setup_runtime.sh；渲染入口自动加载 scripts/runtime_env.sh。
单帧图像输出成功，但目视兼容检查未通过：Isaac Sim 6.1 下机械臂/手掌几何显示不完整。资产迁移与依赖闭合已通过，显示兼容仍待处理。，详情见 reports/runtime_smoke.json。

## 2026-10-11 黑手白桌路径追踪版

用户要求删除 PRO5000 原视频：14 个 release MP4 已删除，删除清单 reports/deleted_original_videos.json；imports 源压缩包保留。原 HTTP 视频通过记录仅为删除前历史。

新输出 runs/wuji_blackwhite_pt4k_20261011/living_mcl_wuji_v1_20261011/，7 项任务。入口 code/batch_pt.py、code/render_pt.py，先 source scripts/pathtrace_env.sh。3840×2880 / 30 fps / PathTracing / 1024 spp / 12 bounces，镜头保持桌面 EGO，near clip 0.025 m。黑手白桌只改材质，原桌垫保留厚度/碰撞/摩擦。早先 smoke 缺臂原因是相机默认 near clip 1 m，不是 USD 资产丢失；完整显示已通过新配色样帧检查。

容器缺少 nvoptix.bin。仅提取与驱动版本一致的 NVIDIA 580.126.09 原包到 runtime/optix580，用 code/optix_weights_redirect.c 编译的项目局部 LD_PRELOAD 将精确文件路径重定向至项目，不修改系统库；其他文件访问不受影响。包 SHA256 b05fe4bec0607ad8e2ed547608408d623a4455d05ec11ed6552186a9c443b046。

渲染状态见新输出 status.json 和 reports/pt4k_batch.log。每项任务分片可续跑，完成后拼接、解码检查并更新页面。新页面仅监听 127.0.0.1:8796，通过 Mac Studio 隧道从本机访问同端口。该批次是既有动作重渲染，不新增物理任务验收。

采样配置依据 NVIDIA Omniverse Path Tracing 文档：https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_pt.html 。必须在 stage 加载和 app warmup 后设定，避免 USD render settings 将 totalSpp 覆盖回 32。运行报告同时记录请求配置及实际 carb settings。

## 当前版本：全色碗与动作优化

最新输出根目录为 runs/wuji_blackwhite_smooth_pt4k_20261011/living_mcl_wuji_v1_20261011/。原 blackwhite_pt4k 队列已按用户改动停止，记录 reports/pt4k_stopped_for_revision.json。当前 8796 端口已切至新版。

运行 code/refine_motion.py 从保留 release 构建新版：两个碗各 33 个几何部分统一绑定绿色/黄色材质；机械臂和场景物体共享单调时间映射，空间路径不变。Wuji 手指在关节空间进行有界平滑、PCHIP 插值，再从手部 URDF 做 FK，关节限位保持，指尖相对原关节插值偏差不超过 3 mm。仅优化展示动作，不声称新增接触或物理任务通过。微波炉碰撞仍暂不处理。

抓放/堆叠峰值掌速从 1.13/1.11 m/s 降至 0.57/0.57 m/s；钢琴均匀减速到原来一半，视频约 24 s，音频用 atempo=0.5 保持音高与动作同步，并补齐至视频长度。完整数据见 reports/motion_refinement_v2.json、reports/refinement_static_checks.json。7 个场景所有已写入 physics/physx 参数保持不变，抽样匹配时间下的机械臂变换误差为 0。

当前批量入口 code/batch_pt.py；通过 DEX_RENDER_ROOT 可以覆盖输入/输出根目录。日志 reports/pt4k_smooth_batch.log、reports/pt4k_smooth/；完成报告 reports/pt4k_smooth_batch.json。每 60 帧一个分片，8 GPU 租用队列避免同卡重复运行。重复启动会检查现有 coordinator PID 并拒绝。可恢复已完成分片。

短动作预览仅为 1280×960/32 spp 的 6 秒片段，位于 produce_sort/quick_preview/ego.mp4；与 3840×2880/1024 spp 正式渲染分开。短预览完成后 GPU 7 自动回到正式队列。

机械臂保留 USD 原烘焙路径，本轮不直接使用整机 URDF 覆盖手臂 FK：整机 URDF 的坐标约定与当前烘焙帧定义尚未对齐。手部 URDF 与 retarget.json 的 FK 约定已用于本次有界手指优化。

## Complete seven-video previews (2026-10-11)

`code/batch_complete_previews.py` renders every frame of all seven refined tasks into each task's `complete_preview/ego.mp4`, at 1280x960, 30 fps, PathTracing 32 spp. It never writes into the 4K master chunks. The gallery reads `complete_previews.json` independently of master status, prefers complete masters when available, and preserves active video elements during progress updates. Six-second samples are no longer used as task videos.

The preview coordinator temporarily suspends only the verified project master process group and resumes it on exit. Current launcher receipts are under reports. Complete-preview validation requires the exact full timeline frame count and an error-free full FFmpeg decode. Piano audio uses the same 0.5 timing scale as its refined motion. These are USD animation playback renders, not new contact-physics or policy-success validation.

## 2026-10-11 Contact audit (current Wuji previews)

All seven stages have 52 hand rigid bodies, with zero enabled. The renderer stops the timeline and samples animated USD transforms without stepping physics. An offline robot-removal check across all frames leaves the animated task-object world matrices identical (maximum difference 0) for pick/place, stack, sorting and microwave. This test is USD playback dependency inspection, not a dynamics ablation. Gestures are non-contact demonstrations; piano has no new verified key-contact-force result.

`code/audit_current_contact.py` reproduces these findings. `code/audit_microwave_current.py` uses project-local python-fcl 0.7.0.11 and visible triangle meshes: 418/452 frames have robot-microwave intersections; 414/452 involve non-handle components, beginning at frame 0 (19 intersecting mesh pairs). This detects surface intersections at discrete video frames, not continuous collision or calibrated penetration depth. Reports are in `reports/current_contact_audit.json`, `reports/microwave_current_intersections.json` and `reports/microwave_contact_summary.json`.

The user has now requested investigation of the previously deferred microwave issue. It is confirmed and NOT repaired. A valid repair requires a collision-aware Wuji grasp/arm path, restored articulated dynamics with finite joint drives, and a door/latch driven by actual contact rather than authored transforms. Revalidate measured hand/object forces, release behavior and spring response before describing any task as physically accepted. Source scene and videos remain available as kinematic previews.
