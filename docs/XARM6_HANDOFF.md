# xArm6 抓方块放箱子：交接文档（2026-10-19，更新至 2026-10-22）

任务：UFactory xArm6 + xArm Gripper 把蓝/黄方块抓起放进已打开的箱子。用 pi0.5 在真机数据 + 对齐真机的仿真数据上微调，CHTC 训练，仿真和真机评测。

**一句话现状**：第 5 轮模型（末端空间动作，50 真机×4 过采样 + 447 仿真）在对齐仿真里 12/20，**真机上 1/1**（黄块成功，第 2 次蓝块测试被人为急停，非模型问题）。第 7 轮（同样的数据配比，但演示做了约 2.1 倍的"数据加速"）刚训练完（loss 0.017），**还没有评测**，是接手后第一件要做的事。另外发现：直接在部署时加速现有模型效果很差（仿真 2 倍速成功率从 10/20 掉到 2/20），真正有效的是让训练数据本身更快，这是做第 7 轮的原因。

详细背景另见：
- `docs/XARM6_REAL_DEPLOY_GUIDE.md`（中文）/ `_en.md`（英文）：**真机部署完整操作指南**，接手真机测试先读这个
- `docs/XARM6_SIM_REAL_ALIGNMENT.md`：仿真与真机对齐的全部测量和改动
- `docs/XARM6_EVAL_RUNS.md`：每次仿真/真机评测的目录、模型、结果
- `docs/XARM6_PI05_SIM_FINETUNE_NOTES.md`、`docs/ARTICULATION_AND_VLA_GUIDE.md` §7：前三轮的过程记录

---

## 1. 结果总览

| 轮次 | 基座 | 训练数据 | CHTC 任务 | 训练 | 对齐仿真 | 真机 |
|---|---|---|---|---|---|---|
| 1 | pi05_droid_jointpos | 100 条纯仿真（白地面，旧相机） | 6780959 | bs8 × 20k | 0/10（旧仿真） | — |
| 3 | pi05_droid_jointpos | 450 条纯仿真（近相机，8 维 DROID 布局） | 6782318 | bs32 × 16k | 0/10、0/10（旧仿真） | — |
| **4** | **pi05_base** | **50 真机 + 447 对齐仿真** | 6787525 | bs32 × 16k，loss 0.024 | **12/20** | **3/5** |
| 真机-only | pi05_base | 50 真机 | 6800348 | bs32 × 8k，loss 0.020 | 1/13（第 14 条 Isaac Sim 崩溃） | 未测 |
| **5（末端空间）** | pi05_base | 50 真机 ×4 过采样 + 447 仿真，**TCP 动作** | 6806585 | bs32 × 16k，loss 0.021 | 12/20 | 1/1（黄，第 2 次被急停） |
| 6a（末端，只用仿真） | pi05_base | 447 仿真，TCP 动作（`chtc_eef_sim/`） | 6845773 | — | **已取消**（腾 GPU 给第 7 轮；未训练） | — |
| 6b（末端，只用真机） | pi05_base | 50 真机，TCP 动作（`chtc_eef_real/`） | 6859131 | — | **已取消**（同上；未训练） | — |
| **7（数据加速）** | pi05_base | 50 真机 ×4 过采样 + 447 仿真，TCP 动作，演示约 2.1 倍提速 | 6896347 | bs32 × 16k，loss 0.017 | **还没评测——接手后先做这个** | 未测 |

（第 2 轮在 CHTC 上被撤销，没有结果。6a/6b 的 `.sub` 已经修好 GPU 架构限制，想继续可以直接重新提交，见 §8d 末尾的 GPU 坑说明。）

真机 5 次的明细、视频路径见 `docs/XARM6_EVAL_RUNS.md` 末尾。规律：黄块每次一抓就起；蓝块的问题在最后的对准，常在方块侧边闭合导致夹空或滑落。

### 仿真速度扫描（第 5 轮模型，同一批 20 个布局，模拟真机速度环执行）

| 倍速 | 动作频率 | 成功 | 成功用时中位 |
|---|---|---|---|
| 1× | 20 Hz | 10/20 | 15.6 s |
| 2× | 40 Hz | 2/20 | 11.8 s |
| 3× | 60 Hz | 0/20 | — |
| 5× | 100 Hz | 0/20 | — |
| 10× | 200 Hz | 0/20 | — |

**结论：部署时直接加速现有模型行不通**，2 倍速就明显跟不上（相对动作的每一块都以"当前位置"为起点，机械臂追不上目标，新的一块又从没走到位的地方重新规划，陷入原地游荡）。这是做第 7 轮"数据加速"而不是"部署加速"的原因。细节见 §8e。

W&B（tingtingdu06-uw-madison）：第 4 轮 `pi05_xarm6_real_sim/runs/bsgqso6b`，真机-only `pi05_xarm6_real/runs/97pi9s3w`，第 5 轮项目 `pi05_xarm6_eef`，第 7 轮项目 `pi05_xarm6_eef_fast`（任务 6896347）。

---

## 2. 文件在哪

### 数据
| 路径 | 内容 |
|---|---|
| `/home/tingting/Documents/data/autoalloc_real/epXXXXXX/episode.h5` | 真机原始数据，51 条（ep8 空），从 `/media/leo/Elements/COLLAB_POLICY/datasets/autoalloc` 拷贝 |
| `Data/cube2_scene/xarm6_demos_real_{0,1,2}/` | 对齐真机的仿真原始数据，450 条，真机同格式 h5（约 10 GB） |
| `Data/cube2_scene/lerobot/xarm6_real` | LeRobot：50 条真机，22,772 帧，7 维（关节），20 Hz，chunk 50 相对动作统计 |
| `Data/cube2_scene/lerobot/xarm6_real_sim` | LeRobot：50 真机 + 447 仿真，205,683 帧，7 维（关节）（3 条撞盖子的仿真被剔除） |
| `Data/cube2_scene/lerobot/xarm6_real_sim_eef` | 同上转成末端空间（`make_eef_dataset.py`）：10 维动作 / 16 维状态，第 5 轮用的数据 |
| `Data/cube2_scene/lerobot/xarm6_sim_eef`、`xarm6_real_eef` | 第 6a/6b 轮的单数据源版本（`filter_dataset_by_source.py`、`make_eef_dataset.py`），任务已取消但数据留着 |
| `Data/cube2_scene/lerobot/xarm6_real_sim_fast` | **数据加速**版（`convert_real_h5_to_lerobot.py --speedup`），关节空间，98,431 帧（约 2.1 倍提速） |
| `Data/cube2_scene/lerobot/xarm6_real_sim_fast_eef` | 上面转成末端空间，**第 7 轮用的数据** |
| `Data/cube2_scene/lerobot/xarm6_cube_in_box_{sim,rand,close7,close8}` | 前几轮的纯仿真数据集（旧布局，可删，CHTC 上有） |

### 模型
| 路径 | 内容 |
|---|---|
| `outputs/chtc/xarm6_eef_cotrain_16k/checkpoints/016000/pretrained_model` | **第 5 轮**（末端空间，真机上验证过） |
| CHTC `/staging/.../checkpoints/xarm6_eef_fast_cotrain_16k.tar` | **第 7 轮**（数据加速），还没拉到本地，拉取命令见下 |
| CHTC `/staging/t/tdu35/simfoundry/checkpoints/` | 全部 checkpoint 打包，含第 1/3/4/真机-only/5/7 轮，各 19–37 GB。第 4 轮、真机-only 的本地副本已删除（传到了 HF，见下） |
| HF `TingtingDu/pi05-xarm6-real-sim-16k` | 第 4 轮（关节空间），公开仓库 |
| HF `TingtingDu/pi05-xarm6-real-only-8k` | 只用真机（关节空间），公开仓库 |

只要某一步：`ssh tdu35@transfer.chtc.wisc.edu 'cat <tar>' | tar -xf - -C outputs/chtc <exp>/checkpoints/<step>/pretrained_model`（不落本地临时文件）。

**拉第 7 轮（接手后第一步）：**
```bash
ssh tdu35@transfer.chtc.wisc.edu 'cat /staging/t/tdu35/simfoundry/checkpoints/xarm6_eef_fast_cotrain_16k.tar' \
    | tar -xf - -C outputs/chtc xarm6_eef_fast_cotrain_16k/checkpoints/016000/pretrained_model
```

### 评测结果
- 仿真：`Data/cube2_scene/s15_eval/{realsim_base_16k, real_base_8k, eef_cotrain_16k}/`（视频 + results.json，第 4/真机-only/5 轮）；`Data/cube2_scene/s15_eval/speed/eef_x{1,2,3,5,10}/`（第 5 轮的速度扫描）
- 真机：`Data/cube2_scene/real_deploy/<时间>/`（`cameras.mp4` = 固定相机 | 腕部相机，H.264，`log.jsonl` = 每步关节/目标/夹爪，`args.json` = 本次参数）

### 标定与布局
- `scripts/cfg/real/xarm6_mount_extrinsics.json`、`xarm6_wrist_extrinsics.json`：两台 D435i 外参（从数据反推）
- `scripts/cfg/external_sensors/sim_xarm6_real.yaml`：仿真相机（真机内参 + 外参）
- `scripts/cfg/real/xarm6_real_layout.json`：每条真机 episode 的箱子位姿、方块位置、起止关节

---

## 3. 代码（都在 SimFoundry 仓库）

| 脚本 | 作用 |
|---|---|
| `scripts/xarm6/real/estimate_mount_extrinsics.py` / `estimate_wrist_extrinsics.py` | 从录制数据估相机外参 |
| `scripts/xarm6/real/extract_real_layout.py` | 从真机数据提取布局（深度点云 + 正运动学） |
| `scripts/xarm6/real/compare_sim_real.py` | 仿真按真机某帧的关节/布局渲染，和真机图像并排对比 |
| `simfoundry/robots/xarm6_real.py` | 对齐真机的场景：箱子 0.70 倍、方块 25 mm、纯色、指尖黑色、盖子限位与半透明、桌边/面包板/地毯、真机格式 h5 写入、碰盖/推箱检测 |
| `scripts/xarm6/generate_scripted_demos.py --real-layout ...` | 生成对齐真机的仿真演示（20 Hz、慢速、夹爪斜坡、真机格式输出） |
| `scripts/xarm6/convert_real_h5_to_lerobot.py` | 真机/仿真 h5 → LeRobot（同一个转换器；`--workers N` 并行；自动剔除撞盖的仿真条目） |
| `scripts/xarm6/eval_pi05_xarm6.py --real-layout ...` | 对齐仿真评测（严格判据、碰盖检测） |
| `simfoundry/policies/pi05_serve/server.py` | 推理服务（已改为直接加载到 GPU，否则 30 GB 内存的机器会 OOM） |
| `scripts/xarm6/real/deploy_pi05_xarm.py` | **真机部署**（xArm SDK，不经 ROS，见 §5；`--action-space joint/eef`，默认异步推理；`--speed` 加速，见 §8e） |
| `scripts/xarm6/make_eef_dataset.py` | 关节空间数据集 → 末端空间数据集（见 §8b） |
| `scripts/xarm6/demo_speedup.py` | 选帧做"数据加速"（见 §8d）；直接跑会打印每类数据集的压缩效果 |
| `scripts/xarm6/filter_dataset_by_source.py` | 从混合数据集里只留 real 或只留 sim（第 6 轮用） |
| `scripts/xarm6/chtc_eef/` | 第 5/7 轮训练包（末端动作 + 真机过采样，共享训练代码） |
| `scripts/xarm6/chtc_eef_sim/`、`chtc_eef_real/` | 第 6a/6b 轮训练包（已取消，未训练，代码留着） |
| `scripts/xarm6/real/go_to_start_pose.py` | 真机回到训练起始姿态均值 |
| `scripts/xarm6/chtc/stage_to_chtc.sh` | 上传 base / dataset / 作业文件（流式，不占本地磁盘） |
| `scripts/xarm6/chtc/submit_run.sh` | 一行：上传数据集（如需）+ 提交训练 |
| `docs/XARM6_REAL_DEPLOY_GUIDE.md`（中）/ `_en.md`（英） | **真机部署完整指南**：环境检查、启动服务、安全机制、加速部署、故障排查 |

常用命令：

```bash
cd /home/tingting/Documents/SimFoundry
# 推理服务（末端空间模型；第 5 轮路径，第 7 轮拉下来后把路径换成 xarm6_eef_fast_cotrain_16k/.../016000/...）
.cache/pi05-serve-venv/bin/python simfoundry/policies/pi05_serve/server.py \
    --checkpoint outputs/chtc/xarm6_eef_cotrain_16k/checkpoints/016000/pretrained_model --device cuda --port 8420
# 对齐仿真评测（末端空间模型加 --action-space eef；--speed 可测加速，见 §8e）
OMNIGIBSON_HEADLESS=1 mamba run -n simfoundry python scripts/xarm6/eval_pi05_xarm6.py \
    --scene-json Data/cube2_scene/s14_og/reconstructed_og_scene.json --task cube2_pick_place_cube_in_box \
    --out-dir Data/cube2_scene/s15_eval/<name> --prompt "Pick up the {color} cube and place it into the box." \
    --real-layout scripts/cfg/real/xarm6_real_layout.json --action-freq 20 --timeout-s 30 --execute-horizon 15 \
    --camera-keys droid --grip-mode raw --randomize --seed 3 --episodes 20 --action-space eef
# 真机（先起推理服务；完整指南见 docs/XARM6_REAL_DEPLOY_GUIDE.md）
.cache/radio-prep-venv/bin/python scripts/xarm6/real/go_to_start_pose.py
.cache/radio-prep-venv/bin/python scripts/xarm6/real/deploy_pi05_xarm.py --action-space eef --prompt "Pick up the yellow cube and place it into the box." --dry-run
.cache/radio-prep-venv/bin/python scripts/xarm6/real/deploy_pi05_xarm.py --action-space eef --prompt "Pick up the yellow cube and place it into the box." --max-seconds 60
# CHTC 训练（末端空间 + 真机过采样，用 chtc_eef 包）
bash scripts/xarm6/chtc_eef/stage_and_submit.sh <exp_name> Data/cube2_scene/lerobot/<dataset>
```

---

## 4. 仿真与真机对齐（已完成）

完整测量见 `docs/XARM6_SIM_REAL_ALIGNMENT.md`。要点：

- **数据格式**：仿真直接写真机 `episode.h5`（同键名、20 Hz、关节 rad、link6 位姿、夹爪 0–850 原始单位、JPEG 图像）。仿真不写深度。
- **相机**：两台 D435i（固定 SN 243222070233，腕部 SN 243122074665），内参从相机读出；外参从数据反推（固定相机重投影 6.2 px，深度验证桌面 z = 0 ± 5 mm）。
- **物体**：真箱子 = 重建工具箱 × 0.70；方块 25 mm；夹住方块时仿真读数 224，真机 223。
- **箱盖**：SimFoundry 第 9 步把铰链放在短边，已手改 `Data/cube2_scene/s9_articulate_objects/cube2_scene/toolbox/results/mobility.urdf` 为长边并重跑第 11–14 步（原文件备份 `Data/cube2_scene/backup_shortedge_lid_20261017/`）。盖子是带碰撞的关节，每条随机开到 125–165° 并作为限位。资产里箱体和盖子都是 1 kg，会被盖子拽翻，加载后改为 0.8 / 0.2 kg。
- **布局**：箱子位姿、方块位置、起止关节从 50 条真机数据采样加噪声。

### 已知未解决的差异：控制动力学（Leo 指出）
真机的"期望关节角"是 xarm_stack IK 在 80 Hz 的输出，由速度 P 控制器跟踪；仿真是脚本轨迹 + 关节位置驱动。实测：

| | 真机 | 仿真 |
|---|---|---|
| 实际落后指令 | 2 步（100 ms） | 4 步（200 ms） |
| 对齐后跟踪误差 | 0.75 mrad | 4.4 mrad |
| 关节速度 p50/p95 | 0.18 / 0.43 rad/s | 0.28 / 0.80 rad/s |

训练用相对动作，这个差异会改变动作分布。**建议的修正（尚未实施，等决定）**：仿真里复现 joint_vel_controller（80 Hz，v = 10·(q_des − q)，限幅 1 rad/s），轨迹先给末端目标再由"从实测关节出发"的 IK 得到期望关节角，速度降到约 0.05 m/s，然后重新生成数据。

---

## 5. 真机部署

### xarm_stack 怎么控制（`/home/tingting/Documents/xarm_stack`）
SpaceMouse → 累加末端目标位姿 → simple_ik（80 Hz，从实测关节出发）→ joint_vel_controller（mode 4，Kp 10，限幅 1 rad/s，`vc_set_joint_velocity`）。夹爪 `set_gripper_position`，速度 1500。**录制程序不在这个仓库**（leo 的 data_collection，bobby 那里有 `rt2-lab-data_collection`）。xarm_stack 的推理界面要的是**末端位姿**，我们的模型输出**关节角**，不能直接用。

### 我们的部署（`deploy_pi05_xarm.py`，完整指南见 `docs/XARM6_REAL_DEPLOY_GUIDE.md`）
- 不经 ROS：xArm SDK 直连 `192.168.1.111`，照搬 joint_vel_controller（mode 4，Kp 10，80 Hz）；夹爪 = 850·(1 − 输出)；pyrealsense2 直读两台相机 640×480。
- **异步推理**：执行到动作块第 10 步时请求下一块，到达后跳过已过去的步数、5 步交叉过渡；`--sync` 复现最早的同步方式（会有约 1.3 Hz 的走停）。
- **`--speed`（2026-10-21 新增）**：把同一条预测轨迹播放得更快，不改模型。控制循环最高 200 Hz（读关节/夹爪各一次、夹爪读数缓存 50 ms、视频编码放独立线程）；`--vmax`、`--max-jump`、`--replan-every`、夹爪速度都随 `--speed` 自动放宽（公式见指南 §6b）。推理跟不上时保持最后目标等新块（`starved`），连续 1 秒没等到才停。**最高可用倍速 ≈ 1.25 / 推理延迟（秒）**，GPU 独占时约 6 倍，和别的任务共享 GPU 时会明显更低。
- **`--num-inference-steps`**（推理服务参数）：减少 flow matching 积分步数（训练用 10）以降低推理延迟，配合加速部署。
- 安全：`--dry-run` 不动；目标偏离超过阈值或法兰过低停止；Ctrl-C 停；退出时零速 + 位置模式。末端空间目标超出工作空间时取最近可达解（`ik_nearest`），不会报错卡死。
- **每次 rollout 前必须先 `go_to_start_pose.py`**：模型只见过训练起始姿态附近的起点。
- 视频从 OpenCV `mp4v` 改成 H.264 + 分片 MP4（2026-10-21）：VS Code/浏览器能直接播放，进程被杀（急停、kill）也不会损坏已写入的部分。
- 真机实测：第 4 轮（关节空间）5 次，3 次成功；第 5 轮（末端空间）1 次成功（黄块）、1 次因急停中断（非模型问题）。**加速部署还没有真正在真机上跑过**（仿真扫描显示 2 倍速已明显变差，见下）。

### ROS 环境（未完成）
这台机器的 ROS（jazzy，Nix）环境在 bobby/leo 的 home 下：
- bobby：`/home/bobby/snow_ws/enter_snow_nix.sh` → `vendor/rt2-lab-snow_ws` 的 flake（`nix develop --no-update-lock-file --impure`），`source install_nix/setup.bash`；跨用户通信需要 `FASTDDS_BUILTIN_TRANSPORTS=UDPv4`。
- `ROS_DOMAIN_ID=83`，`ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`。
- leo 的 xarm 驱动（`xarm6_driver.launch.py robot_ip:=192.168.1.111`）和 data_collection 服务一直在跑。
- 用户已授权用 sudo 复制 bobby 的 snow_ws 来复用；**还没复制**。复制后的价值：和录数据时完全同一套节点/参数/相机设置，并能用 data_collection 录 rollout（格式和训练数据一致）。

### 对别人的影响（需要告知）
- 为测试**停掉了 bobby 的 xarm_stack 推理节点**（`inference_test.launch.py`：simple_ik、joint_vel_controller、gripper_controller、inference_test_vis）**和相机查看程序**（`snow_camera_viewer.py`）。
- 前几天停掉过 leo 的两个 RealSense ROS 节点（gripper_cam / mount_cam），之后没有重开。
- leo 的 xarm 驱动保持运行。

---

## 6. CHTC

- 访问节点 `ap2002.chtc.wisc.edu`，传输 `transfer.chtc.wisc.edu`，用户 tdu35，staging `/staging/t/tdu35/simfoundry`，作业目录 `~/simfoundry_chtc`，镜像 `tdu35/simfoundry_lerobot_train:0.6.1`。
- 已上传：`pi05_base_base.tar`、`pi05_droid_jointpos_base.tar`、`xarm6_real_dataset.tar`、`xarm6_real_sim_dataset.tar` 及旧数据集。
- 登录要密码 + Duo。tmux `ssh_keepalive` 每 5 分钟用一下 ssh ControlMaster（`~/.ssh/cm-tdu35@<host>:22`），连接从 10-17 晚上一直保持；断了就需要重新 Duo。
- W&B 凭据在 ap2002 的 `~/.bashrc`，`train.sub` 用 getenv 读取。
- 坑：`-append 'exp_name=..._$(Cluster)'` 在输出文件名里不会展开（第 4 轮的 tar 叫 `xarm6_realsim_base_.tar`），`submit_run.sh` 改为显式给 exp_name；双引号里写 `$(Cluster)` 会被本地 shell 执行。
- 训练速度：H100/A100/H200 上 bs32 约 1.5–1.6 s/步；**L40 约 7.0 s/步**（4.5 倍慢），16k 步会超过 24 小时的 job 上限，要避免。
- **GPU 架构坑（2026-10-20 发现）**：镜像里的 torch 是按 CUDA 12.6、最高 sm_90 编译的。分到更新的卡（RTX PRO 6000 Blackwell，架构 12.0；或任何架构 12.x）会在数据传完后立刻报 `cudaErrorNoKernelImageForDevice`。三个 `.sub`（`chtc_eef`、`chtc_eef_sim`、`chtc_eef_real`）已加 `gpus_maximum_capability = 9.0`。想进一步排除慢卡（L40/L40S/A40），提交时加
  `-append 'require_gpus = (Capability == 8.0 || Capability == 9.0)'`（第 7 轮这样做过，见 §8d）。
- `scripts/xarm6/chtc_eef/stage_and_submit.sh` 现在接受 `MIN_GPU_MEM` 环境变量（默认 80000M，40000M 也能跑，候选卡更多）。

---

## 7. 当前运行状态（交接时，2026-10-22）

| tmux | 内容 | 可否关闭 |
|---|---|---|
| `pi05_srv` | 推理服务，**第 5 轮**末端空间模型已加载（`xarm6_eef_cotrain_16k/016000`，约 10 GB 显存，GPU 共占 20 GB/32 GB） | 评测/部署第 7 轮前需要换成第 7 轮的 checkpoint 重启 |
| `ssh_keepalive` | 保持 CHTC 连接（每 5 分钟一次 ControlMaster 保活） | 不用 CHTC 就可以关 |
| `chtc_login`、`chtc_transfer` | 旧的 CHTC 登录会话 | 可以关 |

- 机械臂：最后一次部署结束的位置，位置模式，没有报错。
- **磁盘很紧张：根分区约 9.8 GB 可用**。可清理：`Data/cube2_scene/s1_video`（8.1 GB，只重跑第 11–14 步用不到）、旧 LeRobot 数据集 `xarm6_cube_in_box_*`（约 2.6 GB，CHTC 上有备份）、`xarm6_sim_eef`/`xarm6_real_eef`（第 6 轮数据，已取消训练，约 1.9 GB）、原始 h5 `xarm6_demos_real_*`（10 GB，如果不重新生成仿真数据可删，有 `xarm6_real_sim*` LeRobot 数据集留着就够）。**拉第 7 轮 checkpoint（约 9.4 GB）前，先腾出空间。**
- CHTC 上没有本次会话提交、还在排队或训练的任务（6a/6b 已取消，第 7 轮已训练完）。

---

## 8. 下一步（按优先级）

1. **评测第 7 轮**（最优先）：拉取 checkpoint，在对齐仿真里用同一批 20 个布局跑 `--action-space eef`，和第 5 轮（12/20）对比成功率和用时；重点看方块离底座近的几条（ep0/1/2/8，第 5 轮在这类布局上明显弱）。
2. **真机测第 7 轮**：确认仿真结果后上真机，同时验证数据加速是否真的让任务变快（第 5 轮中位 15.6 s，见 §8e 的仿真数字）。部署时注意调高 `--vmax`（见 §8d 末尾）。
3. **真机测加速部署**：GPU 已空闲，按指南 §6b 从 `--speed 1.5` 开始，仿真显示 2 倍速会明显下降，真机上要谨慎验证。
4. **真机对比关节空间 vs 末端空间**：都用异步部署、同样摆放，分开看动作表示和部署方式各自的贡献（HF 上有第 4 轮和真机-only 的关节空间模型可下载）。
5. **补蓝块**：蓝块（尤其靠近底座左侧）的对准是主要失败模式。用 SpaceMouse 补采演示，或在仿真里增加这些位置。
6. **修正仿真控制动力学**（§4），重新生成仿真数据并训练，看是否缩小真机差距。
7. **ROS 部署 + 录 rollout**：复制 bobby 的 snow_ws，用原版 xarm_stack 节点执行，用 data_collection 录 rollout（可做 DAgger）。
8. 部署时加成功检测（目前按时长结束，成功后策略会继续在剩下的方块附近徘徊）。
9. **如果想继续第 6 轮的数据消融**（只用仿真/只用真机，末端空间）：`.sub` 已修好 GPU 架构限制，`scripts/xarm6/chtc_eef_sim/submit.sh`、`chtc_eef_real/submit.sh` 直接可跑，数据已经在本地（`xarm6_sim_eef`、`xarm6_real_eef`）。

---

## 8b. 第 5 轮：末端（TCP）动作 + 真机过采样（2026-10-19 提交）

- 数据集 `Data/cube2_scene/lerobot/xarm6_real_sim_eef`（`scripts/xarm6/make_eef_dataset.py` 从 `xarm6_real_sim` 生成，视频复用）：动作 10 维 = TCP 目标 [xyz, 旋转矩阵前两列, 夹爪]，状态 16 维 = TCP 当前 + 夹爪 + 6 个关节。TCP = URDF `eef_link`（link6 + 0.172 m）。
- 训练包 `scripts/xarm6/chtc_eef/`（独立于 `chtc/`，说明见其 README）：真机 episode 重复 4 次（占 33%），学习率衰减步数 = 总步数。CHTC 作业目录 `~/simfoundry_chtc_eef`，W&B `pi05_xarm6_eef`，输出 `/staging/.../checkpoints/xarm6_eef_cotrain_16k.tar`。
- 部署：`deploy_pi05_xarm.py --action-space eef`（IK 从上一目标出发，和 simple_ik 一致）。部署脚本已改为**异步推理 + 5 步交叉过渡**，`--sync` 复现旧行为。
- 关节空间动作"不连贯"的主要原因是旧部署方式：同步推理每 15 步（0.75 s）冻结目标 0.25 s，Kp = 10 下机械臂基本停下，形成约 1.3 Hz 的走停。演示数据本身在关节空间和末端空间一样平滑（粗糙度 0.173 vs 0.172）。

## 8c. 第 6 轮：末端空间的数据消融（2026-10-20 提交）

和第 5 轮只差数据：`xarm6_sim_eef`（`filter_dataset_by_source.py --keep sim`，447 条）和 `xarm6_real_eef`（`make_eef_dataset.py --src xarm6_real`，50 条）。训练包 `scripts/xarm6/chtc_eef_sim/`、`chtc_eef_real/`（各自的 .sub 和 submit.sh，训练代码共用 `chtc_eef/`）。输出 `checkpoints/xarm6_eef_sim_16k.tar`、`xarm6_eef_real_8k.tar`。

第 4、5 轮仿真评测（同一批 20 个布局）的失败分析：两者只在 ep0–2 同时失败，任一成功 17/20，说明多数失败是随机的抓取失误；方块离基座 < 0.18 m（ep0/1/2/8）是系统性弱点（关节 1/4，末端 0/4，末端有 3 个卡在 IK 不收敛：工具垂直向下时 x = 0.137 m 处 TCP 最低只能到 0.07 m，真机操作员会倾斜手腕，仿真演示几乎不倾斜）。

## 8d. 第 7 轮：数据加速（2026-10-21 提交）

目标：让策略本身更快（演示中位 22.7 s，其中 28% 静止、放下后收尾占 27–31%）。

- 选帧：`scripts/xarm6/demo_speedup.py`（夹爪在动的帧全留；夹爪开始动作前 1.5 s 的精细段全留、停顿最多留 4 帧；静止段每段留 1 帧；自由空间隔帧保留 = 2 倍速；放下结束 1 s 后截断；被跳过的区间里关节累计变化 > 0.02 rad 或 TCP > 3 mm 就强制保留一帧，避免跳变）。数据集仍按 20 fps 标注，所以演示整体变快约 2.1 倍（真机 22.6 → 10.1 s，仿真 20.1 → 9.7 s）。
- 数据：`convert_real_h5_to_lerobot.py --speedup`（与 `xarm6_real_sim` 同一批 497 个文件）→ `xarm6_real_sim_fast`（98,431 帧）→ `make_eef_dataset.py` → `xarm6_real_sim_fast_eef`。`sources.jsonl` 记录每条的原始帧数 `orig_length`。
- 训练：`chtc_eef` 包，设置与第 5 轮相同（16k 步、真机 ×4），任务 6896347（限定 A100 / H100 / H200：`-append "require_gpus = (Capability == 8.0 || Capability == 9.0)"`；第一次提交 6893855 被分到 L40，7.0 s/步，16k 步要 31 h，超过 24 h 上限，已取消），W&B `pi05_xarm6_eef_fast`，输出 `checkpoints/xarm6_eef_fast_cotrain_16k.tar`。
- 部署注意：加速后的数据关节速度 p99 约 1.3 rad/s（真机）/ 1.8 rad/s（仿真），1 倍速部署时默认 `--vmax 0.6` 会截速，要加 `--vmax 1.5` 左右。

**CHTC 新坑**：第 6 轮的两个任务被分到 `sgomezgpu6000`（RTX PRO 6000 Blackwell，架构 12.0），容器里的 torch 没有对应内核，报 `cudaErrorNoKernelImageForDevice` 后被挂起。三个 .sub 已加 `gpus_maximum_capability = 9.0`。第 7 轮第一次提交又被分到 L40（7.0 s/步，16k 步要 31 h，超过 24 h 上限），取消后加 `require_gpus = (Capability == 8.0 || Capability == 9.0)` 限定 A100/H100/H200 重新提交，分到 H200，1.56 s/步，6 小时 54 分跑完，**loss 0.017**（第 5 轮同样设置是 0.021，说明数据量少了并没有让 loss 变差——压缩掉的大多是低信息量的静止帧）。

## 8e. 加速部署与仿真测速（2026-10-21）

动机：27 s 左右的任务时长里，演示本身 28% 是静止、27–31% 是放下方块后的收尾，机械臂只用到硬件能力（关节 π rad/s、TCP 1 m/s）的约 14%。先测了"部署时直接加速"这条路。

**部署脚本改动**（`deploy_pi05_xarm.py`，不改模型和推理）：
- `--speed`：把预测轨迹播放得更快（20 × speed 个动作/秒，动作间线性插值）。
- `--loop-hz`（默认 min(200, 20×speed)）：控制循环频率，优化后（每周期只读一次关节/夹爪、夹爪读数缓存、视频编码放独立线程）实测 10 倍速下能到 177 Hz。
- `--vmax`、`--max-jump`、`--replan-every`、夹爪速度随 `--speed` 自动放宽（公式见 `docs/XARM6_REAL_DEPLOY_GUIDE.md` §6b）。
- 推理跟不上时保持最后目标等新块（不再直接报错停止），连续 1 秒没等到才停；日志记录 `starved`。
- 推理服务新增 `--num-inference-steps`，可以降到 5（训练用 10）换取更低延迟。
- 视频编码从 OpenCV `mp4v` 改成 H.264 + 分片 MP4（见下面"踩过的坑"）。

**仿真里测了 1/2/3/5/10 倍速**（第 5 轮模型，同一批 20 个布局，模拟真机速度环）：1 倍 10/20、2 倍 2/20、3 倍起 0/20（结果见第 1 节表格）。**结论：直接加速现有模型不可行**，2 倍速就已经让机械臂追不上相对动作目标、陷入原地游荡。**这是转向"数据加速"（第 7 轮）的直接原因**——不是推理跟不上（推理延迟约 0.2–0.3 s，远小于能支持的速度上限），而是模型学到的动作节奏本身就慢，加速播放只会让跟随误差变大、任务没有进展。

**仿真测速时顺带的发现**：仿真环境的 `physics_frequency` 原来硬编码 120 Hz，跑 200 Hz 动作频率时物理步进不够密，已改成 `max(120, action_freq)` 自适应（`simfoundry/robots/xarm6_env.py` 新增 `physics_freq` 参数，`eval_pi05_xarm6.py` 的 `--speed` 会自动算）。

真机上还没有真正用 `--speed > 1` 跑过；仿真结果建议先在真机上从 1.25–1.5 倍小心验证。

## 9. 踩过的坑（排查时先看）

- 真机深度图**没有对齐到彩色**：要用深度内参 + 深度→彩色外参。
- 腕部相机离指尖约 10 cm：指尖轮廓顶角是指垫远端（x = +0.016 m），不是中线。
- 在已摆放好的刚体上加 USD 材质绑定，PhysX 会把它弹回初始位置：材质只在初始化时绑定一次，之后只改颜色。
- 任务 YAML 只认蓝块，其宽松判据（箱子包围盒含竖起的盖子）会在方块还悬在箱子上方时提前结束蓝块 episode；对齐模式下已改为只用严格判据。
- 推理服务按默认方式加载会峰值 18 GB 内存被 OOM 杀掉；已改为直接加载到 GPU 并清缓存。
- 仿真和推理服务同时跑时内存紧张，Isaac Sim 偶尔崩溃（carb assets plugin）。
- 磁盘满时连命令输出都写不出来；相对路径的 `rm` 要在 SimFoundry 目录下运行。
- OpenCV `VideoWriter` 的 `mp4v` 编码（MPEG-4 Part 2）在 VS Code/浏览器里打不开；进程被杀时索引没写完，文件会 `moov atom not found` 彻底打不开（抢救方法：从样本文件里扒 `extradata` 补到裸码流前面再转码，很麻烦，不如一开始就用对编码）。已改成 H.264 + 分片 MP4（`VideoThread`，独立线程写，边写边建索引）。
- `.env` 里有 sudo、CHTC、W&B、HF、GitHub、Docker 等密钥，其中 CHTC 密码和 W&B key 在早期对话输出里出现过，建议更换。所有新增脚本/文档提交前都检查过没有带出真实密钥（只提到环境变量名）。
