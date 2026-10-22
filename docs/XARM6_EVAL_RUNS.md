# `Data/cube2_scene/s15_eval/` 各评测目录对应的模型与设置

> 整理于 2026-10-16。"成功"一列要看**判据**：`松` = 任务自带的 `InsideAABB`（箱子整体包围盒含竖起的盖子，会把"推歪箱子盖住方块"误判为成功）；`严` = 方块中心落在箱体底座 AABB 内且低于箱沿（2026-10-15 之后的评测脚本）。
> 带 † 的成功数是**误判**，视频里方块仍在箱子外。

## 总表

| 目录 | 日期 | 机器人 | 评测的模型 | 模型来源 | 训练数据 | 桌面/相机 | 夹爪处理 | episode | 成功 | 判据 |
|---|---|---|---|---|---|---|---|---|---|---|
| `pi05_lerobot/` | 10-13 | **Franka Panda** | `pi05_base`（LeRobot 格式，**未微调**） | 本地 `allocator/models/pi05_base` | 无（zero-shot） | 白地面，Franka 相机 | 反相 + 0.5 二值化 | 5 | 0 / 5 | 松 |
| `xarm6_ys903/` | 10-14 | xArm6 | `ys903/pi05_xarm6_pick_place_50ep` | Hugging Face，第三方 | 作者的 50 条真机演示（红块放黄盒，黑色光学平台） | 白地面，旧相机 | 0.5 | 5 | 0 / 5 | 松 |
| `xarm6_round1_20k/` | 10-15 | xArm6 | **第 1 轮** 20K 步（`pi05_droid_jointpos` 微调） | CHTC 任务 6780959 | 100 条，白桌面，旧相机，7 维布局 | 白地面，旧相机 | 0.5 | 10 | 0 / 10 | 松 |
| `grip_0.5/` | 10-15 | xArm6 | 同上（第 1 轮 20K） | 同上 | 同上 | 白地面，旧相机 | 阈值 0.5 | 3 | 0 / 3 | 松 |
| `grip_0.85/` | 10-15 | xArm6 | 同上（第 1 轮 20K） | 同上 | 同上 | 白地面，旧相机 | 阈值 0.85 | 3 | 1 / 3 † | 松 |
| `round1_thr085/` | 10-15 | xArm6 | 同上（第 1 轮 20K） | 同上 | 同上 | 白地面，旧相机 | 阈值 0.85 | 10 | 2 / 10 † | 松（人工看视频核实，真实 0/10） |
| `round3_nominal/` | 10-16 | xArm6 | **第 3 轮** 16K 步（`pi05_droid_jointpos` 微调） | CHTC 任务 6782318 | 450 条随机化，近相机，**8 维 DROID 布局** | 白地面，**近相机** | 滞回（3 步 ≥0.8 闭合 / ≤0.3 张开） | 10 | 0 / 10 | 严 |
| `round3_wood/` | 10-16 | xArm6 | 同上（第 3 轮 16K） | 同上 | 同上 | **随机木纹桌面** + 随机光照/相机抖动，近相机 | 滞回 | 10 | 0 / 10 | 严 |
| `realsim_base_16k_v1_earlyterm/` | 10-18 | xArm6 | **第 4 轮** 16K 步（`pi05_base` 微调） | CHTC 任务 6787525 | 50 真机 + 447 对齐真机的仿真，7 维，20 Hz | 对齐真机：标定相机、真机布局、0.70 倍箱子 + 长边带碰撞盖子，浅木纹随机化 | raw | 10 | 4 / 10 | 严（蓝块 3 条被任务的松判据提前结束，脚本已修） |
| `realsim_base_16k/` | 10-18 | xArm6 | 同上（第 4 轮 16K） | 同上 | 同上 | 同上 | raw | 20 | **12 / 20** | 严（蓝 6/10，黄 6/10；1 条撞到盖子 15.6° 并推动箱子 3 cm） |
| `real_base_8k/` | 10-19 | xArm6 | **只用真机数据** 8K 步（`pi05_base` 微调） | CHTC 任务 6800348 | 50 条真机，7 维，20 Hz | 同上（对齐真机的仿真） | raw | 13（第 14 条 Isaac Sim 崩溃） | 1 / 13 | 严（同样布局下第 4 轮前 13 条是 8/13；多数条目在方块附近徘徊、不闭合夹爪） |
| `eef_cotrain_16k/` | 10-20 | xArm6 | **第 5 轮**：50 真机 ×4 过采样 + 447 仿真，TCP 动作（10 维），16K 步，CHTC 6806585 | 同上 | 同上 | 同上（`--action-space eef`，IK 取最近可达解） | raw | 20 | 12 / 20 | 严；视频名带颜色和成功/失败（`episode_03_yellow_success.mp4`）。失败 8 条：3 条卡在工作空间边缘（IK 一直不收敛，夹爪从未闭合），5 条闭合后没抓成 |

## 各行补充说明

### `pi05_lerobot/`（最早的 Franka zero-shot）
- 那时场景里机器人是 Franka，离箱子约 1.24 m（超出臂展），所以这次 0/5 **没有诊断意义**。
- 里面的 `gr00t_base` 只是配置默认值留下的名字，实际模型是 pi05_base，不是 GR00T。
- `results_2026-10-13_21-2x ~ 21-3x` 三个目录只有 `rollouts.hdf5`，是调试跑，没有评测结果文件。
- prompt：`pick up cube and place it into box`；`execute_horizon` 8；动作 8 维（Franka 7 关节 + 夹爪）。

### `xarm6_ys903/`
- prompt：`Pick up the blue block and drop it into the box.`，30 秒，相机命名 `front` / `wrist`。
- **这次评测发生在夹爪物理修复之前**（当时 xArm6 的关节驱动还是 acceleration 类型，夹不紧），所以也不能完全说明模型本身的能力。不过方块一次都没碰到，主要还是视觉差异太大。
- 该 checkpoint 之后已删除以释放磁盘。

### `xarm6_round1_20k/`、`grip_0.5/`、`grip_0.85/`、`round1_thr085/`（同一个模型，不同夹爪阈值）
- 模型：第 1 轮 CHTC 任务 6780959 的 step 20K，本地路径 `outputs/chtc/xarm6_sim_6780959/checkpoints/020000/`。
- 基座 `DAVIAN-Robotics/pi05_droid_jointpos`，冻结视觉 + 语言主干，只训练动作专家，相对动作，batch 8 × 20000 步，训练 loss 0.043。
- 这四个目录是同一个模型在**不同夹爪二值化阈值**下的对比：
  - 阈值 0.5：夹爪在第 0–8 步就闭合，手臂直奔箱子，0 次碰到方块；
  - 阈值 0.85：夹爪不再提前闭合，但 `round1_thr085` 里标成功的 2 个 episode 是误判（手臂把箱子推歪，包围盒盖住了方块）。
- 评测脚本当时还没有严格判据，所以这几个目录的 `success` 都是松判据。

### `round3_nominal/`、`round3_wood/`（同一个模型，不同桌面）
- 模型：第 3 轮 CHTC 任务 6782318 的 step 16K，本地路径 `outputs/chtc/xarm6_close8_bs32/checkpoints/016000/`。
- 基座同上，batch 32 × 16000 步，有图像增强，训练 loss 约 0.042。
- 评测命令里必须带：`--sensors sim_xarm6_front_close --camera-keys droid --droid-layout`；`round3_wood` 另加 `--randomize --wood-only --seed 7`。
- 夹爪不再提前闭合（`first_close_step` 都落在 100–300 步，木纹那组有两个在第 2、47 步），但手臂仍然径直去箱子上方、方块被留在原地。

## 一眼看懂

| 变量 | 第 1 轮（`round1_thr085` 等） | 第 3 轮（`round3_*`） |
|---|---|---|
| 训练数据量 | 100 条 | 450 条 |
| 数据外观 | 纯白桌面 | 木纹/纯色桌面 + 光照 + 相机抖动 |
| 前置相机 | 旧（方块约 8–13 px） | 近相机（方块约 13–19 px） |
| 动作布局 | 7 维 `[J1..J6, 夹爪]` | 8 维 `[J1..J6, 0, 夹爪]`（夹爪对齐 DROID 槽位） |
| 夹爪后处理 | 阈值 | 滞回 |
| batch / 步数 | 8 × 20K | 32 × 16K |
| 成功（严格） | 0/10 | 0/10 + 0/10 |

## 下一轮（第 4 轮）

- 同样的 450 条数据（改成 7 维布局），基座换成 **`pi05_base`**（本地 `allocator/models/pi05_base`），其余设置一致，用来检验 DROID 预训练到底有没有帮助。评测时用 `--sensors sim_xarm6_front_close`，**不要** `--droid-layout`。

## 复现评测的最小命令

```bash
# 服务（tmux）
.cache/pi05-serve-venv/bin/python simfoundry/policies/pi05_serve/server.py \
    --checkpoint <checkpoint>/pretrained_model --device cuda --port 8420
# 评测（tmux）。第 3 轮示例：
OMNIGIBSON_HEADLESS=1 mamba run -n simfoundry python scripts/xarm6/eval_pi05_xarm6.py \
    --scene-json Data/cube2_scene/s14_og/reconstructed_og_scene.json --task cube2_pick_place_cube_in_box \
    --out-dir Data/cube2_scene/s15_eval/<name> --prompt "Pick up the blue cube and place it into the box." \
    --episodes 10 --timeout-s 30 --execute-horizon 15 \
    --sensors sim_xarm6_front_close --camera-keys droid --droid-layout [--randomize --wood-only]
```

## 真机 rollout（`Data/cube2_scene/real_deploy/<时间>/`，每条有 cameras.mp4 = 固定相机 | 腕部相机，log.jsonl = 每步关节/目标/夹爪）

部署脚本 `scripts/xarm6/real/deploy_pi05_xarm.py`（不经 ROS：xArm SDK 复现 xarm_stack 的 joint_vel_controller，mode 4、v = 10·(q_des − q)、限幅 0.6 rad/s、80 Hz；夹爪 850·(1 − g)；20 Hz、每次执行 15 步）。每条开始前机械臂移到训练起始姿态均值 [-0.052, -0.332, -0.517, 0.026, 0.957, 0]。

| 时间 | 模型 | 指令 | 时长 | 结果 | 备注 |
|---|---|---|---|---|---|
| 20261019_200742 | 第 4 轮 `realsim_base_16k` | yellow | 40 s | 成功 | 一次抓起放入；最大目标跳变 0.13 rad，夹住读数 226 |
| 20261019_200858 | 同上 | blue | 40 s | 失败 | 方块靠近底座左侧，多次夹空（读数降到 7） |
| 20261019_201003 | 同上 | blue | 60 s | 成功 | 抓起放入 |
| 20261019_201435 | 同上（reset 后） | yellow | 60 s | 成功 | 约 25 s 放入，之后在蓝块附近徘徊（无成功检测，按时长结束） |
| 20261019_201553 | 同上 | blue | 60 s | 失败 | 反复在方块侧边闭合，夹空或夹住后滑落，方块被推动 |

小计（第 4 轮模型）：黄 2/2，蓝 1/3，共 3/5。部署走 xArm SDK 而非 ROS（leo/bobby 的 snow_ws 没有读权限）；控制逻辑与 xarm_stack joint_vel_controller 相同（Kp 10，限幅改为 0.6 rad/s）。
