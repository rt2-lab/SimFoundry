# 在真机 xArm6 上部署 pi0.5 策略

本文说明如何在实验室的 UFactory xArm6 + xArm Gripper（主机 `heron`，机械臂 IP `192.168.1.111`）上运行
SimFoundry 训练的 pi0.5 模型，完成"把方块放进箱子"的任务：启动推理服务、准备机械臂和桌面、执行
rollout、查看结果。（英文版：`XARM6_REAL_DEPLOY_GUIDE_en.md`）

以下命令都在 SimFoundry 仓库根目录下运行：

```bash
cd /home/tingting/Documents/SimFoundry
PY=.cache/radio-prep-venv/bin/python        # 部署端：相机、机械臂、客户端
SRV=.cache/pi05-serve-venv/bin/python       # 推理服务：LeRobot + pi0.5
```

---

## 1. 整体结构

```
 D435i 固定相机 (front) ─┐
 D435i 腕部相机 (wrist) ─┤  deploy_pi05_xarm.py  ──HTTP /infer──►  pi05 推理服务（GPU，端口 8420）
 xArm 关节 / 夹爪 ───────┘        │                                  返回 50 步的动作块
                                  │
                                  ├─ 末端空间模型：TCP 目标 ─► IK（最近可达解，以实测关节为初值）
                                  ▼
                     关节目标 ─► 80 Hz 速度 P 控制：v = Kp (q_des − q)，|v| ≤ vmax
                                 （与 xarm_stack 的 joint_vel_controller 相同）
                     夹爪 ─────► set_gripper_position(850 · (1 − g))
```

- **不经过 ROS。** 脚本用 xArm Python SDK 控制机械臂，用 pyrealsense2 读相机。执行链路复现了采集演示数据时的方式（SpaceMouse → simple_ik → joint_vel_controller），所以策略看到的动力学和训练数据一致。
- **异步执行动作块。** 策略一次预测 50 步（20 Hz 下 2.5 秒）。执行到当前块的第 10 步时请求下一块，推理期间机械臂继续运动；新块到达后，跳过推理期间已经过去的步数，再用 5 步从旧块过渡到新块。机械臂不会因为推理而停顿。（`--sync` 恢复第一次部署时的方式：每 0.75 秒停约 0.25 秒，动作看起来一顿一顿的。）
- **没有成功检测。** 放好方块后策略会继续动作，通常会去够另一个方块。rollout 在到达 `--max-seconds` 或按 Ctrl-C 时结束。

---

## 2. 可用的模型

| 模型 | 训练数据 | 动作空间 | checkpoint | 结果 |
|---|---|---|---|---|
| 第 5 轮，末端空间 | 50 真机（过采样 ×4）+ 447 仿真 | `eef` | `outputs/chtc/xarm6_eef_cotrain_16k/checkpoints/016000/pretrained_model` | 仿真 12/20；真机黄块 1/1 |
| 第 4 轮，关节空间 | 50 真机 + 447 仿真 | `joint` | HF `TingtingDu/pi05-xarm6-real-sim-16k` | 仿真 12/20；真机 3/5 |
| 只用真机，关节空间 | 50 真机 | `joint` | HF `TingtingDu/pi05-xarm6-real-only-8k` | 仿真 1/13；真机未测 |
| 第 6a/6b 轮，末端空间 | 只用仿真 / 只用真机 | `eef` | CHTC `checkpoints/xarm6_eef_sim_16k.tar`、`xarm6_eef_real_8k.tar` | 训练中（2026-10-20） |

**`--action-space` 必须和模型匹配。** 末端空间模型的输入状态是 16 维、输出动作是 10 维；关节空间模型输入输出都是 7 维。不匹配时服务会报错，或者机械臂收到毫无意义的目标。试运行（§5）能发现这个问题。

从 Hugging Face 下载模型（公开仓库，不需要 token，每个约 9.4 GB，下载前先看 `df -h /`）：

```bash
$SRV -c "from huggingface_hub import snapshot_download; \
snapshot_download('TingtingDu/pi05-xarm6-real-sim-16k', local_dir='outputs/hf/pi05-xarm6-real-sim-16k')"
```

从 CHTC 只拉取某一步的 checkpoint，不在本地落 37 GB 的 tar（需要到 `transfer.chtc.wisc.edu` 的 ssh ControlMaster 连接，否则要输密码 + Duo）：

```bash
ssh tdu35@transfer.chtc.wisc.edu 'cat /staging/t/tdu35/simfoundry/checkpoints/xarm6_eef_sim_16k.tar' \
  | tar -xf - -C outputs/chtc xarm6_eef_sim_16k/checkpoints/016000/pretrained_model
```

---

## 3. 开始前检查

**现场**
- [ ] 看守机械臂的人能随时按到急停；工作空间内没有人。
- [ ] 桌面和演示数据一致：工具箱放在桌上、盖子**打开**（向后靠），蓝色和黄色 25 mm 方块放在箱子前方的桌面上，没有其他障碍物。
- [ ] 如果之前按过急停，先解除（见 §8）。

**进程。** 同一时间只能有一个程序控制机械臂，每台相机也只能被一个程序打开。

```bash
ps aux | grep -E "realsense2_camera_node|simple_ik|joint_vel_controller|gripper_controller|inference_test|snow_camera_viewer|deploy_pi05" | grep -v grep
```

- **不能**在运行：ROS 的 RealSense 节点（`gripper_cam`、`mount_cam`，会占住相机）、xarm_stack 的 `simple_ik` / `joint_vel_controller` / `gripper_controller` / `inference_test_vis`（会发送自己的关节目标）、另一个 `deploy_pi05_xarm.py`。
- 可以保留：xArm 的 ROS 驱动（`xarm_driver_node`）、Foxglove、data_collection 服务（只要它没有通过相机在录制）。
- 这些进程通常属于其他同学（leo、bobby），停掉之前先打招呼。用 sudo 停止：`sudo kill -INT <pid>`（先停 launch 进程）。

**GPU。** 推理服务需要约 10 GB 显存：`nvidia-smi --query-gpu=memory.used,memory.total --format=csv`。

---

## 4. 启动推理服务

推理服务加载一个 checkpoint，可以一直开着，多次 rollout 共用。放在 tmux 里运行，关掉终端也不会停：

```bash
tmux new-session -d -s pi05_srv "cd /home/tingting/Documents/SimFoundry && \
  $SRV simfoundry/policies/pi05_serve/server.py \
  --checkpoint outputs/chtc/xarm6_eef_cotrain_16k/checkpoints/016000/pretrained_model \
  --device cuda --port 8420 2>&1 | tee /tmp/pi05_srv.log"
# 高速执行时可加 --num-inference-steps 5 降低推理延迟（见 §6b）

curl -s localhost:8420/health        # 等到返回 {"ready":true,"status":"ok"}（约 15 秒）
```

- 必须加 `--device cuda`。默认的 `cpu` 会把 14 GB 的模型加载进内存，在这台 30 GB 内存的机器上会被系统杀掉。
- 启动时日志里的 `huggingface.co ... 200 OK` 是在检查 PaliGemma 分词器缓存，不是报错。

**tmux 基本操作。** `tmux attach -t pi05_srv` 进入查看；服务所在的窗口只显示日志，不接受键盘输入。**不要在那个窗口按 Ctrl-C**，会把服务停掉。按 `Ctrl-b c` 新建一个窗口来运行部署命令，`Ctrl-b n` / `Ctrl-b p` 切换窗口，`Ctrl-b d` 退出 tmux（服务继续运行）。

**同时开两个模型。** 在另一个端口再启动一个服务（等第一个加载完再启动第二个，避免内存峰值叠加），部署时用 `--port` 选择：

```bash
tmux new-session -d -s pi05_srv2 "cd /home/tingting/Documents/SimFoundry && \
  $SRV simfoundry/policies/pi05_serve/server.py --checkpoint outputs/hf/pi05-xarm6-real-sim-16k \
  --device cuda --port 8421 2>&1 | tee /tmp/pi05_srv2.log"
```

---

## 5. 执行一次 rollout

每次 rollout 都是同样三步：

```bash
# (a) 回到起始姿态：以 0.25 rad/s 慢速移动到演示数据起始姿态的平均值，并张开夹爪。
#     如果直线路径会让法兰低于 0.2 m，会先抬高 6 cm。
$PY scripts/xarm6/real/go_to_start_pose.py

# (b) 试运行：读相机和关节、请求模型、打印目标。机械臂不会动。
$PY scripts/xarm6/real/deploy_pi05_xarm.py --action-space eef \
    --prompt "Pick up the yellow cube and place it into the box." --dry-run

# (c) 真正运动，手放在急停上。随时可以按 Ctrl-C 停止。
$PY scripts/xarm6/real/deploy_pi05_xarm.py --action-space eef \
    --prompt "Pick up the yellow cube and place it into the box." --max-seconds 60
```

- **一定要先做 (a)。** 策略只见过这个姿态附近的起点，从别的位置出发行为不可预测。
- **指令。** 严格使用 `Pick up the {blue|yellow} cube and place it into the box.`（真机演示数据只用了这一种说法）。
- **正常的试运行输出：** 大约 12 行类似 `[dry] t=0.50 k=10 jump=0.120 flange_z=0.281 g=-0.00 tcp=[0.311, 0.028, 0.110]`，最后一行包含 `inference latency median ~0.2s`。
  - 试运行时机械臂不跟着动，所以 `jump` 会逐渐变大，但应远小于 0.35。
  - `flange_z` 开始时应该在 0.28 m 左右。
  - `g`（夹爪）开始时约为 0。
  - 如果出现形状或维度相关的报错，说明 `--action-space` 和服务加载的模型不匹配。
- **一次摆放测两个指令。** 两个方块都在桌上时，可以先测黄块，再（重新执行 (a) 之后）测蓝块。每测完一对就重新摆放方块。
- 关节空间模型用 `--action-space joint`（如果在第二个服务上，再加 `--port 8421`）。

---

## 6. 参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--action-space` | `joint` | 末端空间模型（第 5、6 轮）用 `eef`；第 1–4 轮和只用真机的关节模型用 `joint` |
| `--prompt` | 必填 | 任务指令 |
| `--max-seconds` | 40 | rollout 时长；建议 60（演示用时 17–29 秒） |
| `--dry-run` | 关 | 不使能运动；60 步后自动结束 |
| `--port` | 8420 | 使用哪个推理服务 |
| `--vmax` | 0.6 × speed，最多 π | 关节速度上限，rad/s（xarm_stack 用 1.0；演示中很少超过 0.45；硬件上限 π）。第一次谨慎测试可以调到 0.3 |
| `--kp` | 10 | 速度环的比例增益（xarm_stack 的配置值） |
| `--max-jump` | 0.35（随 speed 放宽，最多 0.8） | rad；任一关节目标与实测值相差超过此值就停止 |
| `--min-flange-z` | 0.15 | m；目标让法兰低于此高度就停止（抓取时法兰约 0.17） |
| `--replan-every` | round(10 / speed) | 执行到当前块第几个动作时请求下一块 |
| `--blend` | 5 | 新旧动作块之间的过渡步数 |
| `--sync` | 关 | 旧的"停下推理"方式，仅用于对比 |
| `--execute-horizon` | 15 | 每块执行的步数，仅 `--sync` 时有效 |
| `--hz` | 20 | 数据集的动作频率，不要改；要加速用 `--speed` |
| `--speed` | 1 | 动作播放倍速，见 §6b |
| `--loop-hz` | min(200, 20 × speed) | 控制循环频率 |
| `--tracker-hz` | 80 | 速度环频率（xarm_stack 是 80），高速时可调到 150–200 |
| `--robot-ip` | 192.168.1.111 | |
| `--out-dir` | `Data/cube2_scene/real_deploy` | 结果保存位置 |

---

## 6b. 加速执行（`--speed`）

模型学到的是演示的节奏（演示中位 22.7 s），机械臂只用到了硬件能力的约 14%（关节上限 π rad/s，TCP 上限 1 m/s）。`--speed` 把模型输出的同一条轨迹播放得更快：模型、推理都不变，一个 50 步的动作块从 2.5 s 压缩到 2.5 / speed s。

```bash
# 先试运行（机械臂不动），看最后一行的循环频率和等待比例
$PY scripts/xarm6/real/deploy_pi05_xarm.py --action-space eef --speed 2 \
    --prompt "Pick up the yellow cube and place it into the box." --dry-run
# 真正运动：从 1.5 倍开始，确认安全再往上加
$PY scripts/xarm6/real/deploy_pi05_xarm.py --action-space eef --speed 1.5 \
    --prompt "Pick up the yellow cube and place it into the box." --max-seconds 40
```

**`--speed` 会自动调整的设置**（启动时第一行会打印出来，都可以手动覆盖）：

| 设置 | 自动取值 | 原因 |
|---|---|---|
| 动作播放速率 | 20 × speed 个动作/秒，相邻动作之间线性插值 | |
| `--loop-hz` 控制循环 | min(200, 20 × speed) Hz | 实测能跑到 177 Hz（10 倍速时） |
| `--vmax` 关节限速 | min(π, 0.6 × speed) rad/s | 保持和 1 倍速相同的余量；π 是硬件上限 |
| `--max-jump` 跳变阈值 | min(0.8, 0.35 + 0.05 × (speed − 1)) rad | 跟随误差和速度成正比 |
| `--replan-every` | round(10 / speed)，最少 1 | 高速时更早请求下一块 |
| 夹爪速度 | min(5000, 1500 × speed) | 1500 是 xarm_stack 的设定，5000 是上限 |
| `--tracker-hz` 速度环 | 80 Hz（不随速度变） | 高速时可以手动调到 150–200，跟得更紧 |

**最高能到多快：由推理延迟决定。** 每个动作块到手时已经过去了一个推理延迟，下一块又要再等一个延迟，所以连续运行需要 2 × 推理延迟 × 动作速率 < 50，即

> 最高倍速 ≈ 1.25 ÷ 推理延迟（秒）

| 推理延迟 | 最高约 | 情况 |
|---|---|---|
| 0.36 s | 3.5 倍 | GPU 同时在跑仿真评测 |
| 0.2 s | 6 倍 | GPU 独占（正常情况） |
| 0.1 s | 12 倍 | 需要减少积分步数等优化 |

推理跟不上时，脚本会保持最后一个目标等待新动作块（机械臂减速，日志里 `starved` 为 true），连续等待超过 1 秒才停止。结束时最后一行会打印"多少比例的周期在等动作块"，这个比例明显大于 0 就说明已经超过了推理能支撑的速度。

**降低推理延迟：**
- 测试时不要让别的程序占用同一块 GPU（比如仿真评测）。
- 推理服务加 `--num-inference-steps 5`（训练时是 10）：flow matching 的积分步数减半，推理更快，动作精度可能略降，需要实测。

**高速下的风险：**
- 跟随延迟（约 0.1 s）不变，速度越快位置误差越大，轨迹会切角，靠近箱盖时可能碰撞。
- 夹爪的开合速度有物理上限，模型可能在夹紧之前就抬起，导致掉块。
- 人按急停的反应时间不变，高速时机械臂在这段时间里走得更远。先在空桌面上以试运行和低倍速确认行为，再逐步加速。

---

## 7. 安全机制

出现以下情况时，脚本会停止本次运行（先发零速度，再切回位置模式）：
- 某个关节目标与实测值相差超过 `--max-jump`；
- 目标让法兰低于 `--min-flange-z`；
- 机械臂 SDK 返回错误，或推理服务出错；
- 推理严重落后，连续 1 秒没有拿到新的动作块（短暂落后时会保持最后一个目标等待）；
- 按下 Ctrl-C，或到达 `--max-seconds`。

末端空间的目标如果超出关节限位或工作空间边界，会取**最近的可达位姿**（和 simple_ik 一样），而不是报错。在靠近机械臂底座的位置，夹爪垂直向下时 TCP 最低只能到约 7 cm 高度；方块离底座太近时，机械臂可能悬在上方反复尝试而抓不到。

急停永远是最后一道防线：软件检查只看目标位置，不检测接触力。

---

## 8. 急停或报错之后

1. 确认现场安全，解除急停。
2. 运行 `$PY scripts/xarm6/real/go_to_start_pose.py`：它会在移动前清除警告和错误、重新使能电机，并切到位置模式。
3. 如果仍然报错误码（比如发生过碰撞），在 UFactory Studio（浏览器打开 `http://192.168.1.111:18333`）里清除，然后重复第 2 步。

被 Ctrl-C 或安全检查中断的运行，机械臂会停在位置模式，直接从第 2 步继续即可。

---

## 9. 结果

每次运行会生成 `Data/cube2_scene/real_deploy/<YYYYMMDD_HHMMSS>/`：

| 文件 | 内容 |
|---|---|
| `cameras.mp4` | 固定相机和腕部相机并排的视频，20 fps，H.264（VS Code 可直接播放）；采用分片 MP4，运行中途被杀（急停、kill）也能播放已写入的部分 |
| `log.jsonl` | 每步一行：时间 `t`、动作块内的步号 `k`、实测关节 `q`、夹爪读数 `grip_raw`（0 = 合上 … 850 = 张开）、关节目标 `q_des`、夹爪目标 `g_des`、TCP 目标 `tcp_des`（末端模型）、`jump`、`flange_z_des`、`fading`（是否处于块间过渡）、`starved`（是否在等新的动作块）。日志每 50 ms 落盘一次 |
| `args.json` | 本次使用的参数（服务端口、动作空间、指令、安全阈值） |

脚本不知道任务是否成功，需要看视频判断并记下结果。建议的统一做法是重命名运行目录，比如 `20261020_191417_yellow_success`。

快速查看一次运行的概况：

```bash
$PY - Data/cube2_scene/real_deploy/<run>/ <<'EOF'
import json, sys, numpy as np
L = [json.loads(l) for l in open(sys.argv[1] + "log.jsonl")]
g = np.array([r["grip_raw"] for r in L])
print(f"{len(L)} steps; gripper min {g.min():.0f} (closed on a cube ≈ 220, on nothing ≈ 0); "
      f"max target jump {max(r['jump'] for r in L):.3f} rad; lowest flange target {min(r['flange_z_des'] for r in L):.3f} m")
EOF
```

夹爪读数最小值约 220 表示夹住了方块，接近 0 表示夹空。

---

## 10. 常见问题

| 现象 | 原因 / 解决 |
|---|---|
| `RuntimeError: xioctl(VIDIOC_S_FMT) failed ... Device or resource busy` | 有别的进程占用相机（ROS RealSense 节点、查看程序、另一个部署脚本），见 §3 |
| `curl: (7) Failed to connect to localhost port 8420` | 服务没启动或还在加载；`tail /tmp/pi05_srv.log` 查看 |
| 服务加载时被杀掉 | 没加 `--device cuda`，或者同时在加载另一个模型 |
| 服务报形状 / 维度错误 | `--action-space` 和 checkpoint 不匹配 |
| 机械臂不动，也没有报错 | 急停没解除或机械臂没使能，见 §8；或者有别的节点在控制机械臂，见 §3 |
| `stopped: safety: ... jump` | 策略要求的跳变太大。通常是没先回起始姿态（执行 §5 (a)），或者桌面和演示差别太大 |
| 机械臂在底座附近悬停，一直抓不到 | 方块离底座太近（< 0.18 m），这是已知弱点，把方块往外挪 |
| 在 tmux 窗口里打字只显示 `^[[A` | 那个窗口是推理服务，用 `Ctrl-b c` 新建窗口 |
| 动作一顿一顿的 | 开了 `--sync`，或推理太慢（最后一行的 `inference latency` 应约 0.2 秒） |

---

## 11. 相关文件

| 路径 | 作用 |
|---|---|
| `scripts/xarm6/real/deploy_pi05_xarm.py` | 部署脚本（本文） |
| `scripts/xarm6/real/go_to_start_pose.py` | 回到演示起始姿态 |
| `simfoundry/policies/pi05_serve/server.py` | 推理服务 |
| `simfoundry/policies/pi05_lerobot.py` | HTTP 客户端（`PI05XArm6Client`） |
| `simfoundry/robots/xarm6_kinematics.py` | 正 / 逆运动学（末端目标用 `ik_nearest`） |
| `scripts/xarm6/make_eef_dataset.py` | 末端空间状态 / 动作的定义（`tcp_pose9`） |
| `docs/XARM6_HANDOFF.md` | 项目现状、各轮训练、CHTC |
| `docs/XARM6_EVAL_RUNS.md` | 所有仿真和真机评测记录 |
