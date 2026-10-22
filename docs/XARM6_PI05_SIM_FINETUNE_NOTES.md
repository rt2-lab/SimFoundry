# xArm6 + pi0.5：仿真数据生成、微调与评测（汇报参考笔记）

> 时间范围：2026-10-13 至 10-16。任务：xArm6 把方块放进"已被打开的"工具箱（task graph 里"开箱"分配给人，机器人只做 *pick up a cube and place it into the box*）。
> 本文按"做了什么 → 每一轮的配置与结果 → 踩过的坑 → 复现命令"组织。第三轮评测结果见 §9.1；第 4 轮（base 基座消融）在准备中。

---

## 1. 一页总结

| 轮次 | 数据 | 主要改动 | 训练 | 仿真评测（严格判据） |
|---|---|---|---|---|
| zero-shot | — | 用 xArm6 上微调过的 `ys903/pi05_xarm6_pick_place_50ep` | 无 | 0/5 |
| 第 0 轮（作废） | 100 条，**吸附抓取**（方块悬在张开的手指之间） | — | 本地，10K 步磁盘写满崩溃 | 数据无效，已删除 |
| 第 1 轮 | 100 条，物理抓取，白色桌面，旧相机 | 相对动作 + DROID 基座 | CHTC H200，20K 步，batch 8，2.7 h，loss 0.043 | **0/10** |
| 第 2 轮 | 200 条，桌面/光照/相机随机化 + 图像增强 | 域随机化 | CHTC L40，short 队列 12 h 超时在 25.5K/30K 被驱逐，**模型丢失** | 无 |
| 第 3 轮 | 450 条，随机化 + **近相机** + **DROID 8 维布局** | 近相机、夹爪对齐 DROID 槽位、推理端夹爪滞回 | CHTC H100，16K 步，batch 32，7.6 h | **0/10（白桌面）、0/10（木纹桌面）**，见 §9.1 |

**核心结论（目前）：**
1. 仿真里的关节预测其实学得不错（离线对比 demo，平均误差 0.02–0.18 rad），失败来自**闭环行为**：夹爪输出糊，被二值化后第一步就闭合，模型误以为已经抓着方块，直接去箱子。
2. 任务自带的成功判据太松（整个箱子的包围盒含竖起的盖子），第一轮曾出现 2/10 "成功"的误判，实际方块还在箱子外。
3. 最关键的是**让仿真数据本身可信**：xArm6 夹爪的物理抓取、成功判据、训练格式（相对动作、维度对齐）都出过问题，每一个都足以让整套实验失去意义。

---

## 2. 整体流程

```
真实视频 scene.MOV
   │  SimFoundry A（重建 + 铰链关节）           ← 盖子是真的 revolute 关节，0–90°
   ▼
OmniGibson 场景 (箱子 + 蓝/黄方块 + xArm6)
   │  scripts/xarm6/generate_scripted_demos.py   ← 脚本规划 + 逆运动学，物理抓取，域随机化
   ▼
原始 demo (mp4 + npz)
   │  scripts/xarm6/convert_demos_to_lerobot.py  ← LeRobot v3 数据集，重算相对动作统计
   ▼
LeRobot 数据集
   │  scripts/xarm6/chtc/ (Docker + HTCondor)    ← pi0.5 微调 (lerobot-train)，wandb 记录
   ▼
checkpoint
   │  pi05_serve/server.py + eval_pi05_xarm6.py  ← 仿真闭环评测，严格成功判据
   ▼
成功率 / 视频 / 失败分析 → 决定下一轮改什么
```

---

## 3. 仿真环境

### 3.1 机器人：xArm6 + xArm 夹爪（`simfoundry/robots/xarm6.py`）
- 从官方 `xarm_description` 的 xacro 生成 URDF，经 `assets/robots/xarm6/clean_urdf.py` 清洗（去掉无几何的 frame link、STL 转 OBJ、新增 `eef_link` = TCP、四连杆右侧 mimic 重挂）。
- 导入 OmniGibson 后用 `scripts/assets/fix_xarm6_usd.py` 修复（见 §7 的坑）：右内指节 mimic 引用/限位、**驱动类型 acceleration → force**、**mimic 软弹簧 → 刚性**。
- 控制：机械臂绝对关节位置控制；夹爪 0–1 命令（0 开 / 1 关，对应 drive_joint 0–0.85 rad），夹爪 kp=30。
- 自写的 numpy 正逆运动学 `simfoundry/robots/xarm6_kinematics.py`（与仿真 FK 完全吻合；该机器人没有 cuRobo/Lula 描述）。

### 3.2 场景与任务前提
- 工具箱盖子是真实铰链关节。**重要**：articulation 修好之后必须重跑 stage 11–14，否则 USD 里盖子仍是固定关节；而 `group_init_joint_positions` 对没有关节的物体会静默跳过，"盒子已打开"这个前提会悄悄失效。现在每个 episode 开始都断言盖子 > 1.0 rad。
- 机械臂放在方块和箱子之间（世界坐标 (0, -0.45, 0)），距离约 0.55 m，在臂展内。

### 3.3 相机（与 ys903 真机数据对齐：`front` + `wrist`，30 Hz）
| 配置 | 前置相机 | 方块在 224 px 输入里的大小 |
|---|---|---|
| `sim_xarm6_front_wrist.yaml`（第 1、2 轮） | 机械臂左前方，69° 视场 | 约 8–13 px |
| `sim_xarm6_front_close.yaml`（第 3 轮） | 更近，42° 视场（D435 彩色流裁成正方形），位置由投影搜索得到 | 约 13–19 px |
手腕相机：UFactory D435i 标准支架位置（**与真机实际安装是否一致还没核对**）。

### 3.4 抓取
- 物理抓取，靠夹爪夹紧 + 手指摩擦（手指材质摩擦 2.0）。
- 曾经用过的 `GraspLatch`（几何吸附）因为会把夹爪冻结在 0.3 开度、方块悬空，第 0 轮数据作废；现在仅作调试备用。

---

## 4. 训练数据：脚本自动生成（`scripts/xarm6/generate_scripted_demos.py`）

**每条 demo：**
1. reset；随机化布局：两个方块位置 ±8 cm + 随机朝向，箱子 ±4 cm，机械臂初始关节 ±0.05 rad，速度 ×0.8–1.2，路径点小抖动。
2. 确认盖子是打开的。
3. 蓝/黄方块交替作为目标。
4. 8 个路径点：方块上方 8–12 cm → 下降到方块底面上方 6 mm → 闭合夹爪（停留 25 步）→ 抬起到 12–18 cm → 箱口上方 → 下放 → 张开（停留 12 步）→ 撤离 10 cm。位置用最小加加速度曲线，朝向用球面插值，每个点用逆运动学求关节角，等关节误差 < 0.01 rad 再前进。
5. 只保留方块最终落在箱体底座内的 episode。

**每帧记录：** 前置 + 手腕图像（320×320）、state（6 关节 + 夹爪）、action（下一步绝对关节目标 + 夹爪命令）、语言指令（三种说法随机，颜色对应目标方块）。

**域随机化**（`simfoundry/robots/xarm6_randomize.py`，第 2、3 轮）：
- 桌面：80% 程序化木纹（7 种木色，随机纹理/尺度/旋转/色调/粗糙度，色调围绕真实桌面的浅橡木 RGB≈215/180/130），20% 纯色垫子；
- 光照：穹顶光强度 ×0.25–0.9、色温 3500–7500 K；
- 相机：前置位置 ±3 cm、角度 ±3°、焦距 ±6%；手腕 ±4 mm、±2°；
- 随机参数记录在 `episodes.jsonl` 的 `domain_rand` 字段。
- 没有随机化的：天空背景、方块和箱子的颜色/材质、干扰物。

| 数据集 | 条数 | 帧数 | 成功率 | 备注 |
|---|---|---|---|---|
| 第 1 轮 `xarm6_cube_in_box_sim` | 100 | 34,419 | 100/100 | 白桌面，旧相机，7 维 |
| 第 2 轮 `xarm6_cube_in_box_rand` | 200 | 83,604 | 200/221（逆运动学失败 21） | 随机化，旧相机，7 维 |
| 第 3 轮 `xarm6_cube_in_box_close8` | 450（生成 455） | 187,939 | 约 90%（逆运动学失败 47、抓取失败 5） | 随机化，近相机，**8 维 DROID 布局** |

三个生成器并行（不同随机种子）把第 3 轮数据的生成时间从约 3 小时压到约 1.3 小时。

---

## 5. 模型与训练

### 5.1 为什么选 `DAVIAN-Robotics/pi05_droid_jointpos` 作基座
- pi0.5 base 是 Physical Intelligence 在大规模多机器人数据 + 网络图文上预训练的通用起点。
- `pi05_droid_jointpos` = pi0.5 base 再在 DROID（Franka，约 7.6 万条真机轨迹）上以**关节位置动作空间**微调，抓放能力和"一个外部相机 + 一个手腕相机"的设置最接近我们的任务。
- 它是 OpenPI 官方 `gs://openpi-assets/checkpoints/pi05_droid_jointpos`（JAX）经 OpenPI 官方脚本转成 LeRobot PyTorch 格式的版本，权重官方、格式第三方（DAVIAN 实验室；其模型卡在 sim 评测上 7/8，对比原 JAX 6/8）。
- **它预测的是相对当前关节的增量，夹爪是绝对值**（我最初误以为是绝对关节角，后来读模型卡才发现），所以训练必须用相对动作，并按相对空间重算动作统计量。
- 官方 `pi05_droid` 很可能输出关节速度（官方文档未写明，是由 sim-evals 另用关节位置版推断的），不适合我们的绝对关节数据。

### 5.2 训练配方（`scripts/xarm6/finetune_pi05.sh`，LeRobot 0.6.1）
- 冻结视觉编码器，只训练动作专家（6.9 亿 / 40 亿参数），bf16，gradient checkpointing，显存约 16 GB。
- `use_relative_actions`（夹爪排除），数据集的动作统计量按 chunk=15 重算为相对空间。
- 相机映射：`front → base_0_rgb`，`wrist → left_wrist_0_rgb`，第二个手腕相机留空。
- 图像增强（第 2、3 轮）：亮度/对比度/饱和度/色调/锐度/小幅仿射，每帧随机 2 种。
- 第 3 轮把 state/action 改成 **8 维 `[J1..J6, 0, gripper]`**：DROID 基座的夹爪在第 8 维（索引 7），我们之前把夹爪放在第 7 维（DROID 第 7 关节的位置），预训练的夹爪知识用不上。
- 超参：第 1、2 轮 batch 8；第 3 轮 batch 32 × 16000 步（约 2.8 个 epoch）。wandb 项目 `pi05_xarm6_sim`。

### 5.3 训练运行记录
| 运行 | 硬件 | 配置 | 结果 |
|---|---|---|---|
| 本地（第 1 轮数据） | RTX 5090 32 GB | batch 8，20K，约 1.1 s/步 | 到 20K 步时日志显示 loss 0.032；写最后一个 checkpoint 时被打断，只保留 15K 步权重 |
| CHTC 第 1 轮 | H200 | batch 8，20K，2.24 步/s，2.7 h（9806 s） | loss 5K 0.064 → 12K 0.052 → 20K 0.043（wandb 曲线为准） |
| CHTC 第 2 轮 | L40（short 队列） | batch 8，30K，1.68 s/步 | 12 h 超时在 25.5K 被驱逐，checkpoint 丢失 |
| CHTC 第 3 轮 | H100 80 GB | batch 32，16K，1.71 s/步 | 15K 步 loss 0.042（等完成） |

---

## 6. CHTC 训练流程（`scripts/xarm6/chtc/`）

```
本机: stage_to_chtc.sh → 上传数据集 tar、基座模型 tar（含 PaliGemma 分词器缓存）、任务文件
CHTC 接入点 ap2002: condor_submit train.sub -append "exp_name=..." -append "dataset_name=..." ...
节点: Docker 容器（torch 2.11 / cu126，包版本与本地一致）→ 解包 → 离线训练 → 打包 checkpoint
回传: /staging/t/tdu35/simfoundry/checkpoints/<exp>.tar → 只抽出最后一步 scp 回本地
```
- 登录需要密码 + Duo；用 SSH ControlMaster 在 tmux 里保持连接，8 小时内无需再验证。
- GPU Lab：L40/L40S（45 GB）、A100（40/80 GB）、H100（80 GB）、H200（141 GB）。job 长度：short 12 h、medium 24 h、long 7 天。
- 训练全程离线（分词器随输入文件上传），只有 wandb 联网。
- **重要经验**：不要用 short 队列跑正式训练；同一个任务在 H200 上 0.45 s/步，在 L40 上 1.68 s/步。`train.sub` 默认已改成 medium。

---

## 7. 踩过的坑（按对结果的影响排序）

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| 1 | 物理抓取 0/24，手指碰到方块但一抬就滑掉 | URDF 导入器把所有关节驱动写成 acceleration 类型，刚度乘连杆惯量（夹爪零件仅 20–50 g），夹紧力约 1e-3 N·m；mimic 又是软弹簧 | 所有驱动改成 force，mimic 改刚性，夹爪 kp=30；12/12 抓起 |
| 2 | 第 0 轮 demo 里手指没碰到方块 | GraspLatch 在夹爪闭合到 0.3 时就挂约束，OmniGibson 随即冻结夹爪开度，方块悬在张开的手指间 | 弃用吸附，数据全部重做 |
| 3 | OmniGibson 自带辅助抓取失效 | 要求两个手指上报接触（导入的夹爪不上报），并把"下限=闭合"写死，xArm 相反 | 同上 |
| 4 | "盒子已打开"没生效 | articulation 修复发生在 stage 10–14 之后，USD 里盖子仍是固定关节；`group_init_joint_positions` 静默跳过 | 重跑 stage 11–14；加断言 |
| 5 | 第 1 轮评测 2/10 "成功"是误判 | 任务的 `InsideAABB` 用整个箱子（含竖起的盖子）的包围盒，手臂推歪箱子就盖住方块 | 评测用严格判据：方块中心在箱体底座 AABB 内且低于箱沿 |
| 6 | `touched_cube` / `picked_cube` 永远是 0 | 夹爪不上报手指接触 | 忽略这两个里程碑，只看严格的 in-box |
| 7 | 地面换材质没效果 | 地面 mesh 没有 UV，且 Isaac 把材质绑在父节点上覆盖子节点绑定 | 在地面上方 0.3 mm 放一个带 UV 的纯视觉四边形，隐藏原地面 |
| 8 | 第一次本地训练 10K 步磁盘写满崩溃 | 每个 checkpoint 约 11 GB（权重 8.8 + 优化器 2.1），保存时新旧同时存在 | 保存前清理旧的；CHTC 上 bundle 去掉优化器状态 |
| 9 | 第 1 轮 0/10 | 夹爪输出糊（0.4–0.6），按 0.5 二值化第一步就闭合，模型误以为在运方块，直奔箱子；方块只有约 8 px | 近相机、夹爪滞回、8 维 DROID 布局（第 3 轮） |
| 10 | CHTC 第 2 轮丢失 | short 队列 12 h 超时 | 默认 medium；估算时长要按最慢的卡 |
| 11 | CHTC 提交参数没生效 | `condor_submit train.sub var=value` 被文件里的默认值覆盖 | 用 `-append "var=value"` |
| 12 | 容器里 `getuser()` 报 KeyError | 容器以提交者 uid 运行，无 passwd 条目 | 设置 `USER` / `LOGNAME` |
| 13 | 某台 A100 节点首次用 GPU 就报 device busy | 节点本身问题 | 启动前加 CUDA 自检并排除该节点 |
| 14 | 仿真里 Isaac 退出时 exit 139 | 已知的 Isaac 关闭阶段段错误 | 无害，可忽略 |

---

## 8. 评测方法

- **服务**：`simfoundry/policies/pi05_serve/server.py` 常驻进程，模型只加载一次，HTTP 接口；图像键和 state 宽度从 checkpoint 配置读取；DROID 系 checkpoint 的 postprocessor 会把相对动作转回绝对关节角。
- **闭环**：`scripts/xarm6/eval_pi05_xarm6.py`。每个 episode：reset → 断言盖子打开 → 发送（两路图像缩放到 224、state、语言指令）→ 取回一个 chunk（15 步）→ 执行 15 步 → 重复，超时 30 s（900 步，30 Hz）。
- **成功判据（严格）**：方块中心落在箱体底座（不含盖子）的 AABB 内，且低于箱沿。
- **夹爪后处理**：滞回（连续 3 步 ≥ 0.8 闭合，连续 3 步 ≤ 0.3 张开）。第 1 轮评测用的是阈值法。
- **评测配置必须和训练数据一致**：
  | 项目 | 第 1 轮 | 第 3 轮 |
  |---|---|---|
  | `--sensors` | `sim_xarm6_front_wrist`（默认） | `sim_xarm6_front_close` |
  | `--droid-layout` | 否 | **是** |
  | `--camera-keys` | `droid` | `droid` |
  | `--randomize` | 否（白桌面） | 建议两组都跑：白桌面 + `--randomize` |
- **离线诊断**（很有用）：把训练集里的真实观测喂给服务，对比预测和 demo 的动作，用来区分"模型没学会"和"闭环/流程有问题"。第 1 轮的结论就是这么得出的。

第 3 轮评测命令：
```bash
# 1) 服务（tmux）
.cache/pi05-serve-venv/bin/python simfoundry/policies/pi05_serve/server.py \
    --checkpoint outputs/chtc/xarm6_close8_bs32/checkpoints/016000/pretrained_model --device cuda --port 8420
# 2) 评测（tmux），两组
OMNIGIBSON_HEADLESS=1 mamba run -n simfoundry python scripts/xarm6/eval_pi05_xarm6.py \
    --scene-json Data/cube2_scene/s14_og/reconstructed_og_scene.json --task cube2_pick_place_cube_in_box \
    --out-dir Data/cube2_scene/s15_eval/round3_nominal --prompt "Pick up the blue cube and place it into the box." \
    --episodes 10 --timeout-s 30 --execute-horizon 15 \
    --sensors sim_xarm6_front_close --camera-keys droid --droid-layout
#   加 --randomize --wood-only 再跑一组，输出目录换成 round3_wood
```
注意：本机内存只有 30 GB，训练、推理服务、仿真三者不要同时跑。

---

## 9. 第 1 轮评测详情（0/10）

| 设置 | 结果 | 说明 |
|---|---|---|
| 夹爪阈值 0.5 | 0/10 | 夹爪在第 0–8 步就闭合，手臂直奔箱子，一次都没碰到方块 |
| 夹爪阈值 0.85 | 系统标 2/10，**实际 0/10** | 看视频：方块还在箱子外，手臂把箱子推歪导致包围盒误判 |
| 离线（训练观测→预测） | 关节平均误差 0.02–0.18 rad | 模型、相对动作转换、归一化、服务流程都正常；夹爪预测起始时 0.5–0.77，demo 是 0 |

---

## 9.1 第 3 轮评测详情（0/10 + 0/10）

| 评测桌面 | 成功 | 夹爪第一次闭合的步数 |
|---|---|---|
| 白桌面（10 个 episode） | 0/10 | 107–243（一个 522），全部闭合过 |
| 木纹桌面，随机木色/光照/相机（10 个 episode） | 0/10 | 2–294 |

- **改进了的**：离线对比训练观测，夹爪起始预测 ≈ 0（第 1 轮是 0.5–0.77），搬运阶段 ≈ 1.04；关节平均误差 0.059 rad。夹爪不再开局就闭合。
- **没改进的**：看视频，手臂仍然径直去箱子上方，在那里合上夹爪，方块留在原地。第一步的预测和训练观测下的预测相近（joint1 都在 0.1 附近），问题出在后续闭环里没有"跟着方块走"。
- 关节误差 0.059 rad ≈ 在 0.5 m 臂展下约 3 cm，而抓 4 cm 的方块需要约 1 cm 的精度。
- 可能原因（尚未验证）：视觉编码器冻结、数据量仍小、只有"干净路径"没有恢复行为、4 cm 方块在 224 px 里仍然很小。
- 下一步（第 4 轮）：用 pi0.5 base 作基座做消融；之后可试解冻 Gemma / 视觉、缩短执行 horizon（每 5 步重新观测）。

## 10. 目前还没有的东西 / 风险

- 没有做过"只加随机化"或"只加近相机"的消融（第 2 轮丢失）；第 3 轮同时改了多件事，若成功率提升，无法分辨各自的贡献。
- 仿真 → 真机的差距：抓取靠物理摩擦（真机夹爪的接触细节不同）、夹持读数约 0.5（真机约 0.6）、手腕相机安装位姿未标定、背景（天空）未随机化。
- 数据都是脚本生成的、路径很干净；没有恢复行为（失败后的重试），闭环里一旦偏离分布，模型没见过怎么回来。

## 11. 文件索引

| 作用 | 路径 |
|---|---|
| 机器人类 / 运动学 / 环境 / 随机化 | `simfoundry/robots/xarm6.py`、`xarm6_kinematics.py`、`xarm6_env.py`、`xarm6_randomize.py`、`tabletop_textures.py` |
| 资产导入与修复 | `assets/robots/xarm6/`、`scripts/assets/import_custom_robot.py`、`fix_xarm6_usd.py` |
| demo 生成 / 转换 / 训练 / 评测 | `scripts/xarm6/generate_scripted_demos.py`、`convert_demos_to_lerobot.py`、`finetune_pi05.sh`、`eval_pi05_xarm6.py` |
| CHTC | `scripts/xarm6/chtc/`（`train.Dockerfile`、`train.sub`、`run_train_chtc.sh`、`stage_to_chtc.sh`、`pull_checkpoints.sh`、`README.md`） |
| 夹爪滞回 / 推理客户端 / 服务 | `simfoundry/policies/gripper_filter.py`、`pi05_lerobot.py`、`pi05_serve/server.py` |
| 相机配置 | `scripts/cfg/external_sensors/sim_xarm6_front_wrist.yaml`、`sim_xarm6_front_close.yaml` |
| 详细操作步骤（中文） | `docs/ARTICULATION_AND_VLA_GUIDE.md` 第 7 节 |
| 数据 | `Data/cube2_scene/lerobot/`（三个数据集）、`Data/cube2_scene/s15_eval/`（评测视频与 `results.json`） |
| 训练曲线 | wandb 项目 `pi05_xarm6_sim` |
