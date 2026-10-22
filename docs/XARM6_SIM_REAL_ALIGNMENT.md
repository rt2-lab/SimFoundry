# xArm6 仿真 ↔ 真机对齐（2026-10-17）

目标：让仿真数据在**格式、相机、布局、物体尺寸、时序**上都和真机采集的 `autoalloc` 数据一致，真机数据和仿真数据走同一个转换脚本进 LeRobot，可以直接混合训练。

## 1. 真机数据（`/media/leo/Elements/COLLAB_POLICY/datasets/autoalloc`，本地拷贝 `/home/tingting/Documents/data/autoalloc_real`）

| 项 | 内容 |
|---|---|
| 条数 | 51（ep8 是空录制，剔除；剩 50 条） |
| 频率 | 20 Hz（`timestamps` 间隔 0.05 s） |
| 长度 | 285–1065 步，中位数 453 步（22.6 s）；抓取中位数第 140 步，松开第 277 步 |
| 图像 | `mount_image_rgb_compressed`（固定相机）、`gripper_image_rgb_compressed`（腕部相机），640×480 JPEG；深度为 12 字节头 + 16 位 PNG（毫米，**深度相机坐标系，未对齐到彩色**） |
| 关节 | `robot_current/desired_joint_angles`（rad）、`..._joint_velocity` |
| 末端位姿 | `robot_current/desired_pose` = link6（法兰）xyz（m）+ 四元数 xyzw |
| 夹爪 | `gripper_actual/commanded_position`，原始单位 0（闭合）… 850（全开）= 开口 0.1 mm；指令按 ≤50/步 斜坡变化；夹住 25 mm 方块时读数 200–230 |
| prompt | 没有存。ep0–27 抓蓝块，ep28–50 抓黄块 |
| 特殊条目 | ep10 开始时方块已在夹爪里（只有"放"）；ep45 夹爪只合到 306（仍成功） |

## 2. 相机

**内参**（直接从插在本机的两台 D435i 读出，畸变为 0）：

| 相机 | 序列号 | fx | fy | cx | cy | 水平 FOV |
|---|---|---|---|---|---|---|
| mount（固定） | 243222070233 | 605.75 | 605.36 | 324.01 | 239.67 | 55.7° |
| gripper（腕部） | 243122074665 | 600.89 | 599.88 | 331.04 | 244.82 | 56.1° |

深度流（640×480）：fx=fy≈386.6；深度→彩色外参平移 14.6 mm。

**外参**（没有标定板，从录制数据反推）：

- 固定相机：机器人夹着方块移动时，方块在法兰系里的偏移固定，正运动学给法兰位姿，图像里按颜色找方块 → 1085 组 2D–3D 对应，RANSAC-PnP + 鲁棒最小二乘，同时解相机位姿和方块偏移。重投影中位误差 **6.2 px**；一个位姿适用于全部 50 条。相机在基座系的位置 (−0.035, −0.252, 0.685) m。
  **独立验证**：用这个外参把深度图投到基座系，桌面落在 z = 0 ± 5 mm。
- 腕部相机：抓取前方块不动，其位置 = 抓取时刻法兰位姿 × 方块偏移；加上夹爪全开时两指尖在图像里的固定像素作为近场约束。相机在法兰系的位置 (0.073, −0.005, 0.085) m，方块重投影中位误差 23.5 px（受方块位置约 1 cm 误差限制）。
  注意：相机离指尖只有约 10 cm，指垫 32 mm 宽在图像里纵向占约 170 px，轮廓顶角是指垫远端（x = +0.016 m），不是指垫中线；第一版用了中线，仿真手指比真机高约 90 px。

脚本：`scripts/xarm6/real/estimate_mount_extrinsics.py`、`estimate_wrist_extrinsics.py`
结果：`scripts/cfg/real/xarm6_{mount,wrist}_extrinsics.json`
仿真相机配置：`scripts/cfg/external_sensors/sim_xarm6_real.yaml`（front_cam = mount，wrist_cam = gripper，640×480）

## 3. 布局与物体

`scripts/xarm6/real/extract_real_layout.py` → `scripts/cfg/real/xarm6_real_layout.json`，每条 episode（基座坐标系）：

- 起始关节（真机每条从上一条结束的位置开始，没有回零）；
- 目标方块位置（抓取时刻正运动学 × 方块偏移）：x 0.12–0.45 m，y −0.17–0.20 m；
- 箱子位姿：固定相机深度点云在箱壁高度带内的最小外接矩形，再用"松开点必须在箱内"修正只拍到一部分箱子的情况。四组摆放：(0.33, −0.23, 0°)、(0.25, −0.29, 0°)、(0.25, −0.40 ~ −0.43, 0–30°)、(0.345, −0.38, 29°)；
- 干扰方块（另一颜色）位置。

**物体尺寸**：

- 真箱子就是重建出来的那个工具箱，尺寸约为仿真的 **0.70 倍**（外廓 0.32 × 0.24 × 0.083 m）。0.58 倍（按内沿角点量）在渲染对比里明显偏小，因为重建网格的卡扣、把手撑大了包围盒；
- 真机箱盖是透明的，铰链在**长边**，打开后向外倒着靠在铰链限位上；SimFoundry 第 9 步（铰链估计）把铰链放在了**短边**。已直接修正 `s9_articulate_objects/.../toolbox/results/mobility.urdf` 的关节（长边顶沿、绕长边转、范围 0–165°），并重跑第 11–14 步生成新资产（原文件备份在 `Data/cube2_scene/backup_shortedge_lid_20261017/`）。盖子是真实的带碰撞的关节连杆：每条 episode 随机打开到 125–165°，并把这个角度设成上限（限位），盖子在重力下靠在限位上；机械臂如果撞到盖子会真的把它推动，数据里能看出来。关节驱动刚度设为 0（真机盖子是自由铰链）。材质改成半透明淡蓝白（真机是透明塑料）。箱子旋转 180°，让内部窄格的位置和真机一致；
- 方块 25 mm（深度图测得方块顶面 z = 0.025 m；夹爪读数 210 也对应约 21 mm + 指垫压缩），仿真方块按比例缩到 25 mm，颜色改成纯色（重建的黄块贴图发白且有绿斑）。

**桌面环境**：浅色木纹（只用 light_oak / ash / beech 贴图），桌子远边 x = 0.55 m，左侧 y > 0.30 m 是黑色光学面包板，桌外是深色地毯；指尖改成黑色（真机是黑色）。

`scripts/xarm6/real/compare_sim_real.py` 把仿真设成真机某一帧的关节角和布局，渲染两台相机，和真机图像并排 + 叠加：机械臂、箱子、方块、指尖在两路图像里都基本重合。

## 4. 仿真数据生成（真机格式）

```bash
OMNIGIBSON_HEADLESS=1 mamba run -n simfoundry python scripts/xarm6/generate_scripted_demos.py \
    --scene-json Data/cube2_scene/s14_og/reconstructed_og_scene.json --task cube2_pick_place_cube_in_box \
    --out-dir Data/cube2_scene/xarm6_demos_real_0 --n-demos 80 --seed 100 \
    --real-layout scripts/cfg/real/xarm6_real_layout.json --action-freq 20 --speed 0.08 --gripper-rate 0.059 --p-wood 0.9
```

和以前的生成相比（`--real-layout` 打开）：

| 项 | 以前 | 现在（对齐真机） |
|---|---|---|
| 频率 | 30 Hz | 20 Hz |
| 末端速度 | 0.15 m/s | 0.08 m/s ±20%（真机中位数 0.068） |
| 布局 | 重建位姿附近 ±8 cm | 从真机 50 条的箱子/方块位置采样 + 噪声 |
| 起始/结束姿态 | 固定 home ±0.05 | 真机起始关节 ±0.03；结束时回到另一条真机的起始姿态 |
| 开始前 | 立即动 | 静止 5–30 步 |
| 夹爪指令 | 0/1 跳变 | 每步 ≤0.059（=50 原始单位），和真机遥操作一样是斜坡 |
| 投放点 | 箱子中心上方 | 箱内任意位置、箱沿上方 2–10 cm（真机松开点分布） |
| 相机 | 猜的位姿 ±3 cm/±3° | 标定位姿 ±1 cm/±1°，焦距 ±2% |
| 输出 | mp4 + npz | `epXXXXXX/episode.h5`，键名和真机完全一致（深度除外），attrs 里存 prompt / 布局 / 随机化参数 |
| 碰盖检测 | 无 | 每步记录 `sim_lid_angle_deg`、`sim_box_shift_m`、`sim_box_dyaw_deg`；`episodes.jsonl` 里的 `disturbance` 给出盖子最大角度变化和箱子最大位移，机械臂碰到盖子或推动箱子一眼可见 |

测试：3/3 成功，长度 325–470 步（真机中位数 453）；夹住方块时仿真夹爪读数 224，真机 223。

## 5. 转 LeRobot（真机和仿真同一个脚本）

```bash
.cache/pi05-serve-venv/bin/python scripts/xarm6/convert_real_h5_to_lerobot.py \
    --src /home/tingting/Documents/data/autoalloc_real Data/cube2_scene/xarm6_demos_real_0 ... \
    --exclude 8 --root Data/cube2_scene/lerobot/<name> --repo-id local/<name> --relative-action-chunk 50
```

- `observation.images.front` ← mount，`observation.images.wrist` ← gripper；
- `observation.state` = [当前关节, 1 − 夹爪实际/850]，`action` = [期望关节, 1 − 夹爪指令/850]（0 开 … 1 合）；
- 真机 prompt：`Pick up the {blue|yellow} cube and place it into the box.`；仿真用文件里存的 prompt；
- `--exclude` 只对真机文件生效；`meta/sources.jsonl` 记录每条来自真机还是仿真。

## 6. 评测（对齐后的仿真）

```bash
OMNIGIBSON_HEADLESS=1 mamba run -n simfoundry python scripts/xarm6/eval_pi05_xarm6.py \
    --scene-json Data/cube2_scene/s14_og/reconstructed_og_scene.json --task cube2_pick_place_cube_in_box \
    --out-dir Data/cube2_scene/s15_eval/<name> --prompt "Pick up the {color} cube and place it into the box." \
    --real-layout scripts/cfg/real/xarm6_real_layout.json --action-freq 20 --randomize --grip-mode raw --episodes 10
```

真机格式数据训练的模型，夹爪状态要用开口换算（脚本在 `--real-layout` 下自动处理），夹爪动作是斜坡值，建议 `--grip-mode raw`。

## 7. 还没对齐 / 已知差异

- 仿真不写深度（pi0.5 不用深度）；
- 资产里箱体和箱盖都是 1 kg，盖子向外倒 150° 时整个箱子会翻倒（倾斜 41°）；现在加载后设为箱体 0.8 kg、箱盖 0.2 kg（估计值，没有称重），箱子稳定；半透明效果靠 RTX 的 fractional cutout opacity，反光不如真机；
- 背景：真机桌外有人、椅子、线缆，仿真只有一块深色地毯；
- 光照：仿真偏亮、阴影弱，靠随机化覆盖；
- 机械臂颜色：仿真偏灰，真机偏白；
- 少数真机条目的干扰方块检测错了（检测到夹爪上的蓝色指示灯），只影响干扰方块的采样分布。
