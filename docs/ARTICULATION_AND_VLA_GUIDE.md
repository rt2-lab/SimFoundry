# Articulation 重建 + VLA 仿真测试 指南

记录 2026-10-13 这次会话里踩过的坑和验证过的做法，供下次（新视频）直接照着跑，不用重新摸索。

当前状态：`Data/cube_closed` 场景的 stage 1-14（不含 articulation）已完整跑通，机械臂/6 个方块/工具箱都在。Stage 9（`--detect-articulation`）反复调试到接近跑通，最后卡在 Qwen 多图推理的退化重复问题上——本文档末尾给出了诊断结论和下次应该直接带上的修复参数，没有重新验证通过，下次跑之前请先小范围验证一次。

---

## 1. 必需的环境变量

每次跑 reconstruction 或 articulation 之前，先 export 这一整套（没有 Gemini/Vertex 付费额度，全部走本地 Qwen3.5-4B + FLUX.1-Kontext-dev）：

```bash
export PATH="/home/tingting/miniforge3/condabin:/home/tingting/miniforge3/bin:$PATH"
export CONDA_EXE=/home/tingting/miniforge3/bin/conda          # 见下方"坑 1"，必须显式设置
export PYOPENGL_PLATFORM=egl                                   # 见下方"坑 3"，pyrender 离屏渲染必需

export SIMFOUNDRY_TEXT_VLM=qwen
export SIMFOUNDRY_QWEN_MODEL_PATH=/home/tingting/Documents/allocator/models/Qwen3.5-4B
export SIMFOUNDRY_QWEN_PYTHON=/home/tingting/Documents/SimFoundry/.cache/radio-prep-venv/bin/python
export SIMFOUNDRY_QWEN_MAX_TOKENS=4096
export SIMFOUNDRY_QWEN_QUANTIZATION=nf4
export SIMFOUNDRY_QWEN_REPETITION_PENALTY=1.1                   # 见下方"坑 5"，articulation 的合并步骤必须加；1.05 已验证不够（2026-10-13）
export SIMFOUNDRY_QWEN_TEMPERATURE=0.7
export SIMFOUNDRY_QWEN_TOP_P=0.8

export SIMFOUNDRY_FLUX_QUANTIZATION=nf4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## 2. 已知的坑（按本次踩坑顺序）

1. **`CONDA_EXE` 未设置时，conda 路径解析会解析到错误的嵌套安装**
   `simfoundry` conda env 自己装了一份 `conda`/`mamba`。在该 env 里调用裸的 `conda info --base` 或 `conda run --prefix ...`，会解析到 `.../envs/simfoundry` 而不是真正的 `/home/tingting/miniforge3`，导致：
   - `get_conda_env_path()` 找不到 `articulate-anything-hunyuan` 等 env（`scripts/pipeline/A_reconstruction/stages/9_articulate_objects.py`）
   - `run_articulation()` 里的 `conda run --prefix` 实际上用了错的 Python 解释器去跑目标命令，导致 `ModuleNotFoundError: No module named 'articulate_anything'`（即使 `pip show` 显示包明明装着）

   **已修复**：显式 `export CONDA_EXE=/home/tingting/miniforge3/bin/conda`，并且 `9_articulate_objects.py` 的 `run_articulation()` 已改为优先用 `CONDA_EXE` 而不是裸 `conda`。

2. **主机内存（30GB）比显存（32GB）更紧张，FLUX / Hunyuan 的默认"省显存"选项反而有害**
   这台机器上其他用户（leo/bobby）的常驻进程占掉了相当一部分主机内存，swap 经常接近打满。`enable_cpu_offload=True`（FLUX 默认）、`s7_mesh.low_vram=true`（Hunyuan 的省显存模式）都会把模型主体堆进**主机内存**，在这台机器上触发 OOM-kill，而显存其实很空。

   **已修复/建议**：
   - `simfoundry/models/vlm.py` 的 `FLUX1` 类：`enable_cpu_offload=False` 时补上 `self.pipeline.to(device=self.device)`（原来这个分支是空的，会导致 VAE/CLIP 留在 CPU、报 dtype 不匹配）
   - `5_decompose_scene.py` / `6_upsample_object_images.py`：构造 `FLUX1(...)` 时都传 `enable_cpu_offload=False`
   - 跑 stage 7 mesh 生成时**不要**传 `s7_mesh.low_vram=true`，前提是显存有 25GB+ 空闲（必要时跟同机的其他人协调一下显存，比口头沟通更靠谱）

3. **pyrender 离屏渲染在这台机器上默认失败：`Pyrender offscreen failed: Cannot connect to "None"`**
   没有 `DISPLAY`，pyrender 默认走 GLX 会直接失败。机器上装了 EGL（`libEGL_nvidia.so` 和 `libEGL_mesa.so` 都在）。

   **已修复**：`export PYOPENGL_PLATFORM=egl`。

4. **trimesh 场景图的节点名和几何体名可能不一致，导致 `No path from world->xxx.001!`**
   `deps/articulate-anything/simfoundry/render.py` 里 `render_scene_with_labels()` 直接拿 `loaded.geometry.keys()` 的名字去查 `loaded.graph[...]`，但 trimesh 在只有一个几何体时，有时会给几何体 dict 的 key 加 `.001` 后缀、而场景图节点名不加，两者对不上。

   **已修复**：改成先查 `loaded.graph.geometry_nodes[geom_name]` 拿到真正的节点名，再去索引 `loaded.graph[node_name]`（`deps/articulate-anything/simfoundry/render.py`）。

5. **（未完全解决）Qwen 做"12 张图综合分析+合并"这类长输出任务时，容易陷入退化重复**
   Stage 9 的"合并部件"步骤（`merge_mesh_parts`）要求模型依次分析 12 张分割标注图再给出合并结果，这种长结构化输出对 4B 模型不友好。诊断发现：**不是 `max_tokens` 不够**（调到 20000 依然截断），而是模型在原地重复同一句话（"绿色是底座、蓝色是盖子、紫色是侧边把手"无限循环），根本不会自然结束——调大 `max_tokens` 只会让它循环更久，纯属浪费时间。

   直接验证方法（已用这个脚本复现过问题）：
   ```bash
   export PATH="/home/tingting/miniforge3/condabin:/home/tingting/miniforge3/bin:$PATH"
   export SIMFOUNDRY_QWEN_MODEL_PATH=/home/tingting/Documents/allocator/models/Qwen3.5-4B
   export SIMFOUNDRY_QWEN_PYTHON=/home/tingting/Documents/SimFoundry/.cache/radio-prep-venv/bin/python
   export SIMFOUNDRY_QWEN_QUANTIZATION=nf4
   mamba run -n articulate-anything-hunyuan python3 -c "
   import sys
   sys.path.insert(0, '/home/tingting/Documents/SimFoundry/deps/articulate-anything/simfoundry')
   sys.path.insert(0, '/home/tingting/Documents/SimFoundry')
   from articulate_anything.utils.prompts import merge_parts
   from simfoundry.models.local_qwen import LocalQwen
   from pathlib import Path
   import glob
   images = sorted(glob.glob('<场景目录>/s9_articulate_objects/<scene>/<object>/s3_segment_mesh/rendered_parts/render_view*.png'))[:12]
   images = [str(Path(p).resolve()) for p in images]
   user_prompt, system_prompt = merge_parts('<object_name>', ['part_a', 'part_b'])
   qwen = LocalQwen()
   request = {
       'prompt': system_prompt + '\n\n' + user_prompt,
       'image_inputs': __import__('simfoundry.models.remote_cache', fromlist=['image_digests']).image_digests(images),
       'temperature': 0, 'top_p': 0, 'seed': 0,
       'max_tokens': 4096, 'max_side': qwen.max_side,
       'quantization': qwen.quantization, 'model_identity': qwen._model_identity(),
       'enable_thinking': False, 'adapter_version': 2,
       'repetition_penalty': 1.05,   # <-- 换成 1.0 能立刻复现"无限重复"的问题
   }
   result = qwen._infer(dict(request, model_path=qwen.model_path, image_paths=images))
   print('finish_reason:', result.get('finish_reason'), '| length:', len(result.get('text','')))
   "
   ```
   **（2026-10-13 已在 `cube2_scene` 上完整验证，更新结论）`repetition_penalty=1.05` 不够，必须用 `1.1`。** 用本节上面的独立诊断脚本，带着 `cube2_scene` 真实跑出来的 12 张 `rendered_parts` 图复现过一次：

   - 关键坑：`local_qwen.py` 里 `temperature`/`top_p` 是 `os.environ.get("SIMFOUNDRY_QWEN_TEMPERATURE", temperature)` 这种写法——**环境变量会无条件覆盖调用方显式传入的值**，包括 stage 5 物体检测那种特意传 `temperature=0, top_p=0` 想要确定性输出的调用点。所以只要本文档第 1 节的环境变量块（`SIMFOUNDRY_QWEN_TEMPERATURE=0.7`、`SIMFOUNDRY_QWEN_TOP_P=0.8`）是 export 着的，stage 9 合并部件这一步实际跑的就是 temperature=0.7/top_p=0.8 的采样解码，不是贪婪解码。诊断脚本按这个真实配置复现：
     - `repetition_penalty=1.05`（配合 temperature=0.7/top_p=0.8，seed=0）：`finish_reason=length`，没有逐字复读同一句话，但对 12 张图的逐图分析写得过于啰嗦，4096 token 预算在走到 `part_segment_dict`之前就耗尽了——是"卡在没完没了的描述里" 而不是"复读同一句话"，但表现（不收敛、截断）跟文档最初描述的退化是一回事。
     - `repetition_penalty=1.1`（其余参数不变）：`finish_reason=stop`，输出完整，`part_segment_dict` 正常生成。**已确认可用，下次直接把第 1 节的 `SIMFOUNDRY_QWEN_REPETITION_PENALTY` 改成 `1.1`。**
   - 因为 `seed=0` 固定、`temperature>0` 时内部仍然 `set_seed(seed)`，这个退化是**确定性复现**的，不是偶发的；原地重跑（不改参数）不会自愈，不要盲目重试。
   - 如果以后哪次用不同物体又在 1.1 下卡住，再试 1.15；如果还不行，参照本节诊断脚本直接查看生成文本判断是"真复读"还是"预算不够的啰嗦"，这两种现象的处理思路不一样（前者应该继续调高 repetition_penalty，后者也可以考虑让 prompt 要求更简洁的逐图描述，而不是一味加 max_tokens——加 max_tokens 本身不改变模型会不会啰嗦/复读，只是给它更多空间啰嗦）。

6. **某个物体的 articulation 工作目录如果是从崩溃的尝试里留下的半成品，`--skip-successful` 会误判"已完成"**
   `cfg.rerun` 的完成检测是按"某个中间产物文件是否存在"判断的，不会检查更后面步骤需要的最终产物是否也生成了。崩溃在渲染标注图这一步中途的尝试，会让 Step 3（分割网格）被误判为"已完成"从而跳过重跑，直接进 Step 4 却发现找不到图片。

   **解决办法**：如果某个物体反复失败，直接整个删掉它的工作目录重来，不要依赖缓存续跑：
   ```bash
   rm -rf "Data/<scene_name>/s9_articulate_objects/<scene_name>/<object_name>/"
   ```

7. **（2026-10-13 新发现）Stage 5 的"物体检测"调用即使有 `torch.cuda.empty_cache()` + `gc.collect()` 兜底，仍可能在显存紧张的场景里 OOM** —— `cube2_scene`（2 个方块）在 stage 5 跑到第 3 次检测调用时触发：

   ```
   torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total
   capacity of 31.35 GiB of which 87.25 MiB is free. Process <pid> has 26.77 GiB memory in use.
   ```

   根因：stage 5 这一个 Python 进程里同时常驻 SAM3（分割）+ 深度合成模型（DepthPro/DA3 + PriorDepthAnything）+ FLUX（nf4 量化的 removal_model），每轮迭代都要用到，不能卸载；"检测物体类别"这一步又会 fork 一个独立子进程去跑 Qwen3.5-4B（代码里 `5_decompose_scene.py:1224` 附近的注释已经解释过这个子进程为什么要独立跑）。`empty_cache()`/`gc.collect()` 只能回收"缓存但未占用"的显存块，而上面四个模型本身的常驻显存就已经把 32GB 卡几乎打满，子进程的 Qwen 连 20MB 都挤不出来——从日志看子进程已经把模型权重 100% 加载完，只是在 vision tower 前向传播时差了最后一点,说明缺口其实不大。

   **已修复**：给 `LocalQwen.__call__` 加了一个可选的 `max_side` 参数（`simfoundry/models/local_qwen.py`），不传时行为不变（仍用 `self.max_side`，默认来自 `SIMFOUNDRY_QWEN_MAX_SIDE`）。Qwen 的 vision tower 显存消耗跟图像边长大致是平方关系，图小一半、激活显存大约减半。`5_decompose_scene.py` 里物体类别检测这一次调用改成显式传 `max_side=768`（原图检测单张图足够用了，不影响方块/箱子这种大物体的 2D 框精度），其余调用点（比如 stage 9 合并部件那种需要更多细节的多图场景）不受影响、仍用默认 1024。用 `hasattr(vlm_pro, "max_side")` 做门控，跟旁边已有的 `getattr(vlm_pro, "bbox_format", ...)` 一个套路，这样 Gemini 后端（不认识 `max_side` 这个参数）不会被这个改动影响到。

   **如果换一台显存更紧张的机器，或者这个改动后还是偶尔 OOM**：可以把 768 再调小（512 应该还够用），或者在 `create_text_vlm`/调用处加一层"失败后自动降分辨率重试一次"的逻辑——但目前这次改动已经足够让 cube2_scene 的三次检测调用都通过。

---

## 3. 怎么跑 reconstruction + articulation

推荐流程：先跑完整条 pipeline（不开 articulation），确认 stage 1-14 都成功，再单独补跑 stage 9。

```bash
cd /home/tingting/Documents/SimFoundry
# ... 第 1 节的环境变量 ...

# 第一步：先跑主流程（不含 articulation）
bash scripts/pipeline/A_reconstruction/run.sh \
  --scene-name <新场景名> \
  --video-fpath <新视频路径.mp4> \
  --env-da3 simfoundry --no-stream \
  -- \
  s5_scene.removal_model=flux s5_scene.use_upsampled_source_image=false \
  s6_upsample.model=flux s6_upsample.include_invalid=false

# 第二步：确认 nvidia-smi / free -h 都有余量后，补跑 articulation
bash scripts/pipeline/A_reconstruction/run.sh \
  --scene-name <新场景名> \
  --video-fpath <新视频路径.mp4> \
  --env-da3 simfoundry --no-stream --skip-successful --detect-articulation \
  -- \
  s5_scene.removal_model=flux s5_scene.use_upsampled_source_image=false \
  s6_upsample.model=flux s6_upsample.include_invalid=false \
  s9_articulate_objects.tree_model=qwen \
  s9_articulate_objects.merge_model=qwen \
  s9_articulate_objects.s5_model=qwen
```

开跑前检查清单：
- `nvidia-smi --query-gpu=memory.used,memory.total --format=csv`：Hunyuan mesh 全精度生成大约需要 25-29GB 空闲显存
- `free -h`：至少留 10GB available 再开始，stage 7 加载 checkpoint 那一下会有瞬时 ~28GB 的主机内存尖峰（属于正常现象，只要能回落就没事）
- stage 5 的物体检测如果因为中途重启触发过 VLM 幻觉（`obj_cat_list/iter_N.json` 里出现不存在的物体名），用 `s5_scene.force_categories=[...]` 锁定正确类别列表，跳过检测调用，例如：
  ```
  's5_scene.force_categories=["purple cube / purple block / cube","toolbox / tool kit / plastic box", ...]'
  ```

---

## 4. 拍新视频时需要注意的点（针对 articulation）

- 整条重建流程本质上只用**一帧静态画面**建模（深度、分割、mesh、位姿全部基于自动选中的那一帧），所以视频里箱子保持"合上"状态拍摄最稳妥——不需要专门拍"开合"的动作过程。
- Articulation（stage 9）对关节（铰链位置、开合范围）的估计，是让 VLM 看**一张静态图+mesh**去推断/生成候选方案，再跑 actor-critic 迭代打分，**不是从视频动作里学出来的**。所以"录一段开箱子的视频"对 articulation 本身没有帮助，纯静态的"合上"视频就够用。
- 如果重建出来的关节不准（轴向、开合范围、支点位置），正确的修正方式是跑交互式关节修正界面，而不是重新拍视频：
  ```
  s9_articulate_objects.interactive_joint_refinement=true
  ```
  这个会打开一个 web UI，带实时 3D 预览，手动调整 joint type / axis / pivot / limits，结果写到 `physics_overrides.json`，后续 stage 会优先采用人工修正值。

---

## 5. 机械臂摆放

`cube_closed` 场景的 stage 14 默认会插入一个 FrankaPanda 机械臂（场景里叫 `robot0`），位置 `[-1.0, 0, 0]`，朝向是默认朝向（**没有现成的"自动转向任务物体"功能**，只有相机有 look-at，机器人底座朝向需要手动算）。

重新摆放机械臂，用浏览器版 light editor（推荐，所见即所得）：

```bash
mamba run -n simfoundry-editor python scripts/interactive/light_editor/server.py \
  --scene /home/tingting/Documents/SimFoundry/Data/<scene_name>/s14_og/reconstructed_og_scene.json \
  --cameras nv_franka_droid
```

打开 `http://localhost:8770`，选中机器人，Transform 面板里改位置（米）/旋转（度），保存会生成一个新的 `*_light_edit_*.json`（不会覆盖原始场景文件）。

也可以直接改 JSON（`AGENTS.md` 里记录的字段）：
```
state.registry.object_registry["robot0"].root_link.pos   # [x, y, z]
state.registry.object_registry["robot0"].root_link.ori   # 四元数 XYZW
```
改完位置记得把 `lin_vel`/`ang_vel` 清零（避免残留速度导致物理启动瞬间漂移）。

---

## 6. 在仿真里测试 VLA

### 6.1 整体流程

策略评估走 `C_application` 的 stage 15（`1_eval_policy_og_scene.py`），支持两种策略后端（`s15_eval.policy`）：
- `gr00t`（默认）
- `openpi`（即 pi0.5 系列 VLA；`/home/tingting/Documents/allocator/models/pi05_base` 应该就是这个）

**关键点**：策略模型本身跑在**独立的 server 进程**里（`openpi_client.websocket_client_policy.WebsocketClientPolicy`），仿真这边（`OpenPIClient`）只是个 websocket client 去连接它，不会在 OmniGibson 进程里直接加载 checkpoint。所以：

1. 先在 openpi 那边（通常是单独的 conda env/项目）启动一个策略 server，指向你要测的 checkpoint，监听端口（openpi 默认 `8000`，gr00t 默认 `5555`）。
2. 再跑仿真这边的 eval，指定 host/port 去连接：

```bash
bash scripts/pipeline/C_application/run.sh \
  --mode eval \
  --scene-name <scene_name> \
  --root-dir /home/tingting/Documents/SimFoundry/Data \
  -- \
  s15_eval.policy=openpi \
  s15_eval.host=localhost \
  s15_eval.port=8000 \
  s15_eval.prompt="put the colored cubes into the box" \
  s15_eval.n_episodes=20
```

### 6.2 常用参数（`scripts/cfg/real2sim_cfg.yaml` 的 `s15_eval` 块）

| 参数 | 说明 |
|---|---|
| `execute_horizon` | 每次策略推理后连续执行几步动作（默认 8） |
| `action_freq` | 动作频率 Hz（默认 15） |
| `timeout_s` | 单局最长时长（默认 90s） |
| `save_video` / `video_fps` | 是否保存评估视频，便于回看 |
| `init_states_path` | 固定一批初始状态，用于可复现对比实验 |
| `scene_json` | 不想直接用 `s14_og` 的输出时，指向手动编辑过的场景 JSON（比如 light editor 导出的 `*_light_edit_*.json`） |
| `asset_id` | 动作空间归一化方式：`franka`（关节位置）或 `droid`（关节速度），要跟训练 checkpoint 用的一致 |
| `base_camera_1_name` / `base_camera_2_name` | 外部相机名（对应 DROID 的 exterior_image_1/2），需要跟 `external_sensors_cfg` 里定义的相机名对上 |

### 6.3 任务目标定义（"什么算成功"）

任务目标（谓词）走 `simfoundry/tasks/pick_place_task.py` + 一个 YAML 配置，可以照抄模板改：
```
scripts/cfg/task/droid/droid_desk_put_cup_in_bowl.yaml
```
主要字段：
- `semantic_group_mapping`：把场景里的物体名映射到语义组（比如 `{robot: [agent], cube: [purple_cube, orange_cube, ...], box: [toolbox]}`）
- `goal_predicates` / `init_predicates`：用 `InsideAABB`、`IsGrasping`、`Touching` 等谓词定义初始/目标状态
- `workspace_bounds`：机械臂活动范围限制
- `robot_pose`：episode reset 时的机器人位姿覆盖（跟第 5 节的场景级摆放是两回事，这个是每局重置时用的）

跑的时候用 `task=load_scene task.task_name=<你的task文件名>` 指定（参考 `scripts/cfg/real2sim_cfg.yaml` 里 `task_name` 字段的注释）。

**注意**：如果箱子最终还是刚体（没做成功 articulation），任务目标就只能设计成"把方块放进已经开着的格子里"这类，不能要求机械臂去开合箱盖——因为刚体没有关节可动。

## 7. xArm6（真机同款）接入、zero-shot 测试与脚本化 demo

### 7.1 机器人资产
- URDF：`assets/robots/xarm6/clean_urdf.py` 由官方 `xarm_description`（xacro，`dof:=6 add_gripper:=true`）生成的 URDF 转换而来（去掉无几何的 frame link、STL→OBJ、新增 `eef_link` = UFactory TCP（gripper base +0.172 m）、右侧 mimic 改挂 `right_outer_knuckle_joint`）。**导入器会原地改写 URDF，每次导入前先重新生成。**
- 导入：`OMNIGIBSON_HEADLESS=1 mamba run -n simfoundry python scripts/assets/import_custom_robot.py --config assets/robots/xarm6/import_xarm6.yaml --dataset-root $PWD/assets/og_custom`，然后 `mamba run -n simfoundry-editor python scripts/assets/fix_xarm6_usd.py --usd assets/og_custom/objects/robot/xarm6/usd/xarm6.usda`（补 `right_inner_knuckle_joint` 的 mimic 引用和关节限位）。
- 类：`simfoundry/robots/xarm6.py`（`XArm6`，`import_og_dependencies()` 注册）；检查：`scripts/assets/smoke_test_xarm6.py`。
- 约定（与 UFACTORY LeRobot 真机数据一致）：state / action = `[J1..J6 (rad, 绝对), gripper (0 开, 1 关 = drive_joint / 0.85)]`，30 Hz，相机 `front` + `wrist`。

### 7.2 场景
- Stage 14 用 xArm6 重建场景（机械臂放在方块和箱子之间，距离 ~0.55 m，在臂展内）：
  ```
  bash scripts/pipeline/A_reconstruction/run.sh --scene-name cube2_scene --video-fpath <video> --no-stream --include 14 -- \
    s14_og.robot_config.type=XArm6 '~s14_og.robot_config.end_effector' \
    's14_og.robot_config.position=[0.0,-0.45,0.0]' 's14_og.robot_config.orientation=[0,0,0.70711,0.70711]' \
    's14_og.robot_config.obs_modalities=[rgb,proprio]' s14_og.external_sensors_cfg=sim_xarm6_front_wrist
  ```
- **坑：articulation 修好后必须重跑 stage 11-14。** 之前 hinge 修复（`mobility.urdf` 改成 revolute）发生在 stage 10-14 之后，USD 里盖子还是 fixed joint；而 `PickPlaceTask.group_init_joint_positions` 对没有关节的物体会**静默跳过**，所以"盒子已打开"这个前提一直没生效。现在 `simfoundry/robots/xarm6_env.py::articulated_joint()` 找不到关节会直接报错。
- 检查：`scripts/xarm6/check_scene.py`（盖子 reset 后是否打开并保持、物体是否 IK 可达、相机画面）。当前 cube2_scene：盖子 reset 后 1.571 rad（上限）并保持，两个方块和箱口都可达。

### 7.3 Zero-shot（ys903/pi05_xarm6_pick_place_50ep）
```
# 服务端（tmux）
.cache/pi05-serve-venv/bin/python simfoundry/policies/pi05_serve/server.py --checkpoint checkpoints/ys903_pi05_xarm6 --device cuda --port 8420
# 评测（tmux）
OMNIGIBSON_HEADLESS=1 mamba run -n simfoundry python scripts/xarm6/eval_pi05_xarm6.py \
  --scene-json Data/cube2_scene/s14_og/reconstructed_og_scene.json --task cube2_pick_place_cube_in_box \
  --out-dir Data/cube2_scene/s15_eval/xarm6_ys903 --prompt "Pick up the blue block and drop it into the box." --episodes 5
```
- `checkpoints/ys903_pi05_xarm6/config.json` 删掉了 lerobot 0.6.1 不认识的字段（均为关闭/默认值，原文件存为 `config.orig.json`）；server 会自动使用 checkpoint 自带的 `tokenizer/`。

### 7.4 抓取：物理抓取（已修好），GraspLatch 只作备用
- **根因**：URDF 导入器把 xArm6 所有关节驱动写成 `acceleration` 类型，刚度会乘上连杆惯量；夹爪零件只有 20-50 g，所以手指空中开合正常，夹住方块时只有 ~1e-3 N·m，一抬就滑掉。另外 mimic 关节被导入成软弹簧（naturalFrequency 25、dampingRatio 0.005），受力时手指会偏离四连杆。
- **修复**（`scripts/assets/fix_xarm6_usd.py`，导入后必跑）：所有驱动改成 `force` 类型；mimic 改成刚性（naturalFrequency 0）。夹爪 kp=30（`simfoundry/tasks/macros.py` 的 `GAINS["XArm6"]`）。测试：12/12 物理抓起，夹持时夹爪读数 ~0.5（真机 ~0.6），机械臂跟踪误差 < 0.02 rad。
- 抓取高度：TCP 在方块底面上方 6 mm（张开时指尖在 eef_link 上方 ~1 cm，闭合时下摆）。
- 试过但放弃：单电机（右侧外指节也做 mimic）——刚性 mimic 链挂在一个驱动上会失稳。
- `GraspLatch`（`--grasp latch`）只作调试备用。注意：挂上后 OmniGibson 会冻结夹爪开度，若在手指碰到物体前挂上，方块会"悬浮"在张开的手指之间——第一批 100 条 demo 就是这样，已删除重做。

### 7.5 脚本化 demo → LeRobot 数据集 → finetune
```
OMNIGIBSON_HEADLESS=1 mamba run -n simfoundry python scripts/xarm6/generate_scripted_demos.py \
  --scene-json Data/cube2_scene/s14_og/reconstructed_og_scene.json --task cube2_pick_place_cube_in_box \
  --out-dir Data/cube2_scene/xarm6_demos --n-demos 100 --seed 1
.cache/pi05-serve-venv/bin/python scripts/xarm6/convert_demos_to_lerobot.py \
  --demo-dir Data/cube2_scene/xarm6_demos --root Data/cube2_scene/lerobot/xarm6_cube_in_box_sim --repo-id local/xarm6_cube_in_box_sim \
  --relative-action-chunk 15
bash scripts/xarm6/finetune_pi05.sh Data/cube2_scene/lerobot/xarm6_cube_in_box_sim outputs/pi05_xarm6_sim_rel checkpoints/pi05_droid_jointpos
```
- 基座 `pi05_droid_jointpos` 预测的是**相对当前关节的增量**（夹爪绝对），所以训练开 `use_relative_actions`，转换数据时用 `--relative-action-chunk 15`（= chunk_size）重算相对空间的 action stats。本地 checkpoint 的 pre/postprocessor JSON 里补了（默认关闭的）`relative_actions_processor` / `absolute_actions_processor`，原文件存为 `*.orig.json`。
- 评测 DROID 系模型：`eval_pi05_xarm6.py --camera-keys droid`（front/wrist → base_0_rgb/left_wrist_0_rgb）；server 输出已转回绝对关节角。
- 磁盘：每个 checkpoint ~11 GB（模型 8.8 + 优化器 2.1），保存时新旧两个会同时存在。
- 每条 demo：随机化两个方块（±8 cm、随机 yaw）和箱子（±4 cm）、初始关节 ±0.05 rad、速度 ×0.8–1.2、路径点抖动；先确认盖子打开；IK 逐步跟踪（`simfoundry/robots/xarm6_kinematics.py`，与仿真 FK 完全一致）；只保留方块最终在箱内的 episode。prompt 在三种说法里随机，颜色对应目标方块。
