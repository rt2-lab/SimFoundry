# CHTC package: task-space (TCP) actions, real + sim co-training

Separate from `../chtc/` (joint-space rounds 1–4 and real-only). Same container image
(`tdu35/simfoundry_lerobot_train:0.6.1`), same staged base (`pi05_base_base.tar`); its own job
directory on the access point (`~/simfoundry_chtc_eef`) and W&B project (`pi05_xarm6_eef`).

| File | Role |
|---|---|
| `stage_and_submit.sh` | Lab machine: stream the dataset to `/staging` (once), copy job files, `condor_submit` |
| `train_eef.sub` | HTCondor submit file (80 GB GPU, 64 GB RAM, "medium" 24 h) |
| `run_train_eef.sh` | Job wrapper: unpack inputs, CUDA self-test, train, bundle checkpoints (also on eviction) |
| `finetune_eef.sh` | The training command (relative actions, LR decays over exactly `STEPS`) |
| `train_cotrain.py` | `lerobot-train` with the real episodes oversampled `REAL_REPEAT`× |

```bash
# 1. dataset (reuses the joint-space dataset's videos; ~1 min)
.cache/pi05-serve-venv/bin/python scripts/xarm6/make_eef_dataset.py \
    --src Data/cube2_scene/lerobot/xarm6_real_sim --root Data/cube2_scene/lerobot/xarm6_real_sim_eef
# 2. upload + submit (ssh ControlMaster to ap2002 / transfer must be open, or you get password + Duo)
bash scripts/xarm6/chtc_eef/stage_and_submit.sh xarm6_eef_cotrain_16k
# 3. fetch one step without a local tar
ssh tdu35@transfer.chtc.wisc.edu 'cat /staging/t/tdu35/simfoundry/checkpoints/xarm6_eef_cotrain_16k.tar' \
    | tar -xf - -C outputs/chtc xarm6_eef_cotrain_16k/checkpoints/016000/pretrained_model
# 4. real arm
.cache/radio-prep-venv/bin/python scripts/xarm6/real/deploy_pi05_xarm.py --action-space eef --prompt "..." --dry-run
```

Knobs (env of `stage_and_submit.sh`): `STEPS` 16000, `BATCH` 32, `SAVE_FREQ` 4000, `REAL_REPEAT` 4,
`BASE_NAME` pi05_base, `DRY_RUN=1` (stage only).

## Representation

| | dims | content |
|---|---|---|
| action | 10 | TCP target `[x, y, z, r11, r21, r31, r12, r22, r32, gripper]` |
| observation.state | 16 | TCP current (same 9) + gripper + `J1..J6` |

- **TCP** = URDF `eef_link` (link6 + 0.172 m, fingertip centre), robot base frame. The target is
  FK of the recorded desired joints (xarm_stack IK output; equals the recorded desired pose to
  0.16 mm median), the current pose FK of the measured joints.
- **Rotation** = first two columns of R (6D): continuous, no quaternion double cover (the recorded
  quaternions flip sign between consecutive messages). Decoded by Gram-Schmidt.
- **Relative**: LeRobot's stock `use_relative_actions` subtracts `state[:10]` from every action of
  the 50-step chunk except the gripper, so the model predicts TCP displacement (m, base frame) and
  rotation-column change relative to the chunk's first observation. Stats are computed in that
  space (`recompute_stats(relative_action=True, chunk_size=50)`), quantile-normalised.
- **Joints stay in the state** (not the action) so the model still sees the arm configuration.
- **Gripper** stays absolute, `1 - raw / 850`.

## Data mix

497 episodes: 50 real (22,772 frames) + 447 aligned sim (182,911). With `REAL_REPEAT=4` real
frames are 33% of every epoch instead of 11%. 16k steps × 32 = 512k samples ≈ 1.9 epochs of the
oversampled set, each real frame seen ~7.5×.

## Checks done before the first submit (2026-10-19)

- dataset: same 497 episodes / 205,683 frames / videos as `xarm6_real_sim`; TCP from FK matches the
  h5 `robot_desired_pose` (+0.172 m) to 0.16 mm median on real data, exact on sim; relative q01/q99
  match an independent computation to 0.002.
- sampler: 33.1 % real over 20k drawn samples.
- processors: action enters the model as (B, 50, 10) in [-1.07, 1.0]; relative → normalise →
  unnormalise → absolute round trip error < 1e-7.
- deploy decode: IK from the measured joints reproduces the recorded desired joints to 0.0005 rad.
- Not run locally: the model forward (loading pi05 needs > 30 GB RAM on this machine; the job
  requests 64 GB).
