# Deploying pi0.5 policies on the real xArm6

How to run the SimFoundry pi0.5 checkpoints on the lab's UFactory xArm6 + xArm Gripper (host
`heron`, arm at `192.168.1.111`) for the cube-into-box task: start the policy server, prepare the
arm and table, run a rollout, and read the results.

All commands run from the SimFoundry repo root:

```bash
cd /home/tingting/Documents/SimFoundry
PY=.cache/radio-prep-venv/bin/python        # deployment side: cameras, arm, client
SRV=.cache/pi05-serve-venv/bin/python       # policy server: LeRobot + pi0.5
```

---

## 1. How it fits together

```
 D435i mount (front) ─┐
 D435i wrist ─────────┤   deploy_pi05_xarm.py  ──HTTP /infer──►  pi05 server (GPU, port 8420)
 xArm joints/gripper ─┘         │                                  returns a 50-step action chunk
                                │
                                ├─ eef models: TCP target ─► IK (nearest reachable, seeded from measured joints)
                                ▼
                   joint target ─► P velocity loop, 80 Hz: v = Kp (q_des − q), |v| ≤ vmax
                                   (same law as xarm_stack's joint_vel_controller)
                   gripper ─────► set_gripper_position(850 · (1 − g))
```

- **No ROS.** The script talks to the arm with the xArm Python SDK and reads the cameras with
  pyrealsense2. It reproduces the execution path the demonstrations were recorded with (SpaceMouse
  → simple_ik → joint_vel_controller), so the policy sees the same dynamics.
- **Asynchronous chunks.** The policy predicts 50 steps (2.5 s at 20 Hz). The next chunk is
  requested 10 steps into the current one and computed while the arm keeps moving; when it arrives,
  the steps that passed during inference are skipped and it is cross-faded in over 5 steps. The arm
  never stops for inference. (`--sync` restores the first deployment's behaviour, which paused about
  0.25 s every 0.75 s and looked jerky.)
- **No success detection.** After placing the cube the policy keeps acting, usually reaching for the
  other cube. A rollout ends at `--max-seconds` or on Ctrl-C.

---

## 2. Models

| Model | Data | Action space | Checkpoint | Results |
|---|---|---|---|---|
| Round 5, task space | 50 real (oversampled ×4) + 447 sim | `eef` | `outputs/chtc/xarm6_eef_cotrain_16k/checkpoints/016000/pretrained_model` | sim 12/20; real yellow 1/1 |
| Round 4, joint space | 50 real + 447 sim | `joint` | HF `TingtingDu/pi05-xarm6-real-sim-16k` | sim 12/20; real 3/5 |
| Real only, joint space | 50 real | `joint` | HF `TingtingDu/pi05-xarm6-real-only-8k` | sim 1/13; real not tested |
| Round 6a/6b, task space | sim only / real only | `eef` | CHTC `checkpoints/xarm6_eef_sim_16k.tar`, `xarm6_eef_real_8k.tar` | training (2026-10-20) |

**`--action-space` must match the model.** A task-space model takes a 16-D state and returns
10-D actions; a joint-space model takes 7-D and returns 7-D. If they do not match, the server errors
or the arm gets meaningless targets. The dry run (§5) catches this.

Download a Hugging Face model (public, no token, about 9.4 GB each; check `df -h /` first):

```bash
$SRV -c "from huggingface_hub import snapshot_download; \
snapshot_download('TingtingDu/pi05-xarm6-real-sim-16k', local_dir='outputs/hf/pi05-xarm6-real-sim-16k')"
```

Pull one step of a CHTC checkpoint without writing the 37 GB tar locally (needs the ssh
ControlMaster to `transfer.chtc.wisc.edu`, otherwise password + Duo):

```bash
ssh tdu35@transfer.chtc.wisc.edu 'cat /staging/t/tdu35/simfoundry/checkpoints/xarm6_eef_sim_16k.tar' \
  | tar -xf - -C outputs/chtc xarm6_eef_sim_16k/checkpoints/016000/pretrained_model
```

---

## 3. Before you start (checklist)

**Physical**
- [ ] E-stop within reach of whoever watches the arm; nobody inside the workspace.
- [ ] The scene looks like the demonstrations: toolbox on the table with the lid **open** (leaning
      back), blue and yellow 25 mm cubes on the table in front of it, nothing else in the way.
- [ ] If the e-stop was pressed, release it (see §8).

**Processes.** Only one program may command the arm and only one may open each camera.

```bash
ps aux | grep -E "realsense2_camera_node|simple_ik|joint_vel_controller|gripper_controller|inference_test|snow_camera_viewer|deploy_pi05" | grep -v grep
```

- Must **not** be running: ROS RealSense nodes (`gripper_cam`, `mount_cam`; they hold the cameras),
  xarm_stack `simple_ik` / `joint_vel_controller` / `gripper_controller` / `inference_test_vis`
  (they would send their own joint targets), another `deploy_pi05_xarm.py`.
- May keep running: the xArm ROS driver (`xarm_driver_node`), Foxglove, the data_collection server
  (as long as it is not recording through the cameras).
- These usually belong to other lab members (leo, bobby): ask before stopping them. With sudo:
  `sudo kill -INT <pid>` (the launch process first).

**GPU.** The server needs about 10 GB: `nvidia-smi --query-gpu=memory.used,memory.total --format=csv`.

---

## 4. Start the policy server

The server holds one checkpoint and stays up across rollouts. Run it in tmux so it survives the
terminal:

```bash
tmux new-session -d -s pi05_srv "cd /home/tingting/Documents/SimFoundry && \
  $SRV simfoundry/policies/pi05_serve/server.py \
  --checkpoint outputs/chtc/xarm6_eef_cotrain_16k/checkpoints/016000/pretrained_model \
  --device cuda --port 8420 2>&1 | tee /tmp/pi05_srv.log"
# for fast execution add --num-inference-steps 5 to cut latency (see §6b)

curl -s localhost:8420/health        # wait for {"ready":true,"status":"ok"} (~15 s)
```

- `--device cuda` is required. The default `cpu` loads the 14 GB model into RAM and is killed on
  this 30 GB machine.
- The `huggingface.co ... 200 OK` lines at startup are the PaliGemma tokenizer cache check, not
  errors.

**tmux basics.** `tmux attach -t pi05_srv` to look; the server's window only shows its log and
ignores typing. Do **not** press Ctrl-C there (it stops the server). `Ctrl-b c` opens a new window
for the deployment commands, `Ctrl-b n` / `Ctrl-b p` switch windows, `Ctrl-b d` detaches.

**Two models at once.** Start a second server on another port (load one, wait for ready, then the
next, so RAM is not hit twice at the same time) and select it with `--port`:

```bash
tmux new-session -d -s pi05_srv2 "cd /home/tingting/Documents/SimFoundry && \
  $SRV simfoundry/policies/pi05_serve/server.py --checkpoint outputs/hf/pi05-xarm6-real-sim-16k \
  --device cuda --port 8421 2>&1 | tee /tmp/pi05_srv2.log"
```

---

## 5. Run a rollout

Every rollout follows the same three steps.

```bash
# (a) start pose: slow joint move (0.25 rad/s) to the mean start pose of the demos, gripper open.
#     If the straight path would take the flange below 0.2 m it first lifts 6 cm.
$PY scripts/xarm6/real/go_to_start_pose.py

# (b) dry run: reads cameras and joints, queries the policy, prints targets. The arm does NOT move.
$PY scripts/xarm6/real/deploy_pi05_xarm.py --action-space eef \
    --prompt "Pick up the yellow cube and place it into the box." --dry-run

# (c) the real thing, hand on the e-stop. Ctrl-C stops at any time.
$PY scripts/xarm6/real/deploy_pi05_xarm.py --action-space eef \
    --prompt "Pick up the yellow cube and place it into the box." --max-seconds 60
```

- **Always run (a) first.** The policy only saw starts near this pose; from elsewhere it behaves
  unpredictably.
- **Prompt.** Use exactly `Pick up the {blue|yellow} cube and place it into the box.` (the real
  demos used only this wording).
- **What a sane dry run looks like:** about 12 lines like
  `[dry] t=0.50 k=10 jump=0.120 flange_z=0.281 g=-0.00 tcp=[0.311, 0.028, 0.110]`, ending with
  `inference latency median ~0.2s`.
  - `jump` grows in a dry run because the arm does not follow; it should stay well under 0.35.
  - `flange_z` should start around 0.28 m.
  - `g` (gripper) is about 0 at the start.
  - Errors about shapes or dimensions mean `--action-space` does not match the server's model.
- **One setup, two prompts.** With both cubes on the table you can run yellow, then (after
  re-running (a)) blue. Reset the cubes after each pair.
- For a joint-space model: `--action-space joint` (and `--port 8421` if it is on the second server).

---

## 6. Options

| Option | Default | Meaning |
|---|---|---|
| `--action-space` | `joint` | `eef` for task-space models (rounds 5, 6), `joint` for rounds 1–4 and real-only joint |
| `--prompt` | required | task instruction |
| `--max-seconds` | 40 | rollout length; 60 is a good value (demos take 17–29 s) |
| `--dry-run` | off | never enables motion; stops after 60 steps |
| `--port` | 8420 | which policy server |
| `--vmax` | 0.6 × speed, max π | joint speed clip, rad/s (xarm_stack uses 1.0; demos rarely exceed 0.45; hardware π). Lower it (0.3) for a cautious first run |
| `--kp` | 10 | P gain of the velocity loop (xarm_stack config) |
| `--max-jump` | 0.35 (relaxed with speed, max 0.8) | rad; stop if any joint target is this far from the measured joints |
| `--min-flange-z` | 0.15 | m; stop if a target puts the flange lower (grasping puts it near 0.17) |
| `--replan-every` | round(10 / speed) | actions into a chunk at which the next one is requested |
| `--blend` | 5 | steps to cross-fade between chunks |
| `--sync` | off | old stop-and-infer behaviour, for comparison only |
| `--execute-horizon` | 15 | steps per chunk, `--sync` only |
| `--hz` | 20 | action rate of the dataset; do not change it, use `--speed` to go faster |
| `--speed` | 1 | playback speed-up, see §6b |
| `--loop-hz` | min(200, 20 × speed) | control loop rate |
| `--tracker-hz` | 80 | velocity loop rate (xarm_stack: 80); 150–200 at high speed |
| `--robot-ip` | 192.168.1.111 | |
| `--out-dir` | `Data/cube2_scene/real_deploy` | where runs are written |

---

## 6b. Faster execution (`--speed`)

The policy reproduces the pace of the demonstrations (median 22.7 s) and uses about 14% of what the
arm can do (joints up to π rad/s, TCP up to 1 m/s). `--speed` plays the same predicted trajectory
faster: the model and inference are unchanged, a 50-action chunk takes 2.5 / speed s instead of 2.5 s.

```bash
# dry run first (no motion): check the loop rate and the waiting share in the last line
$PY scripts/xarm6/real/deploy_pi05_xarm.py --action-space eef --speed 2 \
    --prompt "Pick up the yellow cube and place it into the box." --dry-run
# real motion: start at 1.5x, go up once it behaves
$PY scripts/xarm6/real/deploy_pi05_xarm.py --action-space eef --speed 1.5 \
    --prompt "Pick up the yellow cube and place it into the box." --max-seconds 40
```

**What `--speed` adjusts** (printed in the first line, each can be overridden):

| Setting | Automatic value | Why |
|---|---|---|
| action playback | 20 × speed actions/s, linear interpolation between actions | |
| `--loop-hz` control loop | min(200, 20 × speed) Hz | measured 177 Hz at 10× |
| `--vmax` joint speed clip | min(π, 0.6 × speed) rad/s | same headroom as at 1×; π is the hardware limit |
| `--max-jump` | min(0.8, 0.35 + 0.05 (speed − 1)) rad | tracking error grows with speed |
| `--replan-every` | round(10 / speed), at least 1 | ask for the next chunk earlier |
| gripper speed | min(5000, 1500 × speed) | xarm_stack uses 1500, the gripper allows 5000 |
| `--tracker-hz` velocity loop | 80 Hz (unchanged) | raise to 150–200 at high speed for tighter tracking |

**How fast it can go is set by inference latency.** A chunk is already one latency old when it
arrives and the next one takes another latency, so continuous motion needs
2 × latency × action rate < 50, i.e.

> maximum speed ≈ 1.25 / inference latency (s)

| Latency | Max | When |
|---|---|---|
| 0.36 s | 3.5× | GPU shared with a sim evaluation |
| 0.2 s | 6× | GPU to itself (normal) |
| 0.1 s | 12× | needs fewer integration steps or other optimisation |

When inference falls behind, the script holds the last target until the next chunk arrives (the
arm slows down; `starved` is true in the log) and only stops after 1 s without a chunk. The last line
reports the share of loop ticks spent waiting; a clearly non-zero share means the speed is beyond
what inference can sustain.

**Lowering latency**
- Keep the GPU to the policy server during fast tests (no sim evaluation at the same time).
- Start the server with `--num-inference-steps 5` (trained with 10): half the flow-matching steps,
  faster inference, possibly coarser actions; measure it.

**Risks at speed**
- The ~0.1 s tracking lag stays, so position error grows with speed: corners get cut and the arm
  may hit the box lid.
- The gripper has a physical speed limit; the policy may lift before the fingers have closed.
- Human reaction time on the e-stop stays the same while the arm covers more ground. Check the
  behaviour with dry runs and low speeds on a clear table first.

---

## 7. Safety behaviour

The script stops the run (zero velocity, then position mode) when:
- a joint target is more than `--max-jump` from the measured joints;
- a target puts the flange below `--min-flange-z`;
- the arm SDK returns an error, or the policy server fails;
- no new chunk has arrived for 1 s (shorter gaps hold the last target);
- you press Ctrl-C, or `--max-seconds` passes.

Task-space targets beyond a joint limit or the workspace edge get the **nearest reachable pose**
(like simple_ik), not an error. Near the robot base the gripper cannot point straight down below
about 7 cm TCP height; there the arm may hover and retry instead of grasping.

The e-stop is always the last line of defence: the software checks look at targets, not at
contact forces.

---

## 8. After an e-stop or an error

1. Make the area safe, release the e-stop.
2. `$PY scripts/xarm6/real/go_to_start_pose.py`: it clears warnings and errors, re-enables the
   motors and sets the arm to position mode before moving.
3. If it still reports an error code (e.g. after a collision), clear it in UFactory Studio
   (browser: `http://192.168.1.111:18333`), then repeat step 2.

A run interrupted by Ctrl-C or a safety stop leaves the arm in position mode, ready for step 2.

---

## 9. Results

Each run writes `Data/cube2_scene/real_deploy/<YYYYMMDD_HHMMSS>/`:

| File | Content |
|---|---|
| `cameras.mp4` | front and wrist view side by side, 20 fps, H.264 (plays in VS Code); fragmented MP4, so a run that is killed (e-stop, kill) still leaves a playable file |
| `log.jsonl` | one line per step: `t`, chunk step `k`, measured joints `q`, gripper reading `grip_raw` (0 closed … 850 open), joint target `q_des`, gripper target `g_des`, TCP target `tcp_des` (eef), `jump`, `flange_z_des`, `fading` (inside a cross-fade), `starved` (waiting for a chunk). Flushed every 50 ms |
| `args.json` | the options used (model port, action space, prompt, limits) |

The script does not know whether the task succeeded. Watch the video and note the outcome. A
consistent way to record it is to rename the run folder, for example `20261020_191417_yellow_success`.

Quick summary of a run:

```bash
$PY - Data/cube2_scene/real_deploy/<run>/ <<'EOF'
import json, sys, numpy as np
L = [json.loads(l) for l in open(sys.argv[1] + "log.jsonl")]
g = np.array([r["grip_raw"] for r in L])
print(f"{len(L)} steps; gripper min {g.min():.0f} (closed on a cube ≈ 220, on nothing ≈ 0); "
      f"max target jump {max(r['jump'] for r in L):.3f} rad; lowest flange target {min(r['flange_z_des'] for r in L):.3f} m")
EOF
```

---

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `RuntimeError: xioctl(VIDIOC_S_FMT) failed ... Device or resource busy` | another process holds a camera (ROS RealSense node, a viewer, another deploy). See §3 |
| `curl: (7) Failed to connect to localhost port 8420` | server not running or still loading; `tail /tmp/pi05_srv.log` |
| Server killed while loading | started without `--device cuda`, or another model is loading at the same time |
| Shape / dimension error from the server | `--action-space` does not match the checkpoint |
| Arm does not move, no error | e-stop pressed or arm not enabled: §8. Or another node is commanding the arm: §3 |
| `stopped: safety: ... jump` | the policy asked for a large jump. Usually the start pose was skipped (run §5 a) or the scene differs from the demos |
| Arm hovers near the base and never grasps | cube very close to the base (< 0.18 m): known weakness, move the cube farther out |
| Typing in the tmux window shows `^[[A` | that window is the server; use `Ctrl-b c` for a shell |
| Motion looks stop-and-go | `--sync` is on, or inference is slow (`inference latency` in the final line should be about 0.2 s) |

---

## 11. Files

| Path | Role |
|---|---|
| `scripts/xarm6/real/deploy_pi05_xarm.py` | deployment (this guide) |
| `scripts/xarm6/real/go_to_start_pose.py` | move to the demo start pose |
| `simfoundry/policies/pi05_serve/server.py` | policy server |
| `simfoundry/policies/pi05_lerobot.py` | HTTP client (`PI05XArm6Client`) |
| `simfoundry/robots/xarm6_kinematics.py` | FK / IK (`ik_nearest` for task-space targets) |
| `scripts/xarm6/make_eef_dataset.py` | definition of the task-space state / action (`tcp_pose9`) |
| `docs/XARM6_HANDOFF.md` | project status, training rounds, CHTC |
| `docs/XARM6_EVAL_RUNS.md` | all sim and real evaluation runs |
