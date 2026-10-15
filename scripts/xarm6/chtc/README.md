# Training the xArm6 pi0.5 finetune on CHTC

Runs `scripts/xarm6/finetune_pi05.sh` (LeRobot 0.6.1) on a CHTC GPU Lab node, then brings the
checkpoints back for sim eval. The layout follows `openpi/chtc` (container universe, inputs from
`/staging`, one checkpoint tarball back to `/staging`).

```
lab machine                         CHTC                                   lab machine
convert demos -> LeRobot dataset ─> train in container (L40S..H200) ─> pull ckpt -> serve -> sim eval
stage_to_chtc.sh                    condor_submit train.sub               pull_checkpoints.sh
```

## One-time setup

1. **Image** (`train.Dockerfile`): same packages as `.cache/pi05-serve-venv`
   (`requirements-train.txt`), torch 2.11 / torchvision 0.26 from the CUDA 12.6 index.
   ```bash
   # tdu35 = Docker Hub username (.env: dockerhub_user), not an email; run `docker login` first.
   docker build -t tdu35/simfoundry_lerobot_train:0.6.1 -f scripts/xarm6/chtc/train.Dockerfile scripts/xarm6/chtc
   docker push tdu35/simfoundry_lerobot_train:0.6.1
   ```
2. **W&B** on the access point (not in any file in this repo): add
   `export WANDB_API_KEY=... WANDB_ENTITY=...` to `~/.bashrc` there (`chmod 600`).

## Each run

```bash
# 1. Lab machine, repo root: upload dataset, base checkpoint (+ PaliGemma tokenizer cache), job files.
bash scripts/xarm6/chtc/stage_to_chtc.sh tdu35 ap2002.chtc.wisc.edu          # or: dataset | base | jobfiles

# 2. Access point
ssh tdu35@ap2002.chtc.wisc.edu
cd ~/simfoundry_chtc
condor_submit train.sub dockerhub_user=tdu35
condor_q; condor_tail -f <job_id>

# 3. Lab machine, when it finishes
bash scripts/xarm6/chtc/pull_checkpoints.sh tdu35 xarm6_sim_<cluster_id>
```

The base checkpoint only needs re-staging when `checkpoints/pi05_droid_jointpos` changes; a new
dataset only needs `stage_to_chtc.sh ... dataset`.

## Variants (submit-time overrides)

| Run | Overrides |
|---|---|
| Default: local recipe, batch 8 x 20k steps, frozen vision, action expert only (~16 GB) | none |
| Bigger batch, same number of epochs | `batch_size=32 steps=5000 save_freq=2500` |
| Full finetune (vision still frozen) on an 80 GB card | `extra_args="--policy.train_expert_only=false" min_gpu_mem=80000` |
| Over 12 h | `+GPUJobLength="medium"` (24 h) or `"long"` (7 days, max 4 GPUs/user) |

Use a new `exp_name` per submission; the bundle is written to
`/staging/t/tdu35/simfoundry/checkpoints/<exp_name>.tar`, with every saved step's
`pretrained_model/` (optimizer state is left out).

## GPU Lab (from the CHTC GPU jobs guide)

| GPU | Memory | Capability | GPUs |
|---|---|---|---|
| L40 / L40S | 45 GB | 8.9 | 30 / 16 |
| A100-SXM4 | 40 / 80 GB | 8.0 | 8 / 32 |
| H100 | 80 GB | 9.0 | 8 |
| H200 | 141 GB | 9.0 | 16 |

Job lengths: short 12 h (up to 2/3 of the lab), medium 24 h (1/3), long 7 days (4 GPUs per user).

## Notes

- The job runs Hub-offline (`HF_HUB_OFFLINE=1`): the tokenizer (`google/paligemma-3b-pt-224`, gated)
  ships in the base tarball, so no HF token is needed on CHTC. Only W&B needs the network.
- `Requirements` asks for a CUDA 12.6+ driver (cu126 wheels) and `/staging` access.
- `run_train_chtc.sh` was tested end to end on the lab machine outside the container
  (unpack, offline training, bundle). The container itself has not been run on CHTC yet.
