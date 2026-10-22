# CHTC package: task-space (TCP) actions, sim-only data

One of three task-space runs that differ only in the data:

| package | data | steps |
|---|---|---|
| `../chtc_eef` | 50 real (oversampled x4) + 447 sim | 16k |
| `../chtc_eef_sim` | 447 sim only | 16k |
| `../chtc_eef_real` | 50 real only | 8k |

Representation, model, optimiser and augmentation are those of `../chtc_eef` (see its README):
10-D TCP action / 16-D state, relative actions, pi05_base with only the action expert trained.
The trainer files (`run_train_eef.sh`, `finetune_eef.sh`, `train_cotrain.py`) live in
`../chtc_eef` and are copied to the access point by `submit.sh`; this directory holds the
submit file with this run's defaults.

- Data: 447 real-aligned sim episodes (182,911 frames), 3 prompt paraphrases per colour, dataset `Data/cube2_scene/lerobot/xarm6_sim_eef`.
- Run: 16,000 steps x batch 32 (≈2.8 epochs), save every 4,000, LR warmup 1,000 then cosine to 2.5e-6 at 16,000; `REAL_REPEAT=1` (no oversampling).
- Access point dir `~/simfoundry_chtc_eef_sim`, W&B project `pi05_xarm6_eef_sim`, output
  `/staging/t/tdu35/simfoundry/checkpoints/xarm6_eef_sim_16k.tar`.

```bash
# dataset (already built 2026-10-20)
.cache/pi05-serve-venv/bin/python scripts/xarm6/filter_dataset_by_source.py \\
    --src Data/cube2_scene/lerobot/xarm6_real_sim_eef --root Data/cube2_scene/lerobot/xarm6_sim_eef --keep sim
# upload + submit (needs the ssh ControlMaster to ap2002 / transfer, otherwise password + Duo)
bash scripts/xarm6/chtc_eef_sim/submit.sh            # exp_name xarm6_eef_sim_16k
# eval (sim) / deploy (real): --action-space eef
```
