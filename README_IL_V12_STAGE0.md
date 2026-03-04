# Imitation Learning (BC) — V12 Stage0 (Base → Door Anchor)

This folder provides:
- a deterministic teacher controller
- dataset recorder (teacher rollouts)
- Behavior Cloning (PyTorch)
- BC evaluator in the same env

## 0) Assumptions
- You have the Stage0 env:
  - `tidybot_nav_env_v12_stage0.py` with class `TidybotNavEnvV12Stage0`
  - `v12_stage0_config.py` providing `ENV`
- Env expects normalized actions in [-1,1]^3 (true in your step()).
- Collision info exists in `info["collided"]` and `info["collided_with"]`.

## 1) Record teacher dataset (quick test)
From repo root (or wherever python can import your env modules):

```bash
python3 tb_tidybot_nav/il/record_teacher_v12_stage0.py \
  --out_dir ./teacher_data_v12_stage0_test \
  --episodes 10 \
  --shard_size 2000 \
  --seed 1 \
  --terminate_on_collision