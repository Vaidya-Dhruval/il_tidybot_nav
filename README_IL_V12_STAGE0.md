# IL / Behavior Cloning — V12 Stage0 (Base → Door Anchor)

Run from project root:
.../stanford_tidybot

## Why your first dataset was tiny
Your ENV uses:
- spawn_around_target=True
- spawn_target_min_r=0.35
- spawn_target_max_r=0.65

So the robot always spawns near the door. For BC robustness we must record near/mid/far.

## 1) Record dataset (mixed curriculum) — recommended

This will automatically cycle near/mid/far:
- near: 0.35–0.65 m
- mid: 0.8–1.8 m
- far: 2.0–4.0 m

```bash
PYTHONPATH=. python3 tb_tidybot_nav_il/il/record_teacher_v12_stage0.py \
  --out_dir ./teacher_data_v12_stage0 \
  --episodes 3000 \
  --shard_size 50000 \
  --seed 0 \
  --terminate_on_collision \
  --mix_curriculum