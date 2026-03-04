import os
os.environ.setdefault("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))

import argparse
import numpy as np
from pathlib import Path
from dataclasses import replace

from stable_baselines3.common.monitor import Monitor

from tidybot_nav_env_v12_stage0 import TidybotNavEnvV12Stage0
from v12_stage0_config import ENV
from teacher import GoToGoalTeacher


def make_env(env_params):
    env = TidybotNavEnvV12Stage0(env_params, render_mode=None)
    env = Monitor(env)
    return env


def ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    if np.issubdtype(img.dtype, np.floating):
        return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    return img.astype(np.uint8)


def flush(out_dir: Path, shard_idx: int, images, states, actions, infos):
    if len(states) == 0:
        return shard_idx

    path = out_dir / f"teacher_shard_{shard_idx:04d}.npz"

    collided = np.array([int((i or {}).get("collided", 0)) for i in infos], dtype=np.int8)
    d_anchor  = np.array([float((i or {}).get("d_anchor", np.nan)) for i in infos], dtype=np.float32)

    np.savez_compressed(
        path,
        image=np.stack(images, axis=0),      # (N,H,W,3) uint8
        state=np.stack(states, axis=0),      # (N,7) float32 (padded)
        action=np.stack(actions, axis=0),    # (N,3) float32
        collided=collided,
        d_anchor=d_anchor,
    )
    print(f"[flush] wrote {path}  N={len(states)}")
    return shard_idx + 1


def main():
    ap = argparse.ArgumentParser()

    # output / size
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=1200)
    ap.add_argument("--shard_size", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--terminate_on_collision", action="store_true")

    # teacher tuning (defaults set to the recommended non-saturating values)
    ap.add_argument("--kx", type=float, default=0.35)
    ap.add_argument("--ky", type=float, default=0.35)
    ap.add_argument("--kth", type=float, default=1.0)
    ap.add_argument("--stop_radius", type=float, default=1.5)

    # curriculum overrides
    ap.add_argument("--mix_curriculum", action="store_true",
                   help="If set, cycles through near/mid/far spawn rings every episode with max_steps override")

    # manual override mode (only used if mix_curriculum is OFF)
    ap.add_argument("--spawn_around_target", type=int, default=-1,
                   help="0/1 to override ENV.spawn_around_target, -1 keeps ENV")
    ap.add_argument("--spawn_target_min_r", type=float, default=-1.0)
    ap.add_argument("--spawn_target_max_r", type=float, default=-1.0)
    ap.add_argument("--spawn_xy_range", type=float, default=-1.0)
    ap.add_argument("--spawn_yaw_range", type=float, default=-1.0)
    ap.add_argument("--max_steps", type=int, default=-1,
                   help="override ENV.max_steps in manual mode")

    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    teacher = GoToGoalTeacher(
        max_vx=ENV.max_vx, max_vy=ENV.max_vy, max_wz=ENV.max_wz,
        kx=args.kx, ky=args.ky, kth=args.kth,
        stop_radius=args.stop_radius,
    )

    # Curriculum rings + feasible horizons:
    # With max_vx=0.12 and dt=0.05 => ~1.8m max in 300 steps.
    # So we increase max_steps for mid/far so success is possible.
    CURR = [
        ("near", 0.35, 0.65, 300),
        ("mid",  0.80, 1.80, 500),
        ("far",  2.00, 4.00, 1400),
    ]

    # sanity print
    env0 = make_env(ENV)
    obs, info = env0.reset(seed=args.seed)
    print("[obs keys]", obs.keys())
    print("[image]", obs["image"].shape, obs["image"].dtype)
    print("[state]", obs["state"].shape, obs["state"].dtype)
    print("[ENV]", ENV)
    env0.close()

    images, states, actions, infos = [], [], [], []
    shard_idx = 0
    total_steps = 0

    for ep in range(args.episodes):
        env_params = ENV
        mode = "env_default"

        if args.mix_curriculum:
            mode, rmin, rmax, ms = CURR[ep % len(CURR)]
            env_params = replace(
                env_params,
                spawn_around_target=True,
                spawn_target_min_r=float(rmin),
                spawn_target_max_r=float(rmax),
                max_steps=int(ms),
            )
        else:
            kw = {}
            if args.spawn_around_target in (0, 1):
                kw["spawn_around_target"] = bool(args.spawn_around_target)
            if args.spawn_target_min_r >= 0:
                kw["spawn_target_min_r"] = float(args.spawn_target_min_r)
            if args.spawn_target_max_r >= 0:
                kw["spawn_target_max_r"] = float(args.spawn_target_max_r)
            if args.spawn_xy_range >= 0:
                kw["spawn_xy_range"] = float(args.spawn_xy_range)
            if args.spawn_yaw_range >= 0:
                kw["spawn_yaw_range"] = float(args.spawn_yaw_range)
            if args.max_steps > 0:
                kw["max_steps"] = int(args.max_steps)
            if kw:
                env_params = replace(env_params, **kw)
                mode = "custom"

        env = make_env(env_params)
        obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
        done = False

        while not done:
            s = obs["state"].astype(np.float32)

            # pad state to 7 dims for BC training consistency
            if s.shape[0] == 6:
                s = np.concatenate([s, np.array([0.0], dtype=np.float32)], axis=0)

            a = teacher.act(s)

            next_obs, r, terminated, truncated, info = env.step(a)
            done = bool(terminated or truncated)

            images.append(ensure_uint8(obs["image"]))
            states.append(s[:7].astype(np.float32))
            actions.append(a.astype(np.float32))
            infos.append(info if isinstance(info, dict) else {})

            total_steps += 1

            if args.terminate_on_collision and int((info or {}).get("collided", 0)) == 1:
                break

            obs = next_obs

            if total_steps % args.shard_size == 0:
                shard_idx = flush(out_dir, shard_idx, images, states, actions, infos)
                images, states, actions, infos = [], [], [], []

        env.close()

        if (ep + 1) % 50 == 0:
            print(f"[record] ep={ep+1}/{args.episodes} mode={mode} total_steps={total_steps} shards={shard_idx}")

    shard_idx = flush(out_dir, shard_idx, images, states, actions, infos)
    print(f"[done] steps={total_steps} shards={shard_idx}")


if __name__ == "__main__":
    main()