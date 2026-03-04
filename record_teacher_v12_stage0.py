import os
os.environ.setdefault("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))

import argparse
import numpy as np
from pathlib import Path

from stable_baselines3.common.monitor import Monitor

from tidybot_nav_env_v12_stage0 import TidybotNavEnvV12Stage0
from v12_stage0_config import ENV
from teacher import GoToGoalTeacher


def make_env():
    env = TidybotNavEnvV12Stage0(ENV, render_mode=None)
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
    d_anchor = np.array([float((i or {}).get("d_anchor", np.nan)) for i in infos], dtype=np.float32)

    np.savez_compressed(
        path,
        image=np.stack(images, axis=0),      # (N,H,W,3) uint8
        state=np.stack(states, axis=0),      # (N,7) float32
        action=np.stack(actions, axis=0),    # (N,3) float32
        collided=collided,
        d_anchor=d_anchor,
    )
    print(f"[flush] wrote {path}  N={len(states)}")
    return shard_idx + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--shard_size", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)

    # teacher gains (optional)
    ap.add_argument("--kx", type=float, default=0.9)
    ap.add_argument("--ky", type=float, default=0.9)
    ap.add_argument("--kth", type=float, default=1.8)
    ap.add_argument("--stop_radius", type=float, default=1.2)

    # dataset hygiene
    ap.add_argument("--terminate_on_collision", action="store_true")

    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_env()

    # sanity print
    obs, info = env.reset(seed=args.seed)
    print("[obs keys]", obs.keys())
    print("[image]", obs["image"].shape, obs["image"].dtype)
    print("[state]", obs["state"].shape, obs["state"].dtype)
    print("[info example keys]", list(info.keys()) if isinstance(info, dict) else type(info))

    teacher = GoToGoalTeacher(
        max_vx=0.12, max_vy=0.08, max_wz=0.35,
        kx=args.kx, ky=args.ky, kth=args.kth,
        stop_radius=args.stop_radius,
    )

    rng = np.random.default_rng(args.seed)

    images, states, actions, infos = [], [], [], []
    shard_idx = 0
    total_steps = 0

    for ep in range(args.episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
        done = False

        while not done:
            s = obs["state"].astype(np.float32)
            a = teacher.act(s)

            next_obs, r, terminated, truncated, info = env.step(a)
            done = bool(terminated or truncated)

            # store obs -> teacher_action
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

        if (ep + 1) % 10 == 0:
            print(f"[record] ep={ep+1}/{args.episodes} total_steps={total_steps}")

    shard_idx = flush(out_dir, shard_idx, images, states, actions, infos)
    print(f"[done] steps={total_steps} shards={shard_idx}")
    env.close()


if __name__ == "__main__":
    main()