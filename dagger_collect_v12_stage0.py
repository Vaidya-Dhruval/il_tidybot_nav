import os
os.environ.setdefault("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))

import argparse
import numpy as np
from pathlib import Path
from dataclasses import replace

import torch
from stable_baselines3.common.monitor import Monitor

from tidybot_nav_env_v12_stage0 import TidybotNavEnvV12Stage0
from v12_stage0_config import ENV

from teacher import GoToGoalTeacher
from nets import BCPolicy


def make_env(env_params):
    env = TidybotNavEnvV12Stage0(env_params, render_mode=None)
    env = Monitor(env)
    return env


def flush(out_dir: Path, shard_idx: int, states, actions, infos):
    if len(states) == 0:
        return shard_idx

    path = out_dir / f"dagger_shard_{shard_idx:04d}.npz"

    collided = np.array([int((i or {}).get("collided", 0)) for i in infos], dtype=np.int8)
    d_anchor = np.array([float((i or {}).get("d_anchor", np.nan)) for i in infos], dtype=np.float32)

    np.savez_compressed(
        path,
        state=np.stack(states, axis=0),
        action=np.stack(actions, axis=0),   # teacher labels
        collided=collided,
        d_anchor=d_anchor,
    )
    print(f"[flush] wrote {path}  N={len(states)}")
    return shard_idx + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc_ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--shard_size", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--terminate_on_collision", action="store_true")
    ap.add_argument("--mix_curriculum", action="store_true")

    # teacher params
    ap.add_argument("--kx", type=float, default=0.20)
    ap.add_argument("--ky", type=float, default=0.20)
    ap.add_argument("--kth", type=float, default=0.70)
    ap.add_argument("--stop_radius", type=float, default=2.0)

    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load BC policy
    ckpt = torch.load(args.bc_ckpt, map_location=device)
    state_dim = int(ckpt["state_dim"])
    bc = BCPolicy(state_dim=state_dim).to(device)
    bc.load_state_dict(ckpt["model"])
    bc.eval()

    teacher = GoToGoalTeacher(
        max_vx=ENV.max_vx,
        max_vy=ENV.max_vy,
        max_wz=ENV.max_wz,
        kx=args.kx,
        ky=args.ky,
        kth=args.kth,
        stop_radius=args.stop_radius,
    )

    rng = np.random.default_rng(args.seed)

    CURR = [
        ("near", 0.35, 0.65, 300),
        ("mid",  0.80, 1.80, 500),
        ("far",  2.00, 4.00, 1400),
    ]

    # quick shape sanity
    env0 = make_env(ENV)
    obs, info = env0.reset(seed=args.seed)
    print("[obs state shape]", obs["state"].shape)
    env0.close()

    states, actions, infos = [], [], []
    shard_idx = 0
    total_steps = 0

    bc_success = 0
    bc_collisions = 0

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

        env = make_env(env_params)
        obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
        done = False

        while not done:
            s = obs["state"].astype(np.float32)

            # BC executes
            st_t = torch.from_numpy(s).unsqueeze(0).to(device)
            with torch.no_grad():
                a_exec = bc(st_t).cpu().numpy()[0].astype(np.float32)

            # Teacher labels THIS visited state
            a_teacher = teacher.act(s)

            next_obs, r, terminated, truncated, info = env.step(a_exec)
            done = bool(terminated or truncated)

            states.append(s.copy())
            actions.append(a_teacher.astype(np.float32))
            infos.append(info if isinstance(info, dict) else {})

            total_steps += 1

            if int((info or {}).get("collided", 0)) == 1:
                bc_collisions += 1
                if args.terminate_on_collision:
                    break

            obs = next_obs

            if total_steps % args.shard_size == 0:
                shard_idx = flush(out_dir, shard_idx, states, actions, infos)
                states, actions, infos = [], [], []

        bc_success += int(bool((info or {}).get("is_success", False)))
        env.close()

        if (ep + 1) % 50 == 0:
            print(
                f"[dagger] ep={ep+1}/{args.episodes} mode={mode} "
                f"steps={total_steps} shards={shard_idx} "
                f"bc_success={bc_success}/{ep+1}"
            )

    shard_idx = flush(out_dir, shard_idx, states, actions, infos)
    print(
        f"[done] steps={total_steps} shards={shard_idx} "
        f"bc_success={bc_success}/{args.episodes} "
        f"bc_collisions={bc_collisions}"
    )


if __name__ == "__main__":
    main()