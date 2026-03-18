import os
os.environ.setdefault("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))

import argparse
import numpy as np
import torch
from dataclasses import replace

from tidybot_nav_env_v12_stage0 import TidybotNavEnvV12Stage0
from v12_stage0_config import ENV
from nets import BCPolicy


def make_env(
    render_mode=None,
    max_steps=None,
    yaw_tol=None,
):
    p = ENV
    if max_steps is not None or yaw_tol is not None:
        p = replace(
            ENV,
            max_steps=int(max_steps) if max_steps is not None else ENV.max_steps,
            prepose_yaw_tolerance=float(yaw_tol) if yaw_tol is not None else ENV.prepose_yaw_tolerance,
        )
    return TidybotNavEnvV12Stage0(p, render_mode=render_mode)


def reset_env(env, args, seed):
    if args.fixed_spawn:
        return env.reset(
            seed=seed,
            options={
                "manual_spawn": {
                    "x": float(args.start_x),
                    "y": float(args.start_y),
                    "yaw": float(args.start_yaw),
                }
            },
        )
    return env.reset(seed=seed)


def obs_to_torch(obs, device):
    s = obs["state"].astype(np.float32)
    st_t = torch.from_numpy(s).unsqueeze(0)
    return st_t.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render", action="store_true")

    ap.add_argument("--fixed_spawn", action="store_true")
    ap.add_argument("--start_x", type=float, default=0.0)
    ap.add_argument("--start_y", type=float, default=-2.0)
    ap.add_argument("--start_yaw", type=float, default=1.57)

    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--yaw_tol", type=float, default=None)
    args = ap.parse_args()

    env = make_env(
        render_mode="human" if args.render else None,
        max_steps=args.max_steps,
        yaw_tol=args.yaw_tol,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    state_dim = int(ckpt["state_dim"])

    net = BCPolicy(state_dim=state_dim).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()

    rng = np.random.default_rng(args.seed)

    succ = 0
    final_ds = []
    final_yaws = []
    final_holds = []

    for ep in range(args.episodes):
        obs, info = reset_env(env, args, seed=int(rng.integers(0, 1_000_000)))
        done = False
        steps = 0
        best = 1e9

        while not done:
            st_t = obs_to_torch(obs, device)
            with torch.no_grad():
                a = net(st_t).cpu().numpy()[0].astype(np.float32)

            obs, r, terminated, truncated, info = env.step(a)
            done = bool(terminated or truncated)
            steps += 1
            best = min(best, float((info or {}).get("best_d_anchor", 1e9)))

        is_success = bool((info or {}).get("is_success", False))
        d_anchor = float((info or {}).get("d_anchor", float("nan")))
        yaw_err = float((info or {}).get("yaw_err", float("nan")))
        hold_counter = int((info or {}).get("hold_counter", 0))

        succ += int(is_success)
        final_ds.append(d_anchor)
        final_yaws.append(abs(yaw_err))
        final_holds.append(hold_counter)

        print(
            f"[EP {ep+1}/{args.episodes}] steps={steps} success={is_success} "
            f"d_anchor={d_anchor:.3f} best={best:.3f} "
            f"yaw_err={yaw_err:.3f} hold={hold_counter} "
            f"coll={int((info or {}).get('collided', 0))} "
            f"hit='{(info or {}).get('collided_with','')}'"
        )

    print(
        f"[summary] success_rate={succ}/{args.episodes} = {succ/args.episodes:.3f}  "
        f"mean_final_d={np.mean(final_ds):.3f}  "
        f"mean_abs_final_yaw={np.mean(final_yaws):.3f}  "
        f"mean_final_hold={np.mean(final_holds):.2f}"
    )
    env.close()


if __name__ == "__main__":
    main()