import os
os.environ.setdefault("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))

import argparse
import numpy as np
import torch

from stable_baselines3.common.monitor import Monitor

from tidybot_nav_env_v12_stage0 import TidybotNavEnvV12Stage0
from v12_stage0_config import ENV

from nets import BCPolicy


def make_env(render_mode=None):
    env = TidybotNavEnvV12Stage0(ENV, render_mode=render_mode)
    env = Monitor(env)
    return env


def ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    if np.issubdtype(img.dtype, np.floating):
        return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    return img.astype(np.uint8)


def obs_to_torch(obs, device):
    img = ensure_uint8(obs["image"])
    img_t = torch.from_numpy(img).permute(2, 0, 1).contiguous().float().unsqueeze(0) / 255.0
    st_t  = torch.from_numpy(obs["state"].astype(np.float32)).unsqueeze(0)
    return img_t.to(device), st_t.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    env = make_env(render_mode="human" if args.render else None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = BCPolicy(state_dim=7).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    net.load_state_dict(ckpt["model"])
    net.eval()

    rng = np.random.default_rng(args.seed)

    succ = 0
    for ep in range(args.episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
        done = False
        steps = 0
        best = 1e9

        while not done:
            img_t, st_t = obs_to_torch(obs, device)
            with torch.no_grad():
                a = net(img_t, st_t).cpu().numpy()[0].astype(np.float32)

            obs, r, terminated, truncated, info = env.step(a)
            done = bool(terminated or truncated)
            steps += 1
            best = min(best, float((info or {}).get("best_d_anchor", 1e9)))

        is_success = bool((info or {}).get("is_success", False))
        succ += int(is_success)
        print(
            f"[EP {ep+1}/{args.episodes}] steps={steps} success={is_success} "
            f"d_anchor={float((info or {}).get('d_anchor', float('nan'))):.3f} best={best:.3f} "
            f"coll={int((info or {}).get('collided', 0))} hit='{(info or {}).get('collided_with','')}'"
        )

    print(f"[summary] success_rate={succ}/{args.episodes} = {succ/args.episodes:.3f}")
    env.close()

if __name__ == "__main__":
    main()