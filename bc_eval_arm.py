import os
os.environ.setdefault("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))

import argparse
import time
import numpy as np
import torch

from tidybot_arm_reach_env_v2 import ArmReachEnvV2
from nets import BCPolicy


def obs_to_torch(obs, device):
    s = np.asarray(obs, dtype=np.float32)
    st_t = torch.from_numpy(s).unsqueeze(0)
    return st_t.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--xml", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=1500)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--fixed_reset", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.02)
    ap.add_argument("--tau", type=float, default=0.22)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device]", device)

    ckpt = torch.load(args.ckpt, map_location=device)
    state_dim = int(ckpt["state_dim"])
    action_dim = int(ckpt["action_dim"])

    net = BCPolicy(state_dim=state_dim, action_dim=action_dim).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()

    env = ArmReachEnvV2(
        args.xml,
        render_mode="human" if args.render else None,
    )
    env.max_steps = int(args.max_steps)

    if args.fixed_reset:
        env.jitter_x = 0.0
        env.jitter_y = 0.0
        env.jitter_yaw = 0.0

    succ = 0
    start_list = []
    best_list = []
    final_list = []
    steps_list = []

    for ep in range(args.episodes):
        obs, info = env.reset()
        start_dist = float(info.get("distance", obs[3]))
        done = False
        steps = 0
        best = start_dist
        a_smooth = np.zeros(action_dim, dtype=np.float32)

        while not done:
            st_t = obs_to_torch(obs, device)
            with torch.no_grad():
                a_raw = net(st_t).cpu().numpy()[0].astype(np.float32)

            # slightly stronger response than before
            a_smooth = (1.0 - args.tau) * a_smooth + args.tau * a_raw

            # gentler near-target damping so final correction is not suppressed too much
            dist_now = float(info.get("distance", obs[3]))
            if dist_now < 0.12:
                a_smooth *= 0.75
            if dist_now < 0.10:
                a_smooth *= 0.55

            obs, r, terminated, truncated, info = env.step(a_smooth)
            done = bool(terminated or truncated)
            steps += 1
            best = min(best, float(info.get("best_distance", 1e9)))

            if args.render:
                time.sleep(args.sleep)

        final_dist = float(info.get("distance", obs[3]))
        is_success = bool(info.get("success", False))
        succ += int(is_success)

        start_list.append(start_dist)
        best_list.append(best)
        final_list.append(final_dist)
        steps_list.append(steps)

        print(
            f"[EP {ep+1}/{args.episodes}] "
            f"steps={steps} success={is_success} "
            f"start={start_dist:.4f} best={best:.4f} final={final_dist:.4f}"
        )

    print(
        f"[summary] success_rate={succ}/{args.episodes} = {succ/args.episodes:.3f}  "
        f"mean_start={np.mean(start_list):.4f}  "
        f"mean_best={np.mean(best_list):.4f}  "
        f"mean_final={np.mean(final_list):.4f}  "
        f"mean_steps={np.mean(steps_list):.1f}"
    )

    env.close()


if __name__ == "__main__":
    main()