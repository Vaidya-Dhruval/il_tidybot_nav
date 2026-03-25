import os
os.environ.setdefault("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))

import argparse
import time
import numpy as np
import torch

from tidybot_door_open_env_v2 import TidybotDoorOpenEnvV2
from nets import BCPolicy


def obs_to_torch(obs, device):
    s = np.asarray(obs, dtype=np.float32)
    return torch.from_numpy(s).unsqueeze(0).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--xml", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.03)
    ap.add_argument("--tau", type=float, default=0.20)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device]", device)

    ckpt = torch.load(args.ckpt, map_location=device)
    state_dim = int(ckpt["state_dim"])
    action_dim = int(ckpt["action_dim"])

    net = BCPolicy(state_dim=state_dim, action_dim=action_dim).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()

    env = TidybotDoorOpenEnvV2(
        args.xml,
        render_mode="human" if args.render else None,
    )
    env.max_steps = int(args.max_steps)

    succ = 0
    final_progress_list = []
    final_door_q_list = []
    final_handle_dist_list = []
    steps_list = []

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        steps = 0
        a_smooth = np.zeros(action_dim, dtype=np.float32)

        while not done:
            st_t = obs_to_torch(obs, device)
            with torch.no_grad():
                a_raw = net(st_t).cpu().numpy()[0].astype(np.float32)

            a_smooth = (1.0 - args.tau) * a_smooth + args.tau * a_raw

            obs, reward, terminated, truncated, info = env.step(a_smooth)
            done = bool(terminated or truncated)
            steps += 1

            if args.render:
                time.sleep(args.sleep)

        is_success = bool(info.get("success", False))
        succ += int(is_success)

        final_progress = float(info.get("progress", np.nan))
        final_door_q = float(info.get("door_q", np.nan))
        final_handle_dist = float(info.get("handle_dist", np.nan))

        final_progress_list.append(final_progress)
        final_door_q_list.append(final_door_q)
        final_handle_dist_list.append(final_handle_dist)
        steps_list.append(steps)

        print(
            f"[EP {ep+1}/{args.episodes}] "
            f"steps={steps} success={is_success} "
            f"progress={final_progress:.4f} "
            f"door_q={final_door_q:.4f} "
            f"handle_dist={final_handle_dist:.4f}"
        )

    print(
        f"[summary] success_rate={succ}/{args.episodes} = {succ/args.episodes:.3f}  "
        f"mean_final_progress={np.mean(final_progress_list):.4f}  "
        f"mean_final_door_q={np.mean(final_door_q_list):.4f}  "
        f"mean_final_handle_dist={np.mean(final_handle_dist_list):.4f}  "
        f"mean_steps={np.mean(steps_list):.1f}"
    )

    env.close()


if __name__ == "__main__":
    main()