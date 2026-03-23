import argparse
from pathlib import Path
import numpy as np

from tidybot_arm_reach_env_v2 import ArmReachEnvV2
from arm_teacher import ArmTeacherDLS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=800)
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = ArmReachEnvV2(
        args.xml,
        render_mode="human" if args.render else None,
    )
    env.max_steps = int(args.max_steps)

    teacher = ArmTeacherDLS(
        model=env.model,
        data=env.data,
        ee_site_id=env.ee_site,
        target_site_id=env.target_site,
        arm_dof_indices=env.arm_dadr,
        kp_xyz=10.0,
        damping=0.05,
        max_cart_vel=0.40,
        max_joint_cmd=1.5,
        stop_radius=0.06,
        posture_gain=0.25,
        q_nominal=np.array([0.0, 0.4, 0.0, 1.0, 0.0, 0.8, 0.0], dtype=np.float64),
    )

    states = []
    actions = []
    successes = []
    distances = []
    best_distances = []
    episode_ids = []
    timesteps = []
    sources = []

    success_eps = 0
    ep_best_list = []
    ep_final_list = []
    ep_start_list = []

    for ep in range(args.episodes):
        obs, info = env.reset()
        start_dist = float(info.get("distance", obs[3]))
        best_dist = start_dist
        done = False
        steps = 0

        while not done:
            a = teacher.act(obs)
            next_obs, r, term, trunc, info = env.step(a)
            done = bool(term or trunc)

            d = float(info.get("distance", next_obs[3]))
            best_dist = min(best_dist, d)

            states.append(obs.copy())
            actions.append(a.copy())
            successes.append(int(info.get("success", False)))
            distances.append(d)
            best_distances.append(best_dist)
            episode_ids.append(ep)
            timesteps.append(steps)
            sources.append("teacher_dls_v2")

            obs = next_obs
            steps += 1

        final_dist = float(info.get("distance", obs[3]))
        ok = bool(info.get("success", False))
        success_eps += int(ok)

        ep_start_list.append(start_dist)
        ep_best_list.append(best_dist)
        ep_final_list.append(final_dist)

        print(
            f"[record] ep={ep+1}/{args.episodes} "
            f"steps={steps} success={ok} "
            f"start={start_dist:.4f} best={best_dist:.4f} final={final_dist:.4f}"
        )

    np.savez_compressed(
        out_path,
        state=np.asarray(states, dtype=np.float32),
        action=np.asarray(actions, dtype=np.float32),
        success=np.asarray(successes, dtype=np.int8),
        distance=np.asarray(distances, dtype=np.float32),
        best_distance=np.asarray(best_distances, dtype=np.float32),
        episode_id=np.asarray(episode_ids, dtype=np.int32),
        timestep=np.asarray(timesteps, dtype=np.int32),
        source=np.asarray(sources),
    )

    print(
        f"[summary] success_eps={success_eps}/{args.episodes} "
        f"mean_start={np.mean(ep_start_list):.4f} "
        f"mean_best={np.mean(ep_best_list):.4f} "
        f"mean_final={np.mean(ep_final_list):.4f}"
    )
    print(f"[done] saved={out_path}")


if __name__ == "__main__":
    main()