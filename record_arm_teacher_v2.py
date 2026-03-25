import argparse
from pathlib import Path
import numpy as np

from tidybot_door_open_env_v2 import TidybotDoorOpenEnvV2


class ScriptedStage2Coordinated:
    """
    Same coordinated controller used in debug:
    - move base toward safe open-support pose
    - move arm toward open arm pose
    """

    def __init__(self, env):
        self.env = env

    def action(self):
        base_now = self.env._get_base_xyth()
        arm_now = np.array([self.env.data.qpos[i] for i in self.env.arm_qadr], dtype=np.float64)

        base_err = self.env.goal_base_pose - base_now
        base_err[2] = ((base_err[2] + np.pi) % (2 * np.pi)) - np.pi
        arm_err = self.env.goal_arm_q - arm_now

        a_base = np.zeros(3, dtype=np.float32)
        a_arm = np.zeros(7, dtype=np.float32)

        # intentionally moderate and smooth
        a_base[0] = np.clip(base_err[0] / max(self.env.max_vx * self.env.dt, 1e-8), -1.0, 1.0) * 0.12
        a_base[1] = np.clip(base_err[1] / max(self.env.max_vy * self.env.dt, 1e-8), -1.0, 1.0) * 0.08
        a_base[2] = np.clip(base_err[2] / max(self.env.max_wz * self.env.dt, 1e-8), -1.0, 1.0) * 0.12

        a_arm = np.clip(
            arm_err / max(self.env.max_arm_delta_per_step, 1e-8),
            -1.0,
            1.0
        ).astype(np.float32) * 0.12

        return np.concatenate([a_base, a_arm], axis=0)


def smooth_action(prev_a, a_raw, tau):
    return (1.0 - tau) * prev_a + tau * a_raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--tau", type=float, default=0.20)
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = TidybotDoorOpenEnvV2(
        args.xml,
        render_mode="human" if args.render else None,
    )
    env.max_steps = int(args.max_steps)

    teacher = ScriptedStage2Coordinated(env)

    states = []
    actions = []
    rewards = []
    dones = []
    success = []
    episode_id = []
    timestep = []
    handle_dist = []
    door_q = []
    progress = []
    base_progress = []
    arm_progress = []
    source = []

    success_eps = 0
    ep_final_progress = []
    ep_final_door_q = []
    ep_steps = []

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        steps = 0
        a_smooth = np.zeros(10, dtype=np.float32)

        while not done:
            a_raw = teacher.action()
            a_smooth = smooth_action(a_smooth, a_raw, args.tau).astype(np.float32)

            next_obs, reward, terminated, truncated, info = env.step(a_smooth)
            done = bool(terminated or truncated)

            states.append(obs.copy())
            actions.append(a_smooth.copy())
            rewards.append(float(reward))
            dones.append(bool(done))
            success.append(int(info.get("success", False)))
            episode_id.append(ep)
            timestep.append(steps)
            handle_dist.append(float(info.get("handle_dist", np.nan)))
            door_q.append(float(info.get("door_q", np.nan)))
            progress.append(float(info.get("progress", np.nan)))
            base_progress.append(float(info.get("base_progress", np.nan)))
            arm_progress.append(float(info.get("arm_progress", np.nan)))
            source.append("teacher_stage2_coordinated")

            obs = next_obs
            steps += 1

        ok = bool(info.get("success", False))
        success_eps += int(ok)
        ep_final_progress.append(float(info.get("progress", np.nan)))
        ep_final_door_q.append(float(info.get("door_q", np.nan)))
        ep_steps.append(steps)

        print(
            f"[record] ep={ep+1}/{args.episodes} "
            f"steps={steps} success={ok} "
            f"progress={float(info.get('progress', np.nan)):.4f} "
            f"door_q={float(info.get('door_q', np.nan)):.4f} "
            f"handle_dist={float(info.get('handle_dist', np.nan)):.4f}"
        )

    np.savez_compressed(
        out_path,
        state=np.asarray(states, dtype=np.float32),
        action=np.asarray(actions, dtype=np.float32),
        reward=np.asarray(rewards, dtype=np.float32),
        done=np.asarray(dones, dtype=np.bool_),
        success=np.asarray(success, dtype=np.int8),
        episode_id=np.asarray(episode_id, dtype=np.int32),
        timestep=np.asarray(timestep, dtype=np.int32),
        handle_dist=np.asarray(handle_dist, dtype=np.float32),
        door_q=np.asarray(door_q, dtype=np.float32),
        progress=np.asarray(progress, dtype=np.float32),
        base_progress=np.asarray(base_progress, dtype=np.float32),
        arm_progress=np.asarray(arm_progress, dtype=np.float32),
        source=np.asarray(source),
    )

    print(
        f"[summary] success_eps={success_eps}/{args.episodes} "
        f"mean_final_progress={np.mean(ep_final_progress):.4f} "
        f"mean_final_door_q={np.mean(ep_final_door_q):.4f} "
        f"mean_steps={np.mean(ep_steps):.1f}"
    )
    print(f"[done] saved={out_path}")

    env.close()


if __name__ == "__main__":
    main()