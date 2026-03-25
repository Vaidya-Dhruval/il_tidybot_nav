import argparse
from pathlib import Path
import numpy as np
import mujoco

from tidybot_door_open_env_v2 import TidybotDoorOpenEnvV2


class ScriptedFullTaskTeacher:
    """
    Full scripted teacher for the final task:

    Phase 0:
        base parks near the handle

    Phase 1:
        arm reaches the handle region

    Phase 2:
        base + arm coordinated right-to-left sweep
    """

    def __init__(self, env):
        self.env = env

        self.kx = 1.4
        self.ky = 1.4
        self.kth = 1.2

        self.damping = 0.16
        self.max_cart_vel = 0.06
        self.max_joint_delta = 0.012

    def _arm_dls_to_world_target(self, target_world):
        ee = self.env._get_ee_world()
        err = target_world - ee

        v_des = 4.0 * err
        v_norm = float(np.linalg.norm(v_des))
        if v_norm > self.max_cart_vel:
            v_des *= self.max_cart_vel / max(v_norm, 1e-8)

        jacp = np.zeros((3, self.env.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.env.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.env.model, self.env.data, jacp, jacr, self.env.ee_site)

        J = jacp[:, self.env.arm_dadr]
        A = J @ J.T + (self.damping ** 2) * np.eye(3, dtype=np.float64)
        J_pinv = J.T @ np.linalg.solve(A, np.eye(3, dtype=np.float64))

        dq = J_pinv @ v_des
        dq = np.clip(dq, -self.max_joint_delta, self.max_joint_delta)
        a_arm = dq / max(self.env.max_arm_delta_per_step, 1e-8)
        return np.clip(a_arm, -1.0, 1.0).astype(np.float32)

    def action(self):
        a_base = np.zeros(3, dtype=np.float32)
        a_arm = np.zeros(7, dtype=np.float32)

        if self.env.phase == 0:
            base_now = self.env._get_base_xyth()

            dx = float(self.env.park_pose[0] - base_now[0])
            dy = float(self.env.park_pose[1] - base_now[1])
            yaw_err = ((float(self.env.park_pose[2] - base_now[2]) + np.pi) % (2 * np.pi)) - np.pi

            a_base[0] = np.clip(dx / max(self.env.max_vx * self.env.dt, 1e-8), -1.0, 1.0) * 0.6
            a_base[1] = np.clip(dy / max(self.env.max_vy * self.env.dt, 1e-8), -1.0, 1.0) * 0.6
            a_base[2] = np.clip(yaw_err / max(self.env.max_wz * self.env.dt, 1e-8), -1.0, 1.0) * 0.5

        elif self.env.phase == 1:
            a_base[:] = 0.0
            a_arm = self._arm_dls_to_world_target(self.env.handle_target_world)

        else:
            support_now = self.env._support_target_pose()
            base_now = self.env._get_base_xyth()

            if support_now[0] < base_now[0] - 1e-4:
                a_base[0] = -0.30
            else:
                a_base[0] = 0.0

            ee_target = self.env._moving_ee_target()
            a_arm = self._arm_dls_to_world_target(ee_target)

        return np.concatenate([a_base, a_arm], axis=0)


def smooth_action(prev_a, a_raw, tau):
    return (1.0 - tau) * prev_a + tau * a_raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--max_steps", type=int, default=460)
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

    teacher = ScriptedFullTaskTeacher(env)

    states = []
    actions = []
    rewards = []
    dones = []
    success = []
    episode_id = []
    timestep = []
    phase = []
    d_park = []
    d_handle = []
    stage2_progress = []
    target_dist = []
    base_support_err = []
    phase0_done = []
    phase1_done = []
    phase0_complete_step = []
    phase1_complete_step = []
    source = []

    success_eps = 0
    ep_final_phase = []
    ep_final_stage2_progress = []
    ep_final_d_park = []
    ep_final_d_handle = []
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
            phase.append(int(info.get("phase", -1)))
            d_park.append(float(info.get("d_park", np.nan)))
            d_handle.append(float(info.get("d_handle", np.nan)))
            stage2_progress.append(float(info.get("stage2_progress", np.nan)))
            target_dist.append(float(info.get("target_dist", np.nan)))
            base_support_err.append(float(info.get("base_support_err", np.nan)))
            phase0_done.append(int(info.get("phase0_done", False)))
            phase1_done.append(int(info.get("phase1_done", False)))
            phase0_complete_step.append(int(info.get("phase0_complete_step", -1)))
            phase1_complete_step.append(int(info.get("phase1_complete_step", -1)))
            source.append("teacher_full_task")

            obs = next_obs
            steps += 1

        ok = bool(info.get("success", False))
        success_eps += int(ok)
        ep_final_phase.append(int(info.get("phase", -1)))
        ep_final_stage2_progress.append(float(info.get("stage2_progress", np.nan)))
        ep_final_d_park.append(float(info.get("d_park", np.nan)))
        ep_final_d_handle.append(float(info.get("d_handle", np.nan)))
        ep_steps.append(steps)

        print(
            f"[record] ep={ep+1}/{args.episodes} "
            f"steps={steps} success={ok} "
            f"phase={int(info.get('phase', -1))} "
            f"phase0_done={bool(info.get('phase0_done', False))} "
            f"phase1_done={bool(info.get('phase1_done', False))} "
            f"stage2_progress={float(info.get('stage2_progress', np.nan)):.4f} "
            f"target_dist={float(info.get('target_dist', np.nan)):.4f} "
            f"base_support_err={float(info.get('base_support_err', np.nan)):.4f}"
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
        phase=np.asarray(phase, dtype=np.int32),
        d_park=np.asarray(d_park, dtype=np.float32),
        d_handle=np.asarray(d_handle, dtype=np.float32),
        stage2_progress=np.asarray(stage2_progress, dtype=np.float32),
        target_dist=np.asarray(target_dist, dtype=np.float32),
        base_support_err=np.asarray(base_support_err, dtype=np.float32),
        phase0_done=np.asarray(phase0_done, dtype=np.int8),
        phase1_done=np.asarray(phase1_done, dtype=np.int8),
        phase0_complete_step=np.asarray(phase0_complete_step, dtype=np.int32),
        phase1_complete_step=np.asarray(phase1_complete_step, dtype=np.int32),
        source=np.asarray(source),
    )

    print(
        f"[summary] success_eps={success_eps}/{args.episodes} "
        f"mean_final_phase={np.mean(ep_final_phase):.3f} "
        f"mean_final_stage2_progress={np.mean(ep_final_stage2_progress):.4f} "
        f"mean_final_d_park={np.mean(ep_final_d_park):.4f} "
        f"mean_final_d_handle={np.mean(ep_final_d_handle):.4f} "
        f"mean_steps={np.mean(ep_steps):.1f}"
    )
    print(f"[done] saved={out_path}")

    env.close()


if __name__ == "__main__":
    main()