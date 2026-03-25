import time
import numpy as np
import mujoco

from tidybot_door_open_env_v2 import TidybotDoorOpenEnvV2


class ScriptedStage2Coordinated:
    """
    Very simple coordinated right-to-left sweep:
    - base shifts left
    - arm tracks fixed world-frame leftward sweep target
    """

    def __init__(self, env):
        self.env = env
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

        support_now = self.env._support_target_pose()
        base_now = self.env._get_base_xyth()
        if support_now[0] < base_now[0] - 1e-4:
            a_base[0] = -0.30
        else:
            a_base[0] = 0.0

        ee_target = self.env._moving_ee_target()
        a_arm = self._arm_dls_to_world_target(ee_target)

        return np.concatenate([a_base, a_arm], axis=0)


def main():
    env = TidybotDoorOpenEnvV2("../../tidybot_with_cell.xml", render_mode="human")
    env.max_steps = 260

    ctrl = ScriptedStage2Coordinated(env)

    obs, info = env.reset()
    print(
        f"[debug] start_target_dist={info['target_dist']:.4f} "
        f"start_progress={info['progress']:.4f}"
    )

    for step in range(env.max_steps):
        action = ctrl.action()
        obs, reward, term, trunc, info = env.step(action)

        if step % 10 == 0:
            print(
                f"[step {step:04d}] "
                f"target_dist={float(info['target_dist']):.4f} "
                f"progress={float(info['progress']):.4f} "
                f"base_x_err={float(info['base_x_err']):.4f} "
                f"success={info['success']}"
            )

        if term or trunc:
            print(
                f"[done] step={step} "
                f"target_dist={float(info['target_dist']):.4f} "
                f"progress={float(info['progress']):.4f} "
                f"success={info['success']}"
            )
            break

        time.sleep(0.05)

    env.close()


if __name__ == "__main__":
    main()