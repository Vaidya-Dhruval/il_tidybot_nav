import time
import numpy as np

from tidybot_door_open_env_v2 import TidybotDoorOpenEnvV2


class ScriptedStage2Coordinated:
    """
    Slower scripted coordinated opening:
    - drive base toward goal support pose safely
    - drive arm toward open arm pose
    - intentionally slow so motion is visible
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

        # slower proportional support
        a_base[0] = np.clip(base_err[0] / max(self.env.max_vx * self.env.dt, 1e-8), -1.0, 1.0) * 0.12
        a_base[1] = np.clip(base_err[1] / max(self.env.max_vy * self.env.dt, 1e-8), -1.0, 1.0) * 0.08
        a_base[2] = np.clip(base_err[2] / max(self.env.max_wz * self.env.dt, 1e-8), -1.0, 1.0) * 0.12

        a_arm = np.clip(
            arm_err / max(self.env.max_arm_delta_per_step, 1e-8),
            -1.0,
            1.0
        ).astype(np.float32) * 0.12

        return np.concatenate([a_base, a_arm], axis=0)


def main():
    env = TidybotDoorOpenEnvV2("../../tidybot_with_cell.xml", render_mode="human")
    env.max_steps = 1000

    ctrl = ScriptedStage2Coordinated(env)

    obs, info = env.reset()
    print(
        f"[debug] start_handle_dist={info['handle_dist']:.4f} "
        f"start_door_q={info['door_q']:.4f} "
        f"start_progress={info['progress']:.4f}"
    )

    for step in range(env.max_steps):
        action = ctrl.action()
        obs, reward, term, trunc, info = env.step(action)

        if step % 10 == 0:
            print(
                f"[step {step:04d}] "
                f"handle_dist={float(info['handle_dist']):.4f} "
                f"door_q={float(info['door_q']):.4f} "
                f"progress={float(info['progress']):.4f} "
                f"base_prog={float(info['base_progress']):.4f} "
                f"arm_prog={float(info['arm_progress']):.4f} "
                f"success={info['success']}"
            )

        if term or trunc:
            print(
                f"[done] step={step} "
                f"handle_dist={float(info['handle_dist']):.4f} "
                f"door_q={float(info['door_q']):.4f} "
                f"progress={float(info['progress']):.4f} "
                f"success={info['success']}"
            )
            break

        time.sleep(0.05)

    env.close()


if __name__ == "__main__":
    main()