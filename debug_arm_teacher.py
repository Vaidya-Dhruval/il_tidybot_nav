import time
import numpy as np

from tidybot_arm_reach_env_v2 import ArmReachEnvV2
from arm_teacher import ArmTeacherDLS


def main():
    env = ArmReachEnvV2("../../tidybot_with_cell.xml", render_mode="human")

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

    obs, info = env.reset()
    print(f"[debug] start_dist={float(info['distance']):.4f}")

    for step in range(400):
        a = teacher.act(obs)
        obs, r, term, trunc, info = env.step(a)

        if step % 20 == 0:
            print(
                f"[step {step:03d}] "
                f"dist={float(info['distance']):.4f} "
                f"best={float(info['best_distance']):.4f} "
                f"success={bool(info['success'])}"
            )

        if term or trunc:
            print(
                f"[done] step={step} "
                f"dist={float(info['distance']):.4f} "
                f"best={float(info['best_distance']):.4f} "
                f"success={bool(info['success'])}"
            )
            break

        time.sleep(0.02)


if __name__ == "__main__":
    main()