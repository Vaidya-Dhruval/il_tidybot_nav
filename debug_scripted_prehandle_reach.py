import time
import numpy as np

from tidybot_arm_reach_env_v2 import ArmReachEnvV2
from scripted_prehandle_reach import ScriptedPrehandleReach


def main():
    env = ArmReachEnvV2("../../tidybot_with_cell.xml", render_mode="human")
    env.max_steps = 1500

    controller = ScriptedPrehandleReach(
        env,
        prehandle_offset=np.array([0.0, -0.18, 0.0], dtype=np.float64),
        kp_xyz=8.0,
        damping=0.06,
        max_cart_vel=0.25,
        max_joint_delta=0.03,
        stop_radius=0.08,
        posture_gain=0.02,
    )

    obs, info = env.reset()
    print(f"[debug] start_dist={float(info['distance']):.4f}")

    for step in range(env.max_steps):
        action, ctrl_dist = controller.step_action()
        obs, reward, term, trunc, info = env.step(action)

        if step % 20 == 0:
            tgt = controller.target_world()
            ee = env.data.site_xpos[env.ee_site].copy()
            print(
                f"[step {step:04d}] "
                f"env_dist={float(info['distance']):.4f} "
                f"ctrl_dist={ctrl_dist:.4f} "
                f"ee=({ee[0]:+.3f},{ee[1]:+.3f},{ee[2]:+.3f}) "
                f"tgt=({tgt[0]:+.3f},{tgt[1]:+.3f},{tgt[2]:+.3f}) "
                f"best={float(info['best_distance']):.4f} "
                f"succ={bool(info['success'])}"
            )

        if term or trunc:
            print(
                f"[done] step={step} "
                f"final_dist={float(info['distance']):.4f} "
                f"best={float(info['best_distance']):.4f} "
                f"success={bool(info['success'])}"
            )
            break

        time.sleep(0.02)


if __name__ == "__main__":
    main()