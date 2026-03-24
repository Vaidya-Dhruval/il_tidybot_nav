import numpy as np
from tidybot_arm_reach_env_v2 import ArmManipulationEnv

MODEL_PATH = "/home/vsp-dv/colcon_rox_ws/src/rox-description/rox_assets/stanford_tidybot/tidybot_with_cell.xml"


def run_direction_test(env, direction, steps=150):
    obs, info = env.reset()
    print("\n==============================")
    print(f"Testing direction: {direction}")
    print("reset info:", info)
    print("==============================")

    for t in range(steps):
        ee = env._get_ee_world()
        handle = env._get_handle_target_world()

        vec = handle - ee
        dist = np.linalg.norm(vec)

        if dist > 0.05:
            target_dir = vec / (dist + 1e-6)
        else:
            target_dir = np.array(direction, dtype=np.float32)

        action = np.zeros(7, dtype=np.float32)
        action[1] = target_dir[0] * 0.8
        action[2] = target_dir[1] * 0.8
        action[3] = target_dir[2] * 0.6

        obs, reward, terminated, truncated, info = env.step(action)

        print(
            f"t={t:03d} "
            f"dist={info.get('handle_distance', -1):.4f} "
            f"door={info.get('door_qpos', -1):.4f} "
            f"progress={info.get('door_progress', 0.0):.5f}"
        )

        if info.get("door_qpos", 0.0) > 0.05:
            print("✅ DOOR IS MOVING")
            return True

        if terminated or truncated:
            break

    print("❌ no door movement")
    return False


def main():
    env = ArmManipulationEnv(
        model_path=MODEL_PATH,
        render_mode="human",
        task_mode="door_open",
    )

    directions = [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
    ]

    success_dir = None
    for d in directions:
        ok = run_direction_test(env, d)
        if ok:
            success_dir = d
            break

    print("\n==============================")
    print("SUCCESS DIRECTION:", success_dir)
    print("==============================")

    env.close()


if __name__ == "__main__":
    main()