import time
import numpy as np
import mujoco

from tidybot_door_open_env_v1 import TidybotDoorOpenEnvV1


class DirectionSweepController:
    def __init__(
        self,
        env,
        kp_xyz=6.0,
        damping=0.08,
        max_cart_vel=0.18,
        max_joint_delta=0.02,
        contact_switch_radius=0.10,
        posture_gain=0.02,
        q_nominal=None,
    ):
        self.env = env
        self.kp_xyz = float(kp_xyz)
        self.damping = float(damping)
        self.max_cart_vel = float(max_cart_vel)
        self.max_joint_delta = float(max_joint_delta)
        self.contact_switch_radius = float(contact_switch_radius)
        self.posture_gain = float(posture_gain)

        if q_nominal is None:
            self.q_nominal = np.array([0.0, 0.4, 0.0, 1.0, 0.0, 0.8, 0.0], dtype=np.float64)
        else:
            self.q_nominal = np.asarray(q_nominal, dtype=np.float64)

    def _dls_action_to_world_target(self, target_world):
        model = self.env.model
        data = self.env.data

        ee = data.site_xpos[self.env.ee_site].copy()
        err = target_world - ee
        dist = float(np.linalg.norm(err))

        v_des = self.kp_xyz * err
        v_norm = float(np.linalg.norm(v_des))
        if v_norm > self.max_cart_vel:
            v_des *= self.max_cart_vel / max(v_norm, 1e-8)

        jacp = np.zeros((3, model.nv), dtype=np.float64)
        jacr = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jacSite(model, data, jacp, jacr, self.env.ee_site)

        J = jacp[:, self.env.arm_dadr]

        A = J @ J.T + (self.damping ** 2) * np.eye(3, dtype=np.float64)
        J_pinv = J.T @ np.linalg.solve(A, np.eye(3, dtype=np.float64))

        dq_task = J_pinv @ v_des

        q_now = np.array([data.qpos[i] for i in self.env.arm_qadr], dtype=np.float64)
        dq_posture = self.posture_gain * (self.q_nominal - q_now)

        N = np.eye(len(self.env.arm_dadr), dtype=np.float64) - J_pinv @ J
        dq = dq_task + N @ dq_posture

        dq = np.clip(dq, -self.max_joint_delta, self.max_joint_delta)
        a = dq / max(self.env.max_arm_delta_per_step, 1e-8)
        return np.clip(a, -1.0, 1.0).astype(np.float32), dist

    def approach_handle_action(self):
        handle = self.env._get_handle_world()
        return self._dls_action_to_world_target(handle)

    def biased_contact_action(self, bias_world):
        handle = self.env._get_handle_world()
        target = handle + np.asarray(bias_world, dtype=np.float64)
        return self._dls_action_to_world_target(target)


def find_door_actuator_id(env):
    names_to_try = [
        "cell_door_actuator",
        "door_actuator",
        "cell_door_joint",
    ]
    for name in names_to_try:
        aid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid >= 0:
            return int(aid), name
    return None, None


def run_trial(env, ctrl, bias_name, bias_vec, approach_max_steps=250, push_steps=220, render_sleep=0.02):
    obs, info = env.reset()

    door_aid, door_aname = find_door_actuator_id(env)
    if door_aid is not None:
        try:
            door_ctrl0 = float(env.data.ctrl[door_aid])
        except Exception:
            door_ctrl0 = float("nan")
    else:
        door_ctrl0 = float("nan")

    start_handle_dist = float(info["handle_dist"])
    start_door_q = float(info["door_q"])

    phase = "approach"
    contact_step = None

    best_handle = start_handle_dist
    best_door_q = start_door_q

    # ---- phase 1: approach handle
    for step in range(approach_max_steps):
        action, ctrl_dist = ctrl.approach_handle_action()
        obs, reward, term, trunc, info = env.step(action)

        best_handle = min(best_handle, float(info["handle_dist"]))
        best_door_q = max(best_door_q, float(info["door_q"]))

        if step % 20 == 0:
            print(
                f"[{bias_name}][approach {step:04d}] "
                f"handle_dist={float(info['handle_dist']):.4f} "
                f"door_q={float(info['door_q']):.4f} "
                f"contact={int(info['contact'])}"
            )

        if int(info["contact"]) == 1:
            contact_step = step
            phase = "push"
            break

    if phase != "push":
        return {
            "bias": bias_name,
            "contact_reached": False,
            "contact_step": -1,
            "start_handle_dist": start_handle_dist,
            "best_handle_dist": best_handle,
            "start_door_q": start_door_q,
            "best_door_q": best_door_q,
            "final_door_q": float(info["door_q"]),
            "door_delta": float(info["door_q"] - start_door_q),
            "door_actuator_name": door_aname,
            "door_actuator_ctrl0": door_ctrl0,
        }

    # ---- phase 2: biased push / drag
    push_start_q = float(info["door_q"])
    push_contact_maintained = True

    for step in range(push_steps):
        action, ctrl_dist = ctrl.biased_contact_action(bias_vec)
        obs, reward, term, trunc, info = env.step(action)

        best_handle = min(best_handle, float(info["handle_dist"]))
        best_door_q = max(best_door_q, float(info["door_q"]))

        if int(info["contact"]) == 0:
            push_contact_maintained = False

        if step % 20 == 0:
            door_ctrl_now = float(env.data.ctrl[door_aid]) if door_aid is not None else float("nan")
            print(
                f"[{bias_name}][push {step:04d}] "
                f"handle_dist={float(info['handle_dist']):.4f} "
                f"door_q={float(info['door_q']):.4f} "
                f"best_door_q={best_door_q:.4f} "
                f"door_ctrl={door_ctrl_now:.4f} "
                f"contact={int(info['contact'])}"
            )

        if env.render_mode == "human":
            time.sleep(render_sleep)

    final_door_q = float(info["door_q"])

    return {
        "bias": bias_name,
        "contact_reached": True,
        "contact_step": int(contact_step),
        "start_handle_dist": start_handle_dist,
        "best_handle_dist": best_handle,
        "start_door_q": start_door_q,
        "push_start_q": push_start_q,
        "best_door_q": best_door_q,
        "final_door_q": final_door_q,
        "door_delta": final_door_q - start_door_q,
        "best_door_delta": best_door_q - start_door_q,
        "push_contact_maintained": push_contact_maintained,
        "door_actuator_name": door_aname,
        "door_actuator_ctrl0": door_ctrl0,
    }


def main():
    env = TidybotDoorOpenEnvV1("../../tidybot_with_cell.xml", render_mode="human")
    env.max_steps = 1200

    ctrl = DirectionSweepController(
        env,
        kp_xyz=6.0,
        damping=0.08,
        max_cart_vel=0.18,
        max_joint_delta=0.02,
        contact_switch_radius=0.10,
        posture_gain=0.02,
    )

    # Candidate push / drag directions in world frame
    sweep_dirs = [
        ("push_x_pos", np.array([+0.03,  0.00,  0.00], dtype=np.float64)),
        ("push_x_neg", np.array([-0.03,  0.00,  0.00], dtype=np.float64)),
        ("push_y_pos", np.array([ 0.00, +0.03,  0.00], dtype=np.float64)),
        ("push_y_neg", np.array([ 0.00, -0.03,  0.00], dtype=np.float64)),
        ("push_xy_pospos", np.array([+0.03, +0.03, 0.00], dtype=np.float64)),
        ("push_xy_negpos", np.array([-0.03, +0.03, 0.00], dtype=np.float64)),
        ("push_xy_posneg", np.array([+0.03, -0.03, 0.00], dtype=np.float64)),
        ("push_xy_negneg", np.array([-0.03, -0.03, 0.00], dtype=np.float64)),
    ]

    results = []

    print("\n[direction sweep] starting...\n")
    for name, vec in sweep_dirs:
        print(f"\n===== TRIAL: {name}  bias={vec.tolist()} =====")
        res = run_trial(env, ctrl, name, vec)
        results.append(res)
        print(f"[trial summary] {res}")

    print("\n===== FINAL SUMMARY =====")
    results_sorted = sorted(results, key=lambda r: r.get("best_door_delta", -1e9), reverse=True)
    for r in results_sorted:
        print(
            f"{r['bias']}: "
            f"contact_reached={r['contact_reached']} "
            f"best_door_delta={r.get('best_door_delta', 0.0):+.6f} "
            f"final_door_q={r['final_door_q']:.6f} "
            f"best_handle_dist={r['best_handle_dist']:.4f} "
            f"door_actuator={r['door_actuator_name']} "
            f"door_ctrl0={r['door_actuator_ctrl0']}"
        )

    env.close()


if __name__ == "__main__":
    main()