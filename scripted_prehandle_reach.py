import numpy as np
import mujoco


class ScriptedPrehandleReach:
    def __init__(
        self,
        env,
        prehandle_offset=np.array([0.0, -0.18, 0.0], dtype=np.float64),
        kp_xyz: float = 8.0,
        damping: float = 0.06,
        max_cart_vel: float = 0.25,
        max_joint_delta: float = 0.03,
        stop_radius: float = 0.08,
        posture_gain: float = 0.02,
        q_nominal=None,
    ):
        self.env = env
        self.prehandle_offset = np.asarray(prehandle_offset, dtype=np.float64)
        self.kp_xyz = float(kp_xyz)
        self.damping = float(damping)
        self.max_cart_vel = float(max_cart_vel)
        self.max_joint_delta = float(max_joint_delta)
        self.stop_radius = float(stop_radius)
        self.posture_gain = float(posture_gain)

        if q_nominal is None:
            self.q_nominal = np.array([0.0, 0.4, 0.0, 1.0, 0.0, 0.8, 0.0], dtype=np.float64)
        else:
            self.q_nominal = np.asarray(q_nominal, dtype=np.float64)

    def target_world(self):
        handle = self.env.data.site_xpos[self.env.target_site].copy()
        return handle + self.prehandle_offset

    def step_action(self):
        model = self.env.model
        data = self.env.data

        ee = data.site_xpos[self.env.ee_site].copy()
        target = self.target_world()

        err = target - ee
        dist = float(np.linalg.norm(err))

        if dist < self.stop_radius:
            return np.zeros(7, dtype=np.float32), dist

        v_des = self.kp_xyz * err
        v_norm = float(np.linalg.norm(v_des))
        if v_norm > self.max_cart_vel:
            v_des *= self.max_cart_vel / max(v_norm, 1e-8)

        jacp = np.zeros((3, model.nv), dtype=np.float64)
        jacr = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jacSite(model, data, jacp, jacr, self.env.ee_site)

        dofs = np.asarray(self.env.arm_dadr, dtype=np.int32)
        J = jacp[:, dofs]

        A = J @ J.T + (self.damping ** 2) * np.eye(3, dtype=np.float64)
        J_pinv = J.T @ np.linalg.solve(A, np.eye(3, dtype=np.float64))

        dq_task = J_pinv @ v_des

        q_now = np.array([data.qpos[i] for i in self.env.arm_qadr], dtype=np.float64)
        dq_posture = self.posture_gain * (self.q_nominal - q_now)
        N = np.eye(len(dofs), dtype=np.float64) - J_pinv @ J
        dq = dq_task + N @ dq_posture

        dq = np.clip(dq, -self.max_joint_delta, self.max_joint_delta)
        # convert to normalized action expected by env.step()
        a = dq / max(self.env.max_arm_delta_per_step, 1e-8)
        a = np.clip(a, -1.0, 1.0).astype(np.float32)
        return a, dist