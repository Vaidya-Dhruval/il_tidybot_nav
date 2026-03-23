import numpy as np
import mujoco


class ArmTeacherDLS:
    def __init__(
        self,
        model,
        data,
        ee_site_id: int,
        target_site_id: int,
        arm_dof_indices,
        get_target_fn=None,
        kp_xyz: float = 10.0,
        damping: float = 0.05,
        max_cart_vel: float = 0.40,
        max_joint_cmd: float = 1.5,
        stop_radius: float = 0.06,
        posture_gain: float = 0.25,
        q_nominal=None,
    ):
        self.model = model
        self.data = data
        self.ee_site_id = ee_site_id
        self.target_site_id = target_site_id
        self.arm_dof_indices = np.asarray(arm_dof_indices, dtype=np.int32)
        self.get_target_fn = get_target_fn

        self.kp_xyz = float(kp_xyz)
        self.damping = float(damping)
        self.max_cart_vel = float(max_cart_vel)
        self.max_joint_cmd = float(max_joint_cmd)
        self.stop_radius = float(stop_radius)
        self.posture_gain = float(posture_gain)

        if q_nominal is None:
            self.q_nominal = np.array([0.0, 0.4, 0.0, 1.0, 0.0, 0.8, 0.0], dtype=np.float64)
        else:
            self.q_nominal = np.asarray(q_nominal, dtype=np.float64)

    def act(self, obs=None):
        ee = self.data.site_xpos[self.ee_site_id].copy()
        if self.get_target_fn is not None:
            target = self.get_target_fn().copy()
        else:
            target = self.data.site_xpos[self.target_site_id].copy()

        err = target - ee
        dist = float(np.linalg.norm(err))

        if dist < self.stop_radius:
            return np.zeros(len(self.arm_dof_indices), dtype=np.float32)

        v_des = self.kp_xyz * err
        v_norm = float(np.linalg.norm(v_des))
        if v_norm > self.max_cart_vel:
            v_des = v_des * (self.max_cart_vel / max(v_norm, 1e-8))

        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)

        J = jacp[:, self.arm_dof_indices]

        A = J @ J.T + (self.damping ** 2) * np.eye(3, dtype=np.float64)
        J_pinv = J.T @ np.linalg.solve(A, np.eye(3, dtype=np.float64))

        dq_task = J_pinv @ v_des

        q_now = self.data.qpos[self.arm_dof_indices].astype(np.float64)
        dq_posture = self.posture_gain * (self.q_nominal - q_now)

        N = np.eye(len(self.arm_dof_indices), dtype=np.float64) - J_pinv @ J
        dq = dq_task + N @ dq_posture

        dq = np.clip(dq, -self.max_joint_cmd, self.max_joint_cmd)
        return dq.astype(np.float32)