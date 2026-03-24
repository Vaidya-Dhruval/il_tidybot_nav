import os
import math
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def lerp(a, b, t):
    return (1.0 - t) * a + t * b


class TidybotDoorOpenEnvV2(gym.Env):
    """
    Stage 2:
    Coordinated trajectory-tracking opening phase.

    Abstraction:
    - Grasp is assumed / abstracted.
    - Robot starts in a grasp-ready pose near the door.
    - Base and arm should coordinate from a start pose to an open pose.
    - Door progress is tied to coordinated motion completion.
    - Base is constrained to a safe corridor so it does not move into the cell.
    """

    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(self, model_path: str, render_mode=None):
        self.render_mode = render_mode
        self.viewer = None

        xml_path = os.path.abspath(model_path)
        xml_dir = os.path.dirname(xml_path)

        old_cwd = os.getcwd()
        os.chdir(xml_dir)
        try:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
        finally:
            os.chdir(old_cwd)

        self.data = mujoco.MjData(self.model)

        # ids
        self.ee_site = self._require_site("pinch_site")
        self.handle_site = self._require_site("cell_door_site")
        self.door_joint = self._require_joint("cell_door_joint")

        self.base_joint_names = ["joint_x", "joint_y", "joint_th"]
        self.base_joint_ids = [self._require_joint(n) for n in self.base_joint_names]
        self.base_qadr = [int(self.model.jnt_qposadr[j]) for j in self.base_joint_ids]
        self.base_dadr = [int(self.model.jnt_dofadr[j]) for j in self.base_joint_ids]

        self.base_actuator_names = ["joint_x", "joint_y", "joint_th"]
        self.base_actuator_ids = [self._require_actuator(n) for n in self.base_actuator_names]

        self.arm_joint_names = [
            "joint_1", "joint_2", "joint_3", "joint_4",
            "joint_5", "joint_6", "joint_7",
        ]
        self.arm_joint_ids = [self._require_joint(n) for n in self.arm_joint_names]
        self.arm_qadr = [int(self.model.jnt_qposadr[j]) for j in self.arm_joint_ids]
        self.arm_dadr = [int(self.model.jnt_dofadr[j]) for j in self.arm_joint_ids]

        self.arm_actuator_ids = []
        for jid, jname in zip(self.arm_joint_ids, self.arm_joint_names):
            found = None
            for aid in range(self.model.nu):
                trn_jid = int(self.model.actuator_trnid[aid, 0])
                if trn_jid == jid:
                    found = aid
                    break
            if found is None:
                raise RuntimeError(f"No actuator found that drives joint '{jname}'")
            self.arm_actuator_ids.append(found)

        print("[stage2 base actuators]")
        for name, aid in zip(self.base_actuator_names, self.base_actuator_ids):
            aname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
            print(f"  base={name} -> actuator={aid} name={aname}")

        print("[stage2 arm joints]")
        for jname, jid, aid in zip(self.arm_joint_names, self.arm_joint_ids, self.arm_actuator_ids):
            aname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
            print(f"  joint={jname} jid={jid} -> actuator={aid} name={aname}")

        self.door_qadr = int(self.model.jnt_qposadr[self.door_joint])
        self.door_dadr = int(self.model.jnt_dofadr[self.door_joint])
        self.door_range = self.model.jnt_range[self.door_joint].copy()
        self.door_lo = float(self.door_range[0])
        self.door_hi = float(self.door_range[1])

        # timing
        self.physics_dt = 0.001
        self.dt = 0.05
        self.n_substeps = max(1, int(round(self.dt / self.physics_dt)))
        self.model.opt.timestep = float(self.physics_dt)

        # control
        self.max_steps = 500
        self.max_vx = 0.02
        self.max_vy = 0.015
        self.max_wz = 0.05
        self.max_arm_delta_per_step = 0.02

        self._base_target = np.zeros(3, dtype=np.float64)
        self._arm_target = np.zeros(7, dtype=np.float64)

        # safe staging geometry
        self.safe_yaw = math.pi / 2.0

        # start pose near handle but outside cell
        self.start_offset = np.array([0.0, -0.04, 0.0], dtype=np.float64)

        # coordinated end pose:
        # slight lateral/forward support without entering cell
        # keep y at or below start y
        self.open_base_delta = np.array([+0.06, -0.02, +0.08], dtype=np.float64)

        # corridor limits around start base pose
        self.max_support_dx = 0.10
        self.max_support_dy_outward = 0.08   # can move away from cell more
        self.max_support_dy_inward = 0.00    # cannot move into cell
        self.max_support_yaw = 0.20

        # arm start/end joint targets
        # start_arm_q will be taken from actual current q after reset
        self.open_arm_delta = np.array([+0.10, -0.08, +0.08, +0.05, 0.00, +0.06, 0.00], dtype=np.float64)

        # progress and success
        self.virtual_door_q = self.door_lo
        self.open_threshold = self.door_lo + 0.70 * (self.door_hi - self.door_lo)

        self.start_base_pose = np.zeros(3, dtype=np.float64)
        self.goal_base_pose = np.zeros(3, dtype=np.float64)
        self.start_arm_q = np.zeros(7, dtype=np.float64)
        self.goal_arm_q = np.zeros(7, dtype=np.float64)

        self.step_count = 0
        self.best_progress = 0.0

        # obs:
        # ee->handle xyz(3), dist(1),
        # arm q(7), arm qd(7),
        # door_q(1),
        # progress(1),
        # base error to goal in body frame dx/dy/sin/cos(4),
        # arm target error norm(1)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)

    # ------------------------------------------------------------------ helpers

    def _require_site(self, name: str) -> int:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            raise RuntimeError(f"Site '{name}' not found")
        return int(sid)

    def _require_joint(self, name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"Joint '{name}' not found")
        return int(jid)

    def _require_actuator(self, name: str) -> int:
        aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise RuntimeError(f"Actuator '{name}' not found")
        return int(aid)

    def _get_handle_world(self):
        return self.data.site_xpos[self.handle_site].copy()

    def _get_ee_world(self):
        return self.data.site_xpos[self.ee_site].copy()

    def _get_base_xyth(self):
        return np.array([
            float(self.data.qpos[self.base_qadr[0]]),
            float(self.data.qpos[self.base_qadr[1]]),
            float(self.data.qpos[self.base_qadr[2]]),
        ], dtype=np.float64)

    def _set_base_pose(self, x: float, y: float, yaw: float):
        self.data.qpos[self.base_qadr[0]] = float(x)
        self.data.qpos[self.base_qadr[1]] = float(y)
        self.data.qpos[self.base_qadr[2]] = float(wrap_to_pi(yaw))
        self.data.qvel[self.base_dadr[0]] = 0.0
        self.data.qvel[self.base_dadr[1]] = 0.0
        self.data.qvel[self.base_dadr[2]] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _hold_base(self):
        self.data.ctrl[self.base_actuator_ids[0]] = float(self._base_target[0])
        self.data.ctrl[self.base_actuator_ids[1]] = float(self._base_target[1])
        self.data.ctrl[self.base_actuator_ids[2]] = float(self._base_target[2])

    def _apply_arm_targets(self):
        for i, aid in enumerate(self.arm_actuator_ids):
            self.data.ctrl[aid] = float(self._arm_target[i])

    def _set_arm_to_current(self):
        self._arm_target[:] = np.array([self.data.qpos[i] for i in self.arm_qadr], dtype=np.float64)

    def _mirror_virtual_door(self):
        self.data.qpos[self.door_qadr] = float(self.virtual_door_q)
        self.data.qvel[self.door_dadr] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _get_stage2_start_pose_world(self):
        handle = self._get_handle_world()
        x = float(handle[0] + self.start_offset[0])
        y = float(handle[1] + self.start_offset[1])
        yaw = float(self.safe_yaw)
        return np.array([x, y, yaw], dtype=np.float64)

    def _compute_goal_poses(self):
        self.start_base_pose = self._get_stage2_start_pose_world().copy()
        self.goal_base_pose = self.start_base_pose + self.open_base_delta
        # safety: do not let goal move inward into cell
        self.goal_base_pose[1] = min(self.goal_base_pose[1], self.start_base_pose[1])

        self.start_arm_q = np.array([self.data.qpos[i] for i in self.arm_qadr], dtype=np.float64)
        self.goal_arm_q = self.start_arm_q + self.open_arm_delta

        # clamp goal arm to joint ranges
        for i, jid in enumerate(self.arm_joint_ids):
            lo, hi = [float(x) for x in self.model.jnt_range[jid]]
            if hi > lo:
                self.goal_arm_q[i] = np.clip(self.goal_arm_q[i], lo, hi)

    def _enforce_safe_base_corridor(self):
        bx, by, bth = self._base_target
        sx, sy, sth = self.start_base_pose

        dx = bx - sx
        dy = by - sy
        dyaw = wrap_to_pi(bth - sth)

        dx = np.clip(dx, -self.max_support_dx, self.max_support_dx)
        # outward is more negative y, inward is more positive y
        dy = np.clip(dy, -self.max_support_dy_outward, self.max_support_dy_inward)
        dyaw = np.clip(dyaw, -self.max_support_yaw, self.max_support_yaw)

        self._base_target[0] = sx + dx
        self._base_target[1] = sy + dy
        self._base_target[2] = wrap_to_pi(sth + dyaw)

    def _handle_dist(self):
        return float(np.linalg.norm(self._get_handle_world() - self._get_ee_world()))

    def _progress_metrics(self):
        base_now = self._get_base_xyth()
        arm_now = np.array([self.data.qpos[i] for i in self.arm_qadr], dtype=np.float64)

        base_num = np.linalg.norm(base_now - self.start_base_pose)
        base_den = max(np.linalg.norm(self.goal_base_pose - self.start_base_pose), 1e-8)
        base_prog = np.clip(base_num / base_den, 0.0, 1.0)

        arm_num = np.linalg.norm(arm_now - self.start_arm_q)
        arm_den = max(np.linalg.norm(self.goal_arm_q - self.start_arm_q), 1e-8)
        arm_prog = np.clip(arm_num / arm_den, 0.0, 1.0)

        progress = float(np.clip(0.5 * base_prog + 0.5 * arm_prog, 0.0, 1.0))
        return progress, base_prog, arm_prog

    def _get_obs(self) -> np.ndarray:
        ee = self._get_ee_world()
        handle = self._get_handle_world()
        d = handle - ee
        dist = float(np.linalg.norm(d))

        q = np.array([self.data.qpos[i] for i in self.arm_qadr], dtype=np.float32)
        qd = np.array([self.data.qvel[i] for i in self.arm_dadr], dtype=np.float32)

        progress, _, _ = self._progress_metrics()

        bx, by, bth = self._get_base_xyth()
        gx, gy, gth = self.goal_base_pose
        dx_w = gx - bx
        dy_w = gy - by
        c, s = math.cos(bth), math.sin(bth)
        dx_b = c * dx_w + s * dy_w
        dy_b = -s * dx_w + c * dy_w
        yaw_err = wrap_to_pi(gth - bth)

        arm_now = np.array([self.data.qpos[i] for i in self.arm_qadr], dtype=np.float64)
        arm_goal_err = float(np.linalg.norm(self.goal_arm_q - arm_now))

        return np.concatenate(
            [
                d.astype(np.float32),
                np.array([dist], dtype=np.float32),
                q,
                qd,
                np.array([self.virtual_door_q], dtype=np.float32),
                np.array([progress], dtype=np.float32),
                np.array([dx_b, dy_b, math.sin(yaw_err), math.cos(yaw_err)], dtype=np.float32),
                np.array([arm_goal_err], dtype=np.float32),
            ],
            axis=0,
        ).astype(np.float32)

    # ------------------------------------------------------------------ API

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        start_pose = self._get_stage2_start_pose_world()
        self._set_base_pose(*start_pose.tolist())
        self._base_target[:] = start_pose
        self._set_arm_to_current()
        self._compute_goal_poses()

        self.virtual_door_q = self.door_lo
        self.best_progress = 0.0
        self.step_count = 0

        self._mirror_virtual_door()

        if self.render_mode == "human" and self.viewer is None:
            from mujoco import viewer
            self.viewer = viewer.launch_passive(self.model, self.data)

        obs = self._get_obs()
        info = {
            "handle_dist": float(obs[3]),
            "door_q": float(self.virtual_door_q),
            "progress": float(obs[19]),
        }
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        a_base = action[:3]
        a_arm = action[3:]

        prev_progress, _, _ = self._progress_metrics()

        # base target update (guarded)
        bx, by, bth = self._base_target
        cy, sy = math.cos(bth), math.sin(bth)
        vx_body = self.max_vx * float(a_base[0])
        vy_body = self.max_vy * float(a_base[1])
        wz = self.max_wz * float(a_base[2])

        vx_w = cy * vx_body - sy * vy_body
        vy_w = sy * vx_body + cy * vy_body

        self._base_target[0] = bx + vx_w * self.dt
        self._base_target[1] = by + vy_w * self.dt
        self._base_target[2] = wrap_to_pi(bth + wz * self.dt)
        self._enforce_safe_base_corridor()

        # arm target update
        self._arm_target += self.max_arm_delta_per_step * a_arm.astype(np.float64)
        for i, jid in enumerate(self.arm_joint_ids):
            lo, hi = [float(x) for x in self.model.jnt_range[jid]]
            if hi > lo:
                self._arm_target[i] = np.clip(self._arm_target[i], lo, hi)

        # simulate
        for _ in range(self.n_substeps):
            self._hold_base()
            self._apply_arm_targets()
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        progress, base_prog, arm_prog = self._progress_metrics()
        self.best_progress = max(self.best_progress, progress)

        # door tied to motion completion
        self.virtual_door_q = lerp(self.door_lo, self.door_hi, progress)
        self._mirror_virtual_door()

        obs = self._get_obs()
        handle_dist = self._handle_dist()

        progress_gain = float(progress - prev_progress)

        # tracking penalties
        base_now = self._get_base_xyth()
        arm_now = np.array([self.data.qpos[i] for i in self.arm_qadr], dtype=np.float64)
        base_goal_err = float(np.linalg.norm(base_now - self.goal_base_pose))
        arm_goal_err = float(np.linalg.norm(arm_now - self.goal_arm_q))

        reward = 0.0
        reward += 80.0 * max(progress_gain, 0.0)
        reward += 5.0 * self.virtual_door_q
        reward -= 0.5 * handle_dist
        reward -= 0.5 * base_goal_err
        reward -= 0.2 * arm_goal_err
        reward -= 0.01 * float(np.dot(action, action))
        reward -= 0.01

        success = bool(self.virtual_door_q >= self.open_threshold)
        if success:
            reward += 200.0

        terminated = success
        truncated = bool(self.step_count >= self.max_steps)

        if self.render_mode == "human" and self.viewer is not None:
            self.viewer.sync()

        info = {
            "handle_dist": float(handle_dist),
            "door_q": float(self.virtual_door_q),
            "progress": float(progress),
            "best_progress": float(self.best_progress),
            "base_progress": float(base_prog),
            "arm_progress": float(arm_prog),
            "success": success,
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        self.viewer = None