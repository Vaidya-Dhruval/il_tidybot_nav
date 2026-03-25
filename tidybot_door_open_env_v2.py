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
    Full sequential demo task:

    Phase 0:
        Random base start -> safe parking pose near door

    Phase 1:
        Arm reaches handle region while base stays parked

    Phase 2:
        Coordinated right-to-left sweep:
        base small leftward support + arm larger leftward sweep

    Important:
    - no real door mechanics
    - no dependency on moving handle after reset
    - handle site is only used as a fixed anchor for task setup
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

        self.ee_site = self._require_site("pinch_site")
        self.handle_site = self._require_site("cell_door_site")

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

        # timing
        self.physics_dt = 0.001
        self.dt = 0.05
        self.n_substeps = max(1, int(round(self.dt / self.physics_dt)))
        self.model.opt.timestep = float(self.physics_dt)

        # episode length
        self.max_steps = 500

        # action scaling
        self.max_vx = 0.12
        self.max_vy = 0.12
        self.max_wz = 0.40
        self.max_arm_delta_per_step = 0.012

        # phase hold logic
        self.phase = 0
        self.phase_hold_steps = 6
        self.phase_hold_count = 0

        self.phase0_done = False
        self.phase1_done = False
        self.phase0_complete_step = -1
        self.phase1_complete_step = -1

        # base/arm targets
        self._base_target = np.zeros(3, dtype=np.float64)
        self._arm_target = np.zeros(7, dtype=np.float64)

        # fixed handle anchor read at reset
        self.fixed_handle_world = np.zeros(3, dtype=np.float64)

        # parking pose settings
        self.safe_yaw = math.pi / 2.0
        self.park_offset = np.array([0.0, -0.60, 0.0], dtype=np.float64)

        # random spawn bounds relative to handle
        self.spawn_x_range = (-1.20, -0.35)
        self.spawn_y_range = (-1.20, -0.45)
        self.spawn_yaw_range = (0.80, 2.20)

        # handle target
        self.handle_target_offset = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # coordinated sweep settings
        self.base_support_step_x = 0.0030
        self.max_support_left = 0.60
        self.support_goal_delta = np.array([-0.42, 0.00, 0.00], dtype=np.float64)
        self.ee_sweep_delta = np.array([-0.65, 0.00, +0.01], dtype=np.float64)

        self.stage2_progress = 0.0
        self.stage2_progress_step = 0.012
        self.stage2_success_progress = 0.90

        self.stage2_track_tol = 0.10
        self.stage2_base_tol = 0.04

        self.park_pose = np.zeros(3, dtype=np.float64)
        self.handle_target_world = np.zeros(3, dtype=np.float64)

        self.stage2_base_start_pose = np.zeros(3, dtype=np.float64)
        self.stage2_base_goal_pose = np.zeros(3, dtype=np.float64)
        self.ee_start_world = np.zeros(3, dtype=np.float64)
        self.ee_goal_world = np.zeros(3, dtype=np.float64)

        self.ee_target_filtered = np.zeros(3, dtype=np.float64)
        self.ee_target_alpha = 0.12

        self.prev_d_park = 0.0
        self.prev_d_handle = 0.0
        self.prev_stage2_target_dist = 0.0

        self.step_count = 0

        # obs:
        # [park(5), handle(4), sweep(4), q(7), qd(7), phase_norm(1), stage2_progress(1), base_support_err(1)]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(30,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)

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

    def _get_ee_world(self):
        return self.data.site_xpos[self.ee_site].copy()

    def _get_handle_world(self):
        return self.data.site_xpos[self.handle_site].copy()

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

    def _clip_arm_target_to_limits(self):
        for i, jid in enumerate(self.arm_joint_ids):
            lo, hi = [float(x) for x in self.model.jnt_range[jid]]
            if hi > lo:
                self._arm_target[i] = np.clip(self._arm_target[i], lo, hi)

    def _parking_pose_world(self):
        return np.array([
            float(self.fixed_handle_world[0] + self.park_offset[0]),
            float(self.fixed_handle_world[1] + self.park_offset[1]),
            float(self.safe_yaw),
        ], dtype=np.float64)

    def _sample_random_start_pose(self):
        return np.array([
            float(self.fixed_handle_world[0] + self.np_random.uniform(*self.spawn_x_range)),
            float(self.fixed_handle_world[1] + self.np_random.uniform(*self.spawn_y_range)),
            float(self.np_random.uniform(*self.spawn_yaw_range)),
        ], dtype=np.float64)

    def _support_target_pose(self):
        return lerp(self.stage2_base_start_pose, self.stage2_base_goal_pose, self.stage2_progress)

    def _moving_ee_target(self):
        target_nominal = lerp(self.ee_start_world, self.ee_goal_world, self.stage2_progress)
        self.ee_target_filtered = (
            (1.0 - self.ee_target_alpha) * self.ee_target_filtered
            + self.ee_target_alpha * target_nominal
        )
        return self.ee_target_filtered.copy()

    def _start_stage2(self):
        self.phase = 2
        self.phase_hold_count = 0
        self.stage2_progress = 0.0

        self.stage2_base_start_pose = self._get_base_xyth().copy()
        self.stage2_base_start_pose[1] = self.park_pose[1]
        self.stage2_base_start_pose[2] = self.park_pose[2]

        self.stage2_base_goal_pose = self.stage2_base_start_pose + self.support_goal_delta
        self._base_target[:] = self.stage2_base_start_pose

        self.ee_start_world = self._get_ee_world().copy()
        self.ee_goal_world = self.ee_start_world + self.ee_sweep_delta
        self.ee_target_filtered = self.ee_start_world.copy()

        self.prev_stage2_target_dist = float(np.linalg.norm(self.ee_goal_world - self.ee_start_world))

    def _ik_action_to_world_target(self, target_world, damping=0.16, max_cart_vel=0.06, max_joint_delta=0.012):
        ee = self._get_ee_world()
        err = target_world - ee

        v_des = 4.0 * err
        v_norm = float(np.linalg.norm(v_des))
        if v_norm > max_cart_vel:
            v_des *= max_cart_vel / max(v_norm, 1e-8)

        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site)

        J = jacp[:, self.arm_dadr]
        A = J @ J.T + (damping ** 2) * np.eye(3, dtype=np.float64)
        J_pinv = J.T @ np.linalg.solve(A, np.eye(3, dtype=np.float64))

        dq = J_pinv @ v_des
        dq = np.clip(dq, -max_joint_delta, max_joint_delta)
        return dq / max(self.max_arm_delta_per_step, 1e-8)

    def _get_obs(self):
        base_now = self._get_base_xyth()
        ee_now = self._get_ee_world()

        dx_park = float(self.park_pose[0] - base_now[0])
        dy_park = float(self.park_pose[1] - base_now[1])
        yaw_err_park = wrap_to_pi(float(self.park_pose[2] - base_now[2]))
        d_park = float(np.linalg.norm([dx_park, dy_park]))

        handle_delta = self.handle_target_world - ee_now
        d_handle = float(np.linalg.norm(handle_delta))

        if self.phase >= 2:
            ee_target = self._moving_ee_target()
            sweep_delta = ee_target - ee_now
            base_support_err = float(abs(self._support_target_pose()[0] - base_now[0]))
        else:
            sweep_delta = np.zeros(3, dtype=np.float64)
            base_support_err = 0.0
        d_sweep = float(np.linalg.norm(sweep_delta))

        q = np.array([self.data.qpos[i] for i in self.arm_qadr], dtype=np.float32)
        qd = np.array([self.data.qvel[i] for i in self.arm_dadr], dtype=np.float32)

        phase_norm = float(self.phase / 2.0)

        obs = np.concatenate(
            [
                np.array([dx_park, dy_park, math.sin(yaw_err_park), math.cos(yaw_err_park), d_park], dtype=np.float32),
                np.concatenate([handle_delta.astype(np.float32), np.array([d_handle], dtype=np.float32)], axis=0),
                np.concatenate([sweep_delta.astype(np.float32), np.array([d_sweep], dtype=np.float32)], axis=0),
                q,
                qd,
                np.array([phase_norm], dtype=np.float32),
                np.array([self.stage2_progress], dtype=np.float32),
                np.array([base_support_err], dtype=np.float32),
            ],
            axis=0,
        ).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.phase = 0
        self.phase_hold_count = 0
        self.stage2_progress = 0.0

        self.phase0_done = False
        self.phase1_done = False
        self.phase0_complete_step = -1
        self.phase1_complete_step = -1

        self.fixed_handle_world = self._get_handle_world().copy()
        self.park_pose = self._parking_pose_world()
        self.handle_target_world = self.fixed_handle_world + self.handle_target_offset

        start_pose = self._sample_random_start_pose()
        self._set_base_pose(*start_pose.tolist())
        self._base_target[:] = start_pose

        self._arm_target[:] = np.array([self.data.qpos[i] for i in self.arm_qadr], dtype=np.float64)

        base_now = self._get_base_xyth()
        ee_now = self._get_ee_world()
        self.prev_d_park = float(np.linalg.norm(self.park_pose[:2] - base_now[:2]))
        self.prev_d_handle = float(np.linalg.norm(self.handle_target_world - ee_now))
        self.prev_stage2_target_dist = 0.0

        self.ee_target_filtered[:] = 0.0
        self.stage2_base_start_pose[:] = 0.0
        self.stage2_base_goal_pose[:] = 0.0
        self.ee_start_world[:] = 0.0
        self.ee_goal_world[:] = 0.0

        if self.render_mode == "human" and self.viewer is None:
            from mujoco import viewer
            self.viewer = viewer.launch_passive(self.model, self.data)

        obs = self._get_obs()
        info = {
            "phase": int(self.phase),
            "success": False,
            "d_park": float(self.prev_d_park),
            "d_handle": float(self.prev_d_handle),
            "stage2_progress": float(self.stage2_progress),
            "base_support_err": 0.0,
            "phase0_done": bool(self.phase0_done),
            "phase1_done": bool(self.phase1_done),
            "phase0_complete_step": int(self.phase0_complete_step),
            "phase1_complete_step": int(self.phase1_complete_step),
        }
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.clip(action, -1.0, 1.0)

        a_base = action[:3]
        a_arm = action[3:]

        if self.phase == 0:
            dx = self.max_vx * self.dt * float(a_base[0])
            dy = self.max_vy * self.dt * float(a_base[1])
            dyaw = self.max_wz * self.dt * float(a_base[2])

            self._base_target[0] += dx
            self._base_target[1] += dy
            self._base_target[2] = wrap_to_pi(self._base_target[2] + dyaw)

        elif self.phase == 1:
            self._base_target[:] = self.park_pose
            self._arm_target += self.max_arm_delta_per_step * a_arm.astype(np.float64)
            self._clip_arm_target_to_limits()

        else:
            if float(a_base[0]) < 0.0:
                desired_left = self.base_support_step_x * float(a_base[0])
                self._base_target[0] += 0.65 * desired_left

            min_x = self.stage2_base_start_pose[0] - self.max_support_left
            self._base_target[0] = np.clip(self._base_target[0], min_x, self.stage2_base_start_pose[0])
            self._base_target[1] = self.stage2_base_start_pose[1]
            self._base_target[2] = self.stage2_base_start_pose[2]

            support_now = self._support_target_pose()
            self._base_target[0] = 0.85 * self._base_target[0] + 0.15 * support_now[0]

            self._arm_target += self.max_arm_delta_per_step * a_arm.astype(np.float64)
            self._clip_arm_target_to_limits()

        for _ in range(self.n_substeps):
            self._hold_base()
            self._apply_arm_targets()
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        base_now = self._get_base_xyth()
        ee_now = self._get_ee_world()

        d_park = float(np.linalg.norm(self.park_pose[:2] - base_now[:2]))
        yaw_err_park = abs(wrap_to_pi(float(self.park_pose[2] - base_now[2])))
        d_handle = float(np.linalg.norm(self.handle_target_world - ee_now))

        if self.phase >= 2:
            ee_target = self._moving_ee_target()
            target_dist = float(np.linalg.norm(ee_target - ee_now))
            base_support_err = float(abs(self._support_target_pose()[0] - base_now[0]))
        else:
            target_dist = 0.0
            base_support_err = 0.0

        reward = 0.0
        success = False

        if self.phase == 0:
            park_progress = self.prev_d_park - d_park
            reward += 8.0 * park_progress
            reward -= 1.5 * d_park
            reward -= 0.5 * yaw_err_park
            reward -= 0.01 * float(np.dot(action, action))
            reward -= 0.01

            if (d_park < 0.08) and (yaw_err_park < 0.12):
                self.phase_hold_count += 1
            else:
                self.phase_hold_count = 0

            if self.phase_hold_count >= self.phase_hold_steps:
                self.phase0_done = True
                self.phase0_complete_step = self.step_count
                self.phase = 1
                self.phase_hold_count = 0
                self._base_target[:] = self.park_pose
                reward += 20.0

        elif self.phase == 1:
            handle_progress = self.prev_d_handle - d_handle
            base_drift = float(np.linalg.norm(base_now[:2] - self.park_pose[:2]))

            reward += 10.0 * handle_progress
            reward -= 2.0 * d_handle
            reward -= 0.75 * base_drift
            reward -= 0.01 * float(np.dot(action, action))
            reward -= 0.01

            if d_handle < 0.05:
                self.phase_hold_count += 1
            else:
                self.phase_hold_count = 0

            if self.phase_hold_count >= self.phase_hold_steps:
                self.phase1_done = True
                self.phase1_complete_step = self.step_count
                reward += 25.0
                self._start_stage2()

        else:
            good_track = (target_dist < self.stage2_track_tol) and (base_support_err < self.stage2_base_tol)

            if good_track:
                self.stage2_progress = min(1.0, self.stage2_progress + self.stage2_progress_step)

            reward += 20.0 * self.stage2_progress_step if good_track else 0.0
            reward += 3.0 * self.stage2_progress
            reward -= 2.0 * target_dist
            reward -= 1.0 * base_support_err
            reward -= 0.01 * float(np.dot(action, action))
            reward -= 0.01

            if self.stage2_progress >= self.stage2_success_progress:
                reward += 200.0
                success = True

        self.prev_d_park = d_park
        self.prev_d_handle = d_handle
        if self.phase >= 2:
            self.prev_stage2_target_dist = target_dist

        terminated = bool(success)
        truncated = bool(self.step_count >= self.max_steps)

        if self.render_mode == "human" and self.viewer is not None:
            self.viewer.sync()

        obs = self._get_obs()

        info = {
            "phase": int(self.phase),
            "success": bool(success),
            "d_park": float(d_park),
            "yaw_err_park": float(yaw_err_park),
            "d_handle": float(d_handle),
            "stage2_progress": float(self.stage2_progress),
            "target_dist": float(target_dist),
            "base_support_err": float(base_support_err),
            "phase0_done": bool(self.phase0_done),
            "phase1_done": bool(self.phase1_done),
            "phase0_complete_step": int(self.phase0_complete_step),
            "phase1_complete_step": int(self.phase1_complete_step),
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        self.viewer = None