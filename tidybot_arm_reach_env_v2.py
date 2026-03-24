import os
import math
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class ArmManipulationEnv(gym.Env):
    """
    Unified arm manipulation env.

    Modes
    -----
    task_mode="prehandle"
        Stage 1:
        - base frozen
        - base randomized near safe prepose
        - arm-only control
        - reach pre-handle target

    task_mode="door_open"
        Stage 2:
        - base frozen
        - arm starts near pre-handle pose
        - arm-only control
        - reach door interaction site and open the door

    Current patches
    ---------------
    - handle site is explicitly set to "cell_door_site"
    - Stage 1 prints successful arm joint pose as STAGE1_SUCCESS_Q
    - Stage 2 far_fail is disabled temporarily for debugging
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, model_path: str, render_mode=None, task_mode: str = "prehandle"):
        super().__init__()

        assert task_mode in ("prehandle", "door_open")
        self.task_mode = task_mode
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

        # -------------------------
        # Sites
        # -------------------------
        self.ee_site = self._require_site("pinch_site")
        self.target_site = self._require_site("cell_door_site")

        # Explicitly use the only known door interaction site for now
        self.handle_site_name = "cell_door_site"
        self.handle_site = self._require_site(self.handle_site_name)

        # -------------------------
        # Door joint
        # -------------------------
        self.door_joint_name = "cell_door_joint"
        self.door_joint = self._require_joint(self.door_joint_name)
        self.door_qadr = int(self.model.jnt_qposadr[self.door_joint])
        self.door_dadr = int(self.model.jnt_dofadr[self.door_joint])

        # -------------------------
        # Base joints / actuators
        # -------------------------
        self.base_joint_names = ["joint_x", "joint_y", "joint_th"]
        self.base_joint_ids = [self._require_joint(n) for n in self.base_joint_names]
        self.base_qadr = [int(self.model.jnt_qposadr[j]) for j in self.base_joint_ids]
        self.base_dadr = [int(self.model.jnt_dofadr[j]) for j in self.base_joint_ids]

        self.base_actuator_names = ["joint_x", "joint_y", "joint_th"]
        self.base_actuator_ids = [self._require_actuator(n) for n in self.base_actuator_names]

        # -------------------------
        # 7-DoF arm joints
        # -------------------------
        self.arm_joint_names = [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "joint_7",
        ]
        self.arm_joint_ids = [self._require_joint(n) for n in self.arm_joint_names]
        self.arm_qadr = [int(self.model.jnt_qposadr[j]) for j in self.arm_joint_ids]
        self.arm_dadr = [int(self.model.jnt_dofadr[j]) for j in self.arm_joint_ids]

        # map actuators by driven joint
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

        print("[arm joints]")
        for jname, jid, aid in zip(self.arm_joint_names, self.arm_joint_ids, self.arm_actuator_ids):
            aname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
            print(f"  joint={jname} jid={jid} -> actuator={aid} name={aname}")

        # -------------------------
        # Safe stand-off base pose
        # -------------------------
        self.safe_offset_y = 0.75
        self.safe_yaw = math.pi / 2.0

        # reset randomization for Stage 1
        self.jitter_x = 0.15
        self.jitter_y = 0.15
        self.jitter_yaw = 0.20

        self.min_init_dist = 0.75
        self.max_init_dist = 1.05
        self.max_reset_tries = 60

        # -------------------------
        # Stage 1 pre-handle target
        # -------------------------
        self.pre_handle_offset = np.array([0.0, -0.18, 0.0], dtype=np.float64)

        # -------------------------
        # Stage 2 start pose
        # REPLACE this with printed STAGE1_SUCCESS_Q later
        # -------------------------
        self.stage2_start_q = np.array(
            [0.0, -1.2, 1.8, -1.5, -1.57, 1.2, 0.0],
            dtype=np.float64,
        )
        self.stage2_q_jitter = 0.03

        # -------------------------
        # Control config
        # -------------------------
        self.max_steps = 1500 if self.task_mode == "prehandle" else 300
        self.step_count = 0

        self._base_target = np.zeros(3, dtype=np.float64)
        self._arm_target = np.zeros(7, dtype=np.float64)

        # Keep consistent across record/train/eval
        self.max_arm_delta_per_step = 0.02

        # -------------------------
        # Runtime trackers
        # -------------------------
        self.prev_dist = None
        self.best_dist = float("inf")

        self.prev_handle_dist = None
        self.best_handle_dist = float("inf")
        self.prev_door_qpos = None
        self.contact_hold_steps = 0

        # -------------------------
        # Thresholds
        # -------------------------
        self.prehandle_success_thresh = 0.08

        self.handle_align_radius = 0.08
        self.handle_contact_radius = 0.05
        self.door_success_open = 0.75
        self.hold_contact_steps_for_bonus = 5

        # Observation:
        # ee->prehandle (3)
        # ee->handle (3)
        # dist_prehandle (1)
        # dist_handle (1)
        # arm q (7)
        # arm qd (7)
        # base residual in body frame (2)
        # yaw err sin/cos (2)
        # door qpos (1)
        # door qvel (1)
        # contact_like (1)
        # total = 29
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(29,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    # name helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # geometry helpers
    # ------------------------------------------------------------------

    def _get_base_xyth(self):
        return (
            float(self.data.qpos[self.base_qadr[0]]),
            float(self.data.qpos[self.base_qadr[1]]),
            float(self.data.qpos[self.base_qadr[2]]),
        )

    def _get_safe_prepose_world(self):
        handle = self.data.site_xpos[self.target_site].copy()
        return (
            float(handle[0]),
            float(handle[1] - self.safe_offset_y),
            float(self.safe_yaw),
        )

    def _get_prehandle_target_world(self):
        handle = self.data.site_xpos[self.target_site].copy()
        return handle + self.pre_handle_offset

    def _get_handle_target_world(self):
        return self.data.site_xpos[self.handle_site].copy()

    def _get_ee_world(self):
        return self.data.site_xpos[self.ee_site].copy()

    def _get_door_qpos(self) -> float:
        return float(self.data.qpos[self.door_qadr])

    def _get_door_qvel(self) -> float:
        return float(self.data.qvel[self.door_dadr])

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

    # ------------------------------------------------------------------
    # reset helpers
    # ------------------------------------------------------------------

    def _current_prehandle_distance(self) -> float:
        ee = self._get_ee_world()
        target = self._get_prehandle_target_world()
        return float(np.linalg.norm(target - ee))

    def _current_handle_distance(self) -> float:
        ee = self._get_ee_world()
        handle = self._get_handle_target_world()
        return float(np.linalg.norm(handle - ee))

    def _randomize_base_near_safe_pose(self):
        x0, y0, yaw0 = self._get_safe_prepose_world()

        for _ in range(self.max_reset_tries):
            x = x0 + float(self.np_random.uniform(-self.jitter_x, self.jitter_x))
            y = y0 + float(self.np_random.uniform(-self.jitter_y, self.jitter_y))
            yaw = wrap_to_pi(yaw0 + float(self.np_random.uniform(-self.jitter_yaw, self.jitter_yaw)))

            self._set_base_pose(x, y, yaw)
            self._base_target[:] = (x, y, yaw)

            for _ in range(10):
                self._hold_base()
                mujoco.mj_step(self.model, self.data)

            d = self._current_prehandle_distance()
            if self.min_init_dist <= d <= self.max_init_dist:
                return

        self._set_base_pose(x0, y0, yaw0)
        self._base_target[:] = (x0, y0, yaw0)

    def _reset_stage2_arm_near_prehandle(self):
        if len(self.stage2_start_q) != len(self.arm_qadr):
            raise RuntimeError("stage2_start_q length does not match arm joint count")

        q = self.stage2_start_q.copy()
        q += self.np_random.uniform(-self.stage2_q_jitter, self.stage2_q_jitter, size=q.shape)

        for i, jid in enumerate(self.arm_joint_ids):
            rng = self.model.jnt_range[jid]
            lo, hi = float(rng[0]), float(rng[1])
            if hi > lo:
                q[i] = np.clip(q[i], lo, hi)

        for i, qadr in enumerate(self.arm_qadr):
            self.data.qpos[qadr] = float(q[i])
        for dadr in self.arm_dadr:
            self.data.qvel[dadr] = 0.0

        self._arm_target[:] = q
        mujoco.mj_forward(self.model, self.data)

    def _reset_door(self):
        self.data.qpos[self.door_qadr] = 0.0
        self.data.qvel[self.door_dadr] = 0.0
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------
    # observations
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        ee = self._get_ee_world()
        prehandle = self._get_prehandle_target_world()
        handle = self._get_handle_target_world()

        d_pre = prehandle - ee
        d_handle = handle - ee
        dist_pre = float(np.linalg.norm(d_pre))
        dist_handle = float(np.linalg.norm(d_handle))

        q = np.array([self.data.qpos[i] for i in self.arm_qadr], dtype=np.float32)
        qd = np.array([self.data.qvel[i] for i in self.arm_dadr], dtype=np.float32)

        bx, by, bth = self._get_base_xyth()
        x_t, y_t, yaw_t = self._get_safe_prepose_world()
        dx_w = x_t - bx
        dy_w = y_t - by
        c, s = math.cos(bth), math.sin(bth)
        dx_b = c * dx_w + s * dy_w
        dy_b = -s * dx_w + c * dy_w
        yaw_err = wrap_to_pi(yaw_t - bth)

        door_qpos = self._get_door_qpos()
        door_qvel = self._get_door_qvel()
        contact_like = 1.0 if dist_handle < self.handle_contact_radius else 0.0

        obs = np.concatenate(
            [
                d_pre.astype(np.float32),
                d_handle.astype(np.float32),
                np.array([dist_pre], dtype=np.float32),
                np.array([dist_handle], dtype=np.float32),
                q,
                qd,
                np.array([dx_b, dy_b], dtype=np.float32),
                np.array([math.sin(yaw_err), math.cos(yaw_err)], dtype=np.float32),
                np.array([door_qpos], dtype=np.float32),
                np.array([door_qvel], dtype=np.float32),
                np.array([contact_like], dtype=np.float32),
            ],
            axis=0,
        )
        return obs.astype(np.float32)

    # ------------------------------------------------------------------
    # gym api
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self._reset_door()

        if self.task_mode == "prehandle":
            self._randomize_base_near_safe_pose()
            self._arm_target[:] = np.array([self.data.qpos[i] for i in self.arm_qadr], dtype=np.float64)

        elif self.task_mode == "door_open":
            x0, y0, yaw0 = self._get_safe_prepose_world()
            self._set_base_pose(x0, y0, yaw0)
            self._base_target[:] = (x0, y0, yaw0)
            self._reset_stage2_arm_near_prehandle()

            q_runtime = np.array([self.data.qpos[i] for i in self.arm_qadr], dtype=np.float64)
            print("stage2_start_q_runtime =", q_runtime.tolist())

            for _ in range(10):
                self._hold_base()
                self._apply_arm_targets()
                mujoco.mj_step(self.model, self.data)

        self.step_count = 0
        self.prev_dist = None
        self.best_dist = float("inf")

        self.prev_handle_dist = self._current_handle_distance()
        self.best_handle_dist = self.prev_handle_dist
        self.prev_door_qpos = self._get_door_qpos()
        self.contact_hold_steps = 0

        if self.render_mode == "human" and self.viewer is None:
            from mujoco import viewer
            self.viewer = viewer.launch_passive(self.model, self.data)

        obs = self._get_obs()

        print(
            f"[reset:{self.task_mode}] "
            f"pre={float(obs[6]):.4f} handle={float(obs[7]):.4f} door={self._get_door_qpos():.4f}"
        )

        return obs, {
            "prehandle_distance": float(obs[6]),
            "handle_distance": float(obs[7]),
            "door_qpos": self._get_door_qpos(),
        }

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        obs_before = self._get_obs()
        dist_pre_before = float(obs_before[6])
        dist_handle_before = float(obs_before[7])
        door_before = self._get_door_qpos()

        self._hold_base()

        self._arm_target += self.max_arm_delta_per_step * action.astype(np.float64)

        for i, jid in enumerate(self.arm_joint_ids):
            rng = self.model.jnt_range[jid]
            lo, hi = float(rng[0]), float(rng[1])
            if hi > lo:
                self._arm_target[i] = np.clip(self._arm_target[i], lo, hi)

        self._apply_arm_targets()
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        if self.render_mode == "human" and self.viewer is not None:
            self.viewer.sync()

        obs = self._get_obs()
        dist_pre = float(obs[6])
        dist_handle = float(obs[7])
        door_qpos = self._get_door_qpos()
        door_qvel = self._get_door_qvel()

        self.best_dist = min(self.best_dist, dist_pre)
        self.best_handle_dist = min(self.best_handle_dist, dist_handle)

        # ------------------------------------------------------------
        # Stage 1 reward
        # ------------------------------------------------------------
        if self.task_mode == "prehandle":
            prog = 0.0 if self.prev_dist is None else float(self.prev_dist - dist_pre)
            self.prev_dist = dist_pre

            reward = 0.0
            reward += 10.0 * prog
            reward -= 1.0 * dist_pre
            reward -= 0.01
            reward -= 0.01 * float(np.dot(action, action))

            success = bool(dist_pre < self.prehandle_success_thresh)
            if success:
                q = np.array([self.data.qpos[i] for i in self.arm_qadr], dtype=np.float64)
                print("STAGE1_SUCCESS_Q =", q.tolist())
                reward += 100.0

            terminated = success
            truncated = bool(self.step_count >= self.max_steps)

            info = {
                "distance": dist_pre,
                "best_distance": float(self.best_dist),
                "success": success,
                "progress": float(dist_pre_before - dist_pre),
                "mode": self.task_mode,
            }
            return obs, reward, terminated, truncated, info

        # ------------------------------------------------------------
        # Stage 2 reward
        # ------------------------------------------------------------
        elif self.task_mode == "door_open":
            handle_progress = float(dist_handle_before - dist_handle)
            door_progress = max(0.0, float(door_qpos - door_before))

            in_align_zone = float(dist_handle < self.handle_align_radius)
            in_contact_zone = float(dist_handle < self.handle_contact_radius)

            if in_contact_zone > 0.5:
                self.contact_hold_steps += 1
            else:
                self.contact_hold_steps = 0

            hold_bonus = 1.0 if self.contact_hold_steps >= self.hold_contact_steps_for_bonus else 0.0

            qd = np.array([self.data.qvel[i] for i in self.arm_dadr], dtype=np.float32)
            qvel_pen = float(np.dot(qd, qd))
            ctrl_pen = float(np.dot(action, action))

            reward = 0.0
            reward += 8.0 * handle_progress
            reward += 3.0 * in_align_zone
            reward += 0.75 * hold_bonus
            reward += 25.0 * door_progress
            reward += 2.0 * door_qpos
            reward -= 0.02 * ctrl_pen
            reward -= 0.002 * qvel_pen
            reward -= 0.01

            success = bool(door_qpos >= self.door_success_open)
            if success:
                reward += 200.0

            unstable = not np.isfinite(qvel_pen) or float(np.linalg.norm(qd)) > 8.0

            # TEMP DEBUG PATCH:
            # disable far_fail until stage2_start_q is fixed
            far_fail = False

            terminated = success or unstable
            truncated = bool(self.step_count >= self.max_steps)

            info = {
                "prehandle_distance": dist_pre,
                "handle_distance": dist_handle,
                "best_handle_distance": float(self.best_handle_dist),
                "handle_progress": handle_progress,
                "door_qpos": door_qpos,
                "door_qvel": door_qvel,
                "door_progress": door_progress,
                "contact_hold_steps": int(self.contact_hold_steps),
                "in_align_zone": bool(in_align_zone > 0.5),
                "in_contact_zone": bool(in_contact_zone > 0.5),
                "success": success,
                "unstable": unstable,
                "far_fail": far_fail,
                "mode": self.task_mode,
            }
            return obs, reward, terminated, truncated, info

    def close(self):
        self.viewer = None