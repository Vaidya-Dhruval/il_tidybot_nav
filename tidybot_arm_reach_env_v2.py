import os
import math
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class ArmReachEnvV2(gym.Env):
    """
    Stage 1 corrected:
    - base is FROZEN during episode
    - base is RANDOMIZED at reset around a safe prepose
    - action is ARM ONLY (7 joints)
    - arm actuators are treated as POSITION targets, so we keep persistent arm_target
    """

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

        # sites
        self.ee_site = self._require_site("pinch_site")
        self.target_site = self._require_site("cell_door_site")

        # base joints / actuators
        self.base_joint_names = ["joint_x", "joint_y", "joint_th"]
        self.base_joint_ids = [self._require_joint(n) for n in self.base_joint_names]
        self.base_qadr = [int(self.model.jnt_qposadr[j]) for j in self.base_joint_ids]
        self.base_dadr = [int(self.model.jnt_dofadr[j]) for j in self.base_joint_ids]

        self.base_actuator_names = ["joint_x", "joint_y", "joint_th"]
        self.base_actuator_ids = [self._require_actuator(n) for n in self.base_actuator_names]

        # 7-DoF arm
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

        # safe stand-off center
        self.safe_offset_y = 0.75
        self.safe_yaw = math.pi / 2.0

        # reset randomization bands around safe prepose
        self.jitter_x = 0.15
        self.jitter_y = 0.15
        self.jitter_yaw = 0.20

        # realistic Stage-1 starts
        self.min_init_dist = 0.75
        self.max_init_dist = 1.05
        self.max_reset_tries = 60

        # PRE-HANDLE target offset
        self.pre_handle_offset = np.array([0.0, -0.18, 0.0], dtype=np.float64)

        # control config
        self.max_steps = 1500
        self.step_count = 0
        self.prev_dist = None
        self.best_dist = float("inf")

        self._base_target = np.zeros(3, dtype=np.float64)
        self._arm_target = np.zeros(7, dtype=np.float64)

        # action is interpreted as JOINT DELTA COMMAND, not absolute target
        self.max_arm_delta_per_step = 0.02

        # obs = ee->prehandle (3) + dist (1) + arm q (7) + arm qd (7)
        #     + base residual to safe prepose in body frame (2) + yaw err sin/cos (2)
        # total = 22
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )

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

    def _current_distance(self) -> float:
        ee = self.data.site_xpos[self.ee_site].copy()
        target = self._get_prehandle_target_world()
        return float(np.linalg.norm(target - ee))

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

            d = self._current_distance()
            if self.min_init_dist <= d <= self.max_init_dist:
                return

        self._set_base_pose(x0, y0, yaw0)
        self._base_target[:] = (x0, y0, yaw0)

    def _get_obs(self) -> np.ndarray:
        ee = self.data.site_xpos[self.ee_site].copy()
        target = self._get_prehandle_target_world()

        d = target - ee
        dist = float(np.linalg.norm(d))

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

        obs = np.concatenate(
            [
                d.astype(np.float32),
                np.array([dist], dtype=np.float32),
                q,
                qd,
                np.array([dx_b, dy_b], dtype=np.float32),
                np.array([math.sin(yaw_err), math.cos(yaw_err)], dtype=np.float32),
            ],
            axis=0,
        )
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self._randomize_base_near_safe_pose()

        self._arm_target[:] = np.array([self.data.qpos[i] for i in self.arm_qadr], dtype=np.float64)

        self.step_count = 0
        self.prev_dist = None
        self.best_dist = float("inf")

        if self.render_mode == "human" and self.viewer is None:
            from mujoco import viewer
            self.viewer = viewer.launch_passive(self.model, self.data)

        obs = self._get_obs()
        print(f"[v2 reset] start_dist={float(obs[3]):.4f}")
        return obs, {"distance": float(obs[3])}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        dist_before = float(self._get_obs()[3])

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
        dist = float(obs[3])
        self.best_dist = min(self.best_dist, dist)

        prog = 0.0 if self.prev_dist is None else float(self.prev_dist - dist)
        self.prev_dist = dist

        reward = 0.0
        reward += 10.0 * prog
        reward -= 1.0 * dist
        reward -= 0.01
        reward -= 0.01 * float(np.dot(action, action))

        success = bool(dist < 0.08)
        if success:
            reward += 100.0

        terminated = success
        truncated = bool(self.step_count >= self.max_steps)

        info = {
            "distance": dist,
            "best_distance": float(self.best_dist),
            "success": success,
            "progress": float(dist_before - dist),
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        self.viewer = None