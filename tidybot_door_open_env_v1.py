import os
import math
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class TidybotDoorOpenEnvV1(gym.Env):
    """
    Stage 2:
    - base frozen
    - arm-only control
    - starts near pre-handle-ready configuration
    - objective: approach handle and slide door open
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

        # sites / joints
        self.ee_site = self._require_site("pinch_site")
        self.handle_site = self._require_site("cell_door_site")
        self.door_joint = self._require_joint("cell_door_joint")

        # base
        self.base_joint_names = ["joint_x", "joint_y", "joint_th"]
        self.base_joint_ids = [self._require_joint(n) for n in self.base_joint_names]
        self.base_qadr = [int(self.model.jnt_qposadr[j]) for j in self.base_joint_ids]
        self.base_dadr = [int(self.model.jnt_dofadr[j]) for j in self.base_joint_ids]

        self.base_actuator_names = ["joint_x", "joint_y", "joint_th"]
        self.base_actuator_ids = [self._require_actuator(n) for n in self.base_actuator_names]

        # arm
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

        print("[stage2 arm joints]")
        for jname, jid, aid in zip(self.arm_joint_names, self.arm_joint_ids, self.arm_actuator_ids):
            aname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
            print(f"  joint={jname} jid={jid} -> actuator={aid} name={aname}")

        self.door_qadr = int(self.model.jnt_qposadr[self.door_joint])
        self.door_dadr = int(self.model.jnt_dofadr[self.door_joint])
        self.door_range = self.model.jnt_range[self.door_joint].copy()

        # timing / control
        self.physics_dt = 0.001
        self.dt = 0.05
        self.n_substeps = max(1, int(round(self.dt / self.physics_dt)))
        self.model.opt.timestep = float(self.physics_dt)

        self.max_steps = 1000
        self.step_count = 0

        self._base_target = np.zeros(3, dtype=np.float64)
        self._arm_target = np.zeros(7, dtype=np.float64)
        self.max_arm_delta_per_step = 0.02

        # reset setup
        self.safe_offset_y = 0.75
        self.safe_yaw = math.pi / 2.0
        self.prehandle_offset = np.array([0.0, -0.18, 0.0], dtype=np.float64)

        self.jitter_x = 0.05
        self.jitter_y = 0.05
        self.jitter_yaw = 0.08
        self.max_reset_tries = 40

        # reward bookkeeping
        self.prev_handle_dist = None
        self.prev_door_q = None
        self.best_handle_dist = float("inf")
        self.best_door_q = None

        # opening direction
        self.open_target_q = float(self.door_range[1]) * 0.80
        self.contact_radius = 0.10
        self.success_handle_radius = 0.08

        # obs:
        # ee->handle xyz(3), dist(1), arm q(7), arm qd(7), door q(1), door qd(1), contact_flag(1)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

    # ---------- id helpers

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

    # ---------- geometry helpers

    def _get_handle_world(self):
        return self.data.site_xpos[self.handle_site].copy()

    def _get_prehandle_world(self):
        return self._get_handle_world() + self.prehandle_offset

    def _get_safe_prepose_world(self):
        handle = self._get_handle_world()
        return (
            float(handle[0]),
            float(handle[1] - self.safe_offset_y),
            float(self.safe_yaw),
        )

    def _get_door_q(self) -> float:
        return float(self.data.qpos[self.door_qadr])

    def _get_door_qd(self) -> float:
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

    def _set_arm_to_current(self):
        self._arm_target[:] = np.array([self.data.qpos[i] for i in self.arm_qadr], dtype=np.float64)

    def _current_handle_dist(self) -> float:
        ee = self.data.site_xpos[self.ee_site].copy()
        handle = self._get_handle_world()
        return float(np.linalg.norm(handle - ee))

    def _contact_flag(self) -> float:
        return 1.0 if self._current_handle_dist() < self.contact_radius else 0.0

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
            return

        self._set_base_pose(x0, y0, yaw0)
        self._base_target[:] = (x0, y0, yaw0)

    # ---------- obs/reward

    def _get_obs(self) -> np.ndarray:
        ee = self.data.site_xpos[self.ee_site].copy()
        handle = self._get_handle_world()

        d = handle - ee
        dist = float(np.linalg.norm(d))

        q = np.array([self.data.qpos[i] for i in self.arm_qadr], dtype=np.float32)
        qd = np.array([self.data.qvel[i] for i in self.arm_dadr], dtype=np.float32)

        door_q = np.array([self._get_door_q()], dtype=np.float32)
        door_qd = np.array([self._get_door_qd()], dtype=np.float32)
        contact = np.array([self._contact_flag()], dtype=np.float32)

        return np.concatenate(
            [
                d.astype(np.float32),
                np.array([dist], dtype=np.float32),
                q,
                qd,
                door_q,
                door_qd,
                contact,
            ],
            axis=0,
        ).astype(np.float32)

    # ---------- API

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self._randomize_base_near_safe_pose()
        self._set_arm_to_current()

        self.step_count = 0
        self.prev_handle_dist = None
        self.prev_door_q = self._get_door_q()
        self.best_handle_dist = float("inf")
        self.best_door_q = self._get_door_q()

        if self.render_mode == "human" and self.viewer is None:
            from mujoco import viewer
            self.viewer = viewer.launch_passive(self.model, self.data)

        obs = self._get_obs()
        info = {
            "handle_dist": float(obs[3]),
            "door_q": self._get_door_q(),
        }
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        handle_dist_before = float(self._get_obs()[3])
        door_q_before = self._get_door_q()

        self._hold_base()

        self._arm_target += self.max_arm_delta_per_step * action.astype(np.float64)

        for i, jid in enumerate(self.arm_joint_ids):
            rng = self.model.jnt_range[jid]
            lo, hi = float(rng[0]), float(rng[1])
            if hi > lo:
                self._arm_target[i] = np.clip(self._arm_target[i], lo, hi)

        for _ in range(self.n_substeps):
            self._hold_base()
            self._apply_arm_targets()
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        if self.render_mode == "human" and self.viewer is not None:
            self.viewer.sync()

        obs = self._get_obs()
        handle_dist = float(obs[3])
        door_q = self._get_door_q()

        self.best_handle_dist = min(self.best_handle_dist, handle_dist)
        self.best_door_q = max(self.best_door_q, door_q)

        handle_prog = 0.0 if self.prev_handle_dist is None else float(self.prev_handle_dist - handle_dist)
        self.prev_handle_dist = handle_dist

        door_prog = float(door_q - door_q_before)
        self.prev_door_q = door_q

        contact = float(obs[-1])

        reward = 0.0
        reward += 8.0 * handle_prog
        reward -= 1.0 * handle_dist
        reward += 25.0 * max(door_prog, 0.0)
        reward += 1.0 * contact
        reward -= 0.01
        reward -= 0.01 * float(np.dot(action, action))

        handle_success = bool(handle_dist < self.success_handle_radius)
        door_success = bool(door_q >= self.open_target_q)
        success = door_success

        if handle_success:
            reward += 2.0
        if door_success:
            reward += 200.0

        terminated = success
        truncated = bool(self.step_count >= self.max_steps)

        info = {
            "handle_dist": handle_dist,
            "best_handle_dist": float(self.best_handle_dist),
            "door_q": float(door_q),
            "best_door_q": float(self.best_door_q),
            "door_progress": float(self.best_door_q - float(self.door_range[0])),
            "contact": int(contact > 0.5),
            "handle_success": handle_success,
            "success": success,
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        self.viewer = None