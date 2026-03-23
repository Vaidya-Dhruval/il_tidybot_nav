import os
import math
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class ArmReachEnv(gym.Env):
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

        # ---- sites
        self.ee_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "pinch_site")
        self.target_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "cell_door_site")

        if self.ee_site < 0:
            raise RuntimeError("Site 'pinch_site' not found")
        if self.target_site < 0:
            raise RuntimeError("Site 'cell_door_site' not found")

        # ---- base joints/actuators
        self.base_joint_names = ["joint_x", "joint_y", "joint_th"]
        self.base_joint_ids = []
        for name in self.base_joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"Base joint '{name}' not found")
            self.base_joint_ids.append(jid)

        self.base_qadr = [int(self.model.jnt_qposadr[j]) for j in self.base_joint_ids]
        self.base_dadr = [int(self.model.jnt_dofadr[j]) for j in self.base_joint_ids]

        self.base_actuator_names = ["joint_x", "joint_y", "joint_th"]
        self.base_actuator_ids = []
        for name in self.base_actuator_names:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                raise RuntimeError(f"Base actuator '{name}' not found")
            self.base_actuator_ids.append(aid)

        # ---- arm joints
        self.joint_names = [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "joint_7",
        ]

        self.arm_joints = []
        for name in self.joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"Joint '{name}' not found in model")
            self.arm_joints.append(jid)

        self.qadr = [int(self.model.jnt_qposadr[j]) for j in self.arm_joints]
        self.dadr = [int(self.model.jnt_dofadr[j]) for j in self.arm_joints]

        # ---- arm actuators mapped by driven joint
        self.arm_actuators = []
        for jid, jname in zip(self.arm_joints, self.joint_names):
            found = None
            for aid in range(self.model.nu):
                trn_jid = int(self.model.actuator_trnid[aid, 0])
                if trn_jid == jid:
                    found = aid
                    break
            if found is None:
                raise RuntimeError(f"No actuator found that drives joint '{jname}'")
            self.arm_actuators.append(found)

        print("[arm joints]")
        for jname, jid, aid in zip(self.joint_names, self.arm_joints, self.arm_actuators):
            aname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
            print(f"  joint={jname} jid={jid} -> actuator={aid} name={aname}")

        # ---- Stage-1-specific base placement
        # closer than Stage 0 so the arm can actually reach
        self.reach_offset_y = 0.45
        self.reach_yaw = math.pi / 2.0

        self.max_steps = 500
        self.step_count = 0
        self.prev_dist = None

        self._base_target = np.zeros(3, dtype=np.float64)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

        # obs = ee_to_target(3) + dist(1) + q(7) + qd(7) = 18
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)

    def _get_reach_base_pose_world(self):
        door = self.data.site_xpos[self.target_site].copy()
        x_t = float(door[0])
        y_t = float(door[1] - self.reach_offset_y)
        yaw_t = float(self.reach_yaw)
        return x_t, y_t, yaw_t

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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        # Place base closer for arm reach
        x_t, y_t, yaw_t = self._get_reach_base_pose_world()
        self._set_base_pose(x_t, y_t, yaw_t)
        self._base_target[:] = (x_t, y_t, yaw_t)
        self._hold_base()

        for _ in range(30):
            self._hold_base()
            mujoco.mj_step(self.model, self.data)

        self.step_count = 0
        self.prev_dist = None

        if self.render_mode == "human" and self.viewer is None:
            from mujoco import viewer
            self.viewer = viewer.launch_passive(self.model, self.data)

        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        self._hold_base()

        for i, aid in enumerate(self.arm_actuators):
            self.data.ctrl[aid] = float(action[i] * 1.0)

        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        if self.render_mode == "human" and self.viewer is not None:
            self.viewer.sync()

        obs = self._get_obs()
        dist = float(obs[3])

        prog = 0.0 if self.prev_dist is None else float(self.prev_dist - dist)
        self.prev_dist = dist

        reward = 0.0
        reward += 10.0 * prog
        reward -= 1.0 * dist
        reward -= 0.01
        reward -= 0.01 * float(np.sum(action ** 2))

        success = dist < 0.08
        if success:
            reward += 100.0

        terminated = bool(success)
        truncated = bool(self.step_count >= self.max_steps)

        info = {
            "distance": dist,
            "success": bool(success),
        }

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        ee = self.data.site_xpos[self.ee_site].copy()
        target = self.data.site_xpos[self.target_site].copy()

        d = target - ee
        dist = float(np.linalg.norm(d))

        q = np.array([self.data.qpos[i] for i in self.qadr], dtype=np.float32)
        qd = np.array([self.data.qvel[i] for i in self.dadr], dtype=np.float32)

        return np.concatenate(
            [d.astype(np.float32), np.array([dist], dtype=np.float32), q, qd],
            axis=0
        )

    def close(self):
        # avoid GLX cleanup crashes during debug
        self.viewer = None