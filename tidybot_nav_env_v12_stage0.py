import os
os.environ.setdefault("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import mujoco
import mujoco.viewer


DEFAULT_EE_SITE_NAME = "pinch_site"
DEFAULT_TARGET_SITE_NAME = "cell_door_site"
DEFAULT_CAMERA_NAME = "wrist"
DEFAULT_ROBOT_ROOT_BODY = "base_link"


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class V12Stage0Params:
    xml_path: str = "tidybot_with_cell.xml"
    ee_site_name: str = DEFAULT_EE_SITE_NAME
    target_site_name: str = DEFAULT_TARGET_SITE_NAME
    camera_name: str = DEFAULT_CAMERA_NAME
    robot_root_body: str = DEFAULT_ROBOT_ROOT_BODY

    dt: float = 0.05
    physics_dt: float = 0.001
    max_steps: int = 700

    cam_w: int = 128
    cam_h: int = 128

    max_vx: float = 0.12
    max_vy: float = 0.08
    max_wz: float = 0.35

    spawn_xy_range: float = 0.5
    spawn_yaw_range: float = math.pi

    spawn_around_target: bool = False
    spawn_target_min_r: float = 0.0
    spawn_target_max_r: float = 0.25

    # virtual pre-manipulation pose derived from door site
    prepose_offset_y: float = 0.75
    prepose_yaw: float = math.pi / 2.0
    prepose_yaw_tolerance: float = 0.18

    base_success_radius: float = 0.25
    hold_steps: int = 8

    tau: float = 0.25

    w_prog_anchor: float = 10.0
    w_dist_anchor: float = 0.0
    w_time: float = 0.01
    w_ctrl_base: float = 0.01
    success_bonus: float = 300.0

    terminate_on_collision: bool = False
    collision_penalty: float = 2.0
    collision_grace_radius: float = 1.20

    lidar_max_range: float = 4.0
    spawn_reject_using_lidar: bool = True
    spawn_k_smallest: int = 6
    spawn_min_kmean_lidar: float = 0.18
    spawn_max_tries: int = 60
    settle_steps: int = 30

    base_joint_x: str = "joint_x"
    base_joint_y: str = "joint_y"
    base_joint_th: str = "joint_th"

    base_act_x: str = "joint_x"
    base_act_y: str = "joint_y"
    base_act_th: str = "joint_th"


class TidybotNavEnvV12Stage0(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(self, p: V12Stage0Params, render_mode: Optional[str] = None):
        super().__init__()
        self.p = p
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(self.p.xml_path)
        self.data = mujoco.MjData(self.model)

        self.model.opt.timestep = float(self.p.physics_dt)
        self.n_substeps = max(1, int(round(float(self.p.dt) / float(self.p.physics_dt))))

        self.debug = bool(os.environ.get("V12_DEBUG", "0") == "1")
        self.debug_every = int(os.environ.get("V12_DEBUG_EVERY", "25"))

        self.ee_site_id = self._require_site(self.p.ee_site_name)
        self.target_site_id = self._require_site(self.p.target_site_name)

        self.cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.p.camera_name)
        if self.cam_id < 0:
            raise RuntimeError(f"Camera '{self.p.camera_name}' not found in model.")

        self.jid_x = self._require_joint(self.p.base_joint_x)
        self.jid_y = self._require_joint(self.p.base_joint_y)
        self.jid_th = self._require_joint(self.p.base_joint_th)

        self.qadr_x = int(self.model.jnt_qposadr[self.jid_x])
        self.qadr_y = int(self.model.jnt_qposadr[self.jid_y])
        self.qadr_th = int(self.model.jnt_qposadr[self.jid_th])

        self.dadr_x = int(self.model.jnt_dofadr[self.jid_x])
        self.dadr_y = int(self.model.jnt_dofadr[self.jid_y])
        self.dadr_th = int(self.model.jnt_dofadr[self.jid_th])

        self.aid_x = self._require_actuator(self.p.base_act_x)
        self.aid_y = self._require_actuator(self.p.base_act_y)
        self.aid_th = self._require_actuator(self.p.base_act_th)

        self.step_count = 0
        self.hold_counter = 0
        self.prev_d_anchor: Optional[float] = None
        self._best_d_anchor = float("inf")

        self._a_smooth = np.zeros(3, dtype=np.float32)
        self._base_target = np.zeros(3, dtype=np.float64)

        self._viewer = None
        self.renderer = None

        self._is_robot_body = self._build_robot_body_mask(root_body_name=self.p.robot_root_body)

        self.rf_slices: List[slice] = []
        adr = 0
        for i in range(self.model.nsensor):
            dim = int(self.model.sensor_dim[i])
            if int(self.model.sensor_type[i]) == int(mujoco.mjtSensor.mjSENS_RANGEFINDER):
                self.rf_slices.append(slice(adr, adr + dim))
            adr += dim
        self.n_rf: int = len(self.rf_slices)

        # [dx_b_pre, dy_b_pre, sin(yaw_err), cos(yaw_err), d_prepose, lidar..., hold]
        state_dim = 6 + self.n_rf
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.p.cam_h, self.p.cam_w, 3),
                    dtype=np.uint8,
                ),
                "state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(state_dim,),
                    dtype=np.float32,
                ),
            }
        )

        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )

    # ------------------------- helpers

    def _build_robot_body_mask(self, root_body_name: str = "base_link") -> np.ndarray:
        mask = np.zeros(self.model.nbody, dtype=bool)
        root_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, root_body_name)
        if root_id < 0:
            return mask
        stack = [int(root_id)]
        while stack:
            bid = stack.pop()
            mask[bid] = True
            for child in range(self.model.nbody):
                if int(self.model.body_parentid[child]) == bid:
                    stack.append(child)
        return mask

    def _get_virtual_prepose_world(self) -> Tuple[float, float, float]:
        door = self.data.site_xpos[self.target_site_id].copy()
        x_t = float(door[0])
        y_t = float(door[1] - self.p.prepose_offset_y)
        yaw_t = float(self.p.prepose_yaw)
        return x_t, y_t, yaw_t

    # ------------------------- API

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)

        tries = 0
        while True:
            tries += 1
            mujoco.mj_resetData(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)

            self.step_count = 0
            self.hold_counter = 0
            self.prev_d_anchor = None
            self._best_d_anchor = float("inf")
            self._a_smooth[:] = 0.0

            self._randomize_spawn()

            bx, by, bth = self._get_base_xyth()
            self._base_target[:] = (bx, by, bth)
            self._apply_base_targets()

            for _ in range(int(self.p.settle_steps)):
                mujoco.mj_step(self.model, self.data)

            collided, _ = self._detect_collision_with_name()
            if collided and (tries < int(self.p.spawn_max_tries)):
                continue

            if bool(self.p.spawn_reject_using_lidar) and self.n_rf > 0:
                kmean = self._lidar_kmean_meters(int(self.p.spawn_k_smallest))
                if (kmean < float(self.p.spawn_min_kmean_lidar)) and (tries < int(self.p.spawn_max_tries)):
                    continue

            break

        if self.render_mode == "human" and self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

        return self._get_obs(), self._get_info()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self._a_smooth = (1.0 - float(self.p.tau)) * self._a_smooth + float(self.p.tau) * action
        a = self._a_smooth
        self.step_count += 1

        bx, by, bth = self._get_base_xyth()
        x_t, y_t, yaw_t = self._get_virtual_prepose_world()
        d_pre_before = float(np.linalg.norm(np.array([x_t - bx, y_t - by], dtype=np.float64)))

        if self.debug:
            qpos_before = self.data.qpos.copy()

        vx = float(self.p.max_vx) * float(a[0])
        vy = float(self.p.max_vy) * float(a[1])
        wz = float(self.p.max_wz) * float(a[2])
        self._integrate_base_target(vx, vy, wz, dt=float(self.p.dt))

        for _ in range(self.n_substeps):
            self._apply_base_targets()
            mujoco.mj_step(self.model, self.data)

        bx, by, bth = self._get_base_xyth()
        x_t, y_t, yaw_t = self._get_virtual_prepose_world()
        d_pre = float(np.linalg.norm(np.array([x_t - bx, y_t - by], dtype=np.float64)))
        self._best_d_anchor = min(self._best_d_anchor, d_pre)

        if self.prev_d_anchor is None:
            prog_anchor = 0.0
        else:
            raw = float(np.clip(self.prev_d_anchor - d_pre, -0.20, 0.20))
            prog_anchor = raw / (d_pre + 0.5)
        self.prev_d_anchor = d_pre

        r = 0.0
        r += float(self.p.w_prog_anchor) * prog_anchor
        r += -float(self.p.w_dist_anchor) * d_pre
        r += -float(self.p.w_time)
        r += -float(self.p.w_ctrl_base) * float(np.dot(a, a))

        collided, collided_with = self._detect_collision_with_name()
        collision_terminate = False
        if collided:
            r += -float(self.p.collision_penalty)
            if bool(self.p.terminate_on_collision) and (d_pre > float(self.p.collision_grace_radius)):
                collision_terminate = True

        yaw_err = wrap_to_pi(float(yaw_t - bth))
        pos_ok = d_pre <= float(self.p.base_success_radius)
        yaw_ok = abs(yaw_err) <= float(self.p.prepose_yaw_tolerance)

        self.hold_counter = (self.hold_counter + 1) if (pos_ok and yaw_ok) else 0
        success = self.hold_counter >= int(self.p.hold_steps)

        terminated = bool(success) or bool(collision_terminate)
        truncated = bool(self.step_count >= int(self.p.max_steps))

        if success:
            r += float(self.p.success_bonus)

        if self.render_mode == "human" and self._viewer is not None:
            self._viewer.sync()

        if self.debug and (self.step_count % int(self.debug_every) == 0):
            dq = float(np.linalg.norm(self.data.qpos - qpos_before))
            delta = float(d_pre_before - d_pre)
            print(
                f"[k={self.step_count:04d} t={float(self.data.time):7.3f}] BASE "
                f"d_pre={d_pre:6.3f} (Δ={delta:+.4f}, best={self._best_d_anchor:6.3f}) "
                f"yaw_err={yaw_err:+.3f} "
                f"||Δqpos||={dq:.6f} act_norm={float(np.linalg.norm(a)):.3f} "
                f"coll={int(collided)} hit='{collided_with}' hold={self.hold_counter:02d}"
            )

        obs = self._get_obs()
        info = self._get_info()
        info.update(
            {
                "is_success": bool(success),
                "collided": int(collided),
                "collided_with": collided_with,
                "d_anchor": float(d_pre),
                "best_d_anchor": float(self._best_d_anchor),
                "hold_counter": int(self.hold_counter),
                "act_base_norm": float(np.linalg.norm(a)),
                "yaw_err": float(yaw_err),
            }
        )
        return obs, r, terminated, truncated, info

    def close(self):
        if self.renderer is not None:
            try:
                self.renderer.close()
            except Exception:
                pass
            self.renderer = None
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
        super().close()

    # ------------------------- control

    def _integrate_base_target(self, vx_body, vy_body, wz, dt):
        x, y, th = self._base_target
        cy, sy = math.cos(th), math.sin(th)
        vx_w = cy * vx_body - sy * vy_body
        vy_w = sy * vx_body + cy * vy_body
        self._base_target[0] = x + vx_w * dt
        self._base_target[1] = y + vy_w * dt
        self._base_target[2] = wrap_to_pi(th + wz * dt)

    def _apply_base_targets(self):
        self.data.ctrl[self.aid_x] = float(self._base_target[0])
        self.data.ctrl[self.aid_y] = float(self._base_target[1])
        self.data.ctrl[self.aid_th] = float(self._base_target[2])

    # ------------------------- lidar

    def _lidar_kmean_meters(self, k: int) -> float:
        rf = []
        for sl in self.rf_slices:
            rf.extend(list(np.asarray(self.data.sensordata[sl], dtype=np.float32).reshape(-1)))
        rf = np.asarray(rf, dtype=np.float32)
        rf = np.clip(rf, 0.0, float(self.p.lidar_max_range))

        if rf.size == 0:
            return float("inf")

        k = max(1, min(int(k), int(rf.size)))
        smallest = np.partition(rf, k - 1)[:k]
        return float(np.mean(smallest))

    def _get_lidar_vec(self) -> np.ndarray:
        if self.n_rf == 0:
            return np.zeros((0,), dtype=np.float32)

        rf_vals = []
        for sl in self.rf_slices:
            vals = np.asarray(self.data.sensordata[sl], dtype=np.float32).reshape(-1)
            rf_vals.extend(vals.tolist())

        rf = np.asarray(rf_vals, dtype=np.float32)
        rf = np.clip(rf, 0.0, float(self.p.lidar_max_range))
        rf = rf / max(float(self.p.lidar_max_range), 1e-6)
        return rf.astype(np.float32)

    # ------------------------- obs/render

    def _render_cam(self) -> np.ndarray:
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=self.p.cam_h, width=self.p.cam_w)
        self.renderer.update_scene(self.data, camera=self.cam_id)
        return self.renderer.render()

    def _get_obs(self) -> Dict[str, np.ndarray]:
        return {"image": self._render_cam(), "state": self._get_state_vec()}

    def _get_state_vec(self) -> np.ndarray:
        bx, by, bth = self._get_base_xyth()
        x_t, y_t, yaw_t = self._get_virtual_prepose_world()

        dx_w = float(x_t - bx)
        dy_w = float(y_t - by)

        c, s = math.cos(bth), math.sin(bth)
        dx_b = c * dx_w + s * dy_w
        dy_b = -s * dx_w + c * dy_w

        d_pre = math.hypot(dx_w, dy_w)
        yaw_err = wrap_to_pi(float(yaw_t - bth))

        lidar = self._get_lidar_vec()

        base_state = np.array(
            [
                float(dx_b),
                float(dy_b),
                float(math.sin(yaw_err)),
                float(math.cos(yaw_err)),
                float(d_pre),
            ],
            dtype=np.float32,
        )
        hold = np.array([float(self.hold_counter)], dtype=np.float32)

        return np.concatenate([base_state, lidar, hold], axis=0).astype(np.float32)

    def _get_info(self) -> Dict:
        bx, by, bth = self._get_base_xyth()
        return {"step": int(self.step_count), "base_x": float(bx), "base_y": float(by), "base_th": float(bth)}

    # ------------------------- utils

    def _require_site(self, name: str) -> int:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            raise RuntimeError(f"Site '{name}' missing")
        return int(sid)

    def _require_joint(self, name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"Joint '{name}' missing")
        return int(jid)

    def _require_actuator(self, name: str) -> int:
        aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise RuntimeError(f"Actuator '{name}' missing")
        return int(aid)

    def _get_base_xyth(self):
        return (
            float(self.data.qpos[self.qadr_x]),
            float(self.data.qpos[self.qadr_y]),
            float(self.data.qpos[self.qadr_th]),
        )

    def _randomize_spawn(self):
        if bool(self.p.spawn_around_target):
            anchor = self.data.site_xpos[self.target_site_id].copy()
            ax, ay = float(anchor[0]), float(anchor[1])

            rmin = float(self.p.spawn_target_min_r)
            rmax = float(self.p.spawn_target_max_r)

            u = float(self.np_random.uniform(0.0, 1.0))
            r = math.sqrt((rmax * rmax - rmin * rmin) * u + rmin * rmin)
            ang = float(self.np_random.uniform(-math.pi, math.pi))

            x = ax + r * math.cos(ang)
            y = ay + r * math.sin(ang)
            th = wrap_to_pi(float(self.np_random.uniform(-self.p.spawn_yaw_range, self.p.spawn_yaw_range)))

            self.data.qpos[self.qadr_x] = x
            self.data.qpos[self.qadr_y] = y
            self.data.qpos[self.qadr_th] = th
            mujoco.mj_forward(self.model, self.data)
            return

        x, y, th = self._get_base_xyth()
        x += float(self.np_random.uniform(-self.p.spawn_xy_range, self.p.spawn_xy_range))
        y += float(self.np_random.uniform(-self.p.spawn_xy_range, self.p.spawn_xy_range))
        th = wrap_to_pi(th + float(self.np_random.uniform(-self.p.spawn_yaw_range, self.p.spawn_yaw_range)))
        self.data.qpos[self.qadr_x] = x
        self.data.qpos[self.qadr_y] = y
        self.data.qpos[self.qadr_th] = th
        mujoco.mj_forward(self.model, self.data)

    def _detect_collision_with_name(self) -> Tuple[bool, str]:
        if int(self.data.ncon) <= 0:
            return False, ""
        for ci in range(int(self.data.ncon)):
            c = self.data.contact[ci]
            g1 = int(c.geom1)
            g2 = int(c.geom2)
            b1 = int(self.model.geom_bodyid[g1])
            b2 = int(self.model.geom_bodyid[g2])

            r1 = bool(self._is_robot_body[b1])
            r2 = bool(self._is_robot_body[b2])
            if r1 == r2:
                continue

            other_geom = g2 if r1 else g1
            other_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other_geom) or ""
            if other_name == "floor":
                continue
            return True, other_name
        return False, ""