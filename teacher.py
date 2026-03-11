import math
import numpy as np


class GoToGoalTeacher:
    """
    Teacher for LiDAR-augmented state:

      state = [
          dx_b, dy_b, sin(th), cos(th), d_anchor,
          lidar_0, ..., lidar_{N-1},
          hold_counter
      ]

    Action:
      [a_vx, a_vy, a_wz] in [-1,1]

    Note:
      Teacher currently uses navigation geometry directly.
      LiDAR is not yet used inside teacher action computation.
      That is fine for now because LiDAR is being added primarily
      for the LEARNED policy.
    """

    def __init__(
        self,
        max_vx: float = 0.12,
        max_vy: float = 0.08,
        max_wz: float = 0.35,
        kx: float = 0.20,
        ky: float = 0.20,
        kth: float = 0.70,
        stop_radius: float = 2.0,
        stop_gain_xy: float = 0.15,
        stop_gain_wz: float = 0.35,
        deadband_xy: float = 0.02,
        deadband_th: float = 0.04,
        squash_gain: float = 0.8,
    ):
        self.max_vx = float(max_vx)
        self.max_vy = float(max_vy)
        self.max_wz = float(max_wz)

        self.kx = float(kx)
        self.ky = float(ky)
        self.kth = float(kth)

        self.stop_radius = float(stop_radius)
        self.stop_gain_xy = float(stop_gain_xy)
        self.stop_gain_wz = float(stop_gain_wz)

        self.deadband_xy = float(deadband_xy)
        self.deadband_th = float(deadband_th)

        self.squash_gain = float(squash_gain)

    def act(self, state: np.ndarray) -> np.ndarray:
        vals = state.astype(np.float32)

        if vals.shape[0] < 6:
            raise ValueError(f"Teacher expected at least 6 dims, got {vals.shape[0]}")

        dx_b = float(vals[0])
        dy_b = float(vals[1])
        sin_th = float(vals[2])
        cos_th = float(vals[3])
        d = float(vals[4])

        # reconstruct heading for a simple heading-consistency yaw term
        th = math.atan2(sin_th, cos_th)

        # desired local heading to target in body frame
        e_th = math.atan2(dy_b, dx_b)

        if abs(dx_b) < self.deadband_xy:
            dx_b = 0.0
        if abs(dy_b) < self.deadband_xy:
            dy_b = 0.0
        if abs(e_th) < self.deadband_th:
            e_th = 0.0

        dist_scale = min(1.0, max(0.25, d / 2.5))
        kx = self.kx * dist_scale
        ky = self.ky * dist_scale

        a_vx = (kx * dx_b) / max(self.max_vx, 1e-6)
        a_vy = (ky * dy_b) / max(self.max_vy, 1e-6)
        a_wz = (self.kth * e_th) / max(self.max_wz, 1e-6)

        if d < self.stop_radius:
            a_vx *= self.stop_gain_xy
            a_vy *= self.stop_gain_xy
            a_wz *= self.stop_gain_wz

        a = np.array([a_vx, a_vy, a_wz], dtype=np.float32)
        a = np.tanh(self.squash_gain * a)
        return np.clip(a, -1.0, 1.0).astype(np.float32)