import math
import numpy as np

def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

class GoToGoalTeacher:
    """
    Deterministic teacher for Stage0 base navigation.

    Expects state vector:
      [base_x, base_y, base_theta, goal_x, goal_y, d_anchor, hold_counter]

    Outputs normalized action:
      [a_vx, a_vy, a_wz] in [-1, 1]
    (Your env scales to physical velocities internally.)
    """
    def __init__(
        self,
        max_vx: float = 0.12,
        max_vy: float = 0.08,
        max_wz: float = 0.35,
        kx: float = 0.9,
        ky: float = 0.9,
        kth: float = 1.8,
        stop_radius: float = 1.2,
        stop_gain_xy: float = 0.25,
        stop_gain_wz: float = 0.5,
        deadband_xy: float = 0.01,
        deadband_th: float = 0.02,
        face_goal: bool = True,
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
        self.face_goal = bool(face_goal)

    def act(self, state: np.ndarray) -> np.ndarray:
        x, y, th, gx, gy, d, hold = state[:7].tolist()

        dx, dy = gx - x, gy - y
        c, s = math.cos(th), math.sin(th)

        # world -> body frame error
        ex =  c * dx + s * dy
        ey = -s * dx + c * dy

        if self.face_goal:
            desired_th = math.atan2(dy, dx)
            e_th = wrap_to_pi(desired_th - th)
        else:
            e_th = 0.0

        # deadbands to reduce jitter
        if abs(ex) < self.deadband_xy: ex = 0.0
        if abs(ey) < self.deadband_xy: ey = 0.0
        if abs(e_th) < self.deadband_th: e_th = 0.0

        # normalize to [-1,1]
        a_vx = (self.kx * ex) / max(self.max_vx, 1e-6)
        a_vy = (self.ky * ey) / max(self.max_vy, 1e-6)
        a_wz = (self.kth * e_th) / max(self.max_wz, 1e-6)

        # near-goal settling for hold stability
        if d < self.stop_radius:
            a_vx *= self.stop_gain_xy
            a_vy *= self.stop_gain_xy
            a_wz *= self.stop_gain_wz

        return np.clip(np.array([a_vx, a_vy, a_wz], dtype=np.float32), -1.0, 1.0)