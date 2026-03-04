import math
import numpy as np

def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class GoToGoalTeacher:
    """
    Deterministic teacher for Stage0 base navigation.

    Supports state:
      len=6: [base_x, base_y, base_theta, goal_x, goal_y, d_anchor]
      len=7: [base_x, base_y, base_theta, goal_x, goal_y, d_anchor, hold_counter]

    Outputs normalized action in [-1,1]^3:
      [a_vx, a_vy, a_wz]

    Patches vs earlier version:
      - Lower default gains to reduce action saturation
      - Distance-based speed scaling (far -> moderate, near -> slow)
      - Near-goal settling (stronger damping)
      - Soft-squash (tanh) before clipping to reduce time at hard limits
    """

    def __init__(
        self,
        max_vx: float = 0.12,
        max_vy: float = 0.08,
        max_wz: float = 0.35,

        # smoother defaults (important)
        kx: float = 0.20,
        ky: float = 0.20,
        kth: float = 0.70,

        # near-goal behavior
        stop_radius: float = 2.0,
        stop_gain_xy: float = 0.15,
        stop_gain_wz: float = 0.35,

        # deadbands reduce jitter
        deadband_xy: float = 0.02,
        deadband_th: float = 0.04,

        face_goal: bool = True,

        # additional patch: global soft-squash strength
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

        self.face_goal = bool(face_goal)
        self.squash_gain = float(squash_gain)

    def act(self, state: np.ndarray) -> np.ndarray:
        vals = state.tolist()
        if len(vals) == 6:
            x, y, th, gx, gy, d = vals
            hold = 0.0
        elif len(vals) >= 7:
            x, y, th, gx, gy, d, hold = vals[:7]
        else:
            raise ValueError(f"Teacher expected state length 6 or 7, got {len(vals)}")

        # At reset your env gives d_anchor = 0.0. Compute distance from positions.
        if d <= 1e-9:
            dx0 = gx - x
            dy0 = gy - y
            d = float(math.sqrt(dx0 * dx0 + dy0 * dy0))

        dx = gx - x
        dy = gy - y

        c = math.cos(th)
        s = math.sin(th)

        # world -> body frame error
        ex =  c * dx + s * dy
        ey = -s * dx + c * dy

        if self.face_goal:
            desired_th = math.atan2(dy, dx)
            e_th = wrap_to_pi(desired_th - th)
        else:
            e_th = 0.0

        # deadbands
        if abs(ex) < self.deadband_xy: ex = 0.0
        if abs(ey) < self.deadband_xy: ey = 0.0
        if abs(e_th) < self.deadband_th: e_th = 0.0

        # ------------------------------------------------------------
        # Distance-based speed scaling (reduces saturation far away)
        # ------------------------------------------------------------
        # Scale translational aggressiveness with distance.
        # - very far: cap to ~1.0
        # - mid: normal
        # - near: slower
        #
        # This is intentionally simple, stable, and monotonic.
        dist_scale = min(1.0, max(0.25, d / 2.5))  # d=0.6 -> 0.25, d=2.5 -> 1.0
        kx = self.kx * dist_scale
        ky = self.ky * dist_scale

        # normalize to [-1,1] (pre-squash)
        a_vx = (kx * ex) / max(self.max_vx, 1e-6)
        a_vy = (ky * ey) / max(self.max_vy, 1e-6)
        a_wz = (self.kth * e_th) / max(self.max_wz, 1e-6)

        # near-goal settling (more damping near door for hold stability)
        if d < self.stop_radius:
            a_vx *= self.stop_gain_xy
            a_vy *= self.stop_gain_xy
            a_wz *= self.stop_gain_wz

        a = np.array([a_vx, a_vy, a_wz], dtype=np.float32)

        # ------------------------------------------------------------
        # Soft-squash BEFORE hard clip to reduce time at ±1.
        # This is the big fix for your ~0.49 sat fraction.
        # ------------------------------------------------------------
        a = np.tanh(self.squash_gain * a)

        return np.clip(a, -1.0, 1.0).astype(np.float32)