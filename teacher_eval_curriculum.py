import os
os.environ.setdefault("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))

import numpy as np
from dataclasses import replace
from stable_baselines3.common.monitor import Monitor

from tidybot_nav_env_v12_stage0 import TidybotNavEnvV12Stage0
from v12_stage0_config import ENV
from teacher import GoToGoalTeacher

def d_from_state(s):
    x,y,th,gx,gy = map(float, s[:5])
    dx,dy = gx-x, gy-y
    return float((dx*dx+dy*dy)**0.5)

def main():
    teacher = GoToGoalTeacher(max_vx=ENV.max_vx, max_vy=ENV.max_vy, max_wz=ENV.max_wz,
                              kx=0.35, ky=0.35, kth=1.0, stop_radius=1.5)

    CURR = [
        ("near", 0.35, 0.65, 300),
        ("mid",  0.80, 1.80, 500),
        ("far",  2.00, 4.00, 1400),
    ]

    rng = np.random.default_rng(0)

    for tag,rmin,rmax,ms in CURR:
        p = replace(ENV, spawn_around_target=True, spawn_target_min_r=rmin, spawn_target_max_r=rmax, max_steps=ms)
        env = Monitor(TidybotNavEnvV12Stage0(p, render_mode=None))

        E=30
        succ=0
        coll=0
        starts=[]
        for ep in range(E):
            obs, info = env.reset(seed=int(rng.integers(0,1_000_000)))
            starts.append(d_from_state(obs["state"]))
            done=False
            while not done:
                s = obs["state"].astype(np.float32)
                if s.shape[0]==6:
                    s = np.concatenate([s, np.array([0.0], dtype=np.float32)])
                a = teacher.act(s)
                obs, r, terminated, truncated, info = env.step(a)
                done = bool(terminated or truncated)
                if int((info or {}).get("collided",0))==1:
                    coll += 1
                    break
            succ += int(bool((info or {}).get("is_success", False)))

        starts = np.array(starts)
        print(tag, "max_steps", ms,
              "d_start min/mean/max", float(starts.min()), float(starts.mean()), float(starts.max()),
              "| success", succ, "/", E, "=", succ/E,
              "| coll_eps", coll, "/", E, "=", coll/E)
        env.close()

if __name__ == "__main__":
    main()