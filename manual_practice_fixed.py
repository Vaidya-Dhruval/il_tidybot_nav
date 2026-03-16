import os
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "glfw"

import sys
import time
import select
import termios
import tty
import numpy as np
from dataclasses import replace

from tidybot_nav_env_v12_stage0 import TidybotNavEnvV12Stage0
from v12_stage0_config import ENV


class RawKeyboard:
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = None

    def __enter__(self):
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.old_settings is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def get_key(self, timeout=0.0):
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            return sys.stdin.read(1)
        return None


def clamp(a):
    return np.clip(np.asarray(a, dtype=np.float32), -1.0, 1.0)


def pulse_action(action, idx, val, pulse):
    a = np.asarray(action, dtype=np.float32).copy()
    a[idx] = np.clip(a[idx] + val * pulse, -1.0, 1.0)
    return a


def decay_action(action, decay):
    return np.asarray(action, dtype=np.float32) * float(decay)


def main():
    p = replace(ENV, max_steps=1500)
    env = TidybotNavEnvV12Stage0(p, render_mode="human")

    print("[practice] starting")
    print("[practice] controls: w/s vx, a/d vy, q/e wz, space stop, r reset, esc quit")

    obs, info = env.reset(
        seed=0,
        options={
            "manual_spawn": {
                "x": 0.0,
                "y": -2.0,
                "yaw": 1.57,
            }
        },
    )
    print("[practice] reset ok")

    action = np.zeros(3, dtype=np.float32)
    pulse = 0.45
    decay = 0.86

    with RawKeyboard() as kb:
        step = 0
        while True:
            key = kb.get_key(timeout=0.0)

            if key is not None:
                if ord(key) == 27:
                    print("\n[quit]")
                    break
                elif key == " ":
                    action[:] = 0.0
                elif key == "w":
                    action = pulse_action(action, 0, +1.0, pulse)
                elif key == "s":
                    action = pulse_action(action, 0, -1.0, pulse)
                elif key == "a":
                    action = pulse_action(action, 1, +1.0, pulse)
                elif key == "d":
                    action = pulse_action(action, 1, -1.0, pulse)
                elif key == "q":
                    action = pulse_action(action, 2, +1.0, pulse)
                elif key == "e":
                    action = pulse_action(action, 2, -1.0, pulse)
                elif key == "r":
                    print("\n[reset]")
                    obs, info = env.reset(
                        options={
                            "manual_spawn": {
                                "x": 0.0,
                                "y": -2.0,
                                "yaw": 1.57,
                            }
                        },
                    )
                    action[:] = 0.0
                    step = 0
                    continue

            obs, reward, terminated, truncated, info = env.step(action)
            action = decay_action(action, decay)
            action = clamp(action)

            step += 1
            if step % 10 == 0:
                print(
                    f"\r[step={step:04d}] "
                    f"a=({action[0]:+.2f},{action[1]:+.2f},{action[2]:+.2f}) "
                    f"d={float(info.get('d_anchor', 0.0)):.3f} "
                    f"yaw={float(info.get('yaw_err', 0.0)):+.3f} "
                    f"hold={int(info.get('hold_counter', 0)):02d} "
                    f"coll={int(info.get('collided', 0))}",
                    end="",
                    flush=True,
                )

            if terminated or truncated:
                print("\n[episode ended] resetting same fixed scenario")
                obs, info = env.reset(
                    options={
                        "manual_spawn": {
                            "x": 0.0,
                            "y": -2.0,
                            "yaw": 1.57,
                        }
                    },
                )
                action[:] = 0.0
                step = 0

            time.sleep(0.02)

    env.close()


if __name__ == "__main__":
    main()