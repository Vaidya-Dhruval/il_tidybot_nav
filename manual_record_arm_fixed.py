import os
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "glfw"

import sys
import time
import select
import termios
import tty
import argparse
from pathlib import Path

import numpy as np
import mujoco

from tidybot_arm_reach_env_v2 import ArmReachEnvV2


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


class EpisodeBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.success = []
        self.distance = []
        self.best_distance = []
        self.episode_id = []
        self.timestep = []
        self.source = []

    def add_step(self, obs, action, reward, done, info, episode_id, timestep):
        self.states.append(obs.astype(np.float32).copy())
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.success.append(int(info.get("success", False)))
        self.distance.append(float(info.get("distance", np.nan)))
        self.best_distance.append(float(info.get("best_distance", np.nan)))
        self.episode_id.append(int(episode_id))
        self.timestep.append(int(timestep))
        self.source.append("human_cartesian")

    def __len__(self):
        return len(self.states)

    def summary(self):
        if len(self.states) == 0:
            return {
                "steps": 0,
                "success": 0,
                "final_dist": np.nan,
                "best_dist": np.nan,
                "return": 0.0,
            }
        return {
            "steps": len(self.states),
            "success": int(self.success[-1]),
            "final_dist": float(self.distance[-1]),
            "best_dist": float(np.nanmin(np.asarray(self.best_distance, dtype=np.float32))),
            "return": float(np.sum(self.rewards)),
        }

    def to_payload(self):
        return {
            "state": np.stack(self.states, axis=0).astype(np.float32),
            "action": np.stack(self.actions, axis=0).astype(np.float32),
            "reward": np.asarray(self.rewards, dtype=np.float32),
            "done": np.asarray(self.dones, dtype=np.bool_),
            "success": np.asarray(self.success, dtype=np.int8),
            "distance": np.asarray(self.distance, dtype=np.float32),
            "best_distance": np.asarray(self.best_distance, dtype=np.float32),
            "episode_id": np.asarray(self.episode_id, dtype=np.int32),
            "timestep": np.asarray(self.timestep, dtype=np.int32),
            "source": np.asarray(self.source),
        }


class ShardWriter:
    def __init__(self, out_dir: Path, shard_size: int = 5000, prefix: str = "arm_manual_cartesian"):
        self.out_dir = out_dir
        self.shard_size = int(shard_size)
        self.prefix = prefix
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_idx = 0
        self._reset_buffers()

    def _reset_buffers(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.success = []
        self.distance = []
        self.best_distance = []
        self.episode_id = []
        self.timestep = []
        self.source = []

    def add_episode(self, ep: EpisodeBuffer):
        if len(ep) == 0:
            return
        data = ep.to_payload()

        self.states.extend(list(data["state"]))
        self.actions.extend(list(data["action"]))
        self.rewards.extend(list(data["reward"]))
        self.dones.extend(list(data["done"]))
        self.success.extend(list(data["success"]))
        self.distance.extend(list(data["distance"]))
        self.best_distance.extend(list(data["best_distance"]))
        self.episode_id.extend(list(data["episode_id"]))
        self.timestep.extend(list(data["timestep"]))
        self.source.extend(list(data["source"]))

        while len(self.states) >= self.shard_size:
            self.flush(self.shard_size)

    def flush(self, n_items=None):
        n = len(self.states) if n_items is None else min(int(n_items), len(self.states))
        if n <= 0:
            return

        path = self.out_dir / f"{self.prefix}_shard_{self.shard_idx:04d}.npz"
        np.savez_compressed(
            path,
            state=np.stack(self.states[:n], axis=0).astype(np.float32),
            action=np.stack(self.actions[:n], axis=0).astype(np.float32),
            reward=np.asarray(self.rewards[:n], dtype=np.float32),
            done=np.asarray(self.dones[:n], dtype=np.bool_),
            success=np.asarray(self.success[:n], dtype=np.int8),
            distance=np.asarray(self.distance[:n], dtype=np.float32),
            best_distance=np.asarray(self.best_distance[:n], dtype=np.float32),
            episode_id=np.asarray(self.episode_id[:n], dtype=np.int32),
            timestep=np.asarray(self.timestep[:n], dtype=np.int32),
            source=np.asarray(self.source[:n]),
        )
        print(f"[flush] wrote {path} N={n}")
        self.shard_idx += 1

        self.states = self.states[n:]
        self.actions = self.actions[n:]
        self.rewards = self.rewards[n:]
        self.dones = self.dones[n:]
        self.success = self.success[n:]
        self.distance = self.distance[n:]
        self.best_distance = self.best_distance[n:]
        self.episode_id = self.episode_id[n:]
        self.timestep = self.timestep[n:]
        self.source = self.source[n:]

    def flush_all(self):
        self.flush()


class CartesianTeleop:
    def __init__(
        self,
        env: ArmReachEnvV2,
        damping: float = 0.08,
        max_cart_vel: float = 0.18,
        max_joint_cmd: float = 1.2,
        posture_gain: float = 0.0,
        q_nominal=None,
    ):
        self.env = env
        self.damping = float(damping)
        self.max_cart_vel = float(max_cart_vel)
        self.max_joint_cmd = float(max_joint_cmd)
        self.posture_gain = float(posture_gain)
        if q_nominal is None:
            self.q_nominal = np.array([0.0, 0.4, 0.0, 1.0, 0.0, 0.8, 0.0], dtype=np.float64)
        else:
            self.q_nominal = np.asarray(q_nominal, dtype=np.float64)

        self.v_des = np.zeros(3, dtype=np.float32)

    def zero(self):
        self.v_des[:] = 0.0

    def pulse_axis(self, idx: int, sign: float, mag: float):
        self.v_des[idx] = np.clip(
            self.v_des[idx] + sign * mag,
            -self.max_cart_vel,
            self.max_cart_vel,
        )

    def decay(self, factor: float):
        self.v_des *= float(factor)

    def action(self) -> np.ndarray:
        model = self.env.model
        data = self.env.data
        ee_site_id = self.env.ee_site
        dofs = np.asarray(self.env.arm_dadr, dtype=np.int32)

        jacp = np.zeros((3, model.nv), dtype=np.float64)
        jacr = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)

        J = jacp[:, dofs]
        A = J @ J.T + (self.damping ** 2) * np.eye(3, dtype=np.float64)
        J_pinv = J.T @ np.linalg.solve(A, np.eye(3, dtype=np.float64))

        dq_task = J_pinv @ self.v_des.astype(np.float64)

        q_now = np.array([data.qpos[i] for i in self.env.arm_qadr], dtype=np.float64)
        dq_posture = self.posture_gain * (self.q_nominal - q_now)

        N = np.eye(len(dofs), dtype=np.float64) - J_pinv @ J
        dq = dq_task + N @ dq_posture

        dq = np.clip(dq, -self.max_joint_cmd, self.max_joint_cmd)
        return dq.astype(np.float32)


def print_controls():
    print("")
    print("ARM CARTESIAN MANUAL CONTROL")
    print("----------------------------")
    print("j / l : x- / x+")
    print("u / o : y- / y+")
    print("i / m : z+ / z-")
    print("space : stop Cartesian motion")
    print("p : pause/unpause")
    print("r : reset and discard current episode")
    print("k : keep finished episode")
    print("x : discard finished episode")
    print("esc : quit")
    print("")


def episode_end_prompt(summary):
    print("")
    print("[episode end]")
    print(
        f"steps={summary['steps']}  "
        f"return={summary['return']:.3f}  "
        f"success={summary['success']}  "
        f"final_dist={summary['final_dist']:.4f}  "
        f"best_dist={summary['best_dist']:.4f}"
    )
    print("Press 'k' to keep, 'x' to discard, or 'esc' to quit.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_steps", type=int, default=1500)
    ap.add_argument("--sleep", type=float, default=0.02)
    ap.add_argument("--pulse", type=float, default=0.08)
    ap.add_argument("--decay", type=float, default=0.90)
    ap.add_argument("--shard_size", type=int, default=5000)
    ap.add_argument("--fixed_reset", action="store_true")
    args = ap.parse_args()

    env = ArmReachEnvV2(args.xml, render_mode="human")
    env.max_steps = int(args.max_steps)

    if args.fixed_reset:
        env.jitter_x = 0.0
        env.jitter_y = 0.0
        env.jitter_yaw = 0.0

    teleop = CartesianTeleop(
        env=env,
        damping=0.08,
        max_cart_vel=0.18,
        max_joint_cmd=1.2,
        posture_gain=0.0,
    )

    writer = ShardWriter(Path(args.out_dir), shard_size=int(args.shard_size))

    obs, info = env.reset()
    print_controls()

    episode_id = 0
    t_ep = 0
    paused = False
    total_kept = 0
    total_discarded = 0
    epbuf = EpisodeBuffer()

    with RawKeyboard() as kb:
        while True:
            key = kb.get_key(timeout=0.0)

            if key is not None:
                if ord(key) == 27:
                    print("\n[quit]")
                    break
                elif key == "p":
                    paused = not paused
                    print(f"\n[pause] {paused}")
                elif key == " ":
                    teleop.zero()
                elif key == "j":
                    teleop.pulse_axis(0, -1.0, args.pulse)
                elif key == "l":
                    teleop.pulse_axis(0, +1.0, args.pulse)
                elif key == "u":
                    teleop.pulse_axis(1, -1.0, args.pulse)
                elif key == "o":
                    teleop.pulse_axis(1, +1.0, args.pulse)
                elif key == "i":
                    teleop.pulse_axis(2, +1.0, args.pulse)
                elif key == "m":
                    teleop.pulse_axis(2, -1.0, args.pulse)
                elif key == "r":
                    print("\n[reset] discard current episode")
                    epbuf.clear()
                    teleop.zero()
                    t_ep = 0
                    obs, info = env.reset()
                    total_discarded += 1
                    continue

            if paused:
                time.sleep(args.sleep)
                continue

            action = teleop.action()
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            epbuf.add_step(
                obs=obs,
                action=action,
                reward=reward,
                done=done,
                info=info,
                episode_id=episode_id,
                timestep=t_ep,
            )

            obs = next_obs
            teleop.decay(args.decay)
            t_ep += 1

            if t_ep % 10 == 0:
                print(
                    f"\r[ep={episode_id:04d} step={t_ep:04d}] "
                    f"v=({teleop.v_des[0]:+.3f},{teleop.v_des[1]:+.3f},{teleop.v_des[2]:+.3f}) "
                    f"dist={float(info.get('distance', np.nan)):.4f} "
                    f"best={float(info.get('best_distance', np.nan)):.4f} "
                    f"succ={int(info.get('success', 0))}",
                    end="",
                    flush=True,
                )

            if done:
                summary = epbuf.summary()
                episode_end_prompt(summary)

                decision = None
                while decision is None:
                    k = kb.get_key(timeout=0.1)
                    if k is None:
                        continue
                    if ord(k) == 27:
                        decision = "quit"
                    elif k == "k":
                        decision = "keep"
                    elif k == "x":
                        decision = "discard"

                print("")

                if decision == "keep":
                    writer.add_episode(epbuf)
                    total_kept += 1
                    print(f"[keep] episode={episode_id} kept={total_kept} discarded={total_discarded}")
                elif decision == "discard":
                    total_discarded += 1
                    print(f"[discard] episode={episode_id} kept={total_kept} discarded={total_discarded}")
                elif decision == "quit":
                    print("[quit] stopping")
                    break

                episode_id += 1
                epbuf = EpisodeBuffer()
                teleop.zero()
                t_ep = 0
                obs, info = env.reset()

            time.sleep(args.sleep)

    env.close()
    writer.flush_all()
    print(f"[done] kept={total_kept} discarded={total_discarded} shards={writer.shard_idx}")


if __name__ == "__main__":
    main()