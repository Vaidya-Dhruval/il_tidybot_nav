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


def ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    if np.issubdtype(img.dtype, np.floating):
        return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    return img.astype(np.uint8)


class EpisodeBuffer:
    def __init__(self, save_images: bool = False):
        self.save_images = save_images
        self.clear()

    def clear(self):
        self.images = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.success = []
        self.episode_ids = []
        self.timesteps = []
        self.collided = []
        self.d_anchor = []
        self.yaw_err = []
        self.hold_counter = []
        self.source = []

    def add_step(self, obs, action, reward, done, info, episode_id, timestep):
        if self.save_images:
            self.images.append(ensure_uint8(obs["image"]))
        self.states.append(obs["state"].astype(np.float32).copy())
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.success.append(int((info or {}).get("is_success", 0)))
        self.episode_ids.append(int(episode_id))
        self.timesteps.append(int(timestep))
        self.collided.append(int((info or {}).get("collided", 0)))
        self.d_anchor.append(float((info or {}).get("d_anchor", np.nan)))
        self.yaw_err.append(float((info or {}).get("yaw_err", np.nan)))
        self.hold_counter.append(int((info or {}).get("hold_counter", 0)))
        self.source.append("human")

    def __len__(self):
        return len(self.states)

    def summary(self):
        if len(self.states) == 0:
            return {
                "steps": 0,
                "success": 0,
                "collided": 0,
                "final_d_anchor": np.nan,
                "final_yaw_err": np.nan,
                "return": 0.0,
            }

        return {
            "steps": len(self.states),
            "success": int(self.success[-1]),
            "collided": int(np.max(self.collided)),
            "final_d_anchor": float(self.d_anchor[-1]),
            "final_yaw_err": float(self.yaw_err[-1]),
            "return": float(np.sum(self.rewards)),
        }

    def to_dict(self):
        out = {
            "state": np.stack(self.states, axis=0).astype(np.float32),
            "action": np.stack(self.actions, axis=0).astype(np.float32),
            "reward": np.asarray(self.rewards, dtype=np.float32),
            "done": np.asarray(self.dones, dtype=np.bool_),
            "success": np.asarray(self.success, dtype=np.int8),
            "episode_id": np.asarray(self.episode_ids, dtype=np.int32),
            "timestep": np.asarray(self.timesteps, dtype=np.int32),
            "collided": np.asarray(self.collided, dtype=np.int8),
            "d_anchor": np.asarray(self.d_anchor, dtype=np.float32),
            "yaw_err": np.asarray(self.yaw_err, dtype=np.float32),
            "hold_counter": np.asarray(self.hold_counter, dtype=np.int16),
            "source": np.asarray(self.source),
        }
        if self.save_images:
            out["image"] = np.stack(self.images, axis=0)
        return out


class ShardWriter:
    def __init__(self, out_dir: Path, shard_size: int, prefix: str = "manual_stage0_fixed", save_images: bool = False):
        self.out_dir = out_dir
        self.shard_size = int(shard_size)
        self.prefix = prefix
        self.save_images = save_images
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_idx = 0
        self._reset_buffers()

    def _reset_buffers(self):
        self.images = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.success = []
        self.episode_ids = []
        self.timesteps = []
        self.collided = []
        self.d_anchor = []
        self.yaw_err = []
        self.hold_counter = []
        self.source = []

    def add_episode(self, ep: EpisodeBuffer):
        if len(ep) == 0:
            return
        data = ep.to_dict()
        if self.save_images:
            self.images.extend(list(data["image"]))
        self.states.extend(list(data["state"]))
        self.actions.extend(list(data["action"]))
        self.rewards.extend(list(data["reward"]))
        self.dones.extend(list(data["done"]))
        self.success.extend(list(data["success"]))
        self.episode_ids.extend(list(data["episode_id"]))
        self.timesteps.extend(list(data["timestep"]))
        self.collided.extend(list(data["collided"]))
        self.d_anchor.extend(list(data["d_anchor"]))
        self.yaw_err.extend(list(data["yaw_err"]))
        self.hold_counter.extend(list(data["hold_counter"]))
        self.source.extend(list(data["source"]))

        while len(self.states) >= self.shard_size:
            self.flush(self.shard_size)

    def flush(self, n_items=None):
        n = len(self.states) if n_items is None else min(int(n_items), len(self.states))
        if n <= 0:
            return

        path = self.out_dir / f"{self.prefix}_shard_{self.shard_idx:04d}.npz"

        payload = {
            "state": np.stack(self.states[:n], axis=0).astype(np.float32),
            "action": np.stack(self.actions[:n], axis=0).astype(np.float32),
            "reward": np.asarray(self.rewards[:n], dtype=np.float32),
            "done": np.asarray(self.dones[:n], dtype=np.bool_),
            "success": np.asarray(self.success[:n], dtype=np.int8),
            "episode_id": np.asarray(self.episode_ids[:n], dtype=np.int32),
            "timestep": np.asarray(self.timesteps[:n], dtype=np.int32),
            "collided": np.asarray(self.collided[:n], dtype=np.int8),
            "d_anchor": np.asarray(self.d_anchor[:n], dtype=np.float32),
            "yaw_err": np.asarray(self.yaw_err[:n], dtype=np.float32),
            "hold_counter": np.asarray(self.hold_counter[:n], dtype=np.int16),
            "source": np.asarray(self.source[:n]),
        }
        if self.save_images:
            payload["image"] = np.stack(self.images[:n], axis=0)

        np.savez_compressed(path, **payload)
        print(f"[flush] wrote {path}  N={n}")
        self.shard_idx += 1

        if self.save_images:
            self.images = self.images[n:]
        self.states = self.states[n:]
        self.actions = self.actions[n:]
        self.rewards = self.rewards[n:]
        self.dones = self.dones[n:]
        self.success = self.success[n:]
        self.episode_ids = self.episode_ids[n:]
        self.timesteps = self.timesteps[n:]
        self.collided = self.collided[n:]
        self.d_anchor = self.d_anchor[n:]
        self.yaw_err = self.yaw_err[n:]
        self.hold_counter = self.hold_counter[n:]
        self.source = self.source[n:]

    def flush_all(self):
        self.flush()


def print_controls():
    print("")
    print("Manual fixed-record controls")
    print("----------------------------")
    print("w/s : pulse +vx / -vx")
    print("a/d : pulse +vy / -vy")
    print("q/e : pulse +wz / -wz")
    print("space : hard stop")
    print("p : pause/unpause")
    print("r : reset same fixed scenario and discard current episode")
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
        f"collided={summary['collided']}  "
        f"d_anchor={summary['final_d_anchor']:.3f}  "
        f"yaw_err={summary['final_yaw_err']:.3f}"
    )
    print("Press 'k' to keep, 'x' to discard, 'r' to reset/discard, or 'esc' to quit.")


def reset_fixed(env, x, y, yaw, seed=None):
    return env.reset(
        seed=seed,
        options={
            "manual_spawn": {
                "x": float(x),
                "y": float(y),
                "yaw": float(yaw),
            }
        },
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start_x", type=float, default=0.0)
    ap.add_argument("--start_y", type=float, default=-2.0)
    ap.add_argument("--start_yaw", type=float, default=1.57)
    ap.add_argument("--max_steps", type=int, default=1500)
    ap.add_argument("--pulse", type=float, default=0.45)
    ap.add_argument("--decay", type=float, default=0.86)
    ap.add_argument("--sleep", type=float, default=0.02)
    ap.add_argument("--shard_size", type=int, default=5000)
    ap.add_argument("--save_images", action="store_true")
    args = ap.parse_args()

    p = replace(ENV, max_steps=int(args.max_steps))
    env = TidybotNavEnvV12Stage0(p, render_mode="human")
    writer = ShardWriter(
        out_dir=Path(args.out_dir),
        shard_size=int(args.shard_size),
        prefix="manual_stage0_fixed",
        save_images=bool(args.save_images),
    )

    print("[manual_record_fixed] starting")
    print(f"[manual_record_fixed] MUJOCO_GL={os.environ.get('MUJOCO_GL')}")
    print(f"[manual_record_fixed] fixed pose = ({args.start_x}, {args.start_y}, {args.start_yaw})")

    obs, info = reset_fixed(env, args.start_x, args.start_y, args.start_yaw, seed=args.seed)
    print("[manual_record_fixed] reset ok")
    print_controls()

    action = np.zeros(3, dtype=np.float32)
    episode_id = 0
    t_ep = 0
    paused = False
    total_kept = 0
    total_discarded = 0
    epbuf = EpisodeBuffer(save_images=bool(args.save_images))
    prev_in_goal = False

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
                    action[:] = 0.0
                elif key == "w":
                    action = pulse_action(action, 0, +1.0, float(args.pulse))
                elif key == "s":
                    action = pulse_action(action, 0, -1.0, float(args.pulse))
                elif key == "a":
                    action = pulse_action(action, 1, +1.0, float(args.pulse))
                elif key == "d":
                    action = pulse_action(action, 1, -1.0, float(args.pulse))
                elif key == "q":
                    action = pulse_action(action, 2, +1.0, float(args.pulse))
                elif key == "e":
                    action = pulse_action(action, 2, -1.0, float(args.pulse))
                elif key == "r":
                    print("\n[reset] discard current episode and restart fixed scenario")
                    epbuf.clear()
                    action[:] = 0.0
                    t_ep = 0
                    prev_in_goal = False
                    obs, info = reset_fixed(env, args.start_x, args.start_y, args.start_yaw)
                    total_discarded += 1
                    continue

            if paused:
                time.sleep(args.sleep)
                continue

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

            d_anchor = float(info.get("d_anchor", np.nan))
            yaw_err = float(info.get("yaw_err", np.nan))
            hold_counter = int(info.get("hold_counter", 0))
            pos_ok = int(d_anchor <= env.p.base_success_radius)
            yaw_ok = int(abs(yaw_err) <= env.p.prepose_yaw_tolerance)
            in_goal = int(pos_ok and yaw_ok)

            if bool(in_goal) and not prev_in_goal:
                print(
                    f"\n[enter-goal] "
                    f"d={d_anchor:.3f} yaw={yaw_err:+.3f} "
                    f"HOLD={hold_counter}/{env.p.hold_steps}"
                )
            elif (not bool(in_goal)) and prev_in_goal:
                print(
                    f"\n[exit-goal]  "
                    f"d={d_anchor:.3f} yaw={yaw_err:+.3f} "
                    f"HOLD={hold_counter}/{env.p.hold_steps}"
                )
            prev_in_goal = bool(in_goal)

            obs = next_obs
            action = decay_action(action, float(args.decay))
            action = clamp(action)
            t_ep += 1

            if t_ep % 10 == 0:
                print(
                    f"\r[ep={episode_id:04d} step={t_ep:04d}] "
                    f"a=({action[0]:+.2f},{action[1]:+.2f},{action[2]:+.2f}) "
                    f"d={d_anchor:.3f} "
                    f"yaw={yaw_err:+.3f} "
                    f"POS_OK={pos_ok} "
                    f"YAW_OK={yaw_ok} "
                    f"IN_GOAL={in_goal} "
                    f"HOLD={hold_counter:02d}/{env.p.hold_steps:02d} "
                    f"coll={int(info.get('collided', 0))}",
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
                    elif k == "r":
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
                epbuf = EpisodeBuffer(save_images=bool(args.save_images))
                action[:] = 0.0
                t_ep = 0
                prev_in_goal = False
                obs, info = reset_fixed(env, args.start_x, args.start_y, args.start_yaw)

            time.sleep(args.sleep)

    env.close()
    writer.flush_all()
    print(f"[done] kept={total_kept} discarded={total_discarded} shards={writer.shard_idx}")


if __name__ == "__main__":
    main()