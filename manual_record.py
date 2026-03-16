import os
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "glfw"

import sys
import time
import argparse
import select
import termios
import tty
from pathlib import Path
from dataclasses import replace

import numpy as np
from stable_baselines3.common.monitor import Monitor

from tidybot_nav_env_v12_stage0 import TidybotNavEnvV12Stage0
from v12_stage0_config import ENV
from manual_scenarios_stage0 import SCENARIOS


def make_env(env_params, render_mode="human"):
    env = TidybotNavEnvV12Stage0(env_params, render_mode=render_mode)
    env = Monitor(env)
    return env


def ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    if np.issubdtype(img.dtype, np.floating):
        return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    return img.astype(np.uint8)


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
        self.scenario_name = []

    def add_step(self, obs, action, reward, done, info, episode_id, timestep, scenario_name):
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
        self.scenario_name.append(str(scenario_name))

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
            "scenario_name": np.asarray(self.scenario_name),
        }
        if self.save_images:
            out["image"] = np.stack(self.images, axis=0)
        return out


class ShardWriter:
    def __init__(self, out_dir: Path, shard_size: int, prefix: str = "manual_stage0", save_images: bool = False):
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
        self.scenario_name = []

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
        self.scenario_name.extend(list(data["scenario_name"]))

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
            "scenario_name": np.asarray(self.scenario_name[:n]),
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
        self.scenario_name = self.scenario_name[n:]

    def flush_all(self):
        self.flush()


def clamp_action(a):
    return np.clip(np.asarray(a, dtype=np.float32), -1.0, 1.0)


def print_controls():
    print("")
    print("Manual Stage0 controls")
    print("----------------------")
    print("w/s : pulse +vx / -vx")
    print("a/d : pulse +vy / -vy")
    print("q/e : pulse +wz / -wz")
    print("space : hard stop")
    print("p : pause/unpause")
    print("r : reset same scenario")
    print("n : next scenario")
    print("b : previous scenario")
    print("k : keep current finished episode")
    print("x : discard current finished episode")
    print("esc : quit")
    print("")


def pulse_action(action, idx, val, pulse):
    a = np.asarray(action, dtype=np.float32).copy()
    a[idx] = np.clip(a[idx] + val * pulse, -1.0, 1.0)
    return a


def decay_action(action, decay):
    return np.asarray(action, dtype=np.float32) * float(decay)


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
    print("Press 'k' to keep, 'x' to discard, 'r' to reset same scenario, 'n'/'b' to switch scenario, or 'esc' to quit.")


def get_manual_spawn_for_mode(args, scenario_idx):
    if args.spawn_mode == "fixed":
        return {
            "name": "fixed_manual",
            "x": float(args.start_x),
            "y": float(args.start_y),
            "yaw": float(args.start_yaw),
            "max_steps": int(args.max_steps),
        }

    if args.spawn_mode == "scenario_bank":
        sc = dict(SCENARIOS[scenario_idx % len(SCENARIOS)])
        if args.max_steps is not None:
            sc["max_steps"] = int(args.max_steps)
        return sc

    return None


def reset_env_for_current_mode(env, args, scenario_idx, seed=None):
    if args.spawn_mode == "random":
        obs, info = env.reset(seed=seed)
        scenario = {
            "name": "random",
            "x": np.nan,
            "y": np.nan,
            "yaw": np.nan,
            "max_steps": int(env.unwrapped.p.max_steps),
        }
        return obs, info, scenario, env

    scenario = get_manual_spawn_for_mode(args, scenario_idx)

    max_steps_override = int(scenario["max_steps"])
    if env.unwrapped.p.max_steps != max_steps_override:
        env.close()
        new_params = replace(env.unwrapped.p, max_steps=max_steps_override)
        env = make_env(new_params, render_mode="human")

    obs, info = env.reset(
        seed=seed,
        options={
            "manual_spawn": {
                "x": float(scenario["x"]),
                "y": float(scenario["y"]),
                "yaw": float(scenario["yaw"]),
            }
        },
    )
    return obs, info, scenario, env


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard_size", type=int, default=50000)
    ap.add_argument("--save_images", action="store_true")

    ap.add_argument("--practice_only", action="store_true")

    ap.add_argument("--spawn_mode", type=str, default="fixed", choices=["fixed", "scenario_bank", "random"])
    ap.add_argument("--start_x", type=float, default=0.0)
    ap.add_argument("--start_y", type=float, default=-2.0)
    ap.add_argument("--start_yaw", type=float, default=1.57)

    ap.add_argument("--max_steps", type=int, default=1200)

    ap.add_argument("--pulse", type=float, default=0.45)
    ap.add_argument("--decay", type=float, default=0.86)
    ap.add_argument("--sleep", type=float, default=0.02)

    args = ap.parse_args()

    print("[manual] starting manual recorder")
    print(f"[manual] MUJOCO_GL={os.environ.get('MUJOCO_GL')}")
    print(f"[manual] out_dir={args.out_dir}")
    print(f"[manual] spawn_mode={args.spawn_mode}")
    print(f"[manual] practice_only={args.practice_only}")

    env_params = replace(ENV, max_steps=int(args.max_steps))
    env = make_env(env_params, render_mode="human")

    writer = None
    if not args.practice_only:
        writer = ShardWriter(
            out_dir=Path(args.out_dir),
            shard_size=int(args.shard_size),
            prefix="manual_stage0",
            save_images=bool(args.save_images),
        )

    scenario_idx = 0
    episode_id = 0
    total_kept = 0
    total_discarded = 0

    obs, info, scenario, env = reset_env_for_current_mode(env, args, scenario_idx, seed=args.seed)
    action = np.zeros(3, dtype=np.float32)
    paused = False
    epbuf = EpisodeBuffer(save_images=bool(args.save_images))
    t_ep = 0

    print_controls()
    print(f"[manual] current scenario: {scenario['name']}")

    with RawKeyboard() as kb:
        while True:
            key = kb.get_key(timeout=0.0)

            if key is not None:
                if ord(key) == 27:
                    print("\n[quit] exiting...")
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
                    print(f"\n[reset] same scenario: {scenario['name']}")
                    epbuf.clear()
                    action[:] = 0.0
                    t_ep = 0
                    obs, info, scenario, env = reset_env_for_current_mode(env, args, scenario_idx)
                    continue
                elif key == "n" and args.spawn_mode == "scenario_bank":
                    scenario_idx = (scenario_idx + 1) % len(SCENARIOS)
                    print(f"\n[scenario] next -> {SCENARIOS[scenario_idx]['name']}")
                    epbuf.clear()
                    action[:] = 0.0
                    t_ep = 0
                    obs, info, scenario, env = reset_env_for_current_mode(env, args, scenario_idx)
                    continue
                elif key == "b" and args.spawn_mode == "scenario_bank":
                    scenario_idx = (scenario_idx - 1) % len(SCENARIOS)
                    print(f"\n[scenario] prev -> {SCENARIOS[scenario_idx]['name']}")
                    epbuf.clear()
                    action[:] = 0.0
                    t_ep = 0
                    obs, info, scenario, env = reset_env_for_current_mode(env, args, scenario_idx)
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
                scenario_name=scenario["name"],
            )

            t_ep += 1
            obs = next_obs
            action = decay_action(action, float(args.decay))
            action = clamp_action(action)

            if t_ep % 10 == 0:
                print(
                    f"\r[ep={episode_id:04d} sc={scenario['name']} step={t_ep:04d}] "
                    f"a=({action[0]:+.2f},{action[1]:+.2f},{action[2]:+.2f}) "
                    f"d={float((info or {}).get('d_anchor', np.nan)):.3f} "
                    f"yaw={float((info or {}).get('yaw_err', np.nan)):+.3f} "
                    f"hold={int((info or {}).get('hold_counter', 0)):02d} "
                    f"coll={int((info or {}).get('collided', 0))}",
                    end="",
                    flush=True,
                )

            if done:
                summary = epbuf.summary()
                episode_end_prompt(summary)

                if args.practice_only:
                    print("[practice] auto-discard and reset same/current scenario")
                    epbuf.clear()
                    action[:] = 0.0
                    t_ep = 0
                    obs, info, scenario, env = reset_env_for_current_mode(env, args, scenario_idx)
                    continue

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
                        decision = "reset_same"
                    elif k == "n" and args.spawn_mode == "scenario_bank":
                        decision = "next"
                    elif k == "b" and args.spawn_mode == "scenario_bank":
                        decision = "prev"

                print("")

                if decision == "keep":
                    writer.add_episode(epbuf)
                    total_kept += 1
                    print(f"[keep] episode={episode_id} kept={total_kept} discarded={total_discarded}")
                    episode_id += 1

                elif decision == "discard":
                    total_discarded += 1
                    print(f"[discard] episode={episode_id} kept={total_kept} discarded={total_discarded}")
                    episode_id += 1

                elif decision == "reset_same":
                    total_discarded += 1
                    print(f"[reset] discard and repeat scenario={scenario['name']}")

                elif decision == "next":
                    total_discarded += 1
                    scenario_idx = (scenario_idx + 1) % len(SCENARIOS)
                    print(f"[scenario] next -> {SCENARIOS[scenario_idx]['name']}")

                elif decision == "prev":
                    total_discarded += 1
                    scenario_idx = (scenario_idx - 1) % len(SCENARIOS)
                    print(f"[scenario] prev -> {SCENARIOS[scenario_idx]['name']}")

                elif decision == "quit":
                    print("[quit] stopping collection")
                    break

                epbuf = EpisodeBuffer(save_images=bool(args.save_images))
                action[:] = 0.0
                t_ep = 0
                obs, info, scenario, env = reset_env_for_current_mode(env, args, scenario_idx)

            time.sleep(args.sleep)

    env.close()
    if writer is not None:
        writer.flush_all()
        print(f"[done] kept={total_kept} discarded={total_discarded} shards={writer.shard_idx}")
    else:
        print("[done] practice session ended")