import os
import glob
import json
import argparse
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from nets import BCPolicy


class IndexedShardedNpzDataset(Dataset):
    def __init__(self, shard_glob: str, max_items: int | None = None):
        self.files = sorted(glob.glob(shard_glob))
        if not self.files:
            raise FileNotFoundError(f"No shards matched: {shard_glob}")

        self.index = []
        self._cache = {}

        count = 0
        for fi, f in enumerate(self.files):
            with np.load(f, allow_pickle=True) as data:
                n = int(data["state"].shape[0])
                ep_ids = data["episode_id"] if "episode_id" in data.files else np.full((n,), -1, dtype=np.int32)

            for j in range(n):
                self.index.append((fi, j, int(ep_ids[j])))
                count += 1
                if max_items is not None and count >= max_items:
                    break
            if max_items is not None and count >= max_items:
                break

    def __len__(self):
        return len(self.index)

    def _load_file(self, fi: int):
        if fi in self._cache:
            return self._cache[fi]

        with np.load(self.files[fi], allow_pickle=True) as data:
            payload = {k: data[k] for k in data.files}

        payload["state"] = payload["state"].astype(np.float32)
        payload["action"] = payload["action"].astype(np.float32)

        self._cache = {fi: payload}
        return payload

    def __getitem__(self, idx: int):
        fi, j, ep_id = self.index[idx]
        payload = self._load_file(fi)

        item = {
            "state": torch.from_numpy(payload["state"][j]).float(),
            "action": torch.from_numpy(payload["action"][j]).float(),
            "episode_id": int(ep_id),
        }
        return item


def split_by_episode(ds: IndexedShardedNpzDataset, val_frac: float, seed: int):
    episode_to_indices = {}
    for idx, (_, _, ep_id) in enumerate(ds.index):
        episode_to_indices.setdefault(ep_id, []).append(idx)

    episodes = sorted(episode_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(episodes)

    n_val_eps = max(1, int(round(len(episodes) * val_frac))) if len(episodes) > 1 else 0
    val_eps = set(episodes[:n_val_eps])
    train_eps = set(episodes[n_val_eps:]) if n_val_eps > 0 else set(episodes)

    train_idx = []
    val_idx = []
    for ep_id, idxs in episode_to_indices.items():
        if ep_id in val_eps:
            val_idx.extend(idxs)
        else:
            train_idx.extend(idxs)

    return train_idx, val_idx, len(episodes), len(train_eps), len(val_eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_glob", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--max_items", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_pin = (device.type == "cuda")
    print("[device]", device)

    max_items = None if args.max_items <= 0 else args.max_items
    ds = IndexedShardedNpzDataset(args.data_glob, max_items=max_items)
    print("[dataset] total samples:", len(ds))
    print("[dataset] shards:", len(ds.files))

    sample = ds[0]
    state_dim = int(sample["state"].shape[0])
    print("[state_dim]", state_dim)

    train_idx, val_idx, n_eps, n_train_eps, n_val_eps = split_by_episode(ds, args.val_frac, args.seed)
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    print(f"[episodes] total={n_eps} train={n_train_eps} val={n_val_eps}")
    print(f"[split] train_samples={len(train_ds)} val_samples={len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_pin,
    )

    net = BCPolicy(state_dim=state_dim).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    mse = torch.nn.MSELoss()

    best_val = float("inf")

    for ep in range(1, args.epochs + 1):
        net.train()
        tr_loss = 0.0
        n = 0

        print(f"[epoch {ep}] training...")
        for bi, batch in enumerate(train_loader, start=1):
            st = batch["state"].to(device, non_blocking=use_pin)
            act = batch["action"].to(device, non_blocking=use_pin)

            pred = net(st)
            loss = mse(pred, act)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

            tr_loss += float(loss.item()) * st.size(0)
            n += st.size(0)

            if bi % 200 == 0:
                print(f"  [epoch {ep}] batch {bi}/{len(train_loader)} loss={loss.item():.6f}")

        tr_loss /= max(n, 1)

        net.eval()
        va_loss = 0.0
        n = 0
        print(f"[epoch {ep}] validating...")
        with torch.no_grad():
            for batch in val_loader:
                st = batch["state"].to(device, non_blocking=use_pin)
                act = batch["action"].to(device, non_blocking=use_pin)

                pred = net(st)
                loss = mse(pred, act)
                va_loss += float(loss.item()) * st.size(0)
                n += st.size(0)

        va_loss /= max(n, 1)

        print(f"[epoch {ep}] train_mse={tr_loss:.6f} val_mse={va_loss:.6f}")

        ckpt = {
            "model": net.state_dict(),
            "epoch": ep,
            "train_mse": tr_loss,
            "val_mse": va_loss,
            "state_dim": state_dim,
        }
        torch.save(ckpt, os.path.join(args.out_dir, "bc_last.pt"))
        if va_loss < best_val:
            best_val = va_loss
            torch.save(ckpt, os.path.join(args.out_dir, "bc_best.pt"))
            print(f"  [save] new best: {best_val:.6f}")

    meta = {
        "data_glob": args.data_glob,
        "total_samples": len(ds),
        "total_episodes": n_eps,
        "train_episodes": n_train_eps,
        "val_episodes": n_val_eps,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "state_dim": state_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
    }
    with open(os.path.join(args.out_dir, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("[done] wrote train_meta.json")


if __name__ == "__main__":
    main()