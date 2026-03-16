import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class ShardedNpzDataset(Dataset):
    def __init__(self, shard_glob: str, max_items: int | None = None, return_meta: bool = False):
        self.files = sorted(glob.glob(shard_glob))
        if not self.files:
            raise FileNotFoundError(f"No shards matched: {shard_glob}")

        self.return_meta = return_meta
        self.index = []
        self._cache = {}

        count = 0
        for fi, f in enumerate(self.files):
            with np.load(f, allow_pickle=True) as data:
                n = int(data["state"].shape[0])
            for j in range(n):
                self.index.append((fi, j))
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

        if "state" not in payload or "action" not in payload:
            raise KeyError(f"{self.files[fi]} must contain at least 'state' and 'action'")

        payload["state"] = payload["state"].astype(np.float32)
        payload["action"] = payload["action"].astype(np.float32)

        self._cache = {fi: payload}
        return payload

    def __getitem__(self, idx: int):
        fi, j = self.index[idx]
        payload = self._load_file(fi)

        item = {
            "state": torch.from_numpy(payload["state"][j]).float(),
            "action": torch.from_numpy(payload["action"][j]).float(),
        }

        if self.return_meta:
            meta = {}
            for k, v in payload.items():
                if k in ["state", "action"]:
                    continue
                meta[k] = v[j]
            item["meta"] = meta

        return item