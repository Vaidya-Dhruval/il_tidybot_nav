import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class ShardedNpzDataset(Dataset):
    """
    Loads teacher_shard_*.npz produced by record_teacher_v12_stage0.py

    Returns samples:
      state:  float32 (7,)
      action: float32 (3,)

    Note:
      We intentionally IGNORE image here to make BC lighter and faster.
    """
    def __init__(self, shard_glob: str, max_items: int | None = None):
        self.files = sorted(glob.glob(shard_glob))
        if not self.files:
            raise FileNotFoundError(f"No shards matched: {shard_glob}")

        self.index = []   # (file_idx, local_idx)
        self._cache = {}  # file_idx -> (state, action)

        count = 0
        for fi, f in enumerate(self.files):
            with np.load(f) as data:
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

        data = np.load(self.files[fi])
        state = data["state"].astype(np.float32)    # (N,7)
        action = data["action"].astype(np.float32)  # (N,3)

        # keep only one shard cached to control memory
        self._cache = {fi: (state, action)}
        return self._cache[fi]

    def __getitem__(self, idx: int):
        fi, j = self.index[idx]
        state, action = self._load_file(fi)

        s = torch.from_numpy(state[j]).float()
        a = torch.from_numpy(action[j]).float()

        return {
            "state": s,
            "action": a,
        }