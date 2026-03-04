import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class ShardedNpzDataset(Dataset):
    """
    Loads teacher_shard_*.npz produced by record_teacher_v12_stage0.py
    Provides samples: {"image": CHW float [0,1], "state": float32, "action": float32}
    """
    def __init__(self, shard_glob: str, max_items: int | None = None):
        self.files = sorted(glob.glob(shard_glob))
        if not self.files:
            raise FileNotFoundError(f"No shards matched: {shard_glob}")

        self.index = []  # (file_idx, local_idx)
        self._cache = {} # file_idx -> (image,state,action)

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
        image = data["image"]  # uint8 (N,H,W,3)
        state = data["state"].astype(np.float32)
        action = data["action"].astype(np.float32)
        self._cache = {fi: (image, state, action)}  # keep only one file cached
        return self._cache[fi]

    def __getitem__(self, idx: int):
        fi, j = self.index[idx]
        image, state, action = self._load_file(fi)

        img = image[j]   # (H,W,3) uint8
        s = state[j]     # (7,)
        a = action[j]    # (3,)

        img_t = torch.from_numpy(img).permute(2, 0, 1).contiguous().float() / 255.0
        s_t = torch.from_numpy(s).float()
        a_t = torch.from_numpy(a).float()
        return {"image": img_t, "state": s_t, "action": a_t}