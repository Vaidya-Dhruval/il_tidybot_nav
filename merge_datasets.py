import argparse
import glob
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_dir", type=str, required=True)
    ap.add_argument("--dagger_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    teacher_dir = Path(args.teacher_dir)
    dagger_dir = Path(args.dagger_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    teacher_files = sorted(glob.glob(str(teacher_dir / "*.npz")))
    dagger_files = sorted(glob.glob(str(dagger_dir / "*.npz")))

    if not teacher_files:
        raise FileNotFoundError(f"No teacher shards found in {teacher_dir}")
    if not dagger_files:
        raise FileNotFoundError(f"No dagger shards found in {dagger_dir}")

    idx = 0
    for f in teacher_files:
        dst = out_dir / f"merged_shard_{idx:04d}.npz"
        shutil.copy2(f, dst)
        idx += 1

    for f in dagger_files:
        dst = out_dir / f"merged_shard_{idx:04d}.npz"
        shutil.copy2(f, dst)
        idx += 1

    print(f"[merge] teacher={len(teacher_files)} dagger={len(dagger_files)} total={idx}")
    print(f"[merge] output dir: {out_dir}")


if __name__ == "__main__":
    main()