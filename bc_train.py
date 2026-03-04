import os
import argparse
import torch
from torch.utils.data import DataLoader, random_split

from dataset import ShardedNpzDataset
from nets import BCPolicy

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_glob", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--max_items", type=int, default=0)  # 0 => no limit
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device]", device)

    max_items = None if args.max_items <= 0 else args.max_items
    ds = ShardedNpzDataset(args.data_glob, max_items=max_items)

    n_val = int(len(ds) * args.val_frac)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    net = BCPolicy(state_dim=7).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    mse = torch.nn.MSELoss()

    best_val = 1e9
    for ep in range(1, args.epochs + 1):
        net.train()
        tr_loss = 0.0
        n = 0
        for batch in train_loader:
            img = batch["image"].to(device, non_blocking=True)
            st  = batch["state"].to(device, non_blocking=True)
            act = batch["action"].to(device, non_blocking=True)

            pred = net(img, st)
            loss = mse(pred, act)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

            tr_loss += float(loss.item()) * img.size(0)
            n += img.size(0)

        tr_loss /= max(n, 1)

        net.eval()
        va_loss = 0.0
        n = 0
        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"].to(device, non_blocking=True)
                st  = batch["state"].to(device, non_blocking=True)
                act = batch["action"].to(device, non_blocking=True)
                pred = net(img, st)
                loss = mse(pred, act)
                va_loss += float(loss.item()) * img.size(0)
                n += img.size(0)
        va_loss /= max(n, 1)

        print(f"[epoch {ep}] train_mse={tr_loss:.6f} val_mse={va_loss:.6f}")

        ckpt = {"model": net.state_dict(), "epoch": ep, "train_mse": tr_loss, "val_mse": va_loss}
        torch.save(ckpt, os.path.join(args.out_dir, "bc_last.pt"))
        if va_loss < best_val:
            best_val = va_loss
            torch.save(ckpt, os.path.join(args.out_dir, "bc_best.pt"))
            print(f"  [save] new best: {best_val:.6f}")

if __name__ == "__main__":
    main()