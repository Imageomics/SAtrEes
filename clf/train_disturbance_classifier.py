#!/usr/bin/env python3
"""
Train a patch-level disturbance classifier.

Input:  DINOv3 patch activations (shards in b0d74ed4)
Target: disturbance labels (pre-extracted by prepare_labels.py from CSV)

Train/test split is performed at the *image* level so all 256 patches
from the same 25 m grid cell stay on the same side.

Usage:
    # Step 1: extract labels (run once, no special deps)
    python prepare_labels.py

    # Step 2: train classifier
    python train_disturbance_classifier.py [--target disturb_recent_class] \
        [--test_frac 0.2] [--seed 42] [--epochs 20] [--batch_size 4096] \
        [--lr 1e-3] [--hidden 0] [--out_dir /path/to/save]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

# ─── Defaults ────────────────────────────────────────────────────────────────

SHARD_DIR = Path("/fs/ess/PAS2136/SAtrEes/saev/shards/b0d74ed4")
LABELS_NPZ = Path("/fs/ess/PAS2136/fangxun/FloraPalooza/scripts/output/disturbance_labels.npz")
DEFAULT_OUT = Path("/fs/ess/PAS2136/fangxun/FloraPalooza/scripts/output")

TOKENS_PER_IMAGE = 256  # 16×16 content patches (CLS excluded)
D_MODEL = 1024


# ─── Data loading ────────────────────────────────────────────────────────────


def load_activations(shard_dir: Path) -> np.ndarray:
    """Memory-map shard files and return (n_images, 256, 1024) content tokens."""
    with open(shard_dir / "metadata.json") as f:
        meta = json.load(f)
    with open(shard_dir / "shards.json") as f:
        shards_info = json.load(f)

    n_images = meta["n_examples"]
    has_cls = meta.get("cls_token", False)
    tokens_total = meta["content_tokens_per_example"] + (1 if has_cls else 0)
    d = meta["d_model"]

    arrays = []
    for sinfo in shards_info:
        n = sinfo["n_examples"]
        mm = np.memmap(
            shard_dir / sinfo["name"],
            dtype="float32",
            mode="r",
            shape=(n, tokens_total, d),
        )
        if has_cls:
            arrays.append(mm[:, 1:, :])  # skip CLS token at position 0
        else:
            arrays.append(mm)

    acts = np.concatenate(arrays, axis=0)
    assert acts.shape == (n_images, TOKENS_PER_IMAGE, D_MODEL)
    return acts


def load_labels(npz_path: Path, target: str) -> np.ndarray:
    """Load pre-extracted labels from .npz file."""
    data = np.load(str(npz_path), allow_pickle=True)
    if target not in data:
        available = [k for k in data.files if k != "shard_filenames"]
        raise KeyError(
            f"Target '{target}' not found in {npz_path}. Available: {available}"
        )
    return data[target]


# ─── Dataset ─────────────────────────────────────────────────────────────────


class PatchDataset(Dataset):
    """Flat dataset of (patch_activation, label_index) pairs."""

    def __init__(self, acts: np.ndarray, labels: np.ndarray, image_indices: np.ndarray):
        self.acts = acts
        self.labels = labels
        self.indices = image_indices
        self.n_patches = TOKENS_PER_IMAGE

    def __len__(self):
        return len(self.indices) * self.n_patches

    def __getitem__(self, idx):
        img_pos = idx // self.n_patches
        patch_pos = idx % self.n_patches
        img_idx = self.indices[img_pos]
        x = torch.from_numpy(self.acts[img_idx, patch_pos].copy())
        y = self.labels[img_idx * self.n_patches + patch_pos]
        return x, y


# ─── Model ───────────────────────────────────────────────────────────────────


def build_model(n_classes: int, hidden: int = 0) -> nn.Module:
    """Linear probe (hidden=0) or single-hidden-layer MLP."""
    if hidden > 0:
        return nn.Sequential(
            nn.Linear(D_MODEL, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_classes),
        )
    return nn.Linear(D_MODEL, n_classes)


# ─── Training ────────────────────────────────────────────────────────────────


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        preds = model(x).argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(y)
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    return preds, labels


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Disturbance patch classifier")
    parser.add_argument("--shard_dir", type=str, default=str(SHARD_DIR))
    parser.add_argument("--labels_npz", type=str, default=str(LABELS_NPZ))
    parser.add_argument("--target", type=str, default="disturb_recent_class",
                        choices=["disturb_recent_class", "disturb_majority"],
                        help="Which label column to classify")
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=0,
                        help="Hidden layer size (0 = linear probe)")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Target: {args.target}")

    # ── Load data ──
    print("Loading activations...")
    t0 = time.time()
    acts = load_activations(Path(args.shard_dir))
    n_images = acts.shape[0]
    print(f"  {n_images} images × {TOKENS_PER_IMAGE} patches × {D_MODEL}d "
          f"({time.time()-t0:.1f}s)")

    print("Loading labels...")
    t0 = time.time()
    raw_labels = load_labels(Path(args.labels_npz), args.target)
    assert raw_labels.shape[0] == n_images * TOKENS_PER_IMAGE, (
        f"Label count {raw_labels.shape[0]} != {n_images}*{TOKENS_PER_IMAGE}. "
        f"Run prepare_labels.py first."
    )
    print(f"  {raw_labels.shape[0]} labels ({time.time()-t0:.1f}s)")

    # ── Encode labels ──
    le = LabelEncoder()
    encoded_labels = le.fit_transform(raw_labels)
    n_classes = len(le.classes_)
    print(f"  {n_classes} classes: {le.classes_.tolist()}")

    # ── Image-level train/test split ──
    print("Splitting train/test at image level...")
    image_ids = np.arange(n_images)
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_frac,
                                random_state=args.seed)
    train_idx, test_idx = next(splitter.split(image_ids, groups=image_ids))
    train_idx = np.sort(train_idx)
    test_idx = np.sort(test_idx)
    print(f"  Train: {len(train_idx)} images ({len(train_idx)*256} patches)")
    print(f"  Test:  {len(test_idx)} images ({len(test_idx)*256} patches)")

    # ── Datasets & loaders ──
    train_ds = PatchDataset(acts, encoded_labels, train_idx)
    test_ds = PatchDataset(acts, encoded_labels, test_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    # ── Model ──
    model = build_model(n_classes, hidden=args.hidden).to(device)
    print(f"Model: {model}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # ── Training loop ──
    print("\nTraining...")
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        elapsed = time.time() - t0

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            preds, labels = evaluate(model, test_loader, device)
            acc = accuracy_score(labels, preds)
            bal_acc = balanced_accuracy_score(labels, preds)
            print(f"  Epoch {epoch:3d} | loss {loss:.4f} | "
                  f"acc {acc:.4f} | bal_acc {bal_acc:.4f} | {elapsed:.1f}s")
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), out_dir / "best_model.pt")
        else:
            print(f"  Epoch {epoch:3d} | loss {loss:.4f} | {elapsed:.1f}s")

    # ── Final evaluation ──
    print("\nFinal evaluation on test set...")
    model.load_state_dict(torch.load(out_dir / "best_model.pt", weights_only=True))
    preds, labels = evaluate(model, test_loader, device)
    acc = accuracy_score(labels, preds)
    bal_acc = balanced_accuracy_score(labels, preds)
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  Balanced accuracy: {bal_acc:.4f}")

    report = classification_report(
        labels, preds, target_names=le.classes_, zero_division=0
    )
    print("\nClassification Report:\n")
    print(report)

    # ── Save artifacts ──
    with open(out_dir / "label_encoder_classes.json", "w") as f:
        json.dump(le.classes_.tolist(), f, indent=2)

    split_info = {
        "seed": args.seed,
        "test_frac": args.test_frac,
        "target": args.target,
        "n_classes": n_classes,
        "n_train_images": int(len(train_idx)),
        "n_test_images": int(len(test_idx)),
        "train_image_indices": train_idx.tolist(),
        "test_image_indices": test_idx.tolist(),
    }
    with open(out_dir / "split_info.json", "w") as f:
        json.dump(split_info, f)

    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(f"Target: {args.target}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Balanced accuracy: {bal_acc:.4f}\n\n")
        f.write(report)

    print(f"\nArtifacts saved to {out_dir}")


if __name__ == "__main__":
    main()
