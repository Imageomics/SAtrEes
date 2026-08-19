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
        [--lr 1e-3] [--hidden 0] [--balanced] [--out_dir /path/to/save]
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
from torch.utils.data import DataLoader, Dataset, Sampler

# --- Defaults ----------------------------------------------------------------

SHARD_DIR = Path("/fs/ess/PAS2136/SAtrEes/saev/shards/b0d74ed4")
LABELS_NPZ = Path("/fs/ess/PAS2136/fangxun/FloraPalooza/SAtrEes/clf/output/disturbance_labels.npz")
DEFAULT_OUT = Path("/fs/ess/PAS2136/fangxun/FloraPalooza/SAtrEes/clf/output")

TOKENS_PER_IMAGE = 256  # 16x16 content patches (CLS excluded)
D_MODEL = 1024


# --- Data loading -------------------------------------------------------------


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


def load_temporal_field(npz_path: Path, field: str) -> np.ndarray:
    """Load disturb_recent_yr or disturb_years_since (float32, NaN=missing)."""
    data = np.load(str(npz_path), allow_pickle=True)
    if field not in data:
        raise KeyError(f"Field '{field}' not in {npz_path}. "
                       f"Re-run prepare_labels.py to regenerate.")
    return data[field]


# --- Dataset ------------------------------------------------------------------


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


class FilteredPatchDataset(Dataset):
    """
    Dataset that only includes patches passing all filters:
      - label NOT in exclude_labels
      - patch_mask[global_idx] == True (if provided)
    Stores a precomputed list of valid (image_idx, patch_pos) pairs.
    """

    def __init__(self, acts: np.ndarray, labels: np.ndarray,
                 image_indices: np.ndarray, exclude_labels: set,
                 patch_mask: np.ndarray = None):
        self.acts = acts
        self.labels = labels
        # Build flat index of valid patches
        valid = []
        for img_idx in image_indices:
            for p in range(TOKENS_PER_IMAGE):
                global_idx = img_idx * TOKENS_PER_IMAGE + p
                lbl = labels[global_idx]
                if lbl in exclude_labels:
                    continue
                if patch_mask is not None and not patch_mask[global_idx]:
                    continue
                valid.append((img_idx, p))
        self.valid_indices = valid

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        img_idx, patch_pos = self.valid_indices[idx]
        x = torch.from_numpy(self.acts[img_idx, patch_pos].copy())
        y = self.labels[img_idx * TOKENS_PER_IMAGE + patch_pos]
        return x, y

    def get_labels(self) -> np.ndarray:
        """Return label array for all valid patches (for sampler construction)."""
        return np.array([
            self.labels[img_idx * TOKENS_PER_IMAGE + p]
            for img_idx, p in self.valid_indices
        ])


# --- Balanced Batch Sampler ---------------------------------------------------


class BalancedBatchSampler(Sampler):
    """
    Yields batches where each *non-empty* class contributes equally.
    Empty classes (0 samples after filtering) are automatically skipped.
    Rare classes cycle through their indices (with reshuffle);
    dominant classes are subsampled each epoch.
    """

    def __init__(self, labels: np.ndarray, batch_size: int, n_classes: int,
                 drop_last: bool = False, seed: int = 42):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)

        all_class_indices = [
            np.where(labels == c)[0] for c in range(n_classes)
        ]
        # Only keep classes that have at least 1 sample
        self.active_classes = [c for c in range(n_classes) if len(all_class_indices[c]) > 0]
        self.class_indices = {c: all_class_indices[c] for c in self.active_classes}
        self.n_active = len(self.active_classes)

        if self.n_active == 0:
            raise ValueError("No classes have samples — cannot build sampler.")

        self.n_samples = len(labels)
        self._queues = {}
        self._ptrs = {}

    def _refill(self, c: int):
        q = self.class_indices[c].copy()
        self.rng.shuffle(q)
        self._queues[c] = q
        self._ptrs[c] = 0

    def _draw(self, c: int, n: int) -> np.ndarray:
        drawn = []
        remaining = n
        while remaining > 0:
            if c not in self._queues or self._ptrs[c] >= len(self._queues[c]):
                self._refill(c)
            avail = len(self._queues[c]) - self._ptrs[c]
            take = min(avail, remaining)
            drawn.append(self._queues[c][self._ptrs[c]:self._ptrs[c] + take])
            self._ptrs[c] += take
            remaining -= take
        return np.concatenate(drawn)

    def __iter__(self):
        for c in self.active_classes:
            self._refill(c)

        per_class = self.batch_size // self.n_active
        remainder = self.batch_size % self.n_active
        n_batches = self.n_samples // self.batch_size
        if not self.drop_last and (self.n_samples % self.batch_size > 0):
            n_batches += 1

        for _ in range(n_batches):
            batch = []
            for c in self.active_classes:
                batch.append(self._draw(c, per_class))
            if remainder > 0:
                extra = self.rng.choice(
                    self.active_classes, size=remainder, replace=False
                )
                for c in extra:
                    batch.append(self._draw(c, 1))
            batch = np.concatenate(batch)
            self.rng.shuffle(batch)
            yield batch.tolist()

    def __len__(self):
        if self.drop_last:
            return self.n_samples // self.batch_size
        return (self.n_samples + self.batch_size - 1) // self.batch_size


# --- Model --------------------------------------------------------------------


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


# --- Training -----------------------------------------------------------------


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


# --- Main ---------------------------------------------------------------------


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
    parser.add_argument("--balanced", action="store_true",
                        help="Use class-balanced batch sampling")
    parser.add_argument("--exclude_not_mapped", action="store_true",
                        help="Exclude 'Not Mapped' patches from train and test")
    parser.add_argument("--max_years_since", type=float, default=None,
                        help="Only keep patches with disturb_years_since <= X")
    parser.add_argument("--min_recent_yr", type=float, default=None,
                        help="Only keep patches with disturb_recent_yr >= Y")
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
    print(f"Balanced sampling: {args.balanced}")
    print(f"Exclude Not Mapped: {args.exclude_not_mapped}")

    # -- Load data --
    print("Loading activations...")
    t0 = time.time()
    acts = load_activations(Path(args.shard_dir))
    n_images = acts.shape[0]
    print(f"  {n_images} images x {TOKENS_PER_IMAGE} patches x {D_MODEL}d "
          f"({time.time()-t0:.1f}s)")

    print("Loading labels...")
    t0 = time.time()
    raw_labels = load_labels(Path(args.labels_npz), args.target)
    assert raw_labels.shape[0] == n_images * TOKENS_PER_IMAGE, (
        f"Label count {raw_labels.shape[0]} != {n_images}*{TOKENS_PER_IMAGE}. "
        f"Run prepare_labels.py first."
    )
    print(f"  {raw_labels.shape[0]} labels ({time.time()-t0:.1f}s)")

    # -- Encode labels --
    le = LabelEncoder()
    encoded_labels = le.fit_transform(raw_labels)
    n_classes = len(le.classes_)
    print(f"  {n_classes} classes: {le.classes_.tolist()}")

    # -- Image-level train/test split --
    print("Splitting train/test at image level...")
    image_ids = np.arange(n_images)
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_frac,
                                random_state=args.seed)
    train_idx, test_idx = next(splitter.split(image_ids, groups=image_ids))
    train_idx = np.sort(train_idx)
    test_idx = np.sort(test_idx)
    print(f"  Train: {len(train_idx)} images ({len(train_idx)*256} patches)")
    print(f"  Test:  {len(test_idx)} images ({len(test_idx)*256} patches)")

    # -- Temporal filter mask --
    use_filtering = args.exclude_not_mapped or args.max_years_since is not None or args.min_recent_yr is not None
    patch_mask = None

    if args.max_years_since is not None or args.min_recent_yr is not None:
        # Need temporal data from npz
        if args.max_years_since is not None:
            years_since = load_temporal_field(Path(args.labels_npz), "disturb_years_since")
            patch_mask = ~np.isnan(years_since) & (years_since <= args.max_years_since)
            n_valid = patch_mask.sum()
            print(f"  Temporal filter: disturb_years_since <= {args.max_years_since} "
                  f"-> {n_valid:,} patches pass ({100*n_valid/len(patch_mask):.1f}%)")
        if args.min_recent_yr is not None:
            recent_yr = load_temporal_field(Path(args.labels_npz), "disturb_recent_yr")
            yr_mask = ~np.isnan(recent_yr) & (recent_yr >= args.min_recent_yr)
            if patch_mask is None:
                patch_mask = yr_mask
            else:
                patch_mask = patch_mask & yr_mask
            n_valid = patch_mask.sum()
            print(f"  Temporal filter: disturb_recent_yr >= {args.min_recent_yr} "
                  f"-> {n_valid:,} patches pass ({100*n_valid/len(patch_mask):.1f}%)")

    # -- Datasets & loaders --
    exclude_labels = set()
    if args.exclude_not_mapped:
        not_mapped_idx = int(np.where(le.classes_ == "Not Mapped")[0][0])
        exclude_labels.add(not_mapped_idx)
        kept_classes = [c for c in le.classes_ if c != "Not Mapped"]
        le_filtered = LabelEncoder()
        le_filtered.classes_ = np.array(kept_classes)
        n_classes = len(kept_classes)
        print(f"  Excluding 'Not Mapped' (encoded={not_mapped_idx}), "
              f"{n_classes} classes remain")

    if use_filtering:
        train_ds = FilteredPatchDataset(acts, encoded_labels, train_idx,
                                        exclude_labels, patch_mask)
        test_ds = FilteredPatchDataset(acts, encoded_labels, test_idx,
                                       exclude_labels, patch_mask)
        if args.exclude_not_mapped:
            # Remap labels: drop excluded class, compress to 0..n_classes-1
            remap = np.full(len(le.classes_), -1, dtype=np.int64)
            new_idx = 0
            for old_idx in range(len(le.classes_)):
                if old_idx not in exclude_labels:
                    remap[old_idx] = new_idx
                    new_idx += 1
            encoded_labels = remap[encoded_labels]
            train_ds.labels = encoded_labels
            test_ds.labels = encoded_labels
            le = le_filtered
        print(f"  Train patches: {len(train_ds)}, Test patches: {len(test_ds)}")
    else:
        train_ds = PatchDataset(acts, encoded_labels, train_idx)
        test_ds = PatchDataset(acts, encoded_labels, test_idx)

    # -- Compute class distribution for train and test --
    if use_filtering:
        train_patch_labels = train_ds.get_labels()
        test_patch_labels = test_ds.get_labels()
    else:
        train_patch_labels = np.concatenate(
            [encoded_labels[i * TOKENS_PER_IMAGE:(i + 1) * TOKENS_PER_IMAGE]
             for i in train_idx]
        )
        test_patch_labels = np.concatenate(
            [encoded_labels[i * TOKENS_PER_IMAGE:(i + 1) * TOKENS_PER_IMAGE]
             for i in test_idx]
        )

    train_class_counts = np.bincount(train_patch_labels, minlength=n_classes)
    test_class_counts = np.bincount(test_patch_labels, minlength=n_classes)
    class_names = le.classes_.tolist()

    print(f"  Class distribution (train / test):")
    for i, name in enumerate(class_names):
        print(f"    {name:<35} {train_class_counts[i]:>8,} / {test_class_counts[i]:>8,}")

    if args.balanced:
        sampler = BalancedBatchSampler(
            labels=train_patch_labels,
            batch_size=args.batch_size,
            n_classes=n_classes,
            seed=args.seed,
        )
        train_loader = DataLoader(train_ds, batch_sampler=sampler,
                                  num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=4, pin_memory=True)

    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    # -- Model --
    model = build_model(n_classes, hidden=args.hidden).to(device)
    print(f"Model: {model}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # -- Training loop --
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

    # -- Final evaluation --
    print("\nFinal evaluation on test set...")
    model.load_state_dict(torch.load(out_dir / "best_model.pt", weights_only=True))
    preds, labels = evaluate(model, test_loader, device)
    acc = accuracy_score(labels, preds)
    bal_acc = balanced_accuracy_score(labels, preds)
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  Balanced accuracy: {bal_acc:.4f}")

    report = classification_report(
        labels, preds, labels=list(range(n_classes)),
        target_names=le.classes_, zero_division=0
    )
    print("\nClassification Report:\n")
    print(report)

    # -- Save artifacts --
    with open(out_dir / "label_encoder_classes.json", "w") as f:
        json.dump(le.classes_.tolist(), f, indent=2)

    split_info = {
        "seed": args.seed,
        "test_frac": args.test_frac,
        "target": args.target,
        "n_classes": n_classes,
        "balanced": args.balanced,
        "exclude_not_mapped": args.exclude_not_mapped,
        "max_years_since": args.max_years_since,
        "min_recent_yr": args.min_recent_yr,
        "n_train_images": int(len(train_idx)),
        "n_test_images": int(len(test_idx)),
        "n_train_patches": len(train_ds),
        "n_test_patches": len(test_ds),
        "class_names": class_names,
        "train_class_counts": train_class_counts.tolist(),
        "test_class_counts": test_class_counts.tolist(),
        "train_image_indices": train_idx.tolist(),
        "test_image_indices": test_idx.tolist(),
    }
    with open(out_dir / "split_info.json", "w") as f:
        json.dump(split_info, f)

    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(f"Target: {args.target}\n")
        f.write(f"Balanced sampling: {args.balanced}\n")
        f.write(f"Exclude Not Mapped: {args.exclude_not_mapped}\n")
        if args.max_years_since is not None:
            f.write(f"Max years since: {args.max_years_since}\n")
        if args.min_recent_yr is not None:
            f.write(f"Min recent year: {args.min_recent_yr}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Balanced accuracy: {bal_acc:.4f}\n\n")
        f.write(f"--- Data Distribution ---\n")
        f.write(f"{'Class':<35} {'Train':>8} {'Test':>8} {'Train%':>7} {'Test%':>7}\n")
        f.write(f"{'-'*67}\n")
        for i, name in enumerate(class_names):
            tr_pct = 100 * train_class_counts[i] / max(train_class_counts.sum(), 1)
            te_pct = 100 * test_class_counts[i] / max(test_class_counts.sum(), 1)
            f.write(f"{name:<35} {train_class_counts[i]:>8,} {test_class_counts[i]:>8,} "
                    f"{tr_pct:>6.1f}% {te_pct:>6.1f}%\n")
        f.write(f"{'TOTAL':<35} {train_class_counts.sum():>8,} {test_class_counts.sum():>8,}\n\n")
        f.write(report)

    print(f"\nArtifacts saved to {out_dir}")


if __name__ == "__main__":
    main()
