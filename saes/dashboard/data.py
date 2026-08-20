"""Shared data layer for the SAE feature dashboard.

Handles run discovery, reconstruction of the ordered image paths that were used
to build a shard set, and the per-feature top-activation index built from the
inference artifact `token_acts.npz`.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib

import numpy as np
import scipy.sparse
import torch

from saev import disk
from saev.data.shards import Metadata

IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)
TOKEN_ACTS = "token_acts.npz"
INDEX_NAME = "feature_index.npz"
INDEX_VERSION = 2
DEFAULT_RUNS_ROOT = pathlib.Path("/fs/ess/PAS2136/SAtrEes/saev/runs")


@dataclasses.dataclass(frozen=True)
class RunInfo:
    """A run that has a complete inference artifact set on disk."""

    run: disk.Run
    shards_dir: pathlib.Path
    md: Metadata
    inference_dir: pathlib.Path


def get_run_infos(runs_root: pathlib.Path) -> list[RunInfo]:
    """Discover runs under ``runs_root`` with a completed inference artifact set.

    A run counts as complete only if ``token_acts.npz`` exists for its shard set.
    """
    infos = []
    for d in sorted(runs_root.iterdir()):
        if not d.is_dir():
            continue
        try:
            run = disk.Run(d)
        except (ValueError, FileNotFoundError):
            continue
        shards_dir = run.train_shards
        md = Metadata.load(shards_dir)
        inference_dir = run.inference / md.hash
        if (inference_dir / TOKEN_ACTS).exists():
            infos.append(RunInfo(run, shards_dir, md, inference_dir))
    return infos


def get_run_info(runs_root: pathlib.Path, run_id: str) -> RunInfo | None:
    """Look up a single run by id, or ``None`` if missing/incomplete."""
    for info in get_run_infos(runs_root):
        if info.run.run_id == run_id:
            return info
    return None


def image_paths(md: Metadata, shards_dir: pathlib.Path) -> tuple[pathlib.Path, ...]:
    """Reconstruct the ordered image paths used to build a shard set.

    The shard-time dataset is reproduced by keeping only files that already
    existed when the shards were written (files are ordered by class, then name,
    exactly like torchvision's ``ImageFolder``). The shard timestamp is taken
    from ``metadata.json`` in the shards directory.
    """
    root = md.make_data_cfg().root
    cutoff = (shards_dir / "metadata.json").stat().st_mtime

    paths: list[pathlib.Path] = []
    for cls in sorted(e.name for e in os.scandir(root) if e.is_dir()):
        files = [
            f
            for f in (root / cls).iterdir()
            if f.is_file()
            and f.suffix.lower() in IMG_EXTENSIONS
            and f.stat().st_mtime <= cutoff
        ]
        paths.extend(sorted(files))

    if len(paths) != md.n_examples:
        # Fall back to ImageFolder ordering over the current directory contents.
        paths = []
        for cls in sorted(e.name for e in os.scandir(root) if e.is_dir()):
            files = [
                f
                for f in (root / cls).iterdir()
                if f.is_file() and f.suffix.lower() in IMG_EXTENSIONS
            ]
            paths.extend(sorted(files))
        if len(paths) < md.n_examples:
            raise RuntimeError(
                f"Found {len(paths)} images but shards expect {md.n_examples} "
                f"under {root}. The dataset may have changed since sharding."
            )

    return tuple(paths[: md.n_examples])


def build_index(info: RunInfo, top_k: int = 256, *, force: bool = False) -> pathlib.Path:
    """Build (or return an existing) per-feature top-activation index.

    The index stores, for every SAE feature, the ``top_k`` globally
    highest-activating tokens (as row indices into ``token_acts.npz``) and their
    activation values. Row ``r`` of ``token_acts.npz`` corresponds to content
    token ``r % content_tokens_per_example`` of example ``r // content_tokens_per_example``.
    """
    out_path = info.inference_dir / INDEX_NAME
    if out_path.exists() and not force:
        try:
            index = load_index(out_path)
            if int(index["index_version"]) >= INDEX_VERSION:
                return out_path
        except Exception:
            pass

    acts = scipy.sparse.load_npz(info.inference_dir / TOKEN_ACTS)
    # A CSR stores tokens as rows and features as columns. Convert to CSC in
    # place (rows=tokens, columns=features) so each feature's column is
    # addressable via indptr. Do not use `acts.T`: for csr_array that returns a
    # lazy view whose raw .indptr/.indices/.data are not actually transposed.
    csc = acts.tocsc()
    d_sae = csc.shape[1]
    assert csc.indptr.size == d_sae + 1

    token_ids = np.zeros((d_sae, top_k), dtype=np.int64)
    token_vals = np.zeros((d_sae, top_k), dtype=np.float32)
    for f in range(d_sae):
        sl = slice(csc.indptr[f], csc.indptr[f + 1])
        data = csc.data[sl]
        n = len(data)
        if n == 0:
            continue
        k = min(top_k, n)
        if k < n:
            pick = np.argpartition(data, -k)[-k:]
        else:
            pick = np.arange(n)
        order = pick[np.argsort(-data[pick])]
        token_ids[f, :k] = csc.indices[sl][order]
        token_vals[f, :k] = data[order]

    index = {
        "token_ids": token_ids,
        "token_vals": token_vals,
        "sparsity": torch.load(
            info.inference_dir / "sparsity.pt", weights_only=True
        ).float().numpy(),
        "mean_values": torch.load(
            info.inference_dir / "mean_values.pt", weights_only=True
        ).float().numpy(),
        "top_k": top_k,
        "content_tokens_per_example": info.md.content_tokens_per_example,
        "n_examples": info.md.n_examples,
        "d_sae": d_sae,
        "run": info.run.run_id,
        "shards_hash": info.md.hash,
        "index_version": INDEX_VERSION,
    }
    np.savez_compressed(
        out_path,
        **{k: np.asarray(v) for k, v in index.items()},
    )
    return out_path


def load_index(index_path: pathlib.Path) -> dict:
    """Load a prebuilt feature index from disk."""
    with np.load(index_path) as z:
        return {
            "token_ids": z["token_ids"],
            "token_vals": z["token_vals"],
            "sparsity": z["sparsity"],
            "mean_values": z["mean_values"],
            "top_k": int(z["top_k"]),
            "content_tokens_per_example": int(z["content_tokens_per_example"]),
            "n_examples": int(z["n_examples"]),
            "d_sae": int(z["d_sae"]),
            "run": str(z["run"]),
            "shards_hash": str(z["shards_hash"]),
            "index_version": int(z["index_version"]),
        }


def top_images(
    index: dict, feature: int, n: int
) -> list[tuple[int, float, np.ndarray]]:
    """Return the ``n`` images with the highest activation for ``feature``.

    Each entry is ``(example_idx, max_activation, patches)`` where ``patches``
    holds the feature's activation per content token (patch) for that image.
    """
    ids = index["token_ids"][feature]
    vals = index["token_vals"][feature]
    ctp = index["content_tokens_per_example"]

    keep = vals > 0
    ids, vals = ids[keep], vals[keep]
    if len(vals) == 0:
        return []

    examples = ids // ctp
    tokens = ids % ctp

    seen: set[int] = set()
    selected: list[tuple[int, float]] = []
    for ex, v in zip(examples.tolist(), vals.tolist()):
        if ex not in seen:
            seen.add(ex)
            selected.append((ex, float(v)))
            if len(selected) >= n:
                break

    results = []
    for ex, mx in selected:
        patch_mask = examples == ex
        patches = np.zeros(ctp, dtype=np.float32)
        patches[tokens[patch_mask]] = vals[patch_mask]
        results.append((ex, mx, patches))
    return results