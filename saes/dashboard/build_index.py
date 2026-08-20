"""Prebuild the per-feature activation index for SAE runs.

The Streamlit app builds this index lazily on first use, but running it here
upfront for every run means the dashboard opens instantly later.

Usage:
    uv run python saes/dashboard/build_index.py
    uv run python saes/dashboard/build_index.py --run agw9clvn --top-k 512 --force
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from saes.dashboard.data import (
    DEFAULT_RUNS_ROOT,
    get_run_info,
    get_run_infos,
    build_index,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--runs-root", type=pathlib.Path, default=DEFAULT_RUNS_ROOT,
        help="Parent directory of SAE runs (default: %(default)s).",
    )
    ap.add_argument(
        "--run", default=None,
        help="Build only this run id; defaults to all complete runs.",
    )
    ap.add_argument(
        "--top-k", type=int, default=256,
        help="Number of top tokens to keep per feature (default: %(default)s).",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Rebuild indexes even if they already exist.",
    )
    args = ap.parse_args()

    infos = (
        [get_run_info(args.runs_root, args.run)]
        if args.run
        else get_run_infos(args.runs_root)
    )
    infos = [i for i in infos if i is not None]
    if not infos:
        raise SystemExit("No complete runs found to index.")

    for info in infos:
        path = build_index(info, top_k=args.top_k, force=args.force)
        print(f"{info.run.run_id}: wrote {path}")


if __name__ == "__main__":
    main()