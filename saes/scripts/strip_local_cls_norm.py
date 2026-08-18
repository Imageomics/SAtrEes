"""Drop local_cls_norm weights from a DINOv3 SAT checkpoint.

The SAT493M checkpoints contain local_cls_norm.weight/bias, which the saev
Encoder does not implement. These weights only affect the final CLS/patch
outputs, not the block residuals that saev records, so dropping them is safe.
"""

import argparse
import pathlib

import torch

_KEYS = ("local_cls_norm.weight", "local_cls_norm.bias")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=pathlib.Path, help="Path to the original checkpoint.")
    parser.add_argument("dst", type=pathlib.Path, help="Path to write the stripped checkpoint.")
    args = parser.parse_args()

    state_dict = torch.load(args.src, mmap=True, weights_only=True, map_location="cpu")
    for key in _KEYS:
        state_dict.pop(key, None)
    torch.save(state_dict, args.dst)
    print(f"Wrote {args.dst}")


if __name__ == "__main__":
    main()