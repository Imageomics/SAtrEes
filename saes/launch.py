"""CLI interface for SAE workflows backed by the saev package.

Exposes three subcommands:

- `shards`: generate activation shards from a dataset.
- `train`: train SAEs over sharded activations.
- `features`: generate sparse features for a dataset from a trained SAE.

Each subcommand forwards to the corresponding saev.framework entrypoint.
"""

import dataclasses
import pathlib
import typing as tp

import tyro

import saev.framework.inference
import saev.framework.shards
import saev.framework.train


@dataclasses.dataclass(frozen=True)
class Shards:
    """Generate activation shards from a dataset."""

    cfg: saev.framework.shards.Config

    def __main__(self) -> None:
        saev.framework.shards.cli(self.cfg)


@dataclasses.dataclass(frozen=True)
class Train:
    """Train SAEs on sharded activations."""

    cfg: saev.framework.train.Config
    sweep: pathlib.Path | None = None
    """Path to a .py file defining a hyperparameter sweep."""
    max_parallel: int | None = None
    """Maximum SAEs to train concurrently within a single worker."""

    def __main__(self) -> None:
        saev.framework.train.main(self.cfg, self.sweep, self.max_parallel)


@dataclasses.dataclass(frozen=True)
class Features:
    """Generate sparse features for a dataset from a trained SAE."""

    cfg: saev.framework.inference.Config
    sweep: pathlib.Path | None = None
    """Path to a .py file defining a hyperparameter sweep."""

    def __main__(self) -> None:
        saev.framework.inference.main(self.cfg, self.sweep)


def main() -> None:
    tyro.cli(tp.Union[Shards, Train, Features]).__main__()


if __name__ == "__main__":
    main()