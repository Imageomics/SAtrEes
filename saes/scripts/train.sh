#!/usr/bin/env bash

uv run saes/launch.py train \
    --sweep ./saes/scripts/dino_sweep.py \
    --max-parallel 4 \
    cfg.sae.activation:batch-top-k