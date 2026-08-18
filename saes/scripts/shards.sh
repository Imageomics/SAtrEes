#!/usr/bin/env bash

uv run saes/launch.py shards \
    --cfg.shards-root /fs/ess/PAS2136/SAtrEes/saev/shards/ \
    --cfg.family dinov3 \
    --cfg.ckpt /fs/scratch/PAS2136/jbeattie/models/dino/dinov3_vitl16_pretrain_sat493m-noclslocal.pth \
    --cfg.n-workers 1 \
    --cfg.d-model 1024 \
    --cfg.layers -2 \
    --cfg.content-tokens-per-example 256 \
    --cfg.slurm-acct PAS2136 \
    --cfg.slurm-partition nextgen \
    --cfg.n-hours 2 \
    cfg.data:img-folder \
    --cfg.data.root /fs/ess/PAS2136/SAtrEes/Imagery/NEON/HARV/
