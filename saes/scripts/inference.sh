#!/usr/bin/env bash

uv run saes/launch.py features \
    --sweep ./saes/scripts/dino_inference_sweep.py
