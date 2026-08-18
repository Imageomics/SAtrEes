# SAtrEes
Extract land-use history from RGB images of tree crowns.

# HARV RGB 25 m Gridding & Cropping

Crops the NEON RGB camera mosaic (AOP product [DP3.30010.001](https://data.neonscience.org/data-products/DP3.30010.001)) at the HARV site to a 25 m grid, saving one GeoTIFF per cell, and writes a nested 16 × 16 sub-grid (1.5625 m sub-cells) as a vector file. One preprocessing step in a larger forest-scaling project.

## Requirements

- R (tested with 4.4.0) with `sf` and `terra`
- uv (tested with v0.12.5) for python package management
- DINOv3 SAT493M version (tested with ViT-L/16). Request access [here](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)

## Python Environment Setup
[`uv`](https://docs.astral.sh/uv/) is used for python environment management. Install `uv` using the following command:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, set up the python environment using `uv sync`

## Usage

```bash
Rscript Grid_and_Crop_HARV_RGB.R
```

Paths, `site`, `cell_size` (25 m), and `n_sub` (16) are set at the top of the script. Two toggles control the work:

- `do_crops` — write the 25 m RGB crops
- `do_subgrid` — write the 16 × 16 sub-grid vector

The sub-grid vector is large (~256 × number of cells). If the time usage is tight, run once with `do_subgrid = FALSE`, then again with `do_crops = FALSE`; `grid_id` is deterministic, so IDs match across passes.

## Inputs

- `HARV_Tiles_NorthBoundary.shp` — AOP tiles for the northern boundary (defines the grid extent)
- NEON RGB mosaic tiles (1 km mosaic tiles from 2022 at HARV)

## Outputs

Written to `out_dir` (set in the script):

- `Shapefiles/HARV_grid_25m.shp` — 25 m grid
- `Shapefiles/HARV_grid_25m_sub16.gpkg` — nested 16 × 16 sub-grid
- `Imagery/NEON/HARV/RGB_25m_crops/*.tif` — one RGB crop per 25 m cell

# SAE Training

To automatically learn features corresponding to disturbance types and timelines, we train SAEs on the downloaded and cropped image data using the `saev` libary, which is included as a dependency of this project.


## Preparing Foundation Model

For this project we use DINOv3 SAT493M ViT-L/16 (Request access [here](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)). This model is not directly compatible with the current version of the saev library, though. To address this, certain unused portions of the model need to be dropped. This can be easily done using the provided `saes/scripts/strip_local_cls_norm.py` script. Use `saes/scripts/strip_local_cls_norm.py --help` for more info.

## Generating Foundation Model Activations

First, to generate the training dataset for our SAEs, you must generate vision model activations and save them to disk in shard files. For this work, we use DINOv3-SAT493M ViT-L/16. Download this model to your cluster and modify the `--cfg.ckpt` parameter in `scripts/shards.sh` to point to the location of this model. Additionally, modify `--cfg.shards-root` to point to the location you'd like to save the vision model activations to, as well as all parameters regarding slurm project/partition as needed. Finally, generate vision model activations:

```bash
bash saes/scripts/shards.sh
```


