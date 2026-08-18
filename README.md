# SAtrEes
Extract land-use history from RGB images of tree crowns.

# HARV RGB 25 m Gridding & Cropping

Crops the NEON RGB camera mosaic (AOP product [DP3.30010.001](https://data.neonscience.org/data-products/DP3.30010.001)) at the HARV site to a 25 m grid, saving one GeoTIFF per cell, and writes a nested 16 × 16 sub-grid (1.5625 m sub-cells) as a vector file. One preprocessing step in a larger forest-scaling project.

## Requirements

- R (tested with 4.4.0) with `sf` and `terra`
- uv (tested with v0.12.5) for python package management

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
