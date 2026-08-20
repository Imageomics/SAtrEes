# SAtrEes

Extract land-use disturbance history for NEON Harvard Forest (HARV) tree crowns from RGB airborne imagery, gridded at sub-meter resolution.

The pipeline builds a fine-grained grid over the HARV Airborne Observation Platform (AOP) footprint, crops NEON RGB mosaic imagery to it, and labels every grid cell with its disturbance history (agricultural abandonment, natural disturbance, the 1938 hurricane, and silviculture treatments) drawn from the Harvard Forest land-use history GIS archive.

## Repository structure

```
SAtrEes/
├── Grid_and_crop_HARV_RGB.R          # build the 25m/16x16 sub-grid and crop RGB tiles to it
├── Grid_and_crop_HARV_RGB.sh         # SLURM submission script for the above (OSC)
├── scripts/
│   ├── Label_grid_disturbance_history.R      # label every grid cell with disturbance history
│   └── HARV_grid_25m_disturbance_metadata.csv # field-by-field schema for the labeled grid
└── data/
    ├── hf110_land_use_history/       # source GIS archive (Harvard Forest Data Archive: HF110)
    └── *.csv                         # small derived summary tables (see below)
```

Large geospatial outputs (`.gpkg`/`.shp`/`.kml` grids and disturbance layers) and the full attribute-table export are generated locally by the scripts above and are not tracked in this repository — see [Reproducing the outputs](#reproducing-the-outputs).

## User guide

### 1. Build the grid

`Grid_and_crop_HARV_RGB.R` builds a 25 m grid aligned to the NEON 1 km RGB mosaic tile boundaries (DP3.30010.001), crops the mosaic to each 25 m cell, and nests a 16×16 sub-grid (~1.56 m sub-cells) inside each parent cell. It's designed to run tile-at-a-time to stay memory-flat; `do_crops`/`do_subgrid` toggles let a run be split into a fast grid-only pass and a heavier sub-grid pass. Submit via `Grid_and_crop_HARV_RGB.sh` (SLURM/OSC).

### 2. Label disturbance history

`scripts/Label_grid_disturbance_history.R` intersects the grid against four source layers in `data/hf110_land_use_history/hf110-01-gis.zip`:

| Layer | Covers |
|---|---|
| `ph_ag_abandonment` | field-abandonment + cutting years |
| `ph_natural_disturbance2` | ice/wind/tornado/fire/snow events |
| `1938_hurricane_damage` | 1938 hurricane damage class (fixed year) |
| `silviculture_treatments_08_21_2010` | dated management-treatment log |

Each grid cell is labeled at the sub-cell level (one label per ~1.56 m sub-cell, not pooled across a parent cell) with:

- **`disturb_majority`** — the dominant source layer by area, labeled with its most recent dated event
- **`disturb_recent_class`** — the class of whichever dated event (across all 4 layers) is most recent, using a 9-class taxonomy (see `scripts/HARV_grid_25m_disturbance_metadata.csv` for the full field schema and valid values)
- **`disturb_history`** — every dated event on record for the cell, oldest → newest

Output is written as a labeled grid (`HARV_grid_25m_disturbance` / `HARV_grid_256px_sub16_disturbance`) and a combined source-history layer (`HARV_disturbance_history_combined`, one row per original source polygon), each in `.gpkg` (full fidelity), `.kml` (Google Earth), and `.shp` (lat/lon reprojected, for Google Earth Pro import).

### Input data

Source GIS layers come from the [Harvard Forest Data Archive dataset HF110](https://harvardforest.fas.harvard.edu/harvard-forest-data-archive), Land-use History (Harvard Forest, 1830–2010). The archive is stored locally at `data/hf110_land_use_history/hf110-01-gis.zip` and is not modified by the pipeline.

## Reproducing the outputs

Both scripts require R with the `sf` and `terra` packages (`Label_grid_disturbance_history.R` also uses `dplyr`); the `Grid_and_crop_HARV_RGB.sh` SLURM script loads GDAL/PROJ modules on OSC. Outputs are large (the full 16×16 sub-grid disturbance layer is several GB) and are generated to the repository root / `data/` locally — they are gitignored rather than committed.

## License

[MIT](LICENSE)
