###############################################################################
## Grid_and_Crop_HARV_RGB.R   (tile-major, memory-flat)
##
## Builds a 25 m grid over the HARV AOP, crops the NEON RGB camera mosaic
## (DP3.30010.001) to each cell as an individual GeoTIFF, and builds a nested
## 16 x 16 sub-grid (1.5625 m sub-cells) inside each 25 m cell.
##
## Efficiency triage (vs. the version that OOM'd):
##  * The 25 m grid is ALIGNED to the 1 km NEON tile boundaries. Because
##    1000 / 25 = 40 exactly, every 25 m cell lies wholly inside ONE tile.
##    So there is no mosaicking and no "neighbour tile" problem: each tile is
##    processed independently with its own cells.
##  * Tiles are opened one at a time (header first, pixels only when a 25 m
##    window is cropped). Peak memory is ~one tile + one small crop.
##  * The grid and sub-grid are written INCREMENTALLY (append), so we never
##    hold all cells (or all ~128 M sub-cells) in memory. That is what killed
##    the previous run.
##
## Toggles below let you split the work if the 5 h wall clock is tight:
##   do_crops=TRUE,  do_subgrid=FALSE  -> fast pass, writes grid + all crops
##   do_crops=FALSE, do_subgrid=TRUE   -> second pass, writes only the sub-grid
##   grid_id is assigned deterministically, so IDs match across the two passes.
###############################################################################

library(sf)
library(terra)

## ---- 0. Config -------------------------------------------------------------
site      <- "HARV"
cell_size <- 25    # m, parent grid
n_sub     <- 16    # 16 x 16 sub-cells -> cell_size/n_sub = 1.5625 m each

do_crops   <- FALSE  # write the 25 m RGB crops
do_subgrid <- TRUE   # write the 16x16 sub-grid (HEAVY: 256 x #cells features)

data_dir <- "/fs/ess/PUOM0017/ForestScaling/DeepForest"   # inputs (read-only)
out_dir  <- "/fs/ess/PAS2136/SAtrEes"                     # all outputs

## ---- Output dirs (created up front; hard stop if not writable) -------------
shp_dir  <- file.path(out_dir, "Shapefiles")
crop_dir <- file.path(out_dir, "Imagery", "NEON", site, "RGB_25m_crops")
for (d in c(out_dir, shp_dir, crop_dir))
  dir.create(d, showWarnings = FALSE, recursive = TRUE)
stopifnot(dir.exists(shp_dir), dir.exists(crop_dir))

grid_out    <- file.path(shp_dir, paste0(site, "_grid_25m.shp"))
subgrid_out <- file.path(shp_dir, paste0(site, "_grid_25m_sub16.gpkg"))

## ---- Inputs ----------------------------------------------------------------
aop_path <- file.path(out_dir, "Shapefiles", paste0(site, "_Tiles_NorthBoundary.shp"))

mosaic_dir <- Sys.glob(file.path(
  data_dir, "Imagery/NEON/DP3.30010.001/neon-aop-products/2022/FullSite/D01",
  paste0("2022_", site, "_*"), "L3/Camera/Mosaic"))[1]
stopifnot(!is.na(mosaic_dir), dir.exists(mosaic_dir))

tif_files <- sort(list.files(mosaic_dir, pattern = "_image\\.tif$",
                             full.names = TRUE))
stopifnot(length(tif_files) > 0)
cat("Found", length(tif_files), "RGB tiles\n")

## AOP in the imagery CRS (CRS read from a tile header - cheap). Union to a
## single geometry so the per-tile st_intersects stays fast.
crs_img <- crs(rast(tif_files[1]))
AOP <- st_union(st_transform(st_read(aop_path, quiet = TRUE), crs_img))

## ---- Helper: the aligned 25 m cells that fall inside one tile --------------
tile_cells <- function(xmn, xmx, ymn, ymx) {
  # snap the tile extent to the 25 m lattice (guards against float fuzz)
  xmin <- round(xmn / cell_size) * cell_size
  xmax <- round(xmx / cell_size) * cell_size
  ymin <- round(ymn / cell_size) * cell_size
  ymax <- round(ymx / cell_size) * cell_size
  bb <- st_bbox(c(xmin = xmin, ymin = ymin, xmax = xmax, ymax = ymax),
                crs = crs_img)
  g <- st_make_grid(st_as_sfc(bb), cellsize = cell_size, square = TRUE)
  g[lengths(st_intersects(g, AOP)) > 0]   # keep cells overlapping the AOP
}

## ---- Main tile-major loop --------------------------------------------------
gid        <- 0L      # running parent-cell id (deterministic across runs)
first_grid <- TRUE
first_sub  <- TRUE

for (f in tif_files) {
  e   <- ext(rast(f))                    # header only
  xmn <- xmin(e); xmx <- xmax(e); ymn <- ymin(e); ymx <- ymax(e)  # unnamed

  # Skip tiles that do not touch the AOP.
  tb <- st_as_sfc(st_bbox(c(xmin = xmn, ymin = ymn, xmax = xmx, ymax = ymx),
                          crs = crs_img))
  if (length(st_intersects(tb, AOP)[[1]]) == 0) next

  cells <- tile_cells(xmn, xmx, ymn, ymx)
  if (length(cells) == 0) next

  gsf <- st_sf(grid_id  = gid + seq_along(cells),
               tile     = basename(f),
               geometry = cells)
  gid <- gid + length(cells)

  # (a) append this tile's cells to the parent grid
  st_write(gsf, grid_out, append = !first_grid, quiet = TRUE)
  first_grid <- FALSE

  # (b) crop RGB to each cell (each cell is wholly inside this tile)
  if (do_crops) {
    r <- rast(f)
    for (k in seq_len(nrow(gsf))) {
      crp <- crop(r, vect(gsf[k, ]))
      if (nlyr(crp) > 3) crp <- crp[[1:3]]
      if (sum(as.numeric(global(crp, "notNA")[, 1])) == 0) next  # all nodata
      names(crp) <- c("Red", "Green", "Blue")
      cb <- st_bbox(gsf[k, ])
      fn <- file.path(crop_dir, sprintf("%s_RGB_25m_%d_%d_%d.tif",
             site, gsf$grid_id[k],
             as.integer(round(cb[["xmin"]])), as.integer(round(cb[["ymin"]]))))
      writeRaster(crp, fn, overwrite = TRUE, datatype = "INT1U",
                  gdal = c("COMPRESS=LZW", "TILED=YES", "INTERLEAVE=PIXEL"))
    }
    rm(r)
  }

  # (c) 16 x 16 sub-grid for this tile's cells (streamed, freed each tile)
  if (do_subgrid) {
    subs <- vector("list", nrow(gsf))
    s <- cell_size / n_sub
    for (k in seq_len(nrow(gsf))) {
      sc  <- st_make_grid(gsf[k, ], n = c(n_sub, n_sub), square = TRUE)
      ll  <- st_bbox(gsf[k, ])
      ctr <- st_coordinates(st_centroid(sc))
      subs[[k]] <- st_sf(
        grid_id  = gsf$grid_id[k],
        sub_col  = as.integer(floor((ctr[, 1] - ll[["xmin"]]) / s)) + 1L,
        sub_row  = as.integer(floor((ctr[, 2] - ll[["ymin"]]) / s)) + 1L,
        geometry = sc)
    }
    st_write(do.call(rbind, subs), subgrid_out,
             append = !first_sub, quiet = TRUE)
    first_sub <- FALSE
    rm(subs); gc()
  }

  cat("tile", basename(f), "| cells:", nrow(gsf),
      "| cumulative:", gid, "\n")
}

cat("Done. Grid ->", grid_out, "\n")
if (do_crops)   cat("Crops ->", crop_dir, "\n")
if (do_subgrid) cat("Sub-grid ->", subgrid_out, "\n")

