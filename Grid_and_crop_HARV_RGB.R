###############################################################################
## Grid_and_Crop_HARV_RGB.R   (tile-major, memory-flat, DINOv3-aligned)
##
## Builds a 256 x 256 px crop grid over the HARV AOP, writes one GeoTIFF per
## crop from the NEON RGB camera mosaic (DP3.30010.001), and builds a nested
## 16 x 16 patch sub-grid (16 x 16 px patches) inside each crop.
##
## Why pixels, not metres:
##  * The crops feed a DINOv3 ViT-L/16 transformer. That model ingests a
##    256 px image and tiles it into 16 px x 16 px patches -> a 16 x 16 = 256
##    patch grid, one output token per patch. So the crop MUST be 256 px and
##    each sub-cell MUST be 16 px for the sub-grid to map 1:1 onto the patch
##    tokens. Defining the grid in metres (the old 25 m / 1.5625 m scheme) does
##    NOT line up: at the mosaic's 0.1 m/px, 25 m = 250 px and 1.5625 m =
##    15.625 px, so neither the crop nor the patches land on the model's grid.
##  * Sizes are set in PIXELS here and converted to metres per tile using the
##    tile's own resolution, so the script stays correct if the resolution
##    ever changes (no 0.1 m hard-coded).
##
## Efficiency triage (unchanged intent):
##  * Crops are aligned to EACH 1 km tile's own pixel origin, so every crop
##    lies wholly inside one tile. No mosaicking, no neighbour-tile problem;
##    each tile is processed independently. Peak memory ~ one tile + one crop.
##  * 1000 m / 25.6 m is not an integer (10000 px / 256 = 39.0625), so a 16 px
##    (1.6 m) remainder strip on each tile's east and south edge is left
##    uncropped. These thin seams are the YAGNI trade for keeping crops exactly
##    256 px AND inside a single tile. Full site coverage would need
##    cross-tile mosaicking, which is out of scope here.
##  * Grid and sub-grid are written INCREMENTALLY (append) so we never hold all
##    cells (or all sub-cells) in memory.
##
## Toggles below let you split the work if the wall clock is tight:
##   do_crops=TRUE,  do_subgrid=FALSE  -> fast pass, writes grid + all crops
##   do_crops=FALSE, do_subgrid=TRUE   -> second pass, writes only the sub-grid
##   grid_id is assigned deterministically, so IDs match across the two passes.
###############################################################################

library(sf)
library(terra)

## ---- 0. Config -------------------------------------------------------------
site     <- "HARV"
cell_px  <- 256L   # crop size in pixels  -> DINOv3 input image (256 x 256)
patch_px <- 16L    # patch size in pixels -> DINOv3 ViT-L/16 patch
n_sub    <- cell_px %/% patch_px   # 16 patches per side -> 16 x 16 = 256 tokens
stopifnot(cell_px %% patch_px == 0L)   # crop must be a whole number of patches

do_crops   <- FALSE   # write the 256 px RGB crops
do_subgrid <- TRUE  # write the 16 x 16 patch sub-grid (HEAVY: 256 x #cells)

data_dir <- "/fs/ess/PUOM0017/ForestScaling/DeepForest"   # inputs (read-only)
out_dir  <- "/fs/ess/PAS2136/SAtrEes"                     # all outputs

## ---- Output dirs (created up front; hard stop if not writable) -------------
shp_dir  <- file.path(out_dir, "Shapefiles")
crop_dir <- file.path(out_dir, "Imagery", "NEON", site, "RGB_256px_crops")
for (d in c(out_dir, shp_dir, crop_dir))
  dir.create(d, showWarnings = FALSE, recursive = TRUE)
stopifnot(dir.exists(shp_dir), dir.exists(crop_dir))

grid_out    <- file.path(shp_dir, paste0(site, "_grid_256px.shp"))
subgrid_out <- file.path(shp_dir, paste0(site, "_grid_256px_sub16.gpkg"))

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

## ---- Helper: full 256 px cells (in metres) that fall inside one tile -------
## Anchored at the tile's own top-left pixel; only whole cells are emitted, so
## every cell is exactly cell_px x cell_px and wholly inside this tile. Cells
## not overlapping the AOP are dropped.
tile_cells <- function(rt) {
  res_m       <- res(rt)[1]                 # pixels are square for this product
  cell_size_m <- cell_px * res_m
  ncx <- ncol(rt) %/% cell_px               # whole cells across / down
  ncy <- nrow(rt) %/% cell_px
  if (ncx == 0L || ncy == 0L) return(st_sfc(crs = crs_img))
  x0 <- xmin(rt); y1 <- ymax(rt)            # tile top-left origin
  bb <- st_bbox(c(xmin = x0,
                  xmax = x0 + ncx * cell_size_m,
                  ymin = y1 - ncy * cell_size_m,
                  ymax = y1), crs = crs_img)
  ## Use n = c(ncx, ncy), NOT cellsize: cell_size_m (e.g. 25.6 m) is not exactly
  ## representable, so cellsize would let ceil() add a spurious outer ring of
  ## cells that spill past the tile edge and break the crop. Fixing the count
  ## guarantees exactly ncx x ncy cells, all wholly inside the tile.
  g <- st_make_grid(st_as_sfc(bb), n = c(ncx, ncy), square = TRUE)
  g[lengths(st_intersects(g, AOP)) > 0]     # keep cells overlapping the AOP
}

## ---- Main tile-major loop --------------------------------------------------
gid        <- 0L      # running parent-cell id (deterministic across runs)
first_grid <- TRUE
first_sub  <- TRUE

for (f in tif_files) {
  rt  <- rast(f)                         # header only (pixels read on crop)
  e   <- ext(rt)
  xmn <- xmin(e); xmx <- xmax(e); ymn <- ymin(e); ymx <- ymax(e)

  # Skip tiles that do not touch the AOP.
  tb <- st_as_sfc(st_bbox(c(xmin = xmn, ymin = ymn, xmax = xmx, ymax = ymx),
                          crs = crs_img))
  if (length(st_intersects(tb, AOP)[[1]]) == 0) next

  cells <- tile_cells(rt)
  if (length(cells) == 0) next

  gsf <- st_sf(grid_id  = gid + seq_along(cells),
               tile     = basename(f),
               geometry = cells)
  gid <- gid + length(cells)

  # (a) append this tile's cells to the parent grid
  st_write(gsf, grid_out, append = !first_grid, quiet = TRUE)
  first_grid <- FALSE

  # (b) crop RGB to each cell. Extents are built from integer pixel offsets off
  #     the tile origin, so every crop is EXACTLY cell_px x cell_px.
  if (do_crops) {
    r     <- rast(f)
    res_m <- res(r)[1]
    x0    <- xmin(r); y1 <- ymax(r)
    for (k in seq_len(nrow(gsf))) {
      cb   <- st_bbox(gsf[k, ])
      col1 <- 1L + as.integer(round((cb[["xmin"]] - x0) / res_m))
      row1 <- 1L + as.integer(round((y1 - cb[["ymax"]]) / res_m))
      xw   <- x0 + (col1 - 1L) * res_m           # pixel-aligned window
      yw   <- y1 - (row1 - 1L) * res_m
      win  <- ext(xw, xw + cell_px * res_m, yw - cell_px * res_m, yw)
      crp  <- crop(r, win)
      if (nlyr(crp) > 3) crp <- crp[[1:3]]
      if (sum(as.numeric(global(crp, "notNA")[, 1])) == 0) next  # all nodata
      names(crp) <- c("Red", "Green", "Blue")
      fn <- file.path(crop_dir, sprintf("%s_RGB_256px_%d_%d_%d.tif",
             site, gsf$grid_id[k],
             as.integer(round(xw)), as.integer(round(yw - cell_px * res_m))))
      writeRaster(crp, fn, overwrite = TRUE, datatype = "INT1U",
                  gdal = c("COMPRESS=LZW", "TILED=YES", "INTERLEAVE=PIXEL"))
    }
    rm(r)
  }

  # (c) 16 x 16 patch sub-grid for this tile's cells (streamed, freed each tile)
  #     Each 256 px cell splits into n_sub x n_sub = 256 patches of 16 px.
  if (do_subgrid) {
    subs <- vector("list", nrow(gsf))
    for (k in seq_len(nrow(gsf))) {
      sc  <- st_make_grid(gsf[k, ], n = c(n_sub, n_sub), square = TRUE)
      ll  <- st_bbox(gsf[k, ])
      s   <- (ll[["xmax"]] - ll[["xmin"]]) / n_sub    # patch size in metres
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

