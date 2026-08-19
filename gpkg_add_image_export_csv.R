###############################################################################
## gpkg_add_image_latlon_export_csv.R
##
## Companion to Grid_and_crop_HARV_RGB.R (Imageomics/SAtrEes).
##
## Reads the 16 x 16 sub-grid GeoPackage (here the disturbance version with the
## user's extra columns), keeps EVERY existing attribute unchanged, and appends:
##
##   1. image_file   - the 25 m RGB crop each sub-cell belongs to. Reconstructs
##                     the exact filename written by Grid_and_crop_HARV_RGB.R:
##                     "<site>_RGB_25m_<grid_id>_<xmin>_<ymin>.tif"
##                     (xmin/ymin = the parent 25 m cell's lower-left corner).
##   2. cell_order   - position of the sub-cell WITHIN its image, 1..(n_sub^2),
##                     in raster reading order: row-major from the TOP-LEFT
##                     (north-west) corner. Also writes image_row / image_col.
##   3. centroid_lon / centroid_lat - centroid of each sub-cell in WGS84
##                     (EPSG:4326). Native-CRS centroid_x / centroid_y kept too.
##
## Then exports a CSV containing all original columns + these new ones, and
## auto-writes a column dictionary skeleton (data_dictionary_auto.csv).
##
## Nothing in the source gpkg is modified in place; outputs are new files.
###############################################################################

suppressMessages({
  library(sf)
})

## ---- 0. Config -------------------------------------------------------------
## Project home. Everything below is built from this so you only edit one path.
home_dir  <- "/fs/ess/PAS2136/SAtrEes"
shp_dir   <- file.path(home_dir, "Shapefiles")   # note: pipeline used "Shapefiles"
                                                 #  -> match your actual case here.

in_gpkg   <- file.path(shp_dir, "HARV_grid_25m_sub16_disturbance.gpkg")  # input
out_csv   <- file.path(home_dir, "HARV_grid_25m_sub16_disturbance.csv")   # output CSV
out_dict  <- file.path(home_dir, "data_dictionary_auto.csv")             # auto dict

site      <- "HARV"   # filename prefix used by the crop writer
cell_size <- 25       # m, parent grid cell size (must match the pipeline)
n_sub     <- 16       # sub-cells per side (16 -> 256 sub-cells per cell)
gid_col   <- "grid_id"  # parent-cell id column name in the gpkg

## OPTIONAL extras (set to NA / FALSE to skip) --------------------------------
## Parent 25 m grid: its "tile" column (source NEON mosaic tile) is joined on grid_id.
parent_grid <- file.path(shp_dir, "HARV_grid_25m.shp")
## Crops folder: flag whether each expected crop actually exists on disk
## (all-nodata cells were skipped by the crop writer, so some won't).
crop_dir    <- file.path(home_dir, "Imagery", "NEON", site, "RGB_25m_crops")
include_wkt <- FALSE       # TRUE -> add a geometry_wkt column to the CSV

## ---- 1. Read ---------------------------------------------------------------
stopifnot(file.exists(in_gpkg))
g <- st_read(in_gpkg, quiet = TRUE)
stopifnot(gid_col %in% names(g))
orig_cols <- setdiff(names(g), attr(g, "sf_column"))   # everything but geometry
src_crs   <- st_crs(g)
cat("Read", nrow(g), "sub-cells,", length(orig_cols), "attribute columns.\n")
cat("Source CRS:", src_crs$input, "\n")

s <- cell_size / n_sub                                 # sub-cell size (m)

## ---- 2. Sub-cell centroids (native CRS) ------------------------------------
## suppress the "assumes planar" note; the source CRS is projected (UTM), so a
## planar centroid is exactly what we want.
cen  <- suppressWarnings(st_centroid(st_geometry(g)))
xy   <- st_coordinates(cen)                            # matrix: X (easting), Y (northing)
cx   <- xy[, 1]; cy <- xy[, 2]

## ---- 3. Parent 25 m cell lower-left corner, per grid_id --------------------
## Reconstructed by grouping the sub-cells: the parent xmin/ymin is the min of
## the sub-cell centroids minus half a sub-cell. This does NOT assume any
## particular lattice alignment, so it stays correct for any grid.
gidv  <- g[[gid_col]]
pxmin <- ave(cx, gidv, FUN = min) - s / 2
pymin <- ave(cy, gidv, FUN = min) - s / 2

## integer corner used in the crop filename (crop writer used as.integer(round()))
ixmin <- as.integer(round(pxmin))
iymin <- as.integer(round(pymin))

## ---- 4. (1) image_file -----------------------------------------------------
image_file <- sprintf("%s_RGB_25m_%s_%d_%d.tif", site, as.character(gidv), ixmin, iymin)

## ---- 5. (2) order of the sub-cell within its image -------------------------
## Column index west->east (1..n_sub) and geographic row south->north (1..n_sub).
image_col <- as.integer(floor((cx - pxmin) / s)) + 1L
geo_row   <- as.integer(floor((cy - pymin) / s)) + 1L
## Flip to raster convention: image_row 1 = TOP (north). cell_order is row-major
## from the top-left corner, matching how pixels are laid out in the crop.
image_row  <- n_sub + 1L - geo_row
cell_order <- (image_row - 1L) * n_sub + image_col

## clamp guards against any float edge case landing on 0 or n_sub+1
image_col  <- pmin(pmax(image_col,  1L), n_sub)
image_row  <- pmin(pmax(image_row,  1L), n_sub)
cell_order <- pmin(pmax(cell_order, 1L), n_sub * n_sub)

## ---- 6. (3) centroid lon/lat (WGS84) ---------------------------------------
cen_ll <- st_transform(st_sfc(cen, crs = src_crs), 4326)
ll     <- st_coordinates(cen_ll)

new_cols <- data.frame(
  image_file   = image_file,
  image_col    = image_col,
  image_row    = image_row,
  cell_order   = cell_order,
  centroid_x   = round(cx, 3),           # native CRS easting  (m)
  centroid_y   = round(cy, 3),           # native CRS northing (m)
  centroid_lon = round(ll[, 1], 8),      # WGS84 longitude
  centroid_lat = round(ll[, 2], 8),      # WGS84 latitude
  stringsAsFactors = FALSE
)

## ---- 7. Optional joins / checks --------------------------------------------
if (!is.na(parent_grid) && file.exists(parent_grid)) {
  pg <- st_drop_geometry(st_read(parent_grid, quiet = TRUE))
  if (all(c(gid_col, "tile") %in% names(pg))) {
    tile_lookup <- pg[!duplicated(pg[[gid_col]]), c(gid_col, "tile")]
    new_cols$tile <- tile_lookup$tile[match(gidv, tile_lookup[[gid_col]])]
    cat("Joined 'tile' from parent grid.\n")
  }
}

if (!is.na(crop_dir) && dir.exists(crop_dir)) {
  new_cols$image_exists <- file.exists(file.path(crop_dir, image_file))
  cat("Checked crop existence:", sum(new_cols$image_exists), "of",
      nrow(new_cols), "expected crops found.\n")
}

## ---- 8. Assemble & write CSV ----------------------------------------------
out <- cbind(st_drop_geometry(g)[orig_cols], new_cols)
if (include_wkt) out$geometry_wkt <- st_as_text(st_geometry(g))

write.csv(out, out_csv, row.names = FALSE)
cat("Wrote", out_csv, "with", nrow(out), "rows and", ncol(out), "columns.\n")

## ---- 9. Auto column-dictionary skeleton ------------------------------------
## Known descriptions; anything not listed (your added disturbance columns)
## gets a blank description to fill in.
desc <- c(
  grid_id      = "ID of the parent 25 m cell (deterministic; matches the crop filename and the 25 m grid).",
  sub_col      = "Sub-cell column index within the parent cell, 1..n_sub, west->east (geographic).",
  sub_row      = "Sub-cell row index within the parent cell, 1..n_sub, south->north (geographic).",
  image_file   = "RGB 25 m crop this sub-cell belongs to: <site>_RGB_25m_<grid_id>_<xmin>_<ymin>.tif.",
  image_col    = "Sub-cell column within the image, 1..n_sub, left->right (west->east).",
  image_row    = "Sub-cell row within the image, 1..n_sub, top->bottom (north->south, raster convention).",
  cell_order   = "Sub-cell position within the image, 1..n_sub^2, row-major from the top-left corner.",
  centroid_x   = "Centroid easting in the source projected CRS (metres).",
  centroid_y   = "Centroid northing in the source projected CRS (metres).",
  centroid_lon = "Centroid longitude, WGS84 (EPSG:4326).",
  centroid_lat = "Centroid latitude, WGS84 (EPSG:4326).",
  tile         = "Source NEON 1 km mosaic tile the parent cell came from (if joined).",
  image_exists = "TRUE if the expected crop file was found on disk (if checked).",
  geometry_wkt = "Sub-cell polygon as WKT in the source CRS (if include_wkt=TRUE)."
)
dict <- data.frame(
  column      = names(out),
  r_class     = vapply(out, function(z) class(z)[1], character(1)),
  description = ifelse(names(out) %in% names(desc), desc[names(out)], ""),
  stringsAsFactors = FALSE
)
write.csv(dict, out_dict, row.names = FALSE)
cat("Wrote", out_dict, "- fill in blank descriptions for your added columns.\n")

