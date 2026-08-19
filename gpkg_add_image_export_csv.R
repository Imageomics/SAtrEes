###############################################################################
## gpkg_add_image_latlon_export_csv.R
##
## Companion to Grid_and_crop_HARV_RGB.R (Imageomics/SAtrEes).
##
## Reads the 16 x 16 patch sub-grid GeoPackage (here the disturbance version
## with the user's extra columns), keeps EVERY existing attribute unchanged,
## and appends:
##
##   1. image_file   - the 256 px RGB crop each patch belongs to. Reconstructs
##                     the exact filename written by Grid_and_crop_HARV_RGB.R:
##                     "<site>_RGB_256px_<grid_id>_<xmin>_<ymin>.tif"
##                     (xmin/ymin = the parent crop's lower-left corner).
##   2. cell_order   - position of the patch WITHIN its image, 1..(n_sub^2),
##                     in raster reading order: row-major from the TOP-LEFT
##                     (north-west) corner. This is the same order DINOv3 emits
##                     its 16 x 16 patch tokens, so cell_order indexes the model
##                     output directly. Also writes image_row / image_col.
##   3. centroid_lon / centroid_lat - centroid of each patch in WGS84
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

in_gpkg   <- file.path(shp_dir, "HARV_grid_256px_sub16_disturbance.gpkg")  # input
out_csv   <- file.path(home_dir, "HARV_grid_256px_sub16_disturbance.csv")   # output CSV
out_dict  <- file.path(home_dir, "data_dictionary_auto.csv")               # auto dict

site      <- "HARV"   # filename prefix used by the crop writer
n_sub     <- 16       # patches per side (16 -> 256 patches per 256 px crop)
gid_col   <- "grid_id"  # parent-cell id column name in the gpkg

## OPTIONAL extras (set to NA / FALSE to skip) --------------------------------
## Parent 256 px grid: its "tile" column (source NEON mosaic tile) is joined on grid_id.
parent_grid <- file.path(shp_dir, "HARV_grid_256px.shp")
## Crops folder: flag whether each expected crop actually exists on disk
## (all-nodata cells were skipped by the crop writer, so some won't).
crop_dir    <- file.path(home_dir, "Imagery", "NEON", site, "RGB_256px_crops")
## Site boundary: flag each patch inside/outside HARV_Boundary by intersection.
boundary_shp <- file.path(shp_dir, "HARV_Boundary.shp")
include_wkt <- FALSE       # TRUE -> add a geometry_wkt column to the CSV

## ---- 1. Read ---------------------------------------------------------------
stopifnot(file.exists(in_gpkg))
g <- st_read(in_gpkg, quiet = TRUE)
stopifnot(gid_col %in% names(g))
orig_cols <- setdiff(names(g), attr(g, "sf_column"))   # everything but geometry
src_crs   <- st_crs(g)
cat("Read", nrow(g), "patches,", length(orig_cols), "attribute columns.\n")
cat("Source CRS:", src_crs$input, "\n")

## Patch size (metres) inferred straight from a patch polygon, so this stays
## correct for any imagery resolution (0.1 m -> 1.6 m patches here).
bb1 <- st_bbox(g[1, ])
s   <- as.numeric(bb1[["xmax"]] - bb1[["xmin"]])
cat("Inferred patch size:", s, "m\n")

## ---- 2. Patch centroids (native CRS) ---------------------------------------
## suppress the "assumes planar" note; the source CRS is projected (UTM), so a
## planar centroid is exactly what we want.
cen  <- suppressWarnings(st_centroid(st_geometry(g)))
xy   <- st_coordinates(cen)                            # matrix: X (easting), Y (northing)
cx   <- xy[, 1]; cy <- xy[, 2]

## ---- 3. Parent crop lower-left corner, per grid_id -------------------------
## Reconstructed by grouping the patches: the parent xmin/ymin is the min of
## the patch centroids minus half a patch. This does NOT assume any particular
## lattice alignment, so it stays correct for any grid.
gidv  <- g[[gid_col]]
pxmin <- ave(cx, gidv, FUN = min) - s / 2
pymin <- ave(cy, gidv, FUN = min) - s / 2

## integer corner used in the crop filename (crop writer used as.integer(round()))
ixmin <- as.integer(round(pxmin))
iymin <- as.integer(round(pymin))

## ---- 4. (1) image_file -----------------------------------------------------
image_file <- sprintf("%s_RGB_256px_%s_%d_%d.tif", site, as.character(gidv), ixmin, iymin)

## ---- 5. (2) order of the patch within its image ----------------------------
## Column index west->east (1..n_sub) and geographic row south->north (1..n_sub).
image_col <- as.integer(floor((cx - pxmin) / s)) + 1L
geo_row   <- as.integer(floor((cy - pymin) / s)) + 1L
## Flip to raster convention: image_row 1 = TOP (north). cell_order is row-major
## from the top-left corner, matching how pixels are laid out in the crop AND
## the order DINOv3 emits its patch tokens.
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

## in_boundary: TRUE if the patch intersects the HARV site boundary. Uses the
## patch polygon (not its centroid), so a patch straddling the edge counts as
## inside; switch st_geometry(g) to the centroids `cen` if you want a strict
## in/out split with no straddlers.
if (!is.na(boundary_shp) && file.exists(boundary_shp)) {
  bnd <- st_union(st_transform(st_read(boundary_shp, quiet = TRUE), src_crs))
  new_cols$in_boundary <- lengths(st_intersects(st_geometry(g), bnd)) > 0
  cat("Boundary flag added:", sum(new_cols$in_boundary), "of",
      nrow(new_cols), "patches intersect HARV_Boundary.\n")
} else {
  cat("NOTE: boundary shapefile not found; 'in_boundary' column skipped.\n")
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
  grid_id      = "ID of the parent 256 px crop (deterministic; matches the crop filename and the 256 px grid).",
  sub_col      = "Patch column index within the crop, 1..n_sub, west->east (geographic).",
  sub_row      = "Patch row index within the crop, 1..n_sub, south->north (geographic).",
  image_file   = "RGB 256 px crop this patch belongs to: <site>_RGB_256px_<grid_id>_<xmin>_<ymin>.tif.",
  image_col    = "Patch column within the image, 1..n_sub, left->right (west->east).",
  image_row    = "Patch row within the image, 1..n_sub, top->bottom (north->south, raster convention).",
  cell_order   = "Patch position within the image, 1..n_sub^2, row-major from the top-left corner (matches DINOv3 token order).",
  centroid_x   = "Centroid easting in the source projected CRS (metres).",
  centroid_y   = "Centroid northing in the source projected CRS (metres).",
  centroid_lon = "Centroid longitude, WGS84 (EPSG:4326).",
  centroid_lat = "Centroid latitude, WGS84 (EPSG:4326).",
  tile         = "Source NEON 1 km mosaic tile the parent crop came from (if joined).",
  image_exists = "TRUE if the expected crop file was found on disk (if checked).",
  in_boundary  = "TRUE if the patch intersects the HARV site boundary (HARV_Boundary.shp), else FALSE.",
  geometry_wkt = "Patch polygon as WKT in the source CRS (if include_wkt=TRUE)."
)
dict <- data.frame(
  column      = names(out),
  r_class     = vapply(out, function(z) class(z)[1], character(1)),
  description = ifelse(names(out) %in% names(desc), desc[names(out)], ""),
  stringsAsFactors = FALSE
)
write.csv(dict, out_dict, row.names = FALSE)
cat("Wrote", out_dict, "- fill in blank descriptions for your added columns.\n")
