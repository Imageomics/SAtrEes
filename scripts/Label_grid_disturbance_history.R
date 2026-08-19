###############################################################################
## Label_grid_disturbance_history.R
##
## Labels each cell of the HARV 25 m grid (built by Grid_and_crop_HARV_RGB.R)
## with its disturbance history, drawn from these layers in
## data/hf110_land_use_history/hf110-01-gis.zip:
##
##   ph_ag_abandonment           field-abandonment + cutting years
##   ph_natural_disturbance2     ice/wind/tornado/fire/snow events
##   1938_hurricane_damage       1938 hurricane damage class (fixed year)
##   silviculture_treatments_08_21_2010   dated management-treatment log
##
## The two PH_Specific layers only cover the Prospect Hill tract; the
## hurricane and silviculture layers cover most of the wider property, so
## combining all four maximizes how much of the HARV grid gets a label.
## (ph_landuse_history, the undated Cultivated/Woodlot/Pasture land-use
## classification, and the stands_* historical stand maps are deliberately
## left out - this is disturbance events only.)
##
## Each cell gets two main labels plus a full-history sub-label:
##   disturb_majority       area label - the single dominant event, chosen
##                          by majority-wins area: whichever of the 4 source
##                          layers covers the most of the cell wins, and its
##                          most recent dated event in that cell is the text
##                          (see step 5b for why layer-level, not per-event).
##   disturb_majority_frac  fraction of the cell covered by that layer
##   disturb_recent_class   time label - the class of whichever dated event
##                          in the cell (across all 4 layers) is most recent;
##                          see step 5c for the 9-class taxonomy. Cells whose
##                          only events are undated fall back to that event's
##                          class with disturb_years_since left NA.
##   disturb_recent_label   the actual most-recent event's text + year
##   disturb_years_since    2022 minus the most-recent event's year (2022 is
##                          a fixed reference year, not "today" - re-derive
##                          if a "years since" as of a different date is
##                          ever needed)
##   disturb_history        sub-label - every dated event on record for the
##                          cell, oldest -> newest; undated events (a
##                          handful of natural-disturbance records dated only
##                          "prior"/"repeated") are appended last, tagged
##                          "(undated)".
##
## Two layers are written, each in three formats:
##   HARV_grid_25m_disturbance          the 25 m grid, one row per cell
##   HARV_disturbance_history_combined  the 4 source layers merged into one
##                                      polygon layer, one row per original
##                                      source polygon, each with its own
##                                      chronological history string - useful
##                                      for inspecting the raw combined
##                                      history independent of the grid.
##                                      Overlapping polygons from different
##                                      source layers are NOT dissolved into
##                                      a single non-overlapping mosaic - each
##                                      keeps its own original footprint.
##
## Formats per layer:
##   .gpkg  full-fidelity data for analysis in R/QGIS (no field limits)
##   .kml   full-fidelity, opens directly in Google Earth
##   .shp   reprojected to lat/lon for Google Earth Pro's Import; shortened
##          field names and disturb_history truncated at 254 characters
##          (the .dbf text-field limit) - the run log reports how many rows
##          were truncated. Use the .kml or .gpkg if you need the full text.
##
## Known data-quality caveats (accepted rather than engineered around):
##  * silviculture TREATMENT strings are free text; ~98% of the semicolon-
##    separated entries parse cleanly as "YYYY <code> description", but a
##    few source records are missing their semicolons and two entries get
##    concatenated into one description. Left as-is rather than guessed at.
##  * ph_natural_disturbance2$Date has two rows in "Mon-YY" form (e.g.
##    "Jul-84"); every such case in this dataset is 20th-century, so YY is
##    read as 19YY. Re-check this assumption if new data of that form shows
##    up with a different era.
##  * A cell can pick up more than one record from the same layer/year (e.g.
##    straddling two 1938 hurricane-damage severity zones) - each is listed,
##    since that's real spatial detail, not a duplicate.
###############################################################################

library(sf)
library(dplyr)

## ---- 0. Config ---------------------------------------------------------
## Pass a grid file as a command-line arg to run against a different grid
## (any polygon layer with a grid_id column) - e.g. for a local test file:
##   Rscript Label_grid_disturbance_history.R HARV_grid_25m_sub16.gpkg
## Outputs are then written next to that file instead of the cluster paths.
site <- "HARV"

out_dir <- "/fs/ess/PUOM0017/SAtrEes"   # matches Grid_and_crop_HARV_RGB.R
shp_dir <- file.path(out_dir, "Shapefiles")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 1) {
  grid_in    <- args[1]
  out_base   <- tools::file_path_sans_ext(grid_in)
  out_dir_l  <- dirname(grid_in)
} else {
  grid_in    <- file.path(shp_dir, paste0(site, "_grid_25m.shp"))
  out_base   <- file.path(shp_dir, paste0(site, "_grid_25m"))
  out_dir_l  <- shp_dir
}

grid_out     <- paste0(out_base, "_disturbance.gpkg")
combined_out <- file.path(out_dir_l, paste0(site, "_disturbance_history_combined.gpkg"))

## Google Earth views: KML (full-fidelity, no field-length limit) and an
## ESRI Shapefile (reprojected to lat/lon; the .dbf format caps text fields
## at 254 characters, so long disturb_history strings get truncated - see
## write_earth_views() below).
grid_kml_out      <- paste0(out_base, "_disturbance.kml")
grid_shp_out      <- paste0(out_base, "_disturbance.shp")
combined_kml_out  <- file.path(out_dir_l, paste0(site, "_disturbance_history_combined.kml"))
combined_shp_out  <- file.path(out_dir_l, paste0(site, "_disturbance_history_combined.shp"))

landuse_zip <- "data/hf110_land_use_history/hf110-01-gis.zip"
gis_base    <- file.path("/vsizip", landuse_zip, "Harvard_Forest_Properties_GIS_Layers")

min_frac <- 0.02   # ignore intersection slivers covering less of a cell than this
years_since_ref <- 2022   # fixed reference year for disturb_years_since

## ---- 1. Helpers for pulling years out of free-text fields ---------------
parse_year <- function(txt) {
  txt <- trimws(gsub('"', "", ifelse(is.na(txt), "", txt)))
  year <- rep(NA_real_, length(txt))
  pat4 <- "(17|18|19|20)[0-9]{2}"
  has4 <- grepl(pat4, txt)
  year[has4] <- as.numeric(regmatches(txt[has4], regexpr(pat4, txt[has4])))
  patMY <- "^[A-Za-z]{3}-([0-9]{2})$"
  needMY <- is.na(year) & grepl(patMY, txt)
  if (any(needMY)) {
    yy <- as.numeric(sub(patMY, "\\1", txt[needMY]))
    year[needMY] <- 1900 + yy
  }
  list(year = year, uncertain = grepl("\\?", txt))
}

extract_all_years <- function(txt) {
  regmatches(txt, gregexpr("(17|18|19|20)[0-9]{2}", txt))
}

## one silviculture TREATMENT string -> data.frame(year, desc), one row/event
parse_silv_events <- function(treatment_text) {
  if (is.na(treatment_text)) return(NULL)
  chunks <- strsplit(treatment_text, ";")[[1]]
  chunks <- gsub("^(\\s*\\([^)]*\\)\\s*)+", "", chunks)   # drop leading (compartment) tags
  chunks <- trimws(chunks)
  chunks <- chunks[nchar(chunks) > 0]
  if (length(chunks) == 0) return(NULL)
  rows <- lapply(chunks, function(ch) {
    m <- regmatches(ch, regexec("^(\\d{4})\\s*[A-Za-z0-9()&]*\\s*(.+)$", ch))[[1]]
    if (length(m) == 3) return(data.frame(year = as.numeric(m[2]), desc = trimws(m[3])))
    yy <- regmatches(ch, regexpr("(17|18|19|20)[0-9]{2}", ch))
    if (length(yy) == 1 && nchar(yy) == 4) {
      desc <- trimws(gsub("(17|18|19|20)[0-9]{2}", "", ch))
      desc <- trimws(gsub("^[A-Za-z0-9()&.]+", "", desc))
      if (nchar(desc) == 0) desc <- "treatment"
      return(data.frame(year = as.numeric(yy), desc = desc))
    }
    NULL   # no year anywhere in this entry -> can't place it chronologically, drop
  })
  do.call(rbind, rows)
}

## df must have columns id, ev_type, ev_year -> one summarized row per id,
## history ordered oldest -> newest with undated events appended last.
summarize_events <- function(df) {
  df |>
    distinct(id, ev_type, ev_year) |>
    group_by(id) |>
    arrange(is.na(ev_year), ev_year, .by_group = TRUE) |>
    summarise(
      disturb_history  = paste0(ev_type, " (", ifelse(is.na(ev_year), "undated", ev_year), ")",
                                 collapse = "; "),
      disturb_n_events = n(),
      disturb_first_yr = suppressWarnings(min(ev_year, na.rm = TRUE)),
      disturb_last_yr  = suppressWarnings(max(ev_year, na.rm = TRUE)),
      .groups = "drop"
    ) |>
    mutate(across(c(disturb_first_yr, disturb_last_yr), ~ ifelse(is.infinite(.x), NA_real_, .x)))
}

## Bucket a silviculture TREATMENT description into one of 5 sub-classes.
## Free text, so this is a best-effort keyword classifier, not exact.
## Priority order (first match wins): clearcut > salvage > regen harvest >
## thinning/tending > other. There's no "monitoring" class - inspection/
## photo/measurement-only entries fall into "other" and are flagged
## non-disturbance by is_monitoring_only() below, so step 5c skips past them
## to the next most recent actual disturbance instead of surfacing them as
## the cell's headline event.
classify_silv_desc <- function(desc) {
  d <- tolower(desc)
  ifelse(grepl("clear ?cut|clearcutting|sawtimber|logging|harvest|plantation removal", d),
         "Silviculture - clearcut",
  ifelse(grepl("salvage", d),
         "Silviculture - salvage",
  ifelse(grepl("shelterwood|group selection|group cut|reproduction cut|coppice|cordwood", d),
         "Silviculture - regen harvest",
  ifelse(grepl("thin|weed|prun|plant|seed|fill|girdl|releas|underplant|improvement|selection cut|selective cut", d),
         "Silviculture - thinning",
         "Silviculture - other"))))
}

## An "other"-classed entry counts as non-disturbance (skip it when picking
## the most-recent event) only if it looks purely administrative/monitoring
## and didn't already match a structural keyword above - e.g. "inspection &
## thinning" still classifies (and counts) as thinning; bare "inspection"
## does not count as a disturbance at all.
is_monitoring_only <- function(desc, cls) {
  cls == "Silviculture - other" &
    grepl("inspect|\\binsp\\b|photo|exam|measur|tally|cruise|report", tolower(desc))
}

## Write a layer's KML (full-fidelity) and shapefile (lat/lon, short field
## names, disturb_history truncated to the DBF's 254-char text limit) views.
## hist_col names the long-text column to truncate for the shapefile.
write_earth_views <- function(x, kml_path, shp_path, hist_col, short_names) {
  x_wgs <- st_transform(x, 4326)
  st_write(x_wgs, kml_path, driver = "KML", delete_dsn = TRUE, quiet = TRUE)

  too_long <- nchar(x_wgs[[hist_col]]) > 254
  x_wgs[[hist_col]][too_long] <- paste0(substr(x_wgs[[hist_col]][too_long], 1, 251), "...")
  names(x_wgs)[match(names(short_names), names(x_wgs))] <- unname(short_names)
  st_write(x_wgs, shp_path, delete_dsn = TRUE, quiet = TRUE)

  cat("  KML ->", kml_path, "\n")
  cat("  SHP ->", shp_path, "(", sum(too_long), "/", nrow(x_wgs), "histories truncated )\n")
}

## ---- 2. Build one (key, ev_type, ev_year) event per source record -------
## and one (key, geometry) region per source polygon, kept separate so the
## expensive geometric intersection only runs once per polygon even when a
## polygon (e.g. a silviculture stand) contributes many dated events.

aband  <- st_read(file.path(gis_base, "PH_Specific/ph_ag_abandonment.shp"), quiet = TRUE) |> st_make_valid()
key_ab <- paste0("aband_", seq_len(nrow(aband)))
pa     <- parse_year(aband$FIELD_ABAN)
skip   <- grepl("permanent", aband$FIELD_ABAN, ignore.case = TRUE)   # never abandoned - not a disturbance event
ev_ab1 <- data.frame(
  key     = key_ab,
  ev_type = ifelse(pa$uncertain, "Field abandoned (approx.)", "Field abandoned"),
  ev_year = pa$year,
  ev_class = "Agricultural abandonment",
  ev_is_disturbance = TRUE
) |> filter(!skip, !is.na(ev_year))
cut_years <- extract_all_years(aband$CUTTING)
ev_ab2 <- do.call(rbind, lapply(seq_along(cut_years), function(i) {
  yrs <- cut_years[[i]]
  if (length(yrs) == 0) return(NULL)
  data.frame(key = key_ab[i], ev_type = "Cutting", ev_year = as.numeric(yrs), ev_class = "Agricultural cutting",
             ev_is_disturbance = TRUE)
}))
ev_aband   <- rbind(ev_ab1, ev_ab2)
geom_aband <- st_sf(key = key_ab, geometry = st_geometry(aband))

nd     <- st_read(file.path(gis_base, "PH_Specific/ph_natural_disturbance2.shp"), quiet = TRUE) |> st_make_valid()
key_nd <- paste0("natdist_", seq_len(nrow(nd)))
pn     <- parse_year(nd$Date)
ev_nd  <- data.frame(key = key_nd,
                      ev_type = ifelse(pn$uncertain, paste0(nd$TYPE, " (approx.)"), nd$TYPE),
                      ev_year = pn$year,
                      ev_class = "Natural disturbance",
                      ev_is_disturbance = TRUE)
geom_nd <- st_sf(key = key_nd, geometry = st_geometry(nd))

hur    <- st_read(file.path(gis_base, "1938_hurricane_damage.shp"), quiet = TRUE) |> st_make_valid()
key_hu <- paste0("hurr_", seq_len(nrow(hur)))
ev_hur <- data.frame(key = key_hu, ev_type = paste0("1938 hurricane (", hur$DAMAGE, ")"), ev_year = 1938,
                      ev_class = "1938 hurricane", ev_is_disturbance = TRUE)
geom_hur <- st_sf(key = key_hu, geometry = st_geometry(hur))

silv   <- st_read(file.path(gis_base, "silviculture_treatments_08_21_2010.shp"), quiet = TRUE) |> st_make_valid()
key_sv <- paste0("silv_", seq_len(nrow(silv)))
ev_silv <- do.call(rbind, lapply(seq_len(nrow(silv)), function(i) {
  ev <- parse_silv_events(silv$TREATMENT[i])
  if (is.null(ev)) return(NULL)
  cls <- classify_silv_desc(ev$desc)
  data.frame(key = key_sv[i], ev_type = ev$desc, ev_year = ev$year, ev_class = cls,
             ev_is_disturbance = !is_monitoring_only(ev$desc, cls))
}))
geom_silv <- st_sf(key = key_sv, geometry = st_geometry(silv))

events  <- rbind(ev_aband, ev_nd, ev_hur, ev_silv)
crs0    <- st_crs(aband)
regions <- rbind(geom_aband, geom_nd, geom_hur, st_transform(geom_silv, crs0))

cat("Combined", nrow(regions), "source polygons ->", nrow(events), "dated/undated events",
    "(", sum(!is.na(events$ev_year)), "dated )\n")

## ---- 3. Write the combined layer: one row per source polygon, its own history
poly_history <- events |>
  filter(!is.na(ev_type)) |>
  rename(id = key) |>
  summarize_events() |>
  rename(key = id)

combined <- regions |>
  left_join(poly_history, by = "key") |>
  mutate(
    disturb_history  = ifelse(is.na(disturb_history), "No recorded events", disturb_history),
    disturb_n_events = ifelse(is.na(disturb_n_events), 0L, disturb_n_events)
  )

st_write(combined, combined_out, delete_dsn = TRUE, quiet = TRUE)
cat("Combined history layer (", nrow(combined), "polygons) ->", combined_out, "\n")
write_earth_views(combined, combined_kml_out, combined_shp_out, "disturb_history",
                   c(disturb_history = "dist_hist", disturb_n_events = "dist_nevt",
                     disturb_first_yr = "dist_fyr", disturb_last_yr = "dist_lyr"))

## ---- 4. Load the grid, intersect with the combined regions ---------------
grid <- st_read(grid_in, quiet = TRUE)
grid$cell_area <- as.numeric(st_area(grid))
regions <- st_transform(regions, st_crs(grid))

ov <- st_intersection(st_make_valid(grid), regions)
ov$ov_area <- as.numeric(st_area(ov))
ov <- st_drop_geometry(ov) |>
  mutate(frac = ov_area / cell_area) |>
  filter(frac >= min_frac)

## ---- 5. Expand to events, order chronologically (undated last) ----------
cell_events <- ov |>
  left_join(events, by = "key", relationship = "many-to-many") |>
  filter(!is.na(ev_type)) |>
  rename(id = grid_id) |>
  summarize_events() |>
  rename(grid_id = id)

max_frac <- ov |> group_by(grid_id) |> summarise(disturb_max_frac = max(frac), .groups = "drop")

## ---- 5b. Majority-wins main label -----------------------------------------
## "Majority" is decided at the source-layer level (Field abandonment /
## Natural disturbance / 1938 hurricane / Silviculture treatment), not per
## individual event: three of the four layers already give one clean type
## per record, but silviculture is a log of many co-located treatments on
## the same polygon, so competing on raw event text would let it fragment
## into a dozen small slices and never win against a single clean label from
## another layer. Whichever layer covers the most of the cell's area wins;
## its label text is that layer's most recent dated event in the cell (a
## simple, explainable tie-break for silviculture's multi-event polygons).
layer_label <- c(aband = "Field abandonment", natdist = "Natural disturbance",
                  hurr = "1938 hurricane", silv = "Silviculture treatment")
ov$source_layer <- layer_label[sub("_[0-9]+$", "", ov$key)]

layer_area <- ov |> group_by(grid_id, source_layer) |> summarise(layer_frac = sum(frac), .groups = "drop")
majority_layer <- layer_area |> group_by(grid_id) |> slice_max(layer_frac, n = 1, with_ties = FALSE) |> ungroup()

majority_label <- ov |>
  inner_join(majority_layer, by = c("grid_id", "source_layer")) |>
  left_join(events, by = "key", relationship = "many-to-many") |>
  filter(!is.na(ev_type)) |>
  group_by(grid_id) |>
  arrange(is.na(ev_year), desc(ev_year), .by_group = TRUE) |>
  slice(1) |>
  ungroup() |>
  transmute(grid_id, disturb_majority = paste0(ev_type, ifelse(is.na(ev_year), "", paste0(" (", ev_year, ")"))))

majority <- majority_layer |>
  select(grid_id, disturb_majority_frac = layer_frac) |>
  left_join(majority_label, by = "grid_id")

## ---- 5c. Most-recent-disturbance classification ---------------------------
## Time label, independent of the area-based majority label above: whichever
## dated event is most recent (across all 4 layers) sets the cell's class.
## Monitoring-only silviculture entries (ev_is_disturbance == FALSE) are
## excluded before picking "most recent" so a 2005 inspection visit doesn't
## mask a 1980 clearcut as the cell's headline event - it can still show up
## in disturb_history, just not here. A cell whose only events are undated
## falls back to that event's class with disturb_years_since left NA.
recent <- ov |>
  left_join(events, by = "key", relationship = "many-to-many") |>
  filter(!is.na(ev_type), ev_is_disturbance) |>
  group_by(grid_id) |>
  arrange(is.na(ev_year), desc(ev_year), .by_group = TRUE) |>
  slice(1) |>
  ungroup() |>
  transmute(
    grid_id,
    disturb_recent_class = ev_class,
    disturb_recent_label = paste0(ev_type, ifelse(is.na(ev_year), "", paste0(" (", ev_year, ")"))),
    disturb_recent_yr    = ev_year,
    disturb_years_since  = ifelse(is.na(ev_year), NA_real_, years_since_ref - ev_year)
  )

## ---- 6. Join back onto the full grid --------------------------------------
grid <- grid |>
  left_join(cell_events, by = "grid_id") |>
  left_join(max_frac, by = "grid_id") |>
  left_join(majority, by = "grid_id") |>
  left_join(recent, by = "grid_id") |>
  mutate(
    disturb_majority      = ifelse(is.na(disturb_majority), "Not Mapped", disturb_majority),
    disturb_majority_frac = ifelse(is.na(disturb_majority_frac), 0, disturb_majority_frac),
    disturb_recent_class  = ifelse(is.na(disturb_recent_class), "Not Mapped", disturb_recent_class),
    disturb_recent_label  = ifelse(is.na(disturb_recent_label), "Not Mapped", disturb_recent_label),
    disturb_history        = ifelse(is.na(disturb_history), "Not Mapped", disturb_history),
    disturb_n_events       = ifelse(is.na(disturb_n_events), 0L, disturb_n_events),
    disturb_max_frac       = ifelse(is.na(disturb_max_frac), 0, disturb_max_frac)
  ) |>
  select(-cell_area)

st_write(grid, grid_out, delete_dsn = TRUE, quiet = TRUE)

cat("Labeled", nrow(grid), "cells\n")
cat("Not Mapped:", sum(grid$disturb_history == "Not Mapped"),
    sprintf("(%.1f%%)\n", 100 * mean(grid$disturb_history == "Not Mapped")))
cat("Cells with >1 event:", sum(grid$disturb_n_events > 1), "\n")
cat("\ndisturb_recent_class distribution:\n")
print(sort(table(grid$disturb_recent_class), decreasing = TRUE))
cat("\nGrid ->", grid_out, "\n")
write_earth_views(grid, grid_kml_out, grid_shp_out, "disturb_history",
                   c(disturb_majority = "dist_majr", disturb_majority_frac = "dist_mjfr",
                     disturb_recent_class = "dist_rcls", disturb_recent_label = "dist_rlab",
                     disturb_recent_yr = "dist_ryr", disturb_years_since = "dist_yrsn",
                     disturb_history = "dist_hist", disturb_n_events = "dist_nevt",
                     disturb_first_yr = "dist_fyr", disturb_last_yr = "dist_lyr",
                     disturb_max_frac = "dist_mfrc"))
