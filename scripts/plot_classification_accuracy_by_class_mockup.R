###############################################################################
## plot_classification_accuracy_by_class_mockup.R
##
## MOCKUP figure - simulated data, not real model output. Same idea as
## plot_classification_accuracy_mockup.R (classified year vs. observed year
## on a 1:1 line) but colored by disturb_recent_class instead of years since
## disturbance, using the 8-class taxonomy from Label_grid_disturbance_history.R
## / HARV_grid_25m_disturbance_metadata.csv. Swap sim_data() for a real join
## of disturb_recent_yr/disturb_recent_class (observed) against the model's
## predicted year/class once classification results exist.
##
## 8 classes exceed the categorical palette's all-pairs-validated CVD floor
## (only the first 3 slots clear every pair on a scatter, where any two
## colors can end up adjacent anywhere in the plot - see the dataviz skill's
## color-formula.md). Point shape is layered on as a second identity channel
## so class is never carried by hue alone.
###############################################################################

library(ggplot2)
library(dplyr)

set.seed(42)
ref_year <- 2022

## Fixed 8-hue categorical order (dataviz skill default palette).
class_levels <- c(
  "1938 hurricane", "Silviculture - clearcut", "Agricultural abandonment",
  "Silviculture - thinning", "Natural disturbance", "Agricultural cutting",
  "Silviculture - regen harvest", "Silviculture - salvage/other"
)
class_colors <- c(
  "1938 hurricane"                = "#2a78d6",  # blue
  "Silviculture - clearcut"       = "#eb6834",  # orange
  "Agricultural abandonment"      = "#1baf7a",  # aqua
  "Silviculture - thinning"       = "#eda100",  # yellow
  "Natural disturbance"           = "#e87ba4",  # magenta
  "Agricultural cutting"          = "#008300",  # green
  "Silviculture - regen harvest"  = "#4a3aa7",  # violet
  "Silviculture - salvage/other"  = "#e34948"   # red
)
class_shapes <- c(
  "1938 hurricane"                = 16,  # filled circle
  "Silviculture - clearcut"       = 17,  # filled triangle
  "Agricultural abandonment"      = 15,  # filled square
  "Silviculture - thinning"       = 18,  # filled diamond
  "Natural disturbance"           = 8,   # asterisk
  "Agricultural cutting"          = 3,   # plus
  "Silviculture - regen harvest"  = 4,   # cross
  "Silviculture - salvage/other"  = 6    # open inverted triangle
)

## One block of simulated events per class: n, true-year range, and how
## noisy the AI year-classification is for that class (fixed, well-imaged
## single-year events like the 1938 hurricane are easiest; sparse/older
## classes like salvage & agricultural cutting are hardest).
class_spec <- list(
  "1938 hurricane"               = list(n = 55, yr = c(1938, 1938), sd = 0.6),
  "Silviculture - clearcut"      = list(n = 40, yr = c(1950, 2018), sd = 2.2),
  "Agricultural abandonment"     = list(n = 25, yr = c(1850, 1950), sd = 4.5),
  "Silviculture - thinning"      = list(n = 45, yr = c(1955, 2020), sd = 2.0),
  "Natural disturbance"          = list(n = 20, yr = c(1900, 2015), sd = 3.5),
  "Agricultural cutting"         = list(n = 18, yr = c(1850, 1945), sd = 5.5),
  "Silviculture - regen harvest" = list(n = 22, yr = c(1955, 2010), sd = 2.8),
  "Silviculture - salvage/other" = list(n = 15, yr = c(1955, 2015), sd = 6.0)
)

sim_data <- function() {
  rows <- lapply(names(class_spec), function(cls) {
    s <- class_spec[[cls]]
    true_year <- if (s$yr[1] == s$yr[2]) rep(s$yr[1], s$n) else round(runif(s$n, s$yr[1], s$yr[2]))
    pred_year <- round(true_year + rnorm(s$n, 0, s$sd))
    data.frame(class = cls, true_year = true_year, pred_year = pred_year)
  })
  d <- do.call(rbind, rows)
  d$class <- factor(d$class, levels = class_levels)
  d
}

d <- sim_data()

r2 <- cor(d$true_year, d$pred_year)^2
mae <- mean(abs(d$true_year - d$pred_year))

lims <- range(c(d$true_year, d$pred_year)) + c(-3, 3)

p <- ggplot(d, aes(x = true_year, y = pred_year, color = class, shape = class)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "#c3c2b7", linewidth = 0.7) +
  geom_point(size = 2.6, alpha = 0.85, stroke = 0.9) +
  scale_color_manual(values = class_colors, name = "Disturbance class") +
  scale_shape_manual(values = class_shapes, name = "Disturbance class") +
  coord_equal(xlim = lims, ylim = lims) +
  annotate("label", x = lims[1] + 4, y = lims[2] - 4, hjust = 0, vjust = 1,
           label = sprintf("R² = %.2f\nMAE = %.1f yr", r2, mae),
           fill = "#fcfcfb", color = "#0b0b0b", size = 3.6) +
  labs(
    title = "AI classification of disturbance year, by class",
    subtitle = "MOCKUP — simulated data, not real model output",
    x = "Observed disturbance year (GIS record)",
    y = "AI-classified disturbance year",
    caption = "Dashed line = 1:1 (perfect agreement). Color + shape both carry class identity."
  ) +
  theme_minimal(base_size = 13) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "#e1e0d9", linewidth = 0.4),
    axis.line = element_line(color = "#c3c2b7"),
    plot.title = element_text(face = "bold"),
    plot.subtitle = element_text(color = "#898781", face = "italic"),
    plot.caption = element_text(color = "#898781"),
    legend.position = "right"
  )

out_path <- "classification_accuracy_by_class_mockup.png"
ggsave(out_path, p, width = 8.5, height = 6.5, dpi = 200)
cat("Wrote", out_path, "\n")
