###############################################################################
## plot_classification_accuracy_mockup.R
##
## MOCKUP figure - simulated data, not real model output. Sketches what a
## validation plot for an AI/transformer classifier of disturb_recent_yr
## (see Label_grid_disturbance_history.R) would look like: classified year
## vs. observed (GIS-derived) year on a 1:1 line, points colored by years
## since disturbance. Swap sim_data() for a real join of
## disturb_recent_yr (observed) against the model's predicted year once
## classification results exist.
###############################################################################

library(ggplot2)
library(dplyr)

set.seed(42)
ref_year <- 2022

sim_data <- function(n = 220) {
  ## Mimic the HARV disturbance record: a big spike at the 1938 hurricane,
  ## sparser agricultural/silviculture events spread across the 20th c.,
  ## denser recent silviculture activity near the present.
  true_year <- round(c(
    rep(1938, round(n * 0.22)),
    runif(round(n * 0.38), 1900, 1990),
    runif(round(n * 0.40), 1990, 2020)
  ))
  true_year <- sample(true_year, n)
  years_since <- ref_year - true_year

  ## Classification error grows with age: recent, well-imaged disturbances
  ## are easy; the model has less spectral/temporal signal to work with the
  ## further back an event sits.
  noise_sd <- 1 + years_since / 22
  pred_year <- round(true_year + rnorm(n, mean = 0, sd = noise_sd))

  data.frame(true_year = true_year, pred_year = pred_year, years_since = years_since)
}

d <- sim_data()

r2 <- cor(d$true_year, d$pred_year)^2
mae <- mean(abs(d$true_year - d$pred_year))

lims <- range(c(d$true_year, d$pred_year)) + c(-3, 3)

p <- ggplot(d, aes(x = true_year, y = pred_year, color = years_since)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "#c3c2b7", linewidth = 0.7) +
  geom_point(size = 2.4, alpha = 0.8) +
  scale_color_gradientn(
    colors = c("#cde2fb", "#6da7ec", "#2a78d6", "#184f95"),
    name = "Years since\ndisturbance"
  ) +
  coord_equal(xlim = lims, ylim = lims) +
  annotate("label", x = lims[1] + 4, y = lims[2] - 4, hjust = 0, vjust = 1,
           label = sprintf("R² = %.2f\nMAE = %.1f yr", r2, mae),
           fill = "#fcfcfb", color = "#0b0b0b", label.size = 0, size = 3.6) +
  labs(
    title = "AI classification of disturbance year vs. observed",
    subtitle = "MOCKUP — simulated data, not real model output",
    x = "Observed disturbance year (GIS record)",
    y = "AI-classified disturbance year",
    caption = "Dashed line = 1:1 (perfect agreement)"
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

out_path <- "classification_accuracy_mockup.png"
ggsave(out_path, p, width = 7.5, height = 6.5, dpi = 200)
cat("Wrote", out_path, "\n")
