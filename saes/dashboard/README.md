# SAE Feature Dashboard

Interactive Streamlit dashboard for inspecting the features learned by SAEs
trained on the flyover image data. Pick a run and a feature, and the dashboard
shows the images with the strongest activation for that feature, with the
activating patches highlighted.

## Setup

```bash
uv sync
```

## Build the feature index (optional)

The dashboard builds this automatically on first use of a run, but that takes
~1 minute and blocks the page. Build it upfront instead:

```bash
uv run python saes/dashboard/build_index.py                 # all runs
uv run python saes/dashboard/build_index.py --run agw9clvn  # one run
uv run python saes/dashboard/build_index.py --top-k 512     # more tokens per feature
```

The index is written next to the inference artifacts
(`<run>/inference/<shard-hash>/feature_index.npz`). It stores, for every
feature, its top-256 globally highest-activating tokens.

## Run the dashboard

```bash
uv run streamlit run saes/dashboard/app.py
```

The runs root defaults to `/fs/ess/PAS2136/SAtrEes/saev/runs` and can be
changed in the sidebar. To view the app from your laptop, forward the port:

```bash
ssh -L 8501:localhost:8501 <cluster>
```

Then open http://localhost:8501.

## Notes

- Only runs with a completed inference pass (a `token_acts.npz` under
  `inference/`) are listed. Re-run `saes/launch.py features` for runs missing it.
- Images are shown at model resolution (256×256) so the 16×16 patch grid lines
  up with the overlays.
- Feature statistics (sparsity, mean activation) come from the inference
  artifacts `sparsity.pt` / `mean_values.pt`.