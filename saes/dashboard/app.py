"""Streamlit dashboard for exploring SAE features on flyover imagery.

Given a run and a feature index, this shows the images that activate the
feature most strongly, with the activating patches highlighted.

Usage:
    uv run streamlit run saes/dashboard/app.py

Run ``saes/dashboard/build_index.py`` first to skip the one-time index build
that otherwise happens lazily on first selection of a run.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import streamlit as st
from PIL import Image

from saev.viz import add_highlights

from saes.dashboard.data import (
    DEFAULT_RUNS_ROOT,
    build_index,
    get_run_info,
    get_run_infos,
    image_paths,
    load_index,
    top_images,
)

MODEL_RES = 256
PATCH_SIZE = 16


@st.cache_resource(show_spinner=False)
def _infos(runs_root: str) -> list:
    return get_run_infos(pathlib.Path(runs_root))


@st.cache_resource(show_spinner=False)
def _index(runs_root: str, run_id: str, shards_hash: str) -> dict:
    info = get_run_info(pathlib.Path(runs_root), run_id)
    assert info is not None
    return load_index(build_index(info))


@st.cache_data(show_spinner=False)
def _paths(runs_root: str, run_id: str, shards_hash: str) -> tuple:
    info = get_run_info(pathlib.Path(runs_root), run_id)
    assert info is not None
    return image_paths(info.md, info.shards_dir)


def _feature_order(index: dict, sort_by: str) -> np.ndarray:
    if sort_by == "Sparsity":
        return np.argsort(index["sparsity"], kind="stable")
    if sort_by == "Mean act.":
        return np.argsort(-index["mean_values"], kind="stable")
    return np.argsort(-index["token_vals"][:, 0], kind="stable")


def render_image(
    path: pathlib.Path, patches: np.ndarray, upper: float, opacity: float, show: bool
) -> Image.Image:
    img = Image.open(path).convert("RGB").resize(
        (MODEL_RES, MODEL_RES), Image.LANCZOS
    )
    if show and upper > 0:
        img = add_highlights(img, patches, PATCH_SIZE, upper=upper, opacity=opacity)
    return img


st.set_page_config(page_title="SAE Feature Explorer", layout="wide")

st.sidebar.title("SAE Feature Explorer")
runs_root = st.sidebar.text_input("Runs root", value=str(DEFAULT_RUNS_ROOT))

infos = _infos(runs_root)
if not infos:
    st.error(
        "No runs with inference artifacts found under the runs root. "
        "Run inference (`saes/launch.py features`) first."
    )
    st.stop()

labels = {}
for info in infos:
    top_k = info.run.config.get("sae", {}).get("activation", {}).get("top_k")
    labels[f"{info.run.run_id} · top-k={top_k}"] = info

run_label = st.sidebar.selectbox("Run", list(labels))
info = labels[run_label]

with st.spinner("Loading feature index..."):
    index = _index(runs_root, info.run.run_id, info.md.hash)
    d_sae = index["d_sae"]

st.sidebar.markdown("---")
sort_by = st.sidebar.selectbox(
    "Sort features by", ["Off", "Sparsity", "Mean act.", "Max act."]
)
if sort_by == "Off":
    feature = st.sidebar.slider("Feature", 0, d_sae - 1, 0, key="feature")
    if st.sidebar.button("Random feature"):
        st.session_state.feature = int(np.random.default_rng().integers(0, d_sae))
        st.rerun()
    rank = None
else:
    order = _feature_order(index, sort_by)
    rank = st.sidebar.slider("Rank", 0, d_sae - 1, 0)
    feature = int(order[rank])
n_imgs = st.sidebar.slider("Images", 1, 24, 6)
n_cols = st.sidebar.slider("Columns", 1, 4, 3)
opacity = st.sidebar.slider("Overlay opacity", 0.0, 1.0, 0.85, 0.05)
show_highlights = st.sidebar.checkbox("Show patch highlighting", value=True)

# --- Main panel -------------------------------------------------------------
heading = f"Run `{info.run.run_id}` — feature {feature}"
if rank is not None:
    heading += f" · {sort_by.lower()} rank {rank}"
st.header(heading)

sparsity = float(index["sparsity"][feature])
mean = float(index["mean_values"][feature])
maxv = float(index["token_vals"][feature][0])
s1, s2, s3 = st.columns(3)
s1.metric("Sparsity", f"{sparsity:.4f}", help="Fraction of tokens the feature fires on.")
s2.metric("Mean act. (active)", f"{mean:.2f}", help="Mean activation over active tokens.")
s3.metric("Max activation", f"{maxv:.2f}")

results = top_images(index, feature, n_imgs)
if not results:
    st.warning("This feature never activates. Try another.")
    st.stop()

paths = _paths(runs_root, info.run.run_id, info.md.hash)
n_rows = (len(results) + n_cols - 1) // n_cols
for r in range(n_rows):
    cols = st.columns(n_cols)
    for c, (ex, mx, patches) in enumerate(results[r * n_cols : (r + 1) * n_cols]):
        img = render_image(paths[ex], patches, maxv, opacity, show_highlights)
        cols[c].image(img, width="stretch")
        cols[c].caption(f"img {ex} · max {mx:.2f}\n{paths[ex].name}")