# pages/04_Heatmap.py
from pathlib import Path
import importlib.util
import os
import pandas as pd
import plotly.express as px
import streamlit as st
from common_header import top_nav

top_nav("Heatmap")

HERE = Path(__file__).parent.resolve()
CSV_PATH = HERE / "hes_yearly_totals_2015_2024.csv"
BUILDER  = HERE / "data_builder.py"

# Header
col1, col2 = st.columns([6, 1])
with col1:
    st.title("Trends — Heatmap (Composition)")
with col2:
    st.page_link("pages/03_Trends.py", label="Open Line view →", icon="↩")

# Dev-mode: auto-build CSV if missing
def _dev_mode() -> bool:
    v = os.getenv("DEV_MODE", "").strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    try:
        return bool(st.secrets.get("dev_mode", False))
    except Exception:
        return False

DEV = _dev_mode()

if DEV and (not CSV_PATH.exists()) and BUILDER.exists():
    spec = importlib.util.spec_from_file_location("data_builder", str(BUILDER))
    db = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(db)
    out_csv, _ = db.build_yearly_csv(
        base_folder=str(HERE),
        out_csv=CSV_PATH.name,
        audit_csv="hes_yearly_audit.csv",
    )
    CSV_PATH = HERE / out_csv
elif not DEV and not CSV_PATH.exists():
    st.error("CSV missing. Add the file or enable DEV to auto-build.")
    st.stop()

# Load & order years
df = pd.read_csv(CSV_PATH)
if "YearStart" in df.columns:
    df = df.sort_values("YearStart").reset_index(drop=True)
else:
    df = df.sort_values("YearLabel").reset_index(drop=True)

# Ensure composition shares exist (derive if needed)
need_nums = {"Emergency_FAE", "Waiting_FAE", "Planned_FAE"}
if need_nums.issubset(df.columns):
    denom3 = (
        df[["Emergency_FAE", "Waiting_FAE", "Planned_FAE"]]
        .sum(axis=1)
        .replace(0, pd.NA)
    )
    df["Emergency_pct"] = df["Emergency_FAE"] / denom3
    df["Waiting_pct"]   = df["Waiting_FAE"]   / denom3
    df["Planned_pct"]   = df["Planned_FAE"]   / denom3
else:
    need = {"Emergency_pct", "Waiting_pct", "Planned_pct"}
    if not need.issubset(df.columns):
        st.error("Missing composition columns; cannot build heatmap.")
        st.stop()

# Build matrix: rows = metrics, columns = years
row_order = ["Emergency percent", "Planned percent", "Waiting percent"]
rename_map = {
    "Emergency percent": "Emergency_pct",
    "Planned percent":   "Planned_pct",
    "Waiting percent":   "Waiting_pct",
}
H = (
    df[["YearLabel", *rename_map.values()]]
    .rename(columns={v: k for k, v in rename_map.items()})
    .melt(id_vars="YearLabel", var_name="Metric", value_name="Percent")
    .pivot(index="Metric", columns="YearLabel", values="Percent")
    .reindex(row_order)
)

# Display mode
val_mode = st.radio(
    "Display", ["Share (level)", "Share index (2019/20=100)", "YoY change"],
    index=0, horizontal=True,
)
mode = val_mode.lower()

# Level / Index / YoY matrices
H_level = H.copy()
base_year = "2019/20" if "2019/20" in H_level.columns else H_level.columns[0]
H_index = H_level.div(H_level[base_year], axis=0) * 100.0
H_yoy   = H_level.pct_change(axis=1) * 100.0

# Color scales & hover
if "level" in mode:
    Z, zmin, zmax = H_level, 0, 0.5
    cscale, cbar  = "Blues", dict(title="Percent", tickformat=".0%")
    hover = "<b>%{y}</b> • %{x}<br>Percent: %{z:.1%}<extra></extra>"
elif "index" in mode:
    Z, zmin, zmax = H_index, 60, 140
    cscale, cbar  = "RdBu_r", dict(title="Index")
    hover = "<b>%{y}</b> • %{x}<br>Index: %{z:.0f}<extra></extra>"
else:
    Z, zmin, zmax = H_yoy, -40, 40
    cscale, cbar  = "RdBu_r", dict(title="YoY (pp)")
    hover = "<b>%{y}</b> • %{x}<br>YoY: %{z:.0f}%<extra></extra>"

# Heatmap
fig = px.imshow(
    Z,
    aspect="auto",
    origin="lower",
    color_continuous_scale=cscale,
    labels=dict(color=cbar["title"]),
    zmin=zmin, zmax=zmax,
)
fig.update_coloraxes(colorbar=cbar)
fig.update_yaxes(categoryorder="array", categoryarray=row_order)
fig.update_traces(hovertemplate=hover)
fig.update_layout(margin=dict(l=140, r=70, t=40, b=50))
fig.update_yaxes(title_text="Admission method", title_standoff=12, automargin=True)
fig.update_xaxes(title_text="Year (Apr–Mar)", automargin=True)

st.plotly_chart(fig, use_container_width=True)

# Exports
export_long = (
    H.reset_index()
     .melt(id_vars="Metric", var_name="YearLabel", value_name="Percent")
     .dropna()
     .sort_values(["YearLabel", "Metric"])
)
st.download_button(
    "Download CSV",
    export_long.to_csv(index=False).encode("utf-8"),
    file_name="heatmap_composition.csv",
    mime="text/csv",
)

try:
    png = fig.to_image(format="png", scale=2)  # requires `pip install -U kaleido`
    st.download_button(
        "Download PNG",
        png,
        file_name="heatmap_composition.png",
        mime="image/png",
    )
except Exception:
    st.caption("Tip: use the Plotly toolbar (camera icon) to download PNG, or install `kaleido` for a button here.")

# Figure caption
st.caption(
    "FAE composition by admission method (Apr–Mar). "
    "Note: 2020/21 includes pandemic-related changes in admission recording; compare with caution."
)
