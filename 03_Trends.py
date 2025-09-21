# pages/03_Trends.py — HES · Multi-year trends (English-only, cleaned)

import os
os.environ["STREAMLIT_DATAFRAME_SERIALIZATION"] = "legacy"

from pathlib import Path
import importlib.util, math
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from common_header import top_nav

top_nav("Trends")

# ---------- Load/build CSV (fallback to local data_builder.py if missing) ----------
HERE = Path(__file__).parent.resolve()
CSV_PATH = HERE / "hes_yearly_totals_2015_2024.csv"
BUILDER = HERE / "data_builder.py"

if not CSV_PATH.exists():
    if not BUILDER.exists():
        st.error("Missing CSV and data_builder.py not found.")
        st.stop()
    spec = importlib.util.spec_from_file_location("data_builder", str(BUILDER))
    db = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(db)
    out_csv, _ = db.build_yearly_csv(
        base_folder=str(HERE),
        out_csv=CSV_PATH.name,
        audit_csv="hes_yearly_audit.csv",
    )
    CSV_PATH = HERE / out_csv

# ---------- Header row (title + quick links) ----------
hdr = st.container()
with hdr:
    try:
        c_title, c_heat, c_slope, c_pad = st.columns([0.56, 0.17, 0.20, 0.08], gap="small", vertical_alignment="center")
    except TypeError:
        c_title, c_heat, c_slope, c_pad = st.columns([0.56, 0.17, 0.20, 0.08], gap="small")

    with c_title:
        st.title("HES · Multi-year trends")
    with c_heat:
        st.page_link("pages/04_Heatmap.py", label="Open Heatmap →")
    with c_slope:
        st.page_link("pages/05_Slopegraph.py", label="Open Slopegraph →")
    with c_pad:
        st.write("")

df = pd.read_csv(CSV_PATH).sort_values("YearStart").reset_index(drop=True)

# ---------- Ensure FAE_total exists (reconstruct from components if needed) ----------
if "FAE_total" not in df.columns:
    comp = ["Emergency_FAE", "Waiting_FAE", "Planned_FAE", "Other_FAE"]
    if set(comp).issubset(df.columns):
        df["FAE_total"] = df[comp].sum(axis=1)
        st.caption("Reconstructed FAE_total = Emergency + Waiting + Planned + Other.")
    else:
        st.warning("FAE_total missing and components incomplete; 'FAE total' series will be hidden.")

# ---------- Derive shares if absent ----------
if {"Emergency_FAE", "FAE_total"}.issubset(df.columns) and "Emergency_pct" not in df:
    df["Emergency_pct"] = df["Emergency_FAE"] / df["FAE_total"]
if {"Waiting_FAE", "FAE_total"}.issubset(df.columns) and "Waiting_pct" not in df:
    df["Waiting_pct"] = df["Waiting_FAE"] / df["FAE_total"]
if {"Planned_FAE", "FAE_total"}.issubset(df.columns) and "Planned_pct" not in df:
    df["Planned_pct"] = df["Planned_FAE"] / df["FAE_total"]

# ---------- Year range ----------
labels = df["YearLabel"].tolist()
start_label, end_label = st.select_slider(
    "Year range",
    options=labels,
    value=(labels[0], labels[-1]),
)
i0, i1 = labels.index(start_label), labels.index(end_label)
dfv = df.iloc[i0:i1 + 1].copy()

# Recompute YoY on filtered view (counts only)
for c in ["FCE", "FAE_total"]:
    if c in dfv.columns:
        dfv[f"{c}_yoy"] = dfv[c].pct_change()

# ---------- Color tokens for this page ----------
COL = {
    # totals
    "FAE": "#000000",        # FAE total — black
    "FCE": "#FF9DA7",        # FCE total — soft pink
    # shares by admission type
    "Emergency_pct": "#D55E00",
    "Waiting_pct":   "#F0E442",
    "Planned_pct":   "#009E73",
    # gender shares
    "Male_Share":    "#0074B2",
    "Female_Share":  "#DB4898",
    # timeliness (days)
    "MeanWait_Days":   "#8C564B",
    "MedianWait_Days": "#7F7F7F",
    "MeanLOS_Days":    "#BCBD22",
    "MedianLOS_Days":  "#17BECF",
}
# If you prefer the legacy blue for FCE, set: COL["FCE"] = "#1F77B4"

# ---------- Series registry ----------
SERIES = {
    # counts (left axis)
    "FCE total": {"col": "FCE", "fmt": "count", "axis": "left", "color": COL["FCE"]},
    "FAE total": {"col": "FAE_total", "fmt": "count", "axis": "left", "color": COL["FAE"]},
    # shares (right axis when View=Rates)
    "Emergency share": {"col": "Emergency_pct", "fmt": "pct", "axis": "right", "color": COL["Emergency_pct"]},
    "Waiting share":   {"col": "Waiting_pct",   "fmt": "pct", "axis": "right", "color": COL["Waiting_pct"]},
    "Planned share":   {"col": "Planned_pct",   "fmt": "pct", "axis": "right", "color": COL["Planned_pct"]},
    "Male share":      {"col": "Male_Share",    "fmt": "pct", "axis": "right", "color": COL["Male_Share"]},
    "Female share":    {"col": "Female_Share",  "fmt": "pct", "axis": "right", "color": COL["Female_Share"]},
    # days (right axis when View=Days)
    "Mean wait":   {"col": "MeanWait_Days",   "fmt": "days", "axis": "right", "color": COL["MeanWait_Days"]},
    "Median wait": {"col": "MedianWait_Days", "fmt": "days", "axis": "right", "color": COL["MedianWait_Days"]},
    "Mean LOS":    {"col": "MeanLOS_Days",    "fmt": "days", "axis": "right", "color": COL["MeanLOS_Days"]},
    "Median LOS":  {"col": "MedianLOS_Days",  "fmt": "days", "axis": "right", "color": COL["MedianLOS_Days"]},
}

AVAILABLE = {name: meta for name, meta in SERIES.items() if meta["col"] in dfv.columns}
SERIES_ALL = {
    "Rates": [k for k, m in AVAILABLE.items() if m["fmt"] == "pct"],
    "Days":  [k for k in ["FCE total", "FAE total", "Mean wait", "Median wait", "Mean LOS", "Median LOS"] if k in AVAILABLE],
}

# ---------- UI: view, structure, pick series ----------
colL, colR = st.columns([3, 2])
with colL:
    axis_mode = st.radio("View", ["Rates (shares)", "Days"], horizontal=True, index=0)

structure = "Admission"
with colR:
    if axis_mode.startswith("Rates"):
        structure = st.radio("Structure", ["Admission", "Gender"], horizontal=True, index=0, help="Choose which shares to focus on")

axis_is_rates = axis_mode.startswith("Rates")

# Default combinations
if axis_is_rates:
    basic = (["Emergency share", "Planned share", "Waiting share"] if structure == "Admission"
             else ["Male share", "Female share"])
else:
    basic = ["FAE total", "Mean wait"]

available_list = SERIES_ALL["Rates" if axis_is_rates else "Days"]
options_basic = [s for s in basic if s in available_list]
key_suffix = "rates" if axis_is_rates else "days"

c_sel, c_extra = st.columns([3, 2], gap="large")
with c_sel:
    picked_basic = st.multiselect(
        "Select series",
        options_basic,
        default=options_basic,
        key=f"series_basic_{key_suffix}",
    )
with c_extra:
    pool = [s for s in available_list if s not in options_basic]
    extra = st.multiselect(
        "Add extra metrics (optional)",
        pool,
        default=[],
        key=f"series_extra_{key_suffix}",
    )

picked = picked_basic + [m for m in extra if m not in picked_basic]
if len(picked) > 4:
    st.caption(f"Showing {len(picked)} series — consider ≤4 for readability.")

row = st.columns([1, 1], gap="small")
with row[0]:
    overlay_volume = st.checkbox("Overlay FAE total on right axis", value=False, key="ovl")
with row[1]:
    show_delta = st.toggle("Show 3-point Δ labels", value=True, key="dlt")

fig_slot = st.empty()

# Protect against mixing totals with shares unless overlay is enabled
if axis_is_rates and "FAE total" in picked and not overlay_volume:
    picked.remove("FAE total")
    st.caption("Tip: enable overlay to show totals with shares (right axis).")

# ---------- Summary table controls ----------
col_pos, col_scope = st.columns([1, 1], gap="large")
with col_pos:
    table_pos = st.radio("Summary table layout", ["Below", "Side-by-side"], horizontal=True, index=0, key="tbl_pos")
with col_scope:
    table_scope = st.radio("Table scope", ["Latest year", "Last 3 years", "All years"], horizontal=True, index=2, key="tbl_scope")

# ---------- Hover template ----------
def _hover(fmt: str, show_yoy: bool = False) -> str:
    if fmt == "pct":
        main = "Rate: %{y:.1%}"
    elif fmt == "days":
        main = "Days: %{y:.1f}"
    else:
        main = "Count: %{y:,}"
    yoy = "<br>YoY: %{customdata:+.1%}" if show_yoy else ""
    return "<b>%{fullData.name}</b><br>" + main + yoy + "<extra></extra>"

# ---------- Δ annotations helper ----------
def _annotate_deltas(fig, xs, ys, label, axis_id="y"):
    """Add Δ between adjacent points at midpoints. axis_id: 'y' (left) or 'y2' (right)."""
    for (x0, x1, y0, y1) in zip(xs[:-1], xs[1:], ys[:-1], ys[1:]):
        if pd.isna(y0) or pd.isna(y1):
            continue
        delta = (y1 - y0)
        txt   = f"Δ {delta*100:+.1f} pp" if axis_id == "y" else f"Δ {delta:+.1f} days"
        i0, i1 = xs.index(x0), xs.index(x1)
        xm = xs[(i0 + i1) // 2]
        ym = (y0 + y1) / 2
        fig.add_annotation(
            x=xm, y=ym, xref="x", yref=axis_id,
            text=txt, showarrow=False,
            bgcolor="rgba(255,255,255,.75)", font=dict(size=10)
        )

def _nice_join(items):
    if not items:
        return ""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " & " + items[-1]

def _build_title(axis_is_rates: bool, structure: str, picked: list[str],
                 start_label: str, end_label: str, overlay_volume: bool) -> tuple[str, str]:
    years = f"{start_label}–{end_label}"
    if axis_is_rates:
        main = ("Admissions — shares by admission type" if structure == "Admission" else "Admissions — gender shares")
        sub = f"NHS years {years}" + (" • shares (right) with FAE total overlay" if overlay_volume else "")
    else:
        names = [n.replace("share", "").strip() for n in picked]
        main = f"Admissions — {_nice_join(names)}"
        sub = f"NHS years {years}"
    return main, sub

# ---------- Build figure ----------
fig = make_subplots(specs=[[{"secondary_y": True}]])
X = dfv["YearLabel"]
years_list = X.tolist()

to_plot = picked.copy()
if axis_is_rates and overlay_volume and "FAE total" in SERIES and "FAE total" not in to_plot:
    to_plot.append("FAE total")

added = set()
for name in to_plot:
    s = SERIES[name]
    col = s["col"]; fmt = s["fmt"]
    if col not in dfv.columns:
        continue

    show_yoy = (fmt == "count") and (f"{col}_yoy" in dfv.columns)
    custom = dfv[f"{col}_yoy"].to_numpy() if show_yoy else None

    is_overlay_fae = axis_is_rates and overlay_volume and (name == "FAE total")
    sec_right = is_overlay_fae if axis_is_rates else (s["axis"] == "right")

    line_style = dict(width=2, color=s["color"], dash=("dash" if is_overlay_fae else None))
    yvals = dfv[col].tolist()

    fig.add_trace(
        go.Scatter(
            x=X, y=dfv[col], mode="lines+markers",
            line=line_style, marker=dict(size=6),
            name=name, legendgroup=name, showlegend=(name not in added),
            hovertemplate=_hover(fmt, show_yoy=show_yoy), customdata=custom
        ),
        secondary_y=sec_right
    )

    if show_delta:
        if axis_is_rates and fmt == "pct":
            _annotate_deltas(fig, years_list, yvals, name, axis_id="y")
        elif (not axis_is_rates) and fmt == "days":
            _annotate_deltas(fig, years_list, yvals, name, axis_id="y2")

    added.add(name)

# ---------- End labels (value + Δ from first) ----------
def _fmt_value(v, fmt):
    if pd.isna(v): return "–"
    return f"{v:,.0f}" if fmt=="count" else (f"{v:.1%}" if fmt=="pct" else f"{v:.1f}")

def _fmt_delta(a, b, fmt):
    if pd.isna(a) or pd.isna(b): return "–"
    if fmt=="pct":
        return f"{(b-a)*100:+.1f} pp"
    return f"{(b-a):+,.0f}" if fmt=="count" else f"{(b-a):+.1f}"

def _add_end_labels(fig, traces_info, axis_id="y"):
    """Add end value and Start→End Δ; simple y-shift to reduce overlap."""
    traces_info = sorted(traces_info, key=lambda t: t[4])
    last_y = None; level = 0
    for (name, fmt, colr, x_last, y_last, y_first) in traces_info:
        if pd.isna(y_last):
            continue
        if last_y is not None:
            thr = 0.002 if fmt=="pct" else (0.2 if fmt=="days" else max(50000, abs(y_last)*0.002))
            level = (level + 1) if abs(y_last - last_y) < thr else 0
        yshift = ([-20, -10, 0, 10, 20])[min(level, 4)]
        last_y = y_last

        fig.add_annotation(
            x=x_last, y=y_last, xanchor="left", yanchor="middle",
            xshift=6, yshift=yshift, xref="x", yref=axis_id,
            text=f"{name}: {_fmt_value(y_last, fmt)}",
            font=dict(size=11, color=colr),
            bgcolor="rgba(255,255,255,.65)", bordercolor=colr, borderwidth=0,
            showarrow=False
        )
        fig.add_annotation(
            x=x_last, y=y_last, xanchor="left", yanchor="middle",
            xshift=6, yshift=yshift+16, xref="x", yref=axis_id,
            text=f"Δ {_fmt_delta(y_first, y_last, fmt)}",
            font=dict(size=10, color=colr),
            bgcolor="rgba(255,255,255,.65)", showarrow=False
        )

x_last  = X.iloc[-1]
x_first = X.iloc[0]
info_left, info_right = [], []
for name in to_plot:
    meta = SERIES[name]
    col, fmt, colr = meta["col"], meta["fmt"], meta["color"]
    if col not in dfv:
        continue
    y0 = dfv.loc[dfv["YearLabel"]==x_first, col].iloc[0]
    y1 = dfv.loc[dfv["YearLabel"]==x_last,  col].iloc[0]

    is_overlay_fae = axis_is_rates and overlay_volume and (name == "FAE total")
    use_right = (is_overlay_fae if axis_is_rates else (meta["axis"] == "right"))
    (info_right if use_right else info_left).append((name, fmt, colr, x_last, y1, y0))

if info_left:
    _add_end_labels(fig, info_left, axis_id="y")
if info_right:
    _add_end_labels(fig, info_right, axis_id="y2")

# ---------- Axes ----------
has_count = any(SERIES[name]["fmt"] == "count" for name in to_plot)
has_days  = any(SERIES[name]["fmt"] == "days" for name in to_plot)

if axis_is_rates:
    fig.update_yaxes(title_text="Rate", tickformat=".0%", secondary_y=False)
    fig.update_yaxes(
        title_text=("FAE total" if ("FAE total" in to_plot) else ""),
        tickformat=("~s" if ("FAE total" in to_plot) else None),
        secondary_y=True, showgrid=False
    )
else:
    fig.update_yaxes(
        title_text=("Count" if has_count else ""),
        tickformat=("~s" if has_count else None),
        secondary_y=False
    )
    if has_days:
        days_cols = [SERIES[n]["col"] for n in to_plot if SERIES[n]["fmt"] == "days" and SERIES[n]["col"] in dfv]
        max_days = float(pd.concat([dfv[c] for c in days_cols], axis=1).max().max())
        upper = max_days * 1.1
        fig.update_yaxes(title_text="Days", secondary_y=True, rangemode="tozero", range=[0, upper], tickformat=".1f", showgrid=False)

# ---------- Title/legend/layout ----------
LEGEND_PER_ROW = 5
legend_rows = max(1, math.ceil(max(1, len(picked)) / LEGEND_PER_ROW))
TOP_MARGIN = 88
BOTTOM_MARGIN = 32 + 26 * legend_rows
LEGEND_Y = -0.22

t_main, t_sub = _build_title(axis_is_rates, structure, picked, start_label, end_label, overlay_volume)
fig.update_layout(
    title=dict(text=f"{t_main}<br><sup>{t_sub}</sup>", x=0.01, xanchor="left"),
    title_font=dict(size=20),
    legend=dict(orientation="h", y=LEGEND_Y, yanchor="top", x=0, xanchor="left"),
    margin=dict(l=12, r=8, t=TOP_MARGIN, b=BOTTOM_MARGIN),
    hovermode="x unified"
)

# ---------- Render ----------
fig_slot.empty()
with fig_slot.container():
    st.plotly_chart(fig, use_container_width=True)

# ---------- Summary table ----------
def _format_table(df_show):
    rename_map = {
        "YearLabel": "Year",
        "FCE": "FCE",
        "FAE_total": "FAE",
        "Emergency_pct": "Emergency",
        "Waiting_pct": "Waiting",
        "Planned_pct": "Planned",
        "Male_Share": "Male",
        "Female_Share": "Female",
        "MeanWait_Days": "Mean wait",
        "MedianWait_Days": "Median wait",
        "MeanLOS_Days": "Mean LOS",
        "MedianLOS_Days": "Median LOS",
    }
    out = df_show.rename(columns=rename_map)
    for c in out.columns:
        if c in ["Emergency", "Waiting", "Planned", "Male", "Female"]:
            out[c] = (out[c] * 100).map(lambda v: f"{v:.1f}%")
        elif c in ["FCE", "FAE"]:
            out[c] = out[c].map(lambda v: f"{v:,.0f}")
        elif c in ["Mean wait", "Median wait", "Mean LOS", "Median LOS"]:
            out[c] = out[c].map(lambda v: f"{v:.1f}")
    return out

if table_scope == "Latest year":
    df_show = dfv.iloc[[-1]]
elif table_scope == "Last 3 years":
    df_show = dfv.tail(3)
else:
    df_show = dfv

show_cols = ["YearLabel"] + [SERIES[n]["col"] for n in picked if SERIES[n]["col"] in dfv.columns]
df_small = _format_table(df_show[show_cols])

if table_pos == "Side-by-side":
    fig_slot.empty()
    col_fig, col_gap, col_tbl = st.columns([8, 0.8, 6])
    with col_fig:
        fig.update_layout(margin=dict(l=8, r=6, t=TOP_MARGIN, b=BOTTOM_MARGIN))
        st.plotly_chart(fig, use_container_width=True)
    with col_gap:
        st.empty()
    with col_tbl:
        st.dataframe(df_small, use_container_width=True,
                     height=min(440, 44 * (len(df_small) + 1)), hide_index=True)
        st.caption("Source: NHS HES · Primary Diagnosis Summary (Total row). NHS year = Apr–Mar.")
else:
    st.dataframe(df_small, use_container_width=True,
                 height=min(360, 44 * (len(df_small) + 1)), hide_index=True)
    st.caption("Source: NHS HES · Primary Diagnosis Summary (Total row). NHS year = Apr–Mar.")
