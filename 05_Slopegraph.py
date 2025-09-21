# pages/05_Slopegraph.py
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from common_header import top_nav

top_nav("Slopegraph")
HERE = Path(__file__).parent.resolve()

# ---------- Read yearly totals ----------
CSV_TOTALS = HERE / "hes_yearly_totals_2015_2024.csv"
df_tot = pd.read_csv(CSV_TOTALS)

# Normalize column names and map variants
df_tot.columns = df_tot.columns.str.strip()
col_alias = {
    "Mean wait": "MeanWait_Days",
    "Mean LOS": "MeanLOS_Days",
    "Mean time waited \n(Days)": "MeanWait_Days",
    "Mean length of stay \n(Days)": "MeanLOS_Days",
}
for a, b in col_alias.items():
    if b not in df_tot.columns and a in df_tot.columns:
        df_tot[b] = pd.to_numeric(df_tot[a], errors="coerce")

# Order by year
if "YearStart" in df_tot.columns:
    df_tot = df_tot.sort_values("YearStart")
years_all = df_tot["YearLabel"].tolist()

# Palettes (consistent with other pages)
PALETTE_RATES = {
    "Emergency share": "#D55E00",
    "Planned share":   "#009E73",
    "Waiting share":   "#F0E442",
}
PALETTE_DAYS = {
    "Mean wait": "#8C564B",
    "Mean LOS":  "#BCBD22",
}

# ---------- Controls (top slopegraph) ----------
st.title("Trends — Slopegraph")

c0, c1, c2 = st.columns([1, 1, 1], gap="small")
with c0:
    st.caption("Start")
    y0 = st.selectbox(
        "Start",
        years_all,
        index=(years_all.index("2019/20") if "2019/20" in years_all else 0),
        key="slope_start",
        label_visibility="collapsed",
    )
with c1:
    st.caption("Mid")
    y1 = st.selectbox(
        "Mid",
        years_all,
        index=(years_all.index("2020/21") if "2020/21" in years_all else 1),
        key="slope_mid",
        label_visibility="collapsed",
    )
with c2:
    st.caption("End")
    y2 = st.selectbox(
        "End",
        years_all,
        index=len(years_all) - 1,
        key="slope_end",
        label_visibility="collapsed",
    )

# Enforce strictly increasing years
_order = {y: i for i, y in enumerate(years_all)}
if not (_order[y0] < _order[y1] < _order[y2]):
    y0, y1, y2 = sorted([y0, y1, y2], key=_order.get)

# View toggle (affects the large slopegraph only)
series_kind = st.radio(
    "View",
    ["Rates (E/W/P)", "Days (Wait/LOS)"],
    index=0,
    horizontal=True,
    key="slope_series_kind",
    help="Controls the large slopegraph above only.",
)

# Ensure E/W/P shares exist in totals; derive if needed
need_pct = {"Emergency_pct", "Planned_pct", "Waiting_pct"}
if not need_pct.issubset(df_tot.columns):
    have_parts = {"Emergency_FAE", "Waiting_FAE", "Planned_FAE"}.issubset(df_tot.columns)
    if have_parts:
        denom = (
            df_tot["Emergency_FAE"] + df_tot["Waiting_FAE"] + df_tot["Planned_FAE"]
        ).replace(0, pd.NA)
        df_tot["Emergency_pct"] = df_tot["Emergency_FAE"] / denom
        df_tot["Waiting_pct"] = df_tot["Waiting_FAE"] / denom
        df_tot["Planned_pct"] = df_tot["Planned_FAE"] / denom

# Pick series for the top slopegraph
if series_kind.startswith("Rates"):
    cols = ["Emergency_pct", "Planned_pct", "Waiting_pct"]
    names = ["Emergency share", "Planned share", "Waiting share"]
    fmt = "pct"
    if not set(cols).issubset(df_tot.columns):
        st.error("Missing inputs to compute E/W/P shares for the totals table.")
        st.stop()
else:
    cols = ["MeanWait_Days", "MeanLOS_Days"]
    names = ["Mean wait", "Mean LOS"]
    fmt = "days"
    if not set(cols).issubset(df_tot.columns):
        st.error("Missing mean wait / mean LOS columns in totals table.")
        st.stop()

# Extract three time points
pick_years = [y for y in [y0, y1, y2] if y in years_all]
Z = (
    df_tot[df_tot["YearLabel"].isin(pick_years)][["YearLabel"] + cols]
    .set_index("YearLabel")
    .T.rename(index=dict(zip(cols, names)))
)

# ---------- Top slopegraph ----------
xpos = list(range(Z.shape[1]))
xlabels = list(Z.columns)
fig0 = go.Figure()
palette = PALETTE_RATES if series_kind.startswith("Rates") else PALETTE_DAYS
is_pct = series_kind.startswith("Rates")

for r in Z.index:
    ys = Z.loc[r].values.astype(float)
    texts = [f"{v:.1%}" if is_pct else f"{v:.1f}" for v in ys]
    pos = ["middle left"] + ["top center"] * (len(xpos) - 2) + ["middle right"]
    colr = palette.get(r, "#1f77b4")
    fig0.add_trace(
        go.Scatter(
            x=xpos,
            y=ys,
            mode="lines+markers+text",
            name=r,
            text=texts,
            textposition=pos,
            textfont=dict(size=11),
            cliponaxis=False,
            hovertemplate="%{fullData.name}<br>%{x}: "
            + ("%{y:.1%}" if is_pct else "%{y:.1f}")
            + "<extra></extra>",
            line=dict(width=2, color=colr),
            marker=dict(size=7, color=colr),
        )
    )

# Δ annotations between adjacent points
fmt_d = (lambda d: f"{d:+.1f} pp") if is_pct else (lambda d: f"{d:+.1f}")
for k, tr in enumerate(fig0.data):
    y0_, y1_, y2_ = tr.y[0], tr.y[1], tr.y[2]
    for (xa, xb, ya, yb) in [(xpos[0], xpos[1], y0_, y1_), (xpos[1], xpos[2], y1_, y2_)]:
        d = (yb - ya)
        if abs(d) < (0.001 if is_pct else 0.1):
            continue
        xm, ym = (xa + xb) / 2, (ya + yb) / 2
        fig0.add_annotation(
            x=xm,
            y=ym,
            text=fmt_d(d),
            showarrow=True,
            arrowcolor=tr.line.color or "#777",
            arrowwidth=1,
            arrowsize=0.6,
            ax=0,
            ay=-12 + (k % 3 - 1) * 6,
            bgcolor="rgba(255,255,255,0.8)",
            font=dict(size=11, color=tr.line.color or None),
        )

fig0.update_layout(
    xaxis=dict(tickmode="array", tickvals=xpos, ticktext=xlabels),
    yaxis=dict(
        title=("Rate" if series_kind.startswith("Rates") else "Days"),
        range=[0, 0.6] if series_kind.startswith("Rates") else None,
    ),
    legend=dict(orientation="h", y=-0.22, yanchor="top", x=0, xanchor="left"),
    margin=dict(l=40, r=140, t=40, b=80),
)

# Right-end value labels
valfmt = (lambda v: f"{v:.1%}") if is_pct else (lambda v: f"{v:.1f}")
fig0.update_traces(text=None, texttemplate=None, mode="lines+markers", selector=dict(type="scatter"))
xmin, xmax = min(xpos), max(xpos)
fig0.update_xaxes(range=[xmin - 0.08, xmax + 0.13])
for tr in fig0.data:
    tr.update(cliponaxis=False)

right_y = [tr.y[-1] for tr in fig0.data]
order = sorted(range(len(right_y)), key=lambda i: right_y[i])
min_gap = 0.018 if is_pct else 1.2
groups, cur = [], ([order[0]] if right_y else [])
for a, b in zip(order, order[1:]):
    if abs(right_y[b] - right_y[a]) <= min_gap:
        cur.append(b)
    else:
        groups.append(cur)
        cur = [b]
if cur:
    groups.append(cur)

step_px, ax_off = 14, 36
for g in groups:
    k = len(g)
    base = [i - (k - 1) / 2 for i in range(k)]
    for r, idx in enumerate(g):
        tr, yv = fig0.data[idx], right_y[idx]
        axi = ax_off if (r % 2 == 0) else -ax_off
        anch = "left" if axi > 0 else "right"
        fig0.add_annotation(
            x=xpos[-1],
            y=yv,
            text=valfmt(yv),
            showarrow=True,
            arrowcolor=tr.line.color or "#777",
            arrowwidth=1,
            arrowsize=0.7,
            ax=axi,
            ay=int(base[r] * step_px),
            xanchor=anch,
            bgcolor="rgba(255,255,255,0)",
        )

st.plotly_chart(fig0, use_container_width=True)
st.caption(
    "Top: slopegraph comparing selected NHS years. Rates are E/W/P shares; days include mean wait and mean length of stay."
)

# ---------- Code-level multi-year table ----------
CSV_CODES = HERE / "codes_multi_year.csv"
df_codes = pd.read_csv(CSV_CODES)
df_codes.columns = df_codes.columns.str.strip()

YSEQ = [y for y in ["2019/20", "2020/21", "2023/24"] if y in df_codes["YearLabel"].unique().tolist()]
if not YSEQ:
    st.info("No overlapping years in codes_multi_year.csv for 2019/20, 2020/21, 2023/24.")
    st.stop()

df_codes = df_codes[df_codes["YearLabel"].isin(YSEQ)].copy()

# Sidebar controls for code-level slopegraphs
HAS_EWP = {"FAE_emergency", "FAE_waiting", "FAE_planned"}.issubset(df_codes.columns)
slope_options = ["Index (FAE total)"] + (["Shares (E/W/P)"] if HAS_EWP else [])
_prev = st.session_state.get("slope_code_kind", slope_options[0])

row_ctl = st.columns([2, 5], gap="large")
with row_ctl[0]:
    slope_kind = st.radio(
        "Code slope uses",
        slope_options,
        index=(slope_options.index(_prev) if _prev in slope_options else 0),
        key="slope_code_kind",
        horizontal=True,
        help="This only affects the code-level panels below.",
    )

with row_ctl[1]:
    all_codes = sorted(df_codes["Code"].dropna().unique().tolist())
    pick = st.multiselect(
        "Pick up to 6 codes",
        options=all_codes[:12],
        default=[],
        max_selections=6,
        key="slope_codes_select",
    )
    if st.button("Quick fill top 6", key="slope_pick_top6"):
        st.session_state["slope_codes_select"] = all_codes[:6]

if not pick:
    st.info("Pick some codes to preview slopes.")
    st.stop()

if (not HAS_EWP) and _prev.startswith("Shares"):
    st.info("FAE_emergency / FAE_waiting / FAE_planned not found; falling back to Index.")

st.caption("Below: code-level slope panels. The ‘Code slope uses’ toggle above only affects these panels.")

# ---------- Code-level slopegraphs ----------
fig_rows = len(pick)

# Prepare per-row titles and cached subsets
row_titles, subs = [], []
for code in pick:
    sub = df_codes.loc[(df_codes["Code"] == code) & (df_codes["YearLabel"].isin(YSEQ))].copy()
    sub = sub.sort_values("YearLabel")
    subs.append(sub)
    desc = sub["Description"].dropna().iloc[0] if len(sub) else ""
    row_titles.append(f"{code} — {desc}")

ROW_HEIGHT = 280
ROW_SPACING = 0.22

fig = make_subplots(
    rows=len(pick), cols=1, shared_xaxes=True,
    vertical_spacing=ROW_SPACING / max(1, len(pick) - 1),
    subplot_titles=row_titles
)

for i, (code, sub) in enumerate(zip(pick, subs), start=1):
    if slope_kind.startswith("Index"):
        # Index (base = 2019/20 if present; otherwise the first available year)
        base = sub.loc[
            sub["YearLabel"] == ("2019/20" if "2019/20" in sub["YearLabel"].values else sub["YearLabel"].iloc[0]),
            "FAE_total",
        ]
        base = float(base.iloc[0]) if len(base) else None

        yvals = []
        for y in YSEQ:
            v = sub.loc[sub["YearLabel"] == y, "FAE_total"]
            v = float(v.iloc[0]) if len(v) else None
            yvals.append(None if (v is None or not base or base == 0) else v / base * 100)

        COLOR_IDX = "#1f77b4"
        fig.add_trace(
            go.Scatter(
                x=YSEQ, y=yvals, mode="lines+markers",
                line=dict(width=2, color=COLOR_IDX),
                marker=dict(size=7, color=COLOR_IDX),
                name="Index (FAE total)", showlegend=(i == 1), legendgroup="INDEX",
            ),
            row=i, col=1
        )

        # Δ annotations
        def ann_idx(x0, x1):
            if x0 not in YSEQ or x1 not in YSEQ:
                return
            j0, j1 = YSEQ.index(x0), YSEQ.index(x1)
            if None in (yvals[j0], yvals[j1]):
                return
            v0, v1 = yvals[j0], yvals[j1]
            fig.add_annotation(
                x=x1, y=(v0 + v1) / 2, text=f"Δ {v1 - v0:+.0f}",
                showarrow=False, bgcolor="rgba(255,255,255,.7)",
                bordercolor="rgba(0,0,0,.2)", borderwidth=1,
                font=dict(color=COLOR_IDX),
                row=i, col=1
            )

        if "2019/20" in YSEQ and "2020/21" in YSEQ:
            ann_idx("2019/20", "2020/21")
        if "2020/21" in YSEQ and "2023/24" in YSEQ:
            ann_idx("2020/21", "2023/24")

    else:
        # Shares (E/W/P)
        MAP = {"Emergency": "FAE_emergency", "Planned": "FAE_planned", "Waiting": "FAE_waiting"}
        ORDER = ["Emergency", "Planned", "Waiting"]
        NAMES = {"Emergency": "Emergency share", "Planned": "Planned share", "Waiting": "Waiting share"}
        COLORS = {"Emergency": "#D55E00", "Planned": "#009E73", "Waiting": "#F0E442"}

        sub["_den"] = (
            sub.get("FAE_emergency", 0) + sub.get("FAE_planned", 0) + sub.get("FAE_waiting", 0) + sub.get("FAE_other", 0)
        ).replace(0, pd.NA)

        for part, col in MAP.items():
            num = sub.get(col, 0)
            sub[f"{part}_pct"] = (num / sub["_den"]).astype("float")

        if sub[["Emergency_pct", "Planned_pct", "Waiting_pct"]].isna().all().all():
            fig.add_annotation(
                x=YSEQ[-1], y=0, text=f"{code} — no E/W/P split in CSV",
                showarrow=False, row=i, col=1
            )
            continue

        for part in ORDER:
            yvals = []
            for y in YSEQ:
                v = sub.loc[sub["YearLabel"] == y, f"{part}_pct"]
                v = float(v.iloc[0]) if len(v) and pd.notna(v.iloc[0]) else None
                yvals.append(None if v is None else v * 100)

            fig.add_trace(
                go.Scatter(
                    x=YSEQ, y=yvals, mode="lines+markers",
                    line=dict(width=2, color=COLORS[part]),
                    marker=dict(size=7, color=COLORS[part]),
                    name=NAMES[part], legendgroup="EWP", showlegend=(i == 1),
                ),
                row=i, col=1
            )

            def ann_share(x0, x1):
                if x0 not in YSEQ or x1 not in YSEQ:
                    return
                j0, j1 = YSEQ.index(x0), YSEQ.index(x1)
                if None in (yvals[j0], yvals[j1]):
                    return
                v0, v1 = yvals[j0], yvals[j1]
                fig.add_annotation(
                    x=x1, y=(v0 + v1) / 2, text=f"{part} Δ {v1 - v0:+.0f} pp",
                    showarrow=False, bgcolor="rgba(255,255,255,.7)",
                    bordercolor="rgba(0,0,0,.2)", borderwidth=1,
                    font=dict(color=COLORS[part]),
                    row=i, col=1
                )

            if "2019/20" in YSEQ and "2020/21" in YSEQ:
                ann_share("2019/20", "2020/21")
            if "2020/21" in YSEQ and "2023/24" in YSEQ:
                ann_share("2020/21", "2023/24")

# Unified layout
fig.update_layout(height=ROW_HEIGHT * len(pick), hovermode="x unified")
fig.update_xaxes(automargin=True)
fig.update_yaxes(
    title_text=("Index (2019/20 = 100)" if slope_kind.startswith("Index") else "Share (percentage points)")
)

# Slightly lift the first row title
if row_titles:
    fig.update_annotations(selector=dict(text=row_titles[0]), yshift=10)

st.plotly_chart(fig, use_container_width=True)

# Export PNG (optional; requires Kaleido)
try:
    export_w = 1200
    export_h = ROW_HEIGHT * len(pick)
    png = fig.to_image(format="png", width=export_w, height=export_h, scale=2)
    st.download_button(
        "Download code-slope PNG",
        png,
        file_name=("code_slope_index.png" if slope_kind.startswith("Index") else "code_slope_shares.png"),
        mime="image/png",
    )
except Exception:
    st.caption("Tip: you can also use the camera icon in the Plotly toolbar to download a PNG.")
