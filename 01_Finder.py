# pages/01_Finder.py
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(
    page_title="HES Admissions Dashboard — Finder",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from common_header import top_nav
top_nav("Finder", hide_sidebar=True)

HERE = Path(__file__).parent.resolve()
CSV_CODES = HERE / "codes_multi_year.csv"

st.title("Finder — diagnose lockdown drop & rebound")

if not CSV_CODES.exists():
    st.error(
        f"Missing {CSV_CODES.name}. Expected columns: YearLabel, Code, Description, Group, "
        "FAE_total and/or FCE_total."
    )
    st.stop()

raw = pd.read_csv(CSV_CODES)

# ---- Minimum columns & year subset ----
need_cols = {"YearLabel", "Code", "Description", "Group"}
if not need_cols.issubset(raw.columns):
    st.error(f"Missing required columns: {need_cols - set(raw.columns)}")
    st.stop()

# Prefer these three fiscal years if available (logic works with more years as well)
years_pref = ["2019/20", "2020/21", "2023/24"]
years = [y for y in years_pref if y in raw["YearLabel"].unique().tolist()]
if len(years) < 3:
    st.warning(
        "Finder runs best with 2019/20, 2020/21, 2023/24. Proceeding with whatever is available."
    )
raw = raw[raw["YearLabel"].isin(years)].copy()

# ---- Compact controls (replace sidebar) ----
st.markdown(
    """
<style>
  /* Make widgets more compact */
  .stRadio, .stMultiSelect, .stTextInput, .stSlider { margin-bottom: .25rem; }
  [data-testid="stHorizontalBlock"] { gap: .6rem !important; }
</style>
""",
    unsafe_allow_html=True,
)

# Row 1: Metric / Baseline / Drop / Rebound
r1c1, r1c2, r1c3, r1c4 = st.columns([1.4, 1.0, 1.2, 1.2], gap="small")
with r1c1:
    st.caption("Metric")
    metric_default_idx = 0 if "FAE_total" in raw.columns else 1
    metric = st.radio(
        "",
        ["FAE_total", "FCE_total"],
        index=metric_default_idx,
        horizontal=True,
        key="f_metric",
        label_visibility="collapsed",
    )

with r1c2:
    st.caption("Baseline FY")
    baseline = st.selectbox(
        "",
        ["2019/20", "2018/19"],
        index=0,
        key="f_baseline",
        label_visibility="collapsed",
        help="Baseline year for index (affects Δ in the table only).",
    )

with r1c3:
    st.caption("Drop threshold (pp)")
    drop_pp = st.slider(
        "",
        -60,
        -5,
        value=-30,
        step=1,
        key="f_drop",
        label_visibility="collapsed",
    )

with r1c4:
    st.caption("Rebound threshold (pp)")
    rebound_pp = st.slider(
        "",
        5,
        40,
        value=20,
        step=1,
        key="f_rebd",
        label_visibility="collapsed",
    )

groups_all = sorted(raw["Group"].dropna().unique().tolist())

# Row 2: Group filter / Search / Show hits only
r2c1, r2c2, r2c3 = st.columns([1.6, 2.0, 0.8], gap="small")
with r2c1:
    st.caption("Filter groups")
    picked_groups = st.multiselect(
        "", groups_all, default=[], key="f_groups", label_visibility="collapsed"
    )
with r2c2:
    st.caption("Search code/description")
    q = st.text_input(
        "", "", key="f_q", placeholder="e.g. K50 or Crohn", label_visibility="collapsed"
    )
with r2c3:
    st.caption(" ")
    show_only_hits = st.checkbox("Show only hits", value=True, key="f_hits")

# Ensure metric column exists
if metric not in raw.columns:
    st.error(f"Selected metric '{metric}' not in data.")
    st.stop()

# ---- Build index, drop, rebound ----
d = raw.copy()
if picked_groups:
    d = d[d["Group"].isin(picked_groups)]
if q.strip():
    s = q.lower()
    d = d[
        d["Code"].astype(str).str.lower().str.contains(s)
        | d["Description"].astype(str).str.lower().str.contains(s)
    ]

pv = d.pivot_table(
    index=["Group", "Code", "Description"],
    columns="YearLabel",
    values=metric,
    aggfunc="sum",
)

base_year = (
    baseline
    if baseline in pv.columns
    else ("2019/20" if "2019/20" in pv.columns else pv.columns[0])
)
idx = pv.div(pv[base_year], axis=0) * 100

ix19 = idx.get("2019/20")
ix20 = idx.get("2020/21")
ix23 = idx.get("2023/24")

out = pd.DataFrame(
    {
        "Group": pv.index.get_level_values("Group"),
        "Code": pv.index.get_level_values("Code"),
        "Description": pv.index.get_level_values("Description"),
        "Index 19/20": ix19,
        "Index 20/21": ix20,
        "Index 23/24": ix23,
    }
)

base_col = f"Index {base_year}"
out["Drop (pp)"] = out["Index 20/21"] - (out[base_col] if base_col in out.columns else 100)
out["Rebound (pp)"] = out["Index 23/24"] - out["Index 20/21"]
out["Net vs base (pp)"] = out["Index 23/24"] - (out[base_col] if base_col in out.columns else 100)
out["Hit"] = (out["Drop (pp)"] <= drop_pp) & (out["Rebound (pp)"] >= rebound_pp)

out = (
    out.sort_values(["Hit", "Drop (pp)", "Rebound (pp)"], ascending=[False, True, False])
    .reset_index(drop=True)
)

# ==== Ranked candidates ====
st.subheader("Ranked candidates")

show = out.copy()
if show_only_hits:
    show = show[show["Hit"]]

k_choice = st.selectbox("Show top-K", ["20", "50", "100", "All"], index=1)
show_k = show if k_choice == "All" else show.head(int(k_choice))

# Optional: Top-10 hit list above the table
if "Hit" in show.columns:
    top_list = show[show["Hit"]].head(10)
    if len(top_list):
        st.markdown("**Top hits:**")
        for _, r in top_list.iterrows():
            st.markdown(
                f"- **{r['Code']}** — {r['Description']} · "
                f"Drop {r['Drop (pp)']:+.0f}pp, Rebound {r['Rebound (pp)']:+.0f}pp"
            )

# Display formatting
dsp = show_k.copy()
for c in ["Index 19/20", "Index 20/21", "Index 23/24"]:
    if c in dsp.columns:
        dsp[c] = dsp[c].map(lambda v: "" if pd.isna(v) else f"{v:.0f}")
for c in ["Drop (pp)", "Rebound (pp)", "Net vs base (pp)"]:
    if c in dsp.columns:
        dsp[c] = dsp[c].map(lambda v: "" if pd.isna(v) else f"{v:+.0f}")

cols = [
    "Group",
    "Code",
    "Description",
    "Index 19/20",
    "Index 20/21",
    "Index 23/24",
    "Drop (pp)",
    "Rebound (pp)",
    "Net vs base (pp)",
    "Hit",
]
cols = [c for c in cols if c in dsp.columns]

try:
    st.dataframe(dsp[cols], use_container_width=True, hide_index=True)
except Exception as e:
    st.warning(f"Falling back to markdown table due to: {type(e).__name__}")
    st.markdown(dsp[cols].to_markdown(index=False))

# Downloads: (1) shown Top-K, (2) full candidates
st.download_button(
    "Download shown (Top-K) CSV",
    show_k.to_csv(index=False).encode("utf-8"),
    file_name=f"finder_candidates_top{('All' if k_choice == 'All' else k_choice)}.csv",
    mime="text/csv",
)
st.download_button(
    "Download full candidates CSV",
    out.to_csv(index=False).encode("utf-8"),
    file_name="finder_candidates_full.csv",
    mime="text/csv",
)

# ---- Slope previews for selected codes ----
pick_codes = st.multiselect(
    "Preview slope for… (up to 6)", show["Code"].head(10).tolist(), max_selections=6
)

if pick_codes:
    years_plot = [y for y in ["2019/20", "2020/21", "2023/24"] if y in pv.columns]
    fig = make_subplots(
        rows=len(pick_codes),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12 / max(1, len(pick_codes) - 1),
    )

    show_vals = st.toggle("Show point values", True, help="Display values at each point.")
    include_code = st.toggle("Tail label includes code", value=(len(pick_codes) == 1))

    def fmt_idx(v):
        return "" if pd.isna(v) else f"{v:.0f}"

    for i, c in enumerate(pick_codes, start=1):
        row_idx = idx.xs(c, level="Code", drop_level=False)
        yvals = [row_idx.get(y).values[0] if y in row_idx.columns else None for y in years_plot]
        name = f"{c} — {row_idx.index.get_level_values('Description')[0]}"

        # Point labels (drop the last one to avoid colliding with the tail label)
        if show_vals:
            texts = [fmt_idx(v) for v in yvals]
            last_idx = max((j for j, v in enumerate(yvals) if v is not None), default=None)
            if last_idx is not None:
                texts[last_idx] = ""
            pos = ["middle left"] + ["top center"] * (len(years_plot) - 2) + ["middle right"]
            mode = "lines+markers+text"
        else:
            texts, pos, mode = None, None, "lines+markers"
            last_idx = max((j for j, v in enumerate(yvals) if v is not None), default=None)

        fig.add_trace(
            go.Scatter(
                x=years_plot,
                y=yvals,
                name=name,
                showlegend=(len(pick_codes) >= 2),
                mode=mode,
                text=texts,
                textposition=pos,
                textfont=dict(size=11),
                cliponaxis=False,
                line=dict(width=2),
                marker=dict(size=7),
                hovertemplate="%{fullData.name}<br>%{x}: %{y:.0f}<extra></extra>",
            ),
            row=i,
            col=1,
        )

        # (Optional) Δ annotations between points — intentionally omitted here

        # Tail label following the line color
        if last_idx is not None:
            x_end = years_plot[last_idx]
            y_end = yvals[last_idx]
            tail_txt = f"{y_end:.0f}" + (f" • {c}" if include_code else "")
            series_color = fig.data[-1].line.color
            fig.add_annotation(
                x=x_end,
                y=y_end,
                text=tail_txt,
                xanchor="left",
                yanchor="middle",
                xshift=22,
                showarrow=False,
                font=dict(size=12, color=series_color),
                row=i,
                col=1,
            )

    fig.update_traces(cliponaxis=False)
    fig.update_layout(margin=dict(r=120))

    # Y-axis titles per subplot
    for r in range(1, len(pick_codes) + 1):
        fig.update_yaxes(title_text=f"Index ({base_year}=100)", row=r, col=1)

    # Figure title
    if len(pick_codes) == 1:
        _c = pick_codes[0]
        _desc = (
            idx.xs(_c, level="Code", drop_level=False)
            .index.get_level_values("Description")[0]
        )
        title_main = f"{_c} — {_desc}"
    else:
        head = ", ".join(pick_codes[:3])
        more = f" … (+{len(pick_codes) - 3} more)" if len(pick_codes) > 3 else ""
        title_main = f"{len(pick_codes)} codes: {head}{more}"

    subtitle = f"Index ({base_year}=100)"

    PLOT_HEIGHT = 320
    EXPORT_WIDTH = 1200
    EXPORT_SCALE = 2

    fig.update_layout(
        title=dict(
            text=f"{title_main}<br><sup>{subtitle}</sup>",
            x=0.03,
            xanchor="left",
            y=0.90,
            yanchor="top",
            pad=dict(l=10, r=4, t=2, b=0),
        ),
        height=PLOT_HEIGHT,
        margin=dict(l=110, r=28, t=88, b=40),
    )
    fig.update_yaxes(title_text="Index (2019/20=100)", title_standoff=18, automargin=True)

    # Single color for preview lines/markers (you may replace with your token if desired)
    LINE_COLOR = "#0072B2"
    fig.update_traces(line=dict(color=LINE_COLOR, width=2), marker=dict(color=LINE_COLOR, size=7))

    # Render & export
    st.plotly_chart(fig, use_container_width=True)
    try:
        png = fig.to_image(
            format="png", width=EXPORT_WIDTH, height=PLOT_HEIGHT, scale=EXPORT_SCALE
        )
        st.download_button(
            "Download slope PNG", png, file_name="slope_preview.png", mime="image/png"
        )
    except Exception:
        st.caption("Tip: use the Plotly toolbar (camera icon) to download PNG.")
