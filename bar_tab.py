# bar_tab.py — Controls + Render for Bar tab (used by pages/06_Explore.py)
from __future__ import annotations

import math
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ===== Orders & Palette =====
GENDER_ORDER = ["Male", "Female", "Unknown"]
ADMISSION_ORDER = ["Emergency", "Waiting list", "Planned", "Other"]

COLOR_DISCRETE_MAP = {
    "Emergency":     "#D55E00",  # Vermillion
    "Waiting list":  "#F0E442",  # Mustard Yellow
    "Planned":       "#009E73",  # Teal-green
    "Other":         "#56B4E9",  # Sky blue
    "Male":          "#0074B2",  # Navy blue
    "Female":        "#DB4898",  # Purple
    "Unknown":       "#999999",  # Neutral gray
}

# ===== Helpers =====
def _axis_label(metric_total: str) -> str:
    return {"FAE_total": "FAE", "FCE_total": "FCE"}.get(metric_total, "Count")

def _ensure_code3(s):
    s = str(s) if pd.notna(s) else ""
    return s if "." not in s else s.split(".")[0]

def _pick_level_col(df: pd.DataFrame, want_ui: str) -> Tuple[str, str]:
    """
    Return the actual column for the requested UI level, with graceful fallback.
    Returns (ui_label, col_name).
    """
    prefer = {"Group": "Group", "3-char Code": "Code3", "Code4": "Code4"}
    ui = want_ui if want_ui in prefer else "3-char Code"
    col = prefer[ui]

    # Ensure Code3 exists; fallback if needed
    if col == "Code3" and "Code3" not in df.columns:
        if "Code" in df.columns:
            df["Code3"] = df["Code"].map(_ensure_code3)
        elif "Code4" in df.columns:
            df["Code3"] = df["Code4"].astype(str).str.replace(".", "", regex=False).str[:3]
    if col == "Code4" and "Code4" not in df.columns:
        ui, col = "3-char Code", "Code3"

    # Fallback for Group/Chapter
    if col == "Group" and "Group" not in df.columns:
        if "Chapter" in df.columns:
            ui, col = "Chapter", "Chapter"
        else:
            if "Code3" not in df.columns and "Code" in df.columns:
                df["Code3"] = df["Code"].map(_ensure_code3)
            ui, col = "3-char Code", "Code3"

    return ui, col


def _text_filter(df: pd.DataFrame, q: str) -> pd.DataFrame:
    if not q:
        return df
    s = q.strip().lower()
    cols = [c for c in ["Code", "Code3", "Code4", "Description"] if c in df.columns]
    if not cols:
        return df
    mask = False
    for c in cols:
        mask = mask | df[c].astype(str).str.lower().str.contains(s, na=False)
    return df[mask].copy()


def _group_filter(df: pd.DataFrame, groups: List[str]) -> pd.DataFrame:
    if not groups or "Group" not in df.columns:
        return df
    return df[df["Group"].isin(groups)].copy()


def _melt_gender(d: pd.DataFrame, level_col: str) -> pd.DataFrame | None:
    """Melt gender columns to long format: [level_col, 'Description', 'Segment', 'Value']."""
    d = _align_gender_columns(d)

    need = ["FCE_male", "FCE_female"]  # at least male & female
    if not set(need).issubset(d.columns):
        return None

    keep_cols = [c for c in [level_col, "Description"] if c in d.columns]
    cols = ["FCE_male", "FCE_female"] + (["FCE_unknown"] if "FCE_unknown" in d.columns else [])

    long = (
        d[keep_cols + cols]
        .melt(id_vars=keep_cols, value_vars=cols, var_name="_gcol", value_name="Value")
    )
    m = {"FCE_male": "Male", "FCE_female": "Female", "FCE_unknown": "Unknown"}
    long["Segment"] = long["_gcol"].map(m)
    long.drop(columns=["_gcol"], inplace=True)
    long["Value"] = pd.to_numeric(long["Value"], errors="coerce").fillna(0.0)
    long["Segment"] = pd.Categorical(long["Segment"], categories=GENDER_ORDER, ordered=True)
    return long


def _melt_admission(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    need = {"FAE_emergency", "FAE_waiting", "FAE_planned", "FAE_other"}
    if not need.issubset(df.columns):
        return None
    id_cols = [c for c in df.columns if c not in need]
    amap = {
        "FAE_emergency": "Emergency",
        "FAE_waiting": "Waiting list",
        "FAE_planned": "Planned",
        "FAE_other": "Other",
    }
    a = (
        df.melt(id_vars=id_cols, value_vars=list(amap), var_name="_acol", value_name="Value")
          .assign(Segment=lambda t: t["_acol"].map(amap))
          .drop(columns=["_acol"])
    )
    a["Segment"] = pd.Categorical(a["Segment"], categories=ADMISSION_ORDER, ordered=True)
    return a

def _normalize_within_label(df_plot: pd.DataFrame, level_col: str) -> pd.DataFrame:
    """Convert Value to 0–100 within each label (for 100% stacked)."""
    d = df_plot.copy()
    denom = d.groupby(level_col)["Value"].transform("sum").replace(0, pd.NA)
    d["Value"] = (d["Value"] / denom * 100).fillna(0.0)
    return d

def _c3_to_int(code3: str) -> int:
    s = (code3 or "").replace(".", "").upper()
    if len(s) < 3 or not s[1:3].isdigit():
        return -1
    return (ord(s[0]) - 65) * 100 + int(s[1:3])

def _parse_icd_range(s: str) -> tuple[int, int, str]:
    t = str(s).strip().upper().replace("–", "-").replace("—", "-")
    if "-" in t:
        a, b = t.split("-", 1)
        return _c3_to_int(a[:3]), _c3_to_int(b[:3]), s
    x = _c3_to_int(t[:3])
    return x, x, s

def _assign_summary_bucket(df: pd.DataFrame, ranges: list[str]) -> pd.DataFrame:
    bounds = [_parse_icd_range(r) for r in (ranges or []) if r]
    def pick(c3: str):
        v = _c3_to_int(c3)
        for lo, hi, label in bounds:
            if lo != -1 and hi != -1 and lo <= v <= hi:
                return label
        return None
    out = df.copy()
    if "Code3" not in out.columns:
        if "Code" in out.columns:
            out["Code3"] = out["Code"].astype(str).str.replace(".", "", regex=False).str[:3]
        elif "Code4" in out.columns:
            out["Code3"] = out["Code4"].astype(str).str.replace(".", "", regex=False).str[:3]
    out["SummaryBucket"] = out["Code3"].map(pick)
    return out

def _canon_code3_desc(df: pd.DataFrame) -> pd.DataFrame:
    """Build a stable Code3 → Desc3 mapping."""
    d = df.copy()
    if "Code3" not in d.columns:
        if "Code" in d.columns:
            d["Code3"] = d["Code"].map(_ensure_code3)
        elif "Code4" in d.columns:
            d["Code3"] = d["Code4"].astype(str).str.replace(".", "", regex=False).str[:3]
    if "Description" not in d.columns:
        return pd.DataFrame(columns=["Code3", "Desc3"])

    if "Code4" in d.columns:
        d3 = d[d["Code4"].isna()][["Code3", "Description"]].dropna()
    else:
        d3 = d[["Code3", "Description"]].dropna()

    if d3.empty:
        d3 = (
            d.groupby("Code3", as_index=False)["Description"]
             .agg(lambda s: s.dropna().iloc[0] if not s.dropna().empty else "")
        )

    return d3.drop_duplicates("Code3").rename(columns={"Description": "Desc3"})


GENDER_COL_ALIASES = {
    "FCE_male":    ["FCE_male", "Male", "Male (FCE)", "Male \n(FCE)", "Male_FCE", "FCE Male"],
    "FCE_female":  ["FCE_female", "Female", "Female (FCE)", "Female \n(FCE)", "Female_FCE", "FCE Female"],
    "FCE_unknown": ["FCE_unknown", "Unknown", "Gender Unknown (FCE)", "Gender Unknown \n(FCE)", "Unknown_FCE"],
}

def _align_gender_columns(d: pd.DataFrame) -> pd.DataFrame:
    """Normalize gender columns to FCE_male / FCE_female / FCE_unknown; fill 0 if missing."""
    out = d.copy()
    for tgt, alts in GENDER_COL_ALIASES.items():
        if tgt in out.columns:
            out[tgt] = pd.to_numeric(out[tgt], errors="coerce").fillna(0.0)
            continue
        found = None
        for a in alts:
            if a in out.columns:
                found = a
                break
        if found is not None:
            out[tgt] = pd.to_numeric(out[found], errors="coerce").fillna(0.0)
        else:
            out[tgt] = 0.0
    return out


# ===== Public: Controls =====
def bar_controls(sidebar, df: pd.DataFrame, key_prefix: str = "bar") -> Dict[str, Any]:
    """Render Bar-specific controls in the sidebar; return a config dict."""
    k = lambda name: f"{key_prefix}_{name}"

    level_ui = sidebar.selectbox(
        "Level",
        ["Summary 3-char Code", "3-char Code", "Group"],
        index=1,
        key=k("level_ui"),
    )

    groups_all = sorted(df["Group"].dropna().unique().tolist()) if "Group" in df.columns else []
    groups_pick = sidebar.multiselect("Filter groups (A–Z)", groups_all, default=[], key=k("groups"))

    q = sidebar.text_input("Search code/description", value="", key=k("q"))

    split = sidebar.selectbox("Split by", ["None", "Gender", "Admission"], index=0, key=k("split"))

    # Metric chooser (disabled when split forces metric)
    has_fae = "FAE_total" in df.columns
    has_fce = "FCE_total" in df.columns
    metric_default = "FAE" if has_fae else "FCE"
    disabled = split in {"Gender", "Admission"}
    help_msg = (
        "Gender view always uses FCE; this control is disabled."
        if split == "Gender"
        else ("Admission view always uses FAE; this control is disabled."
              if split == "Admission" else None)
    )
    metric_pick = sidebar.radio(
        "Metric (total)", ["FAE", "FCE"],
        index=(0 if metric_default == "FAE" else 1),
        key=k("metric_total_choice"),
        disabled=disabled,
        help=help_msg,
    )
    metric_total_choice = (
        "FCE_total" if split == "Gender"
        else "FAE_total" if split == "Admission"
        else ("FAE_total" if metric_pick == "FAE" else "FCE_total")
    )

    # Sort / Rows / Paging
    asc = sidebar.toggle("Ascending", value=False, key=k("asc"))
    rows = sidebar.radio("Rows", ["Top 25", "Top 50", "All"], index=2, key=k("rows"))
    page_size = sidebar.selectbox("Page size", [25, 50, 100], index=1, key=k("pgsz"))
    page = sidebar.number_input("Page", min_value=1, value=1, step=1, key=k("pgno"))

    st.session_state.pop(f"{key_prefix}_norm", None)

    return {
        "level_ui": level_ui,
        "groups": groups_pick,
        "q": q,
        "split": split,                      # "None" | "Gender" | "Admission"
        "normalize": False,                  # 100% stacked disabled by default
        "metric_total_choice": metric_total_choice,
        "asc": bool(asc),
        "rows": rows,                        # "Top 25" | "Top 50" | "All"
        "page_size": int(page_size),
        "page": int(page),
        "key_prefix": key_prefix,
    }


# ===== Public: Render =====
def bar_render(
    df_scoped: pd.DataFrame,
    metric_total: str,
    cfg: Dict[str, Any],
    *,
    title_suffix: str = "",
    key_suffix: str = "",
    highlight: Optional[str] = None,
) -> None:
    """
    Render a bar chart according to cfg. This function calls st.plotly_chart directly.
    df_scoped: data already filtered/aggregated by year scope (e.g., 23/24-only or 3-year-sum).
    """
    # 1) Level column & filters
    d = df_scoped.copy()
    ui_req = (
        cfg.get("level_ui")
        or {"Code3": "3-char Code", "Summary3": "Summary 3-char Code", "Group": "Group"}.get(
            cfg.get("level") or cfg.get("level_choice"), "3-char Code"
        )
    )

    if "Code3" not in d.columns:
        if "Code" in d.columns:
            d["Code3"] = d["Code"].astype(str).str.replace(".", "", regex=False).str[:3]
        elif "Code4" in d.columns:
            d["Code3"] = d["Code4"].astype(str).str.replace(".", "", regex=False).str[:3]

    # Summary bucketing only when explicitly chosen
    if ui_req == "Summary 3-char Code":
        ranges = cfg.get("summary_buckets") or []
        if ranges:
            d = _assign_summary_bucket(d, ranges)  # adds d["SummaryBucket"]
            if "SummaryBucket" in d.columns and d["SummaryBucket"].notna().any():
                ui_label, level_col = "Summary 3-char Code", "SummaryBucket"
            else:
                st.info("No codes matched the summary ranges — showing 3-char Code instead.")
                ui_label, level_col = "3-char Code", "Code3"
        else:
            st.info("Summary ranges not provided — showing 3-char Code.")
            ui_label, level_col = "3-char Code", "Code3"
    else:
        ui_label, level_col = _pick_level_col(d, ui_req)

    # Normalize gender columns once
    d = _align_gender_columns(d)

    # Optional whitelist for Summary (not used by default)
    if cfg.get("summary_only", False) and level_col == "Code3":
        white = cfg.get("summary_codes") or []
        if white:
            d = d[d["Code3"].astype(str).isin([str(x) for x in white])].copy()
        else:
            st.info("Summary code list not provided — showing all 3-char codes.")

    # Filters
    d = _group_filter(d, cfg.get("groups") or [])
    d = _text_filter(d, cfg.get("q") or "")

    # Fallback if current level column vanished after filtering
    if level_col not in d.columns:
        ui_label, level_col = _pick_level_col(d, ui_label)

    d[level_col] = d[level_col].astype(str)

    # 2) Split mode & effective metric
    split = cfg.get("split", "None")
    normalize = bool(cfg.get("normalize", False))
    metric_eff = cfg.get("metric_total_choice", metric_total)
    eff_label = _axis_label(metric_eff)

    # Build the plotting frame
    if split == "Gender":
        g_long = _melt_gender(d, level_col)
        if g_long is None:
            st.warning("Split column 'Gender' not found — fallback to None.")
            split = "None"
        else:
            d_use = g_long
            eff_label = "FCE"
    elif split == "Admission":
        a_long = _melt_admission(d)
        if a_long is None:
            st.warning("Split column 'Admission' not found — fallback to None.")
            split = "None"
        else:
            d_use = a_long
            eff_label = "FAE"
    if split == "None":
        if metric_eff not in d.columns:
            st.error(f"Selected metric '{metric_eff}' not in data.")
            return
        d_use = d[[level_col, "Description", metric_eff]].copy()
        d_use = d_use.rename(columns={metric_eff: "Value"})

    # 3) Aggregate (and handle split)
    if split == "None":
        agg = d_use.groupby([level_col], as_index=False)["Value"].sum()
    else:
        if "Segment" not in d_use.columns:
            st.error("Internal error: melted DataFrame missing 'Segment'.")
            return
        agg = d_use.groupby([level_col, "Segment"], as_index=False)["Value"].sum()

    # Map Code3 → Desc3 for hover (Code-level only)
    desc3_map = _canon_code3_desc(d) if level_col == "Code3" else None
    if desc3_map is not None:
        agg = agg.merge(desc3_map, on="Code3", how="left")

    # Y-axis labels (codes/buckets only; descriptions shown in hover)
    agg["Label"] = agg[level_col].astype(str)

    # Compose hover title: “Code — Description” when description exists
    def _join_hover(code_s: pd.Series, desc_s: Optional[pd.Series]) -> pd.Series:
        if desc_s is None:
            return code_s.astype(str)
        raw = desc_s
        s = desc_s.astype(str)
        valid = raw.notna() & s.strip().ne("") & s.str.lower().ne("nan")
        return pd.Series(np.where(valid, code_s.astype(str) + " — " + s, code_s.astype(str)))

    if level_col in ("Code", "Code3", "Code4"):
        desc_series = agg["Desc3"] if "Desc3" in agg.columns else agg.get("Description")
        agg["HoverTitle"] = _join_hover(agg[level_col], desc_series)
    elif level_col == "SummaryBucket":
        rmap = cfg.get("range_desc_map") or {}
        if rmap:
            desc_series = agg[level_col].map(rmap)
            agg["HoverTitle"] = _join_hover(agg[level_col], desc_series)
        else:
            agg["HoverTitle"] = agg[level_col].astype(str)
    else:
        agg["HoverTitle"] = agg[level_col].astype(str)

    # 4) Order, rows, paging
    asc = bool(cfg.get("ascending", False))
    order_all = (
        agg.groupby(level_col)["Value"].sum()
        .sort_values(ascending=asc).index.tolist()
    )

    rows = cfg.get("rows", "All")
    if rows == "Top 25":
        keep = order_all[:25]
    elif rows == "Top 50":
        keep = order_all[:50]
    else:
        page_size = max(1, int(cfg.get("page_size", 50)))
        total_rows = len(order_all)
        pages = max(1, math.ceil(total_rows / page_size))
        page = max(1, min(int(cfg.get("page", 1)), pages))
        i0, i1 = (page - 1) * page_size, (page - 1) * page_size + page_size
        keep = order_all[i0:i1]

    if not keep:
        st.info("No rows after current filters.")
        return

    agg = agg[agg[level_col].astype(str).isin([str(x) for x in keep])].copy()
    agg[level_col] = pd.Categorical(agg[level_col], categories=keep, ordered=True)

    # Y order is just the code/bucket; description is hover-only
    label_keep = [str(x) for x in keep]
    agg["Label"] = pd.Categorical(agg["Label"].astype(str), categories=label_keep, ordered=True)

    # 5) Optional 100% stacked for split views
    if split != "None" and normalize:
        agg = _normalize_within_label(agg, level_col)
        x_title, x_tickformat = "Percent", ".0f"
    else:
        x_title, x_tickformat = _axis_label(metric_eff), "~s"

    # 6) Derivatives (global share, within-group share)
    total_global = float(agg["Value"].sum()) or 1.0
    agg["Share_global"] = agg["Value"] / total_global

    if split != "None":
        if normalize:
            agg["Share_within"] = agg["Value"] / 100.0
        else:
            denom = agg.groupby(level_col)["Value"].transform("sum").replace(0, pd.NA)
            agg["Share_within"] = (agg["Value"] / denom).fillna(0.0)

    # 7) Plot
    if split == "None":
        fig = px.bar(
            agg, x="Value", y="Label", orientation="h",
            custom_data=["Share_global", "HoverTitle"],
            category_orders={"Label": label_keep},
            title=None,
        )
        fig.update_traces(
            hovertemplate="<b>%{customdata[1]}</b><br>"
                          "Count %{x:,.0f} • Percent %{customdata[0]:.1%}"
                          "<extra></extra>"
        )
        fig.update_layout(hovermode="y", hoverlabel=dict(align="left"))

    else:
        seg_order = GENDER_ORDER if split == "Gender" else ADMISSION_ORDER
        fig = px.bar(
            agg, x="Value", y="Label", color="Segment",
            orientation="h", barmode="stack",
            category_orders={"Label": label_keep, "Segment": seg_order},
            color_discrete_map=COLOR_DISCRETE_MAP,
            custom_data=["Segment", "Share_within", "Share_global", "HoverTitle"],
            title=None,
        )
        fig.update_layout(
            hovermode="y unified",
            hoverlabel=dict(align="left"),
            legend_title_text=("Gender" if split == "Gender" else "Admission"),
        )
        for i, tr in enumerate(fig.data):
            if i == 0:
                tr.hovertemplate = (
                    "<b>%{customdata[3]}</b><br>"
                    "%{customdata[0]} • Percent within code %{customdata[1]:.1%} • "
                    "Percent of total %{customdata[2]:.1%}"
                    "<extra></extra>"
                )
            else:
                tr.hovertemplate = (
                    "%{customdata[0]} • Percent within code %{customdata[1]:.1%} • "
                    "Percent of total %{customdata[2]:.1%}"
                    "<extra></extra>"
                )

    # Layout & axes
    n_bars = len(keep)
    fig.update_layout(
        height=min(1100, max(420, 30 * n_bars + 120)),
        bargap=0.18, bargroupgap=0.10,
        margin=dict(t=56, r=12, b=36, l=12),
    )
    fig.update_xaxes(title_text=x_title, tickformat=x_tickformat)
    fig.update_yaxes(title_text=("Codes" if ui_label.startswith("Code") else ui_label), automargin=True)

    rows_on_page = len(keep)
    st.markdown(
        f"""
        <div style="
          display:flex; align-items:center; justify-content:space-between;
          gap:.75rem; margin:.25rem 0 .5rem;">
          <div style="
            font-weight:700; font-size:1.15rem; line-height:1;
            white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
            Bar — {x_title}{f" ({title_suffix})" if title_suffix else ""} by {ui_label}
          </div>
          <div style="white-space:nowrap; opacity:.85;">
            Bar · Page {page}/{pages} · {rows_on_page} rows
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    k_tail = f"{cfg.get('key_prefix', 'bar')}_{ui_label}_{split}_{'norm' if normalize else 'abs'}_{key_suffix}"
    st.plotly_chart(fig, use_container_width=True, key=k_tail)

    if not (split != "None" and normalize):
        st.caption(f"Total = {total_global:,.0f}")
