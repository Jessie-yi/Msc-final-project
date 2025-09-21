# pages/06_Explore.py — Explore: Bar & Treemap (shared Year scope / Metric)
from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st
import re

SHOW_DEV_HINT = False  # set True to show lightweight debug hints

# Dependencies (kept at project root, not inside pages/)
from bar_tab import bar_controls, bar_render, _align_gender_columns
from treemap_tab import treemap_controls, treemap_render
from common_header import top_nav

top_nav("Explore")
st.title("Explore — Bar & Treemap")
st.caption(
    "Summary uses Group ranges (A00–A09, …); 3-Character uses ICD-10 3-char codes. "
    "Default metric: FAE. Gender views use FCE; Admission views use FAE."
)

_bucket_pat = re.compile(r"^([A-Z])(\d{2})\s*-\s*([A-Z])?(\d{2})$")

def add_summary3(df: pd.DataFrame, buckets: list[str]) -> pd.DataFrame:
    """Map Code3 to a Summary3 range tag (e.g., A00-A09) based on ranges parsed from Excel."""
    if "Code3" not in df.columns:
        return df

    def _parse_range(r: str):
        r = r.replace("—", "-").replace("–", "-").replace(" ", "")
        if "-" in r:
            a, b = r.split("-")
            la, na = a[0], int(a[1:3])
            lb = b[0] if b[0].isalpha() else la
            nb = int(b[-2:])
        else:
            la = lb = r[0]; na = nb = int(r[1:3])
        return la, na, lb, nb, r

    parsed = [_parse_range(r) for r in buckets]

    def to_bucket(code3: str):
        if not isinstance(code3, str) or len(code3) < 3:
            return None
        c, n = code3[0], int(code3[1:3])
        for la, na, lb, nb, tag in parsed:
            if c == la == lb and na <= n <= nb:
                return tag
        return None

    out = df.copy()
    out["Summary3"] = out["Code3"].map(to_bucket)
    return out


def load_range_desc_map(xlsx_path: Path) -> dict[str, str]:
    """Read the Summary ranges and their descriptions from Primary Diagnosis Summary (Excel)."""
    if not xlsx_path.exists():
        return {}
    try:
        df = pd.read_excel(
            xlsx_path,
            sheet_name="Primary Diagnosis Summary",
            usecols="A:B",
            header=None
        )
    except Exception:
        return {}

    pat = re.compile(r"^[A-Z]\d{2}(?:\s*[–-]\s*[A-Z]?\d{2})?$")  # A00 or A00-A09
    mp: dict[str, str] = {}
    for _, row in df.dropna(how="all").iterrows():
        key_raw = str(row.iloc[0]).strip().upper()
        if not pat.match(key_raw.replace("—", "-").replace("–", "-").replace(" ", "")):
            continue
        key = key_raw.replace("—", "-").replace("–", "-").replace(" ", "")
        desc = "" if pd.isna(row.iloc[1]) else str(row.iloc[1]).strip()
        mp[key] = desc
    return mp


# -------------------- Load CSVs (project root) --------------------
ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "codes_multi_year.csv"
if not CSV.exists():
    st.error(f"Missing file: {CSV}")
    st.stop()

df_raw = pd.read_csv(CSV)

def load_summary_buckets(xlsx_path: Path) -> list[str]:
    """Extract range tokens from the Primary Diagnosis Summary sheet column A."""
    if not xlsx_path.exists():
        return []
    try:
        xls = pd.ExcelFile(xlsx_path)
        sheet = next(
            (s for s in xls.sheet_names if "primary" in s.lower() and "summary" in s.lower()),
            "Primary Diagnosis Summary"
        )
        colA = (
            pd.read_excel(xls, sheet_name=sheet, usecols="A", header=None)
            .squeeze("columns").dropna().astype(str)
        )
    except Exception:
        return []

    pat = re.compile(r"[A-Z]\d{2}\s*[–-]\s*[A-Z]?\d{2}?")
    seen, out = set(), []
    for cell in colA:
        for t in pat.findall(cell.upper()):
            t = t.replace("—", "-").replace("–", "-").replace(" ", "")
            if t and t not in seen:
                seen.add(t)
                out.append(t)
    return out


SUMMARY_XLSX = ROOT / "hosp-epis-stat-admi-diag-2023-24-tab (1).xlsx"
range_desc_map = load_range_desc_map(SUMMARY_XLSX)
summary_buckets = list(range_desc_map.keys())

if SHOW_DEV_HINT:
    st.caption(
        f"Loaded {len(summary_buckets)} group ranges from Excel."
        if summary_buckets else
        "Primary Diagnosis Summary ranges not found — fallback to 3-char Code."
    )

# Normalize common column names (align 23/24 versus older years)
rename_map = {}
if "FAE_total" not in df_raw.columns and "FAE" in df_raw.columns:
    rename_map["FAE"] = "FAE_total"
if "FCE_total" not in df_raw.columns and "FCE" in df_raw.columns:
    rename_map["FCE"] = "FCE_total"
if rename_map:
    df_raw = df_raw.rename(columns=rename_map)

# Ensure Code3
def ensure_code3(s):
    s = str(s) if pd.notna(s) else ""
    return s if "." not in s else s.split(".")[0]

if "Code3" not in df_raw.columns and "Code" in df_raw.columns:
    df_raw["Code3"] = df_raw["Code"].map(ensure_code3)

# Optionally read 23/24 3/4-char snapshots for richer treemap/explore
CSV3 = ROOT / "codes3_2023_24.csv"
CSV4 = ROOT / "codes4_2023_24.csv"

def _read_csv_safe(p: Path):
    return pd.read_csv(p) if p.exists() else None

df3_2324 = _read_csv_safe(CSV3)
df4_2324 = _read_csv_safe(CSV4)

for _d in (df3_2324, df4_2324):
    if _d is not None:
        _d.columns = _d.columns.str.strip()

df = df_raw.copy()

# Pick default total metric (not exposed as a widget)
if "FAE_total" in df_raw.columns:
    metric = "FAE_total"
elif "FCE_total" in df_raw.columns:
    metric = "FCE_total"
else:
    st.error("No total metric found: need FAE_total or FCE_total in CSV.")
    st.stop()

# -------------------- Year helpers --------------------
YEARS_PREF = ["2019/20", "2020/21", "2023/24"]
years_present = [y for y in YEARS_PREF if y in df_raw["YearLabel"].unique().tolist()]
if not years_present:
    st.error("No expected years in codes_multi_year.csv (need 2019/20, 2020/21, 2023/24).")
    st.stop()

VALUE_COLS = [c for c in [
    "FAE_total", "FCE_total",
    "FAE_emergency", "FAE_waiting", "FAE_planned", "FAE_other",
    "FCE_male", "FCE_female", "FCE_unknown"
] if c in df_raw.columns]

DIM_CANDIDATES = [c for c in [
    "Group", "Chapter", "Code", "Code3", "Code4", "Description",
    "Admission", "Gender"
] if c in df_raw.columns]

def filter_years(df0: pd.DataFrame, years: list[str]) -> pd.DataFrame:
    """Keep rows from selected NHS years; no aggregation."""
    return df0[df0["YearLabel"].isin(years)].copy()

def sum_years(df0: pd.DataFrame, years: list[str]) -> pd.DataFrame:
    """Aggregate totals across selected NHS years (group-by dimensions)."""
    d = df0[df0["YearLabel"].isin(years)].copy()
    dims = [c for c in DIM_CANDIDATES if c in d.columns] or []
    if not VALUE_COLS:
        return d.drop(columns=["YearLabel"], errors="ignore")
    return d.groupby(dims, as_index=False)[VALUE_COLS].sum()

def year_suffix(scope: str) -> str:
    if scope == "23/24 only":   return "23/24"
    if scope == "3-year sum":   return "19/20+20/21+23/24"
    return "23/24 · see Compare for both"

# Precomputed group list (if available)
if "Group" in df_raw.columns:
    groups_all = sorted(df["Group"].dropna().astype(str).unique().tolist())
else:
    groups_all = []

# ---- Helpers: join 23/24 gender/admission columns for bar view ----
def _load_gender_source_csv3(root: Path) -> pd.DataFrame:
    """Read the 23/24 3-char source and normalize gender/admission columns."""
    p = root / "codes3_2023_24.csv"
    if not p.exists():
        return pd.DataFrame()
    g = pd.read_csv(p)
    g.columns = g.columns.str.strip()

    if "Year" not in g.columns and "YearLabel" in g.columns:
        g = g.rename(columns={"YearLabel": "Year"})

    if "Code3" in g.columns:
        g["Code3"] = g["Code3"].astype(str).str.replace(".", "", regex=False).str[:3]
    elif "Code" in g.columns:
        g["Code3"] = g["Code"].astype(str).str.replace(".", "", regex=False).str[:3]
    else:
        g["Code3"] = None

    alias = {
        "Male (FCE)": "FCE_male", "Male \n(FCE)": "FCE_male",
        "Female (FCE)": "FCE_female", "Female \n(FCE)": "FCE_female",
        "Gender Unknown (FCE)": "FCE_unknown", "Gender Unknown \n(FCE)": "FCE_unknown",
    }
    for src, tgt in alias.items():
        if src in g.columns and tgt not in g.columns:
            g = g.rename(columns={src: tgt})

    keep = [
        "Year", "Group", "Code3", "Description",
        "FCE_male", "FCE_female", "FCE_unknown",
        "FCE_total", "FAE_total",
        "FAE_emergency", "FAE_waiting", "FAE_planned", "FAE_other",
    ]
    keep = [c for c in keep if c in g.columns]
    g = g[keep].copy()

    for c in [
        "FCE_male", "FCE_female", "FCE_unknown",
        "FCE_total", "FAE_total", "FAE_emergency", "FAE_waiting", "FAE_planned", "FAE_other",
    ]:
        if c in g.columns:
            g[c] = pd.to_numeric(g[c], errors="coerce").fillna(0.0)
    return g


def _build_df_scoped_for_bar(d_in: pd.DataFrame) -> pd.DataFrame:
    """Construct a bar-friendly DataFrame with normalized dims and numeric columns."""
    d = d_in.copy()

    if "Year" not in d.columns and "YearLabel" in d.columns:
        d["Year"] = d["YearLabel"]
    if "Code3" not in d.columns:
        if "Code" in d.columns:
            d["Code3"] = d["Code"].astype(str).str.replace(".", "", regex=False).str[:3]
        elif "Code4" in d.columns:
            d["Code3"] = d["Code4"].astype(str).str.replace(".", "", regex=False).str[:3]

    gender_src = _load_gender_source_csv3(ROOT)
    if not gender_src.empty:
        if "Year" in d.columns and d["Year"].nunique() == 1 and "Year" in gender_src.columns:
            gender_src = gender_src[gender_src["Year"] == d["Year"].iloc[0]]

        if all(c in d.columns for c in ["Year", "Code3"]) and all(c in gender_src.columns for c in ["Year", "Code3"]):
            join_keys = ["Year", "Code3"]
        elif "Code3" in d.columns and "Code3" in gender_src.columns:
            join_keys = ["Code3"]
        else:
            join_keys = []

        if join_keys:
            base_cols = [
                "FCE_male", "FCE_female", "FCE_unknown",
                "FAE_emergency", "FAE_waiting", "FAE_planned", "FAE_other",
                "FCE_total", "FAE_total", "Description",
            ]
            add_cols = [c for c in base_cols if c in gender_src.columns and c not in d.columns]
            if add_cols:
                d = d.merge(
                    gender_src[join_keys + add_cols].drop_duplicates(join_keys),
                    on=join_keys, how="left"
                )

            for base in base_cols + ["mean_wait", "median_wait", "mean_los", "median_los", "mean_age"]:
                x, y = base + "_x", base + "_y"
                if x in d.columns or y in d.columns:
                    if base == "Description":
                        d[base] = d.get(x).combine_first(d.get(y))
                    else:
                        left  = pd.to_numeric(d.get(x), errors="coerce") if x in d.columns else None
                        right = pd.to_numeric(d.get(y), errors="coerce") if y in d.columns else None
                        if left is None and right is not None:
                            d[base] = right
                        elif right is None and left is not None:
                            d[base] = left
                        else:
                            d[base] = left.fillna(right)
                    d.drop(columns=[c for c in (x, y) if c in d.columns], inplace=True)

    try:
        d = _align_gender_columns(d)
    except Exception:
        pass

    num_cols = [
        "FCE_total", "FAE_total", "FCE_male", "FCE_female", "FCE_unknown",
        "FAE_emergency", "FAE_waiting", "FAE_planned", "FAE_other",
        "mean_wait", "median_wait", "mean_los", "median_los", "mean_age",
    ]
    value_cols = [c for c in num_cols if c in d.columns]
    group_cols = [c for c in ["Year", "Group", "Code", "Code3", "Code4", "Description"] if c in d.columns]
    if not value_cols:
        return d_in.iloc[0:0].copy()

    out = d.groupby(group_cols, as_index=False)[value_cols].sum(numeric_only=True)

    if {"FCE_male", "FCE_female"}.issubset(out.columns):
        if float(out[["FCE_male", "FCE_female"]].sum(numeric_only=True).sum()) == 0.0:
            st.warning("Gender columns are all zero — please check codes3_2023_24.csv.")
    return out


# Year-scoped views for Bar (Treemap will re-prepare separately)
def _scoped(year_scope: str):
    if year_scope == "23/24 only":
        return filter_years(df_raw, ["2023/24"]), None, None, "23/24"
    elif year_scope == "3-year sum":
        return sum_years(df_raw, years_present), None, None, "19/20+20/21+23/24"
    else:
        df_sum3 = sum_years(df_raw, years_present)
        df_2324 = filter_years(df_raw, ["2023/24"])
        return df_2324, df_sum3, df_2324, "23/24"


# -------------------- Tabs --------------------
tab_bar, tab_tm, tab_cmp = st.tabs(["Bar", "Treemap", "Compare"])

# ===================== BAR =====================
with tab_bar:
    bar_host = st.expander("Bar options", expanded=True)

    # Year scope (persisted in URL)
    qs = dict(st.query_params)
    if "shared_year" not in st.session_state:
        st.session_state["shared_year"] = {"y2324": "23/24 only", "sum3": "3-year sum", "both": "Both"}.get(
            qs.get("ys"), "23/24 only"
        )
    year_scope = st.radio(
        "Year scope", ["23/24 only", "3-year sum", "Both"],
        horizontal=True, key="shared_year"
    )
    ys_tag = {"23/24 only": "y2324", "3-year sum": "sum3", "Both": "both"}[year_scope]
    st.query_params.update(ys=ys_tag)

    # Compact widgets
    st.markdown(
        """
    <style>
      .stSelectbox, .stMultiSelect, .stTextInput, .stRadio, .stNumberInput { margin-bottom:.25rem; }
      [data-testid="stHorizontalBlock"] { gap:.6rem !important; }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Two rows of controls (add a unique key prefix tied to year scope)
    def K(name: str) -> str:
        return f"ex_{ys_tag}_{name}"

    levels = ["3-char Code", "Summary 3-char Code", "Group"]
    LEVEL_TO_DIM = {"3-char Code": "Code3", "Summary 3-char Code": "Summary3", "Group": "Group"}
    split_by_opts = ["None", "Admission", "Gender"]

    r1c1, r1s1, r1c2, r1s2, r1c3, r1s3, r1c4 = bar_host.columns(
        [0.8, 0.15, 0.9, 0.15, 1.1, 0.1, 0.6], gap="large"
    )
    with r1c1:
        st.caption("Level")
        level = st.selectbox("", levels, index=0, key=K("level"), label_visibility="collapsed")
    with r1c2:
        st.caption("Filter groups (A–Z)")
        picked_groups = st.multiselect("", groups_all, default=[], key=K("groups"), label_visibility="collapsed")
    with r1c3:
        st.caption("Search code/description")
        query = st.text_input("", "", key=K("query"), placeholder="e.g. K50 or Crohn", label_visibility="collapsed")
    with r1c4:
        st.caption("Split by")
        split_by = st.selectbox("", split_by_opts, index=0, key=K("split"), label_visibility="collapsed")

    r2c1, r2c2, r2c3, r2c4, r2c5 = bar_host.columns([1.0, 0.9, 1.0, 0.9, 0.8], gap="small")
    with r2c1:
        st.caption("Metric")
        metric_pick = st.radio("", ["FAE", "FCE"], index=0, horizontal=True, key=K("metric"), label_visibility="collapsed")
    with r2c2:
        st.caption("Order")
        ascending = st.toggle("Ascending", value=False, key=K("order"))
    with r2c3:
        st.caption("Rows")
        rows_pick = st.selectbox("", ["Top 25", "Top 50", "All"], index=2, key=K("rows"), label_visibility="collapsed")
    with r2c4:
        st.caption("Page size")
        page_size = st.selectbox("", [25, 50, 100], index=1, key=K("pagesize"), label_visibility="collapsed")
    with r2c5:
        st.caption("Page")
        page = st.number_input("", min_value=1, value=1, step=1, key=K("page"), label_visibility="collapsed")

    top_n = None if rows_pick == "All" else int(rows_pick.split()[-1])

    # Year-scoped data + construct bar-friendly df
    df_use, df_sum3, df_2324, suffix = _scoped(year_scope)
    df_scoped = _build_df_scoped_for_bar(df_use)

    # If Summary level is chosen, ensure Summary3 and description mapping
    if level == "Summary 3-char Code":
        if summary_buckets:
            df_scoped = add_summary3(df_scoped, summary_buckets)
        else:
            st.warning("Summary ranges not found — fallback to 3-char Code.")
            level = "3-char Code"

    # Adapt dimension for bar renderer (rename chosen dim to Code3)
    dim_col = {"3-char Code": "Code3", "Summary 3-char Code": "Summary3", "Group": "Group"}[level]
    df_for_bar = df_scoped.copy()

    if dim_col == "Summary3":
        if "Summary3" not in df_for_bar.columns:
            df_for_bar = add_summary3(df_for_bar, summary_buckets)
        df_for_bar["Description"] = df_for_bar.get("Summary3").map(range_desc_map).fillna("")
        if "Code3" in df_for_bar.columns:
            df_for_bar = df_for_bar.rename(columns={"Code3": "Code3_orig"})
        df_for_bar = df_for_bar.rename(columns={"Summary3": "Code3"})

    elif dim_col == "Group":
        df_for_bar["Description"] = ""
        if "Code3" in df_for_bar.columns:
            df_for_bar = df_for_bar.rename(columns={"Code3": "Code3_orig"})
        df_for_bar = df_for_bar.rename(columns={"Group": "Code3"})

    else:
        if "Description" not in df_for_bar.columns:
            df_for_bar["Description"] = ""

    # Update URL params (split + query)
    _split_tag = {"None": "n", "Gender": "g", "Admission": "a"}.get(split_by, "n")
    if query.strip():
        st.query_params.update(split=_split_tag, norm="0", q=query.strip())
    else:
        st.query_params.update(split=_split_tag, norm="0")

    # Assemble config and render bar
    level_map = {"3-char Code": "Code3", "Summary 3-char Code": "Summary3", "Group": "Group"}
    metric_total_choice = "FAE_total" if metric_pick == "FAE" else "FCE_total"

    bar_cfg = {
        "level": "Code3",
        "level_choice": "Code3",
        "groups": picked_groups,
        "filter_groups": picked_groups,
        "q": query.strip(),
        "split": split_by,
        "ascending": bool(ascending),
        "top_n": top_n,
        "page_size": int(page_size),
        "page": int(page),
        "metric_total_choice": metric_total_choice,
        "normalize": False,  # no 100% stacked bars
        "summary_buckets": summary_buckets,
        "range_desc_map": range_desc_map,
        # Hover options consumed by bar_tab if present
        "use_summary": (level == "Summary 3-char Code"),
        "tooltip_show_desc": (level != "Group"),
    }

    if bar_cfg["metric_total_choice"] not in df_scoped.columns:
        fallback = "FCE_total" if "FCE_total" in df_scoped.columns else (
            "FAE_total" if "FAE_total" in df_scoped.columns else None
        )
        if fallback:
            bar_cfg["metric_total_choice"] = fallback
            st.info(f"Selected metric not found; switched to '{fallback}'.")
        else:
            st.error("No valid total metric (need FAE_total or FCE_total).")
            st.stop()

    bar_render(
        df_for_bar,
        metric_total=bar_cfg["metric_total_choice"],
        cfg=bar_cfg,
        title_suffix=ys_tag,
        key_suffix=f"bar_{ys_tag}",
    )


# ===================== TREEMAP =====================
with tab_tm:
    year_scope = st.session_state.get("shared_year", "23/24 only")
    df_use, df_sum3, df_2324, suffix = _scoped(year_scope)

    # Choose treemap input according to scope
    def _ensure_code3(df):
        if df is None:
            return None
        if "Code3" not in df.columns and "Code" in df.columns:
            df = df.copy()
            df["Code3"] = df["Code"].astype(str).str.replace(".", "", regex=False).str[:3]
        return df

    if year_scope == "23/24 only":
        # Prefer 4-char, then 3-char, then fall back to df_use
        treemap_df = df4_2324 if df4_2324 is not None else df3_2324
        treemap_df = treemap_df if treemap_df is not None else df_use

    elif year_scope == "3-year sum":
        # Build 3-year aggregated Code3 table
        d = _ensure_code3(df_raw[df_raw["YearLabel"].isin(years_present)])
        value_cols = [c for c in ["FAE_total", "FCE_total"] if c in d.columns]
        treemap_df = d.groupby(["Group", "Code3", "Description"], as_index=False)[value_cols].sum()

    else:  # Both (main tab still shows 23/24)
        treemap_df = df4_2324 if df4_2324 is not None else df3_2324
        treemap_df = treemap_df if treemap_df is not None else df_2324

    # Validate minimal columns for treemap
    need_metric = any(c in treemap_df.columns for c in ["FAE_total", "FCE_total"])
    need_dims = {"Group", "Description"}
    has_code = ("Code4" in treemap_df.columns) or ("Code3" in treemap_df.columns) or ("Code" in treemap_df.columns)
    if not (need_metric and has_code and need_dims.issubset(set(treemap_df.columns))):
        treemap_df = df_use  # fallback to multi-year view

    # Merge 3-char descriptions into 4-char details (if applicable)
    if (treemap_df is not None) and ("Code4" in treemap_df.columns) and (df3_2324 is not None):
        treemap_df = treemap_df.copy()

        if "Code3" not in treemap_df.columns:
            if "Code" in treemap_df.columns:
                treemap_df["Code3"] = treemap_df["Code"].astype(str).str.replace(".", "", regex=False).str[:3]
            else:
                treemap_df["Code3"] = treemap_df["Code4"].astype(str).str.replace(".", "", regex=False).str[:3]

        if "Code3" in df3_2324.columns:
            c3map_src = df3_2324[["Code3", "Description"]]
        else:
            c3map_src = df3_2324.rename(columns={"Code": "Code3"})[["Code3", "Description"]]

        c3map = (
            c3map_src.dropna(subset=["Code3"])
                     .drop_duplicates("Code3")
                     .rename(columns={"Description": "Desc3"})
        )
        treemap_df = treemap_df.merge(c3map, on="Code3", how="left")

    # Treemap options above the chart
    tm_host = st.expander("Treemap options", expanded=True)
    tm_cfg = treemap_controls(tm_host, treemap_df, metric_total=metric, key_prefix=f"tm_{ys_tag}")

    # Render (trust_df=True to reuse prepared data)
    fig_tm, _ = treemap_render(
        treemap_df,
        metric_total=metric,
        cfg=tm_cfg,
        title_suffix=suffix,
        trust_df=True,
    )
    st.plotly_chart(fig_tm, use_container_width=True, key=f"tm_main_{ys_tag}")


# ===================== COMPARE =====================
with tab_cmp:
    # Follow the Year scope selected in Bar options
    year_scope = st.session_state.get("shared_year", "23/24 only")

    if year_scope != "Both":
        st.info("Set **Year scope = Both** in **Bar options** to compare 3-year sum vs 23/24.")
    else:
        # Obtain both scoped views
        _, df_sum3, df_2324, _ = _scoped(year_scope)
        metric_cmp = ("FAE_total" if bar_cfg["metric_total_choice"] == "FAE_total"
                      else "FCE_total")

        # Bar (side-by-side)
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.subheader("Bar — sum 19/20+20/21+23/24")
            bar_render(
                df_sum3,
                metric_total=metric_cmp,
                cfg=bar_cfg,
                title_suffix="19/20+20/21+23/24",
                key_suffix=f"cmp_sum_{ys_tag}",
            )
        with c2:
            st.subheader("Bar — 23/24")
            bar_render(
                df_2324,
                metric_total=metric_cmp,
                cfg=bar_cfg,
                title_suffix="23/24",
                key_suffix=f"cmp_y1_{ys_tag}",
            )

        st.markdown("---")

        # Treemap (side-by-side)
        c3, c4 = st.columns(2, gap="large")
        with c3:
            st.subheader("Treemap — sum 19/20+20/21+23/24")
            d_sum = df_raw[df_raw["YearLabel"].isin(years_present)].copy()
            if "Code3" not in d_sum.columns and "Code" in d_sum.columns:
                d_sum["Code3"] = d_sum["Code"].astype(str).str.replace(".", "", regex=False).str[:3]
            value_cols = [c for c in ["FAE_total", "FCE_total"] if c in d_sum.columns]
            tm_sum = d_sum.groupby(["Group", "Code3", "Description"], as_index=False)[value_cols].sum()
            fig_sum, _ = treemap_render(tm_sum, metric_total=metric, cfg=tm_cfg, title_suffix="19/20+20/21+23/24")
            st.plotly_chart(fig_sum, use_container_width=True, key=f"tm_cmp_sum_{ys_tag}")

        with c4:
            st.subheader("Treemap — 23/24")
            tm_y1 = df4_2324 if df4_2324 is not None else df3_2324
            tm_y1 = tm_y1 if tm_y1 is not None else df_2324
            fig_y1, _ = treemap_render(tm_y1, metric_total=metric, cfg=tm_cfg, title_suffix="23/24")
            st.plotly_chart(fig_y1, use_container_width=True, key=f"tm_cmp_y1_{ys_tag}")
