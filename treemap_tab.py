# treemap_tab.py — Treemap module reused by Explore (consumes Explore's prepared DataFrame)
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import textwrap
from colors_tokens import SCALES as COLOR_SCALES  # color scales/tokens loaded elsewhere


GENDER_ORDER = ["Male", "Female", "Unknown"]


def _wrap2lines(s: str, width=36) -> str:
    """Wrap a long label to at most two lines using <br>."""
    s = "" if pd.isna(s) else str(s)
    parts = textwrap.wrap(s, width=width, break_long_words=False, break_on_hyphens=False)
    return "<br>".join(parts[:2])


def _mk_label(code, desc):
    """Build a 'CODE — Description' label with wrapped description."""
    code = "" if pd.isna(code) else str(code)
    desc = "" if pd.isna(desc) else str(desc).strip()
    return code if desc == "" else f"{code} — {_wrap2lines(desc)}"


def _ensure_code3(s: Any) -> str:
    s = "" if pd.isna(s) else str(s)
    return s if "." not in s else s.split(".")[0]


def _unify_metric_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize FAE/FCE aliases to *_total."""
    d = df.copy()
    if "FAE_total" not in d.columns and "FAE" in d.columns:
        d["FAE_total"] = d["FAE"]
    if "FCE_total" not in d.columns and "FCE" in d.columns:
        d["FCE_total"] = d["FCE"]
    return d


def _has_gender_cols(df: pd.DataFrame) -> bool:
    return {"FCE_male", "FCE_female", "FCE_unknown"}.issubset(set(df.columns))


def _has_code4(df: pd.DataFrame) -> bool:
    return ("Code4" in df.columns) and (df["Code4"].notna().any())


def _prepare_label_cols(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()
    if "Code" in out.columns:
        out["CodeLabelWrap"] = out.apply(
            lambda r: _mk_label(r.get("Code3", r.get("Code")), r.get("Description", "")), axis=1
        )
    if "Code4" in out.columns:
        out["Code4LabelWrap"] = out.apply(lambda r: _mk_label(r["Code4"], r.get("Description", "")), axis=1)
    return out


def _build_code3_label(d: pd.DataFrame) -> pd.DataFrame:
    """Ensure Code3 and a clean 'Code3LabelWrap' with a 3-char description (Desc3)."""
    out = d.copy()
    if "Code3" not in out.columns and "Code" in out.columns:
        out["Code3"] = out["Code"].map(_ensure_code3)

    if "Desc3" not in out.columns:
        if "Code4" in out.columns:
            d3 = out[out["Code4"].isna()][["Code3", "Description"]].dropna()
        else:
            d3 = out[["Code3", "Description"]].dropna()
        if d3.empty:
            d3 = (
                out.groupby("Code3", as_index=False)["Description"]
                .agg(lambda s: s.dropna().iloc[0] if not s.dropna().empty else "")
            )
        d3 = d3.rename(columns={"Description": "Desc3"}).drop_duplicates("Code3")
        out = out.merge(d3, on="Code3", how="left")

    if "Desc3_x" in out.columns or "Desc3_y" in out.columns:
        out["Desc3"] = out.get("Desc3_x")
        if "Desc3_y" in out.columns:
            out["Desc3"] = out["Desc3"].fillna(out.get("Desc3_y"))
        for c in ("Desc3_x", "Desc3_y"):
            if c in out.columns:
                out.drop(columns=c, inplace=True)

    out["Code3LabelWrap"] = out.apply(lambda r: _mk_label(r["Code3"], r.get("Desc3", "")), axis=1)
    return out


def _apply_percent_colorbar(fig, max_pct: float):
    """Set colorbar to show percentages with minimal ticks (0% and the top end)."""
    if max_pct <= 0 or not np.isfinite(max_pct):
        max_pct = 1e-6
    fig.update_coloraxes(
        cmin=0,
        cmax=max_pct,
        colorbar=dict(title="Percent of total", tickformat=".1%", nticks=2),
    )


# ===================== Controls =====================
def treemap_controls(
    sidebar,
    df: pd.DataFrame,
    *,
    metric_total: str,  # "FAE_total" | "FCE_total" (provided by Explore)
    key_prefix: str = "tm",
) -> Dict[str, Any]:
    """
    Render Treemap-specific controls (Mode / Palette / Hierarchy / Scope) in the sidebar,
    and return a config dict.
    """
    k = lambda n: f"{key_prefix}_{n}"

    # Compact form styling
    sidebar.markdown(
        """
        <style>
          .stSelectbox, .stRadio, .stNumberInput { margin-bottom:.25rem; }
          [data-testid="stHorizontalBlock"] { gap:.6rem !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    d = _prepare_label_cols(_unify_metric_cols(df))
    has_genders = _has_gender_cols(d)
    has_c4 = _has_code4(d)

    # Row 1: Mode | Hierarchy | Color mode | Palette | Scope/Top-N
    c1, s1, c2, s2, c3, s3, c4, s4, c5 = sidebar.columns(
        [0.6, 0.15, 0.9, 0.15, 1.1, 0.15, 0.6, 0.15, 0.9], gap="small"
    )

    with c1:
        st.caption("Mode")
        mode = st.radio(
            "",
            ["Volume (FAE)", "Gender (FCE)"],
            index=0,
            horizontal=True,
            key=k("mode"),
            label_visibility="collapsed",
        )

    with c2:
        st.caption("Hierarchy")
        if mode.startswith("Volume"):
            hier_choices = ["Group > 3-char Code", "Group > 3→4 char Code"]
        else:
            hier_choices = (
                ["Group > Gender", "Group > Gender > 3-char Code", "Group > Gender > 3→4 char Code"]
                if has_genders
                else ["Group > 3-char Code"]
            )
        hier = st.selectbox("", hier_choices, index=0, key=k("hier"), label_visibility="collapsed")

    with c3:
        st.caption("Color mode")
        if mode.startswith("Volume"):
            color_choices = ["Percent", "% Male (diverging, center 50%)", "None"]
            if not has_genders:
                color_choices = [c for c in color_choices if not c.startswith("% Male")]
            color_mode = st.selectbox("", color_choices, index=0, key=k("color_mode"), label_visibility="collapsed")
        else:
            color_mode = "Percent"  # gender mode does not provide % Male coloring

    with c4:
        st.caption("Palette")
        if mode.startswith("Volume") and str(color_mode).startswith("% Male"):
            palette = st.selectbox("", ["Pink→Blue", "Teal→Yellow"], index=0, key=k("palette"), label_visibility="collapsed")
        else:
            palette = "Pink→Blue"  # default placeholder when palette is not shown

    with c5:
        st.caption("Scope / Top-N")
        scope = st.selectbox("", ["All codes", "Top-N"], index=0, key=k("scope"), label_visibility="collapsed")
        topn = int(
            st.number_input(
                "Top-N (0 = All)",
                min_value=0,
                max_value=300,
                value=60,
                step=5,
                key=k("topn"),
            )
        )

    want4 = "4" in hier
    use_code4 = bool(want4 and has_c4)
    if want4 and not has_c4:
        st.info("4-character codes not found — falling back to 3-character codes.")

    cfg: Dict[str, Any] = dict(
        key_prefix=key_prefix,
        mode=mode,
        hier=hier,
        use_code4=use_code4,
        color_mode=color_mode,
        palette=palette,
        scope=scope,
        topn=int(topn),  # 0 means All codes
    )
    return cfg


# ===================== Render =====================
_TITLE_FONT_SIZE = 18  # match the Bar title look (~1.15rem)

def _apply_title(fig, text: str):
    fig.update_layout(
        title=dict(text=text, x=0.01, y=0.98),  # top-left
        title_font=dict(size=_TITLE_FONT_SIZE),
        margin=dict(t=56, l=4, r=4, b=4),
    )


def treemap_render(df_in, metric_total, cfg, title_suffix: str = "", trust_df: bool = True):
    """
    df_in: the Explore-scoped data (e.g., 23/24 or 3-year sum)
    metric_total: "FAE_total" or "FCE_total"
    cfg: result from treemap_controls(...) (color_mode / palette / hier / scope / topn / use_code4)
    trust_df: when True, trust df_in as-is (no extra fallbacks here)
    """
    import plotly.express as px
    import plotly.graph_objects as go

    d = df_in.copy()

    # ---- unify columns & labels ----
    d = _unify_metric_cols(d)
    d = _build_code3_label(d)
    if "Code3" not in d.columns and "Code" in d.columns:
        d["Code3"] = d["Code"].map(_ensure_code3)
    if "Description" not in d.columns:
        d["Description"] = ""
    if "CodeLabelWrap" not in d.columns and "Code3" in d.columns:
        d["CodeLabelWrap"] = d["Code3"].astype(str) + " — " + d["Description"].astype(str)
    if "Code4" in d.columns and "Code4LabelWrap" not in d.columns:
        d["Code4LabelWrap"] = d["Code4"].astype(str) + " — " + d["Description"].astype(str)

    has_gender = _has_gender_cols(d)
    use_code4 = bool(cfg.get("use_code4", False)) and ("Code4" in d.columns) and d["Code4"].notna().any()
    hier = cfg.get("hier", "")
    scope = cfg.get("scope", "All codes")
    topn = int(cfg.get("topn", 60))
    color_mode = cfg.get("color_mode", "Percent")
    palette = cfg.get("palette", "Pink→Blue")
    scale = COLOR_SCALES["pct_male_fb"] if palette.startswith("Pink") else COLOR_SCALES["pct_male_to"]

    mode = cfg.get("mode") or ("Volume (FAE)" if metric_total == "FAE_total" else "Gender (FCE)")
    d["All"] = "All"  # stable root label

    # ========= Volume (FAE) =========
    if mode.startswith("Volume"):
        # leaf column
        leaf = "Code4" if use_code4 else ("Code3" if "Code3" in d.columns else None)
        if leaf is None or "Group" not in d.columns or "FAE_total" not in d.columns:
            fig = go.Figure()
            fig.update_layout(title="Treemap — missing columns for FAE")
            return fig, None

        # Scope: Top-N by leaf FAE_total
        if scope == "Top-N":
            if topn and topn > 0:
                keep = (
                    d.groupby(leaf, as_index=False)["FAE_total"]
                    .sum()
                    .sort_values("FAE_total", ascending=False)
                    .head(topn)
                )[leaf]
                d = d[d[leaf].isin(keep)].copy()

        # path
        if use_code4 and "Code4LabelWrap" in d.columns:
            path = ["All", "Group", "Code3", "Code4LabelWrap"]
        else:
            path = ["All", "Group", "Code3LabelWrap"] if "Code3LabelWrap" in d.columns else ["All", "Group", "Code3"]

        # color config
        kwargs = dict(path=path, values="FAE_total")

        if color_mode == "Percent":
            total = float(d["FAE_total"].sum()) or 1.0
            d["_pct_total"] = d["FAE_total"] / total
            kwargs["color"] = "_pct_total"
            kwargs["color_continuous_scale"] = COLOR_SCALES["total_pct"]

        elif color_mode.startswith("% Male") and has_gender:
            denom = (d.get("FCE_male", 0) + d.get("FCE_female", 0) + d.get("FCE_unknown", 0)).replace(0, np.nan)
            d["_pct_male"] = (d.get("FCE_male", 0) / denom).astype("float64").fillna(0.0)
            kwargs["color"] = "_pct_male"
            kwargs["color_continuous_scale"] = scale
            kwargs["range_color"] = (0.0, 1.0)
            kwargs["color_continuous_midpoint"] = 0.5
        else:
            kwargs["color"] = None  # categorical (no continuous scale)

        fig = px.treemap(d, **kwargs)
        fig.update_traces(
            texttemplate="%{label}",
            textposition="top left",
            hovertemplate="<b>%{label}</b><br>Count %{value:,.0f} • Percent %{percentRoot:.1%}<extra></extra>",
            pathbar=dict(visible=True, side="top", thickness=22, edgeshape=">"),
            tiling=dict(pad=2),
            root_color="rgba(240,240,240,1)",
            selector=dict(type="treemap"),
        )
        fig.update_layout(uniformtext=dict(minsize=10, mode="hide"))

        if color_mode == "Percent":
            fig.update_coloraxes(colorbar=dict(title="Percent of total", tickformat=".1%"))
        elif color_mode.startswith("% Male") and has_gender:
            fig.update_coloraxes(
                colorbar=dict(
                    title="% Male",
                    tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                    ticktext=["0%", "25%", "50%", "75%", "100%"],
                )
            )
        else:
            # fallback qualitative palette used only if no continuous color is applied
            fig.update_layout(
                treemapcolorway=[
                    "#D55E00",  # Emergency
                    "#F0E442",  # Waiting
                    "#009E73",  # Planned
                    "#56B4E9",  # Other
                    "#0072B2",  # Male (if gender is ever used categorically)
                    "#CC79A7",  # Female
                    "#999999",  # Unknown
                ]
            )

        tail = "All codes" if scope == "All codes" else (f"Top-{topn} codes" if topn else "All codes")
        if title_suffix:
            tail += f" • {title_suffix}"
        _apply_title(fig, f"Treemap — Volume (FAE) — {tail}")
        fig.update_layout(height=660)

        return fig, float(d["FAE_total"].sum())

    # ========= Gender (FCE) =========
    if not has_gender:
        # fallback: no gender columns — show FCE_total treemap without a continuous scale
        leaf = "Code3" if "Code3" in d.columns else None
        path = ["All", "Group", ("Code3LabelWrap" if "Code3LabelWrap" in d.columns else "Code3")]
        fig = px.treemap(d, path=path, values="FCE_total", color=None)
        fig.update_traces(
            texttemplate="%{label}",
            textposition="top left",
            hovertemplate="<b>%{label}</b><br>Count %{value:,.0f} • Percent %{percentRoot:.1%}<extra></extra>",
            pathbar=dict(visible=True, side="top", thickness=22, edgeshape=">"),
            tiling=dict(pad=2),
            root_color="rgba(240,240,240,1)",
        )
        fig.update_layout(uniformtext=dict(minsize=10, mode="hide"))
        tail = "All codes" if cfg.get("scope", "All codes") == "All codes" else (f"Top-{topn} codes" if topn else "All codes")
        if title_suffix:
            tail += f" • {title_suffix}"
        _apply_title(fig, f"Treemap — FCE (no gender) — {tail}")
        fig.update_layout(height=660)
        total_val = float(d["FCE_total"].sum()) if "FCE_total" in d.columns else 0.0
        return fig, total_val

    # Ensure FCE_total if missing (sum of genders)
    if "FCE_total" not in d.columns:
        d["FCE_total"] = d.get("FCE_male", 0).fillna(0) + d.get("FCE_female", 0).fillna(0) + d.get("FCE_unknown", 0).fillna(0)

    # melt genders
    extra_ids = ["Group", "Code", "Code3", "Description"] + (["Code4"] if "Code4" in d.columns else [])
    g = d.melt(
        id_vars=[c for c in extra_ids if c in d.columns],
        value_vars=["FCE_male", "FCE_female", "FCE_unknown"],
        var_name="GenderCol",
        value_name="Value",
    )
    g["Gender"] = g["GenderCol"].str.replace("FCE_", "", regex=False).str.capitalize()
    g = g.drop(columns=["GenderCol"])

    # merge back clean Code3 label
    g = g.merge(d[["Code3", "Code3LabelWrap"]].drop_duplicates(), on="Code3", how="left")

    # Group_final for the path (prefer provided Group; fallback to first letter of Code3)
    if "Group" in d.columns:
        map_c3_group = d[["Code3", "Group"]].drop_duplicates().rename(columns={"Group": "Group_final"})
    else:
        map_c3_group = d[["Code3"]].drop_duplicates()
        map_c3_group["Group_final"] = map_c3_group["Code3"].astype(str).str[0]
    g = g.merge(map_c3_group, on="Code3", how="left")

    gender_hier = hier  # e.g. "Group > Gender > 3-char Code" / "Group > Gender > 3→4 char Code" / "Group > Gender"

    # --------- Branch A: Group > Gender > 3-char Code ----------
    if "3-char" in gender_hier or ("3→4" not in gender_hier):
        # Scope: Top-N by Code3 FCE_total
        if scope == "Top-N":
            if topn and topn > 0:
                keep = (
                    d.groupby("Code3", as_index=False)["FCE_total"]
                    .sum()
                    .sort_values("FCE_total", ascending=False)
                    .head(topn)
                )["Code3"]
                g = g[g["Code3"].isin(keep)].copy()

        # aggregate to Code3 to avoid multiple leaves per code
        g3 = g.groupby(["Gender", "Group_final", "Code3", "Code3LabelWrap"], as_index=False)["Value"].sum()

        total_val = float(g3["Value"].sum()) or 1.0
        g3["_pct_total"] = g3["Value"] / total_val

        # within gender share
        g3 = g3.merge(
            g3.groupby("Gender", as_index=False)["Value"].sum().rename(columns={"Value": "_gender_total"}),
            on="Gender",
            how="left",
        )
        g3["Within gender"] = g3["Value"] / g3["_gender_total"]
        g3["Percent of total"] = g3["_pct_total"]

        path = [px.Constant("All"), "Group_final", "Code3LabelWrap"]

        fig = px.treemap(
            g3,
            path=path,
            values="Value",
            color="_pct_total",
            color_continuous_scale=COLOR_SCALES["pct_male_to"],  # teal→yellow style scale
            labels={"_pct_total": "Percent of total"},
        )
        fig.update_traces(
            texttemplate="%{label}",
            textposition="top left",
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Count %{value:,.0f}<br>"
                "Percent of total %{percentRoot:.1%}<br>"
                "Within gender %{percentParent:.1%}<extra></extra>"
            ),
            pathbar=dict(visible=True, side="top", thickness=22, edgeshape=">"),
            tiling=dict(pad=2),
            root_color="rgba(240,240,240,1)",
        )
        _apply_percent_colorbar(fig, float(g3["_pct_total"].max()))
        fig.update_layout(uniformtext=dict(minsize=10, mode="hide"))

        tail = "All codes" if scope == "All codes" else (f"Top-{topn} codes" if topn else "All codes")
        if title_suffix:
            tail += f" • {title_suffix}"
        _apply_title(fig, f"Treemap — Gender (FCE, leaf=3-char) — {tail}")
        fig.update_layout(height=660)

        return fig, total_val

    # --------- Branch B: Group > Gender > 3→4 char Code ----------
    # ensure Code4LabelWrap
    if "Code4LabelWrap" not in d.columns and "Code4" in d.columns:
        d["Code4LabelWrap"] = d["Code4"].astype(str) + " — " + d["Description"].fillna("").astype(str)

    # merge Code4 labels for treemap leaf
    if "Code4" in g.columns and "Code4" in d.columns:
        g = g.merge(d[["Code4", "Code3", "Code4LabelWrap"]].drop_duplicates(), on=["Code4", "Code3"], how="left")

    # prefer Code4→Group mapping where available
    if "Group" in d.columns and "Code4" in d.columns:
        map_c4_group = d[["Code4", "Group"]].dropna().drop_duplicates().rename(columns={"Group": "Group_final"})
        if not map_c4_group.empty:
            g = g.merge(map_c4_group, on="Code4", how="left", suffixes=("", "_c4"))
            g["Group_final"] = g["Group_final"].fillna(g.get("Group_final_c4"))
            if "Group_final_c4" in g.columns:
                g.drop(columns=["Group_final_c4"], inplace=True, errors="ignore")

    # Scope: Top-N by Code4 FCE_total
    if scope == "Top-N" and "Code4" in d.columns:
        if topn and topn > 0:
            keep4 = (
                d.groupby("Code4", as_index=False)["FCE_total"]
                .sum()
                .sort_values("FCE_total", ascending=False)
                .head(topn)
            )["Code4"]
            g = g[g["Code4"].isin(keep4)].copy()

    # aggregate to Code4
    cols_req = ["Gender", "Group_final", "Code3", "Code4", "Code4LabelWrap"]
    exist = [c for c in cols_req if c in g.columns]
    g4 = g.groupby(exist, as_index=False)["Value"].sum()

    # color: percent of total
    g4 = g4.merge(d[["Code3", "Code3LabelWrap"]].drop_duplicates(), on="Code3", how="left")
    total_val = float(g4["Value"].sum()) or 1.0
    g4["_pct_total"] = g4["Value"] / total_val

    path = [px.Constant("All"), "Group_final", "Code3LabelWrap", "Code4LabelWrap"]

    fig = px.treemap(
        g4,
        path=path,
        values="Value",
        color="_pct_total",
        color_continuous_scale=COLOR_SCALES["pct_male_to"],
        labels={"_pct_total": "Percent of total"},
    )
    fig.update_traces(
        texttemplate="%{label}",
        textposition="top left",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Count %{value:,.0f}<br>"
            "Percent of total %{percentRoot:.1%}<br>"
            "Within gender %{percentParent:.1%}<extra></extra>"
        ),
        pathbar=dict(visible=True, side="top", thickness=22, edgeshape=">"),
        tiling=dict(pad=2),
        root_color="rgba(240,240,240,1)",
    )
    fig.update_layout(uniformtext=dict(minsize=10, mode="hide"))
    _apply_percent_colorbar(fig, float(g4["_pct_total"].max()))

    tail = "All codes" if scope == "All codes" else (f"Top-{topn} codes" if topn else "All codes")
    if title_suffix:
        tail += f" • {title_suffix}"
    _apply_title(fig, f"Treemap — Gender (FCE, leaf=3→4) — {tail}")
    fig.update_layout(height=660)

    return fig, total_val
