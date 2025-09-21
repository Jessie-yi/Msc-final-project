# data_io.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

DEFAULT_YEARS = ["2019/20", "2020/21", "2023/24"]

def load_codes(csv_path: str | Path) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Missing CSV: {p}")
    df = pd.read_csv(p)
    df.columns = df.columns.str.strip()

    need = {"YearLabel", "Code", "Description"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Code"] = df["Code"].astype(str).str.strip()
    df["Description"] = df["Description"].astype(str).str.strip()
    return ensure_hierarchy(df)

def ensure_hierarchy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Chapter"] = out["Code"].str[0]
    out["Code3"]   = out["Code"].str.split(".").str[0]
    out["Code4"]   = out["Code"]
    return out

def filter_years(df: pd.DataFrame, years: list[str] | None = None) -> pd.DataFrame:
    years = years or DEFAULT_YEARS
    valid = [y for y in years if y in df["YearLabel"].unique().tolist()]
    return df[df["YearLabel"].isin(valid)].copy()

def apply_text_filter(df: pd.DataFrame, query: str = "") -> pd.DataFrame:
    q = (query or "").strip().lower()
    if not q:
        return df
    return df[
        df["Code"].str.lower().str.contains(q) |
        df["Description"].str.lower().str.contains(q)
    ].copy()

def apply_group_filter(df: pd.DataFrame, groups: list[str] | None = None) -> pd.DataFrame:
    if not groups:
        return df
    if "Group" not in df.columns:
        return df
    return df[df["Group"].isin(groups)].copy()

def aggregate_for_level(
    df: pd.DataFrame,
    level: str,
    metric: str,
    top_n: int | None = None,
    sort_by: str = "value_desc",
) -> pd.DataFrame:
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not in CSV.")
    label = level

    agg = (df.groupby([label], as_index=False)[metric].sum())

    if sort_by == "value_asc":
        agg = agg.sort_values(metric, ascending=True)
    elif sort_by == "alpha":
        agg = agg.sort_values(label, ascending=True)
    else:
        agg = agg.sort_values(metric, ascending=False)

    if isinstance(top_n, int) and top_n > 0:
        agg = agg.head(top_n)
    return agg.reset_index(drop=True)

def prepare_treemap_source(
    df: pd.DataFrame,
    level: str,
    metric: str,
    label_as: str = "Level",
) -> pd.DataFrame:
    agg = aggregate_for_level(df, level, metric, top_n=None, sort_by="value_desc")

    desc = (df.groupby(level)["Description"]
            .apply(lambda s: s.iloc[0] if len(s) else "")
            .reindex(agg[level]).reset_index(drop=True))
    out = agg.copy()
    out[label_as] = out[level]
    out["Description"] = desc
    return out[[label_as, "Description", metric]]
