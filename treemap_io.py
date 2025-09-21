# treemap_io.py
from pathlib import Path
import textwrap
import pandas as pd
import numpy as np
import streamlit as st

def _wrap2(s: str, w=36) -> str:
    s = "" if pd.isna(s) else str(s)
    parts = textwrap.wrap(s, width=w, break_long_words=False, break_on_hyphens=False)
    return "<br>".join(parts[:2])

def _numify(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        else:
            df[c] = 0
    return df

@st.cache_data(show_spinner=False)
def read_code3_from_excel(xlsx_path: str | Path) -> pd.DataFrame:
    import openpyxl
    p = Path(xlsx_path);  assert p.exists(), f"File not found: {p}"
    xl = pd.ExcelFile(p, engine="openpyxl")
    df = pd.read_excel(xl, sheet_name="Primary Diagnosis 3 Character", header=10, dtype=str, keep_default_na=False)

    df = df.rename(columns={
        "Primary diagnosis: 3 character code and description": "Code_raw",
        "Male \n(FCE)": "FCE_male", "Female \n(FCE)": "FCE_female", "Gender Unknown \n(FCE)": "FCE_unknown",
        "Finished consultant episodes": "FCE_total", "Finished Admission Episodes": "FAE_total",
    })
    code_raw = df["Code_raw"].astype(str).str.strip().str.upper()
    df = df[~code_raw.str.fullmatch(r"(?i)total")].copy()
    mask = code_raw.str.match(r"^[A-Z][A-Z0-9]{2}$") | code_raw.str.match(r"^[A-Z]\d{2}$")
    df = df[mask].copy()

    df["Code3"] = code_raw[mask]

    blk = df.iloc[:, 1:7].astype(str).replace({"nan":"", "None":"", "NaN":""})
    df["Description"] = blk.apply(lambda r: " ".join([t for t in r if t]).strip(), axis=1)
    df["Group"] = df["Code3"].str.replace(".", "", regex=False).str[:1]

    _numify(df, ["FCE_male","FCE_female","FCE_unknown","FCE_total","FAE_total"])
    if (df["FCE_total"] == 0).all():
        df["FCE_total"] = df["FCE_male"] + df["FCE_female"] + df["FCE_unknown"]

    df["CodeLabelWrap"] = df.apply(lambda r: f"{r['Code3']} — {_wrap2(r['Description'])}", axis=1)
    keep = ["Group","Code3","Description","FCE_male","FCE_female","FCE_unknown","FCE_total","FAE_total","CodeLabelWrap"]
    return df[keep].copy()

@st.cache_data(show_spinner=False)
def read_code4_from_excel(xlsx_path: str | Path, df3_for_group: pd.DataFrame) -> pd.DataFrame:
    import openpyxl
    p = Path(xlsx_path);  assert p.exists(), f"File not found: {p}"
    xl = pd.ExcelFile(p, engine="openpyxl")
    df = pd.read_excel(xl, sheet_name="Primary Diagnosis 4 Character", header=10, dtype=str, keep_default_na=False)

    df = df.rename(columns={
        "Primary diagnosis: 4 character code and description": "Code4_raw",
        "Male \n(FCE)": "FCE_male", "Female \n(FCE)": "FCE_female", "Gender Unknown \n(FCE)": "FCE_unknown",
        "Finished consultant episodes": "FCE_total", "Finished Admission Episodes": "FAE_total",
    })
    raw = df["Code4_raw"].astype(str).str.strip().str.upper()
    df = df[~raw.str.fullmatch(r"(?i)total")].copy()
    mask = raw.str.match(r"^[A-Z]\d{2}\.[0-9A-Z]$")
    df = df[mask].copy()

    df["Code4"] = raw[mask]
    blk = df.iloc[:, 1:7].astype(str).replace({"nan":"", "None":"", "NaN":""})
    df["Description"] = blk.apply(lambda r: " ".join([t for t in r if t]).strip(), axis=1)

    _numify(df, ["FCE_male","FCE_female","FCE_unknown","FCE_total","FAE_total"])
    if (df["FCE_total"] == 0).all():
        df["FCE_total"] = df["FCE_male"] + df["FCE_female"] + df["FCE_unknown"]

    df["Code3"] = df["Code4"].str.replace(".", "", regex=False).str[:3]
    gmap = df3_for_group[["Code3","Group"]].drop_duplicates().copy()
    gmap["Code3"] = gmap["Code3"].astype("string").str.upper().str.strip()
    df["Code3"] = df["Code3"].astype("string").str.upper().str.strip()
    df = df.merge(gmap, on="Code3", how="left")
    df["Group"] = df["Group"].fillna(df["Code4"].str.replace(".", "", regex=False).str[:1])

    df["Code4LabelWrap"] = df.apply(lambda r: f"{r['Code4']} — {_wrap2(r['Description'])}", axis=1)
    df["CodeLabelWrap"]  = df.apply(lambda r: f"{r['Code3']} — {_wrap2(r['Description'])}", axis=1)
    keep = ["Group","Code3","Code4","Description","FCE_male","FCE_female","FCE_unknown","FCE_total","FAE_total",
            "CodeLabelWrap","Code4LabelWrap"]
    return df[keep].copy()

@st.cache_data(show_spinner=False)
def load_code34(xlsx_path: str | Path):
    df3 = read_code3_from_excel(xlsx_path)
    try:
        df4 = read_code4_from_excel(xlsx_path, df3[["Code3","Group"]])
    except Exception:
        df4 = None
    return df3, df4

def export_2324_csvs(xlsx_path: str | Path, out_dir: str | Path):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df3, df4 = load_code34(xlsx_path)
    df3.assign(YearLabel="2023/24").to_csv(out_dir / "codes3_2023_24.csv", index=False, encoding="utf-8")
    if df4 is not None:
        df4.assign(YearLabel="2023/24").to_csv(out_dir / "codes4_2023_24.csv", index=False, encoding="utf-8")

from treemap_io import export_2324_csvs
export_2324_csvs(r"D:\De\Disease\hosp-epis-stat-admi-diag-2023-24-tab (1).xlsx", r"D:\De\Disease")
