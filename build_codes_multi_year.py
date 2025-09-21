
from pathlib import Path
import pandas as pd
import numpy as np
import re
import argparse
from collections import Counter

SHEET = "Primary Diagnosis 3 Character"

def infer_yearlabel_from_name(p: Path) -> str | None:
    m = re.search(r'(\d{4})[-_](\d{2})', p.name)
    if not m: 
        return None
    return f"{m.group(1)}/{m.group(2)}"

HEADER_GUESS = {
    "2019/20": 10,  # title row 11 (1-based)
    "2020/21": 10,
    "2023/24": 10,
}

def read_sheet_with_fallback(xlsx: Path, sheet: str, header_guess: int):
    tried = []
    for h in [header_guess, 9, 10, 11, 12, 13, 14]:
        if h in tried:
            continue
        tried.append(h)
        try:
            df = pd.read_excel(
                xlsx, sheet_name=sheet, header=h,
                dtype=str, engine="openpyxl", keep_default_na=False
            )
            cols = [str(c).lower() for c in df.columns]
            if any("primary diagnosis" in c for c in cols) or any("finished" in c and "episodes" in c for c in cols):
                return df
        except Exception:
            continue
    return pd.read_excel(
        xlsx, sheet_name=sheet, header=header_guess,
        dtype=str, engine="openpyxl", keep_default_na=False
    )

def normalize_header(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s.strip('"').strip("'")

RAW_TO_STD = {
    "Primary diagnosis: 3 character code and description": "Code_raw",
    "Finished consultant episodes": "FCE_total",
    "Finished Consultant Episodes": "FCE_total",
    "Finished admission episodes": "FAE_total",
    "Admissions": "FAE_total",
    "Male (FCE)": "FCE_male",
    "Male": "FCE_male",
    "Female (FCE)": "FCE_female",
    "Female": "FCE_female",
    "Gender Unknown (FCE)": "FCE_unknown",
    "Gender Unknown": "FCE_unknown",
    "Emergency (FAE)": "FAE_emergency",
    "Emergency": "FAE_emergency",
    "Waiting list (FAE)": "FAE_waiting",
    "Waiting list": "FAE_waiting",
    "Planned (FAE)": "FAE_planned",
    "Planned": "FAE_planned",
    "Other (FAE)": "FAE_other",
    "Other Admission Method": "FAE_other",
    "Mean time waited (Days)": "mean_wait",
    "Mean time waited": "mean_wait",
    "Median time waited (Days)": "median_wait",
    "Median time waited": "median_wait",
    "Mean length of stay (Days)": "mean_los",
    "Mean length of stay": "mean_los",
    "Median length of stay (Days)": "median_los",
    "Median length of stay": "median_los",
}
RAW_TO_STD_NORM = {normalize_header(k): v for k, v in RAW_TO_STD.items()}

def collapse_duplicate_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    counts = Counter(df.columns)
    dups = [c for c, n in counts.items() if n > 1]
    for c in dups:
        cols = [col for col in df.columns if col == c]
        block = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        df[c] = block.sum(axis=1)
        # drop all but the first
        keep_first = True
        newcols = []
        for col in df.columns:
            if col == c:
                if keep_first:
                    newcols.append(col)
                    keep_first = False
                else:
                    newcols.append(None)
            else:
                newcols.append(col)
        df.columns = newcols
        df = df.loc[:, df.columns.notna()]
    return df

def smart_join_description(df: pd.DataFrame, code_col_name: str, max_probe_cols=8) -> pd.Series:
    cols = df.columns.tolist()
    i_code = cols.index(code_col_name) if code_col_name in cols else 0
    probe = cols[i_code+1 : i_code+1+max_probe_cols]
    keep = []
    for c in probe:
        name = str(c)
        s = df[c].astype(str)
        mostly_text = (~s.str.fullmatch(r"\s*[\d,.\-]+%?\s*")).mean() >= 0.8
        if ("Unnamed" in name) or ("description" in name.lower()) or mostly_text:
            keep.append(c)
        else:
            break
    if not keep:
        keep = cols[i_code+1:i_code+2]
    block = df[keep].astype(str).replace({"nan":"", "NaN":"", "None":""})
    return block.apply(lambda r: " ".join([t for t in r if t]).strip(), axis=1)

def read_codes_3char_one(xlsx: Path, year_label: str) -> pd.DataFrame:
    header_guess = HEADER_GUESS.get(year_label, 10)
    df = read_sheet_with_fallback(xlsx, SHEET, header_guess)
    # normalize + rename
    norm_cols = {c: normalize_header(str(c)) for c in df.columns}
    df = df.rename(columns=norm_cols)
    df = df.rename(columns=lambda c: RAW_TO_STD_NORM.get(c, c))
    df = collapse_duplicate_numeric_columns(df)

    if "Code_raw" not in df.columns:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "Code_raw"})

    code_raw = df["Code_raw"].astype(str).str.strip().str.upper()
    mask_total = code_raw.str.fullmatch(r"(?i)total")
    mask_code3 = code_raw.str.match(r"^[A-Z][0-9A-Z]{2}$") | code_raw.str.match(r"^[A-Z]\d{2}$")
    df = df[~mask_total & mask_code3].copy()
    df["Code"] = code_raw[~mask_total & mask_code3]

    df["Description"] = smart_join_description(df, "Code_raw")
    df["Group"] = df["Code"].str.replace(".", "", regex=False).str[:1]

    numcols = [
        "FCE_male", "FCE_female", "FCE_unknown", "FCE_total", "FAE_total",
        "FAE_emergency", "FAE_waiting", "FAE_planned", "FAE_other",
        "mean_wait", "median_wait", "mean_los", "median_los",
    ]
    for c in numcols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        else:
            df[c] = 0

    if (df["FCE_total"] == 0).all():
        df["FCE_total"] = df["FCE_male"] + df["FCE_female"] + df["FCE_unknown"]
    if (df["FAE_total"] == 0).all() and (df[["FAE_emergency","FAE_waiting","FAE_planned","FAE_other"]].sum(axis=1) > 0).any():
        df["FAE_total"] = df[["FAE_emergency","FAE_waiting","FAE_planned","FAE_other"]].sum(axis=1)

    df["YearLabel"] = year_label
    keep = [
        "YearLabel", "Group", "Code", "Description",
        "FAE_total", "FCE_total",
        "FAE_emergency", "FAE_waiting", "FAE_planned", "FAE_other",
        "mean_wait", "median_wait", "mean_los", "median_los",
    ]
    for c in keep:
        if c not in df.columns:
            df[c] = np.nan
    for c in ["YearLabel", "Group", "Code", "Description"]:
        df[c] = df[c].astype("string").str.strip()

    return df[keep].copy()

def build_codes_multi_year(base_folder: str | Path,
                           out_csv: str | Path = "codes_multi_year.csv",
                           years_whitelist: list[str] | None = None) -> Path:
    base = Path(base_folder)
    files = sorted(base.glob("hosp-epis-stat-admi-diag-*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No NHS files found under {base}")

    rows = []
    for f in files:
        yl = infer_yearlabel_from_name(f)
        if years_whitelist and yl not in set(years_whitelist):
            continue
        try:
            df = read_codes_3char_one(f, yl)
            if len(df):
                rows.append(df)
                print(f"[OK] {f.name}: {len(df)} codes")
            else:
                print(f"[WARN] {f.name}: no rows")
        except Exception as e:
            print(f"[ERR] {f.name}: {e}")

    if not rows:
        raise RuntimeError("No rows collected. Check years_whitelist or column mappings.")
    out = pd.concat(rows, ignore_index=True)
    out = (out.dropna(subset=["Code"])
              .drop_duplicates(subset=["YearLabel","Code"], keep="first")
              .reset_index(drop=True))
    out_path = Path(out_csv)
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved -> {out_path.resolve()}  ({len(out)} rows)")
    return out_path

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build codes_multi_year.csv from NHS HES yearly Excel files")
    ap.add_argument("base_folder", help="Folder that contains 'hosp-epis-stat-admi-diag-*.xlsx'")
    ap.add_argument("--out", default="codes_multi_year.csv", help="Output CSV path")
    ap.add_argument("--years", nargs="*", default=["2019/20","2020/21","2023/24"],
                    help="YearLabels to include, e.g. 2019/20 2020/21 2023/24")
    args = ap.parse_args()
    build_codes_multi_year(args.base_folder, args.out, args.years)
