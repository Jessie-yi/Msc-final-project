# data_builder.py
import re, os, pandas as pd
from glob import glob
from pathlib import Path


ROW_MAP = pd.DataFrame({
    "YearLabel": ["2015/16","2016/17","2017/18","2018/19","2019/20","2020/21","2021/22","2022/23","2023/24"],
    "HeaderRowExcel": [11,11,11,11,11,10,10,10,10],
    "TotalRowExcel":  [13,13,13,13,13,12,12,12,12],
    "EndRowExcel":    [162,161,161,161,228,227,227,227,228],
})

def _yearlabel_from_name(name:str):
    m = re.search(r"(\d{4})[-_](\d{2})", name)
    if not m: return None
    s, e2 = int(m.group(1)), int(m.group(2)); e = (s//100)*100 + e2
    return f"{s}/{str(e)[-2:]}"

def _pick_sheet(xl):
    for s in xl.sheet_names:
        l = s.lower()
        if "primary" in l and "summary" in l: return s
    for s in xl.sheet_names:
        if "summary" in s.lower(): return s
    return xl.sheet_names[1] if len(xl.sheet_names)>1 else xl.sheet_names[0]

def _clean_cols(df):
    df = df.copy()
    df.columns = (df.columns.astype(str)
        .str.replace("\xa0"," ", regex=False)
        .str.replace("\n"," ", regex=False)
        .str.replace(r"\s+"," ", regex=True)
        .str.strip())

    ren = {
        "Primary diagnosis: summary code and description":"CodeDesc",
        "Primary diagnoses: summary code and description":"CodeDesc",
        "Primary diagnosis summary code and description":"CodeDesc",
        "Primary diagnosis: summary":"CodeDesc",

        "Finished consultant episodes":"FCE",
        "Finished consultant episodes (FCE)":"FCE",

        "Admissions":"FAE_total",
        "Finished Admission Episodes":"FAE_total",
        "Finished admission episodes":"FAE_total",

        "Male":"Male_FCE","Female":"Female_FCE","Gender Unknown":"Unknown_FCE",
        "Male (FCE)":"Male_FCE","Female (FCE)":"Female_FCE","Gender Unknown (FCE)":"Unknown_FCE",

        "Emergency":"Emergency_FAE","Waiting list":"Waiting_FAE",
        "Planned":"Planned_FAE","Other Admission Method":"Other_FAE",
        "Emergency (FAE)":"Emergency_FAE","Waiting list (FAE)":"Waiting_FAE",
        "Planned (FAE)":"Planned_FAE","Other (FAE)":"Other_FAE",

        "Mean time waited (Days)":"MeanWait_Days","Mean time waited":"MeanWait_Days",
        "Median time waited (Days)":"MedianWait_Days","Median time waited":"MedianWait_Days",
        "Mean length of stay (Days)":"MeanLOS_Days","Mean length of stay":"MeanLOS_Days",
        "Median length of stay (Days)":"MedianLOS_Days","Median length of stay":"MedianLOS_Days",
        "Mean age (Years)":"MeanAge","Mean age":"MeanAge",
    }
    df = df.rename(columns={k:v for k,v in ren.items() if k in df.columns})

    if "CodeDesc" in df.columns:
        if "Unnamed: 1" in df.columns:
            df = df.rename(columns={"Unnamed: 1":"Description"})
            df.insert(0,"Description",df.pop("Description"))
            df.insert(0,"Code",df.pop("CodeDesc"))
        else:
            df.insert(0,"Code",df.pop("CodeDesc"))
            if "Description" not in df.columns:
                df.insert(1,"Description",pd.NA)
    else:
        first = df.columns[0]
        if "Code" not in df.columns: df = df.rename(columns={first:"Code"})
        if "Description" not in df.columns and "Unnamed: 1" in df.columns:
            df = df.rename(columns={"Unnamed: 1":"Description"})
    df["Code"] = df["Code"].astype(str).str.strip()

    if df.columns.duplicated().any():
        for col in pd.unique(df.columns[df.columns.duplicated()]):
            blk = df.loc[:, df.columns == col].apply(pd.to_numeric, errors="coerce")
            df = df.loc[:, ~df.columns.duplicated()]
            df[col] = blk.sum(axis=1, min_count=1)
    return df

def _idx_from_excel(header_excel:int, target_excel:int):
     return target_excel - header_excel - 1

def build_yearly_csv(base_folder:str=".", out_csv:str="hes_yearly_totals_2015_2024.csv", audit_csv:str="hes_yearly_audit.csv"):
    files = sorted(glob(os.path.join(base_folder, "hosp-epis-stat-admi-diag-*.xlsx")))
    rows, audit = [], []

    file_sheet = []
    for f in files:
        yl = _yearlabel_from_name(Path(f).name)
        try:
            xl = pd.ExcelFile(f); sheet = _pick_sheet(xl)
        except Exception:
            sheet = None
        file_sheet.append({"File": Path(f).name, "YearLabel": yl, "Sheet": sheet})
    plan = pd.DataFrame(file_sheet).merge(ROW_MAP, on="YearLabel", how="inner")

    for _, r in plan.iterrows():
        path = os.path.join(base_folder, r["File"])
        h, t, e = int(r["HeaderRowExcel"]), int(r["TotalRowExcel"]), int(r["EndRowExcel"])
        xl = pd.ExcelFile(path)
        df = _clean_cols(xl.parse(r["Sheet"], header=h-1))
        total_idx = _idx_from_excel(h, t); end_idx = _idx_from_excel(h, e)
        data = df.iloc[:end_idx+1]

        def _v(col):
            if col not in data.columns: return pd.NA
            v = data.iloc[total_idx][col]
            if isinstance(v, pd.Series):
                return pd.to_numeric(v, errors="coerce").sum(min_count=1)
            return pd.to_numeric(v, errors="coerce")

        rec = {
            "YearLabel": r["YearLabel"],
            "FCE": _v("FCE"), "FAE_total": _v("FAE_total"),
            "Male_FCE": _v("Male_FCE"), "Female_FCE": _v("Female_FCE"), "Unknown_FCE": _v("Unknown_FCE"),
            "Emergency_FAE": _v("Emergency_FAE"), "Waiting_FAE": _v("Waiting_FAE"),
            "Planned_FAE": _v("Planned_FAE"), "Other_FAE": _v("Other_FAE"),
            "MeanWait_Days": _v("MeanWait_Days"),
            "MedianWait_Days": _v("MedianWait_Days"),
            "MeanLOS_Days": _v("MeanLOS_Days"),
            "MedianLOS_Days": _v("MedianLOS_Days"),
            "MeanAge": _v("MeanAge"),
        }

        rows.append(rec)
        audit.append({"YearLabel": r["YearLabel"], "Sheet": r["Sheet"], "HeaderRow": h, "TotalRow": t, "EndRow": e, **rec})

    yearly = pd.DataFrame(rows).sort_values("YearLabel").reset_index(drop=True)

    # YearStart/End + shares
    def _ys(lbl): return int(lbl.split("/")[0])
    def _ye(lbl): s=int(lbl.split("/")[0]); e2=int(lbl.split("/")[1]); return (s//100)*100 + e2
    yearly["YearStart"] = yearly["YearLabel"].apply(_ys)
    yearly["YearEnd"] = yearly["YearLabel"].apply(_ye)
    for c in ["Male_FCE","Female_FCE","Unknown_FCE"]:
        if c in yearly.columns:
            yearly[c.replace("_FCE","_Share")] = yearly[c] / yearly["FCE"]

    if yearly["FAE_total"].isna().any():
        parts = ["Emergency_FAE","Waiting_FAE","Planned_FAE","Other_FAE"]
        yearly["FAE_total"] = yearly["FAE_total"].fillna(0) + yearly[parts].fillna(0).sum(axis=1)

    yearly.to_csv(os.path.join(base_folder, out_csv), index=False)
    pd.DataFrame(audit).to_csv(os.path.join(base_folder, audit_csv), index=False)
    return out_csv, audit_csv

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    p = argparse.ArgumentParser()
    p.add_argument("--base",  default=".", help="folder containing hosp-epis-stat-admi-diag-*.xlsx")
    p.add_argument("--out",   default="hes_yearly_totals_2015_2024.csv")
    p.add_argument("--audit", default="hes_yearly_audit.csv")
    args = p.parse_args()

    out_csv, audit_csv = build_yearly_csv(base_folder=args.base, out_csv=args.out, audit_csv=args.audit)
    print(f"Wrote {out_csv} and {audit_csv} under {Path(args.base).resolve()}")

