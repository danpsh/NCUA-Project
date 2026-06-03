#!/usr/bin/env python3
"""Ingest NCUA Insurance Report of Activity -> merger records as Parquet.

Downloads each quarterly 'insurance-report-activity-<month>-<year>.zip', reads the
'Mergers' sheet from the detail workbook, and writes tidy merger rows to
    data/MERGERS/cycle=<YYYY-MM>/data.parquet

Usage:  set QUARTERS env var or pass args, e.g.  QUARTERS="2025-12 2025-09 2024"
A bare year expands to all four quarter-ends. Quarterly reports exist from
2018-09 onward (earlier ones are PDFs and are skipped).
"""
import io
import os
import sys
import zipfile

import pandas as pd
import requests

MONTH_TOKEN = {"03": "march", "06": "june", "09": "sept", "12": "dec"}
OUT_ROOT = "data/MERGERS"

# normalized header -> tidy column name
COLMAP = {
    "region": "region",
    "continuing credit union charter": "continuing_charter",
    "continuing name": "continuing_name",
    "continuing location": "continuing_location",
    "continuing assets": "continuing_assets",
    "merging credit union charter": "merging_charter",
    "merging credit union name": "merging_name",
    "merging location": "merging_location",
    "merging assets": "merging_assets",
    "merging reason": "merging_reason",
    "continuing field of membership": "continuing_fom",
    "merging field of membership": "merging_fom",
}


def expand_quarters(tokens):
    out = []
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        if "-" in t:
            out.append(t)
        elif len(t) == 4 and t.isdigit():
            out += [f"{t}-{mm}" for mm in ("03", "06", "09", "12")]
        else:
            print(f"  ! skipping unrecognized token '{t}'")
    return out


def report_url(cycle):
    year, mm = cycle.split("-")
    return ("https://ncua.gov/files/publications/analysis/"
            f"insurance-report-activity-{MONTH_TOKEN[mm]}-{year}.zip")


def parse_mergers(detail_bytes):
    """Return a tidy DataFrame of mergers from a detail workbook's 'Mergers' sheet."""
    xls = pd.ExcelFile(io.BytesIO(detail_bytes))
    sheet = next((s for s in xls.sheet_names if s.strip().lower() == "mergers"), None)
    if sheet is None:
        return pd.DataFrame()
    raw = pd.read_excel(xls, sheet_name=sheet, header=None)
    # locate the header row (it contains 'Merging Credit Union Charter')
    hdr = None
    for i in range(min(6, len(raw))):
        vals = [str(x).strip().lower() for x in raw.iloc[i].tolist()]
        if any("merging credit union charter" in v for v in vals):
            hdr = i
            break
    if hdr is None:
        return pd.DataFrame()
    cols = [str(x).strip() for x in raw.iloc[hdr].tolist()]
    df = raw.iloc[hdr + 1:].copy()
    df.columns = cols
    ren = {c: COLMAP[str(c).strip().lower()] for c in df.columns
           if str(c).strip().lower() in COLMAP}
    df = df.rename(columns=ren)
    df = df[[v for v in COLMAP.values() if v in df.columns]]
    if "merging_charter" not in df.columns:
        return pd.DataFrame()
    df = df[df["merging_charter"].notna()]
    for c in ("continuing_charter", "merging_charter"):
        if c in df.columns:
            df[c] = df[c].apply(
                lambda x: str(int(float(x))) if pd.notna(x) and str(x).strip() else None)
    for c in ("continuing_assets", "merging_assets"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().replace({"nan": None, "None": None})
    return df.reset_index(drop=True)


def main():
    tokens = (os.environ.get("QUARTERS") or " ".join(sys.argv[1:]) or "").split()
    cycles = expand_quarters(tokens)
    if not cycles:
        print("No quarters given. Set QUARTERS or pass args, e.g. '2025-12 2024'.")
        return
    print("Target cycles:", cycles)
    wrote = 0
    for cycle in sorted(set(cycles)):
        try:
            year, mm = cycle.split("-")
        except ValueError:
            print(f"  ! {cycle}: bad format, skip")
            continue
        if mm not in MONTH_TOKEN:
            print(f"  ! {cycle}: not a quarter-end month, skip")
            continue
        if (int(year), int(mm)) < (2018, 9):
            print(f"  ! {cycle}: pre-Sept-2018 reports are PDFs, skip")
            continue
        url = report_url(cycle)
        try:
            r = requests.get(url, timeout=180)
            r.raise_for_status()
        except Exception as e:
            print(f"  ! {cycle}: download failed ({e}), skip")
            continue
        try:
            z = zipfile.ZipFile(io.BytesIO(r.content))
        except Exception as e:
            print(f"  ! {cycle}: not a zip ({e}), skip")
            continue
        detail = next((n for n in z.namelist()
                       if "detail" in n.lower() and n.lower().endswith((".xlsx", ".xls"))), None)
        if detail is None:
            print(f"  ! {cycle}: no detail workbook in zip, skip")
            continue
        try:
            df = parse_mergers(z.read(detail))
        except Exception as e:
            print(f"  ! {cycle}: parse failed ({e}), skip")
            continue
        if df.empty:
            print(f"  · {cycle}: no merger rows found")
            continue
        df.insert(0, "cycle", cycle)
        outdir = os.path.join(OUT_ROOT, f"cycle={cycle}")
        os.makedirs(outdir, exist_ok=True)
        df.to_parquet(os.path.join(outdir, "data.parquet"), compression="zstd", index=False)
        print(f"  \u2713 {cycle}: {len(df)} mergers -> {outdir}/data.parquet")
        wrote += 1
    print(f"Done. Wrote {wrote} cycle(s).")


if __name__ == "__main__":
    main()
