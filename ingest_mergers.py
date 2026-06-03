#!/usr/bin/env python3
"""Ingest NCUA Insurance Report of Activity -> mergers + charter conversions.

For each quarterly 'insurance-report-activity-<month>-<year>.zip' it reads the
detail workbook and writes:
    data/MERGERS/cycle=<YYYY-MM>/data.parquet       (acquirer, merged CU, reason)
    data/CONVERSIONS/cycle=<YYYY-MM>/data.parquet   (old charter -> new charter)

Headers are matched by TOKENS (e.g. any column containing 'merg' + 'name'), so
NCUA's wording changes across years don't drop columns.

Usage:  QUARTERS="2025-12 2024 2023"  (bare year -> all four quarter-ends).
Quarterly reports exist from 2018-09 on (earlier are PDFs, skipped).
"""
import io
import os
import sys
import zipfile

import pandas as pd
import requests

MONTH_TOKEN = {"03": "march", "06": "june", "09": "sept", "12": "dec"}


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


def _find_header(raw, *must_contain):
    """Index of the first row containing all given substrings (across its cells)."""
    for i in range(min(8, len(raw))):
        joined = " | ".join(str(x).strip().lower() for x in raw.iloc[i].tolist())
        if all(m in joined for m in must_contain):
            return i
    return None


def _merger_colname(header):
    """Map a Mergers-sheet header cell to a tidy column name by tokens."""
    h = str(header).strip().lower()
    if not h or h == "nan":
        return None
    if "region" in h:
        return "region"
    side = "continuing" if "continu" in h else ("merging" if "merg" in h else None)
    if side is None:
        return None
    pre = "continuing" if side == "continuing" else "merging"
    if "charter" in h:
        return f"{pre}_charter"
    if "name" in h:
        return f"{pre}_name"
    if "locat" in h:
        return f"{pre}_location"
    if "asset" in h:
        return f"{pre}_assets"
    if "reason" in h:
        return "merging_reason"
    if "field" in h or "membership" in h:
        return f"{pre}_fom"
    return None


def _to_charter(x):
    return str(int(float(x))) if pd.notna(x) and str(x).strip() not in ("", "nan") else None


def _clean(df):
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().replace({"nan": None, "None": None, "": None})
    return df.reset_index(drop=True)


def parse_mergers(detail_bytes):
    xls = pd.ExcelFile(io.BytesIO(detail_bytes))
    sheet = next((s for s in xls.sheet_names if s.strip().lower() == "mergers"), None)
    if sheet is None:
        return pd.DataFrame()
    raw = pd.read_excel(xls, sheet_name=sheet, header=None)
    hdr = _find_header(raw, "merg", "charter")
    if hdr is None:
        return pd.DataFrame()
    names = [_merger_colname(c) for c in raw.iloc[hdr].tolist()]
    df = raw.iloc[hdr + 1:].copy()
    df.columns = [n if n else f"_drop{i}" for i, n in enumerate(names)]
    df = df.loc[:, [c for c in df.columns if not c.startswith("_drop")]]
    df = df.loc[:, ~df.columns.duplicated()]
    if "merging_charter" not in df.columns:
        return pd.DataFrame()
    df = df[df["merging_charter"].notna()]
    for c in ("continuing_charter", "merging_charter"):
        if c in df.columns:
            df[c] = df[c].apply(_to_charter)
    for c in ("continuing_assets", "merging_assets"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return _clean(df)


def parse_conversions(detail_bytes):
    xls = pd.ExcelFile(io.BytesIO(detail_bytes))
    sheet = next((s for s in xls.sheet_names if s.strip().lower() == "conversions"), None)
    if sheet is None:
        return pd.DataFrame()
    raw = pd.read_excel(xls, sheet_name=sheet, header=None)
    hdr = _find_header(raw, "converted charter")
    if hdr is None:
        return pd.DataFrame()
    cmap = {}
    for i, cell in enumerate(raw.iloc[hdr].tolist()):
        k = str(cell).strip().lower()
        if "converted charter" in k:
            cmap[i] = "old_charter"
        elif k == "charter":
            cmap[i] = "new_charter"
        elif k == "region":
            cmap[i] = "region"
        elif k == "name":
            cmap[i] = "name"
        elif "locat" in k:
            cmap[i] = "location"
        elif "asset" in k:
            cmap[i] = "assets"
        elif "date" in k:
            cmap[i] = "date"
        elif k == "type":
            cmap[i] = "conv_type"
    df = raw.iloc[hdr + 1:].copy()
    df.columns = [cmap.get(i, f"_drop{i}") for i in range(len(df.columns))]
    df = df.loc[:, [c for c in df.columns if not c.startswith("_drop")]]
    df = df.loc[:, ~df.columns.duplicated()]
    if "new_charter" not in df.columns or "old_charter" not in df.columns:
        return pd.DataFrame()
    df = df[df["new_charter"].notna() & df["old_charter"].notna()]
    for c in ("old_charter", "new_charter"):
        df[c] = df[c].apply(_to_charter)
    if "assets" in df.columns:
        df["assets"] = pd.to_numeric(df["assets"], errors="coerce")
    df["date"] = df["date"].astype(str) if "date" in df.columns else None
    return _clean(df)


def _write(df, root, cycle):
    if df.empty:
        return 0
    df = df.copy()
    df.insert(0, "cycle", cycle)
    outdir = os.path.join(root, f"cycle={cycle}")
    os.makedirs(outdir, exist_ok=True)
    df.to_parquet(os.path.join(outdir, "data.parquet"), compression="zstd", index=False)
    return len(df)


def main():
    tokens = (os.environ.get("QUARTERS") or " ".join(sys.argv[1:]) or "").split()
    cycles = expand_quarters(tokens)
    if not cycles:
        print("No quarters given. Set QUARTERS or pass args, e.g. '2025-12 2024'.")
        return
    print("Target cycles:", cycles)
    m_total = c_total = 0
    for cycle in sorted(set(cycles)):
        try:
            year, mm = cycle.split("-")
        except ValueError:
            print(f"  ! {cycle}: bad format, skip"); continue
        if mm not in MONTH_TOKEN:
            print(f"  ! {cycle}: not a quarter-end month, skip"); continue
        if (int(year), int(mm)) < (2018, 9):
            print(f"  ! {cycle}: pre-Sept-2018 reports are PDFs, skip"); continue
        try:
            r = requests.get(report_url(cycle), timeout=180); r.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(r.content))
        except Exception as e:
            print(f"  ! {cycle}: download/zip failed ({e}), skip"); continue
        detail = next((n for n in z.namelist()
                       if "detail" in n.lower() and n.lower().endswith((".xlsx", ".xls"))), None)
        if detail is None:
            print(f"  ! {cycle}: no detail workbook, skip"); continue
        data = z.read(detail)
        try:
            nm = _write(parse_mergers(data), "data/MERGERS", cycle)
        except Exception as e:
            nm = 0; print(f"  ! {cycle}: merger parse failed ({e})")
        try:
            nc = _write(parse_conversions(data), "data/CONVERSIONS", cycle)
        except Exception as e:
            nc = 0; print(f"  ! {cycle}: conversion parse failed ({e})")
        print(f"  \u2713 {cycle}: {nm} mergers, {nc} conversions")
        m_total += nm; c_total += nc
    print(f"Done. {m_total} merger rows, {c_total} conversion rows.")


if __name__ == "__main__":
    main()
