#!/usr/bin/env python3
"""
NCUA Call Report ingest
=======================

Downloads a quarter's 5300 Call Report ZIP from NCUA, unzips it, and loads
*every* table for *all* federally insured credit unions into a queryable store.

Two output formats:

  parquet  (default) -- one Hive-partitioned Parquet dataset per NCUA table,
                        partitioned by `cycle`. Compresses all-CU data to ~tens
                        of MB per quarter, so each file stays under GitHub's
                        100 MB limit. Query it with DuckDB (SQL).

  sqlite             -- everything in a single .db file. Fine locally / for one
                        quarter, but a full quarter of all FS220 tables can push
                        past GitHub's 100 MB per-file limit, so use this for
                        local work, not for committing many quarters.

Usage
-----
    python ingest_ncua.py --quarter 2025-09
    python ingest_ncua.py --quarter 2025-09 --quarter 2025-06   # several at once
    python ingest_ncua.py --quarter 2025-09 --format sqlite
    python ingest_ncua.py --quarter 2025-09 --keep-zip         # cache the raw zip

After parquet ingest, query like this:

    import duckdb
    con = duckdb.connect()
    con.sql('''
        SELECT * FROM read_parquet('data/FOICU/**/*.parquet', hive_partitioning=true)
        WHERE CU_NAME ILIKE '%BLUCURRENT%'
    ''').show()
"""

from __future__ import annotations

import argparse
import io
import re
import sqlite3
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import requests

# NCUA posts quarterly files at this stable pattern. The month is the quarter-end
# month: 03 (Q1), 06 (Q2), 09 (Q3), 12 (Q4).
URL_TEMPLATE = "https://ncua.gov/files/publications/analysis/call-report-data-{ym}.zip"

VALID_MONTHS = {"03", "06", "09", "12"}

# Files in the NCUA zip that are NOT comma-delimited data tables, so we skip them:
# Readme.txt is the help file; Report1.txt is a record-count summary.
SKIP_FILES = {"readme.txt", "report1.txt"}


def build_url(quarter: str) -> str:
    """quarter is 'YYYY-MM' with MM in {03,06,09,12}."""
    m = re.fullmatch(r"(\d{4})-(\d{2})", quarter)
    if not m or m.group(2) not in VALID_MONTHS:
        raise ValueError(
            f"Bad quarter {quarter!r}. Use YYYY-MM with MM in 03/06/09/12, e.g. 2025-09."
        )
    return URL_TEMPLATE.format(ym=quarter)


def download_zip(url: str, dest: Path) -> Path:
    """Stream the ZIP to disk so we never hold the whole thing in memory."""
    print(f"  downloading {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                fh.write(chunk)
    print(f"  saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def sanitize_table_name(filename: str) -> str:
    """'Credit Union Branch Information.txt' -> 'Credit_Union_Branch_Information'."""
    stem = Path(filename).stem
    name = re.sub(r"[^0-9A-Za-z]+", "_", stem).strip("_")
    if not name:
        name = "table"
    if name[0].isdigit():            # SQLite/identifier friendliness
        name = "t_" + name
    return name


def read_ncua_txt(raw: bytes) -> pd.DataFrame:
    """NCUA tables are comma-delimited text with a header row.

    Read everything as strings first: account values are numeric but many cells
    are blank, and letting pandas guess column-by-column across thousands of
    account codes invites mixed-type surprises. Parquet/DuckDB handle the casting
    downstream, and you can tighten dtypes later if you want.
    """
    for encoding in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(
                io.BytesIO(raw),
                sep=",",
                dtype=str,
                encoding=encoding,
                low_memory=False,
                keep_default_na=False,   # keep blanks as '' rather than NaN
            )
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("ncua", b"", 0, 1, "could not decode with utf-8 or latin-1")


def load_quarter_from_zip(zip_path: Path, quarter: str) -> dict[str, pd.DataFrame]:
    """Return {table_name: DataFrame} for every .txt in the archive.

    A `cycle` column (= quarter) is stamped onto every row so multiple quarters
    coexist cleanly in the same dataset.
    """
    tables: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(zip_path) as zf:
        txt_members = [n for n in zf.namelist() if n.lower().endswith(".txt")]
        if not txt_members:
            raise RuntimeError(f"No .txt files found in {zip_path}")
        for member in sorted(txt_members):
            if Path(member).name.lower() in SKIP_FILES:
                print(f"    (skipping non-data file {member})")
                continue
            raw = zf.read(member)
            try:
                df = read_ncua_txt(raw)
            except Exception as exc:  # noqa: BLE001 - never let one odd file kill the run
                print(f"    (skipping {member}: not parseable as CSV -- {exc})")
                continue
            df.insert(0, "cycle", quarter)
            name = sanitize_table_name(member)
            # If a name collides (rare), merge rows rather than drop.
            if name in tables:
                tables[name] = pd.concat([tables[name], df], ignore_index=True)
            else:
                tables[name] = df
            print(f"    {name:<40} {len(df):>8,} rows  {df.shape[1]:>4} cols")
    return tables


def write_parquet(tables: dict[str, pd.DataFrame], out_dir: Path, quarter: str) -> None:
    """Hive layout: data/<TABLE>/cycle=<quarter>/data.parquet (zstd compressed)."""
    for name, df in tables.items():
        part_dir = out_dir / name / f"cycle={quarter}"
        part_dir.mkdir(parents=True, exist_ok=True)
        path = part_dir / "data.parquet"
        # drop the partition column from the file body; it's encoded in the path
        df.drop(columns=["cycle"]).to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    total = sum(p.stat().st_size for p in out_dir.rglob("*.parquet"))
    print(f"  parquet dataset now {total / 1e6:.1f} MB total under {out_dir}/")


def write_sqlite(tables: dict[str, pd.DataFrame], db_path: Path, quarter: str) -> None:
    """One table per NCUA file. Re-running a quarter replaces that quarter's rows."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        for name, df in tables.items():
            existing = con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
            ).fetchone()
            if existing:
                con.execute(f'DELETE FROM "{name}" WHERE cycle = ?', (quarter,))
            df.to_sql(name, con, if_exists="append", index=False)
        con.commit()
    print(f"  sqlite written to {db_path} ({db_path.stat().st_size / 1e6:.1f} MB)")


def ingest_quarter(quarter: str, fmt: str, out_dir: Path, keep_zip: bool) -> None:
    print(f"[{quarter}]")
    url = build_url(quarter)
    with tempfile.TemporaryDirectory() as tmp:
        zip_dest = (out_dir / "_zips" / f"call-report-data-{quarter}.zip") if keep_zip \
            else Path(tmp) / f"{quarter}.zip"
        zip_dest.parent.mkdir(parents=True, exist_ok=True)
        if not (keep_zip and zip_dest.exists()):
            download_zip(url, zip_dest)
        else:
            print(f"  using cached {zip_dest}")
        tables = load_quarter_from_zip(zip_dest, quarter)

    if fmt == "parquet":
        write_parquet(tables, out_dir / "data", quarter)
    else:
        write_sqlite(tables, out_dir / f"ncua_{quarter}.db", quarter)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Ingest NCUA Call Report quarters.")
    ap.add_argument("--quarter", action="append", required=True,
                    help="YYYY-MM, e.g. 2025-09. Repeat for multiple quarters.")
    ap.add_argument("--format", choices=["parquet", "sqlite"], default="parquet")
    ap.add_argument("--out", default=".", help="Output root directory (default: .)")
    ap.add_argument("--keep-zip", action="store_true",
                    help="Cache downloaded ZIPs under <out>/_zips for re-runs.")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    succeeded, failed = [], []
    for q in args.quarter:
        try:
            ingest_quarter(q, args.format, out_dir, args.keep_zip)
            succeeded.append(q)
        except Exception as exc:  # noqa: BLE001 - one bad quarter shouldn't abort a batch
            print(f"  !! skipping {q}: {exc}")
            failed.append(q)
    print(f"done. succeeded: {succeeded or 'none'} | failed/skipped: {failed or 'none'}")
    # Return success if at least one quarter came through, so good data still commits.
    return 0 if succeeded else 1


if __name__ == "__main__":
    sys.exit(main())
