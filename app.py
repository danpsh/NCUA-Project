"""
NCUA Call Report Explorer (Streamlit + DuckDB over Parquet)
===========================================================

Reads the partitioned Parquet dataset produced by ingest_ncua.py. DuckDB only
touches the columns/rows a query needs, so the wide FS220 tables never get
loaded into memory wholesale -- which is what made the raw-CSV version fall over
on Streamlit's free tier.

Expects a layout like:
    data/FOICU/cycle=2025-09/data.parquet
    data/FS220/cycle=2025-09/data.parquet
    ...
"""

from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NCUA Call Report Explorer", layout="wide")

DATA_DIR = Path("data")
SKIP_TABLES = {"Readme", "Report1"}  # NCUA help/count files, not data


@st.cache_resource
def get_con() -> duckdb.DuckDBPyConnection:
    return duckdb.connect()


con = get_con()


def glob_for(table: str) -> str:
    return f"{DATA_DIR.as_posix()}/{table}/**/*.parquet"


def available_tables() -> list[str]:
    if not DATA_DIR.exists():
        return []
    return sorted(
        p.name for p in DATA_DIR.iterdir()
        if p.is_dir() and any(p.glob("**/*.parquet"))
    )


@st.cache_data(show_spinner=False)
def cycles() -> list[str]:
    rows = con.execute(
        f"SELECT DISTINCT cycle FROM read_parquet('{glob_for('FOICU')}', "
        "hive_partitioning=true) ORDER BY cycle DESC"
    ).fetchall()
    return [r[0] for r in rows]


@st.cache_data(show_spinner=False)
def acct_names() -> dict:
    """Account code -> readable name from AcctDesc (best effort)."""
    try:
        df = con.execute(
            f"SELECT * FROM read_parquet('{glob_for('AcctDesc')}', hive_partitioning=true)"
        ).df()
        cols = {c.lower(): c for c in df.columns}
        code = cols.get("account")
        name = cols.get("acctname") or cols.get("accountname") or cols.get("acctdesc")
        if code and name:
            return dict(zip(df[code].astype(str), df[name].astype(str)))
    except Exception:
        pass
    return {}


st.title("NCUA Call Report Explorer")

tables = available_tables()
if "FOICU" not in tables:
    st.error(
        "No data found under ./data. Run `python ingest_ncua.py --quarter 2025-09` "
        "locally, then commit the data/ folder."
    )
    st.stop()

cycle = st.sidebar.selectbox("Quarter", cycles())

query = st.text_input("Search a credit union by name", placeholder="e.g. BluCurrent")
if not query:
    st.info("Type part of a credit union name to begin.")
    st.stop()

matches = con.execute(
    f"SELECT * FROM read_parquet('{glob_for('FOICU')}', hive_partitioning=true) "
    "WHERE cycle = ? AND CU_NAME ILIKE ? ORDER BY CU_NAME LIMIT 300",
    [cycle, f"%{query}%"],
).df()

st.caption(f"{len(matches)} match(es) in {cycle}")
if matches.empty:
    st.stop()


def label_for(r: pd.Series) -> str:
    state = r.get("STATE") or r.get("CU_STATE") or ""
    tail = f"#{r['CU_NUMBER']}" + (f", {state}" if state else "")
    return f"{r.get('CU_NAME', '')}  ({tail})"


labels = {r["CU_NUMBER"]: label_for(r) for _, r in matches.iterrows()}
cu = st.selectbox("Select a credit union", list(labels), format_func=lambda n: labels[n])

st.subheader(labels[cu])

foicu_row = con.execute(
    f"SELECT * FROM read_parquet('{glob_for('FOICU')}', hive_partitioning=true) "
    "WHERE cycle = ? AND CU_NUMBER = ?",
    [cycle, cu],
).df()
with st.expander("Identity / FOICU fields", expanded=True):
    st.dataframe(foicu_row.T, use_container_width=True)

st.subheader("Browse a data table")
table = st.selectbox("Table", [t for t in tables if t not in SKIP_TABLES])

try:
    row = con.execute(
        f"SELECT * FROM read_parquet('{glob_for(table)}', hive_partitioning=true) "
        "WHERE cycle = ? AND CU_NUMBER = ?",
        [cycle, cu],
    ).df()
except Exception as e:
    st.warning(f"Could not read {table}: {e}")
    row = pd.DataFrame()

if row.empty:
    st.write("No rows for this credit union in this table.")
else:
    out = row.T.reset_index()
    out.columns = ["account"] + [f"value{i}" if i else "value" for i in range(out.shape[1] - 1)]
    names = acct_names()
    if names:
        out.insert(1, "description", out["account"].map(names).fillna(""))
    st.dataframe(out, use_container_width=True, height=600)
