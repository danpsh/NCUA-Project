"""
NCUA Call Report Explorer (Streamlit + DuckDB over Parquet)
===========================================================

Reads the partitioned Parquet dataset produced by ingest_ncua.py and presents,
for any credit union, a panel of named headline metrics plus the Efficiency
Ratio (with its components), and a browse view of every underlying table with
readable account names.

NOTE on account codes: NCUA's files are inconsistent about casing -- most
columns are ACCT_115 but some are Acct_661A. Every code lookup here is
case-insensitive (everything is upper-cased before matching).
"""

from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NCUA Call Report Explorer", layout="wide")

DATA_DIR = Path("data")
SKIP_TABLES = {"Readme", "Report1"}
META_COLS = {"CU_NUMBER", "CYCLE_DATE", "JOIN_NUMBER", "UPDATE_DATE", "CYCLE"}

# Headline figures: (label, account-code-without-prefix). Codes are unique across
# the FS220 tables, so we don't need to know which table each lives in.
HEADLINE = [
    ("Total Assets", "010"),
    ("Loans & Leases", "025B"),
    ("Shares & Deposits", "018"),
    ("Total Net Worth", "997"),
    ("Net Income", "661A"),
]

# Efficiency Ratio = Operating Expense / (Net Interest Income + Non-Interest Income)
#   Net Interest Income = Total Interest Income (115) - Total Interest Expense (350)
EFF = {
    "int_income": "115",   # Total Interest Income
    "int_expense": "350",  # Total Interest Expense
    "non_int_income": "117",  # Total Non-Interest Income
    "op_expense": "671",   # Total Non-Interest (Operating) Expense
}


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


def fs_tables(tables: list[str]) -> list[str]:
    return [t for t in tables if t.upper().startswith("FS220")]


@st.cache_data(show_spinner=False)
def cycles() -> list[str]:
    rows = con.execute(
        f"SELECT DISTINCT cycle FROM read_parquet('{glob_for('FOICU')}', "
        "hive_partitioning=true) ORDER BY cycle DESC"
    ).fetchall()
    return [r[0] for r in rows]


@st.cache_data(show_spinner=False)
def acct_names() -> dict:
    """Upper-cased account code -> short readable name (AcctName)."""
    try:
        df = con.execute(
            f"SELECT Account, AcctName FROM read_parquet('{glob_for('AcctDesc')}', "
            "hive_partitioning=true)"
        ).df()
        return {str(a).upper(): str(n) for a, n in zip(df["Account"], df["AcctName"])}
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def cu_account_values(cu: str, cycle: str, fs_table_list: tuple) -> dict:
    """Every account value for one CU across all FS220 tables, keyed UPPER-case."""
    vals: dict = {}
    for t in fs_table_list:
        try:
            df = con.execute(
                f"SELECT * FROM read_parquet('{glob_for(t)}', hive_partitioning=true) "
                "WHERE cycle = ? AND CU_NUMBER = ?",
                [cycle, cu],
            ).df()
        except Exception:
            continue
        if df.empty:
            continue
        row = df.iloc[0]
        for c in df.columns:
            if c.upper() in META_COLS:
                continue
            vals[c.upper()] = row[c]
    return vals


def num(vals: dict, code: str):
    v = vals.get(f"ACCT_{code}".upper())
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def money(x) -> str:
    return f"${x:,.0f}" if isinstance(x, (int, float)) else "—"


def pct(x) -> str:
    return f"{x:.1f}%" if isinstance(x, (int, float)) else "—"


# ----------------------------------------------------------------------------- UI

st.title("NCUA Call Report Explorer")

tables = available_tables()
if "FOICU" not in tables:
    st.error(
        "No data found under ./data. Run `python ingest_ncua.py --quarter 2025-09` "
        "(or the GitHub Action), then make sure the data/ folder is committed."
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

# --- Key metrics --------------------------------------------------------------
vals = cu_account_values(cu, cycle, tuple(fs_tables(tables)))

cols = st.columns(len(HEADLINE))
for col, (label, code) in zip(cols, HEADLINE):
    col.metric(label, money(num(vals, code)))

int_inc = num(vals, EFF["int_income"])
int_exp = num(vals, EFF["int_expense"])
non_int = num(vals, EFF["non_int_income"])
op_exp = num(vals, EFF["op_expense"])
nii = (int_inc - int_exp) if None not in (int_inc, int_exp) else None
denom = (nii + non_int) if None not in (nii, non_int) else None
eff = (op_exp / denom * 100) if (op_exp is not None and denom) else None

assets = num(vals, "010")
nw = num(vals, "997")
loans = num(vals, "025B")
shares = num(vals, "018")
nw_ratio = (nw / assets * 100) if (nw is not None and assets) else None
ls_ratio = (loans / shares * 100) if (loans is not None and shares) else None

r2 = st.columns(3)
r2[0].metric("Efficiency Ratio", pct(eff), help="Operating Expense / (Net Interest Income + Non-Interest Income). Lower is better.")
r2[1].metric("Net Worth Ratio", pct(nw_ratio), help="Total Net Worth / Total Assets.")
r2[2].metric("Loan-to-Share Ratio", pct(ls_ratio), help="Loans & Leases / Shares & Deposits.")

with st.expander("Efficiency Ratio breakdown"):
    breakdown = pd.DataFrame(
        [
            ("Total Interest Income", money(int_inc)),
            ("− Total Interest Expense", money(int_exp)),
            ("= Net Interest Income", money(nii)),
            ("+ Non-Interest Income", money(non_int)),
            ("= Revenue (denominator)", money(denom)),
            ("Operating Expense (numerator)", money(op_exp)),
            ("Efficiency Ratio", pct(eff)),
        ],
        columns=["Component", "Value"],
    )
    st.dataframe(breakdown, use_container_width=True, hide_index=True)

st.divider()

# --- Identity + raw table browser --------------------------------------------
with st.expander("Identity / FOICU fields"):
    foicu_row = con.execute(
        f"SELECT * FROM read_parquet('{glob_for('FOICU')}', hive_partitioning=true) "
        "WHERE cycle = ? AND CU_NUMBER = ?",
        [cycle, cu],
    ).df()
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
    out.insert(1, "description", out["account"].str.upper().map(names).fillna(""))
    st.dataframe(out, use_container_width=True, height=600)
