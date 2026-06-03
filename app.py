"""
NCUA Call Report Explorer  (Streamlit + DuckDB over Parquet)
============================================================

Tabs:
  Profile  -- scorecard (incl. growth), trend charts across quarters, peer
              benchmarking, and raw-table browse for one credit union.
  Rankings -- screen/sort all credit unions by any metric (incl. growth),
              filtered by state and asset size.

Notes:
- NCUA mixes column casing (ACCT_115 vs Acct_661A); DuckDB matches identifiers
  case-insensitively, so the SQL uses one consistent ACCT_ form.
- YTD income items are annualized by 12/quarter-month so ROA/ROE/NIM compare
  across quarters. Balance-sheet ratios use period-end balances.
- Growth = vs the prior available quarter, annualized. Extreme values usually
  indicate a merger/acquisition rather than organic growth.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NCUA Call Report Explorer", layout="wide")

DATA_DIR = Path("data")
SKIP_TABLES = {"Readme", "Report1"}
BROWSE_SKIP = SKIP_TABLES | {"AcctDesc", "FOICUDES", "Acct_DescTradeNames"}

# key, label, format, direction ("high"/"low" = better, None = level)
METRICS = [
    ("assets", "Total Assets", "money", None),
    ("loans", "Loans & Leases", "money", None),
    ("shares", "Shares & Deposits", "money", None),
    ("net_worth", "Net Worth", "money", None),
    ("net_income", "Net Income (YTD)", "money", None),
    ("members", "Members", "int", None),
    ("roa", "ROA", "pct", "high"),
    ("roe", "ROE", "pct", "high"),
    ("nim", "Net Interest Margin", "pct", "high"),
    ("efficiency", "Efficiency Ratio", "pct", "low"),
    ("nw_ratio", "Net Worth Ratio", "pct", "high"),
    ("lts", "Loan-to-Share", "pct", None),
    ("delinquency", "Delinquency Ratio", "pct", "low"),
    ("nco", "Net Charge-Off Ratio", "pct", "low"),
    ("assets_growth", "Asset Growth", "pct", "high"),
    ("members_growth", "Member Growth", "pct", "high"),
    ("loans_growth", "Loan Growth", "pct", "high"),
    ("shares_growth", "Share Growth", "pct", "high"),
]
META = {k: (lbl, fmt, dirn) for k, lbl, fmt, dirn in METRICS}
GROWTH_KEYS = ["assets_growth", "members_growth", "loans_growth", "shares_growth"]

BANDS = [
    (0, 10e6, "< $10M"), (10e6, 50e6, "$10M–50M"), (50e6, 100e6, "$50M–100M"),
    (100e6, 250e6, "$100M–250M"), (250e6, 500e6, "$250M–500M"),
    (500e6, 1e9, "$500M–1B"), (1e9, float("inf"), "$1B+"),
]


@st.cache_resource
def get_con():
    return duckdb.connect()


con = get_con()


def glob_for(table):
    return f"{DATA_DIR.as_posix()}/{table}/**/*.parquet"


def available_tables():
    if not DATA_DIR.exists():
        return []
    return sorted(p.name for p in DATA_DIR.iterdir()
                  if p.is_dir() and any(p.glob("**/*.parquet")))


def band_of(assets):
    if assets is None or pd.isna(assets):
        return "Unknown"
    for lo, hi, lbl in BANDS:
        if lo <= assets < hi:
            return lbl
    return "Unknown"


def qidx(c):
    y, m = c.split("-")
    return int(y) * 4 + int(m) // 3


def money(x):
    return f"${x:,.0f}" if isinstance(x, (int, float)) and not pd.isna(x) else "—"


def intfmt(x):
    return f"{x:,.0f}" if isinstance(x, (int, float)) and not pd.isna(x) else "—"


def pct(x):
    return f"{x:.1f}%" if isinstance(x, (int, float)) and not pd.isna(x) else "—"


def fmt(key, x):
    f = META[key][1]
    return money(x) if f == "money" else intfmt(x) if f == "int" else pct(x)


@st.cache_data(show_spinner=False)
def cycles():
    rows = con.execute(
        f"SELECT DISTINCT cycle FROM read_parquet('{glob_for('FOICU')}', "
        "hive_partitioning=true) ORDER BY cycle DESC"
    ).fetchall()
    return [r[0] for r in rows]


def prior_cycle(cycle):
    cs = sorted(cycles())
    if cycle in cs:
        i = cs.index(cycle)
        return cs[i - 1] if i > 0 else None
    return None


def yoy_cycle(cycle):
    """Same quarter one year earlier, if present in the data."""
    y, m = cycle.split("-")
    cand = f"{int(y) - 1}-{m}"
    return cand if cand in cycles() else None


@st.cache_data(show_spinner=False)
def acct_names():
    try:
        df = con.execute(
            f"SELECT Account, AcctName FROM read_parquet('{glob_for('AcctDesc')}', "
            "hive_partitioning=true)"
        ).df()
        return {str(a).upper(): str(n) for a, n in zip(df["Account"], df["AcctName"])}
    except Exception:
        return {}


@st.cache_data(show_spinner="Computing metrics for all credit unions…")
def metrics_table(cycle):
    annualize = 12 / int(cycle[-2:])
    d = con.execute(f"""
      SELECT o.CU_NUMBER AS cu, o.CU_NAME AS cu_name, COALESCE(o.STATE, '') AS state,
        TRY_CAST(f.ACCT_010  AS DOUBLE) AS assets,
        TRY_CAST(f.ACCT_025B AS DOUBLE) AS loans,
        TRY_CAST(f.ACCT_018  AS DOUBLE) AS shares,
        TRY_CAST(f.ACCT_671  AS DOUBLE) AS opex,
        TRY_CAST(f.ACCT_083  AS DOUBLE) AS members,
        TRY_CAST(f.ACCT_550  AS DOUBLE) AS chargeoffs,
        TRY_CAST(f.ACCT_551  AS DOUBLE) AS recoveries,
        TRY_CAST(f.ACCT_041B AS DOUBLE) AS delinquent,
        TRY_CAST(a.ACCT_997  AS DOUBLE) AS net_worth,
        TRY_CAST(a.ACCT_661A AS DOUBLE) AS net_income,
        TRY_CAST(a.ACCT_115  AS DOUBLE) AS int_income,
        TRY_CAST(a.ACCT_350  AS DOUBLE) AS int_expense,
        TRY_CAST(a.ACCT_117  AS DOUBLE) AS non_int_income
      FROM read_parquet('{glob_for('FOICU')}',  hive_partitioning=true) o
      JOIN read_parquet('{glob_for('FS220')}',  hive_partitioning=true) f
        ON o.CU_NUMBER=f.CU_NUMBER AND o.cycle=f.cycle
      JOIN read_parquet('{glob_for('FS220A')}', hive_partitioning=true) a
        ON o.CU_NUMBER=a.CU_NUMBER AND o.cycle=a.cycle
      WHERE o.cycle = ?
    """, [cycle]).df()
    nii = d.int_income - d.int_expense

    def ratio(num, den):
        return np.where(den.notna() & (den != 0), num / den * 100, np.nan)

    d["roa"] = ratio(d.net_income * annualize, d.assets)
    d["roe"] = ratio(d.net_income * annualize, d.net_worth)
    d["nim"] = ratio(nii * annualize, d.assets)
    d["efficiency"] = ratio(d.opex, nii + d.non_int_income)
    d["nw_ratio"] = ratio(d.net_worth, d.assets)
    d["lts"] = ratio(d.loans, d.shares)
    d["delinquency"] = ratio(d.delinquent, d.loans)
    d["nco"] = ratio((d.chargeoffs - d.recoveries) * annualize, d.loans)
    d["band"] = d.assets.apply(band_of)
    return d


@st.cache_data(show_spinner=False)
def growth_for(cycle, basis):
    cols = ["assets", "members", "loans", "shares"]
    cur = metrics_table(cycle)[["cu"] + cols].copy()
    if basis == "YoY":
        prior, factor = yoy_cycle(cycle), 1.0
    else:  # QoQ annualized
        prior = prior_cycle(cycle)
        factor = (4 / (qidx(cycle) - qidx(prior))) if prior else 1.0
    if prior is None:
        for c in cols:
            cur[f"{c}_growth"] = np.nan
        return cur[["cu"] + [f"{c}_growth" for c in cols]]
    p = (metrics_table(prior)[["cu"] + cols]
         .rename(columns={c: f"{c}_p" for c in cols}))
    m = cur.merge(p, on="cu", how="left")
    for c in cols:
        base = m[f"{c}_p"]
        m[f"{c}_growth"] = np.where(base > 0, (m[c] / base - 1) * factor * 100, np.nan)
    return m[["cu"] + [f"{c}_growth" for c in cols]]


@st.cache_data(show_spinner=False)
def enriched_table(cycle, basis):
    return metrics_table(cycle).merge(growth_for(cycle, basis), on="cu", how="left")


@st.cache_data(show_spinner=False)
def cu_timeseries(cu):
    rows = []
    for c in sorted(cycles()):
        r = metrics_table(c)
        rr = r[r.cu == cu]
        if not rr.empty:
            d = rr.iloc[0].to_dict()
            d["cycle"] = c
            rows.append(d)
    return pd.DataFrame(rows).set_index("cycle") if rows else pd.DataFrame()


# ---------------------------------------------------------------------------- UI

st.title("NCUA Call Report Explorer")

tables = available_tables()
if "FOICU" not in tables:
    st.error("No data under ./data. Run the ingest (or GitHub Action) and commit data/.")
    st.stop()

all_cycles = cycles()
cycle = st.sidebar.selectbox("Quarter", all_cycles)
growth_label = st.sidebar.selectbox(
    "Growth basis", ["Year-over-year", "Quarter-over-quarter (annualized)"])
basis = "YoY" if growth_label.startswith("Year") else "QoQ"
mt = enriched_table(cycle, basis)

profile_tab, rankings_tab = st.tabs(["Profile", "Rankings"])

# ============================================================ PROFILE
with profile_tab:
    query = st.text_input("Search a credit union by name", placeholder="e.g. BluCurrent")
    if not query:
        st.info("Type part of a credit union name to begin.")
    else:
        hits = mt[mt.cu_name.str.contains(query, case=False, na=False)].head(300)
        st.caption(f"{len(hits)} match(es) in {cycle}")
        if not hits.empty:
            labels = {r.cu: f"{r.cu_name}  (#{r.cu}, {r.state})" for r in hits.itertuples()}
            cu = st.selectbox("Select a credit union", list(labels),
                              format_func=lambda n: labels[n])
            row = mt[mt.cu == cu].iloc[0]
            st.subheader(labels[cu])
            st.caption(f"Asset peer group: {row.band}")

            c1 = st.columns(5)
            for col, key in zip(c1, ["assets", "loans", "shares", "net_worth", "net_income"]):
                col.metric(META[key][0], fmt(key, row[key]))
            c2 = st.columns(5)
            for col, key in zip(c2, ["roa", "roe", "nim", "nw_ratio", "members"]):
                col.metric(META[key][0], fmt(key, row[key]))
            c3 = st.columns(4)
            for col, key in zip(c3, ["efficiency", "lts", "delinquency", "nco"]):
                col.metric(META[key][0], fmt(key, row[key]))
            c4 = st.columns(4)
            for col, key in zip(c4, GROWTH_KEYS):
                col.metric(META[key][0], fmt(key, row[key]))
            st.caption(f"Growth basis: {growth_label.lower()}")

            with st.expander("Efficiency Ratio breakdown"):
                nii = row.int_income - row.int_expense
                rev = nii + row.non_int_income
                bd = pd.DataFrame([
                    ("Total Interest Income", money(row.int_income)),
                    ("− Total Interest Expense", money(row.int_expense)),
                    ("= Net Interest Income", money(nii)),
                    ("+ Non-Interest Income", money(row.non_int_income)),
                    ("= Revenue (denominator)", money(rev)),
                    ("Operating Expense (numerator)", money(row.opex)),
                    ("Efficiency Ratio", pct(row.efficiency)),
                ], columns=["Component", "Value"])
                st.dataframe(bd, use_container_width=True, hide_index=True)

            # --- trends across quarters ---
            if len(all_cycles) > 1:
                st.subheader("Trends across quarters")
                ts = cu_timeseries(cu)
                trend_opts = [k for k, _, _, _ in METRICS if not k.endswith("_growth")]
                chosen = st.multiselect(
                    "Metrics to chart", trend_opts,
                    default=["assets", "roa", "efficiency", "delinquency"],
                    format_func=lambda k: META[k][0],
                )
                if not ts.empty and chosen:
                    grid = st.columns(2)
                    for i, key in enumerate(chosen):
                        with grid[i % 2]:
                            st.caption(META[key][0])
                            st.line_chart(ts[[key]].rename(columns={key: META[key][0]}))

            # --- peer benchmarking ---
            st.subheader("Peer benchmarking")
            basis = st.radio(
                "Compare against",
                ["Similar asset size", f"Same state ({row.state})",
                 "Same state + asset size", "All credit unions"],
                horizontal=True,
            )
            peers = mt
            if "asset size" in basis and "state" not in basis:
                peers = mt[mt.band == row.band]
            elif basis.startswith("Same state ("):
                peers = mt[mt.state == row.state]
            elif "state + asset" in basis:
                peers = mt[(mt.state == row.state) & (mt.band == row.band)]
            st.caption(f"Peer group: {len(peers):,} credit unions")

            ratio_keys = ["roa", "roe", "nim", "efficiency", "nw_ratio", "lts",
                          "delinquency", "nco"] + GROWTH_KEYS
            bench = []
            for key in ratio_keys:
                lbl, _, dirn = META[key]
                v = row[key]
                series = peers[key].dropna()
                if pd.isna(v) or series.empty:
                    bench.append((lbl, fmt(key, v), "—", "—"))
                    continue
                med = series.median()
                if dirn == "high":
                    rank = f"better than {(series < v).mean() * 100:.0f}% of peers"
                elif dirn == "low":
                    rank = f"better than {(series > v).mean() * 100:.0f}% of peers"
                else:
                    rank = f"{(series < v).mean() * 100:.0f}th percentile"
                bench.append((lbl, fmt(key, v), fmt(key, med), rank))
            st.dataframe(
                pd.DataFrame(bench, columns=["Metric", "This CU", "Peer median", "Standing"]),
                use_container_width=True, hide_index=True,
            )

            with st.expander("Identity / FOICU fields"):
                foicu = con.execute(
                    f"SELECT * FROM read_parquet('{glob_for('FOICU')}', hive_partitioning=true) "
                    "WHERE cycle = ? AND CU_NUMBER = ?", [cycle, cu]).df()
                st.dataframe(foicu.T, use_container_width=True)

            st.subheader("Browse a data table")
            table = st.selectbox("Table", [t for t in tables if t not in BROWSE_SKIP])
            try:
                raw = con.execute(
                    f"SELECT * FROM read_parquet('{glob_for(table)}', hive_partitioning=true) "
                    "WHERE cycle = ? AND CU_NUMBER = ?", [cycle, cu]).df()
            except Exception as e:
                st.warning(f"Could not read {table}: {e}")
                raw = pd.DataFrame()
            if raw.empty:
                st.write("No rows for this credit union in this table.")
            else:
                out = raw.T.reset_index()
                out.columns = ["account"] + [f"value{i}" if i else "value"
                                             for i in range(out.shape[1] - 1)]
                out.insert(1, "description",
                           out["account"].str.upper().map(acct_names()).fillna(""))
                st.dataframe(out, use_container_width=True, height=500)

# ============================================================ RANKINGS
with rankings_tab:
    st.subheader("Screen & rank all credit unions")
    f1, f2, f3 = st.columns([2, 2, 1])
    all_states = sorted(s for s in mt.state.unique() if s)
    sel_states = f1.multiselect("State(s)", all_states, default=[])
    sel_bands = f2.multiselect("Asset size", [b[2] for b in BANDS], default=[])
    top_n = f3.number_input("Show top", min_value=10, max_value=2000, value=100, step=10)

    rankable = [k for k, _, _, _ in METRICS]
    g1, g2 = st.columns([2, 1])
    rank_key = g1.selectbox("Rank by", rankable,
                            format_func=lambda k: META[k][0], index=rankable.index("assets"))
    default_desc = META[rank_key][2] != "low"
    order = g2.radio("Order", ["Top (high→low)", "Bottom (low→high)"],
                     index=0 if default_desc else 1, horizontal=True)
    if rank_key in GROWTH_KEYS:
        st.caption("Heads-up: extreme growth usually reflects a merger/acquisition, not organic growth.")

    view = mt.copy()
    if sel_states:
        view = view[view.state.isin(sel_states)]
    if sel_bands:
        view = view[view.band.isin(sel_bands)]
    view = view.dropna(subset=[rank_key])
    view = view.sort_values(rank_key, ascending=order.startswith("Bottom")).head(int(top_n))

    show_keys = ["assets", "net_worth", "roa", "efficiency", "nw_ratio", "delinquency", "lts"]
    if rank_key not in show_keys:
        show_keys.insert(0, rank_key)
    disp = pd.DataFrame({"Credit Union": view.cu_name.values, "State": view.state.values})
    for k in show_keys:
        disp[META[k][0]] = [fmt(k, x) for x in view[k].values]
    disp.insert(0, "Rank", range(1, len(disp) + 1))
    st.caption(f"{len(view):,} credit unions shown (of {len(mt):,} total)")
    st.dataframe(disp, use_container_width=True, hide_index=True, height=600)
