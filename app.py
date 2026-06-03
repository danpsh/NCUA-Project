"""
NCUA Call Report Explorer  (Streamlit + DuckDB over Parquet)
============================================================

Tabs:
  Profile  -- scorecard, watch flags, trend charts, peer benchmarking
              (incl. custom peer sets), raw-table browse, Excel export.
  Compare  -- side-by-side scorecards for several CUs, trend overlay, Excel.
  Rankings -- screen/sort all CUs by any metric, filtered by state & size.
  Movers   -- biggest YoY gainers/decliners; likely-merger flags.
  Industry -- system-wide totals & medians over time, and by state.

Notes:
- read_parquet uses union_by_name so NCUA's column changes across years don't
  break older quarters (missing fields read as NULL -> shown as "—").
- Functions that depend on the SET of available quarters take a cycle_sig arg
  so their cache refreshes when new data lands.
"""

import io
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
# Derived ratings (not raw call-report metrics, so kept out of the METRICS loops
# but given META entries so fmt() and labels work in rankings/compare).
META["score"] = ("Composite Score", "score", "high")
META["stars"] = ("Star Rating", "stars", "high")
GROWTH_KEYS = ["assets_growth", "members_growth", "loans_growth", "shares_growth"]

# Composite score: weighted blend of band-relative percentiles (capital, earnings,
# efficiency, asset quality, growth). Weights sum to 1.0.
SCORE_WEIGHTS = [
    ("roa", 0.25), ("nw_ratio", 0.20), ("efficiency", 0.20),
    ("delinquency", 0.15), ("nco", 0.10), ("assets_growth", 0.10),
]

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
    return f"{x:.2f}%" if isinstance(x, (int, float)) and not pd.isna(x) else "—"


def stars_str(n):
    if n is None or pd.isna(n):
        return "—"
    n = int(n)
    return "★" * n + "☆" * (5 - n)


def stars_from_score(s):
    if pd.isna(s):
        return np.nan
    return min(5, max(1, int(s // 20) + 1))  # 0-20→1 … 80-100→5


def fmt(key, x):
    f = META[key][1]
    if f == "score":
        return f"{x:.0f}" if isinstance(x, (int, float)) and not pd.isna(x) else "—"
    if f == "stars":
        return stars_str(x)
    return money(x) if f == "money" else intfmt(x) if f == "int" else pct(x)


def to_excel_bytes(sheets):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31])
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def cycles():
    rows = con.execute(
        f"SELECT DISTINCT cycle FROM read_parquet('{glob_for('FOICU')}', "
        "hive_partitioning=true, union_by_name=true) ORDER BY cycle DESC"
    ).fetchall()
    return [r[0] for r in rows]


def prior_cycle(cycle):
    cs = sorted(cycles())
    if cycle in cs:
        i = cs.index(cycle)
        return cs[i - 1] if i > 0 else None
    return None


def yoy_cycle(cycle):
    y, m = cycle.split("-")
    cand = f"{int(y) - 1}-{m}"
    return cand if cand in cycles() else None


@st.cache_data(show_spinner=False)
def acct_names():
    try:
        df = con.execute(
            f"SELECT Account, AcctName FROM read_parquet('{glob_for('AcctDesc')}', "
            "hive_partitioning=true, union_by_name=true)"
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
      FROM read_parquet('{glob_for('FOICU')}',  hive_partitioning=true, union_by_name=true) o
      JOIN read_parquet('{glob_for('FS220')}',  hive_partitioning=true, union_by_name=true) f
        ON o.CU_NUMBER=f.CU_NUMBER AND o.cycle=f.cycle
      JOIN read_parquet('{glob_for('FS220A')}', hive_partitioning=true, union_by_name=true) a
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
def growth_for(cycle, basis, cycle_sig):
    cols = ["assets", "members", "loans", "shares"]
    cur = metrics_table(cycle)[["cu"] + cols].copy()
    if basis == "YoY":
        prior, factor = yoy_cycle(cycle), 1.0
    else:
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
def enriched_table(cycle, basis, cycle_sig):
    df = metrics_table(cycle).merge(growth_for(cycle, basis, cycle_sig), on="cu", how="left")
    # Composite score: weighted blend of band-relative "goodness" percentiles.
    acc = pd.Series(0.0, index=df.index)
    wsum = pd.Series(0.0, index=df.index)
    for key, w in SCORE_WEIGHTS:
        pr = df.groupby("band")[key].rank(pct=True)          # 0–1 within asset band
        good = pr if META[key][2] == "high" else 1 - pr      # flip low-is-better metrics
        df[f"g_{key}"] = good
        acc = acc + good.fillna(0) * w
        wsum = wsum + good.notna().astype(float) * w
    df["score"] = (acc / wsum * 100).where(wsum > 0)
    df["stars"] = df["score"].apply(stars_from_score)
    return df


@st.cache_data(show_spinner=False)
def cu_timeseries(cu, cycle_sig):
    rows = []
    for c in cycle_sig:
        rr = metrics_table(c)
        rr = rr[rr.cu == cu]
        if not rr.empty:
            d = rr.iloc[0].to_dict()
            d["cycle"] = c
            rows.append(d)
    return pd.DataFrame(rows).set_index("cycle") if rows else pd.DataFrame()


@st.cache_data(show_spinner="Building industry history…")
def industry_timeseries(cycle_sig):
    rows = []
    for c in sorted(cycle_sig):
        m = metrics_table(c)
        rows.append({
            "cycle": c,
            "Total Assets": m.assets.sum(),
            "Credit Unions": int(m.assets.notna().sum()),
            "Total Members": m.members.sum(),
            "Total Net Worth": m.net_worth.sum(),
            "Median ROA": m.roa.median(),
            "Median Efficiency": m.efficiency.median(),
            "Median Net Worth Ratio": m.nw_ratio.median(),
        })
    return pd.DataFrame(rows).set_index("cycle")


def compute_flags(row, mt):
    flags = []
    if pd.notna(row.nw_ratio):
        if row.nw_ratio < 6:
            flags.append(("error", f"Net worth ratio {row.nw_ratio:.2f}% — below 6% (PCA: undercapitalized)"))
        elif row.nw_ratio < 7:
            flags.append(("warning", f"Net worth ratio {row.nw_ratio:.2f}% — 6–7% (adequately, not well, capitalized)"))
    if pd.notna(row.roa) and row.roa < 0:
        flags.append(("error", f"Negative ROA ({row.roa:.2f}%)"))
    if pd.notna(row.delinquency):
        p90 = mt.delinquency.quantile(0.90)
        if row.delinquency >= p90:
            flags.append(("warning", f"Delinquency {row.delinquency:.2f}% — worst 10% of all CUs (≥ {p90:.2f}%)"))
    if pd.notna(row.nco):
        p90 = mt.nco.quantile(0.90)
        if row.nco >= p90:
            flags.append(("warning", f"Net charge-offs {row.nco:.2f}% — worst 10% of all CUs (≥ {p90:.2f}%)"))
    if pd.notna(row.efficiency) and row.efficiency > 90:
        flags.append(("warning", f"Efficiency ratio {row.efficiency:.2f}% — above 90% (cost-heavy)"))
    return flags


def comparison_frames(cus, mt, labels):
    keys = ["score", "stars"] + [k for k, _, _, _ in METRICS]
    disp, raw = {}, {}
    for cu in cus:
        r = mt[mt.cu == cu].iloc[0]
        lbl = labels[cu]
        disp[lbl] = {META[k][0]: fmt(k, r[k]) for k in keys}
        raw[lbl] = {META[k][0]: r[k] for k in keys}
    order = [META[k][0] for k in keys]
    return pd.DataFrame(disp).reindex(order), pd.DataFrame(raw).reindex(order)


def score_breakdown(row):
    rows = []
    for key, w in SCORE_WEIGHTS:
        g = row.get(f"g_{key}")
        rows.append((META[key][0], fmt(key, row[key]),
                     f"{g * 100:.0f}th" if pd.notna(g) else "—", f"{w * 100:.0f}%"))
    return pd.DataFrame(rows, columns=["Metric", "Value", "Band percentile", "Weight"])


def multi_cu_series(cus, metric, labels, cycle_sig):
    out = {}
    for cu in cus:
        ts = cu_timeseries(cu, cycle_sig)
        if not ts.empty and metric in ts:
            out[labels[cu]] = ts[metric]
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------- UI

st.title("NCUA Call Report Explorer")

tables = available_tables()
if "FOICU" not in tables:
    st.error("No data under ./data. Run the ingest (or GitHub Action) and commit data/.")
    st.stop()

all_cycles = cycles()
sig = tuple(sorted(all_cycles))
cycle = st.sidebar.selectbox("Quarter", all_cycles)
growth_label = st.sidebar.selectbox(
    "Growth basis", ["Year-over-year", "Quarter-over-quarter (annualized)"])
basis = "YoY" if growth_label.startswith("Year") else "QoQ"
mt = enriched_table(cycle, basis, sig)

# label helpers shared across tabs
ALL_LABELS = {r.cu: f"{r.cu_name} (#{r.cu}, {r.state})" for r in mt.itertuples()}

profile_tab, compare_tab, rankings_tab, movers_tab, industry_tab = st.tabs(
    ["Profile", "Compare", "Rankings", "Movers", "Industry"])

# ============================================================ PROFILE
with profile_tab:
    query = st.text_input("Search a credit union by name", placeholder="e.g. BluCurrent")
    if not query:
        st.info("Type part of a credit union name to begin.")
    else:
        hits = mt[mt.cu_name.str.contains(query, case=False, na=False)].head(300)
        st.caption(f"{len(hits)} match(es) in {cycle}")
        if not hits.empty:
            labels = {r.cu: ALL_LABELS[r.cu] for r in hits.itertuples()}
            cu = st.selectbox("Select a credit union", list(labels),
                              format_func=lambda n: labels[n])
            row = mt[mt.cu == cu].iloc[0]
            st.subheader(labels[cu])
            st.caption(f"Asset peer group: {row.band}")

            # composite rating
            rc = st.columns([1, 3])
            rc[0].metric("Composite Score",
                         f"{row.score:.0f}/100" if pd.notna(row.score) else "—")
            with rc[1]:
                st.markdown(
                    f"<div style='font-size:2.2rem;line-height:1'>{stars_str(row.stars)}</div>",
                    unsafe_allow_html=True)
                st.caption(f"Overall performance vs the {row.band} asset peer group, "
                           "weighting earnings, capital, efficiency, asset quality, and growth.")
            with st.expander("How this score is built"):
                st.dataframe(score_breakdown(row), use_container_width=True, hide_index=True)

            flags = compute_flags(row, mt)
            if flags:
                for sev, msg in flags:
                    (st.error if sev == "error" else st.warning)(msg)
            else:
                st.success("No watch flags — capital, earnings, and asset quality look sound.")

            scorecard_groups = [
                ("Size & membership", ["assets", "loans", "shares", "net_worth", "members"]),
                ("Earnings", ["roa", "roe", "nim", "net_income", "efficiency"]),
                ("Capital & asset quality", ["nw_ratio", "delinquency", "nco", "lts"]),
                (f"Growth ({growth_label.lower()})", GROWTH_KEYS),
            ]
            for title, keys in scorecard_groups:
                st.markdown(f"**{title}**")
                cols = st.columns(len(keys))
                for col, key in zip(cols, keys):
                    col.metric(META[key][0], fmt(key, row[key]))

            # one-click Excel export of this CU's scorecard
            sc = pd.DataFrame(
                {"Value": {META[k][0]: fmt(k, row[k]) for k, _, _, _ in METRICS}})
            st.download_button(
                "Download scorecard (Excel)",
                to_excel_bytes({"Scorecard": sc}),
                file_name=f"{row.cu_name}_{cycle}_scorecard.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

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

            if len(all_cycles) > 1:
                st.subheader("Trends across quarters")
                ts = cu_timeseries(cu, sig)
                trend_opts = [k for k, _, _, _ in METRICS if not k.endswith("_growth")]
                chosen = st.multiselect(
                    "Metrics to chart", trend_opts,
                    default=["assets", "roa", "efficiency", "delinquency"],
                    format_func=lambda k: META[k][0])
                if not ts.empty and chosen:
                    grid = st.columns(2)
                    for i, key in enumerate(chosen):
                        with grid[i % 2]:
                            st.caption(META[key][0])
                            st.line_chart(ts[[key]].rename(columns={key: META[key][0]}))

            st.subheader("Peer benchmarking")
            basis_choice = st.radio(
                "Compare against",
                ["Similar asset size", f"Same state ({row.state})",
                 "Same state + asset size", "All credit unions", "Custom (pick CUs)"],
                horizontal=True)
            if basis_choice == "Custom (pick CUs)":
                picks = st.multiselect("Choose peer credit unions",
                                       list(ALL_LABELS), format_func=lambda n: ALL_LABELS[n])
                peers = mt[mt.cu.isin(picks + [cu])] if picks else mt.iloc[0:0]
            elif basis_choice == "Similar asset size":
                peers = mt[mt.band == row.band]
            elif basis_choice.startswith("Same state ("):
                peers = mt[mt.state == row.state]
            elif "state + asset" in basis_choice:
                peers = mt[(mt.state == row.state) & (mt.band == row.band)]
            else:
                peers = mt
            st.caption(f"Peer group: {len(peers):,} credit unions")

            if len(peers) >= 2:
                ratio_keys = ["roa", "roe", "nim", "efficiency", "nw_ratio", "lts",
                              "delinquency", "nco"] + GROWTH_KEYS
                bench = []
                for key in ratio_keys:
                    lbl, _, dirn = META[key]
                    v = row[key]
                    s = peers[key].dropna()
                    if pd.isna(v) or s.empty:
                        bench.append((lbl, fmt(key, v), "—", "—"))
                        continue
                    med = s.median()
                    if dirn == "high":
                        rank = f"better than {(s < v).mean() * 100:.0f}% of peers"
                    elif dirn == "low":
                        rank = f"better than {(s > v).mean() * 100:.0f}% of peers"
                    else:
                        rank = f"{(s < v).mean() * 100:.0f}th percentile"
                    bench.append((lbl, fmt(key, v), fmt(key, med), rank))
                st.dataframe(pd.DataFrame(
                    bench, columns=["Metric", "This CU", "Peer median", "Standing"]),
                    use_container_width=True, hide_index=True)
            else:
                st.info("Pick at least one peer to benchmark against.")

            with st.expander("Identity / FOICU fields"):
                foicu = con.execute(
                    f"SELECT * FROM read_parquet('{glob_for('FOICU')}', hive_partitioning=true, union_by_name=true) "
                    "WHERE cycle = ? AND CU_NUMBER = ?", [cycle, cu]).df()
                st.dataframe(foicu.T, use_container_width=True)

            st.subheader("Browse a data table")
            table = st.selectbox("Table", [t for t in tables if t not in BROWSE_SKIP])
            try:
                raw = con.execute(
                    f"SELECT * FROM read_parquet('{glob_for(table)}', hive_partitioning=true, union_by_name=true) "
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
                st.dataframe(out, use_container_width=True, height=400)

# ============================================================ COMPARE
with compare_tab:
    st.subheader("Compare credit unions side by side")
    picks = st.multiselect("Pick 2–5 credit unions", list(ALL_LABELS),
                           format_func=lambda n: ALL_LABELS[n], max_selections=5)
    if len(picks) < 2:
        st.info("Choose at least two credit unions to compare.")
    else:
        disp, raw = comparison_frames(picks, mt, ALL_LABELS)
        st.dataframe(disp, use_container_width=True)
        st.download_button(
            "Download comparison (Excel)",
            to_excel_bytes({"Comparison": raw}),
            file_name=f"cu_comparison_{cycle}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        if len(all_cycles) > 1:
            st.subheader("Trend overlay")
            ov_key = st.selectbox(
                "Metric", [k for k, _, _, _ in METRICS if not k.endswith("_growth")],
                format_func=lambda k: META[k][0], index=0)
            series = multi_cu_series(picks, ov_key, ALL_LABELS, sig)
            if not series.empty:
                st.line_chart(series)

# ============================================================ RANKINGS
with rankings_tab:
    st.subheader("Screen & rank all credit unions")
    f1, f2, f3 = st.columns([2, 2, 1])
    all_states = sorted(s for s in mt.state.unique() if s)
    sel_states = f1.multiselect("State(s)", all_states, default=[])
    sel_bands = f2.multiselect("Asset size", [b[2] for b in BANDS], default=[])
    top_n = f3.number_input("Show top", min_value=10, max_value=2000, value=100, step=10)
    rankable = ["score"] + [k for k, _, _, _ in METRICS]
    g1, g2 = st.columns([2, 1])
    rank_key = g1.selectbox("Rank by", rankable,
                            format_func=lambda k: META[k][0], index=0)
    default_desc = META[rank_key][2] != "low"
    order = g2.radio("Order", ["Top (high→low)", "Bottom (low→high)"],
                     index=0 if default_desc else 1, horizontal=True)
    if rank_key in GROWTH_KEYS:
        st.caption("Heads-up: extreme growth usually reflects a merger/acquisition.")
    view = mt.copy()
    if sel_states:
        view = view[view.state.isin(sel_states)]
    if sel_bands:
        view = view[view.band.isin(sel_bands)]
    view = view.dropna(subset=[rank_key]).sort_values(
        rank_key, ascending=order.startswith("Bottom")).head(int(top_n))
    show_keys = ["score", "stars", "assets", "net_worth", "roa", "efficiency",
                 "nw_ratio", "delinquency"]
    if rank_key not in show_keys:
        show_keys.insert(0, rank_key)
    disp = pd.DataFrame({"Credit Union": view.cu_name.values, "State": view.state.values})
    for k in show_keys:
        disp[META[k][0]] = [fmt(k, x) for x in view[k].values]
    disp.insert(0, "Rank", range(1, len(disp) + 1))
    st.caption(f"{len(view):,} credit unions shown (of {len(mt):,} total)")
    st.dataframe(disp, use_container_width=True, hide_index=True, height=560)

# ============================================================ MOVERS
with movers_tab:
    st.subheader("Biggest movers")
    st.caption(f"Growth basis: {growth_label.lower()}. "
               "Very large jumps often signal a merger/acquisition rather than organic growth.")
    m1, m2 = st.columns([2, 1])
    gkey = m1.selectbox("Growth metric", GROWTH_KEYS, format_func=lambda k: META[k][0])
    min_assets = m2.selectbox("Minimum asset size",
                              ["$10M", "$50M", "$100M", "$500M", "$1B"], index=1)
    floor = {"$10M": 10e6, "$50M": 50e6, "$100M": 100e6, "$500M": 500e6, "$1B": 1e9}[min_assets]
    pool = mt[(mt.assets >= floor)].dropna(subset=[gkey])
    if pool.empty:
        st.info("No credit unions with growth data for this basis yet "
                "(needs a prior-period quarter ingested).")
    else:
        cols = ["cu_name", "state", "assets", gkey]
        gain = pool.nlargest(15, gkey)[cols]
        lose = pool.nsmallest(15, gkey)[cols]

        def fmt_movers(df):
            return pd.DataFrame({
                "Credit Union": df.cu_name.values, "State": df.state.values,
                "Assets": [money(x) for x in df.assets.values],
                META[gkey][0]: [pct(x) for x in df[gkey].values]})

        a, b = st.columns(2)
        with a:
            st.markdown("**Top gainers**")
            st.dataframe(fmt_movers(gain), use_container_width=True, hide_index=True)
        with b:
            st.markdown("**Biggest decliners**")
            st.dataframe(fmt_movers(lose), use_container_width=True, hide_index=True)

# ============================================================ INDUSTRY
with industry_tab:
    st.subheader("Industry overview")
    if len(all_cycles) > 1:
        ind = industry_timeseries(sig)
        a, b = st.columns(2)
        with a:
            st.caption("Total system assets")
            st.line_chart(ind[["Total Assets"]])
        with b:
            st.caption("Number of credit unions")
            st.line_chart(ind[["Credit Unions"]])
        c, d = st.columns(2)
        with c:
            st.caption("Median ROA")
            st.line_chart(ind[["Median ROA"]])
        with d:
            st.caption("Median Efficiency Ratio")
            st.line_chart(ind[["Median Efficiency"]])
        with st.expander("Industry table (all quarters)"):
            tbl = ind.copy()
            for c2 in ["Total Assets", "Total Net Worth"]:
                tbl[c2] = tbl[c2].apply(money)
            tbl["Total Members"] = tbl["Total Members"].apply(intfmt)
            tbl["Credit Unions"] = tbl["Credit Unions"].apply(intfmt)
            for c2 in ["Median ROA", "Median Efficiency", "Median Net Worth Ratio"]:
                tbl[c2] = tbl[c2].apply(pct)
            st.dataframe(tbl, use_container_width=True)
    else:
        st.info("Industry trends need more than one quarter of data.")

    st.subheader(f"By state — {cycle}")
    g = mt.groupby("state").agg(
        CUs=("cu", "size"), total_assets=("assets", "sum"),
        median_roa=("roa", "median"), median_eff=("efficiency", "median")).reset_index()
    g = g[g.state != ""].sort_values("total_assets", ascending=False)
    disp = pd.DataFrame({
        "State": g.state.values, "Credit Unions": [intfmt(x) for x in g.CUs.values],
        "Total Assets": [money(x) for x in g.total_assets.values],
        "Median ROA": [pct(x) for x in g.median_roa.values],
        "Median Efficiency": [pct(x) for x in g.median_eff.values]})
    st.dataframe(disp, use_container_width=True, hide_index=True, height=420)
