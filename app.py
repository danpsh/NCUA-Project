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
import numbers
import re
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

# Composite score — GROWTH / MOMENTUM lens: growth weighted heaviest, then
# earnings, with a light asset-quality guardrail. Weights sum to 1.0.
SCORE_WEIGHTS = [
    ("assets_growth", 0.20), ("loans_growth", 0.15),
    ("members_growth", 0.10), ("shares_growth", 0.10),   # growth = 55%
    ("roa", 0.20), ("nim", 0.10),                          # earnings = 30%
    ("efficiency", 0.10),                                  # operating leverage = 10%
    ("delinquency", 0.05),                                 # asset-quality guardrail = 5%
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
    return f"${x:,.0f}" if isinstance(x, numbers.Real) and not pd.isna(x) else "—"


def intfmt(x):
    return f"{x:,.0f}" if isinstance(x, numbers.Real) and not pd.isna(x) else "—"


def pct(x):
    return f"{x:.2f}%" if isinstance(x, numbers.Real) and not pd.isna(x) else "—"


def stars_str(n):
    if n is None or pd.isna(n):
        return "—"
    n = int(n)
    return "★" * n + "☆" * (5 - n)


SEARCH_STOP = {"federal", "credit", "union", "fcu", "cu", "the", "of", "and", "a"}


def name_matches(names, query):
    """Tolerant name search: matches the literal substring, OR all 'distinctive'
    query words (ignoring suffix words like Federal/Credit/Union). So 'ozark
    federal' matches the CU stored simply as 'OZARK'."""
    q = (query or "").strip().lower()
    if not q:
        return pd.Series(False, index=names.index)
    low = names.str.lower()
    matched = low.str.contains(re.escape(q), na=False)
    toks = [t for t in re.split(r"[^a-z0-9]+", q) if t and t not in SEARCH_STOP]
    if toks:
        allt = None
        for t in toks:
            m = low.str.contains(re.escape(t), na=False)
            allt = m if allt is None else (allt & m)
        matched = matched | allt
    return matched


def stars_from_z(z):
    if pd.isna(z):
        return np.nan
    if z >= 0.6:
        return 5
    if z >= 0.2:
        return 4
    if z >= -0.2:
        return 3
    if z >= -0.6:
        return 2
    return 1


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
@st.cache_data(show_spinner=False)
def conversions_table(cycle_sig):
    if not (DATA_DIR / "CONVERSIONS").exists():
        return pd.DataFrame()
    df = con.execute(
        f"SELECT * FROM read_parquet('{glob_for('CONVERSIONS')}', union_by_name=true)").df()
    for c in ("old_charter", "new_charter"):
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df


@st.cache_data(show_spinner=False)
def charter_alias(cycle_sig):
    """old charter -> newest charter id, following multi-step conversion chains."""
    cv = conversions_table(cycle_sig)
    if cv.empty or "old_charter" not in cv.columns:
        return {}
    direct = dict(zip(cv.old_charter, cv.new_charter))

    def resolve(c):
        seen = set()
        while c in direct and c not in seen:
            seen.add(c)
            c = direct[c]
        return c
    return {old: resolve(old) for old in direct}


def canon_charter(cu, alias):
    return alias.get(cu, cu)


def growth_for(cycle, basis, cycle_sig):
    cols = ["assets", "members", "loans", "shares"]
    alias = charter_alias(cycle_sig)
    cur = metrics_table(cycle)[["cu"] + cols].copy()
    cur["_k"] = cur.cu.map(lambda c: alias.get(c, c))   # canonical identity
    if basis == "YoY":
        prior, factor = yoy_cycle(cycle), 1.0
    else:
        prior = prior_cycle(cycle)
        factor = (4 / (qidx(cycle) - qidx(prior))) if prior else 1.0
    if prior is None:
        for c in cols:
            cur[f"{c}_growth"] = np.nan
        return cur[["cu"] + [f"{c}_growth" for c in cols]]
    p = metrics_table(prior)[["cu"] + cols].copy()
    p["_k"] = p.cu.map(lambda c: alias.get(c, c))
    p = (p.rename(columns={c: f"{c}_p" for c in cols}).drop(columns=["cu"])
         .groupby("_k", as_index=False).first())   # collapse any dup canonical keys
    m = cur.merge(p, on="_k", how="left")
    for c in cols:
        base = m[f"{c}_p"]
        m[f"{c}_growth"] = np.where(base > 0, (m[c] / base - 1) * factor * 100, np.nan)
    return m[["cu"] + [f"{c}_growth" for c in cols]]


@st.cache_data(show_spinner=False)
def enriched_table(cycle, basis, cycle_sig):
    df = metrics_table(cycle).merge(growth_for(cycle, basis, cycle_sig), on="cu", how="left")
    # Composite score: weighted blend of band-relative z-scores (SDs from the
    # asset-band peer mean). Direction-adjusted so positive always = better.
    acc = pd.Series(0.0, index=df.index)
    wsum = pd.Series(0.0, index=df.index)
    for key, w in SCORE_WEIGHTS:
        grp = df.groupby("band")[key]
        std = grp.transform("std")
        z = (df[key] - grp.transform("mean")) / std
        z = z.where(std > 0)                              # undefined where no spread
        if META[key][2] == "low":
            z = -z                                        # lower is better → flip sign
        df[f"z_{key}"] = z
        acc = acc + z.fillna(0) * w
        wsum = wsum + z.notna().astype(float) * w
    df["score_z"] = (acc / wsum).where(wsum > 0)          # weighted composite z
    df["score"] = (50 + 30 * df["score_z"]).clip(0, 100)  # 50 = peer average
    df["stars"] = df["score_z"].apply(stars_from_z)
    return df


@st.cache_data(show_spinner=False)
def cu_timeseries(cu, cycle_sig):
    alias = charter_alias(cycle_sig)
    canon = alias.get(cu, cu)
    family = {c for c in set(alias) | set(alias.values()) if alias.get(c, c) == canon}
    family |= {cu, canon}
    rows = []
    for c in cycle_sig:
        rr = metrics_table(c)
        rr = rr[rr.cu.isin(family)]
        if not rr.empty:
            d = rr.iloc[0].to_dict()
            d["cycle"] = c
            rows.append(d)
    return pd.DataFrame(rows).set_index("cycle") if rows else pd.DataFrame()


# ---- Financial statements: line items -> verified NCUA account codes ----
BALANCE_SHEET = [
    ("Assets", "header", None),
    ("Cash & equivalents", "sum", ["ACCT_730A", "ACCT_730B"]),
    ("Loans & leases (gross)", "code", "ACCT_025B"),
    ("Investments & all other assets", "res_assets", None),
    ("Land & building", "code", "ACCT_007"),
    ("Other fixed assets", "code", "ACCT_008"),
    ("NCUA Share Insurance deposit", "code", "ACCT_794"),
    ("Accrued interest & other assets", "sum", ["ACCT_009A", "ACCT_009B", "ACCT_009C"]),
    ("Total assets", "code", "ACCT_010", True),
    ("Liabilities & equity", "header", None),
    ("Total shares & deposits", "code", "ACCT_018"),
    ("Borrowings", "code", "ACCT_860C"),
    ("Accounts payable & other liabilities", "code", "ACCT_825"),
    ("Other liabilities", "res_liab", None),
    ("Net worth (equity)", "code", "ACCT_997"),
    ("Total liabilities, shares & equity", "code", "ACCT_014", True),
]
INCOME_STATEMENT = [
    ("Interest on loans", "code", "ACCT_110"),
    ("Income from investments", "code", "ACCT_120"),
    ("Total interest income", "code", "ACCT_115", True),
    ("Dividends on shares", "code", "ACCT_380"),
    ("Interest on borrowed money", "code", "ACCT_340"),
    ("Total interest expense", "code", "ACCT_350", True),
    ("Net interest income", "nii", None, True),
    ("Provision for credit losses (implied)", "prov", None),
    ("Fee income", "code", "ACCT_131"),
    ("Total non-interest income", "code", "ACCT_117", True),
    ("Employee compensation & benefits", "code", "ACCT_210"),
    ("Office occupancy", "code", "ACCT_250"),
    ("Office operations", "code", "ACCT_260"),
    ("Loan servicing", "code", "ACCT_280"),
    ("Professional & outside services", "code", "ACCT_290"),
    ("Educational & promotional", "code", "ACCT_270"),
    ("Travel & conference", "code", "ACCT_230"),
    ("Miscellaneous operating", "code", "ACCT_360"),
    ("All other operating expense", "other_opex", None),
    ("Total non-interest expense", "code", "ACCT_671", True),
    ("Non-operating income (expense)", "code", "ACCT_440"),
    ("Net income", "code", "ACCT_661A", True),
]
_OPEX_PARTS = ["ACCT_210", "ACCT_250", "ACCT_260", "ACCT_280", "ACCT_290",
               "ACCT_270", "ACCT_230", "ACCT_360"]
_ASSET_PARTS = ["ACCT_730A", "ACCT_730B", "ACCT_025B", "ACCT_007", "ACCT_008",
                "ACCT_794", "ACCT_009A", "ACCT_009B", "ACCT_009C"]


@st.cache_data(show_spinner=False)
def cu_statement_raw(cu, cycle_sig):
    """All FS220/FS220A account values per cycle for a CU and its charter family."""
    alias = charter_alias(cycle_sig)
    canon = alias.get(cu, cu)
    family = {c for c in set(alias) | set(alias.values()) if alias.get(c, c) == canon}
    family |= {cu, canon}
    inlist = ",".join("'%s'" % c for c in family)
    out = {}
    for tbl in ("FS220", "FS220A"):
        try:
            df = con.execute(
                f"SELECT * FROM read_parquet('{glob_for(tbl)}', hive_partitioning=true, "
                f"union_by_name=true) WHERE CAST(CU_NUMBER AS VARCHAR) IN ({inlist})").df()
        except Exception:
            continue
        acct = [c for c in df.columns if c.upper().startswith("ACCT_")]
        for _, r in df.iterrows():
            d = out.setdefault(str(r["cycle"]), {})
            for c in acct:
                try:
                    d[c.upper()] = float(r[c])
                except (TypeError, ValueError):
                    pass
    return out


def _stmt_getter(raw, cyc, flow, mode):
    """Return a function code->value; income (flow) items de-cumulate in Quarters mode."""
    def stock(code):
        return raw.get(cyc, {}).get(code)

    def flow_(code):
        v = raw.get(cyc, {}).get(code)
        if v is None:
            return None
        if mode == "Years":
            return v
        yr, mm = cyc.split("-")
        if mm == "03":
            return v
        pq = prior_cycle(cyc)
        if pq and pq.split("-")[0] == yr:
            return v - (raw.get(pq, {}).get(code) or 0)
        return v
    return flow_ if flow else stock


def _line_value(kind, arg, g):
    if kind == "code":
        return g(arg)
    if kind == "sum":
        vals = [g(c) for c in arg if g(c) is not None]
        return sum(vals) if vals else None
    if kind == "nii":
        a, b = g("ACCT_115"), g("ACCT_350")
        return None if a is None and b is None else (a or 0) - (b or 0)
    if kind == "prov":
        ni = g("ACCT_661A")
        if ni is None:
            return None
        nii = (g("ACCT_115") or 0) - (g("ACCT_350") or 0)
        return (nii + (g("ACCT_117") or 0) - (g("ACCT_671") or 0) + (g("ACCT_440") or 0)) - ni
    if kind == "other_opex":
        tot = g("ACCT_671")
        return None if tot is None else tot - sum((g(c) or 0) for c in _OPEX_PARTS)
    if kind == "res_assets":
        ta = g("ACCT_010")
        return None if ta is None else ta - sum((g(c) or 0) for c in _ASSET_PARTS)
    if kind == "res_liab":
        tot = g("ACCT_014")
        return None if tot is None else tot - sum(
            (g(c) or 0) for c in ["ACCT_018", "ACCT_860C", "ACCT_825", "ACCT_997"])
    return None


def _period_label(cyc, mode):
    y, m = cyc.split("-")
    if mode == "Years":
        return y
    return f"{y} Q{ {'03': 1, '06': 2, '09': 3, '12': 4}[m] }"


def build_statement(cu, schema, flow, mode, anchor, cycle_sig):
    raw = cu_statement_raw(cu, cycle_sig)
    cs = sorted(cycle_sig)
    if mode == "Years":
        periods = [c for c in cs if c.endswith("-12") and c <= anchor][-5:]
    else:
        periods = [c for c in cs if c <= anchor][-6:]
    periods = periods[::-1]                       # newest first
    if not periods:
        return pd.DataFrame()
    labels = [_period_label(p, mode) for p in periods]
    out = []
    for row in schema:
        label, kind, arg = row[0], row[1], row[2]
        if kind == "header":
            out.append({"": label, **{lb: "" for lb in labels}})
            continue
        rec = {"": label}
        for p, lb in zip(periods, labels):
            g = _stmt_getter(raw, p, flow, mode)
            v = _line_value(kind, arg, g)
            rec[lb] = money(v) if v is not None else "—"
        out.append(rec)
    return pd.DataFrame(out)


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


@st.cache_data(show_spinner=False)
def disappeared(cycle, vs, cycle_sig):
    """CUs present in the comparison quarter but gone by `cycle` — i.e. merged
    or liquidated in between. `vs` = 'prior quarter' or 'year ago'."""
    cs = sorted(cycle_sig)
    if cycle not in cs:
        return pd.DataFrame()
    base = yoy_cycle(cycle) if vs == "year ago" else prior_cycle(cycle)
    if base is None:
        return pd.DataFrame()
    current = set(metrics_table(cycle).cu)
    before = metrics_table(base)
    gone = before[~before.cu.isin(current)].copy()
    converted = set(charter_alias(cycle_sig))      # old charters that just renumbered
    gone = gone[~gone.cu.isin(converted)]
    gone["last_cycle"] = base
    return gone[["cu", "cu_name", "state", "assets", "members", "last_cycle"]] \
        .sort_values("assets", ascending=False)


@st.cache_data(show_spinner="Loading mergers…")
def merger_table(cycle_sig):
    """Authoritative mergers from the NCUA Insurance Report of Activity."""
    if not (DATA_DIR / "MERGERS").exists():
        return pd.DataFrame()
    df = con.execute(
        f"SELECT * FROM read_parquet('{glob_for('MERGERS')}', union_by_name=true)").df()
    df = df[df.merging_charter != df.continuing_charter]            # drop self-merge noise
    for c in ("continuing_assets", "merging_assets"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "merging_reason" in df.columns:
        df["reason"] = df.merging_reason.fillna("(unstated)").astype(str).str.title()

    def _state(loc):
        s = str(loc)
        return s.rsplit(",", 1)[1].strip().upper()[:2] if "," in s else ""
    df["merging_state"] = df.merging_location.map(_state) if "merging_location" in df.columns else ""
    df["continuing_state"] = (df.continuing_location.map(_state)
                              if "continuing_location" in df.columns else "")
    return df


@st.cache_data(show_spinner=False)
def acquirers_in_window(cycle, basis, cycle_sig):
    """Charters that absorbed a CU during the growth window -> count absorbed.
    YoY -> the trailing four report quarters; QoQ -> the selected quarter."""
    mg = merger_table(cycle_sig)
    if mg.empty:
        return {}
    base = yoy_cycle(cycle) if basis == "YoY" else prior_cycle(cycle)
    rep = sorted(mg.cycle.unique())
    win = [c for c in rep if c <= cycle and (base is None or c > base)]
    return mg[mg.cycle.isin(win)].continuing_charter.value_counts().to_dict()


@st.cache_data(show_spinner=False)
def acquirers_trailing(cycle, quarters, cycle_sig):
    """Charters that absorbed a CU in the last `quarters` report quarters up to cycle."""
    mg = merger_table(cycle_sig)
    if mg.empty:
        return {}
    rep = [c for c in sorted(mg.cycle.unique()) if c <= cycle][-quarters:]
    return mg[mg.cycle.isin(rep)].continuing_charter.value_counts().to_dict()


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
        z = row.get(f"z_{key}")
        rows.append((META[key][0], fmt(key, row[key]),
                     f"{z:+.2f} SD" if pd.notna(z) else "—", f"{w * 100:.0f}%"))
    return pd.DataFrame(rows, columns=["Metric", "Value", "Peer z-score", "Weight"])


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
page = st.sidebar.radio("View", ["Profile", "Compare", "Rankings", "Movers", "Mergers", "Industry"])
st.sidebar.divider()
cycle = st.sidebar.selectbox("Quarter", all_cycles)
growth_label = st.sidebar.selectbox(
    "Growth basis", ["Year-over-year", "Quarter-over-quarter (annualized)"])
basis = "YoY" if growth_label.startswith("Year") else "QoQ"
mt = enriched_table(cycle, basis, sig)

# label helper shared across pages
ALL_LABELS = {r.cu: f"{r.cu_name} (#{r.cu}, {r.state})" for r in mt.itertuples()}

# ============================================================ PROFILE
if page == "Profile":
    query = st.text_input("Search a credit union by name", placeholder="e.g. BluCurrent")
    if not query:
        st.info("Type part of a credit union name to begin.")
    else:
        hits = mt[name_matches(mt.cu_name, query)].head(300)
        st.caption(f"{len(hits)} match(es) in {cycle}")
        if not hits.empty:
            labels = {r.cu: ALL_LABELS[r.cu] for r in hits.itertuples()}
            cu = st.selectbox("Select a credit union", list(labels),
                              format_func=lambda n: labels[n])
            row = mt[mt.cu == cu].iloc[0]
            st.subheader(labels[cu])
            st.caption(f"Asset peer group: {row.band}")
            _cv = conversions_table(sig)
            if not _cv.empty and "new_charter" in _cv.columns:
                _pred = _cv[_cv.new_charter == cu]
                if not _pred.empty:
                    _p = _pred.sort_values("cycle").iloc[-1]
                    st.caption(f"Formerly charter #{_p.old_charter} — "
                               f"{str(_p.conv_type).title()} conversion ({_p.cycle}). "
                               "History below is linked across the change.")

            # composite rating
            rc = st.columns([1, 3])
            rc[0].metric("Composite Score",
                         f"{row.score:.0f}/100" if pd.notna(row.score) else "—")
            with rc[1]:
                st.markdown(
                    f"<div style='font-size:2.2rem;line-height:1'>{stars_str(row.stars)}</div>",
                    unsafe_allow_html=True)
                st.caption(f"Growth/momentum score vs the {row.band} asset peer group — "
                           "a z-score blend weighting growth and earnings most heavily. "
                           "50 = peer average; higher means stronger momentum.")
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
                st.subheader("Trends")
                ts = cu_timeseries(cu, sig)
                trend_opts = [k for k, _, _, _ in METRICS if not k.endswith("_growth")]
                tc1, tc2 = st.columns([1, 3])
                tspan = tc1.radio("Period", ["Quarters", "Years"], horizontal=True)
                chosen = tc2.multiselect(
                    "Metrics to chart", trend_opts,
                    default=["assets", "roa", "efficiency", "delinquency"],
                    format_func=lambda k: META[k][0])
                if tspan == "Years" and not ts.empty:
                    ts = ts[[str(i).endswith("-12") for i in ts.index]]
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

            st.subheader("Financial statements")
            sc1, sc2 = st.columns(2)
            stmt = sc1.radio("Statement", ["Balance sheet", "Income statement"], horizontal=True)
            pmode = sc2.radio("Periods", ["Quarters", "Years"], horizontal=True)
            schema = BALANCE_SHEET if stmt == "Balance sheet" else INCOME_STATEMENT
            sdf = build_statement(cu, schema, stmt == "Income statement", pmode, cycle, sig)
            if sdf.empty:
                st.info("No statement data available for this credit union and period.")
            else:
                st.dataframe(sdf, use_container_width=True, hide_index=True,
                             height=38 * len(sdf) + 38)
                note = ("Built from NCUA call report accounts and tied to the reported totals. "
                        "Lines marked “implied” (investments, provision for credit losses, other) "
                        "are derived so the statement foots exactly.")
                if stmt == "Income statement":
                    note += (" Income figures are year-to-date in the call report; the Quarters "
                             "view de-cumulates them into standalone quarters.")
                st.caption(note)

            mg = merger_table(sig)
            if not mg.empty:
                mine = mg[mg.continuing_charter == cu].sort_values("cycle", ascending=False)
                if not mine.empty:
                    st.subheader("Mergers absorbed")
                    st.caption(f"This credit union has absorbed {len(mine)} other "
                               f"{'institution' if len(mine) == 1 else 'institutions'} since 2018, "
                               "per the NCUA Insurance Report of Activity.")
                    st.dataframe(pd.DataFrame({
                        "Quarter": mine.cycle.values,
                        "Absorbed": mine.merging_name.values,
                        "Assets at merger": [money(x) for x in mine.merging_assets.values],
                        "Reason": mine.reason.values}),
                        use_container_width=True, hide_index=True)

            with st.expander("Raw call report tables (advanced)"):
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
elif page == "Compare":
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
elif page == "Rankings":
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
    acq = acquirers_trailing(cycle, 4, sig)
    excl = False
    if acq:
        excl = st.checkbox("Exclude credit unions that absorbed another in the last 4 quarters "
                           "(merger-driven results)", value=False)
    if rank_key in GROWTH_KEYS and not excl:
        st.caption("Heads-up: extreme growth usually reflects a merger/acquisition — "
                   "tick the box above to exclude those.")
    view = mt.copy()
    if sel_states:
        view = view[view.state.isin(sel_states)]
    if sel_bands:
        view = view[view.band.isin(sel_bands)]
    if excl:
        view = view[~view.cu.isin(acq)]
    view = view.dropna(subset=[rank_key]).sort_values(
        rank_key, ascending=order.startswith("Bottom")).head(int(top_n))
    if rank_key in ("score", "stars"):
        show_keys = ["score", "stars"] + [k for k, _ in SCORE_WEIGHTS]
    else:
        show_keys = ["score", "stars", "assets", "net_worth", "roa", "efficiency",
                     "nw_ratio", "delinquency"]
        if rank_key not in show_keys:
            show_keys.insert(2, rank_key)
    disp = pd.DataFrame({"Credit Union": view.cu_name.values, "State": view.state.values})
    for k in show_keys:
        disp[META[k][0]] = [fmt(k, x) for x in view[k].values]
    if acq and not excl:
        disp["Merger"] = [f"\u2713 \u00d7{acq[c]}" if c in acq else "" for c in view.cu.values]
    disp.insert(0, "Rank", range(1, len(disp) + 1))
    st.caption(f"{len(view):,} credit unions shown (of {len(mt):,} total)")
    st.dataframe(disp, use_container_width=True, hide_index=True, height=560)

# ============================================================ MOVERS
elif page == "Movers":
    st.subheader("Biggest movers")
    st.caption(f"Growth basis: {growth_label.lower()}. A ✓ in the Merger column marks a credit "
               "union that absorbed another in the last 4 quarters — toggle the box to exclude them.")
    m1, m2 = st.columns([2, 1])
    gkey = m1.selectbox("Growth metric", GROWTH_KEYS, format_func=lambda k: META[k][0])
    min_assets = m2.selectbox("Minimum asset size",
                              ["$10M", "$50M", "$100M", "$500M", "$1B"], index=1)
    floor = {"$10M": 10e6, "$50M": 50e6, "$100M": 100e6, "$500M": 500e6, "$1B": 1e9}[min_assets]
    acq = acquirers_trailing(cycle, 4, sig)
    hide = False
    if acq:
        hide = st.checkbox("Exclude merger-driven growth (credit unions that absorbed "
                           "another in the last 4 quarters)", value=False)
    pool = mt[(mt.assets >= floor)].dropna(subset=[gkey]).copy()
    pool["_merger"] = pool.cu.map(lambda c: acq.get(c, 0))
    if hide:
        pool = pool[pool._merger == 0]
    if pool.empty:
        st.info("No credit unions with growth data for this basis yet "
                "(needs a prior-period quarter ingested).")
    else:
        cols = ["cu_name", "state", "assets", gkey, "_merger"]
        gain = pool.nlargest(15, gkey)[cols]
        lose = pool.nsmallest(15, gkey)[cols]

        def fmt_movers(df):
            d = pd.DataFrame({
                "Credit Union": df.cu_name.values, "State": df.state.values,
                "Assets": [money(x) for x in df.assets.values],
                META[gkey][0]: [pct(x) for x in df[gkey].values]})
            if acq and not hide:
                d["Merger"] = [f"\u2713 \u00d7{n}" if n else "" for n in df._merger.values]
            return d

        a, b = st.columns(2)
        with a:
            st.markdown("**Top gainers**")
            st.dataframe(fmt_movers(gain), use_container_width=True, hide_index=True)
        with b:
            st.markdown("**Biggest decliners**")
            st.dataframe(fmt_movers(lose), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Recent exits — merged or liquidated")
    vs = st.radio("Compared to", ["prior quarter", "year ago"], horizontal=True)
    gone = disappeared(cycle, vs, sig)
    if gone.empty:
        st.info("No comparison quarter available, or no exits detected.")
    else:
        mg = merger_table(sig)
        cap = (f"{len(gone):,} credit unions were in the {gone.last_cycle.iloc[0]} data "
               f"but gone by {cycle} — i.e. they merged or were liquidated in between.")
        ex = pd.DataFrame({
            "Credit Union": gone.cu_name.values, "State": gone.state.values,
            "Last assets": [money(x) for x in gone.assets.values],
            "Last seen": gone.last_cycle.values})
        if not mg.empty:
            info = (mg[["merging_charter", "continuing_name", "reason"]]
                    .drop_duplicates("merging_charter"))
            j = gone.merge(info, left_on="cu", right_on="merging_charter", how="left")
            ex["Absorbed by"] = j.continuing_name.fillna("— (liquidated or not yet reported)").values
            ex["Reason"] = j.reason.fillna("—").values
            cap += " Acquirer and reason come from the NCUA Insurance Report of Activity."
        else:
            cap += (" Charter disappearance alone can't name the acquirer — ingest the merger "
                    "report to add that.")
        st.caption(cap)
        st.dataframe(ex, use_container_width=True, hide_index=True, height=460)

# ============================================================ INDUSTRY
elif page == "Mergers":
    st.subheader("Mergers")
    mg = merger_table(sig)
    if mg.empty:
        st.info("No merger data yet. Run the **Ingest mergers** workflow, then reboot the app "
                "(⋮ → Reboot) to pick it up.")
    else:
        st.caption(f"Authoritative NCUA Insurance Report of Activity — {mg.cycle.min()} to "
                   f"{mg.cycle.max()}, {len(mg):,} mergers. The *merging* credit union is the one "
                   "that disappeared; the *continuing* credit union absorbed it.")
        cs = sorted(sig)
        f1, f2 = st.columns([1, 1])
        period = f1.selectbox("Period", ["Selected quarter", "Trailing 4 quarters", "All time"],
                              index=2)
        states = sorted(s for s in set(mg.merging_state) | set(mg.continuing_state)
                        if s and len(s) == 2)
        state = f2.selectbox("State", ["All states"] + states)
        if period == "Selected quarter":
            sel = mg[mg.cycle == cycle]
        elif period == "Trailing 4 quarters":
            i = cs.index(cycle) if cycle in cs else len(cs) - 1
            sel = mg[mg.cycle.isin(set(cs[max(0, i - 3):i + 1]))]
        else:
            sel = mg
        if state != "All states":
            sel = sel[(sel.merging_state == state) | (sel.continuing_state == state)]

        scope = f"{period.lower()}" + ("" if state == "All states" else f", {state}")
        st.markdown(f"**{len(sel):,} mergers — {scope}**")
        tbl = pd.DataFrame({
            "Quarter": sel.cycle.values,
            "Merged (disappeared)": sel.merging_name.values,
            "Assets": [money(x) for x in sel.merging_assets.values],
            "Absorbed by": sel.continuing_name.values,
            "Reason": sel.reason.values,
        }).sort_values(["Quarter", "Merged (disappeared)"], ascending=[False, True])
        st.dataframe(tbl, use_container_width=True, hide_index=True, height=460)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Most active acquirers**")
            agg = sel.groupby("continuing_name").agg(
                absorbed=("merging_charter", "count"),
                assets=("merging_assets", "sum")).sort_values("absorbed", ascending=False).head(15)
            st.dataframe(pd.DataFrame({
                "Acquirer": agg.index,
                "CUs absorbed": agg.absorbed.values,
                "Assets absorbed": [money(x) for x in agg.assets.values]}),
                use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**Why they merged**")
            rc = sel.reason.value_counts()
            st.dataframe(pd.DataFrame({"Reason": rc.index, "Count": rc.values}),
                         use_container_width=True, hide_index=True)

# ============================================================ INDUSTRY
elif page == "Industry":
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
