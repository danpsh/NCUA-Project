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
META["score"] = ("Peer Score", "score", "high")
META["stars"] = ("Star Rating", "stars", "high")
GROWTH_KEYS = ["assets_growth", "members_growth", "loans_growth", "shares_growth"]

# Composite score — two selectable "lenses" (weight sets), each summing to 1.0.
# Momentum = growth-led (who's expanding fastest); Performance = earnings, capital,
# and asset-quality led (who's financially strongest), with growth a light factor.
SCORE_LENSES = {
    "Momentum (growth-led)": [
        ("assets_growth", 0.20), ("loans_growth", 0.15),
        ("members_growth", 0.10), ("shares_growth", 0.10),    # growth = 55%
        ("roa", 0.20), ("nim", 0.10),                          # earnings = 30%
        ("efficiency", 0.10),                                  # operating leverage = 10%
        ("delinquency", 0.05),                                 # asset-quality guardrail = 5%
    ],
    "Performance (earnings & health)": [
        ("roa", 0.25), ("nim", 0.10), ("efficiency", 0.15),    # profitability/ops = 50%
        ("nw_ratio", 0.15),                                    # capital = 15%
        ("delinquency", 0.10), ("nco", 0.10),                  # asset quality = 20%
        ("assets_growth", 0.10), ("loans_growth", 0.05),       # growth = 15%
    ],
}

SCORE_MIN_COVERAGE = 0.60   # composite assigned only if ≥60% of the lens weight is present
RANK_MIN_ASSETS = 25e6      # Rankings screen floor — tiny books yield unstable ratios

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


def money_compact(x):
    """Abbreviated dollars for tight/mobile KPI cards: $847K / $387.0M / $1.24B.
    Tables keep full precision via money()."""
    if not (isinstance(x, numbers.Real) and not pd.isna(x)):
        return "—"
    a, sign = abs(x), ("-" if x < 0 else "")
    if a >= 1e9:
        return f"{sign}${a / 1e9:.2f}B"
    if a >= 1e6:
        return f"{sign}${a / 1e6:.1f}M"
    if a >= 1e3:
        return f"{sign}${a / 1e3:.0f}K"
    return f"{sign}${a:.0f}"


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


def stars_from_pct(p):
    if pd.isna(p):
        return np.nan
    if p >= 0.80:
        return 5
    if p >= 0.60:
        return 4
    if p >= 0.40:
        return 3
    if p >= 0.20:
        return 2
    return 1


def stars_from_score(s):
    """Assign stars directly from the 0–100 peer score."""
    if pd.isna(s):
        return np.nan
    if s >= 80:
        return 5
    if s >= 60:
        return 4
    if s >= 40:
        return 3
    if s >= 20:
        return 2
    return 1


def fmt(key, x):
    f = META[key][1]
    if f == "score":
        return f"{x:.0f}" if isinstance(x, (int, float)) and not pd.isna(x) else "—"
    if f == "stars":
        return stars_str(x)
    return money(x) if f == "money" else intfmt(x) if f == "int" else pct(x)


def _delta(key, cur, prev):
    """(delta_text, delta_color) for st.metric, or (None, 'normal') when N/A."""
    if prev is None or pd.isna(cur) or pd.isna(prev):
        return None, "normal"
    diff = cur - prev
    f = META[key][1]
    if f == "money":
        a = abs(diff)
        mag = (f"{a/1e9:.2f}B" if a >= 1e9 else f"{a/1e6:.1f}M" if a >= 1e6
               else f"{a/1e3:.0f}K" if a >= 1e3 else f"{a:.0f}")
        text = ("-$" if diff < 0 else "+$") + mag
    elif f == "int":
        text = f"{diff:+,.0f}"
    elif f == "pct":
        text = f"{diff:+.2f} pp"
    else:
        return None, "normal"
    return text, ("inverse" if META[key][2] == "low" else "normal")


def metric_card(col, key, row, prev_row):
    """Render an st.metric with a prior-quarter delta (direction-aware coloring).
    Money values use compact notation so large figures fit narrow/mobile cards."""
    prev = prev_row[key] if (prev_row is not None and key in prev_row) else None
    text, color = _delta(key, row[key], prev)
    val = money_compact(row[key]) if META[key][1] == "money" else fmt(key, row[key])
    col.metric(META[key][0], val, delta=text, delta_color=color)


def color_scale(series, direction):
    """Per-cell CSS backgrounds for a Styler: green = good, red = bad, scaled by the
    value's within-column percentile. direction 'high' -> higher is better, 'low' ->
    lower is better, None -> no color."""
    s = pd.to_numeric(pd.Series(list(series)), errors="coerce")
    if direction is None or s.notna().sum() < 3:
        return ["" for _ in range(len(s))]
    ranks = s.rank(pct=True)
    css = []
    for v, p in zip(s, ranks):
        if pd.isna(v) or pd.isna(p):
            css.append("")
            continue
        good = p if direction == "high" else 1 - p        # 1 = best in column
        if good >= 0.5:
            css.append(f"background-color: rgba(22,163,74,{(good - 0.5) * 0.7:.2f})")
        else:
            css.append(f"background-color: rgba(220,38,38,{(0.5 - good) * 0.7:.2f})")
    return css


def mix_frame(vals, parts, total_code, residual_label):
    """Build a composition DataFrame (Category, Amount $M, Share %) that foots to the
    reported total, with a residual line for anything not separately reported."""
    total = vals.get(total_code.upper())
    if not total or total <= 0:
        return pd.DataFrame()
    rows, used = [], 0.0
    for lbl, codes in parts:
        v = sum((vals.get(c.upper(), 0.0) or 0.0) for c in codes)
        if v > total * 0.0005:        # ≥0.05% of total; tiny lines fold into residual
            rows.append((lbl, v))
            used += v
    resid = total - used
    if resid > total * 0.001:
        rows.append((residual_label, resid))
    rows.sort(key=lambda r: -r[1])
    return pd.DataFrame({
        "Category": [r[0] for r in rows],
        "Amount ($M)": [r[1] / 1e6 for r in rows],
        "Share": [r[1] / total * 100 for r in rows]})


def mix_dataframe(df):
    st.dataframe(df, use_container_width=True, hide_index=True, column_config={
        "Amount ($M)": st.column_config.NumberColumn("Amount ($M)", format="$%.1f"),
        "Share": st.column_config.ProgressColumn(
            "Share of total", min_value=0, max_value=100, format="%.1f%%")})


PEER_BAR_CSS = """<style>
.pb-wrap{font-size:0.9rem;margin-top:.25rem}
.pb-row{display:grid;grid-template-columns:165px 90px 90px 1fr;align-items:center;
        gap:12px;padding:5px 0;border-top:1px solid #eef0f2}
.pb-head{font-size:.75rem;color:#9097a1;text-transform:uppercase;letter-spacing:.04em;
         border-top:none}
.pb-lbl{color:#374151}
.pb-num{text-align:right;font-variant-numeric:tabular-nums}
.pb-val{font-weight:600}
.pb-track{position:relative;height:14px;background:#f1f3f5;border-radius:7px}
.pb-iqr{position:absolute;top:3px;height:8px;background:#ccd6e4;border-radius:4px}
.pb-med{position:absolute;top:-1px;width:2px;height:16px;background:#6b7280}
.pb-dot{position:absolute;top:1px;width:12px;height:12px;border-radius:50%;
        transform:translateX(-6px);border:2px solid #fff;box-shadow:0 0 2px rgba(0,0,0,.35)}
</style>"""


def peer_bars_html(items):
    """Render a small-multiple of peer-distribution bars. Each item:
    {label, value, median, stats|None}. stats: {lo,hi,p25,p75,median,value,better}.
    Track spans the peer 5th–95th percentile (trimmed so a few extreme outliers —
    common in growth metrics — don't compress the whole distribution to one edge);
    shaded band = middle 50%, tick = median, dot = this CU (green if better than the
    peer median, red if worse). A dot pinned to an edge means this CU is beyond the
    trimmed range."""
    head = ("<div class='pb-row pb-head'><div>Metric</div>"
            "<div class='pb-num'>This CU</div><div class='pb-num'>Peer median</div>"
            "<div>Position in peer range, 5th–95th pct "
            "(●=this CU, ▏=median, band=middle 50%)</div></div>")
    rows = [head]
    for it in items:
        lbl, vstr, mstr, s = it["label"], it["value"], it["median"], it["stats"]
        if s is None:
            bar = "<div class='pb-track'></div>"
        else:
            rng = (s["hi"] - s["lo"]) or 1
            def pos(x):
                return max(0.0, min(100.0, (x - s["lo"]) / rng * 100))
            q1, q3, med = pos(s["p25"]), pos(s["p75"]), pos(s["median"])
            val = max(1.5, min(98.5, pos(s["value"])))        # keep dot fully on-track
            color = "#16a34a" if s["better"] else "#dc2626"
            bar = (f"<div class='pb-track'>"
                   f"<div class='pb-iqr' style='left:{q1:.1f}%;width:{max(0.6, q3 - q1):.1f}%'></div>"
                   f"<div class='pb-med' style='left:{med:.1f}%'></div>"
                   f"<div class='pb-dot' style='left:{val:.1f}%;background:{color}'></div></div>")
        rows.append(f"<div class='pb-row'><div class='pb-lbl'>{lbl}</div>"
                    f"<div class='pb-num pb-val'>{vstr}</div>"
                    f"<div class='pb-num'>{mstr}</div>{bar}</div>")
    return PEER_BAR_CSS + "<div class='pb-wrap'>" + "".join(rows) + "</div>"


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
    pye = f"{int(cycle[:4]) - 1}-12"          # prior year-end, for FPR average balances
    _foicu_cols = {c.upper() for c in con.execute(
        f"SELECT * FROM read_parquet('{glob_for('FOICU')}', hive_partitioning=true, "
        "union_by_name=true) LIMIT 0").df().columns}
    city_sel = "COALESCE(o.CITY, '')" if "CITY" in _foicu_cols else "''"
    d = con.execute(f"""
      SELECT o.CU_NUMBER AS cu, o.CU_NAME AS cu_name, COALESCE(o.STATE, '') AS state,
        {city_sel} AS city,
        TRY_CAST(f.ACCT_010  AS DOUBLE) AS assets,
        TRY_CAST(p.assets_pye AS DOUBLE) AS assets_pye,
        TRY_CAST(p.loans_pye  AS DOUBLE) AS loans_pye,
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
      LEFT JOIN (
        SELECT CU_NUMBER,
               TRY_CAST(ACCT_010  AS DOUBLE) AS assets_pye,
               TRY_CAST(ACCT_025B AS DOUBLE) AS loans_pye
        FROM read_parquet('{glob_for('FS220')}', hive_partitioning=true, union_by_name=true)
        WHERE cycle = '{pye}'
      ) p ON o.CU_NUMBER = p.CU_NUMBER
      WHERE o.cycle = ?
    """, [cycle]).df()
    nii = d.int_income - d.int_expense
    # FPR "Average Assets" = (current period + prior year-end) / 2; fall back to the
    # period-end balance when no prior year-end is available (new/renumbered charters).
    avg_assets = d.assets.where(d.assets_pye.isna() | (d.assets_pye == 0),
                                (d.assets + d.assets_pye) / 2)
    # FPR "Average Loans" = (current period + prior year-end) / 2, same fallback rule.
    avg_loans = d.loans.where(d.loans_pye.isna() | (d.loans_pye == 0),
                              (d.loans + d.loans_pye) / 2)
    # PCA total assets (NW0010) for the Net Worth ratio denominator, matching the FPR
    # page (997 / NW0010). NW0010 can sit on any FS220* schedule and may be absent on
    # older data, so it is pulled defensively per schedule and falls back to total
    # assets (010) when unavailable — leaving the prior behaviour intact in that case.
    nw0010 = {}
    for _tbl in [t for t in available_tables() if t.upper().startswith("FS220")]:
        try:
            for _cu, _v in con.execute(
                    f"SELECT CU_NUMBER, TRY_CAST(ACCT_NW0010 AS DOUBLE) "
                    f"FROM read_parquet('{glob_for(_tbl)}', hive_partitioning=true, "
                    "union_by_name=true) WHERE cycle = ?", [cycle]).fetchall():
                if _v is not None and _cu not in nw0010:
                    nw0010[_cu] = _v
        except Exception:
            continue
    nw_den = d.cu.map(nw0010)
    nw_den = nw_den.where(nw_den.notna() & (nw_den != 0), d.assets)

    def ratio(num, den):
        return np.where(den.notna() & (den != 0), num / den * 100, np.nan)

    d["roa"] = ratio(d.net_income * annualize, avg_assets)   # NCUA ROAA: NI / average assets
    d["roe"] = ratio(d.net_income * annualize, d.net_worth)
    d["nim"] = ratio(nii * annualize, avg_assets)   # NCUA FPR: NII / average assets
    d["efficiency"] = ratio(d.opex, nii + d.non_int_income)
    d["nw_ratio"] = ratio(d.net_worth, nw_den)      # 997 / NW0010 (FPR PCA basis); 010 fallback
    d["lts"] = ratio(d.loans, d.shares)
    d["delinquency"] = ratio(d.delinquent, d.loans)
    d["nco"] = ratio((d.chargeoffs - d.recoveries) * annualize, avg_loans)  # over average loans
    d["band"] = d.assets.apply(band_of)
    return d


# Plausibility bands for derived ratios; values outside these signal suspect data
# rather than a real credit union, and are counted toward a cycle's health status.
_RATIO_BANDS = {"nim": (-1.0, 9.0), "roa": (-6.0, 6.0), "nw_ratio": (0.0, 40.0),
                "delinquency": (0.0, 25.0), "efficiency": (0.0, 250.0)}


@st.cache_data(show_spinner="Validating data…")
def data_health(cycle_sig):
    """Per-cycle data-quality report. Catches the failure modes that otherwise render
    as fact: partial/failed ingests (missing income or assets), coverage drops, broken
    year-to-date accumulation, and clusters of implausible ratios. Returns
    {cycle: {cu_count, total_income, ..., status, issues}} with status ok/warn/error."""
    cs = sorted(cycle_sig)
    raw = {}
    for c in cs:
        m = metrics_table(c)
        n = len(m)
        inc = pd.to_numeric(m.int_income, errors="coerce") if n else pd.Series(dtype=float)
        ast = pd.to_numeric(m.assets, errors="coerce") if n else pd.Series(dtype=float)
        impl = pd.Series(False, index=m.index) if n else pd.Series(dtype=bool)
        for k, (lo, hi) in _RATIO_BANDS.items():
            if k in m:
                v = pd.to_numeric(m[k], errors="coerce")
                impl = impl | (v.notna() & ((v < lo) | (v > hi)))
        raw[c] = {
            "cu_count": n,
            "total_income": float(inc[inc > 0].sum()) if n else 0.0,
            "zero_income_pct": float((inc.isna() | (inc <= 0)).mean()) if n else 1.0,
            "zero_assets_pct": float((ast.isna() | (ast <= 0)).mean()) if n else 1.0,
            "implausible_pct": float(impl.mean()) if n else 0.0,
        }

    report = {}
    for i, c in enumerate(cs):
        r = dict(raw[c])
        issues, status = [], "ok"

        def bump(level):
            nonlocal status
            order = {"ok": 0, "warn": 1, "error": 2}
            if order[level] > order[status]:
                status = level

        if r["zero_income_pct"] > 0.30:
            issues.append(f"{r['zero_income_pct']*100:.0f}% of credit unions report no "
                          "interest income — likely a partial or failed ingest.")
            bump("error")
        if r["zero_assets_pct"] > 0.10:
            issues.append(f"{r['zero_assets_pct']*100:.0f}% report no total assets.")
            bump("error")
        if i > 0:
            prev = raw[cs[i - 1]]["cu_count"]
            if prev and r["cu_count"] < prev * 0.75:
                issues.append(f"Credit-union count fell {(1-r['cu_count']/prev)*100:.0f}% "
                              f"vs {cs[i-1]} ({prev:,} → {r['cu_count']:,}).")
                bump("error")
            elif prev and r["cu_count"] < prev * 0.90:
                issues.append(f"Credit-union count down {(1-r['cu_count']/prev)*100:.0f}% "
                              f"vs {cs[i-1]}.")
                bump("warn")
        yr = c.split("-")[0]
        same_year_prior = [p for p in cs[:i] if p.split("-")[0] == yr]
        if same_year_prior:
            pj = same_year_prior[-1]
            pinc = raw[pj]["total_income"]
            if pinc > 0 and r["total_income"] < pinc * 0.8:
                issues.append(f"Year-to-date interest income fell vs {pj} "
                              f"(${pinc/1e9:.2f}B → ${r['total_income']/1e9:.2f}B); "
                              "year-to-date figures should accumulate within a year.")
                bump("error")
        if r["implausible_pct"] > 0.05:
            issues.append(f"{r['implausible_pct']*100:.1f}% of credit unions have "
                          "out-of-range ratios.")
            bump("warn")

        report[c] = {**r, "status": status, "issues": issues}
    return report


# Vectorized yield & spread for every CU in a cycle (FPR average-balance basis),
# mirroring _ys_ratios but computed across the whole industry for the Rates board.
_RATE_BAL = ["ACCT_010", "ACCT_025B", "ACCT_018", "ACCT_860C", "ACCT_730A",
             "ACCT_007", "ACCT_008", "ACCT_794", "ACCT_009A", "ACCT_009B", "ACCT_009C"]


@st.cache_data(show_spinner="Computing yields & spreads…")
def rate_table(cycle, cycle_sig):
    def read(tbl, cyc):
        try:
            df = con.execute(
                f"SELECT * FROM read_parquet('{glob_for(tbl)}', hive_partitioning=true, "
                "union_by_name=true) WHERE cycle = ?", [cyc]).df()
        except Exception:
            return pd.DataFrame()
        if df.empty or "CU_NUMBER" not in df.columns:
            return pd.DataFrame()
        df.columns = [c.upper() for c in df.columns]
        df["CU_NUMBER"] = df["CU_NUMBER"].astype(str)
        return df.drop_duplicates("CU_NUMBER").set_index("CU_NUMBER")

    f, a = read("FS220", cycle), read("FS220A", cycle)
    if f.empty or a.empty:
        return pd.DataFrame()
    pye = f"{int(cycle[:4]) - 1}-12"
    fp = read("FS220", pye) if pye in cycle_sig else pd.DataFrame()
    # Bring granular investment codes (AS0007/AS0008/AS0013/AS0017) in from any other
    # FS220 schedule so Yield on Average Investments uses the exact NCUA denominator.
    for t in available_tables():
        tu = t.upper()
        if not tu.startswith("FS220") or tu in ("FS220", "FS220A"):
            continue
        xc = read(t, cycle)
        if not xc.empty:
            cols = [c for c in xc.columns if c not in a.columns and c not in f.columns]
            if cols:
                a = a.join(xc[cols], how="left")
        if pye in cycle_sig and not fp.empty:
            xp = read(t, pye)
            if not xp.empty:
                cols = [c for c in xp.columns if c not in fp.columns]
                if cols:
                    fp = fp.join(xp[cols], how="left")
    idx = f.index

    def col(df, code):
        if df is None or df.empty or code not in df.columns:
            return pd.Series(np.nan, index=idx)
        return pd.to_numeric(df[code], errors="coerce").reindex(idx)

    bal = lambda code: col(f, code).fillna(col(a, code))      # balances: FS220, then FS220A
    balp = lambda code: col(fp, code)                          # prior year-end

    def avg(cur_s, pye_s):                                     # FPR (cur + PYE) / 2; robust
        if fp.empty or pye_s is None:
            return cur_s
        use = pye_s.notna() & (pye_s > 0)
        return cur_s.where(~use, (cur_s.fillna(0) + pye_s) / 2)

    def inv_base(getter):
        as_cols = [getter("ACCT_AS0007"), getter("ACCT_AS0008"),
                   getter("ACCT_AS0013"), getter("ACCT_AS0017")]
        have = pd.concat(as_cols, axis=1).notna().any(axis=1)
        exact = sum(c.fillna(0) for c in as_cols) + getter("ACCT_730B").fillna(0)
        resid = (getter("ACCT_010") - getter("ACCT_025B") - getter("ACCT_730A")
                 - getter("ACCT_007") - getter("ACCT_008") - getter("ACCT_794")
                 - getter("ACCT_009A") - getter("ACCT_009B") - getter("ACCT_009C"))
        return exact.where(have, resid)

    ann = 12 / int(cycle[-2:])
    avg_loans = avg(bal("ACCT_025B"), balp("ACCT_025B"))
    avg_inv = avg(inv_base(bal), inv_base(balp) if not fp.empty else None)
    avg_ea = avg_loans + avg_inv
    avg_fund = avg(bal("ACCT_018"), balp("ACCT_018")) + avg(bal("ACCT_860C"), balp("ACCT_860C"))
    avg_assets = avg(bal("ACCT_010"), balp("ACCT_010"))

    i110, i119, i120 = col(a, "ACCT_110"), col(a, "ACCT_119"), col(a, "ACCT_120")
    i115, i350 = col(a, "ACCT_115"), col(a, "ACCT_350")

    def rate(num, den):
        return (num * ann / den * 100).where(den.notna() & (den > 0))

    yl = rate(i110.fillna(0) - i119.fillna(0), avg_loans)
    yi = rate(i120, avg_inv)
    yea = rate(i115, avg_ea)
    cof = rate(i350, avg_fund)
    nim = rate(i115 - i350.fillna(0), avg_assets)

    valid = i115.notna() & (i115 > 0)                          # income guard
    for s in (yl, yi, yea, cof, nim):
        s[~valid] = np.nan

    def clamp(s, lo, hi):                                      # plausibility backstop
        return s.where((s >= lo) & (s <= hi))
    yl, yi, yea = clamp(yl, -2, 35), clamp(yi, -2, 35), clamp(yea, -2, 35)
    cof, nim = clamp(cof, -2, 15), clamp(nim, -6, 15)
    spread = yea - cof

    rdf = pd.DataFrame({"cu": idx, "yl": yl.values, "yi": yi.values, "yea": yea.values,
                        "cof": cof.values, "spread": spread.values, "nim": nim.values})
    base = metrics_table(cycle)[["cu", "cu_name", "state", "city", "band", "assets"]].copy()
    base["cu"] = base.cu.astype(str)
    return base.merge(rdf, on="cu", how="left")


# Labels / sort direction for the Rates leaderboard (higher better, except cost of funds).
RATE_COLS = [("nim", "Net Interest Margin", "high"),
             ("spread", "Net Interest Spread", "high"),
             ("yea", "Yield on Earning Assets", "high"),
             ("yl", "Yield on Loans", "high"),
             ("yi", "Yield on Investments", "high"),
             ("cof", "Cost of Funds", "low")]


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
def enriched_table(cycle, basis, lens, cycle_sig):
    df = metrics_table(cycle).merge(growth_for(cycle, basis, cycle_sig), on="cu", how="left")
    weights = SCORE_LENSES[lens]
    # Recent acquirers (a CU they absorbed left the call reports in the last 4 quarters),
    # plus likely-but-unpublished mergers inferred from the call reports themselves.
    acq = merger_acquirers(cycle, cycle_sig)
    inf = inferred_acquirers(cycle, cycle_sig)
    is_acq = df.cu.isin(acq) | df.cu.isin(inf)
    df["is_acquirer"] = is_acq
    # Composite score: weighted blend of band-relative PERCENTILE RANKS (0–1) per metric,
    # direction-adjusted so higher percentile = better. Percentile is bounded and robust
    # to outliers/skew, so it neither saturates (no clip needed) nor lets one merger/junk
    # row distort the baseline — acquirers stay in the ranking but get no credit for
    # merger-bought growth (their own growth contribution is masked).
    acc = pd.Series(0.0, index=df.index)
    wsum = pd.Series(0.0, index=df.index)
    for key, w in weights:
        pr = df.groupby("band")[key].rank(pct=True)       # 0–1 within band; ties averaged
        if META[key][2] == "low":
            pr = 1 - pr                                   # lower value is better → flip
        df[f"pct_{key}"] = pr
        contrib = pr.mask(is_acq) if key.endswith("_growth") else pr  # no merger-growth credit
        acc = acc + contrib.fillna(0) * w
        wsum = wsum + contrib.notna().astype(float) * w
    df["score_pct"] = (acc / wsum).where(wsum >= SCORE_MIN_COVERAGE)  # need ≥60% weight present
    df["score"] = (100 * df["score_pct"]).round(0)        # 0–100; 50 = band median
    df["stars"] = df["score"].apply(stars_from_score)
    return df


@st.cache_data(show_spinner="Computing score history…")
def score_history(cu, basis, lens, cycle, cycle_sig):
    """Composite score, stars, and all-CU percentile at each year-end (plus the
    selected quarter), following the charter family across conversions."""
    cs = sorted(cycle_sig)
    picks = [c for c in cs if c.endswith("-12")]
    if cycle in cs and cycle not in picks:
        picks.append(cycle)
    picks = sorted(set(picks))
    alias = charter_alias(cycle_sig)
    canon = alias.get(cu, cu)
    family = {c for c in set(alias) | set(alias.values())
              if alias.get(c, c) == canon} | {cu, canon}
    rows = []
    for c in picks:
        et = enriched_table(c, basis, lens, cycle_sig)
        r = et[et.cu.isin(family)]
        if r.empty or pd.isna(r.iloc[0].score):
            continue
        s = float(r.iloc[0].score)
        rows.append({"cycle": c, "score": s, "stars": r.iloc[0].stars})
    return pd.DataFrame(rows).set_index("cycle") if rows else pd.DataFrame()


@st.cache_data(show_spinner=False)
def acct_values(cu, cycle, cycle_sig):
    """All FS220 + FS220A account balances for one credit union / cycle as
    {ACCT_CODE_UPPER: float}. Reads via the cursor (no .df()) so it is resilient to
    pandas/duckdb/runtime version differences on Streamlit Cloud."""
    vals = {}
    for table in [t for t in available_tables() if t.upper().startswith("FS220")]:
        try:
            cur = con.execute(
                f"SELECT * FROM read_parquet('{glob_for(table)}', "
                "hive_partitioning=true, union_by_name=true) "
                "WHERE cycle = ? AND CU_NUMBER = ?", [cycle, cu])
            rows = cur.fetchall()
        except Exception:
            continue
        if not rows:
            continue
        cols = [d[0] for d in cur.description]
        for c, v in zip(cols, rows[0]):
            try:
                x = float(v)
                if x == x:                       # skip NaN
                    vals.setdefault(c.upper(), x)
            except (TypeError, ValueError):
                pass
    return vals


# Loan & deposit composition — NCUA call report Section 1 amount codes (verified to foot
# to total loans / total shares). Loan detail lives in FS220A + FS220L (+ FS220H student).
LOAN_MIX = [("First mortgage (1st lien)", ["ACCT_703A"]),
            ("Other real estate", ["ACCT_386A", "ACCT_386B"]),
            ("Used vehicle", ["ACCT_370"]),
            ("New vehicle", ["ACCT_385"]),
            ("Commercial / business", ["ACCT_718A5", "ACCT_400P"]),
            ("Credit card", ["ACCT_396"]),
            ("Other unsecured", ["ACCT_397", "ACCT_397A"]),
            ("Student, other secured & leases",
             ["ACCT_698A", "ACCT_698C", "ACCT_002"])]
DEPOSIT_MIX = [("Share drafts (checking)", ["ACCT_902"]),
               ("Regular shares", ["ACCT_657"]),
               ("Money market", ["ACCT_911"]),
               ("Share certificates (CDs)", ["ACCT_908C"]),
               ("Non-member deposits", ["ACCT_457"])]
# Operating-expense composition — labels mirror the income statement's opex lines.
OPEX_MIX = [("Employee compensation & benefits", ["ACCT_210"]),
            ("Office occupancy", ["ACCT_250"]),
            ("Office operations", ["ACCT_260"]),
            ("Loan servicing", ["ACCT_280"]),
            ("Professional & outside services", ["ACCT_290"]),
            ("Educational & promotional", ["ACCT_270"]),
            ("Travel & conference", ["ACCT_230"]),
            ("Miscellaneous operating", ["ACCT_360"])]
# source label -> (parts, total account, residual label) for the Composition chart.
MIX_SOURCES = {
    "Loan mix": (LOAN_MIX, "ACCT_025B", "Other"),
    "Deposit mix": (DEPOSIT_MIX, "ACCT_018", "Other (incl. IRA / Keogh)"),
    "Operating expense": (OPEX_MIX, "ACCT_671", "All other operating"),
}


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
    """All FS220* account values per cycle for a CU and its charter family. Reads every
    FS220 schedule so newer FPR codes (AS*, NV*, IS*, NW*, etc.) are available."""
    alias = charter_alias(cycle_sig)
    canon = alias.get(cu, cu)
    family = {c for c in set(alias) | set(alias.values()) if alias.get(c, c) == canon}
    family |= {cu, canon}
    inlist = ",".join("'%s'" % c for c in family)
    out = {}
    for tbl in [t for t in available_tables() if t.upper().startswith("FS220")]:
        try:
            curx = con.execute(
                f"SELECT * FROM read_parquet('{glob_for(tbl)}', hive_partitioning=true, "
                f"union_by_name=true) WHERE CAST(CU_NUMBER AS VARCHAR) IN ({inlist})")
            rows = curx.fetchall()
        except Exception:
            continue
        ix = {d[0].upper(): i for i, d in enumerate(curx.description)}
        cyc_i = ix.get("CYCLE")
        acc = [(c, i) for c, i in ix.items() if c.startswith("ACCT_")]
        if cyc_i is None:
            continue
        for r in rows:
            d = out.setdefault(str(r[cyc_i]), {})
            for name, i in acc:
                v = r[i]
                if v is None:
                    continue
                try:
                    d[name] = float(v)
                except (TypeError, ValueError):
                    if name == "ACCT_NW0001":          # CECL adoption date (text)
                        d[name] = str(v)
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
        v = g("ACCT_IS0017")                             # Total Credit Loss Expense
        if v is not None:
            return v
        direct = [g("ACCT_300"), g("ACCT_IS0011")]
        if any(x is not None for x in direct):
            return sum(x or 0 for x in direct)
        ni = g("ACCT_661A")                              # fall back to the net-income identity
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
    if kind == "neg_sum":                                  # contra line (e.g. allowance)
        vals = [g(c) for c in arg if g(c) is not None]
        return -sum(vals) if vals else None
    if kind == "na":                                       # reported but unavailable here
        return None
    if kind == "total_liab":                               # AP + accrued div + borrowings + shares
        parts = ["ACCT_825", "ACCT_820A", "ACCT_860C", "ACCT_018"]
        vals = [g(c) for c in parts]
        return sum(v or 0 for v in vals) if any(v is not None for v in vals) else None
    if kind == "total_equity":                             # total assets − total liabilities
        ta = g("ACCT_010")
        if ta is None:
            return None
        return ta - (_line_value("total_liab", None, g) or 0)
    if kind == "other_reserves":                           # equity not in undivided earnings
        te = _line_value("total_equity", None, g)
        return None if te is None else te - (g("ACCT_940") or 0)
    if kind == "all_other_assets":                         # residual asset line (foots to 010)
        ta = g("ACCT_010")
        if ta is None:
            return None
        itemized = sum((g(c) or 0) for c in ["ACCT_AS0009", "ACCT_AS0013", "ACCT_AS0017",
                                             "ACCT_003", "ACCT_025B", "ACCT_007",
                                             "ACCT_008", "ACCT_794"])
        allowance = (g("ACCT_AS0048") or 0)
        return ta - itemized + allowance
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


# ---- NCUA-style Financial Summary (mirrors the FPR "Quarterly, Ending …" report) ----
FPR_FS_CSS = ("<style>"
              ".fprmeta{border-collapse:collapse;font-size:.78rem;margin:.1rem 0 .7rem}"
              ".fprmeta th,.fprmeta td{border:1px solid #b9c0c9;padding:3px 9px;text-align:center}"
              ".fprmeta th{background:#f3f5f7;font-weight:700}.fprmeta td.l{text-align:left}"
              ".fprfs{border-collapse:collapse;font-size:.78rem;width:100%}"
              ".fprfs th,.fprfs td{border:1px solid #cbd0d7;padding:3px 8px}"
              ".fprfs thead th{background:#f3f5f7;text-align:center;font-weight:700;"
              "white-space:nowrap}"
              ".fprfs td.lbl{text-align:left;white-space:nowrap}"
              ".fprfs td.num{text-align:right;font-variant-numeric:tabular-nums}"
              ".fprfs td.pct{text-align:right;color:#475569;background:#fafbfc;"
              "font-variant-numeric:tabular-nums}"
              ".fprfs tr.sec td{background:#e9edf1;font-weight:700;letter-spacing:.02em}"
              ".fprfs tr.tot td{font-weight:700;border-top:1.4px solid #98a2b0}"
              ".fprfs.fprkr td.lbl{white-space:normal}"
              ".fprfs td.li{text-align:left;white-space:nowrap}"
              ".fprfs td.rt{text-align:right;font-variant-numeric:tabular-nums}"
              ".fprfs th.pa{background:#dfeee8}"
              ".fprfs td.pa{background:#eef4f1;font-weight:600}"
              ".fprfs sup{font-size:.62rem;color:#64748b}"
              ".fprtitle{font-weight:700;margin:.2rem 0 .35rem;font-size:.92rem}"
              ".fprnote{font-size:.7rem;color:#6b7280;margin-top:.5rem;line-height:1.55}"
              "</style>")

FPR_BALANCE = [
    ("Assets", "header", None),
    ('Cash &amp; Other Deposits<sup>1</sup>', "code", "ACCT_AS0009"),
    ("Total Investments", "sum", ["ACCT_AS0013", "ACCT_AS0017"]),
    ("Loans Held for Sale", "code", "ACCT_003"),
    ("Total Loans", "code", "ACCT_025B"),
    ("(Allowance for Loan &amp; Lease Losses or Allowance for Credit Losses on Loans &amp; "
     "Leases)", "neg_sum", ["ACCT_AS0048"]),
    ("Land And Building", "code", "ACCT_007"),
    ("Other Fixed Assets", "code", "ACCT_008"),
    ("NCUSIF Deposit", "code", "ACCT_794"),
    ("All Other Assets", "all_other_assets", None),
    ("Total Assets", "code", "ACCT_010", True),
    ("Liabilities, Shares & Equity", "header", None),
    ("Accounts Payable, Accrued Interest on Borrowings, &amp; Other Liabilities<sup>2</sup>",
     "code", "ACCT_825"),
    ("Accrued Dividends &amp; Interest Payable on Shares &amp; Deposits", "code", "ACCT_820A"),
    ("Allowance for Credit Losses on Off-Balance Sheet Credit Exposures", "na", None),
    ("Borrowings Notes &amp; Interest Payable", "code", "ACCT_860C"),
    ("Total Shares &amp; Deposits", "code", "ACCT_018"),
    ("Total Liabilities", "total_liab", None, True),
    ("Undivided Earnings", "code", "ACCT_940"),
    ("Other Reserves", "other_reserves", None),
    ("Total Equity", "total_equity", None, True),
    ("Total Liabilities, Shares, &amp; Equity", "code", "ACCT_014", True),
]
FPR_INCOME = [
    ("Income & Expense", "header", None),
    ("Interest Income*", "code", "ACCT_115"),
    ("Interest Expense*", "code", "ACCT_350"),
    ("Net Interest Income*", "nii", None),
    ("Provision for Loan/Lease Losses or Total Credit Loss Expense*", "prov", None),
    ("Non-Interest Income*", "code", "ACCT_117"),
    ("Non-Interest Expense*", "code", "ACCT_671"),
    ("Net Income (Loss)*", "code", "ACCT_661A", True),
]
_FPR_MON = {"03": "Mar", "06": "Jun", "09": "Sep", "12": "Dec"}


def _fpr_collabel(cyc, mode):
    y, m = cyc.split("-")
    return y if mode == "Years" else f"{_FPR_MON[m]}-{y}"


def _fpr_amt(v):
    if v is None:
        return ""
    n = int(round(v))
    return f"({abs(n):,})" if n < 0 else f"{n:,}"


def _fpr_pct(cur, prev, mc, mp, income):
    if cur is None or prev is None:
        return "N/A"
    if income:                                   # annualize YTD run-rates before comparing
        cur, prev = cur * 12.0 / mc, prev * 12.0 / mp
    if prev == 0:
        return "N/A"
    return f"{(cur - prev) / prev * 100:.1f}"


@st.cache_data(show_spinner=False)
def foicu_row(cu, anchor, cycle_sig):
    """Latest FOICU record (name/address/region) for a CU at/just before the anchor cycle."""
    try:
        cur = con.execute(
            f"SELECT * FROM read_parquet('{glob_for('FOICU')}', hive_partitioning=true, "
            "union_by_name=true) WHERE CAST(CU_NUMBER AS VARCHAR) = ? AND cycle <= ? "
            "ORDER BY cycle DESC LIMIT 1", [str(cu), anchor])
        rows = cur.fetchall()
    except Exception:
        return {}
    if not rows:
        return {}
    return {d[0].upper(): v for d, v in zip(cur.description, rows[0])}


def fpr_meta_html(cu, anchor, cycle_sig):
    row = foicu_row(cu, anchor, cycle_sig)

    def f(*keys):
        for k in keys:
            v = row.get(k)
            if v not in (None, "") and str(v).lower() != "nan":
                return str(v).strip()
        return ""

    name = (ALL_LABELS.get(cu, "").split(" (#")[0]) or f("CU_NAME")
    cells = [("Charter", str(cu)), ("Name", name),
             ("Street", f("STREET", "ADDRESS", "STREET_ADDRESS")),
             ("City", f("CITY")), ("State", f("STATE", "PHYSICAL_STATE")),
             ("ZipCode", f("ZIP_CODE", "ZIPCODE", "ZIP", "PHYSICAL_ZIP")),
             ("Region", f("REGION", "NCUA_REGION", "REGION_CODE", "RGN"))]
    head = "".join(f"<th>{h}</th>" for h, _ in cells)
    body = "".join(f'<td class="{"l" if h in ("Name", "Street") else ""}">{v}</td>'
                   for h, v in cells)
    return (f'<table class="fprmeta"><thead><tr>{head}</tr></thead>'
            f'<tbody><tr>{body}</tr></tbody></table>')


def fpr_summary_html(cu, mode, anchor, cycle_sig, n=5):
    raw = cu_statement_raw(cu, cycle_sig)
    cs = sorted(cycle_sig)
    if mode == "Years":
        periods = [c for c in cs if c.endswith("-12") and c <= anchor][-n:]
    else:
        periods = [c for c in cs if c <= anchor][-n:]
    if not periods:
        return None
    cols = [_fpr_collabel(p, mode) for p in periods]
    mons = [12 if mode == "Years" else {"03": 3, "06": 6, "09": 9, "12": 12}[p[-2:]]
            for p in periods]
    span_all = 2 * len(periods)
    h1, h2 = ['<th rowspan="2">Line Item</th>'], []
    for i, c in enumerate(cols):
        if i == 0:
            h1.append(f"<th>{c}</th>"); h2.append("<th>Amount</th>")
        else:
            h1.append(f'<th>{c}</th><th rowspan="2">%Chg</th>'); h2.append("<th>Amount</th>")
    thead = f"<thead><tr>{''.join(h1)}</tr><tr>{''.join(h2)}</tr></thead>"

    def rows_for(schema, income):
        out = []
        for row in schema:
            label, kind, arg = row[0], row[1], row[2]
            if kind == "header":
                out.append(f'<tr class="sec"><td colspan="{span_all}">'
                           f'{label.upper().replace("&", "&amp;")}:</td></tr>')
                continue
            is_tot = len(row) > 3 and row[3]
            vals = [_line_value(kind, arg, _stmt_getter(raw, p, False, mode)) for p in periods]
            cells = [f'<td class="lbl">{label}</td>']
            for i, v in enumerate(vals):
                cells.append(f'<td class="num">{_fpr_amt(v)}</td>')
                if i > 0:
                    cells.append('<td class="pct">'
                                 f'{_fpr_pct(v, vals[i-1], mons[i], mons[i-1], income)}</td>')
            out.append(f'<tr class="{"tot" if is_tot else ""}">{"".join(cells)}</tr>')
        return out

    body = rows_for(FPR_BALANCE, False) + rows_for(FPR_INCOME, True)
    return f'<table class="fprfs">{thead}<tbody>{"".join(body)}</tbody></table>'


# ---- Yield & spread decomposition ----------------------------------------
# Splits the net interest margin into its drivers: what the credit union earns on
# loans vs. investments, and what it pays for funding. Average balances follow the
# NCUA FPR convention (current period + prior year-end) ÷ 2; income is annualized.
# Total investments are not a single FS220/FS220A line, so the investment+cash base
# is derived as total assets net of loans, cash on hand, fixed assets, the NCUSIF
# deposit, and accrued/other assets (mirrors the balance-sheet residual).
_NONINV_ASSET = ["ACCT_025B", "ACCT_730A", "ACCT_007", "ACCT_008", "ACCT_794",
                 "ACCT_009A", "ACCT_009B", "ACCT_009C"]


def _inv_base(v):
    """Average-investments base for Yield on Average Investments. Per the NCUA FPR
    (3/31/2022+): total investments (AS0007 + AS0008 + AS0013 + AS0017) + cash on deposit
    (730B). Falls back to a residual estimate when those codes aren't reported."""
    inv = [v.get(c) for c in ("ACCT_AS0007", "ACCT_AS0008", "ACCT_AS0013", "ACCT_AS0017")]
    if any(isinstance(x, (int, float)) and x == x for x in inv):
        return sum((x or 0) for x in inv) + (v.get("ACCT_730B") or 0)
    ta = v.get("ACCT_010")
    if ta is None:
        return None
    return ta - sum((v.get(c) or 0) for c in _NONINV_ASSET)


def _ys_ratios(cur, pye, factor):
    """Yield & spread figures from current + prior-year-end account dicts.
    pye may be None -> point-in-time balances are used as the 'average'.

    Robust to bad partitions: a missing/zero prior-year-end balance falls back to
    the current balance (rather than averaging with 0, which would halve the
    denominator and double the ratio); missing interest income yields blanks, not
    zeros; and clearly-implausible outputs are suppressed as suspect."""
    def num(d, c):
        v = d.get(c) if d else None
        return v if isinstance(v, (int, float)) and v == v else None      # not NaN

    def avg(c):
        a, b = num(cur, c), (num(pye, c) if pye else None)
        if not pye or b is None or b <= 0:        # no prior / prior missing -> point-in-time
            return a or 0.0
        return ((a or 0.0) + b) / 2.0

    def rate(n, den):
        return n * factor / den * 100 if (n is not None and den and den > 0) else None

    # Interest income must be present and positive, or the whole decomposition is moot.
    ii = num(cur, "ACCT_115")
    if ii is None or ii <= 0:
        return {"yl": None, "yi": None, "yea": None, "cof": None,
                "spread": None, "nim": None, "avg_ea": None, "avg_assets": None}

    avg_loans = avg("ACCT_025B") + avg("ACCT_003")          # incl. loans held for sale
    ib_cur = _inv_base(cur)
    ib_pye = _inv_base(pye) if pye else None
    avg_inv = (((ib_cur or 0) + ib_pye) / 2.0
               if (pye and ib_pye and ib_pye > 0) else (ib_cur or 0))
    avg_ea = avg_loans + avg_inv
    avg_fund = avg("ACCT_018") + avg("ACCT_860C")           # shares + borrowings
    avg_assets = avg("ACCT_010")

    yl = rate((num(cur, "ACCT_110") or 0) - (num(cur, "ACCT_119") or 0), avg_loans)
    yi = rate(num(cur, "ACCT_120"), avg_inv)
    yea = rate(ii, avg_ea)
    cof = rate(num(cur, "ACCT_350"), avg_fund)
    nim = rate(ii - (num(cur, "ACCT_350") or 0), avg_assets)

    # Plausibility backstop — egregious values signal bad data, not a real CU.
    def sane(v, lo, hi):
        return v if (v is not None and lo <= v <= hi) else None
    yl, yi, yea = sane(yl, -2, 35), sane(yi, -2, 35), sane(yea, -2, 35)
    cof, nim = sane(cof, -2, 15), sane(nim, -6, 15)
    spread = (yea - cof) if (yea is not None and cof is not None) else None
    return {"yl": yl, "yi": yi, "yea": yea, "cof": cof, "spread": spread, "nim": nim,
            "avg_ea": avg_ea, "avg_assets": avg_assets}


@st.cache_data(show_spinner=False)
def ys_point(cu, cycle, cycle_sig):
    """Yield & spread ratios for one CU at one cycle -> (ratios_dict, used_avg),
    or None if there's no data. Averages use (cycle + prior year-end) ÷ 2."""
    cur = acct_values(cu, cycle, cycle_sig)
    if not cur:
        return None
    pye_c = f"{int(cycle[:4]) - 1}-12"
    pye = acct_values(cu, pye_c, cycle_sig) if pye_c in cycle_sig else None
    return _ys_ratios(cur, pye, 12 / int(cycle[-2:])), (pye is not None)


# Measures offered in the Chart builder, grouped for the picker.
CHART_METRICS = [
    ("roa", "ROA", "Profitability & efficiency"),
    ("roe", "ROE", "Profitability & efficiency"),
    ("nim", "Net Interest Margin", "Profitability & efficiency"),
    ("efficiency", "Efficiency Ratio", "Profitability & efficiency"),
    ("yl", "Yield on Loans", "Yields & spread"),
    ("yi", "Yield on Investments", "Yields & spread"),
    ("yea", "Yield on Earning Assets", "Yields & spread"),
    ("cof", "Cost of Funds", "Yields & spread"),
    ("spread", "Net Interest Spread", "Yields & spread"),
    ("nw_ratio", "Net Worth Ratio", "Capital & asset quality"),
    ("delinquency", "Delinquency Ratio", "Capital & asset quality"),
    ("nco", "Net Charge-Off Ratio", "Capital & asset quality"),
    ("lts", "Loan-to-Share", "Capital & asset quality"),
    ("assets", "Total Assets", "Size ($)"),
    ("loans", "Loans & Leases", "Size ($)"),
    ("shares", "Shares & Deposits", "Size ($)"),
    ("net_worth", "Net Worth", "Size ($)"),
    ("net_income", "Net Income (YTD)", "Size ($)"),
    ("members", "Members", "Size ($)"),
]
CHART_LABEL = {k: lbl for k, lbl, _ in CHART_METRICS}
_YS_KEYS = ("yl", "yi", "yea", "cof", "spread")
# Measures a band median is meaningful for (rates/ratios) — dollar & count measures excluded.
RATIO_PEER_KEYS = {"roa", "roe", "nim", "efficiency", "nw_ratio", "lts", "delinquency",
                   "nco", "yl", "yi", "yea", "cof", "spread"}


def chart_kind(k):
    """'pct' / 'money' / 'int' — drives y-axis tick formatting on the Chart page."""
    if k in _YS_KEYS:
        return "pct"
    return META[k][1] if k in META else "pct"


@st.cache_data(show_spinner="Building the series…")
def chart_series(cu, cycle_sig):
    """Per-cycle time series for one CU: the standard scorecard metrics plus the
    yield & spread decomposition, indexed by cycle (chronological)."""
    base = cu_timeseries(cu, cycle_sig)
    if base.empty:
        return base
    base = base.sort_index()
    ys = {c: (ys_point(base.loc[c, "cu"], c, cycle_sig) or ({},))[0] for c in base.index}
    for k in _YS_KEYS:
        base[k] = [ys.get(c, {}).get(k) for c in base.index]
    return base


@st.cache_data(show_spinner="Computing peer median…")
def peer_median_line(metric, band, in_range, cycle_sig):
    """Median of `metric` across the asset band at each cycle in `in_range`.
    Returns (dict cycle->median, peer_count). Yields come from rate_table and the
    scorecard metrics from metrics_table — the same sources the per-CU lines use, so
    the benchmark is computed identically to the plotted series."""
    out, ncts = {}, []
    is_yield = metric in ("yl", "yi", "yea", "cof", "spread")
    for cyc in in_range:
        try:
            tbl = rate_table(cyc, cycle_sig) if is_yield else metrics_table(cyc)
        except Exception:
            continue
        if tbl is None or tbl.empty or metric not in tbl.columns or "band" not in tbl.columns:
            continue
        sub = pd.to_numeric(tbl.loc[tbl["band"] == band, metric], errors="coerce").dropna()
        if len(sub):
            out[cyc] = float(sub.median())
            ncts.append(len(sub))
    return out, (max(ncts) if ncts else 0)


@st.cache_data(show_spinner=False)
def cu_band(cu, cycle):
    """Asset band for one CU at (or as of) a cycle, from the metrics cross-section."""
    try:
        mt = metrics_table(cycle)
        row = mt.loc[mt.cu.astype(str) == str(cu), "assets"]
        return band_of(float(row.iloc[0])) if len(row) else None
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def yield_spread(cu, cycle, cycle_sig):
    """Yield & spread for one CU at `cycle`, plus the same period one year earlier
    (for pp deltas). `used_avg` is False when no prior year-end is in range."""
    now = ys_point(cu, cycle, cycle_sig)
    if now is None or now[0]["yl"] is None:
        return None
    cur_figs, used_avg = now
    prior = ys_point(cu, f"{int(cycle[:4]) - 1}-{cycle[-2:]}", cycle_sig)
    return {"cur": cur_figs, "prior": (prior[0] if prior else None),
            "used_avg": used_avg, "factor": 12 / int(cycle[-2:])}


# Rows for the yield & spread trend table (grouped like the financial statements).
_YS_ROWS = [
    ("Asset yields", None, "header"),
    ("Yield on loans", "yl", None),
    ("Yield on investments", "yi", None),
    ("Yield on earning assets", "yea", None),
    ("Funding & margin", None, "header"),
    ("Cost of funds", "cof", None),
    ("Net interest spread", "spread", None),
    ("Net interest margin", "nim", None),
]


def build_yield_spread_table(cu, mode, anchor, cycle_sig):
    """Yield & spread over time: rows = ratios, columns = periods (newest first),
    mirroring build_statement's period selection. Every period shows the annualized
    year-to-date ratio (NCUA FPR basis) — ratios are not de-cumulated."""
    cs = sorted(cycle_sig)
    if mode == "Years":
        periods = [c for c in cs if c.endswith("-12") and c <= anchor][-5:]
    else:
        periods = [c for c in cs if c <= anchor][-6:]
    periods = periods[::-1]
    if not periods:
        return pd.DataFrame()
    labels = [_period_label(p, mode) for p in periods]
    pts = {p: (ys_point(cu, p, cycle_sig) or (None,))[0] for p in periods}
    out = []
    for label, key, kind in _YS_ROWS:
        if kind == "header":
            out.append({"": label, **{lb: "" for lb in labels}})
            continue
        rec = {"": label}
        for p, lb in zip(periods, labels):
            v = pts[p].get(key) if pts[p] else None
            rec[lb] = pct(v) if v is not None else "—"
        out.append(rec)
    return pd.DataFrame(out)


def render_yield_spread(cu, cycle, cycle_sig, mode="Quarters"):
    st.markdown("**Yield & spread**")
    ys = yield_spread(cu, cycle, cycle_sig)
    if ys:
        cur, prior = ys["cur"], ys["prior"]

        def card(col, label, key, good_high=True):
            v = cur.get(key)
            d = None
            if prior and prior.get(key) is not None and v is not None:
                d = f"{v - prior[key]:+.2f} pp"
            col.metric(label, pct(v), delta=d,
                       delta_color=(("normal" if good_high else "inverse") if d else "normal"))

        r1 = st.columns(3)
        card(r1[0], "Yield on loans", "yl")
        card(r1[1], "Yield on investments", "yi")
        card(r1[2], "Yield on earning assets", "yea")
        r2 = st.columns(3)
        card(r2[0], "Cost of funds", "cof", good_high=False)
        card(r2[1], "Net interest spread", "spread")
        card(r2[2], "Net interest margin", "nim")
    else:
        st.caption("Yield & spread isn't available for the selected quarter — interest-income "
                   "data is missing or didn't pass validation (see Data health in the sidebar).")

    tbl = build_yield_spread_table(cu, mode, cycle, cycle_sig)
    if not tbl.empty and len(tbl.columns) > 2:        # only worth a table with ≥2 periods
        st.dataframe(tbl, use_container_width=True, hide_index=True,
                     height=38 * len(tbl) + 38)

    st.caption(
        "Income annualized over average balances — (period + prior year-end) ÷ 2. Yield on "
        "investments uses total investments plus cash on deposit, derived as assets net of "
        "loans, fixed assets, the NCUSIF deposit and cash on hand. Cost of funds is total "
        "interest expense over average shares + borrowings; net interest spread is the "
        "earning-asset yield minus cost of funds; net interest margin is net interest income "
        "over average assets (NCUA FPR method). The cards’ deltas are vs. one year prior. In "
        "the trend table each quarter is the annualized year-to-date ratio — unlike the income "
        "statement, ratios are not de-cumulated. Cells that fail validation show “—”.")


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
def merger_acquirers(cycle, cycle_sig, lookback=4):
    """Continuing (acquiring) charters whose absorbed CU actually DISAPPEARED from
    the call reports within the last `lookback` quarters -> count absorbed.

    Keyed off when assets transferred (the merged charter vanishes), not the merger
    approval date — approvals can precede the call-report impact by several quarters.
    """
    mg = merger_table(cycle_sig)
    if mg.empty:
        return {}
    cs = sorted(cycle_sig)
    if cycle not in cs:
        return {}
    base = yoy_cycle(cycle) if lookback == 4 else None
    if base not in cs:
        i = cs.index(cycle)
        base = cs[max(0, i - lookback)]
    cur = set(metrics_table(cycle).cu)
    were = set(metrics_table(base).cu)
    vanished = were - cur                      # charters that left during the window
    sub = mg[mg.merging_charter.isin(vanished)]
    return sub.groupby("continuing_charter")["merging_charter"].nunique().to_dict()


@st.cache_data(show_spinner=False)
def inferred_acquirers(cycle, cycle_sig, lookback=4):
    """Likely (not-yet-published) mergers flagged by footprint, not the merger report.
    A single-quarter jump in BOTH members and assets that's too large to be organic:
    even a hot deposit campaign brings in dollars, not tens of thousands of members in
    90 days, so a big member jump alongside an asset jump is a near-certain absorption.
    Requires both within the last `lookback` quarters; excludes already-confirmed
    acquirers. Returns {cu: 1}."""
    confirmed = set(merger_acquirers(cycle, cycle_sig))
    conv = set(charter_alias(cycle_sig))
    cs = sorted(cycle_sig)
    if cycle not in cs:
        return {}
    i = cs.index(cycle)
    window = cs[max(1, i - lookback + 1): i + 1]
    cache = {}

    def met(c):
        if c not in cache:
            cache[c] = metrics_table(c)[["cu", "assets", "members"]]
        return cache[c]

    out = {}
    for q in window:
        qi = cs.index(q)
        if qi == 0:
            continue
        m = met(q).merge(met(cs[qi - 1]), on="cu", suffixes=("", "_0"))
        dm = m.members - m.members_0
        mg = dm / m.members_0.where(m.members_0 > 0)
        ag = (m.assets - m.assets_0) / m.assets_0.where(m.assets_0 > 0)
        hit = m[(mg >= 0.20) & (ag >= 0.15) & (dm >= 1500) & (~m.cu.isin(conv))]
        for cu in hit.cu:
            if cu not in confirmed:
                out[cu] = 1
    return out


def merger_tag(cu, confirmed, inferred):
    """Display tag: confirmed mergers get '✓ ×N', inferred ones a softer '≈ likely'."""
    if confirmed.get(cu):
        return f"\u2713 \u00d7{confirmed[cu]}"
    if cu in inferred:
        return "\u2248 likely"
    return ""


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


def score_breakdown(row, weights):
    rows = []
    is_acq = bool(row.get("is_acquirer"))
    for key, w in weights:
        pr = row.get(f"pct_{key}")
        if is_acq and key.endswith("_growth"):
            rows.append((META[key][0], fmt(key, row[key]), "excluded — merger", "—"))
        else:
            rows.append((META[key][0], fmt(key, row[key]),
                         f"{pr * 100:.0f}th pct" if pd.notna(pr) else "—", f"{w * 100:.0f}%"))
    return pd.DataFrame(rows, columns=["Metric", "Value", "Peer Percentile", "Weight"])


def multi_cu_series(cus, metric, labels, cycle_sig):
    out = {}
    for cu in cus:
        ts = cu_timeseries(cu, cycle_sig)
        if not ts.empty and metric in ts:
            out[labels[cu]] = ts[metric]
    return pd.DataFrame(out)


# ---- NCUA FPR (Financial Performance Report) ratio engine ----------------
# Reproduces the FPR's Key Ratios / Historical Ratios from the 5300 data, on the
# FPR average-balance basis (current period + prior year-end ÷ 2); income is
# annualized year-to-date. Only verified account codes are used; line items that
# would need codes not in the dataset are omitted rather than guessed.
FPR_SECTIONS = [
    ("Capital Adequacy", [
        ("Net Worth / Total Assets", "nw_ratio", "pct"),
        ("Delinquent Loans / Net Worth", "delinq_nw", "pct"),
    ]),
    ("Asset Quality", [
        ("Delinquent Loans / Total Loans", "delinq_loans", "pct"),
        ("Net Charge-Offs / Average Loans", "nco", "pct"),
        ("Delinquent Loans / Total Assets", "delinq_assets", "pct"),
    ]),
    ("Earnings", [
        ("Return on Average Assets", "roaa", "pct"),
        ("Gross Income / Average Assets", "gross_avg", "pct"),
        ("Yield on Average Loans", "yl", "pct"),
        ("Yield on Average Investments", "yi", "pct"),
        ("Yield on Average Earning Assets", "yea", "pct"),
        ("Fee & Other Income / Average Assets", "fee_avg", "pct"),
        ("Cost of Funds / Average Funding", "cof", "pct"),
        ("Interest Expense / Average Assets", "intexp_avg", "pct"),
        ("Net Interest Margin / Average Assets", "nim", "pct"),
        ("Operating Expense / Average Assets", "opex_avg", "pct"),
        ("Provision for Loan Losses / Average Assets", "prov_avg", "pct"),
        ("Operating Expense / Gross Income", "op_gross", "pct"),
    ]),
    ("Asset / Liability Management", [
        ("Total Loans / Total Shares", "loans_shares", "pct"),
        ("Total Loans / Total Assets", "loans_assets", "pct"),
        ("Total Shares / Total Assets", "shares_assets", "pct"),
        ("Borrowings / Total Shares", "borrow_shares", "pct"),
        ("Cash / Total Assets", "cash_assets", "pct"),
    ]),
    ("Productivity", [
        ("Members", "members", "num"),
        ("Average Share Balance per Member", "avg_shares_member", "money"),
    ]),
    ("Growth Rates (year-over-year)", [
        ("Total Assets", "g_assets", "pct"),
        ("Total Loans", "g_loans", "pct"),
        ("Total Shares", "g_shares", "pct"),
        ("Net Worth", "g_nw", "pct"),
        ("Members", "g_members", "pct"),
    ]),
]


def _fpr_ratio_dict(raw, cyc, cycle_sig):
    """Every FPR ratio for one CU at one cycle, per the NCUA FPR Ratio & Formula Guide
    (v3.3). Codes follow the current-era formulas; any missing code yields None (→ N/A)."""
    cur = raw.get(cyc) or {}
    if not cur:
        return {}
    yr, mm = int(cyc[:4]), cyc[-2:]
    pye_c = f"{yr - 1}-12"
    pye = raw.get(pye_c) if pye_c in cycle_sig else None
    yago = raw.get(f"{yr - 1}-{mm}")                       # prior-year quarter end (PYQE)
    factor = 12 / int(mm)                                  # Mar 4, Jun 2, Sep 1.333, Dec 1
    ys = _ys_ratios(cur, pye, factor)
    aa = ys.get("avg_assets")

    def val(c, d=cur):
        v = d.get(c) if d else None
        return v if isinstance(v, (int, float)) and v == v else None

    def vz(c, d=cur):
        v = val(c, d)
        return v if v is not None else 0.0

    def present(*cs, d=cur):
        return any(val(c, d) is not None for c in cs)

    def sumc(codes, d=cur):
        return sum(vz(c, d) for c in codes) if present(*codes, d=d) else None

    def avg(c):
        a, b = val(c), (val(c, pye) if pye else None)
        if not pye or b is None or b <= 0:
            return a
        return ((a or 0.0) + b) / 2.0

    def rt(n, den):
        return 100 * n / den if (n is not None and den not in (None, 0)) else None

    def ann(x):
        return rt(x * factor, aa) if (x is not None and aa) else None

    def growth(c, absden=False):                           # (AC − PYE)/PYE, annualized
        a, b = val(c), (val(c, pye) if pye else None)
        if a is None or b is None:
            return None
        den = abs(b) if absden else b
        return 100 * (a - b) / den * factor if den else None

    assets, loans, shares = val("ACCT_010"), val("ACCT_025B"), val("ACCT_018")
    nw, members, deln = val("ACCT_997"), val("ACCT_083"), val("ACCT_041B")
    ni, opex, ie = val("ACCT_661A"), val("ACCT_671"), val("ACCT_350")
    ii, noni = val("ACCT_115"), val("ACCT_117")
    fee, other_op = val("ACCT_131"), val("ACCT_IS0020")
    borrow = val("ACCT_860C")
    avgL = avg("ACCT_025B")

    # Gross income = interest income + fee income + other operating income (115+131+IS0020)
    if fee is not None or other_op is not None:
        gross = (ii or 0) + (fee or 0) + (other_op or 0)
        feeother = (fee or 0) + (other_op or 0)
    elif ii is not None or noni is not None:
        gross, feeother = (ii or 0) + (noni or 0), noni
    else:
        gross, feeother = None, None

    # Provision = PLLL + credit-loss expense (300 + IS0011); fall back to implied plug
    if present("ACCT_300", "ACCT_IS0011"):
        prov = vz("ACCT_300") + vz("ACCT_IS0011")
    else:
        nii_ = (ii or 0) - (ie or 0)
        prov = ((nii_ + (noni or 0) - (opex or 0) + (val("ACCT_440") or 0)) - ni
                if ni is not None else None)

    # Rolling-12 net charge-offs / average loans
    def nco_of(d):
        return (vz("ACCT_550", d) - vz("ACCT_551", d)) if present("ACCT_550", "ACCT_551", d=d) else None
    nco_amt = nco_of(cur)
    loans_pyqe = val("ACCT_025B", yago) if yago else None
    avg_loans_roll = (((loans or 0) + loans_pyqe) / 2
                      if (loans is not None and loans_pyqe is not None) else None)
    if pye and yago and nco_amt is not None and avg_loans_roll:
        rn = nco_amt + (nco_of(pye) or 0) - (nco_of(yago) or 0)
        nco_roll = rt(rn, avg_loans_roll)
        delnco = rt((deln or 0) + rn, avg_loans_roll)
    else:                                                  # fall back to simple annualized
        nco_roll = rt((nco_amt or 0) * factor, avgL) if nco_amt is not None else None
        delnco = (rt((deln or 0) + (nco_amt or 0) * factor, avgL)
                  if (avgL and (deln is not None or nco_amt is not None)) else None)
    nco_simple = rt((nco_amt or 0) * factor, avgL) if nco_amt is not None else None

    # Investments = total investments + cash on deposit (NV0158 + 730B); residual fallback
    def inv_total(d):
        if val("ACCT_NV0158", d) is not None:
            return vz("ACCT_NV0158", d) + vz("ACCT_730B", d)
        ta = val("ACCT_010", d)
        if ta is None:
            return None
        parts = ["ACCT_025B", "ACCT_730A", "ACCT_730B", "ACCT_007", "ACCT_008",
                 "ACCT_794", "ACCT_009A", "ACCT_009B", "ACCT_009C"]
        return ta - sum((val(c, d) or 0) for c in parts)
    it_cur, it_pye = inv_total(cur), (inv_total(pye) if pye else None)
    g_invest = (100 * (it_cur - it_pye) / it_pye * factor
                if (it_cur is not None and it_pye not in (None, 0)) else None)

    # Capital adequacy
    nw_den = val("ACCT_NW0010") if val("ACCT_NW0010") is not None else assets
    nw0004 = vz("ACCT_NW0004")
    acl, alll = vz("ACCT_AS0048"), vz("ACCT_719")
    nw_acl = (rt((nw or 0) + acl + alll, (assets or 0) + acl + alll)
              if (present("ACCT_719", "ACCT_AS0048") and nw is not None and assets) else None)
    rbc = val("ACCT_RB0172")
    if rbc is None and val("ACCT_RB0171"):
        rbc = rt(val("ACCT_RB0012"), val("ACCT_RB0171"))
    gaap_eq = sumc(["ACCT_940", "ACCT_668", "ACCT_658", "ACCT_658A", "ACCT_996",
                    "ACCT_945B", "ACCT_945A", "ACCT_EQ0009", "ACCT_945C", "ACCT_602"])
    lc_num = sumc(["ACCT_020B", "ACCT_041B", "ACCT_644", "ACCT_798A", "ACCT_1001F"])
    loss_cov = rt(lc_num, (nw or 0) + alll + acl) if lc_num is not None else None
    solv_num = ((assets or 0) - vz("ACCT_860C") - vz("ACCT_925A") - vz("ACCT_825")
                - vz("ACCT_668") - vz("ACCT_820A"))
    solvency = rt(solv_num, shares) if (assets is not None and shares) else None
    classified = (rt(vz("ACCT_719") + vz("ACCT_AS0048") + vz("ACCT_668"), nw)
                  if (present("ACCT_719", "ACCT_AS0048") and nw) else None)
    nw_excl_cecl = (rt((nw or 0) - nw0004, (nw_den or 0) - nw0004)
                    if nw is not None else None)
    cecl_date = cur.get("ACCT_NW0001")
    if cecl_date in (None, "", 0, 0.0):
        cecl_date = None
    elif isinstance(cecl_date, float) and cecl_date == int(cecl_date):
        cecl_date = str(int(cecl_date))

    # Asset quality
    htm = rt(val("ACCT_801"), val("ACCT_AS0073"))
    eq9, as67 = val("ACCT_EQ0009"), val("ACCT_AS0067")
    afs = (rt(eq9, as67 - eq9) if (eq9 is not None and as67 is not None) else None)
    other_npa = rt(val("ACCT_798A"), assets)

    # Earnings
    extra = sum(vz(c) for c in ["ACCT_IS0046", "ACCT_IS0047", "ACCT_421", "ACCT_430",
                                "ACCT_431", "ACCT_IS0029", "ACCT_IS0030"])
    fixed_repo = (rt(vz("ACCT_007") + vz("ACCT_008") + vz("ACCT_798A"), assets)
                  if present("ACCT_007", "ACCT_008", "ACCT_798A") else None)
    net_op_exp = (ann((opex or 0) - (fee or 0)) if (opex is not None) else None)

    # Asset / liability management
    lt_num = (vz("ACCT_703A") + vz("ACCT_386A") + vz("ACCT_386B") - vz("ACCT_RL0050")
              + vz("ACCT_718A3") + vz("ACCT_718A4") - vz("ACCT_CM0099") + vz("ACCT_NV0155")
              + vz("ACCT_NV0156") + vz("ACCT_NV0157") + vz("ACCT_007") + vz("ACCT_008")
              + vz("ACCT_794"))
    net_lt = (rt(lt_num, assets)
              if (present("ACCT_703A", "ACCT_NV0155", "ACCT_718A3") and assets) else None)
    sh_borr = ((shares or 0) + (borrow or 0)) if present("ACCT_018", "ACCT_860C") else None
    ea = (loans + it_cur) if (loans is not None and it_cur is not None) else None
    reg_sh, drafts = val("ACCT_657"), val("ACCT_902")
    rd_num = ((reg_sh or 0) + (drafts or 0)) if present("ACCT_657", "ACCT_902") else None

    # Productivity
    fte = (vz("ACCT_564A") + vz("ACCT_564B") / 2) if present("ACCT_564A", "ACCT_564B") else None
    nloans = val("ACCT_025A")

    d = dict(ys)                                          # yl, yi, yea, cof, spread, nim
    d.update({
        # capital adequacy
        "nw_ratio": rt(nw, nw_den),
        "nw_acl": nw_acl,
        "rbc": rbc,
        "gaap_eq": rt(gaap_eq, assets) if gaap_eq is not None else None,
        "loss_cov": loss_cov,
        "nw_excl_cecl": nw_excl_cecl,
        "cecl_adopted": ("Yes" if cecl_date else None),
        "cecl_date": cecl_date,
        "solvency": solvency,
        "classified_nw": classified,
        # asset quality
        "delinq_nw": rt(deln, nw),
        "delinq_loans": rt(deln, loans),
        "delinq_assets": rt(deln, assets),
        "nco": nco_simple,                                # simple annualized (Historical)
        "nco_roll": nco_roll,                             # rolling 12-month (Key)
        "delnco_loans": delnco,
        "other_npa": other_npa,
        "htm_fair": htm,
        "afs_gl": afs,
        # earnings
        "roaa": ann(ni),
        "roaa_ex": ann((ni - extra)) if ni is not None else None,
        "gross_avg": ann(gross),
        "fee_avg": ann(feeother),
        "intexp_avg": ann(ie),                            # cost of funds / avg assets
        "net_margin_avg": ann((gross - (ie or 0))) if gross is not None else None,
        "opex_avg": ann(opex),
        "prov_avg": ann(prov),
        "op_gross": rt(opex, gross) if (opex is not None and gross) else None,
        "net_op_exp": net_op_exp,
        "fixed_repo": fixed_repo,
        # asset / liability management
        "net_lt": net_lt,
        "loans_shares": rt(loans, shares),
        "loans_assets": rt(loans, assets),
        "shares_assets": rt(shares, assets),
        "borrow_shares": rt(borrow, shares),
        "borrow_sh_nw": rt(borrow, ((shares or 0) + (nw or 0)) if present("ACCT_018", "ACCT_997") else None),
        "reg_shares": rt(reg_sh, sh_borr),
        "reg_drafts": rt(rd_num, sh_borr),
        "shares_dep_borr_ea": rt(sh_borr, ea),
        "cash_assets": rt(vz("ACCT_730A") + vz("ACCT_730B"), assets),
        "cash_st_assets": (rt(vz("ACCT_730A") + vz("ACCT_730B") + vz("ACCT_NV0153"), assets)
                           if present("ACCT_730A", "ACCT_730B", "ACCT_NV0153") else None),
        # productivity
        "members": members,
        "mem_potential": rt(members, val("ACCT_084")),
        "borrowers_mem": rt(nloans, members),
        "mem_fte": (members / fte if (members is not None and fte) else None),
        "avg_shares_member": (shares / members if (shares is not None and members) else None),
        "avg_loan_bal": (loans / nloans if (loans is not None and nloans) else None),
        "salary_fte": ((vz("ACCT_210") * factor) / fte if (present("ACCT_210") and fte) else None),
        # growth (NCUA basis: (current − prior year-end) / prior year-end, annualized)
        "g_assets": growth("ACCT_010"), "g_loans": growth("ACCT_025B"),
        "g_shares": growth("ACCT_018"), "g_nw": growth("ACCT_997", absden=True),
        "g_members": growth("ACCT_083"), "g_invest": g_invest,
    })
    return d


def _fmt_ratio(v, fmt):
    if v is None:
        return "—"
    if fmt == "pct":
        return pct(v)
    if fmt == "money":
        return money(v)
    if fmt == "num":
        return f"{v:,.0f}"
    return str(v)


def fpr_ratio_table(raw, periods, mode, cycle_sig):
    labels = [_period_label(p, mode) for p in periods]
    dicts = {p: _fpr_ratio_dict(raw, p, cycle_sig) for p in periods}
    rows = []
    for sec, items in FPR_SECTIONS:
        rows.append({"": sec, **{lb: "" for lb in labels}})
        for lbl, key, fmt in items:
            rec = {"": lbl}
            for p, lb in zip(periods, labels):
                rec[lb] = _fmt_ratio(dicts[p].get(key), fmt)
            rows.append(rec)
    return pd.DataFrame(rows)


# ---- NCUA Key Ratios layout (mirrors the FPR "Key Ratios" report) ----------
# (label, ratio-key | None for N/A, format, footnote-marker)
FPR_KEY_SECTIONS = [
    ("Capital Adequacy Ratios", [
        ("Net Worth / Total Assets for Prompt Corrective Action", "nw_ratio", "pct", "6"),
        ("Net Worth + ALLL or ACL / Total Assets + ALLL or ACL", "nw_acl", "pct", ""),
        ("Risk-Based Capital Ratio", "rbc", "pct", ""),
        ("GAAP Equity / Total Assets", "gaap_eq", "pct", ""),
        ("Loss Coverage", "loss_cov", "pct", ""),
    ]),
    ("Asset Quality Ratios", [
        ("Delinquent Loans / Total Loans", "delinq_loans", "pct", ""),
        ("Delinquent Loans / Net Worth", "delinq_nw", "pct", ""),
        ("Rolling 12 Month Net Charge Offs / Average Loans", "nco_roll", "pct", "2"),
        ("Delinquent Loans + Net Charge-Offs / Average Loans", "delnco_loans", "pct", ""),
        ("Other Non-Performing Assets / Total Assets", "other_npa", "pct", ""),
    ]),
    ("Management Ratios", [
        ("Net Worth Growth", "g_nw", "pct", "1"),
        ("Share Growth", "g_shares", "pct", "1"),
        ("Loan Growth", "g_loans", "pct", "1"),
        ("Asset Growth", "g_assets", "pct", "1"),
        ("Investment Growth", "g_invest", "pct", "1"),
        ("Membership Growth", "g_members", "pct", "1"),
    ]),
    ("Earnings Ratios", [
        ("Net Income / Average Assets (ROAA)", "roaa", "pct", "1"),
        ("Net Income - Extraordinary Gains(Losses) / Average Assets", "roaa_ex", "pct", "1"),
        ("Non-Interest Expense / Average Assets", "opex_avg", "pct", "1"),
        ("PLLL or Credit Loss Expense / Average Assets", "prov_avg", "pct", "1"),
    ]),
    ("Liquidity", [
        ("Total Loans / Total Assets", "loans_assets", "pct", ""),
        ("Cash + Short-Term Investments / Assets", "cash_st_assets", "pct", "3"),
    ]),
    ("Sensitivity to Market Risk", [
        ("Est. NEV Tool Post Shock Ratio", None, "pct", "4"),
        ("Est. NEV Tool Post Shock Sensitivity", None, "pct", "4"),
    ]),
]

# ---- NCUA Historical Ratios layout (mirrors the FPR "Historical Ratios" report) ----
FPR_HIST_SECTIONS = [
    ("Capital Adequacy", [
        ("Has the credit union adopted ASC topic 326 (CECL)?", "cecl_adopted", "text", ""),
        ("Effective date of adoption of ASC Topic 326 - Financial Instruments - "
         "Credit Losses (CECL)", "cecl_date", "text", ""),
        ("Net Worth / Total Assets excluding CECL Transition Provision", "nw_excl_cecl", "pct", "3"),
        ("Net Worth / PCA Opt. Total Assets (if applies)", None, "pct", ""),
        ("Net Worth / Total Assets excluding one-time adjustment to undivided earnings for "
         "the adoption of ASC topic 326 (CECL)", None, "pct", "1"),
        ("Solvency Evaluation (Estimated)", "solvency", "pct", ""),
        ("Classified Assets (Estimated) / Net Worth", "classified_nw", "pct", ""),
    ]),
    ("Asset Quality", [
        ("Net Charge-Offs / Average Loans", "nco", "pct", "*"),
        ("Fair (Market) HTM Invest Value / Book Value HTM Invest.", "htm_fair", "pct", ""),
        ("Accum Unreal G/L On AFS / Cost Of AFS", "afs_gl", "pct", ""),
        ("Delinquent Loans / Assets", "delinq_assets", "pct", ""),
    ]),
    ("Earnings", [
        ("Gross Income / Average Assets", "gross_avg", "pct", "*"),
        ("Yield on Average Loans", "yl", "pct", ""),
        ("Yield on Average Investments", "yi", "pct", "*"),
        ("Fee & Other Op. Income / Avg. Assets", "fee_avg", "pct", "*"),
        ("Cost of Funds / Avg. Assets", "intexp_avg", "pct", "*"),
        ("Net Margin / Avg. Assets", "net_margin_avg", "pct", "*"),
        ("Net Interest Margin / Avg. Assets", "nim", "pct", "*"),
        ("Non-Interest Expense / Gross Income", "op_gross", "pct", ""),
        ("Fixed Assets & Foreclosed & Repossessed Assets / Total Assets", "fixed_repo", "pct", ""),
        ("Net Operating Exp. / Avg. Assets", "net_op_exp", "pct", "*"),
    ]),
    ("Asset / Liability Management", [
        ("Net Long-Term Assets / Total Assets", "net_lt", "pct", ""),
        ("Reg. Shares / Total Shares & Borrowings", "reg_shares", "pct", ""),
        ("Total Loans / Total Shares", "loans_shares", "pct", ""),
        ("Total Shares, Dep. & Borrs / Earning Assets", "shares_dep_borr_ea", "pct", ""),
        ("Reg Shares + Share Drafts / Total Shares & Borrs", "reg_drafts", "pct", ""),
        ("Borrowings / Total Shares & Net Worth", "borrow_sh_nw", "pct", ""),
    ]),
    ("Productivity", [
        ("Members / Potential Members", "mem_potential", "pct", ""),
        ("Borrowers / Members", "borrowers_mem", "pct", ""),
        ("Members / Full-Time Empl.", "mem_fte", "num", ""),
        ("Avg. Shares Per Member", "avg_shares_member", "money", ""),
        ("Avg. Loan Balance", "avg_loan_bal", "money", ""),
        ("Salary And Benefits / Full-Time Empl.", "salary_fte", "money", "*"),
    ]),
]


def _fpr_ratio_cell(v, fmt):
    if v is None or (isinstance(v, float) and v != v):
        return "N/A"
    if fmt == "text":
        return str(v)
    if fmt == "pct":
        return f"{v:.2f}"
    if fmt == "money":
        return money(v)
    if fmt == "num":
        return f"{v:,.0f}"
    return str(v)


@st.cache_data(show_spinner=False)
def fpr_peer_avg(band, cycle, cycle_sig):
    """Peer-group (asset-band) average of every FPR ratio at one cycle.
    Bulk-pulls the band's accounts, then reuses _fpr_ratio_dict per CU."""
    mt_c = metrics_table(cycle)
    cus = [str(c) for c, a in zip(mt_c.cu, mt_c.assets) if band_of(a) == band]
    if not cus:
        return {}, 0
    yr, mm = int(cycle[:4]), cycle[-2:]
    want = {cycle, f"{yr - 1}-12", f"{yr - 1}-{mm}"} & set(cycle_sig)
    raw_by = {c: {} for c in cus}
    inlist = ",".join("'%s'" % c for c in cus)
    cyclist = ",".join("'%s'" % c for c in want)
    for tbl in [t for t in available_tables() if t.upper().startswith("FS220")]:
        try:
            curx = con.execute(
                f"SELECT * FROM read_parquet('{glob_for(tbl)}', hive_partitioning=true, "
                f"union_by_name=true) WHERE cycle IN ({cyclist}) "
                f"AND CAST(CU_NUMBER AS VARCHAR) IN ({inlist})")
            rows = curx.fetchall()
        except Exception:
            continue
        ix = {d[0].upper(): i for i, d in enumerate(curx.description)}
        cu_i, cyc_i = ix.get("CU_NUMBER"), ix.get("CYCLE")
        acc = [(c, i) for c, i in ix.items() if c.startswith("ACCT_")]
        if cu_i is None or cyc_i is None:
            continue
        for r in rows:
            slot = raw_by.get(str(r[cu_i]))
            if slot is None:
                continue
            dd = slot.setdefault(str(r[cyc_i]), {})
            for name, i in acc:
                v = r[i]
                if v is not None:
                    try:
                        dd[name] = float(v)
                    except (TypeError, ValueError):
                        pass
    agg, n_cu = {}, 0
    for c in cus:
        raw = raw_by.get(c) or {}
        if cycle not in raw:
            continue
        n_cu += 1
        for k, v in _fpr_ratio_dict(raw, cycle, cycle_sig).items():
            if isinstance(v, (int, float)) and v == v:
                agg.setdefault(k, []).append(v)
    return {k: sum(vs) / len(vs) for k, vs in agg.items() if vs}, n_cu


def roa_bridge_steps(subj, peer):
    """Decompose the ROAA gap between a CU and its peer average into the standard NCUA
    earnings drivers, each as % of average assets. Returns
    {"start": (label, peer_roaa), "steps": [(label, contribution), ...],
     "end": (label, subj_roaa)} with start + sum(steps) == end, or None when ROAA is
    unavailable on either side. Cost lines (operating expense, provision) contribute
    negatively — spending more than peers lowers ROAA. A residual "Other" line absorbs
    gains/taxes/stabilization so the bars always foot to the reported ROAA."""
    def g(d, k):
        v = d.get(k) if d else None
        return v if isinstance(v, (int, float)) and v == v else None
    sr, pr = g(subj, "roaa"), g(peer, "roaa")
    if sr is None or pr is None:
        return None
    comps = [("Net interest margin", "nim", 1),
             ("Fee & other income", "fee_avg", 1),
             ("Operating expense", "opex_avg", -1),
             ("Provision expense", "prov_avg", -1)]
    steps, explained = [], 0.0
    for label, key, sign in comps:
        contrib = sign * ((g(subj, key) or 0.0) - (g(peer, key) or 0.0))
        explained += contrib
        steps.append((label, contrib))
    steps.append(("Other (gains, taxes)", (sr - pr) - explained))   # forces the foot
    return {"start": ("Peer avg ROAA", pr), "steps": steps, "end": ("This CU ROAA", sr)}


@st.cache_data(show_spinner="Aggregating the peer band…")
def peer_mix_frame(band, cycle, src, cycle_sig):
    """Dollar-weighted composition of an asset band for a mix source ('Loan mix' /
    'Deposit mix'), returned as a (Category, Amount $M, Share) frame matching mix_frame,
    plus the peer count. Component dollars are summed across the band's CUs (first
    non-null per code per CU, mirroring acct_values), then run through mix_frame on the
    aggregate so the categories and residual line up exactly with the single-CU view."""
    if src not in MIX_SOURCES:
        return pd.DataFrame(), 0
    parts, total_code, resid = MIX_SOURCES[src]
    mt_c = metrics_table(cycle)
    cus = [str(c) for c, a in zip(mt_c.cu, mt_c.assets) if band_of(a) == band]
    if not cus:
        return pd.DataFrame(), 0
    codes = {total_code.upper()} | {c.upper() for _, cs in parts for c in cs}
    per = {c: {} for c in cus}
    inlist = ",".join("'%s'" % c for c in cus)
    for tbl in [t for t in available_tables() if t.upper().startswith("FS220")]:
        try:
            curx = con.execute(
                f"SELECT * FROM read_parquet('{glob_for(tbl)}', hive_partitioning=true, "
                f"union_by_name=true) WHERE cycle = ? "
                f"AND CAST(CU_NUMBER AS VARCHAR) IN ({inlist})", [cycle])
            rows = curx.fetchall()
        except Exception:
            continue
        ix = {d[0].upper(): i for i, d in enumerate(curx.description)}
        cu_i = ix.get("CU_NUMBER")
        present = [(c, ix[c]) for c in codes if c in ix]
        if cu_i is None or not present:
            continue
        for r in rows:
            slot = per.get(str(r[cu_i]))
            if slot is None:
                continue
            for c, i in present:
                if c not in slot and r[i] is not None:
                    try:
                        slot[c] = float(r[i])
                    except (TypeError, ValueError):
                        pass
    agg, n = {c: 0.0 for c in codes}, 0
    tc = total_code.upper()
    for c in cus:
        d = per[c]
        if d.get(tc, 0) and d[tc] > 0:
            n += 1
        for code, val in d.items():
            agg[code] += val
    return mix_frame(agg, parts, total_code, resid), n


@st.cache_data(show_spinner="Finding statistical twins…")
def find_twins(cu, cycle, cycle_sig, n=20):
    """Up to n CU numbers most similar to `cu` at `cycle`, blending asset size (|log
    ratio|), loan-mix profile (Euclidean distance between component share-vectors), and
    region (same-state preferred). The candidate pool is pre-filtered to a 0.4x–2.5x
    asset window so the loan-mix pull stays bounded. Lower score = more similar. Returns
    [] when the subject or pool is unavailable; mix distance is neutral when a CU doesn't
    report a loan book, so size/region still rank it."""
    import math
    mt_c = metrics_table(cycle)
    srow = mt_c[mt_c.cu == cu]
    if srow.empty:
        return []
    s_assets, s_state = srow.assets.iloc[0], srow.state.iloc[0]
    if not (isinstance(s_assets, (int, float)) and s_assets and s_assets > 0):
        return []
    pool = mt_c[(mt_c.cu != cu) & mt_c.assets.between(s_assets * 0.4, s_assets * 2.5)]
    if pool.empty:
        return []
    parts, total_code, resid = MIX_SOURCES["Loan mix"]
    codes = {total_code.upper()} | {c.upper() for _, cs in parts for c in cs}
    cand = [str(c) for c in pool.cu] + [str(cu)]
    per = {c: {} for c in cand}
    inlist = ",".join("'%s'" % c for c in cand)
    for tbl in [t for t in available_tables() if t.upper().startswith("FS220")]:
        try:
            curx = con.execute(
                f"SELECT * FROM read_parquet('{glob_for(tbl)}', hive_partitioning=true, "
                f"union_by_name=true) WHERE cycle = ? "
                f"AND CAST(CU_NUMBER AS VARCHAR) IN ({inlist})", [cycle])
            rows = curx.fetchall()
        except Exception:
            continue
        ix = {d[0].upper(): i for i, d in enumerate(curx.description)}
        cu_i = ix.get("CU_NUMBER")
        present = [(c, ix[c]) for c in codes if c in ix]
        if cu_i is None or not present:
            continue
        for r in rows:
            slot = per.get(str(r[cu_i]))
            if slot is None:
                continue
            for c, i in present:
                if c not in slot and r[i] is not None:
                    try:
                        slot[c] = float(r[i])
                    except (TypeError, ValueError):
                        pass

    def share_vec(d):
        tot = d.get(total_code.upper())
        if not tot or tot <= 0:
            return None
        return [sum((d.get(c.upper(), 0.0) or 0.0) for c in cs) / tot for _, cs in parts]

    s_vec = share_vec(per.get(str(cu), {}))
    scored = []
    for prow in pool.itertuples():
        a = prow.assets
        size_d = abs(math.log((a or 1e-9) / s_assets)) if (a and a > 0) else 3.0
        cvec = share_vec(per.get(str(prow.cu), {}))
        mix_d = (math.sqrt(sum((x - y) ** 2 for x, y in zip(s_vec, cvec)))
                 if (s_vec is not None and cvec is not None) else 0.5)
        state_pen = 0.0 if (prow.state == s_state and s_state) else 0.25
        scored.append((1.2 * size_d + 1.0 * mix_d + state_pen, prow.cu))
    scored.sort(key=lambda t: t[0])
    return [c for _, c in scored[:n]]


def fpr_ratio_html(cu, sections, mode, anchor, cycle_sig, band, n, peer=True):
    raw = cu_statement_raw(cu, cycle_sig)
    cs = sorted(cycle_sig)
    if mode == "Years":
        periods = [c for c in cs if c.endswith("-12") and c <= anchor][-n:]
    else:
        periods = [c for c in cs if c <= anchor][-n:]
    if not periods:
        return None, 0
    dicts = {p: _fpr_ratio_dict(raw, p, cycle_sig) for p in periods}
    pavg, n_peer = (fpr_peer_avg(band, periods[-1], cycle_sig) if peer else ({}, 0))
    cols = [_fpr_collabel(p, mode) for p in periods]
    ncol = len(cols) + (1 if peer else 0)
    head = "".join(f"<th>{c}</th>" for c in cols)
    if peer:
        head += '<th class="pa">Peer Avg.</th>'
    thead = f'<thead><tr><th class="li">Line Item</th>{head}</tr></thead>'
    body = []
    for sec, items in sections:
        body.append(f'<tr class="sec"><td colspan="{ncol + 1}">{sec.upper()}</td></tr>')
        for it in items:
            label, key, fmt = it[0], it[1], it[2]
            note = it[3] if len(it) > 3 else ""
            lbl_html = label + (f"<sup>{note}</sup>" if note else "")
            cells = [f'<td class="lbl">{lbl_html}</td>']
            for p in periods:
                cells.append(f'<td class="rt">'
                             f'{_fpr_ratio_cell(dicts[p].get(key) if key else None, fmt)}</td>')
            if peer:
                cells.append(f'<td class="rt pa">'
                             f'{_fpr_ratio_cell(pavg.get(key) if key else None, fmt)}</td>')
            body.append(f'<tr>{"".join(cells)}</tr>')
    return f'<table class="fprfs fprkr">{thead}<tbody>{"".join(body)}</tbody></table>', n_peer


# ---------------------------------------------------------------------------- UI

st.title("NCUA Call Report Explorer")

tables = available_tables()
if "FOICU" not in tables:
    st.error("No data under ./data. Run the ingest (or GitHub Action) and commit data/.")
    st.stop()

all_cycles = cycles()
sig = tuple(sorted(all_cycles))
health = data_health(sig)

# ---- Sidebar: brand ----
st.sidebar.markdown("#### Call Report Explorer")

# ---- Sidebar: navigation (grouped by workflow, icon-labeled) ----
NAV = ["Profile", "FPR", "ROA Bridge", "Compare", "Chart", "Rankings", "Yields", "M&A Targets",
       "Merger History", "Industry", "Data Health"]
NAV_ICON = {"Profile": "👤", "FPR": "📄", "ROA Bridge": "🌉", "Compare": "⚖️", "Chart": "📊", "Rankings": "🏆",
            "Yields": "📈", "M&A Targets": "🎯",
            "Merger History": "🔀", "Industry": "🏛️", "Data Health": "🩺"}
# Deep-link routing: ranked tables link to ?view=<page>&cu=<id> to jump straight there.
_qp = st.query_params
if _qp.get("view") in NAV:
    st.session_state["nav_page"] = _qp.get("view")
if _qp.get("cu"):
    st.session_state["profile_pending_cu"] = _qp.get("cu")
if ("view" in _qp) or ("cu" in _qp):
    st.query_params.clear()

st.sidebar.markdown("""<style>
section[data-testid="stSidebar"] div[data-testid="stButton"]{margin:1px 0}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button{
  border:1px solid #e2e5ea !important;box-shadow:none !important;width:100% !important;
  background:transparent !important;color:rgb(49,51,63) !important;
  text-align:left !important;justify-content:flex-start !important;
  padding:7px 12px !important;border-radius:9px !important;
  font-size:.9rem !important;font-weight:normal !important;
  transition:background .12s !important}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover{
  background:#eceff4 !important;border-color:#c8cdd6 !important;color:rgb(49,51,63) !important}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button[data-testid="baseButton-primary"]{
  background:#e6ebf5 !important;font-weight:600 !important;color:#1a2a4a !important;
  border-color:#b8c4da !important}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button[data-testid="baseButton-primary"]:hover{
  background:#dce4f0 !important}
</style>""", unsafe_allow_html=True)

# ---- Dev gate -------------------------------------------------------
st.session_state.setdefault("nav_page", NAV[0])
for _p in NAV:
    if st.sidebar.button(f"{NAV_ICON[_p]}  {_p}", key=f"nav_{_p}",
                         use_container_width=True,
                         type="primary" if st.session_state.get("nav_page") == _p else "secondary"):
        st.session_state["nav_page"] = _p
        st.rerun()
page = st.session_state.get("nav_page", NAV[0])


def pick(label, options, key, help=None):
    """Tap-to-select control matching the nav's radio style. Radio always returns a
    valid option (unlike segmented pills, which can be deselected to None). Persists
    across pages via session_state."""
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[0]
    return st.sidebar.radio(label, options, key=key, help=help)


# ---- Sidebar: controls (Quarter always; growth/lens only where they apply) ----
st.sidebar.divider()
st.sidebar.caption("Settings")
cycle = st.sidebar.selectbox("Quarter", all_cycles)

SCORING_PAGES = {"Profile", "Compare", "Rankings"}        # show the score-lens picker
GROWTH_PAGES = {"Profile", "Compare", "Rankings", "M&A Targets"}  # show growth basis
GROWTH_OPTS = ["Year-Over-Year", "Quarter-Over-Quarter (Annualized)"]
LENS_OPTS = list(SCORE_LENSES)
st.session_state.setdefault("growth_label", GROWTH_OPTS[0])
st.session_state.setdefault("score_lens", LENS_OPTS[0])

if page in GROWTH_PAGES:
    pick("Growth basis", GROWTH_OPTS, "growth_label")
growth_label = st.session_state.get("growth_label")
if growth_label not in GROWTH_OPTS:
    growth_label = GROWTH_OPTS[0]
basis = "YoY" if growth_label.startswith("Year") else "QoQ"

if page in SCORING_PAGES:
    pick("Score lens", LENS_OPTS, "score_lens",
         help="Momentum weights growth heaviest (who's expanding fastest). Performance "
              "weights earnings, capital, and asset quality (who's financially strongest).")
lens = st.session_state.get("score_lens")
if lens not in SCORE_LENSES:
    lens = LENS_OPTS[0]
weights = SCORE_LENSES[lens]

# ---- Sidebar: data-health status — surfaced only when a cycle is flagged ----
_worst = ("error" if any(h["status"] == "error" for h in health.values())
          else "warn" if any(h["status"] == "warn" for h in health.values()) else "ok")
if _worst != "ok":
    _bad = sum(1 for h in health.values() if h["status"] != "ok")
    _icon = {"warn": "🟡", "error": "🔴"}[_worst]
    st.sidebar.divider()
    with st.sidebar.expander(
            f"{_icon} Data health — {_bad} cycle{'s' if _bad != 1 else ''} flagged",
            expanded=True):
        st.caption("Flagged cycles may show missing (—) or unreliable figures. "
                   "Re-check the affected partitions and re-run the ingest.")
        for c in sorted(health, reverse=True):
            h = health[c]
            if h["status"] == "ok":
                continue
            mark = {"warn": "🟡", "error": "🔴"}[h["status"]]
            st.markdown(f"{mark} **{c}** · {h['cu_count']:,} CUs")
            for msg in h["issues"]:
                st.caption("• " + msg)

mt = enriched_table(cycle, basis, lens, sig)
# Always pull city directly from metrics_table (the rate_table path that works correctly)
# rather than relying on it surviving the enriched_table merge chain.
_city_direct = metrics_table(cycle).set_index("cu")["city"]
mt["city"] = mt.cu.map(_city_direct).fillna("")

# label helper shared across pages
ALL_LABELS = {r.cu: f"{r.cu_name} (#{r.cu}, {r.state})" for r in mt.itertuples()}


def universe_picker(df, key):
    """Industry / band / state filter shared by the Rates and Targets pages.
    Returns (filtered_df, human_label)."""
    mode = st.radio("Universe", ["Whole Industry", "By Asset Band", "By State"],
                    horizontal=True, key=f"{key}_mode")
    if mode == "By Asset Band":
        b = st.selectbox("Asset band", [x[2] for x in BANDS], index=4, key=f"{key}_band")
        return df[df.band == b], f"the {b} band"
    if mode == "By State":
        states = sorted(s for s in df.state.dropna().unique() if s)
        if not states:
            return df, "all credit unions"
        s = st.selectbox("State", states, key=f"{key}_state")
        return df[df.state == s], s
    return df, "all credit unions"


def screen_filters(df, key, default_top=100, show_top=True):
    """State(s) + Asset size multiselects (both optional, empty = all) and an optional
    Show-top control. Returns (filtered_df, top_n_or_None, human_label)."""
    cols = st.columns([2, 2, 1] if show_top else [1, 1])
    all_states = sorted(s for s in df.state.dropna().unique() if s)
    sel_states = cols[0].multiselect("State(s)", all_states, default=[], key=f"{key}_states")
    sel_bands = cols[1].multiselect("Asset size", [b[2] for b in BANDS], default=[],
                                    key=f"{key}_bands")
    top_n = (cols[2].number_input("Show top", min_value=10, max_value=2000,
                                  value=default_top, step=10, key=f"{key}_top")
             if show_top else None)
    out = df
    if sel_states:
        out = out[out.state.isin(sel_states)]
    if sel_bands:
        out = out[out.band.isin(sel_bands)]
    parts = []
    if sel_states:
        parts.append(", ".join(sel_states))
    if sel_bands:
        parts.append(", ".join(sel_bands))
    return out, (int(top_n) if top_n is not None else None), \
        (" · ".join(parts) if parts else "all credit unions")


# ============================================================ PROFILE
if page == "Profile":
    _pend = st.session_state.pop("profile_pending_cu", None)
    if _pend is not None:
        try:
            _pend = int(_pend)
        except (TypeError, ValueError):
            pass
        _nm = ALL_LABELS.get(_pend, "").split(" (#")[0]
        if _nm:
            st.session_state["profile_search"] = _nm
            st.session_state["profile_pick"] = _pend
    st.session_state.setdefault("profile_search", "BluCurrent")
    query = st.text_input("Search a credit union by name", key="profile_search",
                          placeholder="e.g. BluCurrent")
    if not query:
        st.info("Type part of a credit union name to begin.")
    else:
        hits = mt[name_matches(mt.cu_name, query)].head(300)
        st.caption(f"{len(hits)} match(es) in {cycle}")
        if not hits.empty:
            labels = {r.cu: ALL_LABELS[r.cu] for r in hits.itertuples()}
            _opts = list(labels)
            _pp = st.session_state.get("profile_pick")
            _idx = _opts.index(_pp) if _pp in labels else 0
            cu = st.selectbox("Select a credit union", _opts, index=_idx,
                              format_func=lambda n: labels[n])
            row = mt[mt.cu == cu].iloc[0]
            ts = cu_timeseries(cu, sig)
            prev_cy = prior_cycle(cycle)
            prev_row = (ts.loc[prev_cy].to_dict()
                        if (not ts.empty and prev_cy in ts.index) else None)

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

            # ---- KPI hero row (persistent above the tabs) ---------------------
            h = st.columns(5)
            for c, key in zip(h[:4], ["assets", "members", "nw_ratio", "roa"]):
                metric_card(c, key, row, prev_row)
            h[4].metric("Peer Score",
                        f"{row.score:.0f}/100" if pd.notna(row.score) else "—",
                        help=f"{lens} percentile rank within the {row.band} asset band; "
                             "50 = band median, 80+ = ★★★★★.")
            st.markdown(
                f"<div style='font-size:1.6rem;line-height:1.1'>{stars_str(row.stars)}"
                "</div>", unsafe_allow_html=True)
            cap = f"{lens} peer score vs {row.band} peers — 50 = band median, 80+ = ★★★★★."
            if prev_row is not None:
                cap += f"  ·  ▲▼ deltas vs prior quarter ({prev_cy})."
            st.caption(cap)
            with st.expander("How This Score Is Built"):
                st.dataframe(score_breakdown(row, weights), use_container_width=True,
                             hide_index=True)

            tab_ov, tab_fin, tab_tr, tab_peer = st.tabs(
                ["Overview", "Financials", "Trends", "Peers"])

            # ===================================================== OVERVIEW
            with tab_ov:
                scorecard_groups = [
                    ("Size & Balance Sheet",
                     ["assets", "loans", "shares", "net_worth", "net_income", "members"]),
                    (f"Growth ({growth_label})", GROWTH_KEYS),
                    ("Profitability", ["roa", "roe", "nim", "efficiency"]),
                    ("Capital, Asset Quality & Liquidity",
                     ["nw_ratio", "delinquency", "nco", "lts"]),
                ]
                for i, (title, keys) in enumerate(scorecard_groups):
                    if i:
                        st.divider()
                    st.markdown(f"**{title}**")
                    cols = st.columns(len(keys))
                    for col, key in zip(cols, keys):
                        metric_card(col, key, row, prev_row)

                st.divider()
                st.markdown("**Composite Score History**")
                hist = score_history(cu, basis, lens, cycle, sig)
                if len(hist) > 1:
                    labels = [c[:4] if c.endswith("-12") else c for c in hist.index]
                    chart_df = pd.DataFrame({"Peer Score": hist.score.values}, index=labels)
                    st.line_chart(chart_df, y="Peer Score")
                    htbl = pd.DataFrame({
                        "Period": labels,
                        "Peer Score": [f"{v:.0f}/100" for v in hist.score],
                        "Stars": [stars_str(v) for v in hist.stars],
                    })
                    st.dataframe(htbl, use_container_width=True, hide_index=True)
                    st.caption(f"{lens} peer score at each year-end (plus the selected "
                               "quarter). Score = percentile rank within the asset band "
                               "(50 = band median). History follows the charter across conversions.")
                else:
                    st.caption("Composite history needs more than one period of data.")

                st.divider()
                sc = pd.DataFrame(
                    {"Value": {META[k][0]: fmt(k, row[k]) for k, _, _, _ in METRICS}})
                st.download_button(
                    "Download scorecard (Excel)",
                    to_excel_bytes({"Scorecard": sc}),
                    file_name=f"{row.cu_name}_{cycle}_scorecard.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
                with st.expander("Efficiency Ratio Breakdown"):
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

            # ===================================================== FINANCIALS
            with tab_fin:
                sc1, sc2 = st.columns(2)
                stmt = sc1.radio("Statement", ["Balance Sheet", "Income Statement"],
                                 horizontal=True)
                pmode = sc2.radio("Periods", ["Quarters", "Years"], horizontal=True)
                schema = BALANCE_SHEET if stmt == "Balance Sheet" else INCOME_STATEMENT
                sdf = build_statement(cu, schema, stmt == "Income Statement", pmode, cycle, sig)
                if sdf.empty:
                    st.info("No statement data available for this credit union and period.")
                else:
                    st.dataframe(sdf, use_container_width=True, hide_index=True,
                                 height=38 * len(sdf) + 38)
                    note = ("Built from NCUA call report accounts and tied to the reported "
                            "totals. Lines marked “implied” (investments, provision for credit "
                            "losses, other) are derived so the statement foots exactly.")
                    if stmt == "Income Statement":
                        note += (" Income figures are year-to-date in the call report; the "
                                 "Quarters view de-cumulates them into standalone quarters.")
                    st.caption(note)

                st.divider()
                render_yield_spread(cu, cycle, sig, pmode)

                st.divider()
                vals = acct_values(cu, cycle, sig)
                mc1, mc2 = st.columns(2)
                with mc1:
                    st.markdown("**Loan Mix**")
                    lm = mix_frame(vals, LOAN_MIX, "ACCT_025B", "Other")
                    if lm.empty:
                        st.caption("No loan data for this credit union.")
                    else:
                        mix_dataframe(lm)
                        st.caption("Loan composition from Section 1 of the NCUA 5300 "
                                   "(first mortgage, other RE, vehicle, commercial, and "
                                   "consumer categories); foots to total loans & leases.")
                with mc2:
                    st.markdown("**Deposit Mix**")
                    dm = mix_frame(vals, DEPOSIT_MIX, "ACCT_018",
                                   "Other (incl. IRA / Keogh)")
                    if dm.empty:
                        st.caption("No deposit data for this credit union.")
                    else:
                        mix_dataframe(dm)
                        st.caption("Share composition from the NCUA call report; the residual "
                                   "captures IRA/Keogh and any other shares.")

                with st.expander("Raw Call Report Tables (Advanced)"):
                    table = st.selectbox("Table", [t for t in tables if t not in BROWSE_SKIP])
                    try:
                        raw = con.execute(
                            f"SELECT * FROM read_parquet('{glob_for(table)}', "
                            "hive_partitioning=true, union_by_name=true) "
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

            # ===================================================== TRENDS
            with tab_tr:
                if len(all_cycles) > 1:
                    tsd = cu_timeseries(cu, sig)
                    trend_opts = [k for k, _, _, _ in METRICS if not k.endswith("_growth")]
                    tc1, tc2 = st.columns([1, 3])
                    tspan = tc1.radio("Period", ["Quarters", "Years"], horizontal=True)
                    chosen = tc2.multiselect(
                        "Metrics to chart", trend_opts,
                        default=trend_opts,
                        format_func=lambda k: META[k][0])
                    peer_tr = st.checkbox(
                        "Add peer median (asset band)", value=False, key="prof_peer_tr",
                        help="Adds the asset-band median line to ratio measures "
                             "(not dollar or count measures).")
                    if tspan == "Years" and not tsd.empty:
                        tsd = tsd[[str(i).endswith("-12") for i in tsd.index]]
                    if not tsd.empty and chosen:
                        cyc_list = tuple(str(i) for i in tsd.index)
                        grid = st.columns(2)
                        for i, key in enumerate(chosen):
                            with grid[i % 2]:
                                st.caption(META[key][0])
                                dfc = tsd[[key]].rename(columns={key: META[key][0]})
                                if peer_tr and key in RATIO_PEER_KEYS:
                                    pm, _ = peer_median_line(key, row.band, cyc_list, sig)
                                    if pm:
                                        dfc["Peer median"] = [pm.get(str(c)) for c in tsd.index]
                                st.line_chart(dfc)
                else:
                    st.info("Only one quarter of data is available — trends need more history.")

            # ===================================================== PEERS
            with tab_peer:
                basis_choice = st.radio(
                    "Compare against",
                    ["Similar asset size", f"Same state ({row.state})",
                     "Same state + asset size", "Statistical twins (auto)",
                     "All credit unions", "Custom (pick CUs)"],
                    horizontal=True)
                if basis_choice == "Custom (pick CUs)":
                    picks = st.multiselect("Choose peer credit unions",
                                           list(ALL_LABELS),
                                           format_func=lambda n: ALL_LABELS[n])
                    peers = mt[mt.cu.isin(picks + [cu])] if picks else mt.iloc[0:0]
                elif basis_choice == "Similar asset size":
                    peers = mt[mt.band == row.band]
                elif basis_choice.startswith("Same state ("):
                    peers = mt[mt.state == row.state]
                elif "state + asset" in basis_choice:
                    peers = mt[(mt.state == row.state) & (mt.band == row.band)]
                elif basis_choice.startswith("Statistical twins"):
                    n_tw = st.slider("Number of twins", 5, 40, 20, key="ntwins")
                    tw = find_twins(cu, cycle, sig, n_tw)
                    peers = mt[mt.cu.isin(tw + [cu])] if tw else mt.iloc[0:0]
                    st.caption("Twins = most similar by asset size, loan-mix profile, and "
                               "region (same state preferred).")
                else:
                    peers = mt
                st.caption(f"Peer group: {len(peers):,} credit unions")
                if len(peers) >= 2:
                    ratio_keys = ["roa", "roe", "nim", "efficiency", "nw_ratio", "lts",
                                  "delinquency", "nco"] + GROWTH_KEYS
                    items = []
                    for key in ratio_keys:
                        lbl, _, dirn = META[key]
                        v = row[key]
                        s = peers[key].dropna()
                        if pd.isna(v) or len(s) < 2:
                            items.append({"label": lbl, "value": fmt(key, v),
                                          "median": "—", "stats": None})
                            continue
                        med = s.median()
                        better = (v >= med) if dirn != "low" else (v <= med)
                        lo, hi = float(s.quantile(.05)), float(s.quantile(.95))
                        if hi <= lo:                       # little spread -> use full range
                            lo, hi = float(s.min()), float(s.max())
                        items.append({"label": lbl, "value": fmt(key, v),
                                      "median": fmt(key, med),
                                      "stats": {"lo": lo, "hi": hi,
                                                "min": float(s.min()), "max": float(s.max()),
                                                "p25": float(s.quantile(.25)),
                                                "p75": float(s.quantile(.75)),
                                                "median": float(med), "value": float(v),
                                                "better": bool(better)}})
                    st.markdown(peer_bars_html(items), unsafe_allow_html=True)
                    st.caption("Green dot = better than the peer median, red = worse "
                               "(direction-adjusted, so lower is 'better' for efficiency, "
                               "delinquency, and charge-offs).")
                else:
                    st.info("Pick at least one peer to benchmark against.")
                with st.expander("Identity / FOICU Fields"):
                    foicu = con.execute(
                        f"SELECT * FROM read_parquet('{glob_for('FOICU')}', "
                        "hive_partitioning=true, union_by_name=true) "
                        "WHERE cycle = ? AND CU_NUMBER = ?", [cycle, cu]).df()
                    st.dataframe(foicu.T, use_container_width=True)

            # ===================================================== MERGERS
            # Merger footnote — quiet inline note at the bottom of Profile
            _mg = merger_table(sig)
            if not _mg.empty:
                _mine = _mg[_mg.continuing_charter == cu].sort_values("cycle", ascending=False)
                if not _mine.empty:
                    st.divider()
                    _shown = _mine.head(3)
                    _mlist = ", ".join(
                        f"{r.merging_name} ({r.cycle[:4]}, {money(r.merging_assets)})"
                        for _, r in _shown.iterrows())
                    _extra = (f" and {len(_mine) - 3} more" if len(_mine) > 3 else "")
                    st.caption(f"🔀 **Merger activity:** absorbed {len(_mine)} "
                               f"institution{'s' if len(_mine) > 1 else ''} since 2018 — "
                               f"{_mlist}{_extra}. "
                               "Source: NCUA Insurance Report of Activity.")

# ============================================================ FPR
elif page == "FPR":
    st.subheader("Financial Performance Report")
    cu_keys = list(ALL_LABELS)
    if not cu_keys:
        st.info("No credit unions available for this cycle.")
    else:
        default_cu = next((c for c in cu_keys if str(c) == "61790"), cu_keys[0])
        h1, h2, h3 = st.columns([2, 2, 1])
        cu_pick = h1.selectbox("Credit union", cu_keys,
                               index=cu_keys.index(default_cu),
                               format_func=lambda c: ALL_LABELS[c])
        view = h2.radio("Report", ["Financial Summary", "Key Ratios", "Historical Ratios"],
                        horizontal=True)
        pmode = h3.radio("Period", ["Quarters", "Years"], horizontal=True, key="fpr_span")
        info = mt[mt.cu == cu_pick]
        st_, band = (info.state.iloc[0] if not info.empty else ""), \
                    (info.band.iloc[0] if not info.empty else "")
        nm = ALL_LABELS[cu_pick].split(" (#")[0]
        st.caption(f"**{nm}** · Charter #{cu_pick} · {st_} · Peer group (asset band): "
                   f"{band} · As of {cycle}")
        raw = cu_statement_raw(cu_pick, sig)

        if view == "Financial Summary":
            cs_pool = [c for c in sorted(sig)
                       if c <= cycle and (pmode != "Years" or c.endswith("-12"))]
            html = fpr_summary_html(cu_pick, pmode, cycle, sig)
            if html is None or not cs_pool:
                st.info("No periods available for this selection.")
            else:
                last_lbl = _fpr_collabel(cs_pool[-1], pmode).replace("-", " ")
                title = f"{'Yearly' if pmode == 'Years' else 'Quarterly'}, Ending {last_lbl}"
                st.markdown(FPR_FS_CSS, unsafe_allow_html=True)
                st.markdown(f'<div class="fprtitle">{title}</div>', unsafe_allow_html=True)
                st.markdown(fpr_meta_html(cu_pick, cycle, sig), unsafe_allow_html=True)
                st.markdown(html, unsafe_allow_html=True)
                st.markdown(
                    '<div class="fprnote">'
                    '* Income / expense items are year-to-date; the related %Chg ratios are '
                    'annualized. &nbsp; %Chg is N/A when the prior period is zero.<br>'
                    '1 Prior to March 2022, Time and Other Deposits were included in '
                    'Investments. &nbsp; 2 Prior to 3/31/2022 includes “Subordinated Debt '
                    'Included in Net Worth” and “Non-Trading Derivative Liabilities”.<br>'
                    'All Other Assets, Total Liabilities, Other Reserves, and Total Equity are '
                    'derived to foot to the reported totals. The off-balance-sheet credit-loss '
                    'allowance is shown N/A (account not available in this dataset).'
                    '</div>', unsafe_allow_html=True)
        else:
            cs_pool = [c for c in sorted(sig)
                       if c <= cycle and (pmode != "Years" or c.endswith("-12"))]
            sections = FPR_KEY_SECTIONS if view == "Key Ratios" else FPR_HIST_SECTIONS
            n = 5 if view == "Key Ratios" else (10 if pmode == "Years" else 13)
            html, n_peer = fpr_ratio_html(cu_pick, sections, pmode, cycle, sig, band, n,
                                          peer=True)
            if html is None or not cs_pool:
                st.info("No periods available for this selection.")
            else:
                last_lbl = _fpr_collabel(cs_pool[-1], pmode).replace("-", " ")
                title = f"{'Yearly' if pmode == 'Years' else 'Quarterly'}, Ending {last_lbl}"
                st.markdown(FPR_FS_CSS, unsafe_allow_html=True)
                st.markdown(f'<div class="fprtitle">{title}</div>', unsafe_allow_html=True)
                st.markdown(fpr_meta_html(cu_pick, cycle, sig), unsafe_allow_html=True)
                st.markdown(html, unsafe_allow_html=True)
                peer_note = (f"Peer Avg. = mean across {n_peer} credit unions in the "
                             f"{band} asset band as of {last_lbl}.")
                if view == "Key Ratios":
                    note = (
                        '<div class="fprnote">'
                        '1 Exam-date ratios are annualized. &nbsp; 2 Net charge-offs over the '
                        'trailing 12 months (approximated here as the annualized year-to-date '
                        'figure). &nbsp; 3 Relies on the maturity distribution of investments. '
                        '&nbsp; 4 Applies to credit unions under $500M. &nbsp; 6 Net-worth ratio '
                        'per NCUA Part 702 (Account 998).<br>' + peer_note + '<br>'
                        'Ratios needing account codes not verified in this dataset — ALLL/ACL-'
                        'adjusted capital, GAAP equity, risk-based capital, short-term '
                        'investments, other non-performing assets, and the NEV tool — are shown '
                        'N/A.</div>')
                else:
                    note = ('<div class="fprnote">'
                            '* Annualization factor: March = 4; June = 2; September = 4/3; '
                            'December = 1.<br>' + peer_note + '<br>'
                            'Ratios needing account codes not verified in this dataset — CECL '
                            'adoption flags, PCA-optional and CECL-adjusted net worth, solvency '
                            'and classified-assets estimates, HTM/AFS fair-value ratios, '
                            'long-term-asset and net-operating-expense ratios, and the FTE / '
                            'borrower / potential-member productivity lines — are shown N/A. '
                            'Computations use the NCUA FPR average-balance basis '
                            '(current period + prior year-end ÷ 2).</div>')
                st.markdown(note, unsafe_allow_html=True)

# ============================================================ ROA BRIDGE
elif page == "ROA Bridge":
    st.subheader("ROA Bridge — What Drives the Gap to Peers")
    cu_keys = list(ALL_LABELS)
    if not cu_keys:
        st.info("No credit unions available for this cycle.")
    else:
        default_cu = next((c for c in cu_keys if str(c) == "61790"), cu_keys[0])
        cu_pick = st.selectbox("Credit union", cu_keys,
                               index=cu_keys.index(default_cu),
                               format_func=lambda c: ALL_LABELS[c])
        info = mt[mt.cu == cu_pick]
        band = info.band.iloc[0] if not info.empty else "Unknown"
        nm = ALL_LABELS[cu_pick].split(" (#")[0]
        st.caption(f"**{nm}** · Peer group (asset band): {band} · As of {cycle}. "
                   "Each driver is a share of average assets; the bars walk from the "
                   "peer-average ROAA up or down to this credit union's ROAA.")
        try:
            subj = _fpr_ratio_dict(cu_statement_raw(cu_pick, sig), cycle, sig)
            peer, n_peer = fpr_peer_avg(band, cycle, sig)
            bridge = roa_bridge_steps(subj, peer)
            if bridge is None or n_peer == 0:
                st.info("ROAA isn't available for this credit union or its peer "
                        "group in this cycle.")
            else:
                import plotly.graph_objects as go
                start_lbl, start_val = bridge["start"]
                end_lbl, end_val = bridge["end"]
                xs = [start_lbl] + [s[0] for s in bridge["steps"]] + [end_lbl]
                ys = [start_val] + [s[1] for s in bridge["steps"]] + [end_val]
                measures = ["absolute"] + ["relative"] * len(bridge["steps"]) + ["total"]
                fig = go.Figure(go.Waterfall(
                    orientation="v", measure=measures, x=xs, y=ys,
                    text=[(f"{v:+.2f}" if m == "relative" else f"{v:.2f}")
                          for v, m in zip(ys, measures)],
                    textposition="outside",
                    connector={"line": {"color": "rgb(160,160,160)"}},
                    increasing={"marker": {"color": "#2e8b57"}},
                    decreasing={"marker": {"color": "#c0392b"}},
                    totals={"marker": {"color": "#34495e"}}))
                fig.update_layout(height=460, showlegend=False,
                                  margin=dict(t=30, b=90, l=50, r=20),
                                  yaxis_title="% of average assets")
                st.plotly_chart(fig, use_container_width=True)
                gap = end_val - start_val
                st.caption(f"ROAA gap vs {n_peer} {band} peers: {gap:+.2f} pp "
                           f"({end_val:.2f}% vs {start_val:.2f}% peer average). "
                           "Green drivers add to ROAA, red subtract.")
                tbl = pd.DataFrame(
                    [(lbl, f"{val:+.2f} pp") for lbl, val in bridge["steps"]],
                    columns=["Driver (this CU vs peer avg)", "Contribution to ROAA gap"])
                st.dataframe(tbl, use_container_width=True, hide_index=True)
                st.caption("Peer comparison uses the asset-band average, matching the "
                           "FPR page's Peer Avg. column.")
        except Exception as exc:
            st.info(f"ROA bridge unavailable ({exc}).")

# ============================================================ COMPARE
elif page == "Compare":
    st.subheader("Compare Credit Unions Side by Side")
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
            st.subheader("Trend Overlays")
            oc1, oc2 = st.columns([1, 3])
            ov_span = oc1.radio("Period", ["Quarters", "Years"], horizontal=True,
                                key="cmp_span")
            ov_opts = [k for k, _, _, _ in METRICS if not k.endswith("_growth")]
            ov_keys = oc2.multiselect(
                "Metrics to chart", ov_opts, default=ov_opts,
                format_func=lambda k: META[k][0])
            peer_ov = st.checkbox(
                "Add peer median (asset band)", value=False, key="cmp_peer_ov",
                help="Adds the asset-band median line to ratio measures (not dollar or "
                     "count measures). Band is taken from the first selected credit union.")
            ov_band = cu_band(picks[0], cycle) if (peer_ov and picks) else None
            grid = st.columns(2)
            for i, key in enumerate(ov_keys):
                series = multi_cu_series(picks, key, ALL_LABELS, sig)
                if ov_span == "Years" and not series.empty:
                    series = series[[str(ix).endswith("-12") for ix in series.index]]
                if not series.empty:
                    with grid[i % 2]:
                        cap = META[key][0]
                        if ov_band and key in RATIO_PEER_KEYS:
                            pm, npeer = peer_median_line(
                                key, ov_band, tuple(str(ix) for ix in series.index), sig)
                            if pm:
                                series = series.copy()
                                series[f"Peer median ({ov_band})"] = [
                                    pm.get(str(ix)) for ix in series.index]
                                cap = f"{cap} · peer median n={npeer}"
                        st.caption(cap)
                        st.line_chart(series)

# ============================================================ CHART
elif page == "Chart":
    import plotly.graph_objects as go
    st.subheader("Chart Builder")
    PALETTES = {
        "Datawrapper default": ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2"],
        "Colorblind-safe":     ["#0072b2", "#e69f00", "#009e73", "#d55e00", "#cc79a7", "#56b4e9"],
        "Vivid":               ["#4f46e5", "#ef4444", "#10b981", "#f59e0b", "#ec4899", "#06b6d4"],
        "Muted":               ["#5b7aa6", "#b5685f", "#6a9a78", "#c39a52", "#8a76a8", "#5f97a3"],
        "Blue ramp":           ["#08306b", "#2171b5", "#4292c6", "#6baed6", "#9ecae1", "#74a9cf"],
        "Grayscale":           ["#1f1f1f", "#555555", "#777777", "#999999", "#b5b5b5", "#cfcfcf"],
    }
    MUTED_COLOR = "#c9ced6"          # greyed-out lines when emphasizing a subset

    CHART_TYPES = [
        ("Compare credit unions on one measure", "📈", "Lines · Compare"),
        ("One credit union across measures", "📉", "Lines · One CU"),
        ("Bars & Columns", "📊", "Bars & Columns"),
        ("Composition (mix)", "🍩", "Composition"),
    ]
    _valid_modes = [v for v, _, _ in CHART_TYPES]
    if st.session_state.get("chart_mode") not in _valid_modes:
        st.session_state.chart_mode = CHART_TYPES[0][0]

    # Handle tile click via query param (?ct=index)
    _ct_idx = st.query_params.get("ct")
    if _ct_idx is not None:
        try:
            _ct_i = int(_ct_idx)
            if 0 <= _ct_i < len(CHART_TYPES):
                st.session_state.chart_mode = CHART_TYPES[_ct_i][0]
        except (ValueError, TypeError):
            pass
        st.query_params.clear()

    mode = st.session_state.chart_mode

    # Chart type tile grid
    _tiles = [
        '<div style="display:grid;grid-template-columns:repeat(auto-fill,'
        'minmax(130px,1fr));gap:10px;margin:0 0 20px">']
    for _i, (_val, _icon, _lbl) in enumerate(CHART_TYPES):
        _a   = (mode == _val)
        _bg  = "#e6ebf5" if _a else "#ffffff"
        _bdr = "1.5px solid #b8c4da" if _a else "1px solid #e2e5ea"
        _fw  = "600" if _a else "400"
        _tiles.append(
            f'<a href="?ct={_i}" style="text-decoration:none">'
            f'<div style="background:{_bg};border:{_bdr};border-radius:10px;'
            f'padding:16px 8px 12px;text-align:center;cursor:pointer;'
            f'display:flex;flex-direction:column;align-items:center;'
            f'justify-content:center;min-height:86px">'
            f'<div style="font-size:1.9rem;line-height:1;margin-bottom:9px">{_icon}</div>'
            f'<div style="font-size:0.79rem;font-weight:{_fw};color:#1a2a4a;'
            f'line-height:1.35">{_lbl}</div>'
            f'</div></a>')
    _tiles.append('</div>')
    st.markdown(''.join(_tiles), unsafe_allow_html=True)

    main = st.container()   # full-width (was a narrow rail column)
    with main:
        chart_slot = st.container()                   # live preview on top
        tab_data, tab_refine, tab_annot, tab_layout = st.tabs(
            ["Data", "Refine", "Annotate", "Layout"])

    with tab_data:
        tf1, tf2 = st.columns([1, 3])
        span = tf1.radio("Period", ["Quarters", "Years"], horizontal=True, key="chart_span")
        cyc_opts = [c for c in sorted(all_cycles) if span == "Quarters" or c.endswith("-12")]
        if len(cyc_opts) >= 2:
            lo, hi = tf2.select_slider("Range", options=cyc_opts,
                                       value=(cyc_opts[0], cyc_opts[-1]),
                                       format_func=lambda c: _period_label(c, span))
            in_range = [c for c in cyc_opts if lo <= c <= hi]
        else:
            in_range = cyc_opts
        value_mode = st.radio(
            "Values", ["Levels", "% change from start", "Period-over-period %"],
            horizontal=True, key="chart_valuemode",
            help="Plot raw levels, cumulative % change from the first period in range, or "
                 "period-over-period % change. Applies to the line views.")
    idx = [_period_label(c, span) for c in in_range]

    with tab_refine:
        cc1, cc2, cc3 = st.columns(3)
        markers = cc1.checkbox("Markers", value=True)
        label_mode = cc2.selectbox("Data labels",
                                   ["None", "All points", "First & last", "Last only"])
        label_pos = cc3.selectbox("Label position", ["Top", "Bottom", "Right", "Left"])
        ce1, ce2, ce3 = st.columns(3)
        smooth = ce1.checkbox("Smooth lines", value=False)
        line_width = ce2.slider("Line width", 1, 6, 2)
        area_fill = ce3.checkbox("Fill area under lines", value=False,
                                 help="Shade a translucent area beneath each line "
                                      "(Compare and small-multiples views).")
        palette_name = st.selectbox("Color palette", list(PALETTES),
                                     help="Sets the default colours for every line. "
                                          "Per-series pickers below still override.")
        PALETTE = PALETTES[palette_name]
        accent = st.color_picker("Single-series color", value=PALETTE[0],
                                 key=f"accent_{palette_name}")
        colors_box = st.container()                   # per-series pickers fill this
        range_box = st.container()                    # manual-range controls fill this

    with tab_annot:
        subtitle = st.text_input("Subtitle", value="")
        source = st.text_input("Source / footnote", value="Source: NCUA 5300 Call Reports")
        cg1, cg2 = st.columns(2)
        ev_choice = cg1.selectbox("Mark a period", ["(none)"] + idx)
        ev_label = cg2.text_input("Marker label", value="")
        ev_x = None if ev_choice == "(none)" else ev_choice
        cb1, cb2, cb3 = st.columns([2, 2, 3])
        band_a = cb1.selectbox("Shade from", ["(none)"] + idx, key="band_from",
                               help="Highlight a span of periods with a shaded band "
                                    "(e.g. a rate-hike cycle).")
        band_b = cb2.selectbox("Shade to", ["(none)"] + idx, key="band_to")
        band_label = cb3.text_input("Band label", value="", key="band_lbl")

    with tab_layout:
        la1, la2, la3 = st.columns(3)
        height = la1.slider("Height", 320, 760, 460, step=20)
        title_align = la2.selectbox("Title align", ["Left", "Center", "Right"])
        legend_on = la3.checkbox("Legend", value=True)
        lb1, lb2 = st.columns(2)
        gridlines = lb1.checkbox("Y gridlines", value=True)
        direct_labels = lb2.checkbox("Label lines at end", value=False,
                                     help="Datawrapper-style: name each line at its right "
                                          "end and hide the legend.")
        lc1, lc2 = st.columns(2)
        export_fmt = lc1.selectbox("Download format", ["PNG", "SVG"],
                                   help="Sets the camera-icon download button on the chart "
                                        "toolbar. SVG is vector — best for decks and print.")
        export_scale = lc2.select_slider("PNG resolution", options=[1, 2, 3], value=2,
                                         disabled=export_fmt != "PNG",
                                         help="1×/2×/3× pixel density for the PNG download.")

    labels_on = label_mode != "None"
    line_mode = ("lines+markers" if markers else "lines") + ("+text" if labels_on else "")
    LSHAPE = "spline" if smooth else "linear"
    POS = {"Top": "top center", "Bottom": "bottom center",
           "Right": "middle right", "Left": "middle left"}[label_pos]
    TITLE_X = {"Left": (0.0, "left"), "Center": (0.5, "center"),
               "Right": (1.0, "right")}[title_align]
    metric_keys = [k for k, _, _ in CHART_METRICS]
    default_cu = next((c for c in ALL_LABELS if str(c) == "61790"), next(iter(ALL_LABELS), None))
    x_is_category = True   # False for horizontal bars (value axis on x) -> skip period markers

    def axis_kw(kind):
        if kind == "pct":
            return dict(ticksuffix="%")
        if kind == "money":
            return dict(tickprefix="$", tickformat="~s")
        return dict(tickformat=",")

    CHANGED = value_mode != "Levels"

    def chg(series):
        """Apply the chosen % view to a level series; pass through when on Levels."""
        s = pd.to_numeric(series, errors="coerce")
        if value_mode == "% change from start":
            nz = s.dropna()
            base = nz.iloc[0] if len(nz) else None
            return (s / base - 1) * 100 if base not in (None, 0) else s * np.nan
        if value_mode == "Period-over-period %":
            return s.pct_change() * 100
        return s

    def vkind(metric_kind):
        return "pct" if CHANGED else metric_kind

    def vlabel(label):
        if value_mode == "% change from start":
            return f"{label} — % change"
        if value_mode == "Period-over-period %":
            return f"{label} — QoQ %"
        return label

    def data_download(df, fname, key):
        if df is None or getattr(df, "empty", True):
            return
        chart_slot.download_button("⬇ Download chart data (CSV)",
                                   df.to_csv().encode("utf-8"), file_name=fname,
                                   mime="text/csv", key="dl_" + key)

    def rgba(hexc, a):
        h = hexc.lstrip("#")
        r, g, bl = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{bl},{a})"

    def fill_kw(col):
        return dict(fill="tozeroy", fillcolor=rgba(col, 0.12)) if area_fill else {}

    def lab(kind, v):
        if pd.isna(v):
            return ""
        if kind == "money":
            return money_compact(v)
        if kind == "int":
            return f"{v:,.0f}"
        return f"{v:.2f}%"

    def txt(series, kind):
        if not labels_on:
            return None
        vals = series.values
        full = [lab(kind, v) for v in vals]
        valid = [i for i, v in enumerate(vals) if pd.notna(v)]
        if not valid or label_mode == "All points":
            return full
        keep = {valid[0], valid[-1]} if label_mode == "First & last" else {valid[-1]}
        return [full[i] if i in keep else "" for i in range(len(full))]

    def minmax(series):
        v = pd.to_numeric(series, errors="coerce").dropna()
        return (float(v.min()), float(v.max())) if len(v) else (0.0, 1.0)

    def manual_range(box, label, dflt, key):
        if not box.checkbox(f"Set {label} range manually", key=f"{key}_on"):
            return None
        c1, c2 = box.columns(2)
        a = c1.number_input(f"{label} min", value=dflt[0], key=f"{key}_lo", format="%g")
        b = c2.number_input(f"{label} max", value=dflt[1], key=f"{key}_hi", format="%g")
        return [a, b] if b > a else None

    def series_colors(box, names, key):
        cols = box.columns(min(len(names), 6) or 1)
        return {nm: cols[i % len(cols)].color_picker(nm, value=PALETTE[i % len(PALETTE)],
                                                     key=f"{key}_{i}")
                for i, nm in enumerate(names)}

    def _last_valid(tr):
        ys = tr.y
        if ys is None:
            return None
        for j in range(len(ys) - 1, -1, -1):
            v = ys[j]
            if v is not None and not (isinstance(v, float) and v != v):
                return tr.x[j], v
        return None

    def styled(fig, title):
        b = 30 + (46 if (legend_on and not direct_labels) else 0) + (26 if source else 0)
        fig.update_layout(
            title=dict(text=title, x=TITLE_X[0], xanchor=TITLE_X[1], font=dict(size=18),
                       subtitle=dict(text=subtitle) if subtitle else None),
            height=height, hovermode="x unified",
            showlegend=legend_on and not direct_labels,
            margin=dict(l=10, r=(110 if direct_labels else 12), t=64 if subtitle else 48, b=b),
            legend=dict(orientation="h", yanchor="top", y=-0.12, x=0))
        fig.update_yaxes(showgrid=gridlines)
        fig.update_xaxes(showgrid=False)
        fig.update_traces(textfont_size=10)
        if source:
            sy = -0.26 if (legend_on and not direct_labels) else -0.12
            fig.add_annotation(text=source, xref="paper", yref="paper", x=0, y=sy,
                               showarrow=False, xanchor="left", yanchor="top",
                               font=dict(size=11, color="#888"))
        if ev_x and x_is_category:
            fig.add_shape(type="line", xref="x", x0=ev_x, x1=ev_x, yref="paper", y0=0, y1=1,
                          line=dict(color="#999", width=1, dash="dash"))
            if ev_label:
                fig.add_annotation(x=ev_x, xref="x", yref="paper", y=1.0, yanchor="bottom",
                                   text=ev_label, showarrow=False, font=dict(size=11, color="#666"))
        if x_is_category and band_a != "(none)" and band_b != "(none)":
            ia, ib = idx.index(band_a), idx.index(band_b)
            if ib < ia:
                ia, ib = ib, ia
            fig.add_vrect(x0=ia - 0.5, x1=ib + 0.5, fillcolor="#9aa7b8", opacity=0.14,
                          line_width=0, layer="below")
            if band_label:
                fig.add_annotation(x=(ia + ib) / 2, xref="x", yref="paper", y=1.0,
                                   yanchor="bottom", text=band_label, showarrow=False,
                                   font=dict(size=11, color="#666"))
        if direct_labels:
            for tr in fig.data:
                if tr.type != "scatter":          # end-labels are a line feature
                    continue
                lv = _last_valid(tr)
                if not lv:
                    continue
                x_, y_ = lv
                col = tr.line.color if (tr.line and tr.line.color) else "#333"
                fig.add_annotation(x=x_, y=y_, xref="x", yref=(tr.yaxis or "y"), text=tr.name,
                                   showarrow=False, xanchor="left", xshift=8,
                                   font=dict(size=11, color=col))
        return fig

    def export_config(title):
        fn = re.sub(r"[^A-Za-z0-9._-]+", "_", title or "chart").strip("_") or "chart"
        opts = {"format": export_fmt.lower(), "filename": fn}
        if export_fmt == "PNG":
            opts["scale"] = export_scale
        return dict(displaylogo=False, toImageButtonOptions=opts, responsive=True)

    if len(in_range) < 2:
        chart_slot.info("Pick a range spanning at least two periods to plot a trend.")

    elif mode == "Bars & Columns":
        with tab_data:
            bo1, bo2, bo3 = st.columns(3)
            orient = bo1.radio("Orientation", ["Columns", "Bars"], horizontal=True,
                               help="Columns = vertical, Bars = horizontal.")
            bar_basis = bo2.radio("Put on the axis", ["Periods", "Credit unions"],
                                  horizontal=True,
                                  help="Periods = one measure over time (e.g. 2024 vs 2025 "
                                       "vs 2026). Credit unions = rank the peer universe at "
                                       "the quarter selected in the sidebar.")
            layout = bo3.radio("Bar layout", ["Grouped", "Stacked"], horizontal=True,
                               help="Stacked suits series that sum to a meaningful total; "
                                    "grouped is the usual peer comparison.")
        with tab_refine:
            cz1, cz2 = st.columns(2)
            vlab = cz1.selectbox("Value labels", ["None", "Outside", "Inside"], key="bar_vlab")
            bargap = cz2.slider("Space between bars (%)", 0, 80, 30, key="bar_gap") / 100.0
        horiz = orient == "Bars"
        x_is_category = not horiz
        barmode = "stack" if layout == "Stacked" else "group"
        tpos = None if vlab == "None" else ("inside" if vlab == "Inside" else "outside")

        def bar_text(series, kind):
            return None if vlab == "None" else [lab(kind, v) for v in series.values]

        def add_bar(fig, cats, vals, name=None, color=None, text=None):
            kw = dict(name=name, marker_color=color, text=text, textposition=tpos)
            fig.add_trace(go.Bar(y=cats, x=vals, orientation="h", **kw) if horiz
                          else go.Bar(x=cats, y=vals, **kw))

        if bar_basis == "Periods":
            with tab_data:
                metric = st.selectbox("Measure", metric_keys,
                                      format_func=lambda k: CHART_LABEL[k],
                                      index=metric_keys.index("assets"), key="bar_metric")
                picks = st.multiselect("Credit unions (up to 6)", list(ALL_LABELS),
                                       default=[default_cu] if default_cu is not None else [],
                                       max_selections=6, format_func=lambda c: ALL_LABELS[c],
                                       key="bar_cus")
            kind = chart_kind(metric)
            series = []
            for c in picks:
                s = chart_series(c, sig)
                if s.empty or metric not in s:
                    continue
                y = pd.to_numeric(s[metric], errors="coerce").reindex(in_range)
                series.append((ALL_LABELS[c].split(" (#")[0], y))
            if not series:
                chart_slot.info("Choose at least one credit union with data for this measure.")
            else:
                names = [nm for nm, _ in series]
                who = (", ".join(names) if len(names) <= 3 else f"{len(names)} credit unions")
                with tab_data:
                    title = st.text_input("Chart title",
                                          value=f"{CHART_LABEL[metric]} — {who}",
                                          key="bt_" + metric + "_" + "_".join(map(str, picks)))
                colors = series_colors(colors_box, names, "barcol_" + palette_name)
                fig = go.Figure()
                for nm, y in series:
                    add_bar(fig, idx, y.values, name=nm, color=colors[nm],
                            text=bar_text(y, kind))
                fig.update_layout(barmode=barmode, bargap=bargap)
                vkw = dict(title=CHART_LABEL[metric], **axis_kw(kind))
                (fig.update_xaxes if horiz else fig.update_yaxes)(**vkw)
                styled(fig, title)
                if horiz:                       # keep gridlines on the value (x) axis
                    fig.update_xaxes(showgrid=gridlines)
                    fig.update_yaxes(showgrid=False)
                chart_slot.plotly_chart(fig, use_container_width=True,
                                        config=export_config(title))
                data_download(pd.DataFrame({nm: y.values for nm, y in series}, index=idx),
                              f"{metric}_bars.csv", "bars_" + metric + "_".join(map(str, picks)))
                chart_slot.caption(f"{CHART_LABEL[metric]} ({span.lower()}), "
                                   f"{idx[0]}–{idx[-1]}. {layout.lower().capitalize()} "
                                   f"{'bars' if horiz else 'columns'}.")
        else:                                   # rank the peer universe at one cycle
            rank_metrics = [k for k in metric_keys if k in getattr(mt, "columns", [])]
            with tab_data:
                metric = st.selectbox("Measure", rank_metrics,
                                      format_func=lambda k: CHART_LABEL[k],
                                      index=(rank_metrics.index("assets")
                                             if "assets" in rank_metrics else 0),
                                      key="bar_rank_metric")
                rc1, rc2 = st.columns(2)
                topn = rc1.slider("Show top", 5, 40, 15, key="bar_topn")
                lowest = rc2.radio("Order", ["Highest first", "Lowest first"],
                                   horizontal=True, key="bar_order") == "Lowest first"
            kind = chart_kind(metric)
            d = mt[["cu", "cu_name", metric]].copy()
            d[metric] = pd.to_numeric(d[metric], errors="coerce")
            d = d.dropna(subset=[metric]).sort_values(metric, ascending=lowest).head(topn)
            if d.empty:
                chart_slot.info("No data for this measure at the selected quarter.")
            else:
                has_blu = (d.cu.astype(str) == "61790").any()
                hi_col, base_col = PALETTE[0], ("#aab2bd" if has_blu else PALETTE[0])
                cols = [hi_col if str(c) == "61790" else base_col for c in d.cu]
                cats, vals = d.cu_name.tolist(), d[metric].tolist()
                with tab_data:
                    title = st.text_input(
                        "Chart title",
                        value=f"{CHART_LABEL[metric]} — top {len(d)} "
                              f"({_period_label(cycle, 'Quarters')})",
                        key=f"btr_{metric}_{cycle}")
                txt_ = None if vlab == "None" else [lab(kind, v) for v in vals]
                fig = go.Figure()
                add_bar(fig, cats, vals, color=cols, text=txt_)
                fig.update_layout(bargap=bargap, showlegend=False)
                vkw = dict(title=CHART_LABEL[metric], **axis_kw(kind))
                if horiz:
                    fig.update_yaxes(autorange="reversed")     # rank 1 on top
                    fig.update_xaxes(**vkw)
                else:
                    fig.update_yaxes(**vkw)
                styled(fig, title)
                if horiz:
                    fig.update_xaxes(showgrid=gridlines)
                    fig.update_yaxes(showgrid=False, autorange="reversed")
                chart_slot.plotly_chart(fig, use_container_width=True,
                                        config=export_config(title))
                chart_slot.caption(f"{CHART_LABEL[metric]} across the peer universe at "
                                   f"{_period_label(cycle, 'Quarters')}; BluCurrent highlighted "
                                   "in the accent colour.")

    elif mode.startswith("Compare"):
        with tab_data:
            metric = st.selectbox("Measure", metric_keys, format_func=lambda k: CHART_LABEL[k],
                                  index=metric_keys.index("nim"))
            picks = st.multiselect("Credit unions (up to 6)", list(ALL_LABELS),
                                   default=[default_cu] if default_cu is not None else [],
                                   max_selections=6, format_func=lambda c: ALL_LABELS[c])
            show_peer = st.checkbox("Add peer median line", value=False,
                                    help="Dashed benchmark = median of this measure across the "
                                         "first credit union's asset band, computed per period.")
        if not picks:
            chart_slot.info("Choose at least one credit union.")
        else:
            series, allvals = [], []
            for c in picks:
                s = chart_series(c, sig)
                if s.empty or metric not in s:
                    continue
                y = chg(pd.to_numeric(s[metric], errors="coerce").reindex(in_range))
                allvals += list(y.dropna().values)
                series.append((ALL_LABELS[c].split(" (#")[0], y))
            if not allvals:
                chart_slot.info("No data for this measure over the selected range.")
            else:
                names = [nm for nm, _ in series]
                who = (names[0] if len(names) == 1
                       else ", ".join(names) if len(names) <= 3
                       else f"{len(names)} credit unions")
                with tab_data:
                    title = st.text_input("Chart title", value=f"{vlabel(CHART_LABEL[metric])} — {who}",
                                          key="ct_cmp_" + metric + "_" + "_".join(map(str, picks)))
                    emph = st.multiselect("Emphasize lines (others muted)", names, default=[],
                                          key="emph_" + "_".join(map(str, picks)),
                                          help="Datawrapper-style highlight: keep the chosen "
                                               "lines in colour and grey out the rest.")
                colors = series_colors(colors_box, names, "cmpcol_" + palette_name)
                pmed, npeer, pband, pvals = {}, 0, None, None
                if show_peer and in_range:
                    pband = cu_band(picks[0], in_range[-1])
                    if pband:
                        pmed, npeer = peer_median_line(metric, pband, tuple(in_range), sig)
                        if pmed:
                            pvals = chg(pd.Series([pmed.get(c) for c in in_range]))
                            allvals += [v for v in pvals.tolist() if v == v]
                yr = manual_range(range_box, "y-axis", (min(allvals), max(allvals)), "cmp_y")
                kind = vkind(chart_kind(metric))
                ordered = sorted(series, key=lambda t: bool(emph) and t[0] in emph)
                fig = go.Figure()
                for nm, y in ordered:
                    on = (not emph) or (nm in emph)
                    col = colors[nm] if on else MUTED_COLOR
                    wid = line_width if on else max(1, line_width - 1)
                    fig.add_trace(go.Scatter(
                        x=idx, y=y.values, mode=line_mode, name=nm, connectgaps=True,
                        text=txt(y, kind) if on else None, textposition=POS,
                        marker=dict(color=col),
                        line=dict(color=col, width=wid, shape=LSHAPE),
                        **(fill_kw(col) if on else {})))
                if pvals is not None:
                    fig.add_trace(go.Scatter(
                        x=idx, y=pvals.tolist(), mode="lines",
                        name=f"Peer median · {pband} (n={npeer})", connectgaps=True,
                        line=dict(color="#6b7280", width=max(2, line_width), dash="dash"),
                        hovertemplate="Peer median: %{y}<extra></extra>"))
                fig.update_yaxes(title=vlabel(CHART_LABEL[metric]), **axis_kw(kind))
                if yr:
                    fig.update_yaxes(range=yr)
                styled(fig, title)
                chart_slot.plotly_chart(fig, use_container_width=True,
                                        config=export_config(title))
                dl = pd.DataFrame({nm: y.values for nm, y in series}, index=idx)
                if pvals is not None:
                    dl[f"Peer median ({pband})"] = pvals.tolist()
                data_download(dl, f"{metric}_{value_mode}.csv".replace(" ", "_"),
                              "cmp_" + metric + "_".join(map(str, picks)))
                chart_slot.caption(f"{vlabel(CHART_LABEL[metric])} ({span.lower()}), "
                                   f"{idx[0]}–{idx[-1]}. Drag on the plot to zoom; double-click "
                                   "to autoscale. Yields use the NCUA FPR average-balance basis.")

    elif mode == "Composition (mix)":
        cat_pal = PALETTE + ["#64748b", "#a855f7", "#14b8a6", "#f97316", "#84cc16", "#e11d48"]
        with tab_data:
            src = st.radio("Composition", list(MIX_SOURCES), horizontal=True)
            cu_keys = list(ALL_LABELS)
            comp_cu = st.selectbox("Credit union", cu_keys,
                                   index=cu_keys.index(default_cu) if default_cu in cu_keys else 0,
                                   format_func=lambda c: ALL_LABELS[c], key="comp_cu")
            disp = st.radio("Display", ["Snapshot (Pie / Donut)", "Over Time (Stacked)"],
                            horizontal=True)
        parts, total_code, resid = MIX_SOURCES[src]
        cu_name = ALL_LABELS[comp_cu].split(" (#")[0]

        if disp.startswith("Snapshot"):
            with tab_data:
                _cp, _cs = st.columns(2)
                period = _cp.selectbox("Period", in_range, index=len(in_range) - 1,
                                       format_func=lambda c: _period_label(c, span),
                                       key="comp_period")
                shape = _cs.radio("Chart type",
                                  ["Donut", "Pie", "Horizontal Bar"],
                                  horizontal=True, key="comp_shape")
                _cl, _cpr, _clg = st.columns(3)
                label_mode = _cl.radio("Labels",
                                        ["Name + %", "% only", "Name only", "None"],
                                        horizontal=True, key="comp_labels")
                precision  = _cpr.radio("Precision", ["0 dec", "1 dec", "2 dec"],
                                         horizontal=True, index=1, key="comp_prec")
                legend_pos = _clg.radio("Legend", ["Right", "Bottom", "Off"],
                                         horizontal=True, key="comp_legend_pos")
                other_thresh = st.slider("Combine slices below",
                                         0.0, 10.0, 0.0, 0.5,
                                         format="%.1f%%", key="comp_other",
                                         help="Fold all slices smaller than this "
                                              "share into the residual category.")
            v = acct_values(comp_cu, period, sig)
            mf = mix_frame(v, parts, total_code, resid)
            if mf.empty:
                chart_slot.info(f"No {src.lower()} reported for {cu_name} at "
                                f"{_period_label(period, span)}.")
            else:
                with tab_data:
                    title = st.text_input("Chart title",
                                          value=f"{cu_name} — {src.lower()} "
                                                f"({_period_label(period, span)})",
                                          key=f"comp_t_{src}_{comp_cu}_{period}")
                # --- Combine small slices into residual ---
                if other_thresh > 0:
                    _mask = (mf["Share"] < other_thresh) & (mf["Category"] != resid)
                    if _mask.sum() >= 1 and not _mask.all():
                        _small = mf[_mask]
                        _kept  = mf[~_mask].copy()
                        _e_amt = _small["Amount ($M)"].sum()
                        _e_shr = _small["Share"].sum()
                        _rm    = _kept["Category"] == resid
                        if _rm.any():
                            _kept.loc[_rm, "Amount ($M)"] += _e_amt
                            _kept.loc[_rm, "Share"] += _e_shr
                        else:
                            _kept = pd.concat([_kept, pd.DataFrame([{
                                "Category": resid,
                                "Amount ($M)": _e_amt,
                                "Share": _e_shr}])], ignore_index=True)
                        mf = _kept
                # --- Precision & legend helpers ---
                _dec_n   = {"0 dec": 0, "1 dec": 1, "2 dec": 2}[precision]
                _dec_sp  = f":.{_dec_n}%"
                if legend_pos == "Right":
                    _leg_kw = dict(showlegend=True,
                                   legend=dict(orientation="v", x=1.02, y=0.5))
                elif legend_pos == "Bottom":
                    _leg_kw = dict(showlegend=True,
                                   legend=dict(orientation="h", x=0.5,
                                               xanchor="center", y=-0.12))
                else:
                    _leg_kw = dict(showlegend=False)
                colors = [cat_pal[i % len(cat_pal)] for i in range(len(mf))]
                if shape == "Horizontal Bar":
                    _mfs = mf.sort_values("Share", ascending=True).reset_index(drop=True)
                    _bar_clr = [cat_pal[i % len(cat_pal)] for i in range(len(_mfs))]
                    _bar_txt = ([f"{v:.{_dec_n}f}%" for v in _mfs["Share"]]
                                if label_mode != "None" else None)
                    fig = go.Figure(go.Bar(
                        y=_mfs["Category"].tolist(),
                        x=_mfs["Share"].tolist(),
                        orientation="h",
                        marker_color=_bar_clr,
                        text=_bar_txt,
                        textposition="outside",
                        hovertemplate="%{y}: %{x:." + str(_dec_n) + "f}%<extra></extra>"))
                    fig.update_layout(
                        xaxis=dict(title="Share of total", ticksuffix="%",
                                   range=[0, _mfs["Share"].max() * 1.2]),
                        yaxis=dict(automargin=True),
                        **_leg_kw)
                else:
                    # Label template for pie / donut
                    if label_mode == "Name + %":
                        _tmpl = f"%{{label}}<br>%{{percent{_dec_sp}}}"
                    elif label_mode == "% only":
                        _tmpl = f"%{{percent{_dec_sp}}}"
                    elif label_mode == "Name only":
                        _tmpl = "%{label}"
                    else:
                        _tmpl = ""
                    fig = go.Figure(go.Pie(
                        labels=mf["Category"].tolist(),
                        values=mf["Amount ($M)"].tolist(),
                        hole=0.55 if shape == "Donut" else 0.0,
                        sort=False, direction="clockwise",
                        marker=dict(colors=colors),
                        texttemplate=_tmpl,
                        textposition=("none" if label_mode == "None" else "auto"),
                        hovertemplate="%{label}: $%{value:.1f}M "
                                      f"(%{{percent{_dec_sp}}})<extra></extra>"))
                    fig.update_layout(**_leg_kw)
                fig.update_layout(
                    title=dict(text=title, x=TITLE_X[0], xanchor=TITLE_X[1],
                               font=dict(size=18),
                               subtitle=dict(text=subtitle) if subtitle else None),
                    height=height,
                    margin=dict(l=10,
                                r=160 if legend_pos == "Right" else 10,
                                t=64 if subtitle else 48,
                                b=30 + (26 if source else 0)))
                if source:
                    fig.add_annotation(text=source, xref="paper", yref="paper",
                                       x=0, y=-0.02, showarrow=False,
                                       xanchor="left", yanchor="top",
                                       font=dict(size=11, color="#888"))
                chart_slot.plotly_chart(fig, use_container_width=True,
                                        config=export_config(title))
                tot = v.get(total_code.upper())
                chart_slot.caption(f"{src} for {cu_name}, {_period_label(period, span)} — "
                                   f"foots to {money(tot)}. Slices under 0.05% fold into "
                                   f"“{resid}”.")
        else:
            with tab_data:
                comp_basis = st.radio("Show", ["Share %", "Dollars"], horizontal=True)
                comp_orient = st.radio("Orientation", ["Columns", "Bars"], horizontal=True,
                                       key="comp_orient")
            horiz = comp_orient == "Bars"
            x_is_category = not horiz
            col_key = "Share" if comp_basis == "Share %" else "Amount ($M)"
            per = {c: (lambda m: dict(zip(m["Category"], m[col_key])) if not m.empty else {})(
                       mix_frame(acct_values(comp_cu, c, sig), parts, total_code, resid))
                   for c in in_range}
            totals = {}
            for dmap in per.values():
                for k, val in dmap.items():
                    totals[k] = totals.get(k, 0.0) + (val or 0.0)
            cats = sorted(totals, key=lambda k: (k == resid, -totals[k]))
            if not cats:
                chart_slot.info(f"No {src.lower()} history for {cu_name} over this range.")
            else:
                with tab_data:
                    title = st.text_input("Chart title",
                                          value=f"{cu_name} — {src.lower()} over time",
                                          key=f"comp_ts_{src}_{comp_cu}")
                fig = go.Figure()
                for i, cat in enumerate(cats):
                    yv = [per[c].get(cat) for c in in_range]
                    col = cat_pal[i % len(cat_pal)]
                    fig.add_trace(go.Bar(y=idx, x=yv, orientation="h", name=cat,
                                         marker_color=col) if horiz
                                  else go.Bar(x=idx, y=yv, name=cat, marker_color=col))
                fig.update_layout(barmode="stack", bargap=0.25)
                if comp_basis == "Share %":
                    vkw = dict(title="Share of total", ticksuffix="%")
                else:
                    vkw = dict(title="Amount", tickprefix="$", ticksuffix="M")
                (fig.update_xaxes if horiz else fig.update_yaxes)(**vkw)
                styled(fig, title)
                if horiz:
                    fig.update_xaxes(showgrid=gridlines)
                    fig.update_yaxes(showgrid=False)
                chart_slot.plotly_chart(fig, use_container_width=True,
                                        config=export_config(title))
                chart_slot.caption(f"{src} for {cu_name}, {idx[0]}–{idx[-1]} "
                                   f"({'share of total' if comp_basis == 'Share %' else '$ millions'}, "
                                   "stacked).")

        # ---- Peer-band overlay: this CU's mix vs the asset-band aggregate ----
        with tab_data:
            show_peer_mix = st.checkbox("Compare to peer band", key="comp_peer_mix",
                                        help="Overlay the dollar-weighted composition of "
                                             "this CU's asset band.")
        if show_peer_mix:
            try:
                ref_period = (period if disp.startswith("Snapshot")
                              else (in_range[-1] if in_range else cycle))
                _brow = mt[mt.cu == comp_cu]
                band = (_brow.band.iloc[0] if not _brow.empty
                        else band_of((acct_values(comp_cu, ref_period, sig) or {}).get("ACCT_010")))
                cu_mf = mix_frame(acct_values(comp_cu, ref_period, sig), parts, total_code, resid)
                pm, n_peer = peer_mix_frame(band, ref_period, src, sig)
                if cu_mf.empty or pm.empty or n_peer == 0:
                    chart_slot.info("Peer-band composition isn't available for this "
                                    "selection.")
                else:
                    cu_share = dict(zip(cu_mf["Category"], cu_mf["Share"]))
                    pe_share = dict(zip(pm["Category"], pm["Share"]))
                    cats = list(dict.fromkeys(list(cu_share) + list(pe_share)))
                    cats.sort(key=lambda k: (k == resid or k.startswith("Other"),
                                             -cu_share.get(k, 0)))
                    ov = go.Figure()
                    ov.add_trace(go.Bar(name=cu_name, x=cats,
                                        y=[cu_share.get(k, 0) for k in cats],
                                        marker_color=PALETTE[0]))
                    ov.add_trace(go.Bar(name=f"{band} peer band", x=cats,
                                        y=[pe_share.get(k, 0) for k in cats],
                                        marker_color="#c9ced6"))
                    ov.update_layout(
                        barmode="group", height=380,
                        title=dict(text=f"{cu_name} vs {band} peer band — "
                                        f"{src.lower()} ({_period_label(ref_period, span)})"),
                        yaxis=dict(title="Share of total", ticksuffix="%"),
                        legend=dict(orientation="h", y=1.14, x=0),
                        margin=dict(l=10, r=10, t=70, b=90))
                    chart_slot.plotly_chart(ov, use_container_width=True,
                                            config=export_config("peer_mix"))
                    gaps = pd.DataFrame({
                        "Category": cats,
                        cu_name: [cu_share.get(k) for k in cats],
                        "Peer band": [pe_share.get(k) for k in cats],
                        "Gap (pp)": [cu_share.get(k, 0) - pe_share.get(k, 0) for k in cats]})
                    chart_slot.dataframe(
                        gaps, use_container_width=True, hide_index=True, column_config={
                            cu_name: st.column_config.NumberColumn(format="%.1f%%"),
                            "Peer band": st.column_config.NumberColumn(format="%.1f%%"),
                            "Gap (pp)": st.column_config.NumberColumn(format="%+.1f")})
                    chart_slot.caption(
                        f"Peer band = dollar-weighted {src.lower()} across {n_peer} credit "
                        f"unions in the {band} asset band at "
                        f"{_period_label(ref_period, span)}. Gap = this CU − peer band (pp).")
            except Exception as _exc:
                chart_slot.info(f"Peer-band overlay unavailable ({_exc}).")

    elif mode == "Peer plots":
        x_is_category = False                  # cross-section: period band/marker don't apply
        cs_metrics = [k for k in metric_keys if k in getattr(mt, "columns", [])]
        qlabel = _period_label(cycle, "Quarters")
        with tab_data:
            ptype = st.radio("Plot type",
                             ["Scatter (X vs Y)", "Dot (rank)",
                              "Range (two periods)", "Arrow (change)"], horizontal=True)

        def hl_colors(cu_series):
            has = (cu_series.astype(str) == "61790").any()
            base = "#aab2bd" if has else PALETTE[0]
            return [PALETTE[0] if str(c) == "61790" else base for c in cu_series]

        if ptype.startswith("Scatter"):
            with tab_data:
                sc1, sc2 = st.columns(2)
                xk = sc1.selectbox("X axis", cs_metrics, format_func=lambda k: CHART_LABEL[k],
                                   index=cs_metrics.index("assets") if "assets" in cs_metrics else 0,
                                   key="sc_x")
                yk = sc2.selectbox("Y axis", cs_metrics, format_func=lambda k: CHART_LABEL[k],
                                   index=cs_metrics.index("roa") if "roa" in cs_metrics else 0,
                                   key="sc_y")
                sizek = st.selectbox("Size by (optional)", ["(none)"] + cs_metrics,
                                     format_func=lambda k: "(none)" if k == "(none)"
                                     else CHART_LABEL[k], key="sc_size")
            need = list({xk, yk} | ({sizek} if sizek != "(none)" else set()))
            d = mt[["cu", "cu_name"] + need].copy()
            for c in need:
                d[c] = pd.to_numeric(d[c], errors="coerce")
            d = d.dropna(subset=[xk, yk])
            if d.empty:
                chart_slot.info("No credit unions report both measures at this quarter.")
            else:
                with tab_data:
                    title = st.text_input("Chart title",
                                          value=f"{CHART_LABEL[yk]} vs {CHART_LABEL[xk]} ({qlabel})",
                                          key=f"sc_t_{xk}_{yk}_{cycle}")
                if sizek != "(none)":
                    sv = pd.to_numeric(d[sizek], errors="coerce").fillna(0).clip(lower=0)
                    sizes = 8 + 34 * (sv / (sv.max() or 1)) ** 0.5
                else:
                    sizes = 9
                txt_ = [nm if str(c) == "61790" else "" for c, nm in zip(d.cu, d.cu_name)]
                fig = go.Figure(go.Scatter(
                    x=d[xk], y=d[yk], mode="markers+text", text=txt_, textposition="top center",
                    textfont=dict(size=10), customdata=d.cu_name,
                    marker=dict(color=hl_colors(d.cu), size=sizes, opacity=0.85,
                                line=dict(width=0.5, color="white")),
                    hovertemplate="%{customdata}<br>" + CHART_LABEL[xk] + ": %{x}<br>"
                                  + CHART_LABEL[yk] + ": %{y}<extra></extra>"))
                fig.update_xaxes(title=CHART_LABEL[xk], **axis_kw(chart_kind(xk)))
                fig.update_yaxes(title=CHART_LABEL[yk], **axis_kw(chart_kind(yk)))
                styled(fig, title)
                fig.update_layout(showlegend=False, hovermode="closest")
                fig.update_xaxes(showgrid=gridlines)
                cap = f"Each dot is a credit union at {qlabel}; BluCurrent highlighted."
                if sizek != "(none)":
                    cap += f" Dot size ∝ {CHART_LABEL[sizek]}."
                chart_slot.plotly_chart(fig, use_container_width=True, config=export_config(title))
                chart_slot.caption(cap)

        elif ptype.startswith("Dot"):
            with tab_data:
                metric = st.selectbox("Measure", cs_metrics, format_func=lambda k: CHART_LABEL[k],
                                      index=cs_metrics.index("roa") if "roa" in cs_metrics else 0,
                                      key="dot_metric")
                dc1, dc2 = st.columns(2)
                topn = dc1.slider("Show top", 5, 40, 15, key="dot_topn")
                lowest = dc2.radio("Order", ["Highest first", "Lowest first"],
                                   horizontal=True, key="dot_order") == "Lowest first"
            kind = chart_kind(metric)
            d = mt[["cu", "cu_name", metric]].copy()
            d[metric] = pd.to_numeric(d[metric], errors="coerce")
            d = d.dropna(subset=[metric]).sort_values(metric, ascending=lowest).head(topn)
            if d.empty:
                chart_slot.info("No data for this measure at the selected quarter.")
            else:
                with tab_data:
                    title = st.text_input("Chart title",
                                          value=f"{CHART_LABEL[metric]} — top {len(d)} ({qlabel})",
                                          key=f"dot_t_{metric}_{cycle}")
                fig = go.Figure(go.Scatter(
                    x=d[metric], y=d.cu_name, mode="markers",
                    marker=dict(color=hl_colors(d.cu), size=12,
                                line=dict(width=0.5, color="white")),
                    hovertemplate="%{y}<br>" + CHART_LABEL[metric] + ": %{x}<extra></extra>"))
                fig.update_xaxes(title=CHART_LABEL[metric], **axis_kw(kind))
                styled(fig, title)
                fig.update_layout(showlegend=False, hovermode="closest")
                fig.update_xaxes(showgrid=gridlines)
                fig.update_yaxes(showgrid=False, autorange="reversed")
                chart_slot.plotly_chart(fig, use_container_width=True, config=export_config(title))
                chart_slot.caption(f"{CHART_LABEL[metric]} across peers at {qlabel}; "
                                   "BluCurrent highlighted.")

        else:                                  # Range or Arrow — two periods, one measure
            is_arrow = ptype.startswith("Arrow")
            with tab_data:
                metric = st.selectbox("Measure", cs_metrics, format_func=lambda k: CHART_LABEL[k],
                                      index=cs_metrics.index("assets") if "assets" in cs_metrics else 0,
                                      key="rng_metric")
                rc1, rc2, rc3 = st.columns(3)
                pA = rc1.selectbox("From period", in_range, index=0,
                                   format_func=lambda c: _period_label(c, span), key="rng_a")
                pB = rc2.selectbox("To period", in_range, index=len(in_range) - 1,
                                   format_func=lambda c: _period_label(c, span), key="rng_b")
                topn = rc3.slider("Show top", 5, 30, 12, key="rng_topn")
            if pA == pB:
                chart_slot.info("Pick two different periods.")
            else:
                kind = chart_kind(metric)
                mA = metrics_table(pA)[["cu", "cu_name", metric]].rename(columns={metric: "a"})
                mB = metrics_table(pB)[["cu", metric]].rename(columns={metric: "b"})
                d = mA.merge(mB, on="cu", how="inner")
                d["a"] = pd.to_numeric(d.a, errors="coerce")
                d["b"] = pd.to_numeric(d.b, errors="coerce")
                d = d.dropna(subset=["a", "b"])
                d["chg"] = d.b - d.a
                d = d.reindex(d.chg.abs().sort_values(ascending=False).index).head(topn)
                la, lb = _period_label(pA, span), _period_label(pB, span)
                if d.empty:
                    chart_slot.info("No credit unions have this measure in both periods.")
                else:
                    with tab_data:
                        dflt = (f"{CHART_LABEL[metric]} change, {la} → {lb}" if is_arrow
                                else f"{CHART_LABEL[metric]}: {la} vs {lb}")
                        title = st.text_input("Chart title", value=dflt,
                                              key=f"rng_t_{metric}_{pA}_{pB}_{is_arrow}")
                    UP, DN, NEU = "#059669", "#dc2626", "#9aa7b8"
                    fig = go.Figure()
                    for _, r in d.iterrows():
                        hov = (f"{r.cu_name}<br>{la}: {lab(kind, r.a)} → "
                               f"{lb}: {lab(kind, r.b)}<extra></extra>")
                        if is_arrow:
                            c = UP if r.chg > 0 else DN if r.chg < 0 else NEU
                            sym = "triangle-right" if r.b >= r.a else "triangle-left"
                            fig.add_trace(go.Scatter(
                                x=[r.a, r.b], y=[r.cu_name, r.cu_name], mode="lines+markers",
                                line=dict(color=c, width=2),
                                marker=dict(color=c, size=[4, 13], symbol=["circle", sym]),
                                showlegend=False, hovertemplate=hov))
                        else:
                            fig.add_trace(go.Scatter(
                                x=[r.a, r.b], y=[r.cu_name, r.cu_name], mode="lines+markers",
                                line=dict(color="#cbd2da", width=2),
                                marker=dict(color=[NEU, PALETTE[0]], size=11),
                                showlegend=False, hovertemplate=hov))
                    # legend keys
                    if is_arrow:
                        for nm, c in [("Increase", UP), ("Decrease", DN)]:
                            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                          marker=dict(color=c, size=11, symbol="triangle-right"),
                                          name=nm))
                    else:
                        for nm, c in [(la, NEU), (lb, PALETTE[0])]:
                            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                          marker=dict(color=c, size=11), name=nm))
                    fig.update_xaxes(title=CHART_LABEL[metric], **axis_kw(kind))
                    styled(fig, title)
                    fig.update_layout(hovermode="closest")
                    fig.update_xaxes(showgrid=gridlines)
                    fig.update_yaxes(showgrid=False, autorange="reversed")
                    chart_slot.plotly_chart(fig, use_container_width=True,
                                            config=export_config(title))
                    noun = "biggest movers" if is_arrow else "widest spreads"
                    chart_slot.caption(f"{CHART_LABEL[metric]}, {la} → {lb} — top {len(d)} by "
                                       f"absolute change ({noun}).")

    else:
        with tab_data:
            cu_keys = list(ALL_LABELS)
            cu_pick = st.selectbox("Credit union", cu_keys,
                                   index=cu_keys.index(default_cu) if default_cu in cu_keys else 0,
                                   format_func=lambda c: ALL_LABELS[c])
            dual = st.checkbox("Overlay two measures on dual axes")
            peer_on = st.checkbox("Add peer median line", value=False,
                                  help="Dashed benchmark = median of each measure across this "
                                       "credit union's asset band, per period.")
        cu_name = ALL_LABELS[cu_pick].split(" (#")[0]
        pband = cu_band(cu_pick, in_range[-1]) if (peer_on and in_range) else None
        s = chart_series(cu_pick, sig)
        s = s.reindex(in_range) if not s.empty else s
        if s.empty:
            chart_slot.info("No history for this credit union over the selected range.")
        elif dual:
            with tab_data:
                d1, d2 = st.columns(2)
                mA = d1.selectbox("Left axis", metric_keys, format_func=lambda k: CHART_LABEL[k],
                                  index=metric_keys.index("nim"), key="dual_a")
                mB = d2.selectbox("Right axis", metric_keys, format_func=lambda k: CHART_LABEL[k],
                                  index=metric_keys.index("assets"), key="dual_b")
                title = st.text_input("Chart title",
                                      value=f"{cu_name}: {CHART_LABEL[mA]} vs {CHART_LABEL[mB]}",
                                      key=f"ct_dual_{cu_pick}_{mA}_{mB}")
            yA = chg(pd.to_numeric(s[mA], errors="coerce")) if mA in s else pd.Series(dtype=float)
            yB = chg(pd.to_numeric(s[mB], errors="coerce")) if mB in s else pd.Series(dtype=float)
            kA, kB = vkind(chart_kind(mA)), vkind(chart_kind(mB))
            cmap = series_colors(colors_box, [CHART_LABEL[mA], CHART_LABEL[mB]],
                                 "dualcol_" + palette_name)
            colA, colB = cmap[CHART_LABEL[mA]], cmap[CHART_LABEL[mB]]
            rngL = manual_range(range_box, f"{CHART_LABEL[mA]} (left)", minmax(yA), "dualL")
            rngR = manual_range(range_box, f"{CHART_LABEL[mB]} (right)", minmax(yB), "dualR")
            yL = dict(title=vlabel(CHART_LABEL[mA]), **axis_kw(kA))
            yR = dict(title=vlabel(CHART_LABEL[mB]), overlaying="y", side="right",
                      **axis_kw(kB))
            if rngL:
                yL["range"] = rngL
            if rngR:
                yR["range"] = rngR
            posB = {"top center": "bottom center", "bottom center": "top center"}.get(POS, POS)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=idx, y=yA.values, mode=line_mode, name=CHART_LABEL[mA],
                                     yaxis="y", connectgaps=True, text=txt(yA, kA),
                                     textposition=POS, marker=dict(color=colA),
                                     line=dict(color=colA, width=line_width, shape=LSHAPE)))
            fig.add_trace(go.Scatter(x=idx, y=yB.values, mode=line_mode, name=CHART_LABEL[mB],
                                     yaxis="y2", connectgaps=True, text=txt(yB, kB),
                                     textposition=posB, marker=dict(color=colB),
                                     line=dict(color=colB, width=line_width, shape=LSHAPE)))
            fig.update_layout(yaxis=yL, yaxis2=yR)
            if pband:
                pmA, _ = peer_median_line(mA, pband, tuple(in_range), sig)
                pmB, _ = peer_median_line(mB, pband, tuple(in_range), sig)
                if pmA:
                    fig.add_trace(go.Scatter(
                        x=idx, y=chg(pd.Series([pmA.get(c) for c in in_range])).tolist(),
                        mode="lines", yaxis="y",
                        name=f"Peer median · {CHART_LABEL[mA]}", connectgaps=True,
                        line=dict(color="#9aa1ab", width=2, dash="dash"),
                        hovertemplate="Peer median: %{y}<extra></extra>"))
                if pmB:
                    fig.add_trace(go.Scatter(
                        x=idx, y=chg(pd.Series([pmB.get(c) for c in in_range])).tolist(),
                        mode="lines", yaxis="y2",
                        name=f"Peer median · {CHART_LABEL[mB]}", connectgaps=True,
                        line=dict(color="#c2c7cf", width=2, dash="dash"),
                        hovertemplate="Peer median: %{y}<extra></extra>"))
            styled(fig, title)
            chart_slot.plotly_chart(fig, use_container_width=True, config=export_config(title))
            data_download(pd.DataFrame({CHART_LABEL[mA]: yA.values, CHART_LABEL[mB]: yB.values},
                                       index=idx),
                          f"{cu_pick}_{mA}_{mB}_{value_mode}.csv".replace(" ", "_"),
                          f"dual_{cu_pick}_{mA}_{mB}")
            chart_slot.caption(f"{cu_name} — {vlabel(CHART_LABEL[mA])} (left axis) vs "
                               f"{vlabel(CHART_LABEL[mB])} (right axis), {idx[0]}–{idx[-1]} "
                               f"({span.lower()}).")
        else:
            with tab_data:
                sel = st.multiselect("Measures", metric_keys,
                                     format_func=lambda k: CHART_LABEL[k],
                                     default=["nim", "roa", "efficiency", "nw_ratio"])
                title = st.text_input("Chart title", value=f"{cu_name} — selected measures",
                                      key=f"ct_sm_{cu_pick}_" + "_".join(sel))
            if not sel:
                chart_slot.info("Choose at least one measure.")
            else:
                with chart_slot:
                    st.markdown(f"<h4 style='text-align:{title_align.lower()};margin:0'>{title}"
                                "</h4>", unsafe_allow_html=True)
                    st.caption(f"{cu_name} — each measure on its own axis, {idx[0]}–{idx[-1]} "
                               f"({span.lower()}). Tick “Overlay two measures on dual axes” in the "
                               "Data tab to combine two on a single chart with independent scales."
                               + (f" Dashed grey = peer median ({pband})." if pband else ""))
                    grid = st.columns(2)
                    dl_sm = {}
                    for i, k in enumerate(sel):
                        y = chg(pd.to_numeric(s[k], errors="coerce")) if k in s else pd.Series(dtype=float)
                        kk = vkind(chart_kind(k))
                        dl_sm[CHART_LABEL[k]] = y.reindex(in_range).values if not y.empty else None
                        fig = go.Figure(go.Scatter(
                            x=idx, y=y.values, mode=line_mode, name=CHART_LABEL[k],
                            connectgaps=True, text=txt(y, kk), textposition=POS,
                            marker=dict(color=accent),
                            line=dict(color=accent, width=line_width, shape=LSHAPE),
                            **fill_kw(accent)))
                        if pband:
                            pm, _ = peer_median_line(k, pband, tuple(in_range), sig)
                            if pm:
                                fig.add_trace(go.Scatter(
                                    x=idx, y=chg(pd.Series([pm.get(c) for c in in_range])).tolist(),
                                    mode="lines", name="Peer median", connectgaps=True,
                                    line=dict(color="#6b7280", width=2, dash="dash"),
                                    hovertemplate="Peer median: %{y}<extra></extra>"))
                        fig.update_yaxes(showgrid=gridlines, **axis_kw(kk))
                        fig.update_xaxes(showgrid=False)
                        fig.update_traces(textfont_size=9)
                        fig.update_layout(height=max(220, height // 2), title=vlabel(CHART_LABEL[k]),
                                          showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
                        grid[i % 2].plotly_chart(fig, use_container_width=True,
                                                 key=f"sm_{i}_{k}",
                                                 config=export_config(f"{cu_name} {CHART_LABEL[k]}"))
                    data_download(
                        pd.DataFrame({c: v for c, v in dl_sm.items() if v is not None}, index=idx),
                        f"{cu_pick}_measures_{value_mode}.csv".replace(" ", "_"),
                        f"sm_{cu_pick}_" + "_".join(sel))

# ============================================================ RANKINGS
elif page == "Rankings":
    st.subheader("Screen & Rank All Credit Unions")
    f1, f2 = st.columns(2)
    all_states = sorted(s for s in mt.state.unique() if s)
    sel_states = f1.multiselect("State(s)", all_states, default=[])
    sel_bands = f2.multiselect("Asset size", [b[2] for b in BANDS], default=[])
    include_small = st.checkbox(
        f"Include credit unions under ${RANK_MIN_ASSETS / 1e6:.0f}M in assets "
        "(off by default — tiny books produce unstable ratios and skew the ranking)",
        value=False)
    rankable = ["score"] + [k for k, _, _, _ in METRICS]
    g1, g2 = st.columns([2, 1])
    rank_key = g1.selectbox("Rank by", rankable,
                            format_func=lambda k: META[k][0], index=0)
    default_desc = META[rank_key][2] != "low"
    order = g2.radio("Order", ["Top (high→low)", "Bottom (low→high)"],
                     index=0 if default_desc else 1, horizontal=True)
    acq = merger_acquirers(cycle, sig)
    inf = inferred_acquirers(cycle, sig)
    tagged = set(acq) | set(inf)
    excl = False
    if tagged:
        excl = st.checkbox("Exclude credit unions that absorbed another in the last 4 quarters "
                           "(merger-driven results)", value=False)
    if rank_key in GROWTH_KEYS and not excl:
        st.caption("Heads-up: extreme growth usually reflects a merger/acquisition — "
                   "tick the box above to exclude those.")
    view = mt.copy()
    if not include_small:
        view = view[view.assets >= RANK_MIN_ASSETS]
    if sel_states:
        view = view[view.state.isin(sel_states)]
    if sel_bands:
        view = view[view.band.isin(sel_bands)]
    if excl:
        view = view[~view.cu.isin(tagged)]
    view = view.dropna(subset=[rank_key]).sort_values(
        rank_key, ascending=order.startswith("Bottom"))
    if rank_key in ("score", "stars"):
        show_keys = ["score", "stars"] + [k for k, _ in weights]
    else:
        show_keys = ["score", "stars", "net_worth", "roa", "efficiency",
                     "nw_ratio", "delinquency"]
        if rank_key not in show_keys and rank_key != "assets":
            show_keys.insert(2, rank_key)
    # Always show Total Assets right after the star rating, for size context.
    show_keys = [k for k in show_keys if k != "assets"]
    show_keys.insert(2, "assets")
    disp = pd.DataFrame({"Credit Union": view.cu_name.values, "City": view.city.values,
                         "State": view.state.values})
    colcfg = {}
    for k in show_keys:
        lbl, f = META[k][0], META[k][1]
        if f == "stars":
            disp[lbl] = [stars_str(x) for x in view[k].values]            # keep ★ string
        elif f == "money":
            if k == "assets":
                disp[lbl] = view[k].round(0).values                       # full dollars
                colcfg[lbl] = st.column_config.NumberColumn(lbl, format="$%,.0f")
            else:
                col = f"{lbl} ($M)"
                disp[col] = (view[k] / 1e6).values                        # numeric, in $M
                colcfg[col] = st.column_config.NumberColumn(col, format="$%.1f")
        elif f == "score":
            disp[lbl] = view[k].round(0).values                           # numeric 0–100
            colcfg[lbl] = st.column_config.ProgressColumn(
                lbl, min_value=0, max_value=100, format="%d")
        elif f == "int":
            disp[lbl] = view[k].values
            colcfg[lbl] = st.column_config.NumberColumn(lbl, format="%d")
        else:                                                             # pct
            disp[lbl] = view[k].values
            colcfg[lbl] = st.column_config.NumberColumn(lbl, format="%.2f%%")
    if tagged and not excl:
        disp["Merger"] = [merger_tag(c, acq, inf) for c in view.cu.values]
    disp.insert(0, "Rank", range(1, len(disp) + 1))
    colcfg["Rank"] = st.column_config.NumberColumn("Rank", format="%d", width="small")
    floor_note = "" if include_small else f" ≥ ${RANK_MIN_ASSETS / 1e6:.0f}M assets"
    st.caption(f"{len(view):,} credit unions shown{floor_note} (of {len(mt):,} total), {cycle}, "
               f"ranked by **{META[rank_key][0]}**. Change **Rank by** above to re-rank by any "
               "measure — the Rank column updates to match. (Clicking a grid header re-sorts the "
               "rows for a quick look, but won't renumber Rank.)")
    disp["Open"] = [f"?view=Profile&cu={c}" for c in view.cu.values]
    colcfg["Open"] = st.column_config.LinkColumn("Open", display_text="Profile ↗", width="small")
    st.dataframe(disp, use_container_width=True, hide_index=True, height=600,
                 column_config=colcfg)

# ============================================================ MERGER HISTORY
elif page == "Merger History":
    st.subheader("Merger History")
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
    st.subheader("Industry Overview")
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
        with st.expander("Industry Table (All Quarters)"):
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

    st.subheader(f"By State — {cycle}")
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

# ============================================================ RATES
elif page == "Yields":
    st.subheader("Rate & Spread Leaderboard")
    rt = rate_table(cycle, sig)
    if rt.empty:
        st.info("No rate data available for this quarter.")
    else:
        sub, _, label = screen_filters(rt, "rates", show_top=False)
        labels = {k: lbl for k, lbl, _ in RATE_COLS}
        dirs = {k: d for k, _, d in RATE_COLS}
        g1, g2 = st.columns([2, 1])
        rank_key = g1.selectbox("Rank by", [k for k, _, _ in RATE_COLS],
                                format_func=lambda k: labels[k])
        order = g2.radio("Order", ["Top (high→low)", "Bottom (low→high)"],
                         index=0 if dirs[rank_key] == "high" else 1, horizontal=True)
        v = sub.dropna(subset=[rank_key]).sort_values(
            rank_key, ascending=order.startswith("Bottom"))
        if v.empty:
            st.info("No credit unions with valid figures for this filter.")
        else:
            disp = pd.DataFrame({"Credit Union": v.cu_name.values, "City": v.city.values,
                                 "State": v.state.values,
                                 "Total Assets": v.assets.round(0).values})
            colcfg = {"Total Assets": st.column_config.NumberColumn("Total Assets",
                                                                    format="$%,.0f")}
            for k, lbl, _ in RATE_COLS:
                disp[lbl] = v[k].values
                colcfg[lbl] = st.column_config.NumberColumn(lbl, format="%.2f%%")
            disp.insert(0, "Rank", range(1, len(disp) + 1))
            colcfg["Rank"] = st.column_config.NumberColumn("Rank", format="%d", width="small")
            st.caption(f"All {len(v):,} credit unions ({label}), {cycle}, ranked by "
                       f"**{labels[rank_key]}**. Change **Rank by** above to re-rank by any "
                       "measure — the Rank column updates to match. (Clicking a grid header "
                       "re-sorts rows only.) Yields on the NCUA FPR average-balance basis "
                       "((current + prior year-end) ÷ 2); cost of funds is total interest "
                       "expense over average shares + borrowings.")
            disp["Open"] = [f"?view=Profile&cu={c}" for c in v.cu.values]
            colcfg["Open"] = st.column_config.LinkColumn("Open", display_text="Profile ↗", width="small")
            st.dataframe(disp, use_container_width=True, hide_index=True, height=600,
                         column_config=colcfg)

# ============================================================ TARGETS (M&A)
elif page == "M&A Targets":
    st.subheader("M&A Target Screener")
    st.caption("Ranks credit unions by acquisition-target attractiveness from four distress "
               "signals — small size, shrinking, thin capital, and weak earnings. Confirmed "
               "and likely acquirers are excluded (they're consolidators, not targets).")
    acq = merger_acquirers(cycle, sig)
    inf = inferred_acquirers(cycle, sig)
    tagged = set(acq) | set(inf)
    pool = mt[~mt.cu.isin(tagged)].copy()
    sub, label = universe_picker(pool, "tgt")
    sub = sub.dropna(subset=["assets", "nw_ratio", "roa", "efficiency"])
    if len(sub) < 5:
        st.info("Not enough credit unions in this universe to score.")
    else:
        def asc_pct(s):
            return pd.to_numeric(s, errors="coerce").rank(pct=True)
        grw = sub[["assets_growth", "members_growth"]].mean(axis=1)
        size = 1 - asc_pct(sub.assets)                       # smaller -> more target-like
        shrink = 1 - asc_pct(grw)                            # lower growth -> more
        cap = 1 - asc_pct(sub.nw_ratio)                      # thinner capital -> more
        earn = 0.5 * (1 - asc_pct(sub.roa)) + 0.5 * asc_pct(sub.efficiency)   # weak earnings
        sub = sub.assign(target=(100 * (0.25 * size + 0.25 * shrink
                                        + 0.25 * cap + 0.25 * earn)).round(0))
        v = sub.sort_values("target", ascending=False)
        disp = pd.DataFrame({"Credit Union": v.cu_name.values, "City": v.city.values,
                             "State": v.state.values,
                             "Band": v.band.values, "Total Assets": v.assets.round(0).values,
                             "Asset Growth": v.assets_growth.values,
                             "Member Growth": v.members_growth.values,
                             "Net Worth Ratio": v.nw_ratio.values, "ROA": v.roa.values,
                             "Efficiency": v.efficiency.values,
                             "Target Score": v.target.values})
        pctcols = ["Asset Growth", "Member Growth", "Net Worth Ratio", "ROA", "Efficiency"]
        colcfg = {"Total Assets": st.column_config.NumberColumn("Total Assets", format="$%,.0f"),
                  "Target Score": st.column_config.ProgressColumn(
                      "Target Score", min_value=0, max_value=100, format="%d")}
        for c2 in pctcols:
            colcfg[c2] = st.column_config.NumberColumn(c2, format="%.2f%%")
        disp.insert(0, "Rank", range(1, len(disp) + 1))
        colcfg["Rank"] = st.column_config.NumberColumn("Rank", format="%d", width="small")
        st.caption(f"All {len(v):,} candidates in {label}, {cycle}, sorted by target score "
                   "(highest = most target-like). (Clicking a grid header re-sorts rows for a "
                   "quick look but won't renumber Rank.) Equal-weighted (25% each): size "
                   "(smaller), growth (shrinking), capital (thinner net worth), and earnings "
                   "(low ROA / high efficiency) — each scored as a percentile within this "
                   "universe. A screen, not a recommendation.")
        disp["Open"] = [f"?view=Profile&cu={c}" for c in v.cu.values]
        colcfg["Open"] = st.column_config.LinkColumn("Open", display_text="Profile ↗", width="small")
        st.dataframe(disp, use_container_width=True, hide_index=True, height=600,
                     column_config=colcfg)

# ============================================================ DATA QUALITY
elif page == "Data Health":
    st.subheader("Data health")
    cs = sorted(health, reverse=True)
    n_err = sum(1 for h in health.values() if h["status"] == "error")
    n_warn = sum(1 for h in health.values() if h["status"] == "warn")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cycles", len(health))
    m2.metric("Latest cycle", cs[0] if cs else "—")
    m3.metric("Errors", n_err)
    m4.metric("Warnings", n_warn)

    if cs:
        cov = pd.DataFrame(
            {"Credit Unions": [health[c]["cu_count"] for c in sorted(health)]},
            index=sorted(health))
        st.caption("Credit-union count by cycle — a sharp drop usually means a partial ingest")
        st.line_chart(cov)

    badge = {"ok": "🟢 OK", "warn": "🟡 Warning", "error": "🔴 Error"}
    rows = []
    for c in cs:
        h = health[c]
        rows.append({"Cycle": c, "Status": badge[h["status"]],
                     "Credit Unions": f"{h['cu_count']:,}",
                     "Missing income": f"{h['zero_income_pct']*100:.0f}%",
                     "Missing assets": f"{h['zero_assets_pct']*100:.0f}%",
                     "Out-of-range ratios": f"{h['implausible_pct']*100:.1f}%",
                     "Notes": " ".join(h["issues"]) if h["issues"] else "—"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=420)
    st.caption("Checks per cycle: missing interest income (>30% → error), missing assets "
               "(>10% → error), credit-union count drop vs the prior cycle (>25% → error, "
               ">10% → warning), broken year-to-date income accumulation, and clusters of "
               "out-of-range ratios (>5% → warning). Flagged cycles also show missing (—) or "
               "suppressed figures in the rate views.")
