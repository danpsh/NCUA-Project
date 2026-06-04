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


def _stars_pdf(n):
    n = int(n) if pd.notna(n) else 0
    return "\u2605" * n + "\u2606" * (5 - n)


def tearsheet_pdf(row, cycle, lens, hist, vals):
    """One-page PDF tearsheet for a single credit union."""
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Table, TableStyle)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=LETTER, topMargin=0.55 * inch, bottomMargin=0.45 * inch,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
        title=f"{row.cu_name} — {cycle} tearsheet")
    S = getSampleStyleSheet()
    teal = colors.HexColor("#0b6b5e")
    grey = colors.HexColor("#666666")
    grid = colors.HexColor("#dddddd")
    headbg = colors.HexColor("#eef2f1")
    title = ParagraphStyle("t", parent=S["Title"], fontSize=17, alignment=0, spaceAfter=1)
    subt = ParagraphStyle("s", parent=S["Normal"], fontSize=9, textColor=grey)
    big = ParagraphStyle("b", parent=S["Normal"], fontSize=11, spaceBefore=4)
    sec = ParagraphStyle("sec", parent=S["Heading4"], fontSize=10.5,
                         textColor=teal, spaceBefore=11, spaceAfter=3)
    foot = ParagraphStyle("f", parent=S["Normal"], fontSize=7.5, textColor=grey)

    def tbl(data, widths, header=True):
        t = Table(data, colWidths=widths)
        sty = [("FONTSIZE", (0, 0), (-1, -1), 9),
               ("GRID", (0, 0), (-1, -1), 0.5, grid),
               ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
               ("LEFTPADDING", (0, 0), (-1, -1), 6),
               ("RIGHTPADDING", (0, 0), (-1, -1), 6),
               ("TOPPADDING", (0, 0), (-1, -1), 3),
               ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
        if header:
            sty += [("BACKGROUND", (0, 0), (-1, 0), headbg),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
        t.setStyle(TableStyle(sty))
        return t

    el = [Paragraph(row.cu_name, title),
          Paragraph(f"Charter #{row.cu} &nbsp;·&nbsp; {row.state} &nbsp;·&nbsp; {cycle} "
                    f"&nbsp;·&nbsp; peer group {row.band}", subt)]
    sc = f"{row.score:.0f}/100" if pd.notna(row.score) else "\u2014"
    el.append(Paragraph(
        f"<b>Composite score {sc}</b> &nbsp; {_stars_pdf(row.stars)} "
        f"&nbsp;&nbsp;<font color='#666' size=8>{lens}</font>", big))

    el.append(Paragraph("Key figures", sec))
    el.append(tbl([
        ["Total Assets", money(row.assets), "Net Worth", money(row.net_worth)],
        ["Members", intfmt(row.members), "Net Worth Ratio", pct(row.nw_ratio)],
        ["ROA", pct(row.roa), "Efficiency Ratio", pct(row.efficiency)],
    ], [1.3 * inch, 1.7 * inch, 1.6 * inch, 1.6 * inch], header=False))

    def col2(left, right):
        t = Table([[left, right]], colWidths=[3.35 * inch, 3.35 * inch])
        t.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (0, 0), 0), ("RIGHTPADDING", (0, 0), (0, 0), 12),
            ("LEFTPADDING", (1, 0), (1, 0), 0), ("RIGHTPADDING", (1, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
        return t

    rk = ["roa", "roe", "nim", "efficiency", "nw_ratio", "delinquency", "nco", "lts"]
    ratio_tbl = tbl([["Metric", "Value"]] + [[META[k][0], fmt(k, row[k])] for k in rk],
                    [2.1 * inch, 1.05 * inch])
    growth_tbl = tbl([["Metric", "Value"]] + [[META[k][0], fmt(k, row[k])] for k in GROWTH_KEYS],
                     [2.1 * inch, 1.05 * inch])
    left = [Paragraph("Profitability, capital &amp; asset quality", sec), ratio_tbl]
    right = [Paragraph("Growth", sec), growth_tbl]
    if hist is not None and len(hist) > 1:
        hdata = [["Yr", "Score", "%ile"]]
        for ix, r in hist.iterrows():
            lbl = ix[:4] if str(ix).endswith("-12") else str(ix)
            hdata.append([lbl, f"{r.score:.0f}", f"{r.pct:.0f}%"])
        right += [Paragraph("Composite score history", sec),
                  tbl(hdata, [1.2 * inch, 1.0 * inch, 0.95 * inch])]
    el.append(col2(left, right))

    lm = mix_frame(vals, LOAN_MIX, "ACCT_025B", "Other")
    dm = mix_frame(vals, DEPOSIT_MIX, "ACCT_018", "Other (incl. IRA / Keogh)")
    cell = ParagraphStyle("cell", parent=S["Normal"], fontSize=8.5, leading=10)

    def mix_tbl(mx):
        data = [["Category", "Amt", "Share"]] + [
            [Paragraph(r["Category"], cell), f"${r['Amount ($M)']:,.1f}M", f"{r['Share']:.1f}%"]
            for _, r in mx.iterrows()]
        return tbl(data, [1.9 * inch, 0.75 * inch, 0.65 * inch])
    mleft = [Paragraph("Loan mix", sec), mix_tbl(lm)] if not lm.empty else []
    mright = [Paragraph("Deposit mix", sec), mix_tbl(dm)] if not dm.empty else []
    if mleft or mright:
        el.append(col2(mleft, mright))

    el.append(Spacer(1, 10))
    el.append(Paragraph(
        "Source: NCUA 5300 Call Report. Composite score is a peer-relative z-score blend "
        "(50 = peer average) under the selected lens. Loan and deposit composition are from "
        "Section 1 and the supplemental share schedule. Generated by NCUA Call Report "
        "Explorer.", foot))
    doc.build(el)
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
    d = con.execute(f"""
      SELECT o.CU_NUMBER AS cu, o.CU_NAME AS cu_name, COALESCE(o.STATE, '') AS state,
        TRY_CAST(f.ACCT_010  AS DOUBLE) AS assets,
        TRY_CAST(p.assets_pye AS DOUBLE) AS assets_pye,
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
        SELECT CU_NUMBER, TRY_CAST(ACCT_010 AS DOUBLE) AS assets_pye
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

    def ratio(num, den):
        return np.where(den.notna() & (den != 0), num / den * 100, np.nan)

    d["roa"] = ratio(d.net_income * annualize, d.assets)
    d["roe"] = ratio(d.net_income * annualize, d.net_worth)
    d["nim"] = ratio(nii * annualize, avg_assets)   # NCUA FPR: NII / average assets
    d["efficiency"] = ratio(d.opex, nii + d.non_int_income)
    d["nw_ratio"] = ratio(d.net_worth, d.assets)
    d["lts"] = ratio(d.loans, d.shares)
    d["delinquency"] = ratio(d.delinquent, d.loans)
    d["nco"] = ratio((d.chargeoffs - d.recoveries) * annualize, d.loans)
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
    idx = f.index

    def col(df, code):
        if df is None or df.empty or code not in df.columns:
            return pd.Series(np.nan, index=idx)
        return pd.to_numeric(df[code], errors="coerce").reindex(idx)

    bal = lambda code: col(f, code).fillna(col(a, code))      # balances: FS220, then FS220A
    balp = lambda code: col(fp, code)                          # prior year-end: FS220

    def avg(cur_s, pye_s):                                     # FPR (cur + PYE) / 2; robust
        if fp.empty or pye_s is None:
            return cur_s
        use = pye_s.notna() & (pye_s > 0)
        return cur_s.where(~use, (cur_s.fillna(0) + pye_s) / 2)

    def inv_base(getter):
        return (getter("ACCT_010") - getter("ACCT_025B") - getter("ACCT_730A")
                - getter("ACCT_007") - getter("ACCT_008") - getter("ACCT_794")
                - getter("ACCT_009A") - getter("ACCT_009B") - getter("ACCT_009C"))

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
    base = metrics_table(cycle)[["cu", "cu_name", "state", "band", "assets"]].copy()
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
    # Composite score: weighted blend of band-relative z-scores (SDs from the asset-band
    # peer mean). The peer mean/std EXCLUDE recent acquirers, so merger-driven growth
    # outliers don't distort the baseline; acquirers are still scored, but get no credit
    # for merger-bought growth. Direction-adjusted so positive always = better.
    acc = pd.Series(0.0, index=df.index)
    wsum = pd.Series(0.0, index=df.index)
    for key, w in weights:
        peers = df.loc[~is_acq]                           # clean baseline (no acquirers)
        stats = peers.groupby("band")[key].agg(["mean", "std"])
        mean = df.band.map(stats["mean"])
        std = df.band.map(stats["std"])
        z = ((df[key] - mean) / std).where(std > 0)       # undefined where no spread
        if META[key][2] == "low":
            z = -z                                        # lower is better → flip sign
        df[f"z_{key}"] = z
        contrib = z.mask(is_acq) if key.endswith("_growth") else z   # no merger-growth credit
        acc = acc + contrib.fillna(0) * w
        wsum = wsum + contrib.notna().astype(float) * w
    df["score_z"] = (acc / wsum).where(wsum > 0)          # weighted composite z
    df["score"] = (50 + 30 * df["score_z"]).clip(0, 100)  # 50 = peer average
    df["stars"] = df["score_z"].apply(stars_from_z)
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
        rows.append({"cycle": c, "score": s, "stars": r.iloc[0].stars,
                     "pct": float((et.score < s).mean() * 100)})
    return pd.DataFrame(rows).set_index("cycle") if rows else pd.DataFrame()


@st.cache_data(show_spinner=False)
def acct_values(cu, cycle, cycle_sig):
    """All FS220 + FS220A account balances for one credit union / cycle as
    {ACCT_CODE_UPPER: float}."""
    vals = {}
    for table in ("FS220", "FS220A", "FS220L", "FS220H"):
        try:
            df = con.execute(
                f"SELECT * FROM read_parquet('{glob_for(table)}', "
                "hive_partitioning=true, union_by_name=true) "
                "WHERE cycle = ? AND CU_NUMBER = ?", [cycle, cu]).df()
        except Exception:
            continue
        if df.empty:
            continue
        r = df.iloc[0]
        for c in df.columns:
            try:
                x = float(r[c])
                if x == x:
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
        z = row.get(f"z_{key}")
        if is_acq and key.endswith("_growth"):
            rows.append((META[key][0], fmt(key, row[key]), "excluded — merger", "—"))
        else:
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
health = data_health(sig)

# ---- Sidebar: brand ----
st.sidebar.markdown("#### 📊 Call Report Explorer")

# ---- Sidebar: navigation (grouped by workflow, icon-labeled) ----
NAV = ["Profile", "Compare", "Chart", "Rankings", "Yields", "M&A Targets",
       "Movers", "Merger History", "Industry", "Data Health"]
NAV_ICON = {"Profile": "👤", "Compare": "⚖️", "Chart": "📉", "Rankings": "🏆",
            "Yields": "📈", "M&A Targets": "🎯", "Movers": "🚀",
            "Merger History": "🔀", "Industry": "🏛️", "Data Health": "🩺"}
page = st.sidebar.radio("View", NAV, format_func=lambda p: f"{NAV_ICON[p]}  {p}",
                        label_visibility="collapsed")


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
GROWTH_PAGES = {"Profile", "Compare", "Rankings", "M&A Targets", "Movers"}  # show growth basis
GROWTH_OPTS = ["Year-over-year", "Quarter-over-quarter (annualized)"]
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

# label helper shared across pages
ALL_LABELS = {r.cu: f"{r.cu_name} (#{r.cu}, {r.state})" for r in mt.itertuples()}


def universe_picker(df, key):
    """Industry / band / state filter shared by the Rates and Targets pages.
    Returns (filtered_df, human_label)."""
    mode = st.radio("Universe", ["Whole industry", "By asset band", "By state"],
                    horizontal=True, key=f"{key}_mode")
    if mode == "By asset band":
        b = st.selectbox("Asset band", [x[2] for x in BANDS], index=4, key=f"{key}_band")
        return df[df.band == b], f"the {b} band"
    if mode == "By state":
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
    query = st.text_input("Search a credit union by name", value="BluCurrent",
                          placeholder="e.g. BluCurrent")
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
            h[4].metric("Composite Score",
                        f"{row.score:.0f}/100" if pd.notna(row.score) else "—",
                        help=f"{lens} z-score blend vs the asset-band peer group; "
                             "50 = peer average.")
            st.markdown(
                f"<div style='font-size:1.6rem;line-height:1.1'>{stars_str(row.stars)}"
                "</div>", unsafe_allow_html=True)
            cap = f"{lens} score vs {row.band} peers — 50 = peer average."
            if prev_row is not None:
                cap += f"  ·  ▲▼ deltas vs prior quarter ({prev_cy})."
            st.caption(cap)
            with st.expander("How this score is built"):
                st.dataframe(score_breakdown(row, weights), use_container_width=True,
                             hide_index=True)

            tab_ov, tab_fin, tab_tr, tab_peer, tab_mrg = st.tabs(
                ["Overview", "Financials", "Trends", "Peers", "Mergers"])

            # ===================================================== OVERVIEW
            with tab_ov:
                scorecard_groups = [
                    ("Size & balance sheet",
                     ["assets", "loans", "shares", "net_worth", "net_income", "members"]),
                    (f"Growth ({growth_label.lower()})", GROWTH_KEYS),
                    ("Profitability", ["roa", "roe", "nim", "efficiency"]),
                    ("Capital, asset quality & liquidity",
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
                st.markdown("**Composite score history**")
                hist = score_history(cu, basis, lens, cycle, sig)
                if len(hist) > 1:
                    labels = [c[:4] if c.endswith("-12") else c for c in hist.index]
                    chart_df = pd.DataFrame({"Composite Score": hist.score.values}, index=labels)
                    st.line_chart(chart_df, y="Composite Score")
                    htbl = pd.DataFrame({
                        "Period": labels,
                        "Composite Score": [f"{v:.0f}" for v in hist.score],
                        "Stars": [stars_str(v) for v in hist.stars],
                        "Percentile (all CUs)": [f"{v:.0f}%" for v in hist.pct],
                    })
                    st.dataframe(htbl, use_container_width=True, hide_index=True)
                    st.caption(f"{lens} composite at each year-end (plus the selected "
                               "quarter). Percentile = share of all credit unions outscored "
                               "that period. History follows the charter across conversions.")
                else:
                    st.caption("Composite history needs more than one period of data.")

                st.divider()
                sc = pd.DataFrame(
                    {"Value": {META[k][0]: fmt(k, row[k]) for k, _, _, _ in METRICS}})
                dl1, dl2 = st.columns(2)
                dl1.download_button(
                    "Download scorecard (Excel)",
                    to_excel_bytes({"Scorecard": sc}),
                    file_name=f"{row.cu_name}_{cycle}_scorecard.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
                try:
                    pdf_bytes = tearsheet_pdf(row, cycle, lens, hist,
                                              acct_values(cu, cycle, sig))
                    dl2.download_button(
                        "Download tearsheet (PDF)", pdf_bytes,
                        file_name=f"{row.cu_name}_{cycle}_tearsheet.pdf",
                        mime="application/pdf", use_container_width=True,
                    )
                except Exception as exc:
                    dl2.caption(f"PDF tearsheet unavailable ({exc}).")
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

            # ===================================================== FINANCIALS
            with tab_fin:
                sc1, sc2 = st.columns(2)
                stmt = sc1.radio("Statement", ["Balance sheet", "Income statement"],
                                 horizontal=True)
                pmode = sc2.radio("Periods", ["Quarters", "Years"], horizontal=True)
                schema = BALANCE_SHEET if stmt == "Balance sheet" else INCOME_STATEMENT
                sdf = build_statement(cu, schema, stmt == "Income statement", pmode, cycle, sig)
                if sdf.empty:
                    st.info("No statement data available for this credit union and period.")
                else:
                    st.dataframe(sdf, use_container_width=True, hide_index=True,
                                 height=38 * len(sdf) + 38)
                    note = ("Built from NCUA call report accounts and tied to the reported "
                            "totals. Lines marked “implied” (investments, provision for credit "
                            "losses, other) are derived so the statement foots exactly.")
                    if stmt == "Income statement":
                        note += (" Income figures are year-to-date in the call report; the "
                                 "Quarters view de-cumulates them into standalone quarters.")
                    st.caption(note)

                st.divider()
                render_yield_spread(cu, cycle, sig, pmode)

                st.divider()
                vals = acct_values(cu, cycle, sig)
                mc1, mc2 = st.columns(2)
                with mc1:
                    st.markdown("**Loan mix**")
                    lm = mix_frame(vals, LOAN_MIX, "ACCT_025B", "Other")
                    if lm.empty:
                        st.caption("No loan data for this credit union.")
                    else:
                        mix_dataframe(lm)
                        st.caption("Loan composition from Section 1 of the NCUA 5300 "
                                   "(first mortgage, other RE, vehicle, commercial, and "
                                   "consumer categories); foots to total loans & leases.")
                with mc2:
                    st.markdown("**Deposit mix**")
                    dm = mix_frame(vals, DEPOSIT_MIX, "ACCT_018",
                                   "Other (incl. IRA / Keogh)")
                    if dm.empty:
                        st.caption("No deposit data for this credit union.")
                    else:
                        mix_dataframe(dm)
                        st.caption("Share composition from the NCUA call report; the residual "
                                   "captures IRA/Keogh and any other shares.")

                with st.expander("Raw call report tables (advanced)"):
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
                    if tspan == "Years" and not tsd.empty:
                        tsd = tsd[[str(i).endswith("-12") for i in tsd.index]]
                    if not tsd.empty and chosen:
                        grid = st.columns(2)
                        for i, key in enumerate(chosen):
                            with grid[i % 2]:
                                st.caption(META[key][0])
                                st.line_chart(tsd[[key]].rename(columns={key: META[key][0]}))
                else:
                    st.info("Only one quarter of data is available — trends need more history.")

            # ===================================================== PEERS
            with tab_peer:
                basis_choice = st.radio(
                    "Compare against",
                    ["Similar asset size", f"Same state ({row.state})",
                     "Same state + asset size", "All credit unions", "Custom (pick CUs)"],
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
                with st.expander("Identity / FOICU fields"):
                    foicu = con.execute(
                        f"SELECT * FROM read_parquet('{glob_for('FOICU')}', "
                        "hive_partitioning=true, union_by_name=true) "
                        "WHERE cycle = ? AND CU_NUMBER = ?", [cycle, cu]).df()
                    st.dataframe(foicu.T, use_container_width=True)

            # ===================================================== MERGERS
            with tab_mrg:
                mg = merger_table(sig)
                mine = (mg[mg.continuing_charter == cu].sort_values("cycle", ascending=False)
                        if not mg.empty else pd.DataFrame())
                if not mine.empty:
                    st.caption(f"This credit union has absorbed {len(mine)} other "
                               f"{'institution' if len(mine) == 1 else 'institutions'} since "
                               "2018, per the NCUA Insurance Report of Activity.")
                    st.dataframe(pd.DataFrame({
                        "Quarter": mine.cycle.values,
                        "Absorbed": mine.merging_name.values,
                        "Assets at merger": [money(x) for x in mine.merging_assets.values],
                        "Reason": mine.reason.values}),
                        use_container_width=True, hide_index=True)
                else:
                    st.info("No absorbed mergers recorded for this credit union in the "
                            "NCUA Insurance Report of Activity.")

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
            st.subheader("Trend overlays")
            oc1, oc2 = st.columns([1, 3])
            ov_span = oc1.radio("Period", ["Quarters", "Years"], horizontal=True,
                                key="cmp_span")
            ov_opts = [k for k, _, _, _ in METRICS if not k.endswith("_growth")]
            ov_keys = oc2.multiselect(
                "Metrics to chart", ov_opts, default=ov_opts,
                format_func=lambda k: META[k][0])
            grid = st.columns(2)
            for i, key in enumerate(ov_keys):
                series = multi_cu_series(picks, key, ALL_LABELS, sig)
                if ov_span == "Years" and not series.empty:
                    series = series[[str(ix).endswith("-12") for ix in series.index]]
                if not series.empty:
                    with grid[i % 2]:
                        st.caption(META[key][0])
                        st.line_chart(series)

# ============================================================ CHART
elif page == "Chart":
    st.subheader("Chart builder")
    mode = st.radio("Chart type",
                    ["Compare credit unions on one measure",
                     "One credit union across measures"], horizontal=True)

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
    idx = [_period_label(c, span) for c in in_range]
    metric_keys = [k for k, _, _ in CHART_METRICS]
    default_cu = next((c for c in ALL_LABELS if str(c) == "61790"), next(iter(ALL_LABELS), None))

    if len(in_range) < 2:
        st.info("Pick a range spanning at least two periods to plot a trend.")
    elif mode.startswith("Compare"):
        metric = st.selectbox("Measure", metric_keys, format_func=lambda k: CHART_LABEL[k],
                              index=metric_keys.index("nim"))
        picks = st.multiselect("Credit unions (up to 6)", list(ALL_LABELS),
                               default=[default_cu] if default_cu is not None else [],
                               max_selections=6, format_func=lambda c: ALL_LABELS[c])
        if not picks:
            st.info("Choose at least one credit union.")
        else:
            data = {}
            for c in picks:
                s = chart_series(c, sig)
                if s.empty or metric not in s:
                    continue
                name = ALL_LABELS[c].split(" (#")[0]
                data[name] = pd.to_numeric(s[metric], errors="coerce").reindex(in_range).values
            chart = pd.DataFrame(data, index=idx)
            if chart.dropna(how="all").empty:
                st.info("No data for this measure over the selected range.")
            else:
                st.line_chart(chart)
                st.caption(f"{CHART_LABEL[metric]} by quarter ({span.lower()}), "
                           f"{idx[0]}–{idx[-1]}. Yields on the NCUA FPR average-balance basis; "
                           "money measures are in dollars.")
    else:
        cu_keys = list(ALL_LABELS)
        cu_pick = st.selectbox("Credit union", cu_keys,
                               index=cu_keys.index(default_cu) if default_cu in cu_keys else 0,
                               format_func=lambda c: ALL_LABELS[c])
        sel = st.multiselect("Measures", metric_keys, format_func=lambda k: CHART_LABEL[k],
                             default=["nim", "roa", "efficiency", "nw_ratio"])
        s = chart_series(cu_pick, sig)
        if s.empty or not sel:
            st.info("Choose at least one measure (and a credit union with history).")
        else:
            s = s.reindex(in_range)
            st.caption(f"{ALL_LABELS[cu_pick].split(' (#')[0]} — each measure on its own axis, "
                       f"{idx[0]}–{idx[-1]} ({span.lower()}).")
            grid = st.columns(2)
            for i, k in enumerate(sel):
                with grid[i % 2]:
                    st.caption(CHART_LABEL[k])
                    col = pd.to_numeric(s[k], errors="coerce") if k in s else pd.Series(dtype=float)
                    st.line_chart(pd.DataFrame({CHART_LABEL[k]: col.values}, index=idx))

# ============================================================ RANKINGS
elif page == "Rankings":
    st.subheader("Screen & rank all credit unions")
    f1, f2 = st.columns(2)
    all_states = sorted(s for s in mt.state.unique() if s)
    sel_states = f1.multiselect("State(s)", all_states, default=[])
    sel_bands = f2.multiselect("Asset size", [b[2] for b in BANDS], default=[])
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
    disp = pd.DataFrame({"Credit Union": view.cu_name.values, "State": view.state.values})
    colcfg = {}
    for k in show_keys:
        lbl, f = META[k][0], META[k][1]
        if f == "stars":
            disp[lbl] = [stars_str(x) for x in view[k].values]            # keep ★ string
        elif f == "money":
            col = f"{lbl} ($M)"
            disp[col] = (view[k] / 1e6).values                            # numeric, in $M
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
    st.caption(f"All {len(view):,} credit unions shown (of {len(mt):,} total), {cycle} — "
               f"initially ranked by **{META[rank_key][0]}**. Click any column header to "
               "re-rank the whole table by that measure (the Rank column reflects the "
               "“Rank by” choice).")
    st.dataframe(disp, use_container_width=True, hide_index=True, height=600,
                 column_config=colcfg)

# ============================================================ MOVERS
elif page == "Movers":
    st.subheader("Biggest movers")
    st.caption(f"Growth basis: {growth_label.lower()}. In the Merger column, ✓ marks a confirmed "
               "acquirer (last 4 quarters) and ≈ a likely one not yet in the NCUA report — "
               "toggle the box to exclude them.")
    m1, m2 = st.columns([2, 1])
    gkey = m1.selectbox("Growth metric", GROWTH_KEYS, format_func=lambda k: META[k][0])
    min_assets = m2.selectbox("Minimum asset size",
                              ["$10M", "$50M", "$100M", "$500M", "$1B"], index=1)
    floor = {"$10M": 10e6, "$50M": 50e6, "$100M": 100e6, "$500M": 500e6, "$1B": 1e9}[min_assets]
    acq = merger_acquirers(cycle, sig)
    inf = inferred_acquirers(cycle, sig)
    tagged = set(acq) | set(inf)
    hide = False
    if tagged:
        hide = st.checkbox("Exclude merger-driven growth (credit unions that absorbed "
                           "another in the last 4 quarters)", value=False)
    pool = mt[(mt.assets >= floor)].dropna(subset=[gkey]).copy()
    pool["_tag"] = [merger_tag(c, acq, inf) for c in pool.cu.values]
    if hide:
        pool = pool[pool._tag == ""]
    if pool.empty:
        st.info("No credit unions with growth data for this basis yet "
                "(needs a prior-period quarter ingested).")
    else:
        cols = ["cu_name", "state", "assets", gkey, "_tag"]
        gain = pool.nlargest(15, gkey)[cols]
        lose = pool.nsmallest(15, gkey)[cols]

        def fmt_movers(df):
            d = pd.DataFrame({
                "Credit Union": df.cu_name.values, "State": df.state.values,
                "Assets": [money(x) for x in df.assets.values],
                META[gkey][0]: [pct(x) for x in df[gkey].values]})
            if tagged and not hide:
                d["Merger"] = df._tag.values
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
elif page == "Merger History":
    st.subheader("Merger history")
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

# ============================================================ RATES
elif page == "Yields":
    st.subheader("Rate & spread leaderboard")
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
            disp = pd.DataFrame({"Credit Union": v.cu_name.values, "State": v.state.values,
                                 "Assets ($M)": (v.assets / 1e6).values})
            colcfg = {"Assets ($M)": st.column_config.NumberColumn("Assets ($M)", format="$%.0f")}
            for k, lbl, _ in RATE_COLS:
                disp[lbl] = v[k].values
                colcfg[lbl] = st.column_config.NumberColumn(lbl, format="%.2f%%")
            disp.insert(0, "Rank", range(1, len(disp) + 1))
            colcfg["Rank"] = st.column_config.NumberColumn("Rank", format="%d", width="small")
            st.caption(f"All {len(v):,} credit unions ({label}), {cycle} — initially ranked by "
                       f"**{labels[rank_key]}**. Click any column header to re-rank the entire "
                       "table by that measure (the Rank column reflects the “Rank by” choice). "
                       "Yields on the NCUA FPR average-balance basis ((current + prior "
                       "year-end) ÷ 2); cost of funds is total interest expense over average "
                       "shares + borrowings.")
            st.dataframe(disp, use_container_width=True, hide_index=True, height=600,
                         column_config=colcfg)

# ============================================================ TARGETS (M&A)
elif page == "M&A Targets":
    st.subheader("M&A target screener")
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
        disp = pd.DataFrame({"Credit Union": v.cu_name.values, "State": v.state.values,
                             "Band": v.band.values, "Assets ($M)": (v.assets / 1e6).values,
                             "Asset Growth": v.assets_growth.values,
                             "Member Growth": v.members_growth.values,
                             "Net Worth Ratio": v.nw_ratio.values, "ROA": v.roa.values,
                             "Efficiency": v.efficiency.values,
                             "Target Score": v.target.values})
        pctcols = ["Asset Growth", "Member Growth", "Net Worth Ratio", "ROA", "Efficiency"]
        colcfg = {"Assets ($M)": st.column_config.NumberColumn("Assets ($M)", format="$%.0f"),
                  "Target Score": st.column_config.ProgressColumn(
                      "Target Score", min_value=0, max_value=100, format="%d")}
        for c2 in pctcols:
            colcfg[c2] = st.column_config.NumberColumn(c2, format="%.2f%%")
        disp.insert(0, "Rank", range(1, len(disp) + 1))
        colcfg["Rank"] = st.column_config.NumberColumn("Rank", format="%d", width="small")
        st.caption(f"All {len(v):,} candidates in {label}, {cycle} — sorted by target score "
                   "(click any column header to re-rank by that measure). Higher score = "
                   "more target-like. Equal-weighted (25% each): size (smaller), growth "
                   "(shrinking), capital (thinner net worth), and earnings (low ROA / high "
                   "efficiency) — each scored as a percentile within this universe. A screen, "
                   "not a recommendation.")
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
