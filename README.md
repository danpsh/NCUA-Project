# NCUA Call Report Explorer

A [Streamlit](https://streamlit.io/) app for exploring the NCUA 5300 Call Report — the quarterly financial filing every U.S. federally insured credit union submits. It computes performance metrics, a peer-relative composite score, and merger/conversion-aware history on top of the raw call report tables, all queried in place with [DuckDB](https://duckdb.org/) over Parquet.

## What it does

Six views (selectable in the sidebar):

- **Profile** — Full scorecard for a single credit union: KPI hero row, grouped metrics with prior-quarter deltas, a composite-score history chart, supervisory watch flags, full balance sheet and income statement, loan/deposit composition, peer benchmarking (including custom peer sets), a raw call-report table browser, and Excel + one-page PDF tearsheet export.
- **Compare** — Side-by-side scorecards for 2–5 credit unions, trend overlays, and Excel export.
- **Rankings** — Screen and sort every credit union by any metric, filtered by state and asset size, with an option to exclude recent acquirers so merger-driven results don't dominate.
- **Movers** — Biggest year-over-year (or annualized quarter-over-quarter) gainers and decliners, plus a list of credit unions that exited (merged or were liquidated) between two periods.
- **Mergers** — The NCUA Insurance Report of Activity: who absorbed whom, assets at merger, stated reasons, most active acquirers, and reason breakdowns, filterable by period and state.
- **Industry** — System-wide totals and medians over time, and a per-state summary for the selected quarter.

Sidebar controls apply across all views: the reporting **quarter**, the **growth basis** (year-over-year, or quarter-over-quarter annualized), and the **score lens** (see below).

## The composite score

Each credit union gets a 0–100 composite score (50 = peer average) and a 1–5 star rating, built as a weighted blend of peer-relative z-scores. Peers are the credit unions in the same asset band, and scores are direction-adjusted so that positive always means better (e.g. a lower efficiency ratio scores higher).

Two selectable lenses weight the inputs differently:

- **Momentum (growth-led)** — emphasizes asset, loan, member, and share growth, with earnings and an asset-quality guardrail.
- **Performance (earnings & health)** — emphasizes ROA, net interest margin, efficiency, capital, and asset quality, with growth as a light factor.

To keep mergers from distorting the baseline, the peer mean and standard deviation **exclude recent acquirers**, and acquirers receive **no credit for merger-bought growth**. Acquirers are still scored on everything else.

## Watch flags

The Profile view surfaces supervisory-style flags, including net worth ratio below PCA thresholds (under 6% / 6–7%), negative ROA, delinquency or net charge-offs in the worst 10% of all credit unions, and efficiency ratios above 90%.

## Merger & conversion awareness

Two things make raw call reports hard to read across time, and the app handles both:

- **Charter conversions** (a credit union renumbers, e.g. on a state-to-federal switch) are followed through multi-step chains via a `CONVERSIONS` table, so a credit union's history stays linked across the change.
- **Mergers** are identified two ways: confirmed acquirers from the NCUA `MERGERS` report (where the absorbed charter actually disappears from the call reports within the lookback window), and *likely* unpublished mergers inferred from a credit union's footprint — a single-quarter jump in both members and assets too large to be organic.

## Data

The app reads Parquet files from a local `./data` directory, one subfolder per call-report table, hive-partitioned:

```
data/
  FOICU/    **/*.parquet      # identity (name, state, charter number)
  FS220/    **/*.parquet      # financial statement schedule
  FS220A/   **/*.parquet      # financial statement schedule
  FS220L/   **/*.parquet      # loan detail
  FS220H/   **/*.parquet      # student-loan detail
  AcctDesc/ **/*.parquet      # account code -> description lookup
  CONVERSIONS/ **/*.parquet   # charter conversions (optional)
  MERGERS/     **/*.parquet   # NCUA Insurance Report of Activity (optional)
```

`FOICU`, `FS220`, and `FS220A` are required; the others are optional and degrade gracefully if absent (merger and conversion features simply turn off). All tables are read with DuckDB's `union_by_name=true`, so NCUA's column changes across years don't break older quarters — missing fields read as `NULL` and display as "—".

The repository tracks the data folder itself; populate it via the project's ingest step (a script or GitHub Action that downloads and converts the NCUA quarterly files to Parquet) and commit `data/`. If you launch the app with no data present, it will tell you to run the ingest.

## Requirements

- Python 3.9+
- `streamlit`, `duckdb`, `pandas`, `numpy`
- `openpyxl` (Excel export)
- `reportlab` (PDF tearsheet)

```bash
pip install streamlit duckdb pandas numpy openpyxl reportlab
```

## Running

From the project root (with `./data` populated):

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (default http://localhost:8501).

## Notes on performance

Heavy computations (per-quarter metrics, composite scores, score histories, industry rollups) are cached with Streamlit's `@st.cache_data`. Functions whose results depend on the full set of available quarters take a `cycle_sig` argument so their cache invalidates automatically when a new quarter is ingested. The DuckDB connection is a cached resource shared across reruns.
