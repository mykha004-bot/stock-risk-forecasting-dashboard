# Multi-Stock Risk & Forecasting Dashboard

A Streamlit dashboard tracking a basket of ~19 equities/ETFs: rolling
correlation & risk structure, and a next-day return forecast (direction +
magnitude) evaluated honestly against naive and ARIMA baselines.

> **Data caveat (read this):** prices come from Yahoo Finance via `yfinance`,
> an unofficial scraper. Data is **end-of-day, delayed, and not guaranteed**.
> Yahoo rate-limits/blocks aggressive access — especially from shared cloud
> IPs — so the deployed app runs off a **committed seed database** and refreshes
> **on demand**, falling back to cached data when a live pull fails. This is
> not a real-time trading tool.

**→ Full methodology, honest results, and limitations: [METHODOLOGY.md](METHODOLOGY.md).**

## Status

- [x] **Step 1 — Data pipeline**: SQLite storage, batched full seed,
      incremental refresh, retry/backoff, graceful failure, tests.
- [x] **Step 2 — Correlation & risk analysis** (this commit): rolling
      correlation (30/60/90d), hierarchical cluster ordering, per-stock
      annualized vol, historical/parametric VaR + CVaR, drawdown, risk summary.
- [x] **Step 3 — Forecasting** (this commit): leak-proof features, XGBoost
      direction + magnitude, naive & ARIMA baselines, expanding-window
      walk-forward backtest with embargo, honest metrics net of costs.
- [x] **Step 4 — Streamlit app** (this commit): 3-tab dashboard (correlation &
      risk, single-stock forecast, cross-section backtest), cached, on-demand
      refresh, data-freshness panel, precomputed backtest artifact.
- [x] **Step 5 — Methodology & limitations write-up** (this commit): see
      [METHODOLOGY.md](METHODOLOGY.md).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python seed_db.py        # full-history pull -> data/prices.db  (run locally)
python precompute.py     # run all backtests -> data/backtests.pkl (minutes)
streamlit run app.py     # launch the dashboard locally

python refresh.py        # later: incremental price update (only new dates)
pytest -q                # run the test suite (28 tests)
```

Then **commit both `data/prices.db` and `data/backtests.pkl`** so the deployed
app renders instantly without a live pull or a cold-start backtest. Re-run
`precompute.py` after a data refresh. Edit the basket in `config.py` (`TICKERS`).

## Deploy (Streamlit Community Cloud)

1. Push the repo to GitHub with `data/prices.db` and `data/backtests.pkl`
   committed (the `.gitignore` keeps them; it only excludes SQLite temp files).
2. On [share.streamlit.io](https://share.streamlit.io), point a new app at the
   repo, main file `app.py`. `requirements.txt` is picked up automatically.
3. The app boots off the committed artifacts. The **Refresh prices** button
   attempts a live pull; on shared cloud IPs Yahoo often rate-limits, so it
   falls back to cached data — which is the honest, documented behavior.

## Layout

| File | Purpose |
|------|---------|
| `config.py` | Basket, sector map, windows, fetch-resilience knobs |
| `db.py` | SQLite schema, idempotent upsert, incremental helpers, reads |
| `data_pipeline.py` | `seed()` (batched full pull) + `refresh()` (incremental) |
| `analysis.py` | Returns, rolling correlation, clustering, vol, VaR/CVaR, risk summary |
| `features.py` | Leak-proof next-day feature/target engineering |
| `forecasting.py` | XGBoost + baselines, walk-forward backtest, honest metrics |
| `charts.py` | Plotly figure builders (styled, pure functions) |
| `app.py` | Streamlit dashboard (thin wiring layer over the modules above) |
| `precompute.py` | Offline backtest runner -> `data/backtests.pkl` |
| `seed_db.py` / `refresh.py` | CLI entry points for the data pipeline |
| `.streamlit/config.toml` | Dashboard theme |
| `tests/` | Storage, resilience, analysis, and forecasting tests |

## Design notes

- **Idempotent by construction.** `PRIMARY KEY (ticker, date)` + `ON CONFLICT
  DO UPDATE` means re-runs never duplicate, and re-pulling the last stored day
  to catch late split/dividend adjustments is safe.
- **Incremental refresh** fetches only from each ticker's last stored date, so
  routine updates are a handful of rows and minimal requests → less
  rate-limiting.
- **Failures are logged, never fatal.** A `refresh_log` table records
  ok/empty/error per ticker so the app can show data freshness honestly.
- **Both raw `close` and `adj_close` stored**; returns use `adj_close`.
