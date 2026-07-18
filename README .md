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

## Status

- [x] **Step 1 — Data pipeline**: SQLite storage, batched full seed,
      incremental refresh, retry/backoff, graceful failure, tests.
- [x] **Step 2 — Correlation & risk analysis** (this commit): rolling
      correlation (30/60/90d), hierarchical cluster ordering, per-stock
      annualized vol, historical/parametric VaR + CVaR, drawdown, risk summary.
- [x] **Step 3 — Forecasting** (this commit): leak-proof features, XGBoost
      direction + magnitude, naive & ARIMA baselines, expanding-window
      walk-forward backtest with embargo, honest metrics net of costs.
- [ ] Step 4 — Streamlit app (3 tabs)
- [ ] Step 5 — Full methodology & limitations write-up

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python seed_db.py        # full-history pull -> data/prices.db  (run locally)
python refresh.py        # later: incremental update (only new dates)
python refresh.py --full # force a full re-pull

pytest -q                # run the pipeline tests
```

Then **commit `data/prices.db`** so the deployed app has data without needing a
live pull. Edit the basket in `config.py` (`TICKERS`).

## Layout

| File | Purpose |
|------|---------|
| `config.py` | Basket, sector map, windows, fetch-resilience knobs |
| `db.py` | SQLite schema, idempotent upsert, incremental helpers, reads |
| `data_pipeline.py` | `seed()` (batched full pull) + `refresh()` (incremental) |
| `analysis.py` | Returns, rolling correlation, clustering, vol, VaR/CVaR, risk summary |
| `features.py` | Leak-proof next-day feature/target engineering |
| `forecasting.py` | XGBoost + baselines, walk-forward backtest, honest metrics |
| `seed_db.py` / `refresh.py` | CLI entry points |
| `tests/` | Storage, resilience, and analysis tests (no network needed) |

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
