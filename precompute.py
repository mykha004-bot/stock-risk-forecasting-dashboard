"""Offline precompute: run every ticker's backtest once and cache the results.

Why: a walk-forward backtest across ~19 tickers takes minutes. Doing it on the
deployed app's cold start would time out or feel broken. Instead we run it here,
locally, and commit data/backtests.pkl so the app loads instantly.

    python precompute.py            # full (includes ARIMA baseline)
    python precompute.py --fast     # skip ARIMA (faster)

Re-run after a data refresh to fold in new bars.
"""
import argparse
import logging
import pickle
import time

import pandas as pd

import config
import db
import forecasting as fc

log = logging.getLogger("precompute")
ARTIFACT = config.DATA_DIR / "backtests.pkl"


def run(tickers=None, use_arima=True):
    tickers = tickers or config.TICKERS
    prices = db.load_prices(config.DB_PATH, tickers)
    if prices.empty:
        raise SystemExit("No prices in DB. Run `python seed_db.py` first.")

    results, summary_rows = {}, []
    for t in tickers:
        try:
            bt = fc.backtest_ticker(prices, t, use_arima=use_arima)
            try:
                bt["latest_signal"] = fc.latest_signal(prices, t)
            except Exception as e:  # noqa: BLE001 — keep the backtest regardless
                log.warning("latest_signal failed for %s: %s", t, e)
                bt["latest_signal"] = None
            results[t] = bt
            d, s = bt["direction"], bt["strategy"]
            summary_rows.append({
                "ticker": t,
                "xgb_hit": d.loc["XGBoost", "hit_rate"],
                "naive_hit": d.loc["Naive", "hit_rate"],
                "xgb_auc": d.loc["XGBoost"].get("auc", float("nan")),
                "xgb_sharpe": s.loc["XGBoost L/S", "sharpe"],
                "naive_sharpe": s.loc["Naive L/S", "sharpe"],
                "bh_sharpe": s.loc["Buy & Hold", "sharpe"],
            })
            log.info("done %s (xgb hit %.3f)", t, d.loc["XGBoost", "hit_rate"])
        except Exception as e:  # noqa: BLE001 — skip a ticker, keep going
            log.warning("skip %s: %s", t, e)

    summary = pd.DataFrame(summary_rows).set_index("ticker")
    payload = {"results": results, "summary": summary,
               "built_at": pd.Timestamp.now("UTC").isoformat(),
               "use_arima": use_arima}
    with open(ARTIFACT, "wb") as f:
        pickle.dump(payload, f)
    return payload


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="skip ARIMA baseline")
    args = ap.parse_args()

    t0 = time.time()
    payload = run(use_arima=not args.fast)
    print(f"\nBacktested {len(payload['results'])} tickers in {time.time()-t0:.0f}s")
    print(f"Artifact -> {ARTIFACT}")
    print("\n=== summary ===")
    print(payload["summary"].round(3).to_string())
