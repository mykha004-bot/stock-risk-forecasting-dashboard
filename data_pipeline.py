"""Data pipeline: fetch daily OHLCV from yfinance into SQLite.

Two entry points:
- seed(...)    : full history for every ticker. Uses a single batched download
                 (1 request instead of N) to minimize rate-limit exposure on the
                 heavy initial pull, with per-ticker fallback for stragglers.
- refresh(...) : incremental. For each ticker, fetch only from its last stored
                 date forward. Cheap (few new rows) and idempotent.

Resilience (the "handle rate limits gracefully" requirement):
- Exponential backoff + retries per ticker on any exception, incl. yfinance's
  YFRateLimitError.
- A failure NEVER wipes existing data. On error we log it and move on, so the
  app always falls back to whatever is already cached in the DB.

- Refresh only on demand
"""
import logging
import time

import pandas as pd

import config
import db

log = logging.getLogger("pipeline")


# --- normalization -------------------------------------------------------

_RENAME = {
    "Open": "open", "High": "high", "Low": "low",
    "Close": "close", "Adj Close": "adj_close", "Volume": "volume",
}


def _normalize(df):
    """yfinance frame -> tidy lowercase OHLCV, tz-naive datetime index."""
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df = df.rename(columns=_RENAME)
    # If auto_adjust collapsed 'Adj Close', fall back to close so downstream
    # return calculations still have a column to use.
    if "adj_close" not in df.columns and "close" in df.columns:
        df = df.assign(adj_close=df["close"])
    keep = [c for c in ("open", "high", "low", "close", "adj_close", "volume")
            if c in df.columns]
    if not keep:
        return pd.DataFrame()
    df = df[keep].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].dropna(how="all")
    return df


def _flatten_columns(raw, ticker=None):
    """Collapse yfinance MultiIndex columns to a single OHLCV level."""
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = set(raw.columns.get_level_values(0))
        # Single-ticker download: ('Open','AAPL') -> take level 0.
        if {"Open", "Close"} & lvl0:
            raw = raw.copy()
            raw.columns = raw.columns.get_level_values(0)
    return raw


# fetching ------------------------------------------------------------

def _fetch_one(ticker, start):
    """Fetch a single ticker with retry + exponential backoff.

    Returns (normalized_df, status, detail) where status is ok/empty/error.
    """
    import yfinance as yf

    last_detail = ""
    for attempt in range(config.MAX_RETRIES):
        try:
            raw = yf.download(
                ticker, start=start, progress=False,
                auto_adjust=False, threads=False, timeout=30,
            )
            raw = _flatten_columns(raw, ticker)
            norm = _normalize(raw)
            if not norm.empty:
                return norm, "ok", ""
            last_detail = "empty response"
        except Exception as e:  # noqa: BLE001 - incl. YFRateLimitError
            last_detail = f"{type(e).__name__}: {e}"
            log.warning("fetch %s attempt %d/%d failed: %s",
                        ticker, attempt + 1, config.MAX_RETRIES, last_detail)
        time.sleep(config.BACKOFF_BASE_SECONDS * (2 ** attempt))

    status = "empty" if last_detail == "empty response" else "error"
    return pd.DataFrame(), status, last_detail


def _fetch_batch(tickers, start):
    """One batched download for the whole basket; per-ticker fallback on gaps.

    Returns {ticker: (df, status, detail)}.
    """
    import yfinance as yf

    try:
        raw = yf.download(
            tickers, start=start, progress=False, auto_adjust=False,
            threads=False, group_by="ticker", timeout=60,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("batch download failed (%s); falling back to per-ticker", e)
        return {t: _fetch_one(t, start) for t in tickers}

    out = {}
    multi = isinstance(raw.columns, pd.MultiIndex)
    for t in tickers:
        try:
            sub = raw[t] if multi else raw
        except (KeyError, TypeError):
            sub = None
        norm = _normalize(sub) if sub is not None else pd.DataFrame()
        if norm.empty:
            # Retry this one individually with backoff.
            out[t] = _fetch_one(t, start)
        else:
            out[t] = (norm, "ok", "")
    return out


# orchestration ------------------------------

def seed(tickers=None, db_path=None):
    """Full-history pull for every ticker (batched). Builds the seed DB."""
    tickers = tickers or config.TICKERS
    db_path = db_path or config.DB_PATH
    db.init_db(db_path)

    results = []
    fetched = _fetch_batch(tickers, config.HISTORY_START)
    for t in tickers:
        norm, status, detail = fetched[t]
        written = db.upsert_prices(db_path, t, norm)
        db.log_refresh(db_path, t, written, status, detail)
        results.append({"ticker": t, "status": status, "rows_written": written,
                        "detail": detail})
        log.info("seed %s: %s (+%d rows) %s", t, status, written, detail)
    return pd.DataFrame(results)


def refresh(tickers=None, db_path=None, full=False):
    """Incremental refresh. Fetch only new dates per ticker; upsert; log.

    full=True forces a re-pull from HISTORY_START (delegates to seed()).
    Never destructive: on failure, existing cached data is left intact.
    """
    tickers = tickers or config.TICKERS
    db_path = db_path or config.DB_PATH
    if full:
        return seed(tickers, db_path)

    db.init_db(db_path)
    results = []
    for t in tickers:
        ld = db.last_date(db_path, t)
        # Start AT the last stored date (not the day after) so we re-pull it and
        # capture any late split/dividend adjustment; the upsert dedupes it.
        start = ld if ld else config.HISTORY_START
        norm, status, detail = _fetch_one(t, start)
        written = db.upsert_prices(db_path, t, norm)
        db.log_refresh(db_path, t, written, status, detail)
        results.append({"ticker": t, "status": status, "rows_written": written,
                        "detail": detail})
        log.info("refresh %s: %s (+%d rows) %s", t, status, written, detail)
        time.sleep(config.REQUEST_SPACING_SECONDS)
    return pd.DataFrame(results)
