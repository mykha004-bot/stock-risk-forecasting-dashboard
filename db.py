"""SQLite persistence layer.

Design:
- Composite PRIMARY KEY (ticker, date) makes writes idempotent: re-running a
  refresh never duplicates rows, and re-fetching the last stored day to catch
  late price adjustments is safe (ON CONFLICT ... DO UPDATE).
- We store BOTH raw close and adj_close. Returns/analysis use adj_close
  (splits + dividends baked in); raw OHLC is kept for reference.
- A refresh_log table records every attempt (ok/empty/error) so the app can
  show data freshness and failures honestly instead of silently serving stale
  data.
"""
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    ticker     TEXT NOT NULL,
    date       TEXT NOT NULL,          -- ISO 'YYYY-MM-DD'
    open       REAL,
    high       REAL,
    low        REAL,
    close      REAL,
    adj_close  REAL,
    volume     INTEGER,
    PRIMARY KEY (ticker, date)
);
CREATE INDEX IF NOT EXISTS idx_prices_ticker ON prices(ticker);
CREATE INDEX IF NOT EXISTS idx_prices_date   ON prices(date);

CREATE TABLE IF NOT EXISTS refresh_log (
    ticker      TEXT,
    run_at      TEXT,      -- ISO UTC timestamp
    rows_written INTEGER,
    status      TEXT,      -- 'ok' | 'empty' | 'error'
    detail      TEXT
);
"""


@contextmanager
def connect(db_path):
    """Context-managed connection that ensures the parent dir exists and commits."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path):
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


# write ---------------------------------------------------------------

def _f(x):
    """Coerce to float, mapping NaN/None -> None (SQLite NULL)."""
    try:
        if x is None or pd.isna(x):
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _i(x):
    v = _f(x)
    return int(v) if v is not None else None


def upsert_prices(db_path, ticker, df):
    """Upsert a normalized single-ticker frame.

    df: index = dates, columns include open/high/low/close/adj_close/volume.
    Returns number of rows written (inserted or updated).
    """
    if df is None or df.empty:
        return 0
    rows = [
        (
            ticker,
            pd.Timestamp(idx).strftime("%Y-%m-%d"),
            _f(r.get("open")), _f(r.get("high")), _f(r.get("low")),
            _f(r.get("close")), _f(r.get("adj_close")), _i(r.get("volume")),
        )
        for idx, r in df.iterrows()
    ]
    with connect(db_path) as conn:
        conn.executemany(
            """INSERT INTO prices
                   (ticker, date, open, high, low, close, adj_close, volume)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(ticker, date) DO UPDATE SET
                   open=excluded.open, high=excluded.high, low=excluded.low,
                   close=excluded.close, adj_close=excluded.adj_close,
                   volume=excluded.volume;""",
            rows,
        )
    return len(rows)


def log_refresh(db_path, ticker, rows_written, status, detail=""):
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO refresh_log (ticker, run_at, rows_written, status, detail) "
            "VALUES (?, ?, ?, ?, ?)",
            (ticker, pd.Timestamp.now("UTC").isoformat(), rows_written, status,
             (detail or "")[:500]),
        )


# --- read ----------------------------------------------------------------

def last_date(db_path, ticker):
    """Most recent stored date for a ticker (ISO string) or None if absent."""
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT MAX(date) FROM prices WHERE ticker = ?", (ticker,)
        ).fetchone()
    return row[0] if row and row[0] else None


def load_prices(db_path, tickers=None, start=None, end=None, field="adj_close"):
    """Load prices.

    field='adj_close' (default) -> wide frame: index=date, columns=tickers.
    field=None                  -> long frame with all OHLCV columns.
    """
    q = ("SELECT ticker, date, open, high, low, close, adj_close, volume "
         "FROM prices")
    clauses, params = [], []
    if tickers:
        clauses.append(f"ticker IN ({','.join('?' * len(tickers))})")
        params += list(tickers)
    if start:
        clauses.append("date >= ?"); params.append(start)
    if end:
        clauses.append("date <= ?"); params.append(end)
    if clauses:
        q += " WHERE " + " AND ".join(clauses)
    q += " ORDER BY date"

    with connect(db_path) as conn:
        df = pd.read_sql_query(q, conn, params=params, parse_dates=["date"])

    if field is None:
        return df
    if df.empty:
        return pd.DataFrame()
    return df.pivot(index="date", columns="ticker", values=field).sort_index()


def coverage(db_path):
    """Per-ticker row counts and date ranges — used for a data-health panel."""
    with connect(db_path) as conn:
        return pd.read_sql_query(
            "SELECT ticker, COUNT(*) AS rows, MIN(date) AS start, MAX(date) AS end "
            "FROM prices GROUP BY ticker ORDER BY ticker",
            conn,
        )


def last_refresh(db_path):
    """Most recent refresh_log rows (one per ticker) for a freshness panel."""
    with connect(db_path) as conn:
        return pd.read_sql_query(
            """SELECT r.ticker, r.run_at, r.rows_written, r.status, r.detail
               FROM refresh_log r
               JOIN (SELECT ticker, MAX(run_at) AS mx FROM refresh_log GROUP BY ticker) m
                 ON r.ticker = m.ticker AND r.run_at = m.mx
               ORDER BY r.ticker""",
            conn,
        )
