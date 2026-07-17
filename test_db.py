"""Unit tests for the parts we can verify without hitting Yahoo:
idempotent upsert, incremental last_date logic, wide/long loads, and the
graceful-failure contract of the fetch layer.
"""
import numpy as np
import pandas as pd
import pytest

import db
import data_pipeline as dp


def _frame(dates, base=100.0):
    idx = pd.to_datetime(dates)
    n = len(idx)
    return pd.DataFrame(
        {
            "open": np.linspace(base, base + n, n),
            "high": np.linspace(base + 1, base + n + 1, n),
            "low": np.linspace(base - 1, base + n - 1, n),
            "close": np.linspace(base, base + n, n),
            "adj_close": np.linspace(base, base + n, n),
            "volume": np.arange(1, n + 1) * 1000,
        },
        index=idx,
    )


@pytest.fixture()
def dbfile(tmp_path):
    p = tmp_path / "test.db"
    db.init_db(p)
    return p


def test_upsert_is_idempotent(dbfile):
    df = _frame(["2024-01-02", "2024-01-03", "2024-01-04"])
    db.upsert_prices(dbfile, "AAPL", df)
    db.upsert_prices(dbfile, "AAPL", df)  # same rows again
    wide = db.load_prices(dbfile, ["AAPL"])
    assert len(wide) == 3, "re-inserting identical rows must not duplicate"


def test_upsert_updates_existing_day(dbfile):
    db.upsert_prices(dbfile, "AAPL", _frame(["2024-01-02"], base=100.0))
    db.upsert_prices(dbfile, "AAPL", _frame(["2024-01-02"], base=200.0))  # revised
    wide = db.load_prices(dbfile, ["AAPL"])
    assert len(wide) == 1
    assert wide["AAPL"].iloc[0] == pytest.approx(200.0), "conflict must update, not append"


def test_last_date_tracks_max(dbfile):
    assert db.last_date(dbfile, "AAPL") is None
    db.upsert_prices(dbfile, "AAPL", _frame(["2024-01-02", "2024-01-03"]))
    assert db.last_date(dbfile, "AAPL") == "2024-01-03"


def test_load_wide_pivot_and_long(dbfile):
    db.upsert_prices(dbfile, "AAPL", _frame(["2024-01-02", "2024-01-03"]))
    db.upsert_prices(dbfile, "MSFT", _frame(["2024-01-02", "2024-01-03"], base=50.0))
    wide = db.load_prices(dbfile, ["AAPL", "MSFT"])
    assert list(wide.columns) == ["AAPL", "MSFT"]
    assert wide.shape == (2, 2)
    long = db.load_prices(dbfile, field=None)
    assert {"open", "high", "low", "close", "adj_close", "volume"} <= set(long.columns)


def test_coverage_report(dbfile):
    db.upsert_prices(dbfile, "AAPL", _frame(["2024-01-02", "2024-01-03"]))
    cov = db.coverage(dbfile)
    row = cov[cov["ticker"] == "AAPL"].iloc[0]
    assert row["rows"] == 2
    assert row["start"] == "2024-01-02" and row["end"] == "2024-01-03"


def test_normalize_handles_missing_adj_close():
    raw = pd.DataFrame(
        {"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.5], "Volume": [10]},
        index=pd.to_datetime(["2024-01-02"]),
    )
    out = dp._normalize(raw)
    assert "adj_close" in out.columns
    assert out["adj_close"].iloc[0] == pytest.approx(1.5)  # falls back to close


def test_empty_frame_upsert_is_noop(dbfile):
    assert db.upsert_prices(dbfile, "AAPL", pd.DataFrame()) == 0


def test_refresh_survives_fetch_failure(dbfile, monkeypatch):
    """If fetching a ticker errors, refresh must log + continue, not crash,
    and must leave any existing cached rows intact."""
    db.upsert_prices(dbfile, "AAPL", _frame(["2024-01-02"]))  # pre-existing cache

    def boom(ticker, start):
        return pd.DataFrame(), "error", "simulated YFRateLimitError"

    monkeypatch.setattr(dp, "_fetch_one", boom)
    res = dp.refresh(tickers=["AAPL"], db_path=dbfile)
    assert res.iloc[0]["status"] == "error"
    # Cache preserved despite the failed fetch:
    assert len(db.load_prices(dbfile, ["AAPL"])) == 1
    # Failure recorded for the freshness panel:
    assert (db.last_refresh(dbfile)["status"] == "error").any()
