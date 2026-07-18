import numpy as np
import pandas as pd
import pytest

import analysis as an


def _prices_from_returns(rets_dict, start="2020-01-01"):
    """Build a wide price frame from per-ticker simple-return arrays."""
    n = len(next(iter(rets_dict.values())))
    idx = pd.bdate_range(start, periods=n + 1)
    out = {}
    for t, r in rets_dict.items():
        out[t] = np.concatenate([[100.0], 100.0 * np.cumprod(1 + np.asarray(r))])
    return pd.DataFrame(out, index=idx)


def test_log_returns_recover_known_series():
    prices = pd.DataFrame({"X": [100, 110, 121]},
                          index=pd.bdate_range("2020-01-01", periods=3))
    r = an.daily_returns(prices, kind="log")
    assert r["X"].iloc[0] == pytest.approx(np.log(1.1))
    assert r["X"].iloc[1] == pytest.approx(np.log(1.1))


def test_perfectly_correlated_series_corr_is_one():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.01, 300)
    prices = _prices_from_returns({"A": base, "B": base})  # identical -> corr 1
    r = an.daily_returns(prices)
    c = an.correlation_matrix(r)
    assert c.loc["A", "B"] == pytest.approx(1.0, abs=1e-6)


def test_independent_series_corr_near_zero():
    rng = np.random.default_rng(1)
    prices = _prices_from_returns({
        "A": rng.normal(0, 0.01, 4000),
        "B": rng.normal(0, 0.01, 4000),
    })
    r = an.daily_returns(prices)
    c = an.correlation_matrix(r)
    assert abs(c.loc["A", "B"]) < 0.05  # independent -> ~0


def test_annualized_vol_matches_formula():
    rng = np.random.default_rng(2)
    daily = rng.normal(0, 0.02, 5000)
    prices = _prices_from_returns({"A": daily})
    r = an.daily_returns(prices, kind="simple")
    expected = r["A"].std() * np.sqrt(252)
    assert an.annualized_volatility(r)["A"] == pytest.approx(expected)


def test_historical_var_is_empirical_quantile():
    rng = np.random.default_rng(3)
    daily = rng.normal(0, 0.02, 6000)
    prices = _prices_from_returns({"A": daily})
    r = an.daily_returns(prices, kind="simple")
    var95 = an.historical_var(r, alpha=0.05)["A"]
    assert var95 == pytest.approx(-r["A"].quantile(0.05))
    assert var95 > 0  # reported as a positive loss


def test_cvar_at_least_var():
    rng = np.random.default_rng(4)
    daily = rng.normal(0, 0.02, 6000)
    prices = _prices_from_returns({"A": daily})
    r = an.daily_returns(prices, kind="simple")
    var = an.historical_var(r, 0.05)["A"]
    cvar = an.historical_cvar(r, 0.05)["A"]
    assert cvar >= var  # expected shortfall is never less than VaR


def test_max_drawdown_known_path():
    # 100 -> 120 -> 60 -> 90 : worst peak-to-trough is 120->60 = 50%
    prices = pd.DataFrame({"A": [100, 120, 60, 90]},
                          index=pd.bdate_range("2020-01-01", periods=4))
    assert an.max_drawdown(prices)["A"] == pytest.approx(0.5)


def test_cluster_order_groups_correlated_names():
    rng = np.random.default_rng(5)
    g1 = rng.normal(0, 0.01, 800)   # cluster 1 driver
    g2 = rng.normal(0, 0.01, 800)   # cluster 2 driver
    noise = lambda: rng.normal(0, 0.002, 800)
    prices = _prices_from_returns({
        "A1": g1 + noise(), "A2": g1 + noise(),
        "B1": g2 + noise(), "B2": g2 + noise(),
    })
    r = an.daily_returns(prices)
    order = an.cluster_order(an.correlation_matrix(r))
    # A-group members should be adjacent, and B-group members adjacent.
    a_positions = sorted(order.index(t) for t in ["A1", "A2"])
    b_positions = sorted(order.index(t) for t in ["B1", "B2"])
    assert a_positions[1] - a_positions[0] == 1
    assert b_positions[1] - b_positions[0] == 1


def test_rolling_mean_correlation_shape_and_range():
    rng = np.random.default_rng(6)
    prices = _prices_from_returns({
        "A": rng.normal(0, 0.01, 200),
        "B": rng.normal(0, 0.01, 200),
        "C": rng.normal(0, 0.01, 200),
    })
    r = an.daily_returns(prices)
    s = an.rolling_mean_correlation(r, window=30)
    assert len(s) == len(r) - 30 + 1
    assert s.dropna().between(-1, 1).all()


def test_risk_summary_columns_and_sort():
    rng = np.random.default_rng(7)
    prices = _prices_from_returns({
        "CALM": rng.normal(0.0003, 0.008, 500),
        "WILD": rng.normal(0.0003, 0.03, 500),
    })
    summary = an.risk_summary(prices)
    for col in ["sector", "ann_return", "ann_vol", "sharpe", "max_drawdown"]:
        assert col in summary.columns
    # sorted by vol descending -> the high-vol name is first
    assert summary.index[0] == "WILD"
