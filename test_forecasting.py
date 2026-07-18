import numpy as np
import pandas as pd
import pytest

import features as F
import forecasting as fc


def _prices(returns, start="2016-01-01", p0=100.0):
    idx = pd.bdate_range(start, periods=len(returns) + 1)
    lvl = np.concatenate([[p0], p0 * np.cumprod(1 + np.asarray(returns))])
    return pd.DataFrame({"A": lvl}, index=idx)


# leakage / causality -------------------------------------------------

def test_features_have_no_lookahead():
    rng = np.random.default_rng(0)
    prices = _prices(rng.normal(0, 0.02, 500))
    full = F.make_features(prices, "A")[F.feature_columns()]
    cut_date = full.index[300]
    truncated = F.make_features(prices.loc[:cut_date], "A")[F.feature_columns()]
    common = full.index.intersection(truncated.index)
    # Features on the common dates must be bit-for-bit identical.
    pd.testing.assert_frame_equal(full.loc[common], truncated.loc[common])


def test_target_is_next_day_return():
    prices = _prices([0.01, -0.02, 0.03, 0.00, 0.015])
    data = F.make_features(prices, "A")
    ret = prices["A"].pct_change()
    # target_ret at date t should equal the actual return realized at t+1.
    for t in data.index:
        nxt = ret.index[ret.index.get_loc(t) + 1]
        assert data.loc[t, F.TARGET_RET] == pytest.approx(ret.loc[nxt])


def test_walk_forward_splits_are_causal():
    for tr, te in fc.walk_forward_splits(n=1000, initial=500, step=50, embargo=1):
        assert tr.max() < te.min()             # train strictly before test
        assert te.min() - tr.max() > 1          # embargo gap respected


def test_walk_forward_covers_tail():
    seen = np.concatenate([te for _, te in
                           fc.walk_forward_splits(1000, 500, 50, 1)])
    assert seen.min() == 500 and seen.max() == 999


# learns real signal, ignores noise -----------------------------------

def _markov_sign_returns(n, stay=0.9, seed=0):
    """Returns whose SIGN persists with prob `stay` -> today predicts tomorrow."""
    rng = np.random.default_rng(seed)
    s = 1
    out = []
    for _ in range(n):
        if rng.random() > stay:
            s = -s
        out.append(s * abs(rng.normal(0, 0.02)))
    return np.array(out)


def test_learns_known_signal():
    prices = _prices(_markov_sign_returns(1600, stay=0.9, seed=1))
    res = fc.backtest_ticker(prices, "A", initial=600, step=60, use_arima=False)
    # A model that can read persistent sign should clear chance comfortably.
    assert res["direction"].loc["XGBoost", "hit_rate"] > 0.75


def test_no_spurious_signal_on_noise():
    rng = np.random.default_rng(2)
    prices = _prices(rng.normal(0, 0.02, 1600))   # i.i.d. -> unpredictable
    res = fc.backtest_ticker(prices, "A", initial=600, step=60, use_arima=False)
    hit = res["direction"].loc["XGBoost", "hit_rate"]
    assert 0.40 < hit < 0.60      # near coin-flip; a leak would push this up


# metric correctness --------------------------------------------------

def test_direction_metrics_perfect():
    y = pd.Series([1, 0, 1, 1, 0])
    m = fc.direction_metrics(y, y, proba=[0.9, 0.1, 0.8, 0.7, 0.2])
    assert m["hit_rate"] == 1.0 and m["auc"] == 1.0


def test_magnitude_metrics_perfect():
    y = pd.Series([0.01, -0.02, 0.03])
    m = fc.magnitude_metrics(y, y)
    assert m["rmse"] == pytest.approx(0.0) and m["dir_acc"] == 1.0


def test_transaction_costs_reduce_return():
    rng = np.random.default_rng(3)
    ret = pd.Series(rng.normal(0, 0.02, 500))
    pos = pd.Series(rng.choice([-1.0, 1.0], 500))   # heavy turnover
    cheap = fc.strategy_metrics(ret, pos, cost_bps=0)["cum_return"]
    pricey = fc.strategy_metrics(ret, pos, cost_bps=20)["cum_return"]
    assert pricey < cheap


def test_buy_and_hold_matches_underlying():
    rng = np.random.default_rng(4)
    ret = pd.Series(rng.normal(0.0005, 0.02, 500))
    pos = pd.Series(1.0, index=ret.index)
    m = fc.strategy_metrics(ret, pos, cost_bps=0)
    assert m["cum_return"] == pytest.approx((1 + ret).prod() - 1)
