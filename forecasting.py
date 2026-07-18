
import warnings

import numpy as np
import pandas as pd

import features as F

TRADING_DAYS = 252

# Shallow, regularized trees: financial signal is faint and noisy, so we bias

XGB_PARAMS = dict(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    n_jobs=2, random_state=0,
)


# models --------------------------------------------------------------

def _xgb_classifier():
    from xgboost import XGBClassifier
    return XGBClassifier(eval_metric="logloss", **XGB_PARAMS)


def _xgb_regressor():
    from xgboost import XGBRegressor
    return XGBRegressor(**XGB_PARAMS)


# walk-forward split --------------------------------------------------

def walk_forward_splits(n, initial, step, embargo=1):
    """Yield (train_slice, test_slice) index arrays.

    train = [0, i - embargo); test = [i, i + step). Expanding train window.
    """
    i = initial
    idx = np.arange(n)
    while i < n:
        train = idx[: max(0, i - embargo)]
        test = idx[i: min(i + step, n)]
        if len(test) == 0:
            break
        yield train, test
        i += step


# baselines -----------------------------------------------------------

def _arima_block_forecast(train_ret, horizon):
    """Fit a small ARIMA on training returns, forecast `horizon` steps.

    On daily returns ARIMA collapses toward the mean fast — that's the point of
    including it: it's the classical baseline the ML model must beat.
    """
    from statsmodels.tsa.arima.model import ARIMA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = ARIMA(train_ret.values, order=(1, 0, 1)).fit()
            return np.asarray(res.forecast(steps=horizon))
        except Exception:
            return np.repeat(train_ret.mean(), horizon)  # graceful fallback


# backtest ------------------------------------------------------------

def backtest_ticker(prices, ticker, initial=756, step=21, embargo=1,
                    cost_bps=5.0, use_arima=True):
    """Run the full walk-forward backtest for one ticker.

    Returns a dict with an out-of-sample predictions frame plus direction,
    magnitude, and strategy metric tables.
    """
    data = F.make_features(prices, ticker)
    if len(data) <= initial + step:
        raise ValueError(f"{ticker}: not enough history for the backtest window")

    X = data[F.feature_columns()]
    y_ret = data[F.TARGET_RET]
    y_dir = data[F.TARGET_DIR]
    dates = data.index

    rows = []
    for tr, te in walk_forward_splits(len(data), initial, step, embargo):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        clf = _xgb_classifier().fit(Xtr, y_dir.iloc[tr])
        reg = _xgb_regressor().fit(Xtr, y_ret.iloc[tr])

        proba = clf.predict_proba(Xte)[:, 1]
        xgb_dir = (proba > 0.5).astype(int)
        xgb_ret = reg.predict(Xte)

        # Naive persistence: tomorrow looks like the most recent completed day.
        naive_ret = Xte["ret_lag_0"].values
        naive_dir = (naive_ret > 0).astype(int)

        arima_ret = (_arima_block_forecast(y_ret.iloc[tr], len(te))
                     if use_arima else np.full(len(te), np.nan))

        block = pd.DataFrame({
            "actual_ret": y_ret.iloc[te].values,
            "actual_dir": y_dir.iloc[te].values,
            "xgb_proba": proba, "xgb_dir": xgb_dir, "xgb_ret": xgb_ret,
            "naive_ret": naive_ret, "naive_dir": naive_dir,
            "arima_ret": arima_ret,
        }, index=dates[te])
        rows.append(block)

    oos = pd.concat(rows)

    dir_tbl = pd.DataFrame({
        "XGBoost": direction_metrics(oos["actual_dir"], oos["xgb_dir"], oos["xgb_proba"]),
        "Naive": direction_metrics(oos["actual_dir"], oos["naive_dir"]),
    }).T
    mag_models = {"XGBoost": "xgb_ret", "Naive": "naive_ret"}
    if use_arima:
        mag_models["ARIMA"] = "arima_ret"
    mag_tbl = pd.DataFrame({
        name: magnitude_metrics(oos["actual_ret"], oos[col])
        for name, col in mag_models.items()
    }).T

    # Strategies: position from each model's predicted direction, ±1 long/short.
    strat_tbl = pd.DataFrame({
        "XGBoost L/S": strategy_metrics(oos["actual_ret"], _pos(oos["xgb_dir"]), cost_bps),
        "Naive L/S": strategy_metrics(oos["actual_ret"], _pos(oos["naive_dir"]), cost_bps),
        "Buy & Hold": strategy_metrics(oos["actual_ret"], pd.Series(1.0, index=oos.index), cost_bps),
    }).T

    return {"ticker": ticker, "oos": oos,
            "direction": dir_tbl, "magnitude": mag_tbl, "strategy": strat_tbl}


def _pos(dir_series):
    """Map {0,1} direction predictions to {-1,+1} positions."""
    return dir_series.replace({0: -1, 1: 1}).astype(float)


# --- metrics -------------------------------------------------------------

def direction_metrics(y_dir, pred_dir, proba=None):
    y_dir = np.asarray(y_dir); pred_dir = np.asarray(pred_dir)
    out = {"hit_rate": float((y_dir == pred_dir).mean()),
           "base_rate_up": float(y_dir.mean()), "n": int(len(y_dir))}
    if proba is not None and len(np.unique(y_dir)) > 1:
        from sklearn.metrics import roc_auc_score
        out["auc"] = float(roc_auc_score(y_dir, np.asarray(proba)))
    return out


def magnitude_metrics(y_ret, pred_ret):
    y = pd.Series(y_ret).reset_index(drop=True)
    p = pd.Series(pred_ret).reset_index(drop=True)
    err = y - p
    ss_res = float((err ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mae": float(err.abs().mean()),
        "dir_acc": float((np.sign(p) == np.sign(y)).mean()),
        "r2": (1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,  # usually ~0 or <0
    }


def strategy_metrics(actual_ret, position, cost_bps=5.0):
    """Net-of-cost strategy performance. position aligned to actual_ret."""
    actual_ret = pd.Series(actual_ret)
    position = pd.Series(position).reindex(actual_ret.index).fillna(0.0)
    turnover = position.diff().abs()
    turnover.iloc[0] = abs(position.iloc[0])
    net = position * actual_ret - (cost_bps / 1e4) * turnover
    ann = np.sqrt(TRADING_DAYS)
    eq = (1 + net).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    sd = net.std()
    return {
        "sharpe": float(net.mean() / sd * ann) if sd > 0 else np.nan,
        "ann_return": float(net.mean() * TRADING_DAYS),
        "ann_vol": float(sd * ann),
        "cum_return": float(eq.iloc[-1] - 1),
        "max_dd": -dd,
        "hit_rate": float((net > 0).mean()),
        "avg_turnover": float(turnover.mean()),
    }
