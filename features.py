import numpy as np
import pandas as pd

# Feature spec (kept here so the app and tests share one source of truth).
LAGS = (0, 1, 2, 4)          # 0 = most recent completed day
VOL_WINDOW = 21
MOM_WINDOWS = (5, 10, 21)
Z_WINDOW = 21

TARGET_RET = "target_ret"
TARGET_DIR = "target_dir"


def feature_columns():
    cols = [f"ret_lag_{L}" for L in LAGS]
    cols += [f"vol_{VOL_WINDOW}"]
    cols += [f"mom_{w}" for w in MOM_WINDOWS]
    cols += [f"z_{Z_WINDOW}"]
    return cols


def build_matrix(prices, ticker):
    """Full feature+target frame for one ticker.

    Feature columns are trailing-only (no lookahead). target_ret is next-day
    return; it is NaN on the final row because tomorrow isn't known yet. Warm-up
    rows (insufficient history for a rolling window) have NaN features. Nothing
    is dropped here — callers decide.
    """
    px = prices[ticker].dropna().sort_index()
    ret = px.pct_change()

    df = pd.DataFrame(index=px.index)
    for L in LAGS:
        df[f"ret_lag_{L}"] = ret.shift(L)                  # trailing: known at t
    df[f"vol_{VOL_WINDOW}"] = ret.rolling(VOL_WINDOW).std()
    for w in MOM_WINDOWS:
        df[f"mom_{w}"] = px.pct_change(w)                  # trailing momentum
    ma = px.rolling(Z_WINDOW).mean()
    sd = px.rolling(Z_WINDOW).std()
    df[f"z_{Z_WINDOW}"] = (px - ma) / sd                   # mean-reversion z-score

    df[TARGET_RET] = ret.shift(-1)                         # NEXT day's return
    df[TARGET_DIR] = (df[TARGET_RET] > 0).astype("float")  # NaN-safe; cast later
    return df


def make_features(prices, ticker):
    """Model-ready frame: features + targets, fully labeled rows only."""
    df = build_matrix(prices, ticker)
    labeled = df.dropna().copy()
    labeled[TARGET_DIR] = labeled[TARGET_DIR].astype(int)
    return labeled


def latest_feature_row(prices, ticker):
    """Most recent row with complete features but an UNKNOWN target.

    This is the row you'd actually act on: all features available at today's
    close, predicting tomorrow. Returns (date, Series of features) or None.
    """
    df = build_matrix(prices, ticker)
    feats = df[feature_columns()]
    complete = feats.dropna()
    if complete.empty:
        return None
    last_date = complete.index[-1]
    # Only "actionable" if its target is unknown (i.e. it's the final bar).
    if not np.isnan(df.loc[last_date, TARGET_RET]):
        # All rows are labeled (shouldn't happen with shift(-1)); no live row.
        return last_date, complete.iloc[-1]
    return last_date, complete.loc[last_date]
