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


def make_features(prices, ticker):
    """Build (features + targets) for one ticker from a wide price frame.

    Returns a DataFrame indexed by date with feature_columns() plus
    target_ret (next-day simple return) and target_dir (1 if up else 0).
    Rows with any NaN (warm-up period, final row with unknown future) dropped.
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

    # Targets: NEXT day's return. shift(-1) is the only forward-looking op, and
    # it's the label — never a feature.
    df[TARGET_RET] = ret.shift(-1)
    df[TARGET_DIR] = (df[TARGET_RET] > 0).astype(int)

    return df.dropna()
