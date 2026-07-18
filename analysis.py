import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

import config

TRADING_DAYS = config.TRADING_DAYS_PER_YEAR


# returns -------------------------------------------------------------

def daily_returns(prices, kind="log"):
    #Daily returns from a wide price frame (index=date, cols=tickers).

    prices = prices.sort_index()
    if kind == "log":
        rets = np.log(prices / prices.shift(1))
    elif kind == "simple":
        rets = prices.pct_change()
    else:
        raise ValueError("kind must be 'log' or 'simple'")
    return rets.iloc[1:]


# correlation ---------------------------------------------------------

def correlation_matrix(returns, window=None, as_of=None):
    #Correlation matrix over the trailing `window` days ending at `as_of`.

    r = returns if as_of is None else returns.loc[:as_of]
    if window is not None:
        r = r.tail(window)
    return r.corr(min_periods=max(2, (window or len(r)) // 2))


def rolling_mean_correlation(returns, window):
    #Time series of average pairwise (off-diagonal) correlation.

    cols = returns.columns
    k = len(cols)
    iu = np.triu_indices(k, k=1)  # upper-triangle indices, excl. diagonal
    out = {}
    for i in range(window, len(returns) + 1):
        w = returns.iloc[i - window:i]
        c = w.corr().values
        out[returns.index[i - 1]] = np.nanmean(c[iu])
    return pd.Series(out, name=f"mean_corr_{window}d")


def pair_correlation_series(returns, a, b, window):
    #Rolling correlation between two named tickers (for the app's drill-down).
    return returns[a].rolling(window).corr(returns[b]).rename(f"{a}~{b}_{window}d")


# ordering for the heatmap -------------------------------------------

def cluster_order(corr, method="average"):
    #Reorder tickers by hierarchical clustering on correlation distance.

    dist = (1.0 - corr).clip(lower=0.0).to_numpy(copy=True)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=method)
    order = leaves_list(Z)
    return [corr.columns[i] for i in order]


def sector_order(tickers, sectors=None):
    #Group tickers by sector label (the a-priori grouping, for comparison).
    sectors = sectors or config.SECTORS
    return sorted(tickers, key=lambda t: (sectors.get(t, "ZZ"), t))


# volatility ----------------------------------------------------------

def rolling_volatility(returns, window, annualize=True):
    #Rolling std of returns, annualized by sqrt(252) by default.
    vol = returns.rolling(window).std()
    return vol * np.sqrt(TRADING_DAYS) if annualize else vol


def annualized_volatility(returns):
    #Full-sample annualized volatility per ticker.
    return returns.std() * np.sqrt(TRADING_DAYS)


# --- tail risk -----------------------------------------------------------

def historical_var(returns, alpha=0.05, horizon=1):
    #Historical VaR as a POSITIVE loss fraction.

    q = returns.quantile(alpha)
    return (-q * np.sqrt(horizon)).rename(f"VaR_{int((1 - alpha) * 100)}")


def historical_cvar(returns, alpha=0.05):
    #Conditional VaR / expected shortfall: mean loss in the worst alpha tail.

    def _es(s):
        s = s.dropna()
        cutoff = s.quantile(alpha)
        tail = s[s <= cutoff]
        return -tail.mean() if len(tail) else np.nan
    return returns.apply(_es).rename(f"CVaR_{int((1 - alpha) * 100)}")


def parametric_var(returns, alpha=0.05):
    #Gaussian (variance-covariance) VaR, for comparison with historical.


    from scipy.stats import norm
    z = norm.ppf(alpha)
    var = -(returns.mean() + z * returns.std())
    return var.rename(f"paramVaR_{int((1 - alpha) * 100)}")


# --- drawdown ------------------------------------------------------------

def max_drawdown(prices):
    #Worst peak-to-trough decline per ticker, as a positive fraction.
    
    prices = prices.sort_index()
    running_max = prices.cummax()
    drawdown = prices / running_max - 1.0
    return (-drawdown.min()).rename("max_drawdown")


# --- summary table -------------------------------------------------------

def risk_summary(prices, alpha=0.05):
    #One row per ticker: the numbers that go straight into the risk table.

    simple = daily_returns(prices, kind="simple")
    ann_ret = simple.mean() * TRADING_DAYS
    ann_vol = simple.std() * np.sqrt(TRADING_DAYS)
    sharpe = ann_ret / ann_vol.replace(0, np.nan)

    out = pd.DataFrame({
        "sector": pd.Series({t: config.SECTORS.get(t, "?") for t in prices.columns}),
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        historical_var(simple, alpha).name: historical_var(simple, alpha),
        historical_cvar(simple, alpha).name: historical_cvar(simple, alpha),
        "max_drawdown": max_drawdown(prices),
    })
    return out.sort_values("ann_vol", ascending=False)
