# Methodology & Limitations

This document explains what the dashboard does, the choices behind it, and —
just as importantly — where it falls short. The project's thesis is deliberate:
**rigorous, honest evaluation beats a good-looking backtest.** Daily equity
returns are close to unpredictable, so the goal here is to build the measurement
apparatus correctly and report what it finds, not to manufacture alpha.

---

## 1. Data

**Source and scope.** End-of-day OHLCV for 19 instruments (tech/semis megacaps,
plus financials, energy, staples, healthcare, a utility, and two contrast
anchors — `TLT` long Treasuries and `GLD` gold), from 2018 to present via
`yfinance`. Each ticker carries ~2,150 trading days.

**Why this basket.** The anchors are the point. Without something structurally
decoupled from equities, a correlation heatmap of megacaps is uniformly high and
says nothing. `TLT` and `GLD` give the clustering something real to separate, so
the analysis demonstrably recovers structure rather than decorating noise.

**The data-delay caveat (stated plainly).** `yfinance` scrapes Yahoo Finance's
unofficial endpoints. The data is **end-of-day, delayed, and not guaranteed**,
and Yahoo rate-limits or temporarily blocks aggressive access — especially from
the shared IPs that host platforms like Streamlit Community Cloud use. This is a
structural constraint, not a bug to be fixed. The architecture responds to it:

- The app ships a **committed seed database** (`data/prices.db`) and renders off
  it, so it never depends on a live pull to work.
- Refresh is **on demand** (a button), with retry + exponential backoff, and it
  **falls back to cached data** when Yahoo throttles. A failed refresh is logged,
  never fatal.
- A `refresh_log` table records the status of every fetch so freshness is
  visible rather than assumed.

This is not a real-time tool, and the app says so.

**Storage.** SQLite, with a `(ticker, date)` primary key so writes are
idempotent — re-running a refresh never duplicates rows, and re-pulling the most
recent day to capture late split/dividend adjustments is safe. Refreshes are
incremental (only dates after each ticker's last stored bar), which keeps request
counts low and reduces rate-limiting. Both raw `close` and `adj_close` are
stored; all return calculations use `adj_close` so dividends and splits don't
appear as phantom jumps.

---

## 2. Correlation & risk analysis

**Return convention.** Log returns for correlation and volatility (they're
time-additive, the standard for second-moment estimation); simple returns for
anything that represents actual P&L (VaR, Sharpe, drawdown, strategy returns). At
daily frequency the two differ by basis points, but the choice is deliberate, not
incidental.

**Rolling correlation.** 30/60/90-day windows, computed on pairwise-complete
observations so a ticker with a shorter history doesn't invalidate the whole
matrix. The heatmap is ordered by **hierarchical clustering** on correlation
distance (`1 − ρ`), which recovers the block structure directly from the data —
the semis (`NVDA`/`AMD`/`AVGO`) cluster tightly, the megacap-tech block glows,
energy (`XOM`/`CVX`) forms its own block, and the defensives and anchors peel
off. That's a stronger, more honest grouping than sorting by a sector label I
assigned by hand.

**Market cohesion.** The mean pairwise correlation over time is a single line
that summarizes "how much everything is moving together." It behaves exactly as
market history would predict — spiking toward crisis co-movement in early 2020,
elevated through the 2022–2023 rate shock, and decaying toward a low-correlation
regime as markets calmed. This is a useful sanity check that the pipeline is
measuring something real.

**Volatility and tail risk.** Annualized volatility is `σ_daily × √252`. Tail
risk is reported three ways on purpose:

- **Historical VaR (95%, 1-day)** — the empirical 5th percentile of returns. No
  distributional assumption.
- **Parametric (Gaussian) VaR** — shown alongside historical specifically to
  expose the gap: the normal assumption *underestimates* the fat, left-skewed
  tails equities actually have.
- **CVaR / expected shortfall** — the average loss in the worst 5% of days, which
  says how bad the tail is *once you're past* VaR (VaR alone is silent on
  severity).

Max drawdown (worst peak-to-trough) rounds out the picture. The risk table makes
the risk/return spread concrete: e.g., `AMD` carries the basket's highest
volatility (~56% annualized) and a ~65% max drawdown — a genuinely different risk
object from a defensive name like `KO` or the `TLT` anchor.

---

## 3. Forecasting

**Targets.** Two, per ticker: next-day return **direction** (up/down,
classification) and **magnitude** (the return itself, regression).

**Features (leak-proof by construction).** Lagged returns (`t, t-1, t-2, t-4`),
21-day rolling volatility, momentum over 5/10/21 days, and a 21-day
mean-reversion z-score. Every feature at day `t` uses only data available at the
close of `t`; the target is `t+1`'s return. The no-lookahead property isn't just
asserted — it's **enforced by a test** that checks features for a given date are
bit-for-bit identical whether computed on the full history or a history truncated
at that date. If a feature could see the future, that test fails.

**Model.** XGBoost, intentionally shallow and regularized (`max_depth=3`,
`learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1`,
200 trees). Financial signal is faint and the noise is enormous, so the model is
biased hard toward underfitting — a deep tree would memorize noise and look great
in-sample while failing out-of-sample.

**Baselines (not strawmen).** XGBoost is measured against:
- **Naive persistence** — tomorrow's direction equals today's; tomorrow's return
  equals today's. This is the honest bar. For a process with any positive
  autocorrelation, persistence is close to the *optimal* rule, so beating it is
  genuinely hard.
- **ARIMA(1,0,1)** — the classical time-series baseline. On daily returns it
  collapses toward the mean, which is exactly why it belongs here: it shows what
  "no real signal" looks like.

**Evaluation protocol.** Expanding-window **walk-forward**, never a single split:
train on all history up to a point, predict the next ~21-day block out-of-sample,
step forward, retrain, repeat. A random train/test split would leak future regime
information into training and is the single most common way these projects lie to
themselves. An **embargo** of one day is dropped between train and test because
the label is next-day return — without it, the last training example's label
would fall on the first test day, a subtle boundary leak.

**Metrics.**
- *Direction:* hit rate, AUC, and the base rate of up-days (so a model that just
  predicts "up" can't masquerade as skill).
- *Magnitude:* RMSE, MAE, directional accuracy, and R². R² near or below zero is
  the expected, honest result — it means next-day return magnitude is essentially
  unpredictable and the model does no better than guessing the mean.
- *Strategy:* a ±1 long/short position from the predicted direction, with Sharpe,
  cumulative return, max drawdown, and turnover — all **net of a 5bps
  per-turnover transaction cost** (a full position flip therefore costs ~10bps).
  Without costs, a daily-flipping strategy looks great and means nothing.

---

## 4. Results

_Fill these from the `precompute.py` summary (averages across the 19 tickers)._

| Metric | XGBoost | Naive | Buy & Hold |
|---|---|---|---|
| Avg direction hit rate | ____ | ____ | — |
| Tickers where XGBoost > naive | ___ / 19 | — | — |
| Avg strategy Sharpe (net) | ____ | ____ | ____ |

**Honest reading.** The expected — and defensible — outcome is that XGBoost's hit
rate sits near 50%, its edge over naive is small or negative, and its net-of-cost
Sharpe does not reliably beat a one-line persistence rule or simple buy-and-hold.
Magnitude R² is at or below zero across the basket. **This is the finding, not a
failure.** It is consistent with weak-form market efficiency at daily horizons:
if a gradient-boosted model on price-derived features could reliably call
tomorrow's direction after costs, that edge would already be arbitraged away. A
model that quietly reported a 65% hit rate would be far more likely to be leaking
than to be right.

---

## 5. Limitations & failure modes

**Data.** The source is unofficial and delayed. The basket is the *current*
membership, so results carry **survivorship bias** — no delisted or acquired
names are included, which flatters aggregate returns. There is no intraday data,
no fundamentals, and corporate-action handling is limited to what `adj_close`
encodes.

**Model.** Features are purely price-derived — no volume, macro, options-implied
vol, or sentiment, all of which carry information this model can't see. Each
ticker has its own model, ignoring cross-sectional structure (relative value,
lead-lag between correlated names). The models are **not hyperparameter-tuned**;
parameters are fixed and conservative, which avoids overfitting through tuning but
also means the model is untuned. And markets are non-stationary: a relationship
learned on 2018–2021 may simply not hold in a later regime.

**Backtest realism.** The 5bps cost is a simplification — it ignores bid-ask
spread, slippage, market impact, and the borrow cost of shorting. Fills are
assumed at the close. There's no position sizing or risk management, and no
capacity analysis. Most importantly, evaluating 19 tickers invites
**multiple-testing bias**: the best-looking ticker's Sharpe is partly luck, and
should not be cherry-picked as evidence of skill.

**Statistics.** Hit rates are not tested for significance against 50% — a hit
rate of 0.51 over a few thousand days may not be distinguishable from a coin flip.
Walk-forward reduces leakage but doesn't prove its absence. VaR is 1-day and
historical, so it assumes the past return distribution repeats and says nothing
about a shock outside the sample.

---

## 6. What I'd do next

- **Significance testing:** a stationary bootstrap on the return series, and a
  deflated Sharpe ratio to correct for the 19-way multiple testing, to ask
  whether any apparent edge survives scrutiny.
- **Cross-sectional model:** pool tickers and add relative-value / lead-lag
  features instead of 19 isolated models.
- **Richer signal:** volume, macro series, and options-implied volatility.
- **A real cost model:** spread- and impact-aware, with borrow cost for shorts.
- **Regime awareness:** label volatility/correlation regimes and evaluate
  conditionally, since a single blended metric hides regime-dependent behavior.
- **A licensed data vendor** to remove the delay/reliability caveat entirely.

---

## Reproducibility

```bash
pip install -r requirements.txt
python seed_db.py        # build data/prices.db
python precompute.py     # run all backtests -> data/backtests.pkl
streamlit run app.py     # launch the dashboard
pytest -q                # 28 tests: storage, resilience, analysis, leakage guards
```

The leakage guards in `tests/test_forecasting.py` are the load-bearing tests: one
proves features are causal, another proves that on i.i.d. noise the out-of-sample
hit rate stays at ~50% — if the pipeline were leaking, that test would fail.
