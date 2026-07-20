import pickle

import pandas as pd
import streamlit as st

import config
import db
import analysis as an
import forecasting as fc
import charts as ch

st.set_page_config(page_title="Risk & Forecast Desk", layout="wide",
                   initial_sidebar_state="expanded")

# --- theme / type (config.toml handles base colors; this adds the fonts) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Mono', ui-monospace, monospace; }
h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; }
.block-container { padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1300px; }
[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; }
.deskhead { border-left: 3px solid #E0A458; padding: 2px 0 2px 14px; margin-bottom: .4rem; }
.deskhead h1 { margin: 0; font-size: 1.55rem; color: #E6EDF3; }
.deskhead p { margin: .15rem 0 0; color: #8B98A5; font-size: .82rem; }
.pill { display:inline-block; padding:2px 10px; border-radius:999px; font-size:.72rem;
        background:#17212B; border:1px solid #25313D; color:#8B98A5; }
</style>
""", unsafe_allow_html=True)

ARTIFACT = config.DATA_DIR / "backtests.pkl"


# --- cached loaders ------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def load_prices():
    return db.load_prices(config.DB_PATH, config.TICKERS)

@st.cache_data(ttl=3600, show_spinner=False)
def load_returns():
    return an.daily_returns(load_prices(), kind="log")

@st.cache_data(ttl=3600, show_spinner=False)
def load_risk():
    return an.risk_summary(load_prices())

@st.cache_data(ttl=3600, show_spinner=False)
def load_coverage():
    try:
        return db.coverage(config.DB_PATH), db.last_refresh(config.DB_PATH)
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def load_backtests():
    if not ARTIFACT.exists():
        return None
    with open(ARTIFACT, "rb") as f:
        return pickle.load(f)

@st.cache_data(ttl=3600, show_spinner="Backtesting…")
def backtest_one(ticker):
    return fc.backtest_ticker(load_prices(), ticker, use_arima=False)


# sidebar -------------------------------------------------------------

prices = load_prices()
cov, last_ref = load_coverage()
data_end = cov["end"].max() if not cov.empty else "—"

with st.sidebar:
    st.markdown("### Risk & Forecast Desk")
    st.caption(f"{len(config.TICKERS)} instruments · data through **{data_end}**")

    if st.button("Refresh prices", width='stretch'):
        import data_pipeline
        with st.spinner("Fetching latest bars (Yahoo may rate-limit)…"):
            res = data_pipeline.refresh()
        ok = int((res["status"] == "ok").sum())
        st.cache_data.clear()
        if ok:
            st.success(f"Updated {ok}/{len(res)} tickers.")
        else:
            st.warning("No live update — showing cached data.")
        st.rerun()

    st.divider()
    st.caption("**Data caveat**")
    st.caption("End-of-day prices via yfinance (unofficial, delayed, "
               "best-effort). Not real-time. The forecast is a methodology "
               "demo, not investment advice.")

if prices.empty:
    st.markdown('<div class="deskhead"><h1>Risk & Forecast Desk</h1>'
                '<p>No data yet.</p></div>', unsafe_allow_html=True)
    st.info("The price database is empty. Run `python seed_db.py` locally, "
            "commit `data/prices.db`, and redeploy.")
    st.stop()

returns = load_returns()

# header --------------------------------------------------------------

st.markdown(
    '<div class="deskhead"><h1>Multi-Stock Risk &amp; Forecasting Desk</h1>'
    '<p>Rolling correlation &amp; risk structure, with a next-day forecast held '
    'honestly against naive and ARIMA baselines.</p></div>',
    unsafe_allow_html=True)
st.markdown(f'<span class="pill">data through {data_end}</span>', unsafe_allow_html=True)
st.write("")

tab1, tab2, tab3 = st.tabs(["Correlation & risk", "Stock forecast", "Backtest performance"])

# TAB 1 — correlation & risk

with tab1:
    c = st.columns([1, 1, 2])
    window = c[0].selectbox("Correlation window", config.CORR_WINDOWS,
                            index=len(config.CORR_WINDOWS) - 1,
                            format_func=lambda w: f"{w} days")
    ordering = c[1].radio("Ordering", ["Cluster", "Sector"], horizontal=True)

    corr = an.correlation_matrix(returns, window=window)
    order = (an.cluster_order(corr) if ordering == "Cluster"
             else an.sector_order(list(corr.columns)))

    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(
            ch.correlation_heatmap(corr, order, f"{window}-day correlation ({ordering.lower()}-ordered)"),
            width='stretch')
        st.plotly_chart(ch.cohesion_line(an.rolling_mean_correlation(returns, window)),
                        width='stretch')
    with right:
        st.plotly_chart(ch.vol_bar(load_risk()), width='stretch')

    st.markdown("##### Risk summary")
    risk = load_risk()
    show = risk.rename(columns={
        "ann_return": "ann. return", "ann_vol": "ann. vol", "sharpe": "Sharpe",
        "max_drawdown": "max DD"})
    st.dataframe(
        show.style.format({
            "ann. return": "{:.1%}", "ann. vol": "{:.1%}", "Sharpe": "{:.2f}",
            "VaR_95": "{:.2%}", "CVaR_95": "{:.2%}", "max DD": "{:.1%}"}),
        width='stretch')
    st.caption("VaR/CVaR are 1-day, 95%. Historical (empirical), so no normality "
               "assumption. Sharpe assumes a zero risk-free rate.")

# TAB 2 — single-stock forecast

with tab2:
    bt = load_backtests()
    ticker = st.selectbox("Instrument", config.TICKERS)

    res = bt["results"].get(ticker) if bt else None
    if res is None:
        res = backtest_one(ticker)
    sig = res.get("latest_signal") or fc.latest_signal(prices, ticker)

    m = st.columns(4)
    if sig:
        arrow = "▲" if sig["direction"] == "up" else "▼"
        m[0].metric(f"Next-day call ({sig['as_of']})", f"{arrow} {sig['direction']}")
        m[1].metric("Model confidence", f"{sig['proba_up']*100:.0f}% up",
                    help="Classifier probability. ~50% means no real conviction.")
        m[2].metric("Predicted return", f"{sig['pred_ret']*100:+.2f}%")
    dirm = res["direction"]
    m[3].metric("Hit rate (OOS)", f"{dirm.loc['XGBoost','hit_rate']*100:.1f}%",
                delta=f"{(dirm.loc['XGBoost','hit_rate']-dirm.loc['Naive','hit_rate'])*100:+.1f} vs naive")

    if abs(sig["proba_up"] - 0.5) < 0.05 if sig else False:
        st.info("The model is near a coin-flip on tomorrow — low conviction. "
                "That's the honest and common case for daily equity direction.")

    st.plotly_chart(ch.equity_curves(res["oos"]), width='stretch')
    st.plotly_chart(ch.rolling_hit_rate(res["oos"]), width='stretch')

    a, b = st.columns(2)
    a.markdown("###### Direction — model vs baseline")
    a.dataframe(res["direction"].style.format({
        "hit_rate": "{:.3f}", "base_rate_up": "{:.3f}", "auc": "{:.3f}", "n": "{:.0f}"}),
        width='stretch')
    b.markdown("###### Magnitude — RMSE, R² (usually ≤0)")
    b.dataframe(res["magnitude"].style.format({
        "rmse": "{:.4f}", "mae": "{:.4f}", "dir_acc": "{:.3f}", "r2": "{:.3f}"}),
        width='stretch')


# TAB 3 — cross-section

with tab3:
    bt = load_backtests()
    if bt is None:
        st.info("No precomputed backtests found. Run `python precompute.py` "
                "locally and commit `data/backtests.pkl`, or explore individual "
                "tickers in the Stock forecast tab.")
    else:
        s = bt["summary"]
        k = st.columns(4)
        k[0].metric("Avg XGBoost hit rate", f"{s['xgb_hit'].mean()*100:.1f}%")
        k[1].metric("Avg naive hit rate", f"{s['naive_hit'].mean()*100:.1f}%")
        k[2].metric("XGBoost beats naive", f"{int((s['xgb_hit']>s['naive_hit']).sum())}/{len(s)}")
        k[3].metric("Avg XGBoost Sharpe", f"{s['xgb_sharpe'].mean():.2f}")

        st.markdown(
            "> **Read this honestly.** If XGBoost sits near the 50% line and its "
            "Sharpe clusters around naive's, that's the expected result for "
            "next-day equity direction after costs — reported, not hidden.")

        left, right = st.columns(2)
        left.plotly_chart(ch.hit_rate_ranked(s), width='stretch')
        right.plotly_chart(ch.sharpe_scatter(s), width='stretch')

        st.markdown("##### Per-ticker results")
        st.dataframe(s.style.format({
            "xgb_hit": "{:.3f}", "naive_hit": "{:.3f}", "xgb_auc": "{:.3f}",
            "xgb_sharpe": "{:.2f}", "naive_sharpe": "{:.2f}", "bh_sharpe": "{:.2f}"}),
            width='stretch')
        st.caption(f"Backtests built {bt.get('built_at','—')[:10]} · "
                   f"walk-forward, expanding window, net of 5bps/turnover.")
