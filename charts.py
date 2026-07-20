"""Plotly figure builders. Pure functions: data in, styled go.Figure out.

Keeping these out of app.py means the visuals are unit-testable and the app
stays a thin wiring layer. Palette is defined once here and mirrored in
.streamlit/config.toml so the charts and the chrome agree.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- palette (quant-desk: ink navy, paper text, calm signal colors) ------
INK = "#0F1720"
SURFACE = "#17212B"
GRID = "#25313D"
TEXT = "#E6EDF3"
MUTED = "#8B98A5"
POS = "#3FB950"       # calm green
NEG = "#F85149"       # calm red
ACCENT = "#E0A458"    # muted amber highlight
NEUTRAL = "#4C9AFF"   # cool blue (negative side of corr scale)

MONO = "IBM Plex Mono, ui-monospace, monospace"

# Diverging scale for correlation: blue (neg) -> ink (0) -> amber/red (pos)
CORR_SCALE = [
    [0.0, "#2B6CB0"], [0.25, "#3B4A5A"], [0.5, "#1C2530"],
    [0.75, "#B9713B"], [1.0, "#E0A458"],
]


def _style(fig, height=420, title=None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=INK, plot_bgcolor=INK,
        font=dict(family=MONO, size=12, color=TEXT),
        title=dict(text=title, font=dict(size=14, color=TEXT)) if title else None,
        margin=dict(l=40, r=20, t=40 if title else 20, b=40),
        height=height,
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
    )
    return fig


# --- tab 1: correlation & risk ------------------------------------------

def correlation_heatmap(corr, order=None, title="Correlation"):
    if order is not None:
        corr = corr.loc[order, order]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=list(corr.columns), y=list(corr.index),
        zmin=-1, zmax=1, colorscale=CORR_SCALE,
        colorbar=dict(title="ρ", outlinewidth=0, tickfont=dict(color=MUTED)),
        hovertemplate="%{y} · %{x}<br>ρ = %{z:.2f}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    return _style(fig, height=560, title=title)


def cohesion_line(series):
    """Mean pairwise correlation over time — the 'everything moves together' line."""
    fig = go.Figure(go.Scatter(
        x=series.index, y=series.values, mode="lines",
        line=dict(color=ACCENT, width=1.6),
        hovertemplate="%{x|%Y-%m-%d}<br>mean ρ = %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=series.mean(), line=dict(color=MUTED, dash="dot", width=1))
    return _style(fig, height=260, title="Mean pairwise correlation")


def vol_bar(risk_df):
    """Annualized volatility per ticker, sorted, highest = most turbulent."""
    d = risk_df.sort_values("ann_vol")
    fig = go.Figure(go.Bar(
        x=d["ann_vol"].values, y=list(d.index), orientation="h",
        marker=dict(color=d["ann_vol"].values, colorscale="Oranges"),
        hovertemplate="%{y}<br>ann vol = %{x:.1%}<extra></extra>",
    ))
    fig.update_xaxes(tickformat=".0%")
    return _style(fig, height=520, title="Annualized volatility")


# --- tab 2: single-stock forecast ---------------------------------------

def equity_curves(oos, cost_bps=5.0):
    """Cumulative net-of-cost equity: XGBoost L/S vs Naive L/S vs Buy & Hold."""
    def curve(pos):
        pos = pd.Series(pos, index=oos.index).astype(float)
        turn = pos.diff().abs(); turn.iloc[0] = pos.iloc[0].__abs__()
        net = pos * oos["actual_ret"] - (cost_bps / 1e4) * turn
        return (1 + net).cumprod() - 1

    xgb = curve(oos["xgb_dir"].replace({0: -1, 1: 1}))
    naive = curve(oos["naive_dir"].replace({0: -1, 1: 1}))
    bh = (1 + oos["actual_ret"]).cumprod() - 1

    fig = go.Figure()
    for name, s, col in [("XGBoost L/S", xgb, ACCENT),
                         ("Naive L/S", naive, NEUTRAL),
                         ("Buy & Hold", bh, MUTED)]:
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=name, mode="lines",
                                 line=dict(color=col, width=1.6)))
    fig.update_yaxes(tickformat=".0%")
    return _style(fig, height=360, title="Out-of-sample cumulative return (net of costs)")


def rolling_hit_rate(oos, window=63):
    """Rolling share of correct direction calls; 0.5 line = coin flip."""
    correct = (oos["xgb_dir"] == oos["actual_dir"]).astype(float)
    roll = correct.rolling(window).mean()
    fig = go.Figure(go.Scatter(
        x=roll.index, y=roll.values, mode="lines",
        line=dict(color=POS, width=1.4),
        hovertemplate="%{x|%Y-%m-%d}<br>hit rate = %{y:.0%}<extra></extra>",
    ))
    fig.add_hline(y=0.5, line=dict(color=NEG, dash="dot", width=1))
    fig.update_yaxes(tickformat=".0%", range=[0.3, 0.7])
    return _style(fig, height=260, title=f"Rolling {window}-day direction hit rate")


# --- tab 3: cross-section ------------------------------------------------

def hit_rate_ranked(summary):
    """Per-ticker XGBoost hit rate vs the 50% chance line."""
    d = summary.sort_values("xgb_hit").copy()
    colors = [POS if v >= 0.5 else NEG for v in d["xgb_hit"]]
    fig = go.Figure(go.Bar(
        x=d["xgb_hit"].values, y=list(d.index), orientation="h",
        marker=dict(color=colors),
        hovertemplate="%{y}<br>hit rate = %{x:.1%}<extra></extra>",
    ))
    fig.add_vline(x=0.5, line=dict(color=MUTED, dash="dot", width=1))
    fig.update_xaxes(tickformat=".0%", range=[0.4, 0.6])
    return _style(fig, height=520, title="Next-day direction hit rate by ticker")


def sharpe_scatter(summary):
    """XGBoost strategy Sharpe vs Naive Sharpe. Points below y=x: naive wins."""
    lo = float(np.nanmin([summary["xgb_sharpe"].min(), summary["naive_sharpe"].min(), 0]))
    hi = float(np.nanmax([summary["xgb_sharpe"].max(), summary["naive_sharpe"].max(), 0]))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=summary["naive_sharpe"], y=summary["xgb_sharpe"], mode="markers+text",
        text=summary.index, textposition="top center",
        textfont=dict(color=MUTED, size=9),
        marker=dict(color=ACCENT, size=9),
        hovertemplate="%{text}<br>naive %{x:.2f} · xgb %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                             line=dict(color=MUTED, dash="dot", width=1),
                             showlegend=False, hoverinfo="skip"))
    fig.update_layout(xaxis_title="Naive Sharpe", yaxis_title="XGBoost Sharpe")
    return _style(fig, height=460, title="Model vs naive — Sharpe (net of costs)")
