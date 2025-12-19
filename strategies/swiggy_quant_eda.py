"""
SWIGGY.BO Quant EDA Dashboard (Plotly + Streamlit)
==================================================

Deep exploratory & risk analysis for SWIGGY.BO (Swiggy) using the existing
Quant Finance stack (Stock class + indicators) with Plotly visuals.

Run with:

    cd /Users/inavlabs/Documents/inavlabs/quant_finance
    streamlit run strategies/swiggy_quant_eda.py
"""

import sys
from pathlib import Path
from typing import List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from assets.stock import Stock
from indicators.technical_indicators import (
    calculate_sma,
    calculate_bollinger_bands,
    calculate_rsi,
    calculate_macd,
)
from telemetries.logger import logger


DEFAULT_SYMBOL = "SWIGGY.BO"


st.set_page_config(
    page_title="Swiggy Quant EDA",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Swiggy (SWIGGY.BO) Quant EDA Dashboard")
st.markdown(
    "Detailed exploratory analysis and risk metrics using Plotly. "
    "Built on the same Quant Finance backend components (Stock, indicators, telemetry)."
)


# Sidebar configuration
st.sidebar.header("Data Configuration")
symbol = st.sidebar.text_input("Symbol", value=DEFAULT_SYMBOL).upper()

period = st.sidebar.selectbox(
    "Lookback Period",
    ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
    index=2,
)

interval = st.sidebar.selectbox(
    "Data Interval",
    ["1d", "1h", "30m", "15m", "5m"],
    index=1,
)

real_time = st.sidebar.checkbox("Enable real-time refresh", value=True)
refresh_seconds = st.sidebar.slider("Refresh interval (seconds)", 1, 30, 5)

if real_time and hasattr(st, "autorefresh"):
    st.autorefresh(interval=refresh_seconds * 1000, key="swiggy_quant_eda_autorefresh")


@st.cache_data(show_spinner=False)
def load_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    stock = Stock(symbol)
    if not stock.validate():
        raise ValueError(f"Invalid symbol: {symbol}")
    df = stock.get_historical_data(period=period, interval=interval)
    return df


try:
    with st.spinner(f"Fetching data for {symbol}..."):
        df = load_data(symbol, period, interval)
except Exception as exc:
    logger.error(f"Error loading data for {symbol}: {exc}")
    st.error(f"Error loading data for {symbol}: {exc}")
    st.stop()

if df.empty:
    st.error("No data returned. Try another symbol, period or interval.")
    st.stop()


st.subheader("Overview & Descriptive Statistics")

df = df.copy()
df["Return"] = df["Close"].pct_change()
df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))

last_close = float(df["Close"].iloc[-1])
first_close = float(df["Close"].iloc[0])
tot_ret = (last_close - first_close) / first_close * 100.0 if first_close != 0 else 0.0
ann_vol = df["LogReturn"].std(ddof=1) * np.sqrt(252) * 100.0

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Last Close", f"{last_close:.2f}")
with c2:
    st.metric("Period Return %", f"{tot_ret:+.2f}%")
with c3:
    st.metric("Ann. Volatility % (log returns)", f"{ann_vol:.2f}%")
with c4:
    st.metric("Observations", len(df))

st.write("**Return distribution (daily / interval returns)**")

hist_fig = go.Figure()
hist_fig.add_trace(
    go.Histogram(
        x=df["Return"].dropna(),
        nbinsx=50,
        name="Returns",
        marker_color="deepskyblue",
        opacity=0.75,
    )
)
hist_fig.update_layout(
    height=400,
    bargap=0.05,
    title="Histogram of Returns",
    xaxis_title="Return",
    yaxis_title="Frequency",
)
st.plotly_chart(hist_fig, width="stretch")


st.subheader("Price, Volume & Indicators")

# Indicators
df["SMA_9"] = calculate_sma(df["Close"], 9)
df["SMA_21"] = calculate_sma(df["Close"], 21)
bb = calculate_bollinger_bands(df["Close"], period=20, std_dev=2.0)
df["BB_Upper"] = bb["Upper"]
df["BB_Lower"] = bb["Lower"]
df["RSI_14"] = calculate_rsi(df["Close"], 14)
macd_df = calculate_macd(df["Close"])

fig_price = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.5, 0.25, 0.25],
    subplot_titles=["Price & SMAs / Bollinger Bands", "Volume", "RSI (14) & MACD"],
)

# Price + SMAs + Bollinger Bands
fig_price.add_trace(
    go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="OHLC",
        increasing_line_color="green",
        decreasing_line_color="red",
        showlegend=False,
    ),
    row=1,
    col=1,
)
fig_price.add_trace(
    go.Scatter(x=df.index, y=df["SMA_9"], name="SMA 9", line=dict(color="cyan")),
    row=1,
    col=1,
)
fig_price.add_trace(
    go.Scatter(x=df.index, y=df["SMA_21"], name="SMA 21", line=dict(color="orange")),
    row=1,
    col=1,
)
fig_price.add_trace(
    go.Scatter(
        x=df.index,
        y=df["BB_Upper"],
        name="BB Upper",
        line=dict(color="magenta", width=1, dash="dot"),
    ),
    row=1,
    col=1,
)
fig_price.add_trace(
    go.Scatter(
        x=df.index,
        y=df["BB_Lower"],
        name="BB Lower",
        line=dict(color="magenta", width=1, dash="dot"),
        fill="tonexty",
        fillcolor="rgba(255,0,255,0.05)",
    ),
    row=1,
    col=1,
)

# Volume
fig_price.add_trace(
    go.Bar(
        x=df.index,
        y=df["Volume"],
        name="Volume",
        marker_color="lightblue",
    ),
    row=2,
    col=1,
)

# RSI
fig_price.add_trace(
    go.Scatter(
        x=df.index, y=df["RSI_14"], name="RSI 14", line=dict(color="purple")
    ),
    row=3,
    col=1,
)
fig_price.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
fig_price.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

# MACD (overlay in same panel as RSI using secondary style)
fig_price.add_trace(
    go.Scatter(
        x=macd_df.index,
        y=macd_df["MACD"],
        name="MACD",
        line=dict(color="yellow"),
        opacity=0.7,
    ),
    row=3,
    col=1,
)
fig_price.add_trace(
    go.Scatter(
        x=macd_df.index,
        y=macd_df["Signal"],
        name="MACD Signal",
        line=dict(color="orange"),
        opacity=0.7,
    ),
    row=3,
    col=1,
)

fig_price.update_layout(
    height=850,
    hovermode="x unified",
    xaxis_rangeslider_visible=False,
)
fig_price.update_yaxes(title_text="Price", row=1, col=1)
fig_price.update_yaxes(title_text="Volume", row=2, col=1)
fig_price.update_yaxes(title_text="Indicator", row=3, col=1)

st.plotly_chart(fig_price, width="stretch")


st.subheader("Volatility Surface (3D) & Risk Metrics")

# Build realized volatility surface: time vs window size vs rolling vol
log_ret = df["LogReturn"].dropna()
windows = [5, 10, 20, 40]

vol_matrix = []
for w in windows:
    vol_series = log_ret.rolling(window=w).std() * np.sqrt(252) * 100.0
    vol_matrix.append(vol_series.values.tolist())

vol_matrix = np.array(vol_matrix)
valid_idx = ~np.isnan(vol_matrix).all(axis=0)
times = df.index[1:][valid_idx]  # align roughly with log_ret index
vol_matrix = vol_matrix[:, valid_idx]

X, Y = np.meshgrid(
    np.arange(len(times)), windows
)  # X is time index, Y is window size

surf_fig = go.Figure(
    data=[
        go.Surface(
            x=X,
            y=Y,
            z=vol_matrix,
            colorscale="Viridis",
            colorbar_title="Vol %",
        )
    ]
)
surf_fig.update_layout(
    title="Realized Volatility Surface (annualized %, windows vs time)",
    scene=dict(
        xaxis_title="Time index (recent → right)",
        yaxis_title="Window size (days)",
        zaxis_title="Volatility %",
    ),
    height=700,
)

left_col, right_col = st.columns([2, 1])
with left_col:
    st.plotly_chart(surf_fig, width="stretch")

# Basic risk metrics: VaR / CVaR (simple historical)
with right_col:
    st.markdown("**Historical Risk (1-day horizon)**")
    clean_ret = df["Return"].dropna()
    if not clean_ret.empty:
        alpha = 0.95
        var = np.percentile(clean_ret, (1 - alpha) * 100)
        cvar = clean_ret[clean_ret <= var].mean()

        st.write(f"**VaR {int(alpha*100)}%** (1-day): {var:.4f} ({var*100:.2f}%)")
        st.write(f"**CVaR {int(alpha*100)}%** (1-day): {cvar:.4f} ({cvar*100:.2f}%)")

        risk_fig = go.Figure()
        risk_fig.add_trace(
            go.Histogram(
                x=clean_ret,
                nbinsx=60,
                name="Returns",
                marker_color="steelblue",
            )
        )
        risk_fig.add_vline(
            x=var,
            line_color="red",
            line_dash="dash",
            annotation_text="VaR",
        )
        risk_fig.update_layout(
            title="Return Distribution with VaR Marker",
            xaxis_title="Return",
            yaxis_title="Frequency",
            height=350,
        )
        st.plotly_chart(risk_fig, width="stretch")
    else:
        st.info("Not enough data to compute risk metrics.")


