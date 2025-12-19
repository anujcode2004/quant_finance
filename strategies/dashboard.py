"""
Market Dashboard Strategy
High-level dashboard with live price, fundamentals, technicals and news.

Run with: streamlit run strategies/dashboard.py
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import List

# Ensure project root is on sys.path for package imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from assets.stock import Stock
from assets.news import NewsService
from indicators.technical_indicators import (
    calculate_sma,
    calculate_rsi,
    calculate_macd,
)
from telemetries.logger import logger
from static_memory_cache import StaticMemoryCache


# Page config
st.set_page_config(
    page_title="Market Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Market Dashboard")
st.markdown("Live overview with price, fundamentals, technical indicators and news.")

# Sidebar controls
st.sidebar.header("Dashboard Settings")

symbol = st.sidebar.text_input("Primary Symbol", value="AAPL").upper()

period = st.sidebar.selectbox(
    "Lookback Period",
    ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"],
    index=3,
)

interval = st.sidebar.selectbox(
    "Data Interval",
    ["1m", "5m", "15m", "30m", "1h", "1d"],
    index=3,
)

real_time = st.sidebar.checkbox("Enable real-time refresh", value=True)
refresh_seconds = st.sidebar.slider("Refresh interval (seconds)", 1, 30, 5)

# Optional watchlist for quick switching
default_watchlist: List[str] = ["AAPL", "MSFT", "GOOG", "TSLA", "NVDA"]
watchlist = st.sidebar.multiselect(
    "Watchlist (click to load symbol)",
    default_watchlist,
    default=default_watchlist[:3],
)

clicked_symbol = st.sidebar.radio(
    "Quick load from watchlist",
    options=["(none)"] + watchlist,
    index=0,
)
if clicked_symbol != "(none)":
    symbol = clicked_symbol

# Handle auto-refresh if supported
if real_time and hasattr(st, "autorefresh"):
    st.autorefresh(interval=refresh_seconds * 1000, key="dashboard_autorefresh")


def render_price_section(stock: Stock, df: pd.DataFrame) -> None:
    """Render top price, fundamentals and overview cards."""
    company_info = stock.get_company_info()
    current_price = stock.get_current_price(refresh=True)

    # Basic metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Company", company_info.get("name", symbol))
    with col2:
        st.metric("Sector", company_info.get("sector", "N/A"))
    with col3:
        st.metric("Current Price", f"${current_price:.2f}")
    with col4:
        price_change = ((df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0]) * 100
        st.metric("Period Change", f"{price_change:.2f}%")

    # Second row of metrics
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Market Cap", f"{company_info.get('market_cap', 0):,}")
    with col6:
        st.metric("Volume (Last)", f"{df['Volume'].iloc[-1]:,}")
    with col7:
        st.metric("PE Ratio (TTM)", f"{company_info.get('pe_ratio', 'N/A')}")
    with col8:
        st.metric("EPS (TTM)", f"{company_info.get('eps', 'N/A')}")


def render_price_chart(df: pd.DataFrame, symbol: str) -> None:
    """Render main price chart with indicators."""
    # Use indicators config for default periods
    periods_cfg = StaticMemoryCache.get_indicators_config()
    sma_short_p = periods_cfg.get("sma_short", 9)
    sma_long_p = periods_cfg.get("sma_long", 21)

    df = df.copy()
    df["SMA_Short"] = calculate_sma(df["Close"], sma_short_p)
    df["SMA_Long"] = calculate_sma(df["Close"], sma_long_p)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{symbol} Price & SMAs", "Volume"),
    )

    fig.add_trace(
        go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="cyan")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["SMA_Short"], name=f"SMA {sma_short_p}", line=dict(color="orange")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["SMA_Long"], name=f"SMA {sma_long_p}", line=dict(color="magenta")),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color="lightblue"),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=600,
        showlegend=True,
        hovermode="x unified",
        xaxis_rangeslider_visible=True,
    )
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


def render_technical_section(df: pd.DataFrame) -> None:
    """Render RSI and MACD technical charts."""
    df = df.copy()
    df["RSI"] = calculate_rsi(df["Close"], 14)
    macd_df = calculate_macd(df["Close"])

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.5, 0.5],
        subplot_titles=("RSI", "MACD"),
    )

    # RSI
    fig.add_trace(
        go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="purple")),
        row=1,
        col=1,
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)

    # MACD
    fig.add_trace(
        go.Scatter(x=macd_df.index, y=macd_df["MACD"], name="MACD", line=dict(color="blue")),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=macd_df.index, y=macd_df["Signal"], name="Signal", line=dict(color="orange")),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=macd_df.index, y=macd_df["Histogram"], name="Histogram", marker_color="grey"),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=500,
        showlegend=True,
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=1, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


def render_news_section(symbol: str) -> None:
    """Render latest news for the symbol (if news provider configured)."""
    st.subheader("📰 Recent News")
    news_service = NewsService()
    articles = news_service.fetch_stock_news(symbol, max_results=5)

    if not articles:
        st.info("No news available or news provider not configured.")
        return

    for article in articles:
        with st.container(border=True):
            st.markdown(f"**{article.get('title', '')}**")
            st.caption(f"{article.get('source', '')} • {article.get('published_at', '')}")
            if article.get("description"):
                st.write(article["description"])
            if article.get("url"):
                st.markdown(f"[Open article]({article['url']})")


if symbol:
    try:
        logger.info(f"Dashboard loading for {symbol}")

        stock = Stock(symbol)
        if not stock.validate():
            st.error(f"Invalid symbol: {symbol}")
            st.stop()

        with st.spinner(f"Fetching data for {symbol}..."):
            df_prices = stock.get_historical_data(period=period, interval=interval)

        if df_prices.empty:
            st.error("No data available for this symbol and period.")
            st.stop()

        # Layout: price + fundamentals top, tabs below
        render_price_section(stock, df_prices)

        tab_price, tab_technical, tab_news = st.tabs(["Price", "Technical", "News"])

        with tab_price:
            render_price_chart(df_prices, symbol)

        with tab_technical:
            render_technical_section(df_prices)

        with tab_news:
            render_news_section(symbol)

        logger.success(f"Dashboard rendered for {symbol}")

    except Exception as exc:
        logger.error(f"Error in dashboard: {str(exc)}")
        st.error(f"Error loading dashboard: {exc}")


