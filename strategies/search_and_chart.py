"""
Yahoo Search + Stock Chart Streamlit App
========================================

This app lets you:
- Search for tickers via Yahoo Finance's public search endpoint.
- Inspect the results (symbol, name, exchange, sector, industry).
- Select a symbol and visualize its price & volume history using yfinance
  through the existing Stock class.

Run with:

    cd /Users/inavlabs/Documents/inavlabs/quant_finance
    streamlit run strategies/search_and_chart.py
"""

import sys
from pathlib import Path
from typing import List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from assets.stock import Stock
from services.yahoo_search_service import get_yahoo_search_service, YahooQuoteResult
from telemetries.logger import logger


st.set_page_config(
    page_title="Ticker Search & Chart",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Yahoo Search + Stock Chart")
st.markdown(
    "Search tickers using Yahoo's search endpoint, then load charts using the existing "
    "Quant Finance stack (yfinance via `Stock`)."
)


search_service = get_yahoo_search_service()

st.sidebar.header("Search Settings")
query = st.sidebar.text_input("Search query", value="swi")
region = st.sidebar.selectbox("Region", ["US", "IN", "GB", "EU"], index=1)
quotes_count = st.sidebar.slider("Max results", 1, 25, 10)

real_time = st.sidebar.checkbox("Enable real-time refresh", value=True)
refresh_seconds = st.sidebar.slider("Refresh interval (seconds)", 1, 30, 5)

if real_time and hasattr(st, "autorefresh"):
    st.autorefresh(interval=refresh_seconds * 1000, key="search_and_chart_autorefresh")


@st.cache_data(show_spinner=False)
def _search_tickers_cached(query: str, region: str, quotes_count: int) -> List[YahooQuoteResult]:
    """Cached wrapper for Yahoo search to avoid hitting rate limits too aggressively."""
    return search_service.search(query=query, region=region, quotes_count=quotes_count)


if "selected_symbol" not in st.session_state:
    st.session_state.selected_symbol = None


with st.sidebar:
    if st.button("Search"):
        results = _search_tickers_cached(query, region, quotes_count)
        st.session_state["last_results"] = results
    else:
        results = st.session_state.get("last_results", [])


st.subheader("Search Results")

results = st.session_state.get("last_results", [])

if not results:
    st.info("No results yet. Enter a query and click **Search**.")
else:
    df_results = pd.DataFrame(
        [
            {
                "Symbol": r.symbol,
                "Name": r.shortname or r.longname,
                "Exchange": r.exchange,
                "Type": r.quote_type,
                "Sector": r.sector,
                "Industry": r.industry,
                "Score": r.score,
            }
            for r in results
        ]
    )

    st.dataframe(df_results, width="stretch", hide_index=True)

    display_options = [f"{r.symbol} — {r.shortname or r.longname} ({r.exchange})" for r in results]
    choice = st.selectbox("Select symbol to chart", display_options)

    if choice:
        idx = display_options.index(choice)
        selected = results[idx]
        st.session_state.selected_symbol = selected.symbol


symbol = st.session_state.get("selected_symbol")

if symbol:
    st.subheader(f"📈 Chart for {symbol}")

    col1, col2, col3 = st.columns(3)
    with col1:
        period = st.selectbox(
            "Lookback Period",
            ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3,
        )
    with col2:
        interval = st.selectbox(
            "Data Interval",
            ["1m", "5m", "15m", "30m", "1h", "1d"],
            index=5,
        )
    with col3:
        chart_type = st.selectbox("Chart Type", ["Line", "Candlestick"], index=0)

    try:
        stock = Stock(symbol)
        if not stock.validate():
            st.error(f"Invalid symbol: {symbol}")
        else:
            with st.spinner(f"Fetching data for {symbol}..."):
                df = stock.get_historical_data(period=period, interval=interval)

            if df.empty:
                st.error("No data available for this symbol / period / interval.")
            else:
                # Basic stats
                last_close = float(df["Close"].iloc[-1])
                first_close = float(df["Close"].iloc[0])
                pct_change = (last_close - first_close) / first_close * 100.0 if first_close != 0 else 0.0

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Last Close", f"{last_close:.2f}")
                with m2:
                    st.metric("Period Change %", f"{pct_change:+.2f}%")
                with m3:
                    st.metric("Last Volume", f"{int(df['Volume'].iloc[-1]):,}")

                # Chart
                fig = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    row_heights=[0.7, 0.3],
                    subplot_titles=["Price", "Volume"],
                )

                if chart_type == "Line":
                    fig.add_trace(
                        go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="cyan")),
                        row=1,
                        col=1,
                    )
                else:
                    fig.add_trace(
                        go.Candlestick(
                            x=df.index,
                            open=df["Open"],
                            high=df["High"],
                            low=df["Low"],
                            close=df["Close"],
                            name="OHLC",
                            increasing_line_color="green",
                            decreasing_line_color="red",
                        ),
                        row=1,
                        col=1,
                    )

                fig.add_trace(
                    go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color="lightblue"),
                    row=2,
                    col=1,
                )

                fig.update_layout(
                    height=750,
                    hovermode="x unified",
                    xaxis_rangeslider_visible=(chart_type == "Candlestick"),
                    showlegend=True,
                )
                fig.update_xaxes(title_text="Time", row=2, col=1)
                fig.update_yaxes(title_text="Price", row=1, col=1)
                fig.update_yaxes(title_text="Volume", row=2, col=1)

                st.plotly_chart(fig, width="stretch")

    except Exception as exc:
        logger.error(f"Error in search_and_chart app for symbol {symbol}: {exc}")
        st.error(f"Error: {exc}")


