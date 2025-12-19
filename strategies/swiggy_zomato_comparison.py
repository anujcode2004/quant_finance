"""
Swiggy & Zomato (and others) comparison strategy.

Run with:
    streamlit run strategies/swiggy_zomato_comparison.py

Features:
- Defaults to Indian food-delivery / consumer names (e.g. ZOMATO.NS, ETERNAL.NS).
- User can dynamically add/remove symbols.
- Single chart with either line or candlestick view.
- Optional real-time refresh (defaults to 1s).
"""

import sys
from pathlib import Path
from typing import List, Dict

# Ensure project root is on sys.path for package imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from assets.stock import Stock
from telemetries.logger import logger


st.set_page_config(
    page_title="Swiggy & Zomato Comparison",
    page_icon="🍽️",
    layout="wide",
)

st.title("🍽️ Swiggy & Zomato Comparison Dashboard")
st.markdown(
    "Compare Zomato, Swiggy-related proxies, and any other stocks on a single chart.\n"
    "Use the controls on the left to add symbols and switch between line and candlestick views."
)


# Sidebar controls
st.sidebar.header("Configuration")

# Initialise comparison list in session state
if "compare_symbols" not in st.session_state:
    # ZOMATO.NS + a couple of defaults (user can change)
    st.session_state.compare_symbols: List[str] = ["ZOMATO.NS", "ETERNAL.NS"]


def normalise_symbol(sym: str) -> str:
    return sym.strip().upper()


base_symbol_input = st.sidebar.text_input(
    "Add Symbol (e.g. ZOMATO.NS, ETERNAL.NS, AAPL)", value=""
)

if st.sidebar.button("➕ Add Symbol"):
    sym = normalise_symbol(base_symbol_input)
    if sym and sym not in st.session_state.compare_symbols:
        st.session_state.compare_symbols.append(sym)
        logger.info(f"Added symbol to comparison list: {sym}")

# Allow user to remove symbols
remove_symbol = st.sidebar.selectbox(
    "Remove Symbol",
    options=["(none)"] + st.session_state.compare_symbols,
    index=0,
)
if remove_symbol != "(none)" and st.sidebar.button("🗑 Remove Selected"):
    st.session_state.compare_symbols = [
        s for s in st.session_state.compare_symbols if s != remove_symbol
    ]
    logger.info(f"Removed symbol from comparison list: {remove_symbol}")


lookback = st.sidebar.selectbox(
    "Lookback Period (fetch from yfinance)",
    ["1d", "5d", "1mo", "3mo", "6mo", "1y", "5y", "max"],
    index=5,
)

interval = st.sidebar.selectbox(
    "Data Interval",
    ["1m", "5m", "15m", "30m", "1h", "1d"],
    index=5,
)

view_mode = st.sidebar.radio("Chart Type", options=["Line", "Candles"], index=0)

real_time = st.sidebar.checkbox("Enable real-time refresh", value=True)
refresh_seconds = st.sidebar.slider("Refresh interval (seconds)", 1, 30, 1)

# Optional auto-refresh (Streamlit 1.26+)
if real_time and hasattr(st, "autorefresh"):
    st.autorefresh(
        interval=refresh_seconds * 1000,
        key="swiggy_zomato_autorefresh",
    )


def fetch_data_for_symbols(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """Fetch historical data for each symbol."""
    data: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            stock = Stock(sym)
            if not stock.validate():
                logger.warning(f"Validation failed for symbol: {sym}")
                continue
            df = stock.get_historical_data(period=lookback, interval=interval)
            if df.empty:
                logger.warning(f"No data returned for {sym}")
                continue
            data[sym] = df
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Error fetching data for {sym}: {exc}")
    return data


symbols = st.session_state.compare_symbols

if not symbols:
    st.info("No symbols selected. Please add at least one symbol from the sidebar.")
    st.stop()

with st.spinner("Fetching data from yfinance..."):
    symbol_data = fetch_data_for_symbols(symbols)

if not symbol_data:
    st.error(
        "No data could be fetched for the selected symbols. "
        "Please verify ticker symbols or try a different period/interval."
    )
    st.stop()


st.subheader("📈 Price Comparison")

# Create figure
fig = go.Figure()

if view_mode == "Line":
    # Normalise each series to 100 at the start so they are comparable
    for sym, df in symbol_data.items():
        series = df["Close"]
        base = series.iloc[0]
        normalised = (series / base) * 100.0
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=normalised,
                mode="lines",
                name=f"{sym} (Indexed=100)",
            )
        )

    fig.update_yaxes(title_text="Indexed Price (100 = first point)")

else:  # Candles
    # Place each symbol as a separate trace; legend can toggle visibility
    for sym, df in symbol_data.items():
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name=sym,
                increasing_line_color="green",
                decreasing_line_color="red",
                visible=True,
            )
        )

    fig.update_yaxes(title_text="Price")

fig.update_layout(
    height=700,
    hovermode="x unified",
    xaxis_rangeslider_visible=True,
    legend_title_text="Symbols",
)
fig.update_xaxes(title_text="Date / Time")

st.plotly_chart(fig, use_container_width=True)


st.subheader("📋 Latest Values")

summary_rows = []
for sym, df in symbol_data.items():
    last = df.iloc[-1]
    first = df.iloc[0]
    pct_change = (last["Close"] - first["Close"]) / first["Close"] * 100.0
    summary_rows.append(
        {
            "Symbol": sym,
            "Last Close": round(float(last["Close"]), 2),
            "First Close": round(float(first["Close"]), 2),
            "Change %": round(float(pct_change), 2),
            "Last Volume": int(last["Volume"]),
        }
    )

st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)


