"""
Indian Consumer & Food-Tech Dashboard

Similar to `dashboard.py` but focused on Indian names like Zomato, Eternal, etc.

Run with:
    streamlit run strategies/indian_consumer_dashboard.py

Features:
- Default Indian symbols (you can extend this list).
- Add/remove more symbols dynamically from the sidebar.
- Line-chart comparison of all symbols.
- Real-time refresh fixed at 1 second.
- Per-symbol cards showing latest price, % change and a small tick/arrow
  indicating direction of the latest move.
"""

import sys
from pathlib import Path
from typing import List, Dict

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from assets.stock import Stock
from telemetries.logger import logger


REFRESH_SECONDS = 1  # fixed 1s refresh across this strategy


st.set_page_config(
    page_title="Indian Consumer Dashboard",
    page_icon="🇮🇳",
    layout="wide",
)

st.title("🇮🇳 Indian Consumer & Food-Tech Dashboard")
st.markdown(
    "Live dashboard for Indian consumer / food-tech stocks such as Zomato, Eternal, etc. "
    "You can add more NSE symbols dynamically (e.g. `ZOMATO.NS`, `ETERNAL.NS`)."
)


# Sidebar configuration
st.sidebar.header("Configuration")

# Default Indian-focused symbols – you can keep extending this list as needed
DEFAULT_SYMBOLS: List[str] = [
    "ZOMATO.NS",   # Zomato
    "ETERNAL.NS",  # Eternal Limited (from your screenshot)
]

if "indian_symbols" not in st.session_state:
    st.session_state.indian_symbols: List[str] = DEFAULT_SYMBOLS.copy()

if "prev_last_prices" not in st.session_state:
    st.session_state.prev_last_prices: Dict[str, float] = {}


def normalise_symbol(sym: str) -> str:
    return sym.strip().upper()


new_symbol_input = st.sidebar.text_input(
    "Add NSE/BSE symbol (e.g. ZOMATO.NS, ETERNAL.NS, RELIANCE.NS)", value=""
)

if st.sidebar.button("➕ Add Symbol"):
    sym = normalise_symbol(new_symbol_input)
    if sym and sym not in st.session_state.indian_symbols:
        st.session_state.indian_symbols.append(sym)
        logger.info(f"Added Indian symbol: {sym}")

remove_symbol = st.sidebar.selectbox(
    "Remove Symbol",
    options=["(none)"] + st.session_state.indian_symbols,
    index=0,
)
if remove_symbol != "(none)" and st.sidebar.button("🗑 Remove Selected"):
    st.session_state.indian_symbols = [
        s for s in st.session_state.indian_symbols if s != remove_symbol
    ]
    logger.info(f"Removed Indian symbol: {remove_symbol}")


lookback = st.sidebar.selectbox(
    "Lookback Period",
    ["1d", "5d", "1mo", "3mo", "6mo", "1y"],
    index=2,
)

interval = st.sidebar.selectbox(
    "Data Interval",
    ["1m", "5m", "15m", "30m", "1h", "1d"],
    index=4,
)

# Real-time refresh fixed to 1s for this strategy
st.sidebar.markdown(f"**Real-time refresh:** every {REFRESH_SECONDS} second")
if hasattr(st, "autorefresh"):
    st.autorefresh(interval=REFRESH_SECONDS * 1000, key="indian_dashboard_autorefresh")


def fetch_data(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """Fetch historical data for the configured Indian-focused symbols."""
    result: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            stock = Stock(sym)
            if not stock.validate():
                logger.warning(f"Validation failed for symbol: {sym}")
                continue
            df = stock.get_historical_data(period=lookback, interval=interval)
            if df.empty:
                logger.warning(f"No data for {sym} (lookback={lookback}, interval={interval})")
                continue
            result[sym] = df
        except Exception as exc:  # pragma: no cover
            logger.error(f"Error fetching data for {sym}: {exc}")
    return result


symbols = st.session_state.indian_symbols
if not symbols:
    st.info("No symbols configured. Please add at least one symbol from the sidebar.")
    st.stop()

with st.spinner("Fetching data from yfinance..."):
    symbol_dfs = fetch_data(symbols)

if not symbol_dfs:
    st.error("Could not fetch data for any of the configured symbols.")
    st.stop()


# Top metrics section – one card per symbol
st.subheader("📊 Symbol Overview")

cols = st.columns(len(symbol_dfs))
for (sym, df), col in zip(symbol_dfs.items(), cols):
    last = df.iloc[-1]
    first = df.iloc[0]
    current_price = float(last["Close"])
    pct_change = (current_price - float(first["Close"])) / float(first["Close"]) * 100.0

    # Detect direction vs last refresh
    prev_price = st.session_state.prev_last_prices.get(sym)
    direction_icon = "•"  # neutral
    direction_color = "gray"
    if prev_price is not None:
        if current_price > prev_price:
            direction_icon = "▲"
            direction_color = "green"
        elif current_price < prev_price:
            direction_icon = "▼"
            direction_color = "red"

    # Store latest price for next tick comparison
    st.session_state.prev_last_prices[sym] = current_price

    with col:
        st.markdown(f"**{sym}**")
        st.metric(
            label="Last Close",
            value=f"{current_price:.2f}",
            delta=f"{pct_change:.2f}%",
        )
        # Small "tick" / arrow next to price range info
        st.markdown(
            f"<span style='color:{direction_color}; font-size: 0.9rem;'>"
            f"{direction_icon} latest tick</span>",
            unsafe_allow_html=True,
        )


# Line-chart comparison of all symbols (indexed to 100 at start)
st.subheader("📈 Price Comparison (Indexed)")

fig = go.Figure()
for sym, df in symbol_dfs.items():
    series = df["Close"]
    base = float(series.iloc[0])
    if base == 0:
        continue
    indexed = (series / base) * 100.0
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=indexed,
            mode="lines",
            name=f"{sym} (100 = first point)",
        )
    )

fig.update_layout(
    height=650,
    hovermode="x unified",
    xaxis_rangeslider_visible=True,
    legend_title_text="Symbols",
)
fig.update_xaxes(title_text="Date / Time")
fig.update_yaxes(title_text="Indexed Price")

st.plotly_chart(fig, use_container_width=True)


st.subheader("📋 Latest Snapshot")

summary_rows = []
for sym, df in symbol_dfs.items():
    last = df.iloc[-1]
    first = df.iloc[0]
    last_close = float(last["Close"])
    first_close = float(first["Close"])
    pct_change = (last_close - first_close) / first_close * 100.0

    summary_rows.append(
        {
            "Symbol": sym,
            "First Close": round(first_close, 2),
            "Last Close": round(last_close, 2),
            "Change %": round(pct_change, 2),
            "Last Volume": int(last["Volume"]),
        }
    )

st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)


