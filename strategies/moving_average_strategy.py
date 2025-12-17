"""
Moving Average Crossover Strategy
Single stock analysis with 9 and 21 period moving averages
Run with: streamlit run strategies/moving_average_strategy.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from assets.stock import Stock
from indicators.technical_indicators import calculate_sma
from telemetries.logger import logger
from static_memory_cache import StaticMemoryCache


# Page config
st.set_page_config(
    page_title="Moving Average Strategy",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Moving Average Crossover Strategy")
st.markdown("Analyze stocks using 9 and 21 period Simple Moving Averages")

# Sidebar for inputs
st.sidebar.header("Strategy Parameters")

symbol = st.sidebar.text_input("Stock Symbol", value="AAPL").upper()
period = st.sidebar.selectbox("Data Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
sma_short = st.sidebar.number_input("Short SMA Period", min_value=1, max_value=50, value=9)
sma_long = st.sidebar.number_input("Long SMA Period", min_value=1, max_value=200, value=21)

# Get default periods from config
config_periods = StaticMemoryCache.get_indicators_config()
if st.sidebar.button("Use Default Periods"):
    sma_short = config_periods.get("sma_short", 9)
    sma_long = config_periods.get("sma_long", 21)

if st.sidebar.button("Analyze"):
    try:
        logger.info(f"Starting analysis for {symbol}")
        
        # Initialize stock
        stock = Stock(symbol)
        
        if not stock.validate():
            st.error(f"Invalid symbol: {symbol}")
            st.stop()
        
        # Get historical data
        with st.spinner(f"Fetching data for {symbol}..."):
            df = stock.get_historical_data(period=period)
        
        if df.empty:
            st.error("No data available for this symbol")
            st.stop()
        
        # Calculate indicators
        df['SMA_Short'] = calculate_sma(df['Close'], sma_short)
        df['SMA_Long'] = calculate_sma(df['Close'], sma_long)
        
        # Identify crossover signals
        df['Signal'] = 0
        df.loc[df['SMA_Short'] > df['SMA_Long'], 'Signal'] = 1  # Buy signal
        df.loc[df['SMA_Short'] < df['SMA_Long'], 'Signal'] = -1  # Sell signal
        
        # Display company info
        company_info = stock.get_company_info()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Company", company_info.get("name", "N/A"))
        with col2:
            st.metric("Sector", company_info.get("sector", "N/A"))
        with col3:
            current_price = stock.get_current_price()
            st.metric("Current Price", f"${current_price:.2f}")
        with col4:
            price_change = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
            st.metric("Period Change", f"{price_change:.2f}%")
        
        # Create interactive chart
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Price & Moving Averages", "Volume"),
            row_heights=[0.7, 0.3]
        )
        
        # Price and moving averages
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Close'], name='Close Price', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA_Short'], name=f'SMA {sma_short}', line=dict(color='orange')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA_Long'], name=f'SMA {sma_long}', line=dict(color='red')),
            row=1, col=1
        )
        
        # Mark crossover points
        crossovers = df[df['Signal'].diff() != 0]
        buy_signals = crossovers[crossovers['Signal'] == 1]
        sell_signals = crossovers[crossovers['Signal'] == -1]
        
        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index,
                    y=buy_signals['Close'],
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(symbol='triangle-up', size=10, color='green')
                ),
                row=1, col=1
            )
        
        if not sell_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals.index,
                    y=sell_signals['Close'],
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(symbol='triangle-down', size=10, color='red')
                ),
                row=1, col=1
            )
        
        # Volume
        fig.add_trace(
            go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='lightblue'),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text=f"{symbol} - Moving Average Analysis",
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        st.subheader("📊 Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Buy Signals", len(buy_signals))
        with col2:
            st.metric("Total Sell Signals", len(sell_signals))
        with col3:
            st.metric("Current Trend", "Bullish" if df['SMA_Short'].iloc[-1] > df['SMA_Long'].iloc[-1] else "Bearish")
        
        # Data table
        with st.expander("View Raw Data"):
            st.dataframe(df[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_Short', 'SMA_Long', 'Signal']].tail(20))
        
        logger.success(f"Analysis completed for {symbol}")
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        st.error(f"Error: {str(e)}")

