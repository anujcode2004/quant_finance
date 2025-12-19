"""
Multi-Stock Analysis Strategy
Compare 2 stocks with 4 charts: 2 price charts and 2 volume/analysis charts
Run with: streamlit run strategies/multi_stock_analysis.py
"""

import sys
from pathlib import Path

# Ensure project root (containing `assets`, `indicators`, etc.) is on sys.path,
# even when this script is run from inside the `strategies/` directory via Streamlit.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from assets.stock import Stock
from indicators.technical_indicators import calculate_sma, calculate_rsi
from telemetries.logger import logger


# Page config
st.set_page_config(
    page_title="Multi-Stock Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Multi-Stock Analysis Dashboard")
st.markdown("Compare two stocks with comprehensive charts and analysis")

# Sidebar for inputs
st.sidebar.header("Stock Selection")

symbol1 = st.sidebar.text_input("Stock 1 Symbol", value="AAPL").upper()
symbol2 = st.sidebar.text_input("Stock 2 Symbol", value="MSFT").upper()

period = st.sidebar.selectbox(
    "Lookback Period",
    ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"],
    index=4,
)

interval = st.sidebar.selectbox(
    "Data Interval",
    ["1m", "5m", "15m", "30m", "1h", "1d"],
    index=5,
)

real_time = st.sidebar.checkbox("Enable real-time refresh", value=True)
refresh_seconds = st.sidebar.slider("Refresh interval (seconds)", 1, 30, 1)

# Optional real-time refresh (Streamlit 1.26+)
if real_time and hasattr(st, "autorefresh"):
    st.autorefresh(interval=refresh_seconds * 1000, key="multi_stock_autorefresh")

if st.sidebar.button("Analyze"):
    try:
        logger.info(f"Starting multi-stock analysis for {symbol1} and {symbol2}")
        
        # Initialize stocks
        stock1 = Stock(symbol1)
        stock2 = Stock(symbol2)
        
        # Validate symbols
        if not stock1.validate():
            st.error(f"Invalid symbol: {symbol1}")
            st.stop()
        
        if not stock2.validate():
            st.error(f"Invalid symbol: {symbol2}")
            st.stop()
        
        # Fetch data
        with st.spinner("Fetching data..."):
            df1 = stock1.get_historical_data(period=period, interval=interval)
            df2 = stock2.get_historical_data(period=period, interval=interval)
        
        if df1.empty or df2.empty:
            st.error("No data available for one or both symbols")
            st.stop()
        
        # Calculate indicators
        df1['SMA_9'] = calculate_sma(df1['Close'], 9)
        df1['SMA_21'] = calculate_sma(df1['Close'], 21)
        df1['RSI'] = calculate_rsi(df1['Close'], 14)
        
        df2['SMA_9'] = calculate_sma(df2['Close'], 9)
        df2['SMA_21'] = calculate_sma(df2['Close'], 21)
        df2['RSI'] = calculate_rsi(df2['Close'], 14)
        
        # Company info
        info1 = stock1.get_company_info()
        info2 = stock2.get_company_info()
        
        # Display metrics
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric(f"{symbol1} Price", f"${stock1.get_current_price():.2f}")
        with col2:
            st.metric(f"{symbol1} Change", f"{((df1['Close'].iloc[-1] - df1['Close'].iloc[0]) / df1['Close'].iloc[0] * 100):.2f}%")
        with col3:
            st.metric(f"{symbol1} RSI", f"{df1['RSI'].iloc[-1]:.1f}")
        with col4:
            st.metric(f"{symbol2} Price", f"${stock2.get_current_price():.2f}")
        with col5:
            st.metric(f"{symbol2} Change", f"{((df2['Close'].iloc[-1] - df2['Close'].iloc[0]) / df2['Close'].iloc[0] * 100):.2f}%")
        with col6:
            st.metric(f"{symbol2} RSI", f"{df2['RSI'].iloc[-1]:.1f}")
        
        # Create 4 charts in 2x2 grid
        st.subheader("📈 Price Charts")
        
        # Chart 1: Stock 1 Price
        fig1 = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f"{symbol1} - Price & Moving Averages", f"{symbol1} - Volume"),
            row_heights=[0.7, 0.3]
        )
        
        fig1.add_trace(
            go.Scatter(x=df1.index, y=df1['Close'], name='Close', line=dict(color='blue')),
            row=1, col=1
        )
        fig1.add_trace(
            go.Scatter(x=df1.index, y=df1['SMA_9'], name='SMA 9', line=dict(color='orange')),
            row=1, col=1
        )
        fig1.add_trace(
            go.Scatter(x=df1.index, y=df1['SMA_21'], name='SMA 21', line=dict(color='red')),
            row=1, col=1
        )
        fig1.add_trace(
            go.Bar(x=df1.index, y=df1['Volume'], name='Volume', marker_color='lightblue'),
            row=2, col=1
        )
        
        fig1.update_layout(height=600, title_text=f"{symbol1} Analysis", showlegend=True)
        fig1.update_xaxes(title_text="Date", row=2, col=1)
        fig1.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig1.update_yaxes(title_text="Volume", row=2, col=1)
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # Chart 2: Stock 2 Price
        fig2 = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f"{symbol2} - Price & Moving Averages", f"{symbol2} - Volume"),
            row_heights=[0.7, 0.3]
        )
        
        fig2.add_trace(
            go.Scatter(x=df2.index, y=df2['Close'], name='Close', line=dict(color='green')),
            row=1, col=1
        )
        fig2.add_trace(
            go.Scatter(x=df2.index, y=df2['SMA_9'], name='SMA 9', line=dict(color='orange')),
            row=1, col=1
        )
        fig2.add_trace(
            go.Scatter(x=df2.index, y=df2['SMA_21'], name='SMA 21', line=dict(color='red')),
            row=1, col=1
        )
        fig2.add_trace(
            go.Bar(x=df2.index, y=df2['Volume'], name='Volume', marker_color='lightgreen'),
            row=2, col=1
        )
        
        fig2.update_layout(height=600, title_text=f"{symbol2} Analysis", showlegend=True)
        fig2.update_xaxes(title_text="Date", row=2, col=1)
        fig2.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig2.update_yaxes(title_text="Volume", row=2, col=1)
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Chart 3: Stock 1 RSI and Volume Analysis
        st.subheader("📊 Technical Analysis")
        
        fig3 = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f"{symbol1} - RSI Indicator", f"{symbol1} - Volume Trend"),
            row_heights=[0.5, 0.5]
        )
        
        # RSI
        fig3.add_trace(
            go.Scatter(x=df1.index, y=df1['RSI'], name='RSI', line=dict(color='purple')),
            row=1, col=1
        )
        fig3.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)", row=1, col=1)
        fig3.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)", row=1, col=1)
        
        # Volume trend (moving average)
        df1['Volume_MA'] = df1['Volume'].rolling(window=20).mean()
        fig3.add_trace(
            go.Bar(x=df1.index, y=df1['Volume'], name='Volume', marker_color='lightblue', opacity=0.6),
            row=2, col=1
        )
        fig3.add_trace(
            go.Scatter(x=df1.index, y=df1['Volume_MA'], name='Volume MA', line=dict(color='darkblue')),
            row=2, col=1
        )
        
        fig3.update_layout(height=600, title_text=f"{symbol1} Technical Indicators", showlegend=True)
        fig3.update_xaxes(title_text="Date", row=2, col=1)
        fig3.update_yaxes(title_text="RSI", row=1, col=1, range=[0, 100])
        fig3.update_yaxes(title_text="Volume", row=2, col=1)
        
        st.plotly_chart(fig3, use_container_width=True)
        
        # Chart 4: Stock 2 RSI and Volume Analysis
        fig4 = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f"{symbol2} - RSI Indicator", f"{symbol2} - Volume Trend"),
            row_heights=[0.5, 0.5]
        )
        
        # RSI
        fig4.add_trace(
            go.Scatter(x=df2.index, y=df2['RSI'], name='RSI', line=dict(color='purple')),
            row=1, col=1
        )
        fig4.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)", row=1, col=1)
        fig4.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)", row=1, col=1)
        
        # Volume trend
        df2['Volume_MA'] = df2['Volume'].rolling(window=20).mean()
        fig4.add_trace(
            go.Bar(x=df2.index, y=df2['Volume'], name='Volume', marker_color='lightgreen', opacity=0.6),
            row=2, col=1
        )
        fig4.add_trace(
            go.Scatter(x=df2.index, y=df2['Volume_MA'], name='Volume MA', line=dict(color='darkgreen')),
            row=2, col=1
        )
        
        fig4.update_layout(height=600, title_text=f"{symbol2} Technical Indicators", showlegend=True)
        fig4.update_xaxes(title_text="Date", row=2, col=1)
        fig4.update_yaxes(title_text="RSI", row=1, col=1, range=[0, 100])
        fig4.update_yaxes(title_text="Volume", row=2, col=1)
        
        st.plotly_chart(fig4, use_container_width=True)
        
        # Comparison table
        st.subheader("📋 Comparison Summary")
        
        comparison_data = {
            "Metric": [
                "Current Price",
                "Period Change %",
                "RSI",
                "SMA 9",
                "SMA 21",
                "Volume (Latest)",
                "Company",
                "Sector"
            ],
            symbol1: [
                f"${stock1.get_current_price():.2f}",
                f"{((df1['Close'].iloc[-1] - df1['Close'].iloc[0]) / df1['Close'].iloc[0] * 100):.2f}%",
                f"{df1['RSI'].iloc[-1]:.1f}",
                f"${df1['SMA_9'].iloc[-1]:.2f}",
                f"${df1['SMA_21'].iloc[-1]:.2f}",
                f"{df1['Volume'].iloc[-1]:,.0f}",
                info1.get("name", "N/A"),
                info1.get("sector", "N/A")
            ],
            symbol2: [
                f"${stock2.get_current_price():.2f}",
                f"{((df2['Close'].iloc[-1] - df2['Close'].iloc[0]) / df2['Close'].iloc[0] * 100):.2f}%",
                f"{df2['RSI'].iloc[-1]:.1f}",
                f"${df2['SMA_9'].iloc[-1]:.2f}",
                f"${df2['SMA_21'].iloc[-1]:.2f}",
                f"{df2['Volume'].iloc[-1]:,.0f}",
                info2.get("name", "N/A"),
                info2.get("sector", "N/A")
            ]
        }
        
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
        
        logger.success(f"Multi-stock analysis completed for {symbol1} and {symbol2}")
        
    except Exception as e:
        logger.error(f"Error in multi-stock analysis: {str(e)}")
        st.error(f"Error: {str(e)}")

