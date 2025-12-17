# Quant Finance - Interactive Trading Analysis Platform

A comprehensive quantitative finance platform built with Python, Streamlit, and yfinance. This project provides interactive stock analysis tools with technical indicators, multi-stock comparisons, and extensible architecture following design patterns similar to pranthora_backend.

## 🏗️ Project Structure

```
quant_finance/
├── config.json                 # Configuration file
├── static_memory_cache.py      # Static memory cache for config
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── data_providers/            # Data provider abstractions
│   ├── __init__.py
│   ├── base_provider.py       # Abstract base class
│   ├── yfinance_provider.py   # YFinance implementation
│   └── provider_factory.py    # Factory pattern for providers
│
├── assets/                    # Asset classes
│   ├── __init__.py
│   ├── stock.py              # Stock asset class
│   ├── bond.py               # Bond asset class
│   └── news.py               # News fetching service
│
├── services/                 # Core services
│   ├── __init__.py
│   └── embedding_service.py  # MiniLM-6 embedding service
│
├── indicators/               # Technical indicators
│   ├── __init__.py
│   └── technical_indicators.py  # All indicator functions
│
├── strategies/               # Strategy Streamlit apps
│   ├── __init__.py
│   ├── moving_average_strategy.py  # Single MA strategy
│   └── multi_stock_analysis.py     # Multi-stock comparison
│
└── telemetries/              # Logging and metrics
    ├── __init__.py
    ├── logger.py             # Rich text logger
    └── metrics.py             # Telemetry collector
```

## 🚀 Quick Start

### Installation

1. **Clone or navigate to the project directory:**
```bash
cd quant_finance
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Running Strategies

Each strategy is a standalone Streamlit app. Run them individually:

**Moving Average Strategy:**
```bash
streamlit run strategies/moving_average_strategy.py
```

**Multi-Stock Analysis:**
```bash
streamlit run strategies/multi_stock_analysis.py
```

## 📊 Features

### 1. **Data Providers**
- **Abstract Base Provider**: Extensible architecture for multiple data sources
- **YFinance Provider**: Current active provider for stock data
- **Factory Pattern**: Easy switching between providers via config.json
- **Future Providers**: Ready for Alpha Vantage, Polygon.io, etc.

### 2. **Asset Classes**
- **Stock Class**: Complete stock data management with caching
- **Bond Class**: Bond asset structure (ready for future implementation)
- **News Service**: News fetching with multiple provider support

### 3. **Technical Indicators**
Modular indicator functions that can be called independently:
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Relative Strength Index (RSI)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Stochastic Oscillator
- Average True Range (ATR)
- Average Directional Index (ADX)

### 4. **Interactive Charts**
- TradingView-like interactive charts using Plotly
- Real-time data visualization
- Multiple chart types (price, volume, indicators)
- Responsive and customizable

### 5. **Embedding Service**
- Sentence-transformers integration
- MiniLM-L6-v2 model for text embeddings
- Singleton pattern for efficient model loading
- Ready for future semantic search and RAG applications

### 6. **Logging & Telemetry**
- **Rich Text Logging**: Colorful, formatted console output
- **File Logging**: Rotating log files
- **Metrics Collection**: Performance tracking and statistics
- **Structured Logging**: JSON-formatted logs for analysis

## ⚙️ Configuration

Edit `config.json` to customize:

### Data Providers
```json
{
  "data_providers": {
    "active_provider": "yfinance",
    "providers": {
      "yfinance": {
        "enabled": true,
        "timeout": 10,
        "retry_count": 3
      }
    }
  }
}
```

### Indicators Default Periods
```json
{
  "indicators": {
    "default_periods": {
      "sma_short": 9,
      "sma_long": 21,
      "rsi_period": 14
    }
  }
}
```

### News API (Optional)
```json
{
  "news": {
    "provider": "newsapi",
    "newsapi": {
      "api_key": "your_api_key_here",
      "enabled": true
    }
  }
}
```

## 📈 Sample Strategies

### 1. Moving Average Strategy
- **File**: `strategies/moving_average_strategy.py`
- **Features**:
  - Single stock analysis
  - 9 and 21 period Simple Moving Averages
  - Crossover signal detection
  - Interactive price and volume charts
  - Company information display

**Usage:**
1. Enter stock symbol (e.g., AAPL)
2. Select data period
3. Adjust SMA periods
4. Click "Analyze"

### 2. Multi-Stock Analysis
- **File**: `strategies/multi_stock_analysis.py`
- **Features**:
  - Compare 2 stocks side-by-side
  - 4 comprehensive charts:
    - Stock 1: Price & Moving Averages + Volume
    - Stock 2: Price & Moving Averages + Volume
    - Stock 1: RSI Indicator + Volume Trend
    - Stock 2: RSI Indicator + Volume Trend
  - Comparison metrics table
  - Real-time RSI analysis

**Usage:**
1. Enter two stock symbols
2. Select data period
3. Click "Analyze"
4. View comprehensive comparison

## 🏛️ Design Patterns

This project follows design patterns similar to `pranthora_backend`:

1. **Factory Pattern**: `DataProviderFactory` for creating provider instances
2. **Singleton Pattern**: Logger, Metrics, EmbeddingService
3. **Abstract Base Classes**: `BaseDataProvider` for extensibility
4. **Static Memory Cache**: Configuration management
5. **Service Layer**: Separation of concerns (data, services, assets)
6. **Modular Indicators**: Independent, reusable indicator functions

## 🔧 Adding New Strategies

Create a new file in `strategies/` folder:

```python
import streamlit as st
from assets.stock import Stock
from indicators.technical_indicators import calculate_rsi

st.title("My New Strategy")

symbol = st.text_input("Symbol", "AAPL")
if st.button("Analyze"):
    stock = Stock(symbol)
    df = stock.get_historical_data(period="1y")
    df['RSI'] = calculate_rsi(df['Close'], 14)
    # Your analysis code here
```

Run with:
```bash
streamlit run strategies/my_strategy.py
```

## 📝 Adding New Indicators

Add functions to `indicators/technical_indicators.py`:

```python
def calculate_my_indicator(data: pd.Series, period: int) -> pd.Series:
    """Calculate my custom indicator."""
    # Your calculation here
    return result
```

Import and use:
```python
from indicators.technical_indicators import calculate_my_indicator
```

## 🔮 Future Enhancements

- [ ] Additional data providers (Alpha Vantage, Polygon.io)
- [ ] More technical indicators
- [ ] Backtesting framework
- [ ] Portfolio management
- [ ] Options analysis
- [ ] Cryptocurrency support
- [ ] Real-time data streaming
- [ ] Machine learning models integration
- [ ] RAG system for financial document analysis

## 📚 Dependencies

- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **yfinance**: Stock data fetching
- **streamlit**: Interactive web apps
- **plotly**: Interactive charts
- **sentence-transformers**: Embedding generation
- **rich**: Beautiful terminal output

## 🐛 Troubleshooting

**Issue**: "No data available"
- **Solution**: Check internet connection and symbol validity

**Issue**: "Embedding model loading error"
- **Solution**: Ensure sentence-transformers is installed and internet is available for first-time model download

**Issue**: "Provider not enabled"
- **Solution**: Check `config.json` and enable the desired provider

## 📄 License

This project is for educational and research purposes.

## 🤝 Contributing

When adding new features:
1. Follow the existing design patterns
2. Use the logger for all operations
3. Add telemetry/metrics tracking
4. Update this README
5. Test with sample strategies

---

**Happy Trading! 📈**

