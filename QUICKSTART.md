# Quick Start Guide

## Installation

```bash
# Navigate to project
cd quant_finance

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running Strategies

### Strategy 1: Moving Average Analysis
```bash
streamlit run strategies/moving_average_strategy.py
```

**Features:**
- Single stock analysis
- 9 and 21 period Simple Moving Averages
- Crossover signal detection
- Interactive charts

### Strategy 2: Multi-Stock Comparison
```bash
streamlit run strategies/multi_stock_analysis.py
```

**Features:**
- Compare 2 stocks
- 4 comprehensive charts
- RSI analysis
- Volume trends

## Example Usage

### Using Stock Class
```python
from assets.stock import Stock

# Create stock instance
stock = Stock("AAPL")

# Get historical data
df = stock.get_historical_data(period="1y")

# Get current price
price = stock.get_current_price()

# Get company info
info = stock.get_company_info()
```

### Using Technical Indicators
```python
from indicators.technical_indicators import calculate_sma, calculate_rsi

# Calculate SMA
sma = calculate_sma(df['Close'], 9)

# Calculate RSI
rsi = calculate_rsi(df['Close'], 14)
```

### Using Embedding Service
```python
from services.embedding_service import EmbeddingService

# Get embedding service
embedding_service = EmbeddingService.get_instance()

# Generate embeddings
embeddings = embedding_service.embed_text("Your text here")
```

## Configuration

Edit `config.json` to:
- Change active data provider
- Adjust indicator default periods
- Configure logging
- Set up news API (optional)

## Troubleshooting

**Import errors**: Make sure you're running from the project root directory.

**No data**: Check internet connection and symbol validity.

**Model loading**: First run will download the embedding model (~80MB).

