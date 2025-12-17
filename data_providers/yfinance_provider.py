import yfinance as yf
import pandas as pd
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import time

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_providers.base_provider import BaseDataProvider
from telemetries.logger import logger
from telemetries.metrics import metrics


class YFinanceProvider(BaseDataProvider):
    """YFinance data provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize YFinance provider."""
        super().__init__(config)
        logger.info("YFinance provider initialized")
    
    def get_historical_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        period: Optional[str] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch historical data using yfinance.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            period: Period string (e.g., '1y', '6mo')
            interval: Data interval
        
        Returns:
            DataFrame with OHLCV data
        """
        start_time = time.time()
        
        try:
            ticker = yf.Ticker(symbol)
            
            if period:
                df = ticker.history(period=period, interval=interval)
            elif start_date and end_date:
                df = ticker.history(start=start_date, end=end_date, interval=interval)
            else:
                # Default to 1 year
                df = ticker.history(period="1y", interval=interval)
            
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Ensure column names are standard
            df.columns = [col.replace(' ', '_') for col in df.columns]
            
            duration = time.time() - start_time
            metrics.track_performance(f"yfinance_get_historical_{symbol}", duration)
            metrics.track_metric("data_fetch_count", 1, {"provider": "yfinance", "symbol": symbol})
            
            logger.info(f"Fetched {len(df)} rows for {symbol} in {duration:.3f}s")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            raise
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price."""
        start_time = time.time()
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Try different price fields
            price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
            
            if price is None:
                # Fallback: get latest from history
                hist = ticker.history(period="1d", interval="1m")
                if not hist.empty:
                    price = hist['Close'].iloc[-1]
                else:
                    raise ValueError(f"Could not fetch price for {symbol}")
            
            duration = time.time() - start_time
            metrics.track_performance(f"yfinance_get_price_{symbol}", duration)
            
            logger.debug(f"Current price for {symbol}: ${price:.2f}")
            return float(price)
            
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {str(e)}")
            raise
    
    def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """Get company information."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract relevant info
            company_info = {
                "symbol": symbol,
                "name": info.get("longName") or info.get("shortName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap"),
                "currency": info.get("currency", "USD"),
                "exchange": info.get("exchange", ""),
                "description": info.get("longBusinessSummary", "")[:500]  # Truncate
            }
            
            logger.debug(f"Fetched company info for {symbol}")
            return company_info
            
        except Exception as e:
            logger.error(f"Error fetching company info for {symbol}: {str(e)}")
            raise
    
    def search_symbols(self, query: str) -> list:
        """Search for symbols (yfinance doesn't have built-in search, so we use ticker lookup)."""
        try:
            # Try direct lookup first
            ticker = yf.Ticker(query.upper())
            info = ticker.info
            
            if info and info.get("symbol"):
                return [{
                    "symbol": info.get("symbol"),
                    "name": info.get("longName") or info.get("shortName", ""),
                    "exchange": info.get("exchange", "")
                }]
            
            # If direct lookup fails, return empty
            logger.warning(f"No results found for search query: {query}")
            return []
            
        except Exception as e:
            logger.debug(f"Search failed for {query}: {str(e)}")
            return []

