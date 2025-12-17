import pandas as pd
from typing import Optional, Dict, Any
from datetime import datetime

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_providers.base_provider import BaseDataProvider
from data_providers.provider_factory import DataProviderFactory
from telemetries.logger import logger
from telemetries.metrics import metrics


class Stock:
    """Stock asset class following design patterns."""
    
    def __init__(self, symbol: str, provider: Optional[BaseDataProvider] = None):
        """Initialize Stock instance.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            provider: Optional data provider. If None, uses active provider.
        """
        self.symbol = symbol.upper()
        self.provider = provider or DataProviderFactory.get_active_provider()
        self._company_info: Optional[Dict[str, Any]] = None
        self._current_price: Optional[float] = None
        
        logger.info(f"Initialized Stock: {self.symbol}")
    
    def get_historical_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        period: Optional[str] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Get historical price data.
        
        Args:
            start_date: Start date
            end_date: End date
            period: Period string (e.g., '1y', '6mo')
            interval: Data interval
        
        Returns:
            DataFrame with OHLCV data
        """
        logger.debug(f"Fetching historical data for {self.symbol}")
        return self.provider.get_historical_data(
            self.symbol, start_date, end_date, period, interval
        )
    
    def get_current_price(self, refresh: bool = False) -> float:
        """Get current price.
        
        Args:
            refresh: If True, fetches fresh data
        
        Returns:
            Current price
        """
        if self._current_price is None or refresh:
            self._current_price = self.provider.get_current_price(self.symbol)
            metrics.track_metric("stock_price_fetch", 1, {"symbol": self.symbol})
        
        return self._current_price
    
    def get_company_info(self, refresh: bool = False) -> Dict[str, Any]:
        """Get company information.
        
        Args:
            refresh: If True, fetches fresh data
        
        Returns:
            Company information dictionary
        """
        if self._company_info is None or refresh:
            self._company_info = self.provider.get_company_info(self.symbol)
        
        return self._company_info
    
    def validate(self) -> bool:
        """Validate if the stock symbol is valid."""
        return self.provider.validate_symbol(self.symbol)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Stock(symbol='{self.symbol}')"

