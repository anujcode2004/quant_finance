import sys
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from telemetries.logger import logger


class BaseDataProvider(ABC):
    """Abstract base class for data providers following design patterns."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the data provider with configuration.
        
        Args:
            config: Provider-specific configuration dictionary
        """
        self.config = config
        self.timeout = config.get("timeout", 10)
        self.retry_count = config.get("retry_count", 3)
        logger.info(f"Initialized {self.__class__.__name__} with timeout={self.timeout}s")
    
    @abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        period: Optional[str] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch historical price data for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date for data
            end_date: End date for data
            period: Period string (e.g., '1y', '6mo', '1mo')
            interval: Data interval (e.g., '1d', '1h', '5m')
        
        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
        """
        pass
    
    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Current price as float
        """
        pass
    
    @abstractmethod
    def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """Get company information.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Dictionary with company information
        """
        pass
    
    @abstractmethod
    def search_symbols(self, query: str) -> list:
        """Search for symbols matching a query.
        
        Args:
            query: Search query (company name or symbol)
        
        Returns:
            List of matching symbols
        """
        pass
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if a symbol exists and is accessible.
        
        Args:
            symbol: Stock symbol to validate
        
        Returns:
            True if symbol is valid, False otherwise
        """
        try:
            price = self.get_current_price(symbol)
            return price > 0
        except Exception as e:
            logger.error(f"Symbol validation failed for {symbol}: {str(e)}")
            return False

