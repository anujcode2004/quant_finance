import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_providers.base_provider import BaseDataProvider
from data_providers.yfinance_provider import YFinanceProvider
from static_memory_cache import StaticMemoryCache
from telemetries.logger import logger


class DataProviderFactory:
    """Factory class for creating data provider instances following design patterns."""
    
    _providers: Dict[str, BaseDataProvider] = {}
    
    @classmethod
    def get_provider(cls, provider_name: Optional[str] = None) -> BaseDataProvider:
        """Get or create a data provider instance.
        
        Args:
            provider_name: Name of the provider. If None, uses active provider from config.
        
        Returns:
            DataProvider instance
        """
        if provider_name is None:
            provider_name = StaticMemoryCache.get_active_provider()
        
        # Return cached instance if exists
        if provider_name in cls._providers:
            return cls._providers[provider_name]
        
        # Get provider config
        providers_config = StaticMemoryCache.get_config("data_providers", "providers")
        provider_config = providers_config.get(provider_name, {})
        
        if not provider_config.get("enabled", False):
            raise ValueError(f"Provider {provider_name} is not enabled")
        
        # Create provider instance
        if provider_name == "yfinance":
            provider = YFinanceProvider(provider_config)
        else:
            raise ValueError(f"Unknown provider: {provider_name}")
        
        # Cache the instance
        cls._providers[provider_name] = provider
        logger.info(f"Created and cached {provider_name} provider")
        
        return provider
    
    @classmethod
    def get_active_provider(cls) -> BaseDataProvider:
        """Get the active provider from config."""
        return cls.get_provider()

