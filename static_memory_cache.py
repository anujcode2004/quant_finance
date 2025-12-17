import json
import os
from typing import Optional, Dict, Any
from pathlib import Path


class StaticMemoryCache:
    """Static memory cache for configuration and models following pranthora_backend pattern."""
    
    config: Dict[str, Any] = {}
    embedding_model = None
    
    @classmethod
    def initialize(cls, config_file: str = "config.json"):
        """Load config into memory at startup."""
        # Get the directory where this file is located
        base_dir = Path(__file__).parent
        config_path = base_dir / config_file
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, "r") as f:
            cls.config = json.load(f)
    
    @classmethod
    def get_config(cls, section: str, key: Optional[str] = None) -> Any:
        """Retrieve configuration value from the static memory cache.
        
        Args:
            section: Configuration section name
            key: Optional key within the section. If None, returns entire section.
        
        Returns:
            Configuration value or entire section dict
        """
        if key is None:
            return cls.config.get(section, {})
        return cls.config.get(section, {}).get(key)
    
    @classmethod
    def get_data_provider_config(cls) -> Dict[str, Any]:
        """Get active data provider configuration."""
        active = cls.config.get("data_providers", {}).get("active_provider", "yfinance")
        providers = cls.config.get("data_providers", {}).get("providers", {})
        return providers.get(active, {})
    
    @classmethod
    def get_active_provider(cls) -> str:
        """Get the name of the active data provider."""
        return cls.config.get("data_providers", {}).get("active_provider", "yfinance")
    
    @classmethod
    def get_embedding_config(cls) -> Dict[str, Any]:
        """Get embedding model configuration."""
        return cls.config.get("embedding", {})
    
    @classmethod
    def get_logging_config(cls) -> Dict[str, Any]:
        """Get logging configuration."""
        return cls.config.get("logging", {})
    
    @classmethod
    def get_telemetry_config(cls) -> Dict[str, Any]:
        """Get telemetry configuration."""
        return cls.config.get("telemetry", {})
    
    @classmethod
    def get_indicators_config(cls) -> Dict[str, Any]:
        """Get indicators default configuration."""
        return cls.config.get("indicators", {}).get("default_periods", {})


# Initialize on import
StaticMemoryCache.initialize()

