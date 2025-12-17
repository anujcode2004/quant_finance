import requests
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from static_memory_cache import StaticMemoryCache
from telemetries.logger import logger
from telemetries.metrics import metrics


class NewsService:
    """News fetching service class following design patterns."""
    
    def __init__(self):
        """Initialize News Service."""
        self.config = StaticMemoryCache.get_config("news")
        self.provider = self.config.get("provider", "newsapi")
        self.api_key = None
        
        # Initialize provider-specific config
        if self.provider == "newsapi":
            newsapi_config = self.config.get("newsapi", {})
            if newsapi_config.get("enabled", False):
                self.api_key = newsapi_config.get("api_key")
        
        logger.info(f"Initialized NewsService with provider: {self.provider}")
    
    def fetch_news(
        self,
        query: str,
        symbol: Optional[str] = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Fetch news articles.
        
        Args:
            query: Search query
            symbol: Optional stock symbol to filter news
            max_results: Maximum number of results
        
        Returns:
            List of news articles
        """
        if self.provider == "newsapi" and self.api_key:
            return self._fetch_newsapi(query, symbol, max_results)
        else:
            logger.warning("News API not configured. Returning empty results.")
            return []
    
    def _fetch_newsapi(
        self,
        query: str,
        symbol: Optional[str] = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Fetch news from NewsAPI."""
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": f"{query} {symbol}" if symbol else query,
                "apiKey": self.api_key,
                "pageSize": max_results,
                "sortBy": "publishedAt",
                "language": "en"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            articles = data.get("articles", [])
            
            # Format articles
            formatted_articles = []
            for article in articles[:max_results]:
                formatted_articles.append({
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "url": article.get("url", ""),
                    "published_at": article.get("publishedAt", ""),
                    "source": article.get("source", {}).get("name", "")
                })
            
            metrics.track_metric("news_fetch_count", len(formatted_articles))
            logger.info(f"Fetched {len(formatted_articles)} news articles for query: {query}")
            
            return formatted_articles
            
        except Exception as e:
            logger.error(f"Error fetching news: {str(e)}")
            return []
    
    def fetch_stock_news(self, symbol: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Fetch news specifically for a stock symbol.
        
        Args:
            symbol: Stock symbol
            max_results: Maximum number of results
        
        Returns:
            List of news articles
        """
        return self.fetch_news(symbol, symbol=symbol, max_results=max_results)

