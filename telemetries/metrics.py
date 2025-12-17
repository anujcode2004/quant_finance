import time
from typing import Dict, Optional, Any
from collections import defaultdict
from datetime import datetime

from rich.console import Console
from rich.table import Table

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from static_memory_cache import StaticMemoryCache
from telemetries.logger import logger


class MetricsCollector:
    """Telemetry and metrics collection system."""
    
    _instance: Optional["MetricsCollector"] = None
    
    def __init__(self):
        """Initialize metrics collector."""
        self.config = StaticMemoryCache.get_telemetry_config()
        self.enabled = self.config.get("enabled", True)
        self.metrics_enabled = self.config.get("metrics_enabled", True)
        self.performance_tracking = self.config.get("performance_tracking", True)
        
        self.metrics: Dict[str, Any] = defaultdict(list)
        self.performance_data: Dict[str, float] = {}
        self.console = Console()
    
    @classmethod
    def get_instance(cls) -> "MetricsCollector":
        """Get singleton instance of metrics collector."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def track_metric(self, name: str, value: Any, tags: Optional[Dict[str, str]] = None):
        """Track a metric value."""
        if not self.metrics_enabled:
            return
        
        metric_entry = {
            "value": value,
            "timestamp": datetime.now().isoformat(),
            "tags": tags or {}
        }
        self.metrics[name].append(metric_entry)
    
    def track_performance(self, operation: str, duration: float):
        """Track performance of an operation."""
        if not self.performance_tracking:
            return
        
        if operation not in self.performance_data:
            self.performance_data[operation] = []
        
        self.performance_data[operation].append(duration)
        logger.debug(f"Performance: {operation} took {duration:.3f}s")
    
    def get_metric_summary(self, name: str) -> Optional[Dict[str, Any]]:
        """Get summary statistics for a metric."""
        if name not in self.metrics or not self.metrics[name]:
            return None
        
        values = [m["value"] for m in self.metrics[name] if isinstance(m["value"], (int, float))]
        if not values:
            return None
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1]
        }
    
    def display_metrics_table(self):
        """Display metrics in a rich table."""
        if not self.metrics:
            logger.info("No metrics to display")
            return
        
        table = Table(title="Metrics Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="magenta")
        table.add_column("Latest", style="green")
        table.add_column("Avg", style="yellow")
        
        for metric_name, entries in self.metrics.items():
            summary = self.get_metric_summary(metric_name)
            if summary:
                table.add_row(
                    metric_name,
                    str(summary["count"]),
                    str(summary["latest"]),
                    f"{summary['avg']:.2f}"
                )
        
        self.console.print(table)
    
    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.performance_data.clear()
        logger.info("Metrics reset")


# Initialize metrics collector
metrics = MetricsCollector.get_instance()

