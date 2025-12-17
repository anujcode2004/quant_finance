import logging
import sys
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme
from rich.traceback import install

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from static_memory_cache import StaticMemoryCache


class RichLogger:
    """Colorful rich text logger following pranthora_backend patterns."""
    
    _instance: Optional["RichLogger"] = None
    
    def __init__(self):
        """Initialize the rich logger with colorful formatting."""
        self.config = StaticMemoryCache.get_logging_config()
        self.logger = logging.getLogger("quant_finance")
        self.logger.setLevel(getattr(logging, self.config.get("log_level", "INFO")))
        self.logger.handlers.clear()
        
        # Install rich traceback handler
        install(show_locals=True)
        
        # Create rich console with theme
        theme_name = self.config.get("rich_theme", "monokai")
        custom_theme = Theme({
            "info": "cyan",
            "warning": "yellow",
            "error": "bold red",
            "critical": "bold white on red",
            "debug": "dim white",
            "success": "bold green"
        })
        self.console = Console(theme=custom_theme, stderr=False)
        
        # Console handler with rich formatting
        if self.config.get("log_console", True):
            rich_handler = RichHandler(
                console=self.console,
                show_time=True,
                show_path=True,
                markup=True,
                rich_tracebacks=True,
                tracebacks_show_locals=True
            )
            rich_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
            self.logger.addHandler(rich_handler)
        
        # File handler with rotation
        log_file = self.config.get("log_file", "logs/quant_finance.log")
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=self.config.get("log_file_max_size", 10485760),
            backupCount=self.config.get("log_file_num_backups", 5)
        )
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
    
    @classmethod
    def get_instance(cls) -> "RichLogger":
        """Get singleton instance of the logger."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def info(self, message: str, **kwargs):
        """Log info message with rich formatting."""
        self.logger.info(f"[info]{message}[/info]", **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with rich formatting."""
        self.logger.debug(f"[debug]{message}[/debug]", **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with rich formatting."""
        self.logger.warning(f"[warning]{message}[/warning]", **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with rich formatting."""
        self.logger.error(f"[error]{message}[/error]", **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with rich formatting."""
        self.logger.critical(f"[critical]{message}[/critical]", **kwargs)
    
    def success(self, message: str, **kwargs):
        """Log success message with rich formatting."""
        self.logger.info(f"[success]{message}[/success]", **kwargs)


# Initialize logger instance
logger = RichLogger.get_instance()

