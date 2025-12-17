import pandas as pd
import numpy as np
from typing import Optional

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from static_memory_cache import StaticMemoryCache
from telemetries.logger import logger


def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average.
    
    Args:
        data: Price series (typically Close)
        period: Period for SMA
    
    Returns:
        Series with SMA values
    """
    return data.rolling(window=period).mean()


def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average.
    
    Args:
        data: Price series
        period: Period for EMA
    
    Returns:
        Series with EMA values
    """
    return data.ewm(span=period, adjust=False).mean()


def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index.
    
    Args:
        data: Price series
        period: Period for RSI (default 14)
    
    Returns:
        Series with RSI values (0-100)
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(
    data: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> pd.DataFrame:
    """Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        data: Price series
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period
    
    Returns:
        DataFrame with columns: MACD, Signal, Histogram
    """
    ema_fast = calculate_ema(data, fast_period)
    ema_slow = calculate_ema(data, slow_period)
    
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    
    return pd.DataFrame({
        "MACD": macd_line,
        "Signal": signal_line,
        "Histogram": histogram
    })


def calculate_bollinger_bands(
    data: pd.Series,
    period: int = 20,
    std_dev: float = 2.0
) -> pd.DataFrame:
    """Calculate Bollinger Bands.
    
    Args:
        data: Price series
        period: Period for moving average
        std_dev: Standard deviation multiplier
    
    Returns:
        DataFrame with columns: Upper, Middle, Lower
    """
    sma = calculate_sma(data, period)
    std = data.rolling(window=period).std()
    
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    
    return pd.DataFrame({
        "Upper": upper,
        "Middle": sma,
        "Lower": lower
    })


def calculate_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3
) -> pd.DataFrame:
    """Calculate Stochastic Oscillator.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        k_period: %K period
        d_period: %D period
    
    Returns:
        DataFrame with columns: %K, %D
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()
    
    return pd.DataFrame({
        "%K": k_percent,
        "%D": d_percent
    })


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """Calculate Average True Range.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: Period for ATR
    
    Returns:
        Series with ATR values
    """
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr


def calculate_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """Calculate Average Directional Index.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: Period for ADX
    
    Returns:
        Series with ADX values
    """
    # Calculate +DM and -DM
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    # Calculate True Range
    atr = calculate_atr(high, low, close, period)
    
    # Calculate +DI and -DI
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    # Calculate DX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    
    # Calculate ADX
    adx = dx.rolling(window=period).mean()
    
    return adx

