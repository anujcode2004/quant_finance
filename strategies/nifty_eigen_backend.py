"""
Backend utilities for Nifty‑50 style eigenvector / PCA analysis.

This is a **script version** of the ideas in `eigenvectors_analysis.ipynb`,
adapted for Indian large‑cap names (NSE symbols) and meant to be reused
by a Tkinter GUI.

Key functionality:
    - Fetch rolling daily log‑returns for a universe of stocks
    - Build aligned returns matrix
    - Compute correlation matrix
    - Eigenvalue / eigenvector (PCA) analysis on the correlation matrix
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


# Universe: subset of Nifty‑50 style names (NSE symbols)
NIFTY_CORE_SYMBOLS: Dict[str, str] = {
    # name            NSE symbol
    "Reliance Industries": "RELIANCE",
    "HDFC Bank": "HDFCBANK",
    "Bharti Airtel": "BHARTIARTL",
    "TCS": "TCS",
    "ICICI Bank": "ICICIBANK",
    "Infosys": "INFY",
    "State Bank of India": "SBIN",
    "LIC": "LICI",
    "ITC": "ITC",
    "Bajaj Finance": "BAJFINANCE",
    "Hindustan Unilever": "HINDUNILVR",
    "Larsen & Toubro": "LT",
    "HCL Tech": "HCLTECH",
    "Kotak Bank": "KOTAKBANK",
    "Sun Pharma": "SUNPHARMA",
    "Maruti Suzuki": "MARUTI",
    "Axis Bank": "AXISBANK",
    "Mahindra & Mahindra": "M&M",
    "NTPC": "NTPC",
    "Titan": "TITAN",
}


def nse_to_yahoo(symbol: str) -> str:
    """
    Convert plain NSE symbol to Yahoo Finance symbol.

    Most Indian equities are available as `<SYMBOL>.NS`.
    """
    symbol = symbol.strip().upper()
    if symbol.endswith(".NS"):
        return symbol
    return f"{symbol}.NS"


@dataclass
class EigenAnalysisResult:
    """Container for correlation / eigen / PCA results."""

    returns: pd.DataFrame          # aligned log‑returns (T × N)
    corr: pd.DataFrame             # N × N correlation matrix
    eigenvalues: np.ndarray        # length N
    eigenvectors: np.ndarray       # N × N (columns = PCs)
    pve: np.ndarray                # variance explained % per PC (length N)
    pve_cumulative: np.ndarray     # cumulative variance %


def fetch_returns_for_universe(
    nse_symbols: List[str],
    period: str = "1y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch daily OHLC from Yahoo Finance and convert to aligned log‑returns.

    Parameters
    ----------
    nse_symbols : list of NSE tickers without suffix (e.g. 'RELIANCE')
    period      : e.g. '1y', '6mo'
    interval    : e.g. '1d', '1h'

    Returns
    -------
    DataFrame indexed by date (UTC‑naive), columns = NSE symbols, values = log‑returns.
    """
    yahoo_symbols = {sym: nse_to_yahoo(sym) for sym in nse_symbols}

    all_returns: Dict[str, pd.Series] = {}
    for nse_sym, y_sym in yahoo_symbols.items():
        try:
            df = yf.download(
                y_sym,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
            )
            if df.empty or "Close" not in df.columns:
                continue

            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            r = np.log(df["Close"] / df["Close"].shift(1)).dropna()
            if not r.empty:
                all_returns[nse_sym] = r
        except Exception:
            # For GUI we silently skip; caller can inspect resulting columns.
            continue

    if not all_returns:
        return pd.DataFrame()

    returns_df = pd.DataFrame(all_returns).dropna()
    return returns_df


def eigen_analysis_from_returns(returns_df: pd.DataFrame) -> EigenAnalysisResult:
    """
    Given a matrix of aligned returns, compute correlation, eigenvalues/vectors,
    and percentage of variance explained.
    """
    if returns_df.empty:
        raise ValueError("Empty returns DataFrame passed to eigen_analysis_from_returns")

    corr_matrix = returns_df.corr()

    # Eigen‑decomposition of correlation matrix
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix.values)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Variance explained in %
    pve = eigenvalues / eigenvalues.sum() * 100.0
    pve_cumulative = np.cumsum(pve)

    return EigenAnalysisResult(
        returns=returns_df,
        corr=corr_matrix,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        pve=pve,
        pve_cumulative=pve_cumulative,
    )


def quick_eigen_for_default_universe(
    period: str = "1y", interval: str = "1d"
) -> EigenAnalysisResult:
    """
    Convenience helper: run eigen / PCA for the hard‑coded Nifty‑style universe.
    """
    returns_df = fetch_returns_for_universe(
        list(NIFTY_CORE_SYMBOLS.values()), period=period, interval=interval
    )
    if returns_df.empty:
        raise RuntimeError("No returns data fetched for default Nifty universe.")
    return eigen_analysis_from_returns(returns_df)


if __name__ == "__main__":
    # Simple CLI sanity check
    res = quick_eigen_for_default_universe(period="6mo", interval="1d")
    print("Stocks:", list(res.returns.columns))
    print("Eigenvalues (top 5):", res.eigenvalues[:5])
    print("PVE (top 5):", res.pve[:5])


