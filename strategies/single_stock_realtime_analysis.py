"""
Single-stock real‑time quantitative analysis script.

You are a senior Quant Researcher + Applied Mathematician.

Goal
-----
Maintain a *live*, rolling 7‑day intraday view of ONE US stock (1‑minute candles),
and continuously recompute / visualize:

    - Statistical foundation (returns, volatility, autocorrelation, distributions)
    - Linear‑algebra view (vectors, covariance, eigenvalues/eigenvectors)
    - Eigen analysis in lagged‑return space
    - PCA on single‑stock multi‑scale features

All logic is written as a plain Python script using matplotlib (no notebooks required).
Run it from a terminal; it will refresh every few seconds until you stop it with Ctrl+C.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, AutoDateLocator

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ============================================================
# Configuration
# ============================================================

# 🔁 CHANGE THIS to the stock you want to analyze
TICKER = "AAPL"  # <INSERT_TICKER>

# Rolling intraday window: last 7 calendar days (filtered to trading hours)
DAYS_BACK = 7

# Data refresh interval (seconds)
REFRESH_SECONDS = 3

# How many of the most recent points to emphasize in some plots (fraction of window)
RECENT_FRACTION = 0.25


# ============================================================
# Data handling
# ============================================================

def fetch_intraday_1m(ticker: str, days_back: int = 7) -> pd.DataFrame:
    """
    Fetch last `days_back` calendar days of intraday 1‑minute data using yfinance.

    We always re‑query the last N days, which implicitly enforces a rolling window.
    The returned DataFrame is:
        index : DatetimeIndex (timezone dropped)
        cols  : ['Open', 'High', 'Low', 'Close', 'Volume', ...]
    """
    period_str = f"{days_back}d"
    df = yf.download(
        ticker,
        period=period_str,
        interval="1m",
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        return df

    # yfinance can sometimes return a MultiIndex of the form
    #   level 0: ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    #   level 1: tickers
    # For this single‑ticker script, collapse that to plain columns.
    if isinstance(df.columns, pd.MultiIndex):
        # If ticker is in the second level, slice it out.
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, axis=1, level=-1)
        else:
            # Fallback: just keep the first level names.
            df.columns = df.columns.get_level_values(0)

    # Normalize index: drop timezone info, sort
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.sort_index()

    # Hard filter to last `days_back` days based on *now*
    cutoff = datetime.now() - timedelta(days=days_back)
    df = df[df.index >= cutoff]

    # Keep only US trading days (Mon–Fri) and market hours 09:30–16:00
    df = df[df.index.dayofweek < 5]  # 0=Mon, ..., 4=Fri
    df = df.between_time("09:30", "16:00")

    return df


def resample_to_1m(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure we have 1‑minute OHLC candles.
    If the data is already 1‑minute, this is essentially a no‑op;
    otherwise we resample explicitly.
    """
    if df.empty:
        return df

    # Detect approximate frequency; if it's about 1 minute we keep as is.
    if len(df.index) > 1:
        inferred = pd.infer_freq(df.index)
    else:
        inferred = None

    if inferred in {"T", "1T", "1min", "min"}:
        return df

    # Robustly handle cases where some columns are missing
    base_cols = ["Open", "High", "Low", "Close", "Volume"]
    cols_present = [c for c in base_cols if c in df.columns]
    if not cols_present:
        # Nothing useful to resample; just return original frame
        return df

    agg_map = {}
    for c in cols_present:
        if c == "High":
            agg_map[c] = "max"
        elif c == "Low":
            agg_map[c] = "min"
        elif c == "Volume":
            agg_map[c] = "sum"
        else:  # Open, Close
            agg_map[c] = "first" if c == "Open" else "last"

    ohlc = df[cols_present].resample("1min").agg(agg_map)
    ohlc = ohlc.dropna(how="all")
    return ohlc


# ============================================================
# Statistical foundation
# ============================================================

def compute_return_stats(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Compute 1‑minute log returns and associated statistics on the rolling window.

    Math:
        r_t = log(P_t / P_{t-1})

    From r_t we compute mean μ, variance σ², standard deviation σ, and an
    annualized volatility approximation (using 252 trading days * 390 minutes).
    """
    stats_df = df.copy().sort_index()
    stats_df["log_return"] = np.log(stats_df["Close"] / stats_df["Close"].shift(1))
    stats_df = stats_df.dropna()

    r = stats_df["log_return"]
    if r.empty:
        return stats_df, {}

    mean_r = float(r.mean())
    var_r = float(r.var())
    std_r = float(r.std())
    vol_annual = std_r * np.sqrt(252 * 390)  # 252 days * 390 minutes/day

    # Simple lag‑1 autocorrelation as a quick mean‑reversion signal
    acf_lag1 = float(r.autocorr(lag=1))

    # Distribution shape
    skew_r = float(r.skew())
    kurt_r = float(r.kurtosis())  # excess kurtosis

    stats = {
        "mean_r": mean_r,
        "var_r": var_r,
        "std_r": std_r,
        "vol_annual": vol_annual,
        "acf_lag1": acf_lag1,
        "skew": skew_r,
        "excess_kurtosis": kurt_r,
    }

    return stats_df, stats


# ============================================================
# Linear‑algebra view & eigen analysis
# ============================================================

def build_lagged_matrix(returns: pd.Series, max_lag: int = 30) -> np.ndarray:
    """
    Build a lagged‑return matrix X where each column is returns shifted by a lag.

    If r_t is the return at time t, then column j contains r_{t-j}.
    Rows correspond to aligned time indices after all lags are available.
    """
    r = returns.dropna().values
    if len(r) < 10:
        return np.empty((0, 0))

    max_lag = min(max_lag, len(r) // 3)
    if max_lag < 3:
        return np.empty((0, 0))

    n = len(r)
    X = np.zeros((n - max_lag, max_lag + 1))
    for lag in range(max_lag + 1):
        X[:, lag] = r[max_lag - lag : n - lag]
    return X


def eigen_analysis_lagged(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform eigen‑decomposition of the covariance matrix of lagged returns.

    If Σ is the covariance matrix of columns of X, we solve:

        Σ v = λ v

    where λ are eigenvalues (variance captured along each eigen‑direction)
    and v are eigenvectors (patterns of movement across lags).
    """
    if X.size == 0:
        return np.array([]), np.empty((0, 0))

    # Covariance in lag space (features = lags)
    cov_lag = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eigh(cov_lag)
    # Sort descending by eigenvalue magnitude
    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return eigvals, eigvecs


# ============================================================
# PCA on single‑stock multi‑scale features
# ============================================================

def build_pca_features(stats_df: pd.DataFrame) -> Tuple[pd.DataFrame, PCA]:
    """
    Build a feature matrix for PCA using a *single* stock.

    Features:
        - 1m log returns
        - 5m and 15m log‑returns (resampled & ffilled)
        - Rolling volatility (in annualized units)
        - Volume ratio (current / 60‑min moving average)
    """
    feat = pd.DataFrame(index=stats_df.index)
    feat["r_1m"] = stats_df["log_return"]

    # 5m and 15m returns via resample, then align back
    close_5m = stats_df["Close"].resample("5T").last()
    r_5m = np.log(close_5m / close_5m.shift(1))
    feat["r_5m"] = r_5m.reindex(stats_df.index, method="ffill")

    close_15m = stats_df["Close"].resample("15T").last()
    r_15m = np.log(close_15m / close_15m.shift(1))
    feat["r_15m"] = r_15m.reindex(stats_df.index, method="ffill")

    # Rolling volatility (already annualized in compute_return_stats if present)
    if "roll_vol" in stats_df.columns:
        feat["roll_vol"] = stats_df["roll_vol"]
    else:
        # Build a quick 60‑minute rolling vol
        roll_window = min(60, len(stats_df))
        vol = (
            stats_df["log_return"]
            .rolling(roll_window)
            .std()
            * np.sqrt(252 * 390)
        )
        feat["roll_vol"] = vol

    # Volume ratio vs 60‑minute moving average
    if "Volume" in stats_df.columns:
        vol_ma = stats_df["Volume"].rolling(60).mean()
        feat["vol_ratio"] = stats_df["Volume"] / vol_ma
    else:
        feat["vol_ratio"] = 1.0

    feat = feat.dropna()
    if len(feat) < 30:
        return pd.DataFrame(), PCA()

    scaler = StandardScaler()
    X = scaler.fit_transform(feat)

    pca = PCA(n_components=min(3, X.shape[1]))
    pcs = pca.fit_transform(X)
    pc_df = pd.DataFrame(
        pcs, index=feat.index, columns=[f"PC{i+1}" for i in range(pca.n_components_)]
    )

    return pc_df, pca


# ============================================================
# Visualization helpers
# ============================================================

def init_figures():
    """
    Create all matplotlib figures/axes once and reuse them on each refresh.
    """
    plt.style.use("dark_background")
    plt.rcParams["font.size"] = 9

    fig1, (ax_price, ax_vol) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True, num="Price & Rolling Volatility"
    )

    fig2, (ax_hist, ax_acf) = plt.subplots(
        1, 2, figsize=(14, 4), num="Return Distribution & Autocorrelation"
    )

    fig3, (ax_eigs, ax_pca) = plt.subplots(
        1, 2, figsize=(14, 4), num="Eigenvalue Spectrum & PCA Explained Variance"
    )

    for ax in (ax_price, ax_vol, ax_hist, ax_acf, ax_eigs, ax_pca):
        ax.grid(alpha=0.3)

    # Configure datetime formatting on price axis
    ax_price.xaxis.set_major_locator(AutoDateLocator())
    ax_price.xaxis.set_major_formatter(DateFormatter("%m-%d\n%H:%M"))

    plt.tight_layout()
    plt.ion()
    plt.show(block=False)

    return fig1, fig2, fig3, ax_price, ax_vol, ax_hist, ax_acf, ax_eigs, ax_pca


def update_plots(
    df_ohlc: pd.DataFrame,
    stats_df: pd.DataFrame,
    stats: Dict[str, float],
    eigvals: np.ndarray,
    pc_df: pd.DataFrame,
    pca: PCA,
    ax_price,
    ax_vol,
    ax_hist,
    ax_acf,
    ax_eigs,
    ax_pca,
):
    """
    Update all plots using the latest rolling‑window data.
    """
    if df_ohlc.empty or stats_df.empty:
        return

    # Focus on recent fraction for zoomed views
    n = len(df_ohlc)
    start_idx = int(max(0, n - n * RECENT_FRACTION))
    df_zoom = df_ohlc.iloc[start_idx:]

    # --- 1) Price & rolling volatility ---
    ax_price.clear()
    ax_vol.clear()

    ax_price.plot(df_ohlc.index, df_ohlc["Close"], color="deepskyblue", linewidth=0.8)
    ax_price.set_title(f"{TICKER} 1‑min Close Price (last {DAYS_BACK} days)")
    ax_price.set_ylabel("Price")

    # Highlight zoomed recent part
    ax_price.plot(
        df_zoom.index,
        df_zoom["Close"],
        color="yellow",
        linewidth=1.2,
        label="recent window",
    )
    ax_price.legend(loc="upper left")

    # Rolling annualized volatility (60‑minute window)
    r = stats_df["log_return"]
    roll_window = min(60, len(r))
    roll_vol = r.rolling(roll_window).std() * np.sqrt(252 * 390)
    stats_df["roll_vol"] = roll_vol
    ax_vol.plot(
        stats_df.index,
        stats_df["roll_vol"],
        color="orange",
        linewidth=0.8,
    )
    ax_vol.set_ylabel("Annualized Vol")
    ax_vol.set_xlabel("Time")
    ax_vol.set_title(f"Rolling Volatility ({roll_window}‑min window)")

    # --- 2) Return distribution & autocorrelation ---
    ax_hist.clear()
    ax_acf.clear()

    r = stats_df["log_return"].dropna()
    if not r.empty:
        ax_hist.hist(
            r,
            bins=50,
            color="cyan",
            alpha=0.7,
            edgecolor="white",
            density=True,
        )
        ax_hist.set_title(
            "Log‑Return Distribution\n"
            f"skew={stats.get('skew', np.nan):.2f}, "
            f"kurt={stats.get('excess_kurtosis', np.nan):.2f}"
        )
        ax_hist.set_xlabel("log return")
        ax_hist.set_ylabel("density")

        # Simple autocorrelation for first N lags
        max_lag = min(50, len(r) - 1)
        acfs = [r.autocorr(lag=k) for k in range(1, max_lag + 1)]
        ax_acf.stem(range(1, max_lag + 1), acfs, linefmt="cyan", markerfmt="bo")
        ax_acf.set_title(
            "Autocorrelation of Log Returns\n"
            f"lag1={stats.get('acf_lag1', np.nan):.2f}"
        )
        ax_acf.set_xlabel("lag (minutes)")
        ax_acf.set_ylabel("ACF")
        ax_acf.axhline(0, color="white", linewidth=0.5)

    # --- 3) Eigenvalue spectrum & PCA explained variance ---
    ax_eigs.clear()
    ax_pca.clear()

    if eigvals.size > 0:
        k = min(15, len(eigvals))
        ax_eigs.bar(range(1, k + 1), eigvals[:k], color="lime", alpha=0.8)
        ax_eigs.set_title("Eigenvalue Spectrum (lagged‑return covariance)")
        ax_eigs.set_xlabel("component index")
        ax_eigs.set_ylabel("eigenvalue (variance)")

    if pca is not None and hasattr(pca, "explained_variance_ratio_"):
        evr = pca.explained_variance_ratio_
        ax_pca.bar(
            np.arange(1, len(evr) + 1),
            evr * 100.0,
            color="magenta",
            alpha=0.8,
        )
        ax_pca.set_xticks(np.arange(1, len(evr) + 1))
        ax_pca.set_xlabel("principal component")
        ax_pca.set_ylabel("variance explained (%)")
        ax_pca.set_title("PCA Explained Variance (single‑stock features)")

    # Flush updates
    for ax in (ax_price, ax_vol, ax_hist, ax_acf, ax_eigs, ax_pca):
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.pause(0.001)


# ============================================================
# Main loop
# ============================================================

def main():
    print(f"Starting real‑time single‑stock analysis for {TICKER}...")
    print(
        "Rolling window: last 7 calendar days of US trading hours "
        "(1‑minute candles, updated every "
        f"{REFRESH_SECONDS} seconds)."
    )
    print("Press Ctrl+C to stop.\n")

    (
        fig1,
        fig2,
        fig3,
        ax_price,
        ax_vol,
        ax_hist,
        ax_acf,
        ax_eigs,
        ax_pca,
    ) = init_figures()

    iteration = 0

    try:
        while True:
            iteration += 1
            df = fetch_intraday_1m(TICKER, DAYS_BACK)
            if df.empty:
                print(f"[{datetime.now()}] No data returned yet for {TICKER}.")
                time.sleep(REFRESH_SECONDS)
                continue

            df_ohlc = resample_to_1m(df)
            stats_df, stats = compute_return_stats(df_ohlc)

            if stats_df.empty:
                print(f"[{datetime.now()}] Not enough data for stats yet.")
                time.sleep(REFRESH_SECONDS)
                continue

            # Build lagged matrix and eigen analysis
            X_lag = build_lagged_matrix(stats_df["log_return"])
            eigvals, eigvecs = eigen_analysis_lagged(X_lag)

            # PCA on multi‑scale features
            pc_df, pca = build_pca_features(stats_df)

            # One‑line textual summary for traders
            if iteration % 5 == 1:
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"{TICKER}: mean 1m log‑return {stats.get('mean_r', np.nan):+.4e}, "
                    f"ann. vol {stats.get('vol_annual', np.nan):.2%}, "
                    f"lag‑1 ACF {stats.get('acf_lag1', np.nan):+.2f}"
                )
                if eigvals.size > 0:
                    print(
                        f"  Top eigenvalues (lagged covariance) ≈ "
                        f"{', '.join(f'{v:.2e}' for v in eigvals[:3])}"
                    )
                if hasattr(pca, 'explained_variance_ratio_'):
                    evr = pca.explained_variance_ratio_
                    print(
                        "  PCA variance share (PC1,PC2,PC3) ≈ "
                        + ", ".join(f"{x:.1%}" for x in evr[:3])
                    )

            update_plots(
                df_ohlc,
                stats_df,
                stats,
                eigvals,
                pc_df,
                pca,
                ax_price,
                ax_vol,
                ax_hist,
                ax_acf,
                ax_eigs,
                ax_pca,
            )

            time.sleep(REFRESH_SECONDS)

    except KeyboardInterrupt:
        print("\nStopping analysis loop. Goodbye.")


if __name__ == "__main__":
    main()


