"""
SWIGGY.BO Quant EDA (CLI + Matplotlib)
======================================

Pure matplotlib version of the Swiggy exploratory quant analysis:
- Fetches OHLCV data for SWIGGY.BO using the existing Stock class
- Computes returns, indicators and risk metrics
- Opens matplotlib windows (price/indicators, return histogram,
  volatility surface, risk histogram).

Run with:

    cd /Users/inavlabs/Documents/inavlabs/quant_finance
    python -m strategies.swiggy_quant_eda_cli

You can change symbol / period / interval via:

    python -m strategies.swiggy_quant_eda_cli --symbol SWIGGY.BO --period 6mo --interval 1d
"""

import argparse
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed for 3D plots
import matplotlib.dates as mdates

from assets.stock import Stock
from indicators.technical_indicators import (
    calculate_sma,
    calculate_bollinger_bands,
    calculate_rsi,
    calculate_macd,
)
from telemetries.logger import logger


REFRESH_SECONDS = 1  # fetch + redraw every second


def fetch_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    stock = Stock(symbol)
    if not stock.validate():
        raise ValueError(f"Invalid symbol: {symbol}")
    df = stock.get_historical_data(period=period, interval=interval)
    if df.empty:
        raise ValueError("No data returned for given symbol/period/interval")
    return df


def plot_candles_with_indicators(ax, df: pd.DataFrame, symbol: str):
    """Candlesticks + SMA 9/21 + Bollinger bands on a single axis."""
    df = df.copy()
    df["SMA_9"] = calculate_sma(df["Close"], 9)
    df["SMA_21"] = calculate_sma(df["Close"], 21)
    bb = calculate_bollinger_bands(df["Close"], period=20, std_dev=2.0)
    df["BB_Upper"] = bb["Upper"]
    df["BB_Lower"] = bb["Lower"]

    ax.set_facecolor("black")
    ax.grid(True, color="white", alpha=0.15)

    dates = mdates.date2num(df.index.to_pydatetime())
    candle_width = 0.6 * (dates[1] - dates[0]) if len(dates) > 1 else 0.0005

    for x, row in zip(dates, df.itertuples()):
        o = row.Open
        h = row.High
        l = row.Low
        c = row.Close
        color = "lime" if c >= o else "red"

        ax.vlines(x, l, h, color=color, linewidth=0.8, alpha=0.9)

        lower = min(o, c)
        height = abs(c - o) or 0.0001
        rect = Rectangle(
            (x - candle_width / 2, lower),
            candle_width,
            height,
            facecolor=color,
            edgecolor=color,
            linewidth=0.8,
        )
        ax.add_patch(rect)

    ax.plot(dates, df["SMA_9"], color="cyan", linewidth=1.0, label="SMA 9")
    ax.plot(dates, df["SMA_21"], color="orange", linewidth=1.0, label="SMA 21")
    ax.plot(dates, df["BB_Upper"], color="magenta", linestyle="--", linewidth=0.8, label="BB Upper")
    ax.plot(dates, df["BB_Lower"], color="magenta", linestyle="--", linewidth=0.8, label="BB Lower")

    ax.set_title(f"{symbol} - Price, SMAs & Bollinger Bands", color="white")
    ax.set_ylabel("Price", color="white")
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    for label in ax.get_xticklabels():
        label.set_color("white")
        label.set_rotation(45)
        label.set_ha("right")
    for label in ax.get_yticklabels():
        label.set_color("white")
    ax.legend(facecolor="black", edgecolor="white")


def plot_volume(ax, df: pd.DataFrame):
    dates = mdates.date2num(df.index.to_pydatetime())
    ax.bar(dates, df["Volume"], color="dodgerblue", alpha=0.7)
    ax.set_ylabel("Volume", color="white")
    ax.grid(True, color="white", alpha=0.15)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    for label in ax.get_xticklabels():
        label.set_color("white")
        label.set_rotation(45)
        label.set_ha("right")
    for label in ax.get_yticklabels():
        label.set_color("white")


def plot_rsi_macd(ax, df: pd.DataFrame):
    df = df.copy()
    df["RSI_14"] = calculate_rsi(df["Close"], 14)
    macd_df = calculate_macd(df["Close"])

    dates = mdates.date2num(df.index.to_pydatetime())
    macd_dates = mdates.date2num(macd_df.index.to_pydatetime())

    ax.plot(dates, df["RSI_14"], color="violet", linewidth=1.0, label="RSI 14")
    ax.axhline(70, color="red", linestyle="--", linewidth=0.8)
    ax.axhline(30, color="green", linestyle="--", linewidth=0.8)

    ax.plot(macd_dates, macd_df["MACD"], color="yellow", linewidth=1.0, label="MACD")
    ax.plot(macd_dates, macd_df["Signal"], color="orange", linewidth=1.0, label="MACD Signal")

    ax.set_ylabel("RSI / MACD", color="white")
    ax.grid(True, color="white", alpha=0.15)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    for label in ax.get_xticklabels():
        label.set_color("white")
        label.set_rotation(45)
        label.set_ha("right")
    for label in ax.get_yticklabels():
        label.set_color("white")
    ax.legend(facecolor="black", edgecolor="white")


def plot_return_hist_with_var(ax, returns: pd.Series, alpha: float = 0.95):
    clean_ret = returns.dropna()
    if clean_ret.empty:
        ax.text(0.5, 0.5, "Not enough data", transform=ax.transAxes, ha="center", va="center")
        return

    var = np.percentile(clean_ret, (1 - alpha) * 100)
    cvar = clean_ret[clean_ret <= var].mean()

    logger.info(
        f"VaR {int(alpha*100)}% (1-day): {var:.4f} ({var*100:.2f}%), "
        f"CVaR: {cvar:.4f} ({cvar*100:.2f}%)"
    )

    ax.hist(clean_ret, bins=60, color="steelblue", alpha=0.8)
    ax.axvline(var, color="red", linestyle="--", linewidth=1.2, label=f"VaR {int(alpha*100)}%")
    ax.set_title("Return Distribution with VaR Marker", color="white")
    ax.set_xlabel("Return", color="white")
    ax.set_ylabel("Frequency", color="white")
    for label in ax.get_xticklabels():
        label.set_color("white")
    for label in ax.get_yticklabels():
        label.set_color("white")
    ax.legend(facecolor="black", edgecolor="white")


def plot_vol_surface(ax, log_ret: pd.Series, cbar=None):
    windows = [5, 10, 20, 40]
    vol_matrix = []
    for w in windows:
        vol_series = log_ret.rolling(window=w).std() * np.sqrt(252) * 100.0
        vol_matrix.append(vol_series.values)
    vol_matrix = np.array(vol_matrix)
    valid_idx = ~np.isnan(vol_matrix).all(axis=0)
    if not valid_idx.any():
        ax.text(0.5, 0.5, "Not enough data", transform=ax.transAxes, ha="center", va="center")
        return cbar
    vol_matrix = vol_matrix[:, valid_idx]

    X, Y = np.meshgrid(np.arange(vol_matrix.shape[1]), windows)

    surf = ax.plot_surface(X, Y, vol_matrix, cmap="viridis")
    ax.set_title("Realized Volatility Surface (annualized %, window vs time)", color="white")
    ax.set_xlabel("Time index", color="white")
    ax.set_ylabel("Window (days)", color="white")
    ax.set_zlabel("Volatility %", color="white")
    ax.tick_params(colors="white")

    # Create a single colorbar and update it each refresh
    if cbar is None:
        cbar = plt.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    else:
        cbar.update_normal(surf)

    return cbar


def main():
    parser = argparse.ArgumentParser(description="Swiggy Quant EDA (pure matplotlib).")
    parser.add_argument("--symbol", type=str, default="SWIGGY.BO", help="Ticker symbol")
    parser.add_argument("--period", type=str, default="6mo", help="Lookback period (e.g. 6mo, 1y)")
    parser.add_argument("--interval", type=str, default="1d", help="Data interval (e.g. 1d, 1h)")
    args = parser.parse_args()

    symbol = args.symbol.upper()

    # Global style
    plt.rcParams.update(
        {
            "figure.facecolor": "black",
            "axes.facecolor": "black",
            "axes.edgecolor": "white",
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
            "text.color": "white",
            "font.size": 11,
        }
    )

    # Prepare figures and axes once
    plt.ion()

    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig1.suptitle(f"{symbol} Quant EDA - Price / Volume / Indicators", color="white")

    fig2, ax4 = plt.subplots(figsize=(10, 5))

    fig3 = plt.figure(figsize=(10, 6))
    ax5 = fig3.add_subplot(111, projection="3d")
    vol_cbar = None

    first_run = True

    try:
        while True:
            loop_start = time.time()
            logger.info(f"Refreshing data for {symbol}, period={args.period}, interval={args.interval}")

            df = fetch_data(symbol, args.period, args.interval)
            df = df.copy()
            df["Return"] = df["Close"].pct_change()
            df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))

            last_close = float(df["Close"].iloc[-1])
            first_close = float(df["Close"].iloc[0])
            tot_ret = (last_close - first_close) / first_close * 100.0 if first_close != 0 else 0.0
            ann_vol = df["LogReturn"].std(ddof=1) * np.sqrt(252) * 100.0

            # Print summary once at start, or occasionally if you like
            if first_run:
                print(f"Symbol: {symbol}")
                print(f"Last close: {last_close:.2f}")
                print(f"Period return: {tot_ret:+.2f}%")
                print(f"Annualized volatility (log returns): {ann_vol:.2f}%")
                print(f"Observations: {len(df)}")

            # Preserve zoom/pan if not first run
            if first_run:
                xlims = {}
                ylims = {}
            else:
                xlims = {
                    "ax1": ax1.get_xlim(),
                    "ax2": ax2.get_xlim(),
                    "ax3": ax3.get_xlim(),
                    "ax4": ax4.get_xlim(),
                    "ax5": ax5.get_xlim(),
                }
                ylims = {
                    "ax1": ax1.get_ylim(),
                    "ax2": ax2.get_ylim(),
                    "ax3": ax3.get_ylim(),
                    "ax4": ax4.get_ylim(),
                    "ax5": ax5.get_ylim(),
                }

            # Clear and redraw
            for ax in [ax1, ax2, ax3, ax4, ax5]:
                ax.clear()

            plot_candles_with_indicators(ax1, df, symbol)
            plot_volume(ax2, df)
            plot_rsi_macd(ax3, df)
            plot_return_hist_with_var(ax4, df["Return"])
            vol_cbar = plot_vol_surface(ax5, df["LogReturn"].dropna(), vol_cbar)

            # Restore zoom/pan limits
            if not first_run and xlims:
                try:
                    ax1.set_xlim(xlims["ax1"])
                    ax2.set_xlim(xlims["ax2"])
                    ax3.set_xlim(xlims["ax3"])
                    ax4.set_xlim(xlims["ax4"])
                    ax5.set_xlim(xlims["ax5"])

                    ax1.set_ylim(ylims["ax1"])
                    ax2.set_ylim(ylims["ax2"])
                    ax3.set_ylim(ylims["ax3"])
                    ax4.set_ylim(ylims["ax4"])
                    ax5.set_ylim(ylims["ax5"])
                except Exception:
                    # If axis domains changed too much, fall back to autoscale
                    pass

            fig1.tight_layout(rect=[0, 0, 1, 0.96])
            fig2.tight_layout()
            fig3.tight_layout()

            plt.draw()
            plt.pause(0.01)

            first_run = False

            sleep_time = max(0.0, REFRESH_SECONDS - (time.time() - loop_start))
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("Swiggy Quant EDA CLI stopped by user.")
    finally:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()


