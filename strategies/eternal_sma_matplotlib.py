"""
Eternal Limited (ETERNAL.NS) SMA 9 & 21 Candlestick Strategy

This is a pure matplotlib script (no Streamlit) that:
- Fetches ETERNAL.NS data from yfinance via the existing Stock class.
- Plots candlesticks with a black background and light grid.
- Overlays SMA 9 and SMA 21 on top of the candles.
- Refreshes the data and redraws the chart every 1 second.

Run with:

    cd /Users/inavlabs/Documents/inavlabs/quant_finance
    python -m strategies.eternal_sma_matplotlib

You can stop the script with Ctrl+C.
"""

import sys
import time
from pathlib import Path

# Ensure project root is on sys.path for imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

from assets.stock import Stock
from indicators.technical_indicators import calculate_sma
from telemetries.logger import logger


SYMBOL = "ETERNAL.NS"
REFRESH_SECONDS = 1  # Fixed 1-second refresh


def plot_candles_with_sma(ax, df):
    """Plot candlesticks and SMA lines on the given axes.

    Args:
        ax: Matplotlib axes instance.
        df: DataFrame with columns: Open, High, Low, Close, Volume
    """
    if df.empty:
        ax.text(
            0.5,
            0.5,
            "No data available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="white",
        )
        return

    df = df.copy()
    df["SMA_9"] = calculate_sma(df["Close"], 9)
    df["SMA_21"] = calculate_sma(df["Close"], 21)

    # Convert index to matplotlib dates
    dates = mdates.date2num(df.index.to_pydatetime())

    ax.set_facecolor("black")
    ax.grid(True, color="white", alpha=0.15)

    candle_width = 0.6 * (dates[1] - dates[0]) if len(dates) > 1 else 0.0005

    for x, row in zip(dates, df.itertuples()):
        open_price = row.Open
        high_price = row.High
        low_price = row.Low
        close_price = row.Close

        color = "lime" if close_price >= open_price else "red"

        # High-low line
        ax.vlines(x, low_price, high_price, color=color, linewidth=0.8, alpha=0.9)

        # Candle body
        lower = min(open_price, close_price)
        height = abs(close_price - open_price)
        if height == 0:
            height = 0.0001  # tiny body so it is still visible

        rect = Rectangle(
            (x - candle_width / 2, lower),
            candle_width,
            height,
            facecolor=color,
            edgecolor=color,
            linewidth=0.8,
        )
        ax.add_patch(rect)

    # Plot SMAs
    ax.plot(dates, df["SMA_9"], label="SMA 9", color="cyan", linewidth=1.2)
    ax.plot(dates, df["SMA_21"], label="SMA 21", color="orange", linewidth=1.2)

    ax.set_title(f"ETERNAL.NS - Candles with SMA 9 & 21", color="white")
    ax.set_xlabel("Time", color="white")
    ax.set_ylabel("Price", color="white")

    # Format x-axis
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_color("white")
    for label in ax.get_yticklabels():
        label.set_color("white")

    ax.legend(facecolor="black", edgecolor="white")


def main():
    stock = Stock(SYMBOL)
    if not stock.validate():
        print(f"Symbol validation failed for {SYMBOL}")
        return
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("black")

    first_run = True

    try:
        while True:
            try:
                # Use a short lookback with intraday interval that Yahoo supports
                df = stock.get_historical_data(period="5d", interval="5m")
            except Exception as exc:  # pragma: no cover
                logger.error(f"Error fetching data for {SYMBOL}: {exc}")
                time.sleep(REFRESH_SECONDS)
                continue

            if first_run:
                # First draw: let Matplotlib pick limits based on data
                ax.clear()
                plot_candles_with_sma(ax, df)
                first_run = False
            else:
                # Preserve current zoom/pan (axis limits) across refreshes
                cur_xlim = ax.get_xlim()
                cur_ylim = ax.get_ylim()

                ax.clear()
                plot_candles_with_sma(ax, df)

                # Reapply previous limits so user zoom/pan state is kept
                ax.set_xlim(cur_xlim)
                ax.set_ylim(cur_ylim)

            fig.tight_layout()
            plt.draw()
            plt.pause(0.01)  # allow GUI to update

            time.sleep(REFRESH_SECONDS)

    except KeyboardInterrupt:
        logger.info("Eternal SMA candlestick script stopped by user.")
    finally:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()


