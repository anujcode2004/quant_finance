"""
Eternal & Swiggy Quant Terminal
===============================

A rich matplotlib-based "terminal" dashboard (no Streamlit) for:
- Eternal Limited (ETERNAL.NS)
- Swiggy (SWIGGY.BO)

It provides:
- Candlestick charts with SMA 9 and 21 for both symbols
- Volume panels with rolling volume averages
- Volatility analysis (rolling std of log returns) for both symbols
- Correlation heatmap of returns / volume changes
- Large, terminal-style fonts and dark theme
- Real-time refresh every 1 second
- Display of API fetch latency and compute loop time

Run with:

    cd /Users/inavlabs/Documents/inavlabs/quant_finance
    python -m strategies.eternal_swiggy_quant_terminal

Stop with Ctrl+C.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Tuple

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

from assets.stock import Stock
from indicators.technical_indicators import calculate_sma
from telemetries.logger import logger


ETERNAL = "ETERNAL.NS"
SWIGGY = "SWIGGY.BO"
REFRESH_SECONDS = 1


# Global visual settings for a "trading terminal" look
plt.rcParams.update(
    {
        "figure.facecolor": "black",
        "axes.facecolor": "black",
        "axes.edgecolor": "white",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "text.color": "white",
        "font.size": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
    }
)


def _plot_candles(ax, df: pd.DataFrame, title: str):
    """Draw candlesticks + SMA 9 & 21 on given axis."""
    if df.empty:
        ax.text(
            0.5,
            0.5,
            "No data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="white",
            fontsize=14,
        )
        return

    df = df.copy()
    df["SMA_9"] = calculate_sma(df["Close"], 9)
    df["SMA_21"] = calculate_sma(df["Close"], 21)

    dates = mdates.date2num(df.index.to_pydatetime())

    ax.grid(True, color="white", alpha=0.15)

    candle_width = 0.6 * (dates[1] - dates[0]) if len(dates) > 1 else 0.0005

    for x, row in zip(dates, df.itertuples()):
        o = row.Open
        h = row.High
        l = row.Low
        c = row.Close

        color = "lime" if c >= o else "red"

        # Wick
        ax.vlines(x, l, h, color=color, linewidth=0.8, alpha=0.9)

        # Body
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

    # SMAs
    ax.plot(dates, df["SMA_9"], color="cyan", linewidth=1.2, label="SMA 9")
    ax.plot(dates, df["SMA_21"], color="orange", linewidth=1.2, label="SMA 21")

    ax.set_title(title)
    ax.set_ylabel("Price")
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    for label in ax.get_xticklabels():
        label.set_color("white")
    for label in ax.get_yticklabels():
        label.set_color("white")

    ax.legend(loc="upper left")


def _plot_volume(ax, df: pd.DataFrame, label_prefix: str):
    """Volume bars + rolling volume average."""
    if df.empty:
        return

    df = df.copy()
    df["Vol_MA_20"] = df["Volume"].rolling(window=20).mean()
    dates = mdates.date2num(df.index.to_pydatetime())

    ax.bar(dates, df["Volume"], color="dodgerblue", alpha=0.6, label=f"{label_prefix} Volume")
    ax.plot(dates, df["Vol_MA_20"], color="yellow", linewidth=1.2, label="Vol MA 20")

    ax.set_ylabel("Volume")
    ax.grid(True, color="white", alpha=0.15)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    for label in ax.get_xticklabels():
        label.set_color("white")
    for label in ax.get_yticklabels():
        label.set_color("white")
    ax.legend(loc="upper left")


def _rolling_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """20-period rolling std of log returns."""
    if df.empty:
        return pd.Series(dtype=float)
    close = df["Close"].astype(float)
    log_ret = np.log(close / close.shift(1)).dropna()
    return log_ret.rolling(window=window).std()


def _plot_volatility(ax, df_et: pd.DataFrame, df_sw: pd.DataFrame):
    """Volatility comparison plot."""
    vol_et = _rolling_volatility(df_et)
    vol_sw = _rolling_volatility(df_sw)

    if vol_et.empty and vol_sw.empty:
        ax.text(
            0.5,
            0.5,
            "No volatility data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="white",
        )
        return

    if not vol_et.empty:
        dates_et = mdates.date2num(vol_et.index.to_pydatetime())
        ax.plot(dates_et, vol_et, color="cyan", label=f"{ETERNAL} Vol (20)")

    if not vol_sw.empty:
        dates_sw = mdates.date2num(vol_sw.index.to_pydatetime())
        ax.plot(dates_sw, vol_sw, color="magenta", label=f"{SWIGGY} Vol (20)")

    ax.set_title("Rolling Volatility (log returns, 20 periods)")
    ax.set_ylabel("Volatility")
    ax.grid(True, color="white", alpha=0.15)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    for label in ax.get_xticklabels():
        label.set_color("white")
    for label in ax.get_yticklabels():
        label.set_color("white")
    ax.legend(loc="upper left")


def _plot_correlation_heatmap(ax, df_et: pd.DataFrame, df_sw: pd.DataFrame, cbar=None):
    """Correlation heatmap of returns and volume changes.

    Args:
        ax: Matplotlib Axes to draw the heatmap on.
        df_et: Eternal dataframe.
        df_sw: Swiggy dataframe.
        cbar: Optional shared Colorbar; created on first call and updated later.

    Returns:
        Colorbar instance (to be reused on subsequent calls).
    """
    if df_et.empty or df_sw.empty:
        ax.text(
            0.5,
            0.5,
            "Insufficient data for correlation",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="white",
        )
        return cbar

    # Align on time index
    et = df_et[["Close", "Volume"]].rename(columns=lambda c: f"ET_{c}")
    sw = df_sw[["Close", "Volume"]].rename(columns=lambda c: f"SW_{c}")
    joined = et.join(sw, how="inner")

    # Compute log returns and volume pct change
    joined["ET_ret"] = np.log(joined["ET_Close"] / joined["ET_Close"].shift(1))
    joined["SW_ret"] = np.log(joined["SW_Close"] / joined["SW_Close"].shift(1))
    joined["ET_vol_chg"] = joined["ET_Volume"].pct_change()
    joined["SW_vol_chg"] = joined["SW_Volume"].pct_change()

    corr = joined[["ET_ret", "SW_ret", "ET_vol_chg", "SW_vol_chg"]].corr()

    # Use a green-blue style theme for the heatmap
    im = ax.imshow(corr, cmap="YlGnBu", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", color="white")
    ax.set_yticklabels(corr.index, color="white")
    ax.set_title("Correlation Heatmap (Returns & Volume Changes)")

    # Annotate cells
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(
                j,
                i,
                f"{corr.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
            )

    # Colorbar: create once and update the same bar each refresh
    if cbar is None:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.get_yticklabels(), color="white")
    else:
        cbar.update_normal(im)

    return cbar


def main():
    et_stock = Stock(ETERNAL)
    sw_stock = Stock(SWIGGY)

    if not et_stock.validate():
        print(f"Symbol validation failed for {ETERNAL}")
        return
    if not sw_stock.validate():
        print(f"Symbol validation failed for {SWIGGY}")
        return

    plt.ion()
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(15, 9),
        gridspec_kw={"height_ratios": [3, 2, 2]},
    )
    fig.patch.set_facecolor("black")

    (ax_price_et, ax_price_sw), (ax_vol_et, ax_vol_sw), (ax_volatility, ax_corr) = axes

    # Status text area at top
    status_text = fig.text(
        0.01,
        0.97,
        "",
        fontsize=12,
        color="yellow",
        ha="left",
        va="top",
    )

    first_run = True
    heatmap_colorbar = None

    try:
        while True:
            loop_start = time.time()

            # Fetch data with latency tracking
            t0 = time.time()
            df_et = et_stock.get_historical_data(period="5d", interval="5m")
            t1 = time.time()
            df_sw = sw_stock.get_historical_data(period="5d", interval="5m")
            t2 = time.time()

            latency_et = (t1 - t0) * 1000.0
            latency_sw = (t2 - t1) * 1000.0

            if df_et.empty or df_sw.empty:
                logger.warning("One or both data frames are empty for Eternal/Swiggy.")

            # Preserve zoom/pan if not first run
            if first_run:
                xlims: Dict[str, Tuple[float, float]] = {}
                ylims: Dict[str, Tuple[float, float]] = {}
            else:
                xlims = {
                    "ax_price_et": ax_price_et.get_xlim(),
                    "ax_price_sw": ax_price_sw.get_xlim(),
                    "ax_vol_et": ax_vol_et.get_xlim(),
                    "ax_vol_sw": ax_vol_sw.get_xlim(),
                    "ax_volatility": ax_volatility.get_xlim(),
                }
                ylims = {
                    "ax_price_et": ax_price_et.get_ylim(),
                    "ax_price_sw": ax_price_sw.get_ylim(),
                    "ax_vol_et": ax_vol_et.get_ylim(),
                    "ax_vol_sw": ax_vol_sw.get_ylim(),
                    "ax_volatility": ax_volatility.get_ylim(),
                }

            # Clear and redraw all axes
            for ax in [ax_price_et, ax_price_sw, ax_vol_et, ax_vol_sw, ax_volatility, ax_corr]:
                ax.clear()

            _plot_candles(ax_price_et, df_et, f"{ETERNAL} - Candles & SMA")
            _plot_candles(ax_price_sw, df_sw, f"{SWIGGY} (Swiggy Proxy) - Candles & SMA")
            _plot_volume(ax_vol_et, df_et, "ETERNAL")
            _plot_volume(ax_vol_sw, df_sw, "SWIGGY")
            _plot_volatility(ax_volatility, df_et, df_sw)
            heatmap_colorbar = _plot_correlation_heatmap(ax_corr, df_et, df_sw, heatmap_colorbar)

            # Price labels with green/red coloring based on latest move
            def _price_label(ax, df, symbol: str):
                if df.empty or len(df) < 2:
                    return
                last = float(df["Close"].iloc[-1])
                prev = float(df["Close"].iloc[-2])
                delta = last - prev
                pct = (delta / prev) * 100.0 if prev != 0 else 0.0
                color = "lime" if delta >= 0 else "red"
                txt = f"{symbol}  {last:.2f}  ({delta:+.2f}, {pct:+.2f}%)"
                ax.text(
                    0.01,
                    0.97,
                    txt,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    color=color,
                    fontsize=13,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", edgecolor=color, alpha=0.6),
                )

            _price_label(ax_price_et, df_et, ETERNAL)
            _price_label(ax_price_sw, df_sw, SWIGGY)

            # On the very first draw, automatically zoom in closer to the latest prices
            # (e.g., last ~30% of available range) so you see current action instead
            # of the entire 5d window zoomed out.
            if first_run:
                try:
                    def _zoom_latest(ax, df):
                        if df.empty or len(df.index) < 5:
                            return
                        dates = mdates.date2num(df.index.to_pydatetime())
                        n = len(dates)
                        start_idx = int(n * 0.7)  # keep roughly last 30%
                        left = dates[start_idx]
                        right = dates[-1]
                        ax.set_xlim(left, right)

                    _zoom_latest(ax_price_et, df_et)
                    _zoom_latest(ax_price_sw, df_sw)
                    _zoom_latest(ax_vol_et, df_et)
                    _zoom_latest(ax_vol_sw, df_sw)
                    _zoom_latest(ax_volatility, df_et)
                except Exception:
                    # If anything goes wrong, fall back to default autoscale
                    pass

                # After applying initial zoom once, from now on we just preserve user zoom
                first_run = False

            # Reapply stored limits to preserve zoom/pan
            if xlims:
                try:
                    ax_price_et.set_xlim(xlims["ax_price_et"])
                    ax_price_sw.set_xlim(xlims["ax_price_sw"])
                    ax_vol_et.set_xlim(xlims["ax_vol_et"])
                    ax_vol_sw.set_xlim(xlims["ax_vol_sw"])
                    ax_volatility.set_xlim(xlims["ax_volatility"])

                    ax_price_et.set_ylim(ylims["ax_price_et"])
                    ax_price_sw.set_ylim(ylims["ax_price_sw"])
                    ax_vol_et.set_ylim(ylims["ax_vol_et"])
                    ax_vol_sw.set_ylim(ylims["ax_vol_sw"])
                    ax_volatility.set_ylim(ylims["ax_volatility"])
                except Exception:
                    # If shapes changed drastically, fall back to autoscale
                    pass

            loop_end = time.time()
            loop_latency = (loop_end - loop_start) * 1000.0

            # Update status text
            status_msg = (
                f"{ETERNAL} fetch: {latency_et:.1f} ms | "
                f"{SWIGGY} fetch: {latency_sw:.1f} ms | "
                f"loop: {loop_latency:.1f} ms | "
                f"last update: {time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            status_text.set_text(status_msg)

            fig.tight_layout(rect=[0, 0, 1, 0.95])
            plt.draw()
            plt.pause(0.01)

            sleep_time = max(0.0, REFRESH_SECONDS - (time.time() - loop_start))
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("Eternal & Swiggy quant terminal stopped by user.")
    finally:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()


