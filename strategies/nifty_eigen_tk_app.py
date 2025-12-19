"""
Tkinter GUI for real‑time eigenvector / PCA analysis on top Indian stocks.

Features
--------
- Uses the backend logic from `nifty_eigen_backend.py`
- Universe: core large‑cap Indian names (NSE symbols)
- Every REFRESH_SECONDS, the app:
    * downloads latest daily prices from Yahoo (`<SYMBOL>.NS`)
    * recomputes log‑returns, correlation matrix
    * recomputes eigenvalues / eigenvectors and variance explained
    * updates three live matplotlib views:
         1) Scree plot of eigenvalues
         2) Cumulative variance explained
         3) PC1 vs PC2 stock scatter

This is intentionally simple and robust – good for "live dashboard" use.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from .nifty_eigen_backend import (
    NIFTY_CORE_SYMBOLS,
    quick_eigen_for_default_universe,
    eigen_analysis_from_returns,
    fetch_returns_for_universe,
)


REFRESH_SECONDS = 120  # update every 2 minutes (tweak as needed)


class NiftyEigenApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Nifty Eigenvectors & PCA – Live Market Structure")
        self.geometry("1400x900")

        # State
        self.analysis_result = None
        self.after_job = None

        # Layout
        self._build_controls()
        self._build_figures()

        # Kick off first analysis
        self.run_analysis(initial=True)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_controls(self):
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

        ttk.Label(top, text="Universe:", font=("Arial", 10, "bold")).pack(
            side=tk.LEFT, padx=(0, 4)
        )

        self.universe_label = ttk.Label(
            top,
            text=f"{len(NIFTY_CORE_SYMBOLS)} Nifty‑style large caps (NSE)",
        )
        self.universe_label.pack(side=tk.LEFT, padx=(0, 16))

        ttk.Label(top, text="Period:").pack(side=tk.LEFT, padx=(0, 2))
        self.period_var = tk.StringVar(value="1y")
        period_entry = ttk.Entry(top, width=6, textvariable=self.period_var)
        period_entry.pack(side=tk.LEFT, padx=(0, 8))

        ttk.Label(top, text="Interval:").pack(side=tk.LEFT, padx=(0, 2))
        self.interval_var = tk.StringVar(value="1d")
        interval_entry = ttk.Entry(top, width=6, textvariable=self.interval_var)
        interval_entry.pack(side=tk.LEFT, padx=(0, 8))

        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(
            top, textvariable=self.status_var, foreground="#00ffcc"
        )
        status_label.pack(side=tk.RIGHT, padx=(8, 0))

        refresh_btn = ttk.Button(
            top,
            text="Refresh Now",
            command=lambda: self.run_analysis(initial=False, manual=True),
        )
        refresh_btn.pack(side=tk.RIGHT, padx=4)

    def _build_figures(self):
        # Use a ttk.Notebook with tabs
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Scree + cumulative variance figure
        self.fig1, (self.ax_scree, self.ax_cumvar) = plt.subplots(
            1, 2, figsize=(12, 4)
        )
        self.fig1.suptitle("Eigenvalues & Variance Explained", fontsize=12)
        self.fig1.tight_layout()

        frame1 = ttk.Frame(notebook)
        notebook.add(frame1, text="Eigenvalues")
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=frame1)
        self.canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # PC1 vs PC2 scatter
        self.fig2, self.ax_pc = plt.subplots(1, 1, figsize=(6, 4))
        self.fig2.suptitle("Stocks in PC1–PC2 Space", fontsize=12)
        self.fig2.tight_layout()

        frame2 = ttk.Frame(notebook)
        notebook.add(frame2, text="PC1 vs PC2")
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=frame2)
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def run_analysis(self, initial: bool = False, manual: bool = False):
        """Fetch latest data and recompute eigen / PCA."""
        period = self.period_var.get().strip() or "1y"
        interval = self.interval_var.get().strip() or "1d"

        self.status_var.set(f"Fetching data… ({period}, {interval})")
        self.update_idletasks()

        try:
            returns_df = fetch_returns_for_universe(
                list(NIFTY_CORE_SYMBOLS.values()),
                period=period,
                interval=interval,
            )
            if returns_df.empty:
                raise RuntimeError("No data returned for universe.")

            self.analysis_result = eigen_analysis_from_returns(returns_df)
            self._update_plots()
            self.status_var.set(
                f"Last update: {returns_df.index[-1].date()} "
                f"({returns_df.shape[0]} days, {returns_df.shape[1]} stocks)"
            )
        except Exception as e:
            self.status_var.set("Error")
            messagebox.showerror(
                "Analysis Error",
                f"Failed to run eigen analysis:\n{e}",
            )

        # Schedule next automatic refresh
        if not manual:
            if self.after_job is not None:
                self.after_cancel(self.after_job)
            self.after_job = self.after(REFRESH_SECONDS * 1000, self.run_analysis)

    def _update_plots(self):
        """Redraw matplotlib figures with current analysis."""
        res = self.analysis_result
        if res is None:
            return

        eigenvalues = res.eigenvalues
        pve = res.pve
        cum_pve = res.pve_cumulative

        # Scree plot
        self.ax_scree.clear()
        x = np.arange(1, len(eigenvalues) + 1)
        self.ax_scree.plot(x, eigenvalues, "o-", color="cyan", linewidth=1.5)
        self.ax_scree.axhline(1.0, color="red", linestyle="--", linewidth=1)
        self.ax_scree.set_xlabel("Principal Component")
        self.ax_scree.set_ylabel("Eigenvalue")
        self.ax_scree.set_title("Scree Plot")
        self.ax_scree.grid(alpha=0.3)

        # Cumulative variance
        self.ax_cumvar.clear()
        self.ax_cumvar.plot(x, cum_pve, "o-", color="lime", linewidth=1.5)
        self.ax_cumvar.axhline(80, color="red", linestyle="--", linewidth=1)
        self.ax_cumvar.axhline(90, color="orange", linestyle="--", linewidth=1)
        self.ax_cumvar.set_xlabel("Number of PCs")
        self.ax_cumvar.set_ylabel("Cumulative variance (%)")
        self.ax_cumvar.set_ylim(0, 105)
        self.ax_cumvar.set_title("Cumulative Variance Explained")
        self.ax_cumvar.grid(alpha=0.3)

        self.fig1.tight_layout()
        self.canvas1.draw_idle()

        # PC1 vs PC2
        self.ax_pc.clear()
        pc_loadings = self._pc_loadings_matrix()
        xpc = pc_loadings["PC1"]
        ypc = pc_loadings["PC2"]

        sc = self.ax_pc.scatter(xpc, ypc, s=80, c=np.arange(len(xpc)), cmap="viridis")
        for sym in pc_loadings.index:
            self.ax_pc.annotate(
                sym,
                (pc_loadings.loc[sym, "PC1"], pc_loadings.loc[sym, "PC2"]),
                fontsize=7,
                alpha=0.7,
            )
        self.ax_pc.axhline(0, color="white", linestyle="--", linewidth=0.8, alpha=0.5)
        self.ax_pc.axvline(0, color="white", linestyle="--", linewidth=0.8, alpha=0.5)
        self.ax_pc.set_xlabel(f"PC1 (Var {pve[0]:.1f}%)")
        self.ax_pc.set_ylabel(f"PC2 (Var {pve[1]:.1f}%)")
        self.ax_pc.set_title("Stocks in PC1–PC2 space")
        self.ax_pc.grid(alpha=0.3)
        self.fig2.colorbar(sc, ax=self.ax_pc, label="Index")

        self.fig2.tight_layout()
        self.canvas2.draw_idle()

    def _pc_loadings_matrix(self):
        """Return DataFrame of PC loadings (stocks × PCs)."""
        res = self.analysis_result
        eigvecs = res.eigenvectors
        symbols = list(res.corr.columns)
        cols = [f"PC{i+1}" for i in range(eigvecs.shape[1])]
        import pandas as pd

        return pd.DataFrame(eigvecs, index=symbols, columns=cols)


def main():
    app = NiftyEigenApp()
    app.mainloop()


if __name__ == "__main__":
    main()


