"""Matplotlib styling helpers for StochParticles example figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STYLE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "legend.frameon": False,
    "figure.dpi": 600,
    "savefig.dpi": 600,
}


def apply_publication_style() -> None:
    plt.rcParams.update(STYLE_RC)


apply_publication_style()


PALETTE = {
    "aerosol": "#1f77b4",
    "cloud": "#17becf",
    "activation_only": "#2ca02c",
    "activation_with_coagulation": "#d62728",
    "so4": "#6baed6",
    "bc": "#252525",
    "brownian": "#9467bd",
    "gravitational": "#8c564b",
    "turbulent": "#ff7f0e",
    "neutral": "#7f7f7f",
    "positive": "#2ca02c",
    "negative": "#d62728",
}


def add_panel_label(ax, label: str, x: float = -0.10, y: float = 1.04):
    return ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def save_png(fig, path: str | Path, dpi: int = 600) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def finish_panel(fig, path: str | Path) -> None:
    fig.tight_layout()
    save_png(fig, path, dpi=600)
    plt.close(fig)


def draw_replicates_with_mean(
    ax, x, y_matrix, color: str, label: str, lw_mean: float = 1.8
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    rows = np.asarray(y_matrix, dtype=float)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    for row in rows:
        ax.plot(x, row, color=color, lw=0.7, alpha=0.25)
    mean = np.nanmean(rows, axis=0)
    ax.plot(x, mean, color=color, lw=lw_mean, label=label)
    return mean
