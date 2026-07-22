"""Analysis helpers for coagulation example plots."""

from __future__ import annotations

import numpy as np

from .stochparticles_io import ReplicateData, stack_common_time


def _normalize_rows(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=float)
    first = rows[:, :1]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(rows, first, out=np.full_like(rows, np.nan), where=first != 0.0)


def normalized_number(reps: list[ReplicateData]) -> tuple[np.ndarray, np.ndarray]:
    time, rows = stack_common_time(reps, "number_concentration")
    return time, _normalize_rows(rows)


def normalized_diameter(
    reps: list[ReplicateData], key: str = "mean_diameter"
) -> tuple[np.ndarray, np.ndarray]:
    time, rows = stack_common_time(reps, key)
    return time, _normalize_rows(rows)


def kernel_fraction_series(
    reps: list[ReplicateData],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    names = {
        "Brownian": "kernel_fraction_brownian",
        "Gravitational": "kernel_fraction_gravitational",
        "Turbulent": "kernel_fraction_turbulent",
    }
    series: dict[str, np.ndarray] = {}
    time: np.ndarray | None = None
    for label, array_name in names.items():
        current_time, rows = stack_common_time(reps, array_name)
        if time is None:
            time = current_time
        series[label] = rows
    return np.array([], dtype=float) if time is None else time, series


def kernel_total_bars(reps: list[ReplicateData]) -> dict[str, np.ndarray]:
    time, series = kernel_fraction_series(reps)
    totals: dict[str, np.ndarray] = {}
    duration = float(time[-1] - time[0]) if time.size > 1 else 0.0
    for label, rows in series.items():
        if time.size > 1 and duration > 0:
            totals[label] = np.trapezoid(rows, time, axis=1) / duration
        else:
            totals[label] = np.nanmean(rows, axis=1)
    return totals
