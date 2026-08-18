"""Smoothing and log-diameter density helpers without SciPy."""

from __future__ import annotations

import numpy as np

from .stochparticles_io import ReplicateData, clean_sample_matrix, require_arrays


def dense_log_grid(min_d: float, max_d: float, n: int = 240) -> np.ndarray:
    if n < 2:
        raise ValueError("n must be at least 2")
    if not np.isfinite(min_d) or not np.isfinite(max_d) or min_d <= 0 or max_d <= 0:
        raise ValueError("diameter bounds must be positive finite values")
    if min_d >= max_d:
        max_d = min_d * 1.01
    return np.logspace(np.log10(min_d), np.log10(max_d), n)


def kde_log_diameter(
    diameters: np.ndarray,
    eval_diameters: np.ndarray,
    volume: float,
    bandwidth_factor: float = 1.0,
    max_bandwidth: float | None = None,
) -> np.ndarray:
    eval_diameters = np.asarray(eval_diameters, dtype=float)
    result = np.zeros_like(eval_diameters, dtype=float)
    valid_eval = np.isfinite(eval_diameters) & (eval_diameters > 0.0)
    samples = np.asarray(diameters, dtype=float).ravel()
    samples = samples[np.isfinite(samples) & (samples > 0.0)]
    if samples.size == 0 or not np.isfinite(volume) or volume <= 0:
        return result

    log_samples = np.log10(samples)
    log_eval = np.log10(eval_diameters[valid_eval])
    if samples.size == 1:
        bandwidth = 0.05
    else:
        std = np.std(log_samples, ddof=1)
        bandwidth = 1.06 * std * samples.size ** (-1.0 / 5.0)
        if not np.isfinite(bandwidth) or bandwidth <= 0:
            bandwidth = 0.05
    bandwidth *= max(float(bandwidth_factor), np.finfo(float).eps)
    if max_bandwidth is not None:
        max_bandwidth = float(max_bandwidth)
        if not np.isfinite(max_bandwidth) or max_bandwidth <= 0.0:
            raise ValueError("max_bandwidth must be positive and finite")
        bandwidth = min(bandwidth, max_bandwidth)

    z = (log_eval[:, None] - log_samples[None, :]) / bandwidth
    kernel = np.exp(-0.5 * z * z) / (np.sqrt(2.0 * np.pi) * bandwidth)
    result[valid_eval] = np.sum(kernel, axis=1) / volume
    return result


def smooth_time(values: np.ndarray, passes: int = 2) -> np.ndarray:
    smoothed = np.asarray(values, dtype=float).copy()
    if passes <= 0 or smoothed.shape[0] < 3:
        return smoothed
    kernel_shape = (3,) + (1,) * (smoothed.ndim - 1)
    kernel = np.array([0.25, 0.5, 0.25], dtype=float).reshape(kernel_shape)
    pad_width = [(1, 1)] + [(0, 0)] * (smoothed.ndim - 1)
    for _ in range(passes):
        padded = np.pad(smoothed, pad_width, mode="edge")
        smoothed = (
            padded[:-2] * kernel[0]
            + padded[1:-1] * kernel[1]
            + padded[2:] * kernel[2]
        )
    return smoothed


def linear_edges_from_centers(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1 or centers.size < 2:
        raise ValueError("centers must be a 1D array with at least two values")
    deltas = np.diff(centers)
    if np.any(~np.isfinite(deltas)) or np.any(deltas <= 0.0):
        raise ValueError("centers must be finite and strictly increasing")
    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * deltas[0]
    edges[-1] = centers[-1] + 0.5 * deltas[-1]
    return edges


def log_edges_from_centers(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1 or centers.size < 2:
        raise ValueError("centers must be a 1D array with at least two values")
    if np.any(~np.isfinite(centers)) or np.any(centers <= 0.0):
        raise ValueError("log centers must be positive and finite")
    log_edges = linear_edges_from_centers(np.log10(centers))
    return np.power(10.0, log_edges)


def _volume_at_time(rep: ReplicateData, time: np.ndarray) -> np.ndarray:
    volume = rep.arrays.get("volume")
    if volume is not None:
        values = np.asarray(volume, dtype=float)
        rep_time = np.asarray(rep.arrays.get("time", time), dtype=float)
        if values.ndim == 1 and values.size == time.size:
            return values
        if values.ndim == 1 and values.size == rep_time.size:
            return np.interp(time, rep_time, values)
        if values.size == 1:
            return np.full(time.size, float(values.ravel()[0]))
    attr_volume = rep.attrs.get("volume", 1.0)
    return np.full(time.size, float(attr_volume))


def replicate_mean_kde_heatmap(
    reps: list[ReplicateData],
    diameter_key: str = "diameter_samples",
    grid: np.ndarray | None = None,
    bandwidth_factor: float = 1.0,
    max_bandwidth: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not reps:
        empty_grid = np.asarray(grid if grid is not None else [], dtype=float)
        return np.array([], dtype=float), empty_grid, np.empty((0, empty_grid.size))

    for rep in reps:
        require_arrays(rep, ("time", diameter_key))
    common_time = np.asarray(reps[0].arrays["time"], dtype=float)

    if grid is None:
        all_samples = []
        for rep in reps:
            samples = np.asarray(rep.arrays[diameter_key], dtype=float)
            samples = samples[np.isfinite(samples) & (samples > 0.0)]
            if samples.size:
                all_samples.append(samples)
        if not all_samples:
            raise ValueError(f"no positive finite samples found for {diameter_key!r}")
        pooled = np.concatenate(all_samples)
        grid = dense_log_grid(float(np.min(pooled)), float(np.max(pooled)))
    else:
        grid = np.asarray(grid, dtype=float)

    heatmaps = []
    for rep in reps:
        rep_time = np.asarray(rep.arrays["time"], dtype=float)
        samples_by_time = clean_sample_matrix(rep.arrays[diameter_key])
        volumes = _volume_at_time(rep, rep_time)
        heatmap = np.vstack(
            [
                kde_log_diameter(
                    samples,
                    grid,
                    volumes[idx],
                    bandwidth_factor,
                    max_bandwidth=max_bandwidth,
                )
                for idx, samples in enumerate(samples_by_time)
            ]
        )
        if not np.array_equal(rep_time, common_time):
            interpolated = np.empty((common_time.size, grid.size), dtype=float)
            for col in range(grid.size):
                interpolated[:, col] = np.interp(common_time, rep_time, heatmap[:, col])
            heatmap = interpolated
        heatmaps.append(heatmap)

    mean_heatmap = np.nanmean(np.stack(heatmaps, axis=0), axis=0)
    return common_time, grid, smooth_time(mean_heatmap)


def replicate_kde_heatmap(
    rep: ReplicateData,
    diameter_key: str = "diameter_samples",
    grid: np.ndarray | None = None,
    bandwidth_factor: float = 1.0,
    max_bandwidth: float | None = None,
    smooth_passes: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    require_arrays(rep, ("time", diameter_key))
    rep_time = np.asarray(rep.arrays["time"], dtype=float)
    samples_by_time = clean_sample_matrix(rep.arrays[diameter_key])
    if len(samples_by_time) != rep_time.size:
        raise ValueError(
            f"Expected one {diameter_key!r} sample row per time value for "
            f"case={rep.case_name!r}, replicate={rep.replicate_name!r}; "
            f"got {len(samples_by_time)} sample rows and {rep_time.size} time values"
        )

    if grid is None:
        all_samples = [samples for samples in samples_by_time if samples.size]
        if not all_samples:
            raise ValueError(f"no positive finite samples found for {diameter_key!r}")
        pooled = np.concatenate(all_samples)
        grid = dense_log_grid(float(np.min(pooled)), float(np.max(pooled)))
    else:
        grid = np.asarray(grid, dtype=float)

    volumes = _volume_at_time(rep, rep_time)
    heatmap = np.vstack(
        [
            kde_log_diameter(
                samples,
                grid,
                volumes[idx],
                bandwidth_factor,
                max_bandwidth=max_bandwidth,
            )
            for idx, samples in enumerate(samples_by_time)
        ]
    )
    return rep_time, grid, smooth_time(heatmap, passes=smooth_passes)


def select_replicate_for_heatmap(
    reps: list[ReplicateData], preferred_index: int = 0
) -> ReplicateData:
    if not reps:
        raise ValueError("cannot select a heatmap replicate from an empty list")
    index = min(max(int(preferred_index), 0), len(reps) - 1)
    return reps[index]
