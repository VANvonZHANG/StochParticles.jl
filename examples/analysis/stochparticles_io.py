"""HDF5 readers and small array utilities for StochParticles examples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


EXAMPLES_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = EXAMPLES_DIR / "data"
FIG_DIR = EXAMPLES_DIR / "fig"

TIME_MAJOR_2D_DATASETS = {
    "diameter_samples",
    "size_distribution_raw",
    "species_mass_concentration",
    "bc_mass_fraction_samples",
    "dry_diameter_samples",
    "activation_flag_samples",
}


@dataclass
class ReplicateData:
    case_name: str
    replicate_name: str
    arrays: dict[str, np.ndarray]
    attrs: dict[str, object]


def _decode_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.astype(str).item()
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode_value(value.item())
        return [_decode_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_decode_value(item) for item in value)
    return value


def _decode_attrs(attrs: h5py.AttributeManager) -> dict[str, object]:
    return {key: _decode_value(value) for key, value in attrs.items()}


def _replicate_dataset_group(rep_group: h5py.Group) -> h5py.Group:
    datasets = rep_group.get("datasets")
    if isinstance(datasets, h5py.Group):
        return datasets
    return rep_group


def _normalize_time_major(name: str, array: np.ndarray, n_time: int | None) -> np.ndarray:
    if n_time is None or name not in TIME_MAJOR_2D_DATASETS or array.ndim != 2:
        return array
    if array.shape[0] == n_time:
        # If both axes equal n_time the orientation is ambiguous, so keep the
        # stored layout rather than guessing.
        return array
    if array.shape[1] == n_time:
        return array.T
    raise ValueError(
        f"Known time-major dataset {name!r} has shape {array.shape}; "
        f"expected one axis to match n_time={n_time}"
    )


def read_scene(path: str | Path) -> dict[str, list[ReplicateData]]:
    """Read one example HDF5 scene into case-keyed replicate records."""

    path = Path(path)
    result: dict[str, list[ReplicateData]] = {}
    with h5py.File(path, "r") as h5:
        if "cases" not in h5:
            raise KeyError(f"{path} does not contain a /cases group")
        for case_name in sorted(h5["cases"].keys()):
            case_group = h5["cases"][case_name]
            if "replicates" not in case_group:
                continue
            case_attrs = _decode_attrs(case_group.attrs)
            reps: list[ReplicateData] = []
            replicate_group = case_group["replicates"]
            for replicate_name in sorted(replicate_group.keys()):
                rep_group = replicate_group[replicate_name]
                data_group = _replicate_dataset_group(rep_group)

                raw_arrays = {
                    name: dataset[()]
                    for name, dataset in data_group.items()
                    if isinstance(dataset, h5py.Dataset)
                }
                time = raw_arrays.get("time")
                n_time = int(len(time)) if time is not None and np.ndim(time) == 1 else None
                arrays = {
                    name: _normalize_time_major(name, np.asarray(array), n_time)
                    for name, array in raw_arrays.items()
                }
                attrs = {**case_attrs, **_decode_attrs(rep_group.attrs)}
                reps.append(
                    ReplicateData(
                        case_name=case_name,
                        replicate_name=replicate_name,
                        arrays=arrays,
                        attrs=attrs,
                    )
                )
            result[case_name] = reps
    return result


def clean_sample_matrix(matrix: np.ndarray) -> list[np.ndarray]:
    """Return one positive finite 1D sample array per time row."""

    values = np.asarray(matrix)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    cleaned: list[np.ndarray] = []
    for row in values:
        row = np.asarray(row, dtype=float).ravel()
        cleaned.append(row[np.isfinite(row) & (row > 0.0)])
    return cleaned


def require_arrays(rep: ReplicateData, names: list[str] | tuple[str, ...]) -> None:
    missing = [name for name in names if name not in rep.arrays]
    if missing:
        joined = ", ".join(missing)
        raise KeyError(
            f"Missing arrays for case={rep.case_name!r}, "
            f"replicate={rep.replicate_name!r}: {joined}"
        )


def stack_common_time(
    reps: list[ReplicateData], array_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Stack 1D replicate time series, interpolating onto the first time grid."""

    if not reps:
        return np.array([], dtype=float), np.empty((0, 0), dtype=float)

    require_arrays(reps[0], ("time", array_name))
    common_time = np.asarray(reps[0].arrays["time"], dtype=float)
    rows: list[np.ndarray] = []
    for rep in reps:
        require_arrays(rep, ("time", array_name))
        time = np.asarray(rep.arrays["time"], dtype=float)
        values = np.asarray(rep.arrays[array_name], dtype=float)
        if values.ndim != 1:
            raise ValueError(
                f"Expected 1D array {array_name!r} for case={rep.case_name!r}, "
                f"replicate={rep.replicate_name!r}; got shape {values.shape}"
            )
        if np.array_equal(time, common_time):
            rows.append(values)
        else:
            rows.append(np.interp(common_time, time, values))
    return common_time, np.vstack(rows)


def geometric_centers(edges: np.ndarray) -> np.ndarray:
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("edges must be a 1D array with at least two entries")
    return np.sqrt(edges[:-1] * edges[1:])
