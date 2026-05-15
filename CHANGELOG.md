# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- HDF5 I/O subsystem for checkpoint/restart and time-series diagnostics.
  - `save_checkpoint` / `load_checkpoint` — HDF5 checkpoint with schema v1.0.0.
  - `save_checkpoint_jld2` / `load_checkpoint_jld2` — JLD2 fallback for Julia-native workflows.
  - `init_diagnostics_file` / `save_diagnostics` — chunked HDF5 append for time-series data.
  - `export_diagnostics_to_csv` — convert HDF5 diagnostics to CSV.
  - `restore_rng` / `list_checkpoints` — RNG restoration and checkpoint enumeration.
- Diagnostics datasets: `time`, `number_concentration`, `mass_concentration`, `species_mass_concentration`, `mean_diameter`, `volume`, `size_distribution`.
- Python post-processing scripts in `examples/`:
  - `analyze_aerosol_brownian_coagulation.py` — single combined figure from HDF5 diagnostics.
  - `analyze_cloud_droplet_turbulent_coagulation.py` — single combined figure from HDF5 diagnostics.
- Example outputs (`.h5`, `.png`) are now written to the `examples/` folder via `@__DIR__`.

### Fixed

- HDF5 2D dataset dimension ordering for cross-language compatibility with h5py.
