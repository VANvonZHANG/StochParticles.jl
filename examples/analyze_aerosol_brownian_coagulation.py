#!/usr/bin/env python3
"""
Post-processing script for aerosol_brownian_coagulation.h5
Reads the HDF5 diagnostics file and generates analysis plots.
"""

import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
H5_PATH = os.path.join(SCRIPT_DIR, "aerosol_brownian_coagulation.h5")
OUT_DIR = SCRIPT_DIR

if not os.path.isfile(H5_PATH):
    print(f"Error: HDF5 file not found: {H5_PATH}")
    print("Run aerosol_brownian_coagulation.jl first.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with h5py.File(H5_PATH, "r") as f:
    time = f["time"][:]
    N_conc = f["number_concentration"][:]
    M_conc = f["mass_concentration"][:]
    species_mass = f["species_mass_concentration"][:]
    mean_d = f["mean_diameter"][:]
    volume = f["volume"][:]
    size_dist = f["size_distribution"][:]

    meta = f["meta"]
    species_names = meta.attrs["species_names"].astype(str).tolist()
    bin_edges = meta.attrs["bin_edges"][:]

# Convert diameter from meters to micrometers
mean_d_um = mean_d * 1e6
bin_edges_um = bin_edges * 1e6
bin_centers_um = np.sqrt(bin_edges_um[:-1] * bin_edges_um[1:])

# ---------------------------------------------------------------------------
# Plot 1: Time series of concentrations and mean diameter
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
fig.suptitle("Aerosol Brownian Coagulation — Time Series", fontsize=14)

ax = axes[0, 0]
ax.plot(time / 60.0, N_conc, color="C0")
ax.set_xlabel("Time (min)")
ax.set_ylabel("Number concentration (m⁻³)")
ax.set_title("Number Concentration")
ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

ax = axes[0, 1]
ax.plot(time / 60.0, M_conc, color="C1")
ax.set_xlabel("Time (min)")
ax.set_ylabel("Mass concentration (kg/m³)")
ax.set_title("Mass Concentration")
ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

ax = axes[1, 0]
for i, name in enumerate(species_names):
    ax.plot(time / 60.0, species_mass[:, i], label=name)
ax.set_xlabel("Time (min)")
ax.set_ylabel("Species mass concentration (kg/m³)")
ax.set_title("Species Mass Concentration")
ax.legend()
ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

ax = axes[1, 1]
ax.plot(time / 60.0, mean_d_um, color="C3")
ax.set_xlabel("Time (min)")
ax.set_ylabel("Mean diameter (μm)")
ax.set_title("Mean Diameter")

out_path = os.path.join(OUT_DIR, "aerosol_brownian_coagulation_timeseries.png")
fig.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")
plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 2: Size distribution heatmap
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

# Subsample time for clearer visualization
nt = len(time)
step = max(1, nt // 200)
time_sub = time[::step]
size_dist_sub = size_dist[::step, :]

# Plot as pcolormesh (pad one row so C matches edge count for nearest shading)
C_padded = np.vstack([size_dist_sub.T, size_dist_sub.T[-1:, :]])
c = ax.pcolormesh(time_sub / 60.0, bin_edges_um, C_padded, shading="nearest", cmap="YlOrRd")
ax.set_yscale("log")
ax.set_ylim(bin_edges_um[0], bin_edges_um[-1])
ax.set_xlabel("Time (min)")
ax.set_ylabel("Diameter (μm)")
ax.set_title("Size Distribution dN/dlogD (m⁻³)")
fig.colorbar(c, ax=ax, label="Concentration (m⁻³)")

out_path = os.path.join(OUT_DIR, "aerosol_brownian_coagulation_size_heatmap.png")
fig.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")
plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 3: Snapshots of size distribution at selected times
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

snapshot_times = [0.0, 600.0, 1800.0, 3600.0]  # seconds
for t_target in snapshot_times:
    idx = np.argmin(np.abs(time - t_target))
    actual_t = time[idx]
    ax.plot(bin_centers_um, size_dist[idx, :], label=f"t = {actual_t/60:.0f} min")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Diameter (μm)")
ax.set_ylabel("Concentration per bin (m⁻³)")
ax.set_title("Size Distribution Snapshots")
ax.legend()
ax.grid(True, which="both", ls="--", alpha=0.5)

out_path = os.path.join(OUT_DIR, "aerosol_brownian_coagulation_snapshots.png")
fig.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")
plt.close(fig)

print("\nAnalysis complete.")
