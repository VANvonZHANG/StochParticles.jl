#!/usr/bin/env python3
"""
Post-processing script for cloud_droplet_turbulent_coagulation.h5
Reads the HDF5 diagnostics file and generates a single combined analysis figure.
"""

import os
import sys

import h5py
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
H5_PATH = os.path.join(SCRIPT_DIR, "cloud_droplet_turbulent_coagulation.h5")
OUT_DIR = SCRIPT_DIR

if not os.path.isfile(H5_PATH):
    print(f"Error: HDF5 file not found: {H5_PATH}")
    print("Run cloud_droplet_turbulent_coagulation.jl first.")
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

# Subsample time for heatmap
nt = len(time)
step = max(1, nt // 200)
time_sub = time[::step]
size_dist_sub = size_dist[::step, :]

# ---------------------------------------------------------------------------
# Combined figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 16))
gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 1.2, 1.2])
fig.suptitle("Cloud Droplet Turbulent Coagulation — Analysis", fontsize=16, y=0.995)

# Row 0, Col 0: Number Concentration
ax = fig.add_subplot(gs[0, 0])
ax.plot(time, N_conc, color="C0")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Number concentration (m⁻³)")
ax.set_title("Number Concentration")
ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

# Row 0, Col 1: Mass Concentration
ax = fig.add_subplot(gs[0, 1])
ax.plot(time, M_conc, color="C1")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Mass concentration (kg/m³)")
ax.set_title("Mass Concentration")
ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

# Row 1, Col 0: Species Mass Concentration
ax = fig.add_subplot(gs[1, 0])
for i, name in enumerate(species_names):
    ax.plot(time, species_mass[:, i], label=name)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Species mass concentration (kg/m³)")
ax.set_title("Species Mass Concentration")
ax.legend()
ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

# Row 1, Col 1: Mean Diameter
ax = fig.add_subplot(gs[1, 1])
ax.plot(time, mean_d_um, color="C3")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Mean diameter (μm)")
ax.set_title("Mean Diameter")

# Row 2: Size distribution heatmap (span both columns)
ax = fig.add_subplot(gs[2, :])
C_padded = np.vstack([size_dist_sub.T, size_dist_sub.T[-1:, :]])
c = ax.pcolormesh(time_sub, bin_edges_um, C_padded, shading="nearest", cmap="YlOrRd")
ax.set_yscale("log")
ax.set_ylim(bin_edges_um[0], bin_edges_um[-1])
ax.set_xlabel("Time (s)")
ax.set_ylabel("Diameter (μm)")
ax.set_title("Size Distribution dN/dlogD (m⁻³)")
fig.colorbar(c, ax=ax, label="Concentration (m⁻³)")

# Row 3: Snapshots (span both columns)
ax = fig.add_subplot(gs[3, :])
snapshot_times = [0.0, 50.0, 100.0, 200.0]  # seconds
for t_target in snapshot_times:
    idx = np.argmin(np.abs(time - t_target))
    actual_t = time[idx]
    ax.plot(bin_centers_um, size_dist[idx, :], label=f"t = {actual_t:.0f} s")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Diameter (μm)")
ax.set_ylabel("Concentration per bin (m⁻³)")
ax.set_title("Size Distribution Snapshots")
ax.legend()
ax.grid(True, which="both", ls="--", alpha=0.5)

fig.tight_layout(rect=[0, 0, 1, 0.99])
out_path = os.path.join(OUT_DIR, "cloud_droplet_turbulent_coagulation_analysis.png")
fig.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")
plt.close(fig)

print("Analysis complete.")
