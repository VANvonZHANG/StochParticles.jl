#!/usr/bin/env python3
"""Merge case-sharded campaign files into the canonical per-arm HDF5.

Each shard (M3_CASE_SELECT) holds exactly one /cases/chi_X group written by
an independent process; merge = copy meta from the first shard + all case
groups. Seed-preserving: shards are bit-identical to the sequential run."""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import glob

import h5py

from .stochparticles_io import DATA_DIR


def merge(basename: str) -> str:
    shards = sorted(glob.glob(str(DATA_DIR / f"{basename}_shard*.h5")))
    if not shards:
        print(f"{basename}: no shards found, skipping")
        return ""
    out = DATA_DIR / f"{basename}.h5"
    with h5py.File(out, "w") as dst:
        with h5py.File(shards[0], "r") as first:
            first.copy("meta", dst)
        for sh in shards:
            with h5py.File(sh, "r") as src:
                for case in src["cases"]:
                    src.copy(f"cases/{case}", dst["cases"] if "cases" in dst
                             else dst.create_group("cases"))
    print(f"{basename}: merged {len(shards)} shards -> {out}")
    return str(out)


if __name__ == "__main__":
    for name in sys.argv[1:]:
        merge(name)
