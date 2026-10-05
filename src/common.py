"""Shared paths, experiment grids and the seed policy.

Seed policy (plan section 9: "random seeds fixed and reported"):
every random stream is derived from a *role* (train / val / test / bank / hardware)
and a *config key* with :func:`derive_seed`, so train, validation and test data for
the same configuration can never share a seed, and re-running anything reproduces
the same numbers bit-for-bit on the same library versions.
"""

from __future__ import annotations

import zlib
from pathlib import Path

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
GAME_BANK_DIR = DATA_DIR / "game_bank"
CALIBRATION_DIR = DATA_DIR / "calibration"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"
TABLES_DIR = RESULTS_DIR / "tables"
HARDWARE_DIR = RESULTS_DIR / "hardware"


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, GAME_BANK_DIR, CALIBRATION_DIR, FIGURES_DIR,
              MODELS_DIR, TABLES_DIR, HARDWARE_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------- experiment grids
#: code distances used throughout (plan E1: d in {3, 5, 7})
DISTANCES = (3, 5, 7)
#: physical error rates (plan E1: p in [0.001, 0.05]); also the game's difficulty grid
P_GRID = (0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05)
#: Z:X bias values r (plan section 4, model 2)
BIAS_GRID = (1.0, 5.0, 20.0)
#: strengths of the correlated two-qubit channel, in units of p (see noise.py)
CORR_GRID = (0.0, 0.25, 0.5, 1.0)
#: IBM fake backend used offline as stand-in for the PINQ2 device (ibm_quebec)
DEFAULT_FAKE_BACKEND = "fake_quebec"
#: noise multipliers offered for the device-derived model in the game
DEVICE_SCALES = (1.0, 2.0, 4.0)

# ----------------------------------------------------------------------------- seeds
BASE_SEEDS = {
    "train": 1_000_003,
    "val": 2_000_003,
    "test": 3_000_017,
    "bank": 4_000_037,
    "hardware": 5_000_011,
    "torch": 6_000_001,
}


def derive_seed(role: str, key: str = "") -> int:
    """Deterministic 31-bit seed for ``(role, key)``; independent across roles."""
    if role not in BASE_SEEDS:
        raise KeyError(f"unknown seed role {role!r}; choose from {sorted(BASE_SEEDS)}")
    h = zlib.crc32(key.encode("utf-8")) & 0xFFFF_FFFF
    return (BASE_SEEDS[role] + 7919 * h) % (2**31 - 1)


def fmt_p(p: float) -> str:
    """Stable string for a probability used in file names (0.01 -> '0.01')."""
    return f"{p:.6g}"
