"""Shared command-line helpers for the scripts (noise-model arguments, circuit building)."""

from __future__ import annotations

import argparse

from src.common import DEFAULT_FAKE_BACKEND
from src.noise import NoiseSpec


def add_noise_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("code and noise model")
    g.add_argument("--code", choices=["rep", "surface"], default="rep",
                   help="repetition code (default) or rotated surface code (stretch)")
    g.add_argument("--d", type=int, default=5, help="code distance")
    g.add_argument("--rounds", type=int, default=None,
                   help="syndrome-extraction rounds (default: d; 0 = final-readout-only code capacity)")
    g.add_argument("--noise", choices=["uniform", "biased", "device"], default="uniform",
                   help="uniform | biased (biased/correlated) | device (calibration-derived)")
    g.add_argument("--p", type=float, default=0.01, help="physical error rate (uniform / biased)")
    g.add_argument("--bias", type=float, default=5.0, help="Z:X bias r (biased)")
    g.add_argument("--corr", type=float, default=0.5, help="correlated pair-flip strength in units of p (biased)")
    g.add_argument("--scale", type=float, default=1.0, help="multiplier on calibrated error rates (device)")
    g.add_argument("--device", default=DEFAULT_FAKE_BACKEND,
                   help="calibration source for --noise device (an IBM fake backend name)")


def spec_from_args(a: argparse.Namespace) -> NoiseSpec:
    if a.noise == "uniform":
        return NoiseSpec.uniform(a.p)
    if a.noise == "biased":
        return NoiseSpec.biased(a.p, a.bias, a.corr)
    return NoiseSpec.from_device(a.scale, a.device)


def rounds_from_args(a: argparse.Namespace) -> int:
    return a.d if a.rounds is None else a.rounds
