"""Sampling syndromes and the on-disk dataset format.

A *dataset* is a set of ``shots`` independent experiments, each giving

``detectors``   uint8 ``(shots, n_det)``  detection events (1 = parity check changed)
``observables`` uint8 ``(shots,)``        1 = the logical observable was flipped

plus a JSON ``meta`` dict (code, distance, rounds, noise spec, source, seeds, ...).
Datasets are stored as compressed ``.npz`` with the bits packed 8-per-byte, so a
million shots of a d=7, 7-round experiment is about 6 MB.

Sources
-------
``stim``      fast bulk sampling of the Pauli-noise circuit (millions of shots/s)
``aer``       Qiskit Aer with a backend noise model (slow, device-like)
``hardware``  real IBM Runtime (PINQ2) shots, written by :mod:`src.hardware`
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import stim


# ============================================================== sampling (Stim)
def sample_syndromes(circuit: stim.Circuit, shots: int, seed: int | None = None,
                     chunk: int = 1_000_000) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``shots`` detection-event vectors and observable flips (uint8).

    Equivalent to the plan's ``sample()`` helper; ``seed`` makes it reproducible.
    """
    sampler = circuit.compile_detector_sampler(seed=seed)
    dets, obss = [], []
    remaining = shots
    while remaining > 0:
        n = min(remaining, chunk)
        det, obs = sampler.sample(n, separate_observables=True)
        dets.append(det.astype(np.uint8))
        obss.append(obs[:, 0].astype(np.uint8))
        remaining -= n
    return np.concatenate(dets), np.concatenate(obss)


class SyndromeSampler:
    """Stateful sampler used for *fresh data every training step* (plan E2)."""

    def __init__(self, circuit: stim.Circuit, seed: int):
        self.circuit = circuit
        self._sampler = circuit.compile_detector_sampler(seed=seed)
        self.n_det = circuit.num_detectors

    def __call__(self, batch: int) -> tuple[np.ndarray, np.ndarray]:
        det, obs = self._sampler.sample(batch, separate_observables=True)
        return det.astype(np.uint8), obs[:, 0].astype(np.uint8)


# ==================================================================== dataset I/O
@dataclass
class SyndromeDataset:
    detectors: np.ndarray
    observables: np.ndarray
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.detectors = np.ascontiguousarray(self.detectors, dtype=np.uint8)
        self.observables = np.ascontiguousarray(self.observables, dtype=np.uint8).reshape(-1)
        if self.detectors.ndim != 2 or len(self.detectors) != len(self.observables):
            raise ValueError("detectors must be (shots, n_det) and match observables")

    def __len__(self) -> int:
        return len(self.observables)

    @property
    def n_det(self) -> int:
        return self.detectors.shape[1]

    @property
    def logical_error_rate_undecoded(self) -> float:
        """Fraction of shots whose observable flipped (a decoder that never corrects)."""
        return float(self.observables.mean())

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            det_packed=np.packbits(self.detectors, axis=1),
            obs_packed=np.packbits(self.observables),
            n_det=np.int64(self.n_det),
            shots=np.int64(len(self)),
            meta=np.array(json.dumps(self.meta)),
        )
        return path if path.suffix == ".npz" else path.with_suffix(path.suffix + ".npz")

    @classmethod
    def load(cls, path: str | Path) -> "SyndromeDataset":
        with np.load(path, allow_pickle=False) as z:
            n_det, shots = int(z["n_det"]), int(z["shots"])
            det = np.unpackbits(z["det_packed"], axis=1)[:, :n_det]
            obs = np.unpackbits(z["obs_packed"])[:shots]
            meta = json.loads(str(z["meta"]))
        return cls(det, obs, meta)

    def split(self, n_first: int) -> tuple["SyndromeDataset", "SyndromeDataset"]:
        return (SyndromeDataset(self.detectors[:n_first], self.observables[:n_first], dict(self.meta)),
                SyndromeDataset(self.detectors[n_first:], self.observables[n_first:], dict(self.meta)))

    def batches(self, batch: int, *, shuffle: bool = True, seed: int = 0) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        idx = np.arange(len(self))
        if shuffle:
            np.random.default_rng(seed).shuffle(idx)
        for start in range(0, len(idx), batch):
            sel = idx[start:start + batch]
            yield self.detectors[sel], self.observables[sel]


def make_dataset(circuit: stim.Circuit, shots: int, seed: int, meta: dict | None = None) -> SyndromeDataset:
    det, obs = sample_syndromes(circuit, shots, seed)
    m = dict(meta or {})
    m.update(shots=shots, seed=seed, n_det=circuit.num_detectors, source=m.get("source", "stim"))
    return SyndromeDataset(det, obs, m)
