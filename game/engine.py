"""Game logic for "Syndrome Hunter" -- pure python / numpy, no Streamlit (so it is unit-tested).

Scope control (plan section 6): the game *consumes pre-generated syndromes and saved models*;
it never trains anything.  Episodes come from ``data/game_bank/`` (made by
``scripts/make_game_bank.py``); if a setting has no bank file the same syndromes are sampled
on the fly from the Stim circuit with a fixed seed (cheap, still no training).

What one episode is
-------------------
A distance-``d`` repetition code is run for ``rounds = d`` rounds and the data qubits are read
out.  The player sees the *detection events* (space-time grid of lit defects, one row per round
plus the final-readout row).  Their job is to say **which data qubits were flipped**.

Scoring is identical for the human and every decoder:

* the final-readout parity pattern ``s`` (XOR of all detector rows) is the syndrome of the
  true data-qubit flip pattern ``e``;
* a *correction* ``c`` (set of data qubits to flip back) is **successful iff ``c == e``**;
* equivalently: ``c`` must clear the syndrome (residual ``s xor syn(c) = 0``) **and** pick the
  right one of the two patterns that clear it (they differ by flipping every qubit = a logical
  flip).  A decoder that predicts the logical flip ``o`` produces ``c = error_estimate(s, o)``.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.circuits import error_estimate, final_syndrome
from src.common import DEVICE_SCALES, GAME_BANK_DIR, MODELS_DIR, P_GRID, RESULTS_DIR, derive_seed
from src.data import SyndromeDataset, sample_syndromes
from src.decoders import Decoder, GreedyDecoder, MWPMDecoder, NeuralDecoder
from src.evaluate import wilson_interval
from src.noise import NoiseSpec

NOISE_LABELS = {
    "uniform": "Uniform noise (control)",
    "biased": "Biased / correlated noise",
    "device": "Real IBM device (calibration-derived approximation)",
}
DEFAULT_BIAS, DEFAULT_CORR = 5.0, 0.5
DISTANCES = (3, 5, 7)
EPISODE_CHOICES = (10, 25, 50)


def leaderboard_file() -> Path:
    """Where finished games are appended (override with the ``QEC_LEADERBOARD`` env var)."""
    return Path(os.environ.get("QEC_LEADERBOARD") or RESULTS_DIR / "game_leaderboard.jsonl")


PLAYER_LABEL = {"human": "You", "greedy": "Greedy (human-like)", "mwpm": "MWPM", "nn": "Neural net"}


# ============================================================== settings -> specs
def levels_for(noise: str) -> tuple[float, ...]:
    return DEVICE_SCALES if noise == "device" else P_GRID


def level_label(noise: str) -> str:
    return "noise multiplier (x calibrated rates)" if noise == "device" else "physical error rate p"


def spec_for(noise: str, level: float) -> NoiseSpec:
    if noise == "uniform":
        return NoiseSpec.uniform(level)
    if noise == "biased":
        return NoiseSpec.biased(level, DEFAULT_BIAS, DEFAULT_CORR)
    if noise == "device":
        return NoiseSpec.from_device(level)
    raise ValueError(f"unknown noise model {noise!r}")


def bank_path(noise: str, level: float, d: int) -> Path:
    return GAME_BANK_DIR / f"{spec_for(noise, level).key}_d{d}_r{d}.npz"


def model_path(noise: str, level: float, d: int, models_dir: Path = MODELS_DIR) -> Path:
    return Path(models_dir) / f"rep_{spec_for(noise, level).key}_d{d}_r{d}_mlp.pt"


def _circuit(noise: str, level: float, d: int):
    from src.experiments import make_circuit
    return make_circuit("rep", d, d, spec_for(noise, level))


# ====================================================================== episodes
@dataclass
class Episodes:
    detectors: np.ndarray        # (n, n_det) uint8
    observables: np.ndarray      # (n,) uint8
    d: int
    rounds: int
    from_bank: bool

    def __len__(self) -> int:
        return len(self.observables)


def load_episodes(noise: str, level: float, d: int, n: int, seed: int, skip_trivial: bool = False) -> Episodes:
    """``n`` episodes for a setting, shuffled reproducibly by ``seed``.

    Uses the pre-generated bank when present; otherwise samples from the circuit with a seed
    derived from the setting (role ``bank``), so the same setting always yields the same pool.
    ``skip_trivial`` drops episodes without a single defect (nothing to decode) and, if the pool
    has too few left (e.g. at p = 0.001), tops it up from the same circuit with seeds derived
    from ``(setting, seed)`` -- still deterministic.
    """
    spec = spec_for(noise, level)
    path = bank_path(noise, level, d)
    if path.exists():
        ds = SyndromeDataset.load(path)
        det, obs, bank = ds.detectors, ds.observables, True
    else:
        det, obs = sample_syndromes(_circuit(noise, level, d), max(n, 500),
                                    derive_seed("bank", f"{spec.key}|d{d}"))
        bank = False
    if skip_trivial:
        keep = det.any(axis=1)
        det, obs = det[keep], obs[keep]
        circuit, batch, attempt = None, 20_000, 0
        while len(obs) < n and attempt < 100:          # top up: rejection-sample non-trivial episodes
            circuit = circuit or _circuit(noise, level, d)
            more_det, more_obs = sample_syndromes(
                circuit, batch, derive_seed("bank", f"{spec.key}|d{d}|topup{attempt}|{seed}"))
            keep = more_det.any(axis=1)
            det, obs = np.concatenate([det, more_det[keep]]), np.concatenate([obs, more_obs[keep]])
            attempt += 1
    order = np.random.default_rng(seed).permutation(len(obs))[:n]
    return Episodes(det[order], obs[order], d, d, bank)


# ====================================================================== scoring
def syndrome_of(correction: np.ndarray) -> np.ndarray:
    """Parity pattern ``c_i xor c_{i+1}`` that a set of qubit flips would produce."""
    c = np.asarray(correction, dtype=np.uint8)
    return c[..., :-1] ^ c[..., 1:]


@dataclass
class Outcome:
    success: bool
    residual: np.ndarray          # defects left after applying the correction (0 = cleared)
    true_error: np.ndarray        # the actual data-qubit flip pattern e
    logical_flip: bool            # did the observable (last data qubit) flip?
    reason: str                   # "ok" | "defects left" | "wrong logical branch"


def true_error(det_row: np.ndarray, obs: int, d: int, rounds: int) -> np.ndarray:
    return error_estimate(final_syndrome(det_row, d, rounds), obs, d)


def score_correction(det_row: np.ndarray, obs: int, correction: np.ndarray, d: int, rounds: int) -> Outcome:
    s = final_syndrome(det_row, d, rounds)
    c = np.asarray(correction, dtype=np.uint8)
    e = error_estimate(s, obs, d)
    residual = s ^ syndrome_of(c)
    ok = bool(np.array_equal(c, e))
    reason = "ok" if ok else ("defects left" if residual.any() else "wrong logical branch")
    return Outcome(ok, residual, e, bool(obs), reason)


def correction_from_prediction(det_row: np.ndarray, pred_obs: int, d: int, rounds: int) -> np.ndarray:
    """Turn a decoder's logical-flip prediction into the full correction it implies."""
    return error_estimate(final_syndrome(det_row, d, rounds), pred_obs, d)


def play_decoder(decoder: Decoder, eps: Episodes) -> np.ndarray:
    """Boolean success per episode for a decoder (success <=> predicted flip == true flip)."""
    return np.asarray(decoder.predict(eps.detectors)) == eps.observables


def longest_run(bits: np.ndarray) -> int:
    best = cur = 0
    for b in np.asarray(bits):
        cur = cur + 1 if b else 0
        best = max(best, cur)
    return best


def chain_bucket(e: np.ndarray) -> str:
    """Teaching-hook bucket from the longest run of adjacent flipped data qubits.

    "no data flips" can still come with defects: a wrong parity-check *measurement* lights up
    defects without any data qubit being flipped."""
    r = longest_run(e)
    return {0: "no data flips", 1: "isolated errors"}.get(r, "chain of 2" if r == 2 else "chain of 3+")


BUCKETS = ["no data flips", "isolated errors", "chain of 2", "chain of 3+"]


def success_by_bucket(eps: Episodes, success: np.ndarray) -> dict[str, tuple[int, int]]:
    """``{bucket: (successes, episodes)}`` for one player over an episode set."""
    out = {b: [0, 0] for b in BUCKETS}
    for det, obs, ok in zip(eps.detectors, eps.observables, success):
        b = chain_bucket(true_error(det, int(obs), eps.d, eps.rounds))
        out[b][0] += int(ok)
        out[b][1] += 1
    return {b: (v[0], v[1]) for b, v in out.items()}


# ======================================================================== players
def load_players(noise: str, level: float, d: int, models_dir: Path = MODELS_DIR) -> dict[str, Decoder | None]:
    """Bots for a setting.  ``nn`` is ``None`` when no saved model exists (never trained here)."""
    spec = spec_for(noise, level)
    circuit = _circuit(noise, level, d)
    cands = {"matched": MWPMDecoder.from_circuit(circuit, "MWPM")}
    if noise == "biased":                       # best available weighting (see experiments.best_mwpm)
        from src.experiments import make_circuit
        cands["uniform"] = MWPMDecoder.from_circuit(make_circuit("rep", d, d, NoiseSpec.uniform(spec.p)), "MWPM")
    if len(cands) > 1:
        det, obs = sample_syndromes(circuit, 20_000, derive_seed("val", f"game|{spec.key}|d{d}"))
        mwpm = min(cands.values(), key=lambda m: float(np.mean(m.predict(det) != obs)))
    else:
        mwpm = cands["matched"]
    mp = model_path(noise, level, d, models_dir)
    nn = NeuralDecoder.load(mp, name="Neural net") if mp.exists() else None
    return {"greedy": GreedyDecoder(d, d, "Greedy"), "mwpm": mwpm, "nn": nn}


# =================================================================== leaderboard
@dataclass
class Entry:
    player: str
    kind: str                       # human | greedy | mwpm | nn
    noise: str
    level: float
    d: int
    successes: int
    episodes: int
    seed: int
    when: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

    @property
    def rate(self) -> float:
        return self.successes / max(self.episodes, 1)

    def ci(self) -> tuple[float, float]:
        return wilson_interval(self.successes, self.episodes)


def append_leaderboard(entry: Entry, path: Path | None = None) -> None:
    path = Path(path) if path else leaderboard_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(entry.__dict__) + "\n")


def read_leaderboard(path: Path | None = None) -> list[Entry]:
    path = Path(path) if path else leaderboard_file()
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                out.append(Entry(**json.loads(line)))
            except (TypeError, json.JSONDecodeError):
                continue
    return out
