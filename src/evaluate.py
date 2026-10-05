"""Evaluation protocol (plan section 9).

* Metric: logical error rate (LER) per shot (and per round), with 95 % Wilson intervals.
* Fairness: *the same test shots for every decoder*; test data is fresh (its seed is
  derived from role ``"test"`` and is never used for training or validation).
* "NN advantage" = LER_MWPM - LER_NN, reported as a **paired** difference on those
  shared shots with a 95 % confidence interval (positive = the network is better).
"""

from __future__ import annotations

import math
from math import comb
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .decoders.base import Decoder

Z95 = 1.959963984540054


# ------------------------------------------------------------------ intervals
def wilson_interval(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion ``k / n`` (well behaved at k = 0)."""
    if n <= 0:
        return 0.0, 1.0
    phat = k / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    # clamp so the interval always contains the point estimate (at k = 0 or k = n,
    # floating-point error can otherwise put a bound ~1e-17 on the wrong side of phat)
    return max(0.0, min(centre - half, phat)), min(1.0, max(centre + half, phat))


def paired_difference(pred_a: np.ndarray, pred_b: np.ndarray, obs: np.ndarray,
                      z: float = Z95) -> tuple[float, float, float]:
    """``LER_a - LER_b`` on shared shots with a normal-approximation 95 % CI.

    Uses the discordant pairs only (McNemar-style): with ``n10`` shots where only A is
    wrong and ``n01`` where only B is wrong, ``diff = (n10 - n01) / n`` and
    ``var = (n10 + n01 - (n10 - n01)^2 / n) / n^2``.  Returns ``(diff, lo, hi)``.
    """
    obs = np.asarray(obs)
    a_wrong = np.asarray(pred_a) != obs
    b_wrong = np.asarray(pred_b) != obs
    n = len(obs)
    n10 = int(np.sum(a_wrong & ~b_wrong))
    n01 = int(np.sum(~a_wrong & b_wrong))
    diff = (n10 - n01) / n
    var = max(n10 + n01 - (n10 - n01) ** 2 / n, 0.0) / (n * n)
    half = z * math.sqrt(var)
    return diff, diff - half, diff + half


def ler_per_round(ler: float, rounds: int) -> float:
    """Per-round logical error probability, ``(1 - (1 - 2 LER)^(1/R)) / 2``."""
    r = max(rounds, 1)
    x = min(max(1.0 - 2.0 * ler, 0.0), 1.0)
    return 0.5 * (1.0 - x ** (1.0 / r))


# ------------------------------------------------------------ analytic references
def code_capacity_flip_prob(p: float) -> float:
    """Net bit-flip probability per data qubit of the ``rounds = 0`` uniform circuit:
    ``DEPOLARIZE1(p)`` (X or Y flips with probability 2p/3) followed by a readout flip p."""
    pf = 2.0 * p / 3.0
    return pf * (1.0 - p) + p * (1.0 - pf)


def majority_vote_failure(d: int, pf: float) -> float:
    """Exact logical error rate of the distance-``d`` (odd) repetition code under i.i.d.
    flips ``pf`` with an optimal (= majority-vote = MWPM) decoder."""
    if d % 2 == 0:
        raise ValueError("closed form given for odd d")
    return sum(comb(d, k) * pf**k * (1 - pf) ** (d - k) for k in range(d // 2 + 1, d + 1))


def fit_loglog_slope(p: Iterable[float], ler: Iterable[float], min_ler: float = 0.0) -> float:
    """Least-squares slope of ``log LER`` vs ``log p`` over points with ``LER > min_ler``.
    Below threshold ``LER ~ C p^((d+1)/2)``, so the slope should approach ``(d+1)/2``."""
    p, ler = np.asarray(list(p), float), np.asarray(list(ler), float)
    ok = ler > max(min_ler, 0.0)
    if ok.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(p[ok]), np.log(ler[ok]), 1)[0])


# ----------------------------------------------------------------- evaluation
def evaluate_decoders(decoders: Mapping[str, Decoder], detectors: np.ndarray,
                      observables: np.ndarray, *, rounds: int = 1,
                      reference: str | None = None) -> list[dict]:
    """Run every decoder on the *same* shots; one result row per decoder.

    ``reference`` (a key of ``decoders``) additionally gets paired differences
    ``adv`` = LER_reference - LER_decoder with a 95 % CI (positive = decoder better).
    """
    obs = np.asarray(observables)
    preds = {name: np.asarray(dec.predict(detectors)) for name, dec in decoders.items()}
    rows = []
    for name, pred in preds.items():
        k = int(np.sum(pred != obs))
        n = len(obs)
        lo, hi = wilson_interval(k, n)
        row = dict(decoder=name, shots=n, errors=k, ler=k / n, ler_lo=lo, ler_hi=hi,
                   ler_per_round=ler_per_round(k / n, rounds))
        if reference is not None and reference in preds:
            diff, dlo, dhi = paired_difference(preds[reference], pred, obs)
            row.update(adv=diff, adv_lo=dlo, adv_hi=dhi)
        rows.append(row)
    return rows


def adaptive_ler(decoder: Decoder, sampler, *, min_errors: int = 100, first: int = 200_000,
                 max_shots: int = 5_000_000, chunk: int = 500_000) -> tuple[int, int]:
    """Keep sampling until ``min_errors`` logical errors are seen or ``max_shots`` is hit.

    ``sampler(n) -> (detectors, observables)``.  Used for the low-p scaling check (E1),
    where the LER is far too small to resolve with a fixed 10^5 shots.
    Returns ``(errors, shots)``.
    """
    errors = shots = 0
    n = first
    while shots < max_shots:
        det, obs = sampler(n)
        errors += int(np.sum(decoder.predict(det) != obs))
        shots += len(obs)
        if errors >= min_errors:
            break
        n = chunk
    return errors, shots


def to_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)
