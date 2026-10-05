"""Evaluation maths, dataset I/O, seeds, and noise / calibration specs."""

from __future__ import annotations

import math
from itertools import product

import numpy as np
import pytest

from src.circuits import build_repetition_circuit
from src.common import BASE_SEEDS, derive_seed
from src.data import SyndromeDataset, SyndromeSampler, make_dataset, sample_syndromes
from src.decoders import MWPMDecoder
from src.evaluate import (adaptive_ler, code_capacity_flip_prob, evaluate_decoders, fit_loglog_slope,
                          ler_per_round, majority_vote_failure, paired_difference, wilson_interval)
from src.noise import (NoiseSpec, best_linear_chain, calibration_from_backend, get_fake_backend,
                       idle_pauli_probs, load_device_calibration, DeviceCalibration)
from src.training import DatasetSampler


# ------------------------------------------------------------------------- intervals
@pytest.mark.parametrize("k,n", [(0, 10), (10, 10), (0, 1), (1, 1), (3, 10), (0, 1_000_000), (5, 5), (500, 1000)])
def test_wilson_contains_point_estimate_and_stays_in_unit_interval(k, n):
    lo, hi = wilson_interval(k, n)
    assert 0.0 <= lo <= k / n <= hi <= 1.0           # regression: k = 0 used to give lo > phat by ~1e-17


def test_wilson_known_value_and_monotonicity():
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=5e-4) and hi == pytest.approx(0.5962, abs=5e-4)
    assert wilson_interval(50, 1000)[1] - wilson_interval(50, 1000)[0] < hi - lo     # more data -> narrower
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_paired_difference_sign_and_zero_case():
    obs = np.zeros(1000, dtype=np.uint8)
    a = np.zeros(1000, dtype=np.uint8)
    b = np.zeros(1000, dtype=np.uint8)
    b[:50] = 1                                        # b wrong on 50 shots, a never wrong
    diff, lo, hi = paired_difference(a, b, obs)       # LER_a - LER_b
    assert diff == pytest.approx(-0.05) and hi < 0    # a is significantly better
    assert paired_difference(a, a, obs) == (0.0, 0.0, 0.0)


def test_paired_difference_ci_covers_truth():
    rng = np.random.default_rng(0)
    hits = 0
    for _ in range(200):
        n = 4000
        obs = np.zeros(n, dtype=np.uint8)
        a = (rng.random(n) < 0.05).astype(np.uint8)
        b = (rng.random(n) < 0.04).astype(np.uint8)
        _, lo, hi = paired_difference(a, b, obs)
        hits += lo <= 0.01 <= hi
    assert hits >= 175                                # ~95 % nominal coverage with slack


# ---------------------------------------------------------------------- analytic references
def test_majority_vote_failure_matches_brute_force():
    for d, pf in [(3, 0.1), (5, 0.2), (7, 0.05)]:
        brute = sum(math.prod(pf if b else 1 - pf for b in bits) for bits in product([0, 1], repeat=d)
                    if sum(bits) > d / 2)
        assert majority_vote_failure(d, pf) == pytest.approx(brute, rel=1e-12)
    with pytest.raises(ValueError):
        majority_vote_failure(4, 0.1)


def test_code_capacity_flip_prob_limits():
    assert code_capacity_flip_prob(0.0) == 0.0
    assert code_capacity_flip_prob(0.01) == pytest.approx(0.01 + 2 * 0.01 / 3 - 2 * 0.01 * 2 * 0.01 / 3)


def test_ler_per_round_inverts_compounding():
    eps, rounds = 0.01, 5
    ler = 0.5 * (1 - (1 - 2 * eps) ** rounds)
    assert ler_per_round(ler, rounds) == pytest.approx(eps)
    assert ler_per_round(0.0, 3) == 0.0


def test_fit_loglog_slope_recovers_exponent():
    p = np.array([0.001, 0.002, 0.004, 0.008])
    assert fit_loglog_slope(p, 7 * p**3) == pytest.approx(3.0)
    assert math.isnan(fit_loglog_slope([0.1], [0.1]))


def test_adaptive_ler_stops_when_enough_errors_seen():
    c = build_repetition_circuit(3, 3, NoiseSpec.uniform(0.05))
    dec = MWPMDecoder.from_circuit(c)
    err, shots = adaptive_ler(dec, SyndromeSampler(c, 1), min_errors=50, first=2000, chunk=2000, max_shots=100_000)
    assert err >= 50 and shots < 100_000
    _, shots2 = adaptive_ler(dec, SyndromeSampler(c, 2), min_errors=10**9, first=1000, chunk=1000, max_shots=3000)
    assert shots2 == 3000                              # respects the cap


def test_evaluate_decoders_uses_same_shots_and_reports_ci():
    c = build_repetition_circuit(3, 3, NoiseSpec.uniform(0.03))
    det, obs = sample_syndromes(c, 5000, 3)
    mw = MWPMDecoder.from_circuit(c)

    class NeverFlip:
        name = "never"
        def predict(self, d):
            return np.zeros(len(d), dtype=np.uint8)

    rows = evaluate_decoders({"mwpm": mw, "never": NeverFlip()}, det, obs, rounds=3, reference="mwpm")
    r = {x["decoder"]: x for x in rows}
    assert r["never"]["ler"] == pytest.approx(obs.mean())
    assert r["mwpm"]["adv"] == 0.0 and r["never"]["adv"] < 0       # LER_mwpm - LER_never < 0
    assert r["mwpm"]["ler_lo"] <= r["mwpm"]["ler"] <= r["mwpm"]["ler_hi"]
    assert all(x["shots"] == 5000 for x in rows)


# ------------------------------------------------------------------------------ data
def test_sampling_is_reproducible_and_seed_dependent():
    c = build_repetition_circuit(3, 3, NoiseSpec.uniform(0.05))
    a = sample_syndromes(c, 5000, seed=1)
    b = sample_syndromes(c, 5000, seed=1)
    d = sample_syndromes(c, 5000, seed=2)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not np.array_equal(a[0], d[0])
    assert a[0].dtype == np.uint8 and a[1].shape == (5000,)
    chunked = sample_syndromes(c, 5000, seed=1, chunk=1000)
    assert chunked[0].shape == a[0].shape


@pytest.mark.parametrize("d,rounds", [(3, 3), (5, 4), (3, 0)])        # n_det = 8, 20, 2: not all multiples of 8
def test_dataset_roundtrip(tmp_path, d, rounds):
    c = build_repetition_circuit(d, rounds, NoiseSpec.uniform(0.05))
    ds = make_dataset(c, 777, seed=5, meta={"code": "rep", "d": d})
    path = ds.save(tmp_path / "x.npz")
    back = SyndromeDataset.load(path)
    assert np.array_equal(back.detectors, ds.detectors) and np.array_equal(back.observables, ds.observables)
    assert back.meta["d"] == d and back.meta["shots"] == 777 and back.n_det == c.num_detectors


def test_dataset_validation_split_and_batches():
    det = np.random.default_rng(0).integers(0, 2, (100, 6), dtype=np.uint8)
    ds = SyndromeDataset(det, np.zeros(100, dtype=np.uint8))
    a, b = ds.split(70)
    assert len(a) == 70 and len(b) == 30
    assert sum(len(x) for x, _ in ds.batches(32)) == 100
    with pytest.raises(ValueError):
        SyndromeDataset(det, np.zeros(99, dtype=np.uint8))


def test_dataset_sampler_draws_valid_batches():
    det = np.arange(40, dtype=np.uint8).reshape(10, 4)
    ds = SyndromeDataset(det % 2, np.arange(10, dtype=np.uint8) % 2)
    x, y = DatasetSampler(ds, 0)(32)
    assert x.shape == (32, 4) and y.shape == (32,)


def test_seed_roles_are_independent_and_deterministic():
    keys = ["rep|d5|r5|uniform_p0.01", "rep|d7|r7|uniform_p0.01"]
    seeds = {(r, k): derive_seed(r, k) for r in BASE_SEEDS for k in keys}
    assert len(set(seeds.values())) == len(seeds)                 # no collisions across roles/configs
    assert derive_seed("test", keys[0]) == derive_seed("test", keys[0])
    with pytest.raises(KeyError):
        derive_seed("nonsense", "x")


# ---------------------------------------------------------------------------------- noise
def test_noise_spec_validation_keys_and_serialization():
    assert NoiseSpec.uniform(0.01).key != NoiseSpec.biased(0.01, 5, 0.5).key != NoiseSpec.from_device(1.0).key
    assert NoiseSpec.biased(0.01, 5, 0.5).key != NoiseSpec.biased(0.01, 5, 0.25).key
    s = NoiseSpec.biased(0.02, 20.0, 1.0)
    assert NoiseSpec.from_dict(s.to_dict()) == s
    for bad in (dict(kind="nope"), dict(p=-0.1), dict(p=0.9), dict(bias=-1.0), dict(kind="biased", p=0.5, bias=10)):
        with pytest.raises(ValueError):
            NoiseSpec(**bad)
    assert NoiseSpec.from_device(2.0).level == 2.0 and NoiseSpec.uniform(0.03).level == 0.03
    assert NoiseSpec.uniform(0.01).with_p(0.02).p == 0.02


def test_idle_pauli_probs_physical():
    px, py, pz = idle_pauli_probs(1e-6, 100e-6, 100e-6)
    assert px == py > 0 and pz >= 0 and px + py + pz < 0.1
    assert idle_pauli_probs(0.0, 1e-4, 1e-4) == (0.0, 0.0, 0.0)
    assert idle_pauli_probs(1e-6, 100e-6, 1e-3)[2] >= 0           # T2 > 2 T1 is clamped, never negative


def test_best_linear_chain_is_a_valid_path_and_deterministic():
    backend = get_fake_backend("fake_quebec")
    chain = best_linear_chain(backend, 9)
    edges = {tuple(sorted(e)) for e in backend.coupling_map.get_edges()}
    assert len(chain) == 9 == len(set(chain))
    assert all(tuple(sorted(pair)) in edges for pair in zip(chain[:-1], chain[1:]))
    assert chain == best_linear_chain(backend, 9)


def test_calibration_snapshot_roundtrip_prefix_and_helpers(tmp_path):
    backend = get_fake_backend("fake_quebec")
    chain = best_linear_chain(backend, 5)
    cal = calibration_from_backend(backend, chain)
    assert cal.n_qubits == 5 and len(cal.cx_error) == 4 and all(0 < e < 0.5 for e in cal.cx_error)
    cal.to_json(tmp_path / "c.json")
    back = DeviceCalibration.from_json(tmp_path / "c.json")
    assert back == cal
    short = cal.prefix(3)
    assert short.chain == chain[:3] and len(short.cx_error) == 2
    with pytest.raises(ValueError):
        cal.prefix(6)
    assert 0 < cal.bare_qubit_error(10e-6) < 0.2
    assert cal.bare_qubit_error(1e-3) > cal.bare_qubit_error(1e-6)


def test_committed_calibration_snapshots_are_used():
    for d in (3, 5, 7):
        cal = load_device_calibration("fake_quebec", d)
        assert cal.n_qubits == 2 * d - 1 and cal.backend == "fake_quebec"


def test_get_fake_backend_rejects_unknown_name():
    with pytest.raises(ValueError):
        get_fake_backend("fake_does_not_exist")
