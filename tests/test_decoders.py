"""Decoders: MWPM correctness, greedy behaviour, neural nets (shapes, features, saving, training)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.circuits import build_repetition_circuit, measurements_to_syndromes
from src.data import SyndromeSampler, sample_syndromes
from src.decoders import (CNNDecoder, GRUDecoder, GreedyDecoder, MLPDecoder, MWPMDecoder, NeuralDecoder,
                          build_network, train)
from src.decoders.greedy import LEFT, RIGHT
from src.decoders.neural import _events_and_record
from src.evaluate import code_capacity_flip_prob, majority_vote_failure
from src.noise import NoiseSpec


# ----------------------------------------------------------------------------- MWPM
@pytest.mark.parametrize("d", [3, 5, 7])
def test_mwpm_equals_majority_vote_in_the_code_capacity_limit(d):
    """rounds=0: MWPM fails exactly when more than half of the data qubits flipped."""
    c = build_repetition_circuit(d, 0, NoiseSpec.uniform(0.1))
    meas = c.compile_sampler(seed=4).sample(20000).astype(np.uint8)
    det, obs = measurements_to_syndromes(meas, d, 0, 0)
    wrong = MWPMDecoder.from_circuit(c).predict(det) != obs
    assert np.array_equal(wrong, meas.sum(axis=1) > d / 2)


@pytest.mark.parametrize("d,p", [(3, 0.05), (5, 0.1), (7, 0.1)])
def test_mwpm_logical_error_rate_matches_exact_formula(d, p):
    c = build_repetition_circuit(d, 0, NoiseSpec.uniform(p))
    n = 400_000
    det, obs = sample_syndromes(c, n, seed=5)
    ler = MWPMDecoder.from_circuit(c).logical_error_rate(det, obs)
    exact = majority_vote_failure(d, code_capacity_flip_prob(p))
    assert abs(ler - exact) < 4.5 * np.sqrt(exact * (1 - exact) / n)


def test_mwpm_ler_decreases_with_distance_below_threshold():
    lers = []
    for d in (3, 5, 7):
        c = build_repetition_circuit(d, d, NoiseSpec.uniform(0.02))
        det, obs = sample_syndromes(c, 100_000, seed=6)
        lers.append(MWPMDecoder.from_circuit(c).logical_error_rate(det, obs))
    assert lers[0] > lers[1] > lers[2]


def test_mwpm_single_row_input_and_perfect_on_trivial_syndrome():
    c = build_repetition_circuit(3, 3, NoiseSpec.uniform(0.01))
    dec = MWPMDecoder.from_circuit(c)
    zero = np.zeros(c.num_detectors, dtype=np.uint8)
    assert dec.predict(zero).shape == (1,) and dec.predict(zero)[0] == 0


# ---------------------------------------------------------------------------- greedy
def test_greedy_basic_cases():
    g = GreedyDecoder(5, 0)                       # code capacity: 4 detectors
    assert g.predict(np.array([0, 0, 0, 0]))[0] == 0
    assert g.predict(np.array([0, 1, 1, 0]))[0] == 0         # adjacent pair: one flipped qubit
    assert g.predict(np.array([0, 0, 0, 1]))[0] == 1         # lone defect at the right edge -> flips q_{d-1}
    assert g.predict(np.array([1, 0, 0, 0]))[0] == 0         # lone defect at the left edge
    assert g.match(np.array([1, 0, 0, 0])) == [(0, LEFT)]
    assert g.match(np.array([0, 0, 0, 1])) == [(3, RIGHT)]


def test_greedy_fails_on_a_chain_where_mwpm_succeeds():
    """The teaching hook: three adjacent flips leave two defects each closer to an edge."""
    d = 7
    truth = np.zeros(d, dtype=np.uint8)
    truth[[2, 3, 4]] = 1
    det = (truth[:-1] ^ truth[1:]).astype(np.uint8)
    obs = truth[d - 1]
    mwpm = MWPMDecoder.from_circuit(build_repetition_circuit(d, 0, NoiseSpec.uniform(0.05)))
    assert mwpm.predict(det)[0] == obs                        # global view: correct
    assert GreedyDecoder(d, 0).predict(det)[0] != obs         # local view: logical error


def test_greedy_dedupe_matches_row_by_row_and_validates_shape():
    c = build_repetition_circuit(5, 5, NoiseSpec.uniform(0.03))
    det, _ = sample_syndromes(c, 3000, seed=8)
    g = GreedyDecoder(5, 5)
    rows = np.array([g.predict(r)[0] for r in det[:300]])
    assert np.array_equal(rows, g.predict(det[:300]))
    with pytest.raises(ValueError):
        g.predict(det[:, :-1])


def test_greedy_is_reasonable_but_worse_than_mwpm_at_larger_distance():
    c = build_repetition_circuit(7, 7, NoiseSpec.uniform(0.01))
    det, obs = sample_syndromes(c, 60_000, seed=9)
    g = GreedyDecoder(7, 7).logical_error_rate(det, obs)
    m = MWPMDecoder.from_circuit(c).logical_error_rate(det, obs)
    assert m < g < obs.mean()                                  # better than doing nothing, worse than MWPM


# ---------------------------------------------------------------------------- networks
@pytest.mark.parametrize("arch,kw", [("mlp", dict(record=True)), ("mlp", dict(record=False)),
                                     ("gru", dict(record=True)), ("gru", dict(record=False)),
                                     ("cnn", dict(record=True)), ("cnn", dict(record=False))])
def test_networks_forward_and_checkpoint_roundtrip(arch, kw, tmp_path):
    net = build_network(arch, 24, 4, **kw)
    x = (torch.rand(7, 24) > 0.8).float()
    assert net(x).shape == (7,)
    nd = NeuralDecoder(net, {"torch": torch.__version__, "note": "meta with a TorchVersion object"})
    path = nd.save(tmp_path / "m.pt")
    loaded = NeuralDecoder.load(path)                 # weights_only=True must accept the metadata
    xx = (np.random.default_rng(0).random((20, 24)) > 0.8).astype(np.uint8)
    assert np.allclose(nd.predict_logits(xx), loaded.predict_logits(xx))
    assert loaded.n_det == 24 and set(np.unique(loaded.predict(xx))) <= {0, 1}


def test_record_feature_is_cumulative_parity_of_events():
    ev = torch.tensor([[1, 0, 0, 0, 1, 0, 1, 1, 0]], dtype=torch.float32)     # 3 layers x 3 ancillas
    events, rec = _events_and_record(ev, 3, 3)
    assert rec[0, 0].tolist() == [1, 0, 0]
    assert rec[0, 1].tolist() == [1, 1, 0]
    assert rec[0, 2].tolist() == [0, 0, 0]          # [1,1,0] xor [1,1,0]
    # last layer of the record = parity of every detector layer = syndrome of the final data readout
    assert torch.equal(rec[:, -1], events.sum(dim=1) % 2)


def test_network_input_validation():
    with pytest.raises(ValueError):
        build_network("gru", 24)                     # needs n_anc
    with pytest.raises(ValueError):
        MLPDecoder(24, record=True)                  # record features need the layout
    with pytest.raises(ValueError):
        build_network("transformer", 24, 4)
    nd = NeuralDecoder(MLPDecoder(24))
    with pytest.raises(ValueError):
        nd.predict(np.zeros((2, 10), dtype=np.uint8))


def test_training_improves_and_restores_best_weights():
    c = build_repetition_circuit(3, 3, NoiseSpec.uniform(0.05))
    val = sample_syndromes(c, 20_000, seed=1)
    net = MLPDecoder(c.num_detectors, hidden=64, n_anc=2, record=True)
    res = train(net, SyndromeSampler(c, seed=2), val_data=val, steps=300, batch=1024, eval_every=100, seed=0)
    assert res.history[-1]["val_loss"] < res.history[0]["val_loss"]
    assert res.best_val_loss == min(h["val_loss"] for h in res.history)
    # restored weights reproduce the recorded best validation loss
    xv = torch.from_numpy(val[0]).float()
    yv = torch.from_numpy(val[1]).float()
    with torch.no_grad():
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(xv), yv).item()
    assert abs(loss - res.best_val_loss) < 1e-4


def test_training_is_bit_reproducible_through_the_public_entry_point():
    """Same seeds -> identical weights (initialisation is seeded before the network is built),
    and independent of what else the process trained before."""
    from src.training import train_on_circuit

    c = build_repetition_circuit(3, 2, NoiseSpec.uniform(0.05))
    kw = dict(d=3, steps=60, batch=256, eval_every=30, val_shots=5000, hidden=32,
              seeds=dict(val=1, train=2, torch=3))
    a, _ = train_on_circuit(c, **kw)
    train_on_circuit(build_repetition_circuit(5, 3, NoiseSpec.uniform(0.02)), d=5, steps=10, batch=64,
                     eval_every=10, val_shots=500, hidden=16)           # perturb global RNG state
    b, _ = train_on_circuit(c, **kw)
    assert all(torch.equal(x, y) for x, y in zip(a.net.state_dict().values(), b.net.state_dict().values()))
    kw["seeds"] = dict(val=1, train=2, torch=4)
    other, _ = train_on_circuit(c, **kw)
    assert not all(torch.equal(x, y) for x, y in zip(a.net.state_dict().values(), other.net.state_dict().values()))


def test_early_stopping_triggers_with_zero_patience_budget():
    c = build_repetition_circuit(3, 1, NoiseSpec.uniform(0.001))   # almost no signal: no steady improvement
    val = sample_syndromes(c, 5000, seed=1)
    net = MLPDecoder(c.num_detectors, hidden=16)
    res = train(net, SyndromeSampler(c, seed=2), val_data=val, steps=2000, batch=256, eval_every=20,
                patience=2, min_delta=1.0, seed=0)               # nothing can improve by >= 1.0
    assert res.stopped_early and res.steps_run < 2000


def test_lr_schedule_validation():
    c = build_repetition_circuit(3, 1, NoiseSpec.uniform(0.01))
    with pytest.raises(ValueError):
        train(MLPDecoder(c.num_detectors), SyndromeSampler(c, 0), val_data=sample_syndromes(c, 100, 1),
              steps=2, lr_schedule="bogus")


@pytest.mark.slow
def test_network_reaches_mwpm_level_on_small_uniform_problem():
    d, p = 3, 0.03
    c = build_repetition_circuit(d, d, NoiseSpec.uniform(p))
    val = sample_syndromes(c, 50_000, seed=11)
    net = build_network("mlp", c.num_detectors, d - 1, record=True)
    train(net, SyndromeSampler(c, seed=12), val_data=val, steps=1500, batch=4096, lr_schedule="cosine", seed=0)
    det, obs = sample_syndromes(c, 200_000, seed=13)
    nn = NeuralDecoder(net).logical_error_rate(det, obs)
    mw = MWPMDecoder.from_circuit(c).logical_error_rate(det, obs)
    assert nn < 1.2 * mw


@pytest.mark.slow
def test_cnn_and_gru_train_without_error():
    c = build_repetition_circuit(3, 3, NoiseSpec.uniform(0.03))
    val = sample_syndromes(c, 5000, seed=1)
    for arch in ("cnn", "gru"):
        net = build_network(arch, c.num_detectors, 2, record=True)
        res = train(net, SyndromeSampler(c, seed=2), val_data=val, steps=40, batch=256, eval_every=20)
        assert np.isfinite(res.best_val_loss)
        assert isinstance(net, (CNNDecoder, GRUDecoder))
