"""Game: engine logic (scoring parity with decoders), episode loading, drawings, and a headless
run through the Streamlit app."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from game import engine as eng
from game import viz
from src.common import GAME_BANK_DIR
from src.data import make_dataset
from src.decoders import NeuralDecoder
from src.training import train_on_circuit

ROOT = Path(__file__).resolve().parent.parent


def _eps(noise="uniform", level=0.02, d=5, n=60, seed=3, skip=False):
    return eng.load_episodes(noise, level, d, n, seed, skip)


def test_decoder_prediction_and_implied_correction_are_scored_identically():
    d = 5
    eps = _eps(d=d)
    players = eng.load_players("uniform", 0.02, d)
    for kind in ("greedy", "mwpm"):
        pred = players[kind].predict(eps.detectors)
        for det, obs, pr in zip(eps.detectors, eps.observables, pred):
            c = eng.correction_from_prediction(det, int(pr), d, d)
            out = eng.score_correction(det, int(obs), c, d, d)
            assert not out.residual.any()                           # decoders always clear the syndrome
            assert out.success == (int(pr) == int(obs))
        assert eng.play_decoder(players[kind], eps).sum() == (pred == eps.observables).sum()


def test_true_error_is_a_perfect_play_and_complement_is_a_logical_error():
    d = 5
    eps = _eps(d=d, n=80)
    for det, obs in zip(eps.detectors, eps.observables):
        e = eng.true_error(det, int(obs), d, d)
        assert eng.score_correction(det, int(obs), e, d, d).success
        wrong = eng.score_correction(det, int(obs), e ^ 1, d, d)          # flip every qubit: same syndrome
        assert not wrong.success and wrong.reason == "wrong logical branch" and not wrong.residual.any()


def test_leftover_defects_are_reported():
    d = 5
    eps = _eps(d=d, skip=True)
    det, obs = eps.detectors[0], int(eps.observables[0])
    e = eng.true_error(det, obs, d, d)
    bad = e.copy()
    bad[0] ^= 1
    out = eng.score_correction(det, obs, bad, d, d)
    assert not out.success and out.reason == "defects left" and out.residual.any()


def test_syndrome_of_correction():
    assert eng.syndrome_of(np.array([0, 1, 1, 0, 0])).tolist() == [1, 0, 1, 0]
    assert eng.syndrome_of(np.ones(5, dtype=np.uint8)).tolist() == [0, 0, 0, 0]


def test_chain_buckets():
    f = eng.chain_bucket
    assert f(np.array([0, 0, 0])) == "no data flips"
    assert f(np.array([0, 1, 0, 1, 0])) == "isolated errors"
    assert f(np.array([0, 1, 1, 0, 0])) == "chain of 2"
    assert f(np.array([1, 1, 1, 0, 0])) == "chain of 3+"
    assert set(eng.BUCKETS) == {"no data flips", "isolated errors", "chain of 2", "chain of 3+"}


def test_greedy_loses_on_chains_but_mwpm_does_not():
    """The game's teaching hook, measured on the committed 1000-episode bank (d=7, p=0.03):
    pooled over error chains (2+ adjacent flips) MWPM succeeds clearly more often than greedy."""
    d = 7
    eps = eng.load_episodes("uniform", 0.03, d, 1000, seed=5)
    players = eng.load_players("uniform", 0.03, d)
    sg, sm = eng.play_decoder(players["greedy"], eps), eng.play_decoder(players["mwpm"], eps)
    g, m = eng.success_by_bucket(eps, sg), eng.success_by_bucket(eps, sm)
    pooled = lambda b: sum(b[k][0] for k in ("chain of 2", "chain of 3+")) / sum(b[k][1] for k in ("chain of 2", "chain of 3+"))  # noqa: E731
    assert sm.mean() > sg.mean()
    assert pooled(m) > pooled(g) + 0.05                      # observed gap ~ 0.09
    assert sum(v[1] for v in g.values()) == 1000             # every episode lands in exactly one bucket


def test_skip_trivial_is_deterministic_and_never_returns_empty_episodes():
    for noise, level in [("uniform", 0.001), ("biased", 0.01), ("uniform", 0.05)]:
        a = _eps(noise, level, 5, 50, seed=3, skip=True)
        b = _eps(noise, level, 5, 50, seed=3, skip=True)
        assert len(a) == 50 and a.detectors.any(axis=1).all()
        assert np.array_equal(a.detectors, b.detectors)
        assert not np.array_equal(a.detectors, _eps(noise, level, 5, 50, seed=4, skip=True).detectors)


def test_committed_game_bank_is_complete_and_consistent():
    for noise in eng.NOISE_LABELS:
        for level in eng.levels_for(noise):
            for d in eng.DISTANCES:
                path = eng.bank_path(noise, level, d)
                assert path.exists(), path
    ds = eng.SyndromeDataset.load(eng.bank_path("biased", 0.01, 5))
    assert len(ds) == 1000 and ds.n_det == 24 and ds.meta["d"] == 5


def test_bank_is_used_when_present(tmp_path, monkeypatch):
    monkeypatch.setattr(eng, "GAME_BANK_DIR", tmp_path)
    spec = eng.spec_for("uniform", 0.02)
    circuit = eng._circuit("uniform", 0.02, 3)
    ds = make_dataset(circuit, 100, seed=99, meta=dict(d=3))
    ds.save(eng.bank_path("uniform", 0.02, 3))
    eps = eng.load_episodes("uniform", 0.02, 3, 20, seed=1)
    assert eps.from_bank and set(map(bytes, eps.detectors)) <= set(map(bytes, ds.detectors))
    monkeypatch.setattr(eng, "GAME_BANK_DIR", tmp_path / "empty")
    assert not eng.load_episodes("uniform", 0.02, 3, 20, seed=1).from_bank        # falls back to live sampling
    assert spec.kind == "uniform"


def test_neural_player_is_loaded_only_if_a_saved_model_exists(tmp_path):
    assert eng.load_players("uniform", 0.01, 3, models_dir=tmp_path)["nn"] is None
    nd, _ = train_on_circuit(eng._circuit("uniform", 0.01, 3), d=3, steps=20, batch=128, eval_every=10,
                             val_shots=500, hidden=16)
    nd.save(eng.model_path("uniform", 0.01, 3, tmp_path))
    nn = eng.load_players("uniform", 0.01, 3, models_dir=tmp_path)["nn"]
    assert isinstance(nn, NeuralDecoder)
    eps = _eps(d=3)
    assert eng.play_decoder(nn, eps).shape == (len(eps),)


def test_biased_game_picks_a_mwpm_weighting():
    players = eng.load_players("biased", 0.01, 5)
    assert players["mwpm"].name == "MWPM" and players["greedy"] is not None


def test_leaderboard_roundtrip_and_robust_reading(tmp_path):
    path = tmp_path / "lb.jsonl"
    assert eng.read_leaderboard(path) == []
    e = eng.Entry("Ada", "human", "uniform", 0.01, 5, 7, 10, seed=1)
    eng.append_leaderboard(e, path)
    with open(path, "a") as fh:
        fh.write("not json\n{\"bad\": 1}\n")
    got = eng.read_leaderboard(path)
    assert len(got) == 1 and got[0].player == "Ada" and got[0].rate == 0.7
    lo, hi = got[0].ci()
    assert lo < 0.7 < hi


def test_leaderboard_path_can_be_redirected(tmp_path, monkeypatch):
    monkeypatch.setenv("QEC_LEADERBOARD", str(tmp_path / "x.jsonl"))
    eng.append_leaderboard(eng.Entry("Bob", "human", "uniform", 0.01, 3, 1, 2, 0))
    assert (tmp_path / "x.jsonl").exists()


# ----------------------------------------------------------------------------- drawings
def test_figures_render_for_all_states(tmp_path):
    d = 5
    eps = _eps(d=d, skip=True)
    det, obs = eps.detectors[0], int(eps.observables[0])
    e = eng.true_error(det, obs, d, d)
    for kw in (dict(), dict(correction=e), dict(correction=e ^ 1, true_err=e, residual=np.array([1, 0, 0, 1], dtype=np.uint8))):
        fig = viz.defect_figure(det, d, d, title="t", **kw)
        fig.savefig(tmp_path / "f.png")
        viz.plt.close(fig)
        assert (tmp_path / "f.png").stat().st_size > 2000
    fig = viz.chain_example_figure(7)
    fig.savefig(tmp_path / "c.png")
    viz.plt.close(fig)


# ------------------------------------------------------------------- headless Streamlit run
def test_streamlit_app_full_flow(tmp_path, monkeypatch):
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    lb = tmp_path / "lb.jsonl"
    monkeypatch.setenv("QEC_LEADERBOARD", str(lb))
    at = AppTest.from_file(str(ROOT / "game" / "app.py"), default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    at.sidebar.select_slider[1].set_value(3)               # d = 3
    at.sidebar.select_slider[2].set_value(10)              # 10 episodes
    at.run()
    [b for b in at.sidebar.button if "New game" in b.label][0].click().run()
    assert not at.exception, at.exception
    played = 0
    while True:
        at.checkbox[played % 3].check().run()
        [b for b in at.button if b.label == "Submit correction"][0].click().run()
        assert not at.exception, at.exception
        nxt = [b for b in at.button if b.label.startswith(("Next episode", "Finish game"))]
        assert nxt
        played += 1
        nxt[0].click().run()
        assert not at.exception, at.exception
        if nxt[0].label.startswith("Finish"):
            break
    assert played == 10
    assert any("Game over" in s.value for s in at.success)
    kinds = [eng.Entry(**__import__("json").loads(l)).kind for l in lb.read_text().splitlines()]
    assert "human" in kinds and "greedy" in kinds and "mwpm" in kinds
    # other tabs render without error after the game
    for tab_index in range(4):
        assert at.tabs[tab_index] is not None
    assert not at.exception
