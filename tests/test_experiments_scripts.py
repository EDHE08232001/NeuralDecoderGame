"""Experiment pipelines (tiny budgets) and command-line scripts (run in-process, outputs in tmp dirs)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src import experiments as ex
from src.data import SyndromeDataset, make_dataset
from src.noise import NoiseSpec

import game.engine as eng


@pytest.fixture()
def cfg(tmp_path):
    return ex.Config.quick(tmp_path / "res", verbose=False)


# ------------------------------------------------------------------------- experiments
def test_e1_reproduces_analytic_checks(cfg):
    out = ex.run_e1(cfg)
    assert out["code_capacity"]["z"].abs().max() < 4.5            # MWPM == exact majority vote
    for r in out["slopes"].itertuples():                          # LER ~ p^((d+1)/2)
        assert abs(r.slope_measured - r.slope_expected) < 0.7
    e1 = out["e1"]
    low = e1[(e1["p"] == 0.005)].set_index("d")["ler"]
    assert low[5] < low[3]                                        # larger code is better below threshold
    assert (cfg.figures_dir / "fig_e1_baseline.png").exists() and (cfg.tables_dir / "e1_slopes.csv").exists()


def test_get_nn_trains_once_then_uses_the_cache(cfg):
    spec = NoiseSpec.uniform(0.02)
    a = ex.get_nn(cfg, "rep", 3, 3, spec)
    path = ex.model_path(cfg, "rep", 3, 3, spec)
    assert path.exists()
    mtime = path.stat().st_mtime_ns
    time.sleep(0.01)
    b = ex.get_nn(cfg, "rep", 3, 3, spec)
    assert path.stat().st_mtime_ns == mtime                       # not retrained
    xx = np.random.default_rng(0).integers(0, 2, (10, a.n_det), dtype=np.uint8)
    assert np.allclose(a.predict_logits(xx), b.predict_logits(xx))
    assert a.meta["noise"]["p"] == 0.02 and a.meta["record"] is True and a.meta["history"]
    ev = ex.get_nn(cfg, "rep", 3, 3, spec, record=False)          # events-only ablation model is separate
    assert ex.model_path(cfg, "rep", 3, 3, spec, "mlp", False).name.endswith("mlp-ev.pt") and ev.meta["record"] is False


def test_prefetch_skips_cached_and_dedupes(cfg, capsys):
    cfg.verbose = True
    spec = NoiseSpec.uniform(0.02)
    jobs = [("rep", 3, 3, spec, "mlp", True)] * 3
    ex.prefetch_models(cfg, jobs)
    assert "training 1 networks" in capsys.readouterr().out
    ex.prefetch_models(cfg, jobs)
    assert capsys.readouterr().out == ""                           # everything cached


def test_best_mwpm_selects_the_better_weighting_on_validation_data(cfg):
    spec_b = NoiseSpec.biased(0.02, 5.0, 0.5)
    cands = {"mwpm_matched": ex.mwpm_for("rep", 3, 3, spec_b), "mwpm_uniform": ex.mwpm_for("rep", 3, 3, NoiseSpec.uniform(0.02))}
    pick = ex.best_mwpm(cfg, "rep", 3, 3, spec_b, cands)
    assert pick in cands
    det, obs = ex.sample_syndromes(ex.make_circuit("rep", 3, 3, spec_b), 50_000, seed=1)
    ler = {k: float(np.mean(v.predict(det) != obs)) for k, v in cands.items()}
    assert ler[pick] <= min(ler.values()) * 1.1                    # chosen weighting is (near-)best on fresh data


def test_e2_and_e3_tables_have_expected_structure(cfg):
    cfg.distances, cfg.p_grid, cfg.ablation = (3,), (0.02,), False
    e2 = ex.run_e2(cfg)
    assert set(e2["decoder"]) == {"mwpm_matched", "nn", "greedy"}
    assert {"ler", "ler_lo", "ler_hi", "adv", "adv_lo", "adv_hi", "shots", "errors"} <= set(e2.columns)
    assert (e2["ler_lo"] <= e2["ler"]).all() and (e2["ler"] <= e2["ler_hi"]).all()
    assert (e2["shots"] == cfg.shots_test).all()                   # identical shots for every decoder
    e3 = ex.run_e3(cfg)
    assert set(e3["decoder"]) == {"mwpm_matched", "mwpm_uniform", "nn", "nn_transfer", "greedy"}
    assert e3["ref_decoder"].isin(["mwpm_matched", "mwpm_uniform"]).all()
    for name in ("fig_e2_uniform_ler_vs_p", "fig_e3_biased_ler_vs_p", "fig_e3_advantage_vs_p"):
        assert (cfg.figures_dir / f"{name}.png").stat().st_size > 5000


def test_bias_invariance_table_for_repetition_code(cfg):
    df = ex.run_e3_bias_check(cfg)
    assert list(df["bias"]) == [1.0, 5.0, 20.0]
    # same DEM => same distribution; allow sampling noise only
    spread = df["ler"].max() - df["ler"].min()
    assert spread < 6 * np.sqrt(df["ler"].mean() * (1 - df["ler"].mean()) / df["shots"].iloc[0])


def test_surface_code_experiment_runs(cfg):
    cfg.p_surface = (0.01,)
    out = ex.run_e5(cfg)
    assert set(out["uniform"]["decoder"]) == {"mwpm_matched", "nn"}
    assert set(out["bias"]["bias"]) == {1.0, 5.0, 20.0}


def test_device_experiment_runs_with_aer_transfer(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr("src.common.RAW_DIR", tmp_path / "raw")
    cfg.distances = (3,)
    out = ex.run_device(cfg, scales=(1.0,), aer_train=300, aer_test=200)
    assert set(out["aer"]["decoder"]) == {"mwpm_matched", "nn", "nn_aer", "greedy"}
    assert (cfg.figures_dir / "fig_e4b_aer_transfer.png").exists()


def test_manifest_accumulates_runs(cfg):
    ex.write_manifest(cfg, dict(experiments=["a"]))
    path = ex.write_manifest(cfg, dict(experiments=["b"]))
    m = json.loads(path.read_text())
    assert [r["experiments"] for r in m["runs"]] == [["a"], ["b"]] and "stim" in m["versions"]


# ------------------------------------------------------------------------------ scripts
def test_collect_dataset_stim_roles_give_disjoint_seeds(tmp_path):
    from scripts import collect_dataset as cd

    outs = {}
    for role in ("train", "test"):
        out = tmp_path / f"{role}.npz"
        assert cd.main(["--source", "stim", "--d", "3", "--p", "0.03", "--shots", "3000", "--role", role,
                        "--out", str(out)]) == 0
        outs[role] = SyndromeDataset.load(out)
    assert outs["train"].n_det == 8 and len(outs["train"]) == 3000
    assert outs["train"].meta["noise"]["p"] == 0.03 and outs["train"].meta["role"] == "train"
    assert not np.array_equal(outs["train"].detectors, outs["test"].detectors)
    again = tmp_path / "again.npz"
    cd.main(["--source", "stim", "--d", "3", "--p", "0.03", "--shots", "3000", "--role", "train", "--out", str(again)])
    assert np.array_equal(SyndromeDataset.load(again).detectors, outs["train"].detectors)   # reproducible


def test_collect_dataset_biased_device_and_aer(tmp_path):
    from scripts import collect_dataset as cd

    for extra in (["--noise", "biased", "--p", "0.02", "--bias", "5", "--corr", "0.5"],
                  ["--noise", "device", "--scale", "2"]):
        out = tmp_path / "x.npz"
        assert cd.main(["--source", "stim", "--d", "3", "--shots", "1000", "--out", str(out)] + extra) == 0
        assert len(SyndromeDataset.load(out)) == 1000
    aer = tmp_path / "aer.npz"
    assert cd.main(["--source", "aer", "--d", "3", "--rounds", "2", "--shots", "200", "--out", str(aer)]) == 0
    ds = SyndromeDataset.load(aer)
    assert len(ds) == 200 and ds.meta["source"] == "aer" and ds.n_det == 6


def test_collect_dataset_from_hardware_run(tmp_path):
    from scripts import collect_dataset as cd
    from scripts import run_hardware as rh

    assert rh.main(["simulate", "--d", "3", "--rounds", "2", "--shots", "200", "--nn-steps", "20", "--stim-shots",
                    "2000", "--aer-shots", "100", "--out-dir", str(tmp_path / "hw"),
                    "--figure", str(tmp_path / "fig.png")]) == 0
    run_dir = next((tmp_path / "hw").iterdir())
    assert (tmp_path / "fig.png").exists()
    out = tmp_path / "hwset.npz"
    assert cd.main(["--source", "hardware", "--run-dir", str(run_dir), "--out", str(out)]) == 0
    ds = SyndromeDataset.load(out)
    assert len(ds) == 200 and "simulated" in ds.meta["source"]
    assert rh.main(["list"]) == 0


def test_train_script_circuit_and_dataset_modes(tmp_path, capsys):
    from scripts import train as tr

    out = tmp_path / "m.pt"
    assert tr.main(["--d", "3", "--noise", "uniform", "--p", "0.03", "--steps", "40", "--eval-every", "20",
                    "--val-shots", "2000", "--test-shots", "5000", "--hidden", "32", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert out.exists() and "mwpm_matched" in text and "greedy" in text and "paired CI" in text
    from src.decoders import NeuralDecoder
    assert NeuralDecoder.load(out).meta["record"] is True
    # dataset mode: stored train / val sets
    from src.experiments import make_circuit
    c = make_circuit("rep", 3, 3, NoiseSpec.uniform(0.03))
    make_dataset(c, 4000, 1).save(tmp_path / "tr.npz")
    make_dataset(c, 1000, 2).save(tmp_path / "va.npz")
    out2 = tmp_path / "m2.pt"
    assert tr.main(["--d", "3", "--p", "0.03", "--steps", "30", "--eval-every", "15", "--hidden", "32", "--dataset",
                    str(tmp_path / "tr.npz"), "--val-dataset", str(tmp_path / "va.npz"), "--test-shots", "3000",
                    "--out", str(out2)]) == 0
    assert out2.exists()
    with pytest.raises(SystemExit):                                # dataset without validation set is an error
        tr.main(["--d", "3", "--dataset", str(tmp_path / "tr.npz")])


def test_train_script_other_architectures_and_surface(tmp_path):
    from scripts import train as tr

    assert tr.main(["--d", "3", "--arch", "cnn", "--steps", "20", "--eval-every", "10", "--val-shots", "1000",
                    "--test-shots", "2000", "--out", str(tmp_path / "c.pt")]) == 0
    assert tr.main(["--code", "surface", "--d", "3", "--p", "0.005", "--steps", "20", "--eval-every", "10",
                    "--val-shots", "1000", "--test-shots", "2000", "--out", str(tmp_path / "s.pt")]) == 0
    assert tr.main(["--d", "3", "--no-record", "--steps", "20", "--eval-every", "10", "--val-shots", "1000",
                    "--test-shots", "2000", "--out", str(tmp_path / "e.pt")]) == 0


def test_make_game_bank_writes_expected_files(tmp_path, monkeypatch):
    from scripts import make_game_bank as mb

    monkeypatch.setattr(eng, "GAME_BANK_DIR", tmp_path)
    assert mb.main(["--noise", "uniform", "--d", "3", "--episodes", "50"]) == 0
    files = sorted(tmp_path.glob("*.npz"))
    assert len(files) == len(eng.levels_for("uniform"))
    assert len(SyndromeDataset.load(files[0])) == 50


def test_run_experiments_cli_quick(tmp_path):
    from scripts import run_experiments as re_

    assert re_.main(["e1", "e3c", "--quick", "--results-dir", str(tmp_path)]) == 0
    assert (tmp_path / "tables" / "e3_bias_invariance.csv").exists() and (tmp_path / "run_manifest.json").exists()
    with pytest.raises(SystemExit):
        re_.main(["e9"])
