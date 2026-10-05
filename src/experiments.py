"""Experiments E1-E3 and E5 (plan section 5); E4 (hardware) lives in :mod:`src.hardware`.

Every function writes a CSV (the numbers behind each figure) and a PNG, and trains
networks through a file cache (``results/models``), so re-running is cheap and an
interrupted run resumes where it stopped.  All randomness is derived from
:func:`src.common.derive_seed`, so results are reproducible.

====  =====================================================================
E1    MWPM baseline sanity check (LER falls with d; analytic checks)
E2    neural decoder vs MWPM on uniform noise (+ events-only ablation)
E3    biased / correlated noise: matched MWPM vs uniform-weight MWPM vs NN,
      transfer (train uniform -> test biased), advantage vs correlation,
      bias-invariance check of the repetition code
E5    rotated surface code d = 3 (stretch): bias matters here
ARCH  MLP vs CNN vs GRU architecture comparison (stretch)
====  =====================================================================
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import stim
import torch

from . import plotting
from .circuits import build_repetition_circuit, build_surface_circuit
from .common import (BIAS_GRID, CORR_GRID, DEFAULT_FAKE_BACKEND, DEVICE_SCALES, DISTANCES, FIGURES_DIR,
                     MODELS_DIR, P_GRID, TABLES_DIR, derive_seed, fmt_p)
from .data import SyndromeSampler, sample_syndromes
from .decoders import GreedyDecoder, MWPMDecoder, NeuralDecoder
from .evaluate import (adaptive_ler, code_capacity_flip_prob, evaluate_decoders, fit_loglog_slope,
                       majority_vote_failure, wilson_interval)
from .noise import NoiseSpec, load_device_calibration
from .training import train_on_circuit

#: wider p range for the E1 threshold-crossing panel (MWPM only, cheap)
P_E1_EXTRA = (0.07, 0.1, 0.13)
#: surface-code physical error rates (threshold of this noise model is ~0.7 %)
P_SURFACE = (0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015)


# =================================================================== config
@dataclass
class Config:
    shots_test: int = 200_000          # plan: >= 1e5 fresh test shots
    shots_val: int = 100_000
    steps: dict = field(default_factory=lambda: {3: 3000, 5: 4000, 7: 6000})
    surface_steps: int = 6000
    batch: int = 4096
    lr: float = 1e-3
    lr_schedule: str = "cosine"
    hidden: int = 256
    depth: int = 3
    eval_every: int = 250
    patience: int = 8
    distances: tuple = DISTANCES
    p_grid: tuple = P_GRID
    p_surface: tuple = P_SURFACE
    bias: float = 5.0                  # default noise model 2
    corr: float = 0.5
    ablation: bool = True              # events-only MLP ablation in E2
    ablation_distances: tuple = (5,)
    workers: int = 1                   # parallel training processes (1 thread each when > 1)
    e1_max_shots: int = 5_000_000
    retrain: bool = False
    results_dir: Path = field(default_factory=lambda: MODELS_DIR.parent)
    verbose: bool = True

    # -------------------------------------------------------------- derived paths
    @property
    def models_dir(self) -> Path:
        return self.results_dir / "models"

    @property
    def tables_dir(self) -> Path:
        return self.results_dir / "tables"

    @property
    def figures_dir(self) -> Path:
        return self.results_dir / "figures"

    def steps_for(self, d: int, arch: str = "mlp") -> int:
        if arch == "gru":
            return min(self.steps.get(d, 3000), 3000)
        return self.steps.get(d, max(self.steps.values()))

    def log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    @classmethod
    def quick(cls, results_dir: Path | str, **kw) -> "Config":
        """Tiny budget for smoke tests / a 2-minute demo (numbers are NOT meaningful)."""
        base = dict(shots_test=20_000, shots_val=10_000, steps={3: 200, 5: 200, 7: 200},
                    surface_steps=200, eval_every=100, distances=(3, 5), p_grid=(0.005, 0.02),
                    p_surface=(0.005, 0.01), e1_max_shots=200_000, results_dir=Path(results_dir),
                    verbose=False)
        base.update(kw)
        return cls(**base)


# ============================================================== circuit / model glue
def make_circuit(code: str, d: int, rounds: int, spec: NoiseSpec) -> stim.Circuit:
    if code == "rep":
        cal = load_device_calibration(spec.device, d) if spec.kind == "device" else None
        return build_repetition_circuit(d, rounds, spec, calibration=cal)
    if code == "surface":
        return build_surface_circuit(d, rounds, spec)
    raise ValueError(f"unknown code {code!r}")


def _key(code: str, d: int, rounds: int, spec: NoiseSpec) -> str:
    return f"{code}|d{d}|r{rounds}|{spec.key}"


def model_path(cfg: Config, code: str, d: int, rounds: int, spec: NoiseSpec, arch: str = "mlp",
               record: bool = True) -> Path:
    tag = arch if (record or code != "rep") else f"{arch}-ev"
    return cfg.models_dir / f"{code}_{spec.key}_d{d}_r{rounds}_{tag}.pt"


def get_nn(cfg: Config, code: str, d: int, rounds: int, spec: NoiseSpec, *, arch: str = "mlp",
           record: bool | None = None, steps: int | None = None) -> NeuralDecoder:
    """Load the cached network for this configuration or train + save it."""
    if record is None:
        record = code == "rep"
    if code != "rep":
        record = False                       # record features need the repetition-code layout
    path = model_path(cfg, code, d, rounds, spec, arch, record)
    if path.exists() and not cfg.retrain:
        return NeuralDecoder.load(path, name=arch.upper())
    circuit = make_circuit(code, d, rounds, spec)
    k = _key(code, d, rounds, spec)
    n_steps = steps or (cfg.surface_steps if code == "surface" else cfg.steps_for(d, arch))
    seeds = dict(val=derive_seed("val", k), train=derive_seed("train", f"{k}|{arch}|{record}"),
                 torch=derive_seed("torch", k))
    cfg.log(f"[train] {path.name}: {n_steps} steps, batch {cfg.batch}, n_det={circuit.num_detectors}")
    nd, res = train_on_circuit(
        circuit, d=d, code=code, arch=arch, record=record, steps=n_steps, batch=cfg.batch, lr=cfg.lr,
        lr_schedule=cfg.lr_schedule, hidden=cfg.hidden, depth=cfg.depth, val_shots=cfg.shots_val,
        eval_every=cfg.eval_every, patience=cfg.patience, seeds=seeds,
        meta=dict(code=code, d=d, rounds=rounds, noise=spec.to_dict()))
    cfg.log(f"        done in {res.seconds:.0f}s  best val BCE {res.best_val_loss:.5f} @ step {res.best_step}")
    nd.name = arch.upper()
    nd.save(path)
    return nd


Job = tuple  # (code, d, rounds, NoiseSpec, arch, record)


def _train_job(args) -> str:
    cfg, code, d, rounds, spec, arch, record = args
    torch.set_num_threads(1)
    get_nn(cfg, code, d, rounds, spec, arch=arch, record=record)
    return model_path(cfg, code, d, rounds, spec, arch, record).name


def prefetch_models(cfg: Config, jobs: list[Job]) -> None:
    """Train (in parallel when ``cfg.workers > 1``) every network in ``jobs`` that is not
    cached yet.  Afterwards :func:`get_nn` is a pure cache read."""
    seen, todo = set(), []
    for code, d, rounds, spec, arch, record in jobs:
        rec = record if code == "rep" else False
        path = model_path(cfg, code, d, rounds, spec, arch, rec)
        if path in seen or (path.exists() and not cfg.retrain):
            continue
        seen.add(path)
        todo.append((code, d, rounds, spec, arch, rec))
    if not todo:
        return
    todo.sort(key=lambda j: (-j[1], j[4]))              # biggest first for load balance
    cfg.log(f"[prefetch] training {len(todo)} networks with {cfg.workers} worker(s)")
    if cfg.workers <= 1:
        for j in todo:
            get_nn(cfg, j[0], j[1], j[2], j[3], arch=j[4], record=j[5])
        return
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=cfg.workers, mp_context=mp.get_context("spawn")) as ex:
        for name in ex.map(_train_job, [(cfg, *j) for j in todo]):
            cfg.log(f"[prefetch] finished {name}")


def mwpm_for(code: str, d: int, rounds: int, spec: NoiseSpec, name: str = "MWPM") -> MWPMDecoder:
    circuit = make_circuit(code, d, rounds, spec)
    return MWPMDecoder.from_circuit(circuit, name)


def best_mwpm(cfg: Config, code: str, d: int, rounds: int, data_spec: NoiseSpec,
              candidates: dict[str, MWPMDecoder]) -> str:
    """Name of the MWPM weighting with the lowest LER on a *validation* set drawn from the
    data distribution (never the test shots, so the choice carries no selection bias).

    Plan sections 5.3 / 9: the MWPM baseline must get "the best available" weights.  DEM
    weights of a correlated circuit are not automatically the best: decomposing a
    correlated event into independent edges inflates those edges' probabilities and can
    make matching *worse* than ignoring the correlation (seen at d = 3).
    """
    circuit = make_circuit(code, d, rounds, data_spec)
    det, obs = sample_syndromes(circuit, cfg.shots_val, derive_seed("val", _key(code, d, rounds, data_spec) + "|mwpm-select"))
    scores = {name: float(np.mean(dec.predict(det) != obs)) for name, dec in candidates.items()}
    return min(scores, key=lambda n: (scores[n], n))


def fresh_test_data(cfg: Config, code: str, d: int, rounds: int, spec: NoiseSpec) -> tuple[np.ndarray, np.ndarray]:
    """Fresh test shots; seed depends only on (code, d, rounds, spec): identical for every decoder."""
    circuit = make_circuit(code, d, rounds, spec)
    return sample_syndromes(circuit, cfg.shots_test, derive_seed("test", _key(code, d, rounds, spec)))


def _point(code: str, d: int, rounds: int, spec: NoiseSpec) -> dict:
    return dict(code=code, noise=spec.kind, p=spec.p if spec.kind != "device" else np.nan,
                bias=spec.bias, corr=spec.corr, scale=spec.scale, d=d, rounds=rounds)


def _save_table(cfg: Config, name: str, df: pd.DataFrame) -> Path:
    cfg.tables_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.tables_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def _fig(cfg: Config, name: str) -> Path:
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    return cfg.figures_dir / f"{name}.png"


# ======================================================================== E1
def run_e1(cfg: Config) -> dict[str, pd.DataFrame]:
    """MWPM baseline: LER falls with d below threshold; analytic cross-checks."""
    cfg.log("== E1: MWPM baseline sanity check")
    rows = []
    for d in cfg.distances:
        for p in tuple(cfg.p_grid) + P_E1_EXTRA:
            spec = NoiseSpec.uniform(p)
            circuit = make_circuit("rep", d, d, spec)
            dec = MWPMDecoder.from_circuit(circuit, "MWPM")
            sampler = SyndromeSampler(circuit, derive_seed("test", _key("rep", d, d, spec) + "|e1"))
            err, shots = adaptive_ler(dec, sampler, min_errors=100, first=cfg.shots_test,
                                      max_shots=cfg.e1_max_shots)
            lo, hi = wilson_interval(err, shots)
            rows.append(dict(**_point("rep", d, d, spec), decoder="mwpm_matched", shots=shots, errors=err,
                             ler=err / shots, ler_lo=lo, ler_hi=hi))
        cfg.log(f"   d={d} done")
    e1 = pd.DataFrame(rows)

    # (a) analytic code-capacity check: MWPM must reproduce the exact majority-vote LER
    cc = []
    for d in cfg.distances:
        for p in (0.02, 0.05, 0.1):
            spec = NoiseSpec.uniform(p)
            circuit = make_circuit("rep", d, 0, spec)
            det, obs = sample_syndromes(circuit, max(cfg.shots_test, 400_000), derive_seed("test", f"cc|{d}|{p}"))
            k = int(np.sum(MWPMDecoder.from_circuit(circuit).predict(det) != obs))
            n = len(obs)
            exact = majority_vote_failure(d, code_capacity_flip_prob(p))
            se = max(np.sqrt(exact * (1 - exact) / n), 1e-12)
            cc.append(dict(d=d, p=p, shots=n, ler_mwpm=k / n, ler_exact=exact, z=(k / n - exact) / se))
    cc = pd.DataFrame(cc)

    # (b) low-p scaling LER ~ p^((d+1)/2)
    sl = []
    for d, sub in e1.groupby("d"):
        use = sub[(sub["p"] <= 0.02) & (sub["errors"] >= 30)]
        sl.append(dict(d=d, slope_measured=fit_loglog_slope(use["p"], use["ler"]),
                       slope_expected=(d + 1) / 2, n_points=len(use)))
    sl = pd.DataFrame(sl)

    _save_table(cfg, "e1_mwpm_uniform", e1)
    _save_table(cfg, "e1_code_capacity_check", cc)
    _save_table(cfg, "e1_slopes", sl)
    plotting.plot_e1(e1, sl, _fig(cfg, "fig_e1_baseline"))
    cfg.log(f"   code-capacity max |z| = {cc['z'].abs().max():.2f};  slopes: "
            + ", ".join(f"d={r.d}: {r.slope_measured:.2f} (expect {r.slope_expected:.0f})" for r in sl.itertuples()))
    return dict(e1=e1, code_capacity=cc, slopes=sl)


# ======================================================================== E2
def run_e2(cfg: Config) -> pd.DataFrame:
    """Neural decoder on uniform noise: expected to (nearly) tie MWPM."""
    cfg.log("== E2: neural decoder on uniform noise")
    rows = []
    jobs = [("rep", d, d, NoiseSpec.uniform(p), "mlp", True) for d in cfg.distances for p in cfg.p_grid]
    if cfg.ablation:
        jobs += [("rep", d, d, NoiseSpec.uniform(p), "mlp", False)
                 for d in cfg.distances if d in cfg.ablation_distances for p in cfg.p_grid]
    prefetch_models(cfg, jobs)
    for d in cfg.distances:
        for p in cfg.p_grid:
            spec = NoiseSpec.uniform(p)
            decs = {"mwpm_matched": mwpm_for("rep", d, d, spec),
                    "nn": get_nn(cfg, "rep", d, d, spec, record=True),
                    "greedy": GreedyDecoder(d, d)}
            if cfg.ablation and d in cfg.ablation_distances:
                decs["nn_events"] = get_nn(cfg, "rep", d, d, spec, record=False)
            det, obs = fresh_test_data(cfg, "rep", d, d, spec)
            for r in evaluate_decoders(decs, det, obs, rounds=d, reference="mwpm_matched"):
                rows.append({**_point("rep", d, d, spec), **r})
            cfg.log(f"   d={d} p={p}: " + "  ".join(f"{r['decoder']}={r['ler']:.5f}" for r in rows[-len(decs):]))
    df = pd.DataFrame(rows)
    _save_table(cfg, "e2_uniform", df)
    plotting.plot_ler_vs_p(df, ["mwpm_matched", "nn", "greedy"], _fig(cfg, "fig_e2_uniform_ler_vs_p"),
                           title="E2 - uniform noise: the network (nearly) ties MWPM")
    if cfg.ablation and "nn_events" in set(df["decoder"]):
        plotting.plot_ler_vs_p(df, ["mwpm_matched", "nn", "nn_events"], _fig(cfg, "fig_e2_ablation_input"),
                               title="E2 ablation - effect of the syndrome-record input feature",
                               distances=[d for d in cfg.distances if d in cfg.ablation_distances])
    adv = df[df["decoder"] == "nn"]
    plotting.plot_advantage(adv, _fig(cfg, "fig_e2_advantage_uniform"), xcol="p", logx=True,
                            xlabel="physical error rate p", title="E2 - NN advantage, uniform noise")
    return df


# ======================================================================== E3
def _biased(cfg: Config, p: float, bias: float | None = None, corr: float | None = None) -> NoiseSpec:
    return NoiseSpec.biased(p, cfg.bias if bias is None else bias, cfg.corr if corr is None else corr)


def run_e3(cfg: Config) -> pd.DataFrame:
    """Main result: LER vs p under biased/correlated noise for matched / uniform MWPM and NNs."""
    cfg.log(f"== E3: biased/correlated noise (bias r={cfg.bias:g}, corr={cfg.corr:g})")
    rows = []
    prefetch_models(cfg, [("rep", d, d, sp, "mlp", True) for d in cfg.distances for p in cfg.p_grid
                          for sp in (_biased(cfg, p), NoiseSpec.uniform(p))])
    for d in cfg.distances:
        for p in cfg.p_grid:
            spec_b, spec_u = _biased(cfg, p), NoiseSpec.uniform(p)
            decs = {"mwpm_matched": mwpm_for("rep", d, d, spec_b),
                    "mwpm_uniform": mwpm_for("rep", d, d, spec_u),
                    "nn": get_nn(cfg, "rep", d, d, spec_b),
                    "nn_transfer": get_nn(cfg, "rep", d, d, spec_u),
                    "greedy": GreedyDecoder(d, d)}
            det, obs = fresh_test_data(cfg, "rep", d, d, spec_b)
            ref = best_mwpm(cfg, "rep", d, d, spec_b, {k: decs[k] for k in ("mwpm_matched", "mwpm_uniform")})
            for r in evaluate_decoders(decs, det, obs, rounds=d, reference=ref):
                rows.append({**_point("rep", d, d, spec_b), "ref_decoder": ref, **r})
            cfg.log(f"   d={d} p={p} (ref {ref}): " + "  ".join(f"{r['decoder']}={r['ler']:.5f}" for r in rows[-len(decs):]))
    df = pd.DataFrame(rows)
    _save_table(cfg, "e3_biased", df)
    plotting.plot_ler_vs_p(
        df, ["mwpm_uniform", "mwpm_matched", "nn_transfer", "nn", "greedy"],
        _fig(cfg, "fig_e3_biased_ler_vs_p"),
        title=f"E3 - biased/correlated noise (r={cfg.bias:g}, corr={cfg.corr:g}): where learning helps")
    plotting.plot_advantage(df[df["decoder"] == "nn"], _fig(cfg, "fig_e3_advantage_vs_p"), xcol="p", logx=True,
                            xlabel="physical error rate p",
                            title="E3 - NN advantage over noise-matched MWPM")
    return df


def run_e3_corr_sweep(cfg: Config, distances: tuple = (3, 5), p: float = 0.01,
                      corrs: tuple = CORR_GRID) -> pd.DataFrame:
    """NN advantage as the strength of the correlated channel grows (fixed p)."""
    cfg.log(f"== E3b: advantage vs correlation strength (p={p})")
    rows = []
    prefetch_models(cfg, [("rep", d, d, _biased(cfg, p, corr=c), "mlp", True) for d in distances for c in corrs])
    for d in distances:
        for c in corrs:
            spec = _biased(cfg, p, corr=c)
            decs = {"mwpm_matched": mwpm_for("rep", d, d, spec), "nn": get_nn(cfg, "rep", d, d, spec),
                    "mwpm_uniform": mwpm_for("rep", d, d, NoiseSpec.uniform(p))}
            det, obs = fresh_test_data(cfg, "rep", d, d, spec)
            ref = best_mwpm(cfg, "rep", d, d, spec, {k: decs[k] for k in ("mwpm_matched", "mwpm_uniform")})
            for r in evaluate_decoders(decs, det, obs, rounds=d, reference=ref):
                rows.append({**_point("rep", d, d, spec), "ref_decoder": ref, **r})
            cfg.log(f"   d={d} corr={c} (ref {ref}): " + "  ".join(f"{r['decoder']}={r['ler']:.5f}" for r in rows[-3:]))
    df = pd.DataFrame(rows)
    _save_table(cfg, "e3_corr_sweep", df)
    plotting.plot_advantage(df[df["decoder"] == "nn"], _fig(cfg, "fig_e3_advantage_vs_corr"), xcol="corr",
                            xlabel="correlated-flip strength (units of p)",
                            title=f"E3 - NN advantage grows with correlation (p={p:g})")
    return df


def run_e3_bias_check(cfg: Config, d: int = 5, p: float = 0.01) -> pd.DataFrame:
    """The repetition code is blind to Z errors: matched MWPM's DEM, hence its LER, must not
    depend on the bias r (at fixed X-flip rate).  Reported as a table (and unit-tested)."""
    cfg.log("== E3c: bias-invariance check of the repetition code")
    rows = []
    for r in BIAS_GRID:
        spec = _biased(cfg, p, bias=r)
        circuit = make_circuit("rep", d, d, spec)
        det, obs = sample_syndromes(circuit, cfg.shots_test, derive_seed("test", f"biascheck|{r}"))
        dec = MWPMDecoder.from_circuit(circuit)
        k = int(np.sum(dec.predict(det) != obs))
        lo, hi = wilson_interval(k, len(obs))
        rows.append(dict(d=d, p=p, bias=r, corr=cfg.corr, shots=len(obs), errors=k, ler=k / len(obs),
                         ler_lo=lo, ler_hi=hi))
    df = pd.DataFrame(rows)
    _save_table(cfg, "e3_bias_invariance", df)
    return df


# ======================================================================== E5
def run_e5(cfg: Config, d: int = 3, p_bias: float = 0.005) -> dict[str, pd.DataFrame]:
    """Stretch: rotated surface code d=3 (circuit-level), MWPM vs NN; bias sweep."""
    cfg.log("== E5: rotated surface code d=3")
    rounds = d
    rows = []
    prefetch_models(cfg, [("surface", d, rounds, NoiseSpec.uniform(p), "mlp", False) for p in cfg.p_surface]
                    + [("surface", d, rounds, NoiseSpec.biased(p_bias, r, 0.0), "mlp", False) for r in BIAS_GRID])
    for p in cfg.p_surface:
        spec = NoiseSpec.uniform(p)
        decs = {"mwpm_matched": mwpm_for("surface", d, rounds, spec),
                "nn": get_nn(cfg, "surface", d, rounds, spec)}
        det, obs = fresh_test_data(cfg, "surface", d, rounds, spec)
        for r in evaluate_decoders(decs, det, obs, rounds=rounds, reference="mwpm_matched"):
            rows.append({**_point("surface", d, rounds, spec), **r})
        cfg.log(f"   p={p}: " + "  ".join(f"{r['decoder']}={r['ler']:.5f}" for r in rows[-2:]))
    uni = pd.DataFrame(rows)
    plotting.plot_ler_vs_p(uni, ["mwpm_matched", "nn"], _fig(cfg, "fig_e5_surface_ler_vs_p"),
                           title="E5 - rotated surface code d=3, uniform circuit noise")
    rows = []
    for r_bias in BIAS_GRID:
        spec = NoiseSpec.biased(p_bias, r_bias, 0.0)
        decs = {"mwpm_matched": mwpm_for("surface", d, rounds, spec),
                "mwpm_uniform": mwpm_for("surface", d, rounds, NoiseSpec.uniform(p_bias)),
                "nn": get_nn(cfg, "surface", d, rounds, spec)}
        det, obs = fresh_test_data(cfg, "surface", d, rounds, spec)
        ref = best_mwpm(cfg, "surface", d, rounds, spec, {k: decs[k] for k in ("mwpm_matched", "mwpm_uniform")})
        for r in evaluate_decoders(decs, det, obs, rounds=rounds, reference=ref):
            rows.append({**_point("surface", d, rounds, spec), "ref_decoder": ref, **r})
        cfg.log(f"   bias r={r_bias} (ref {ref}): " + "  ".join(f"{r['decoder']}={r['ler']:.5f}" for r in rows[-3:]))
    bias = pd.DataFrame(rows)
    _save_table(cfg, "e5_surface_uniform", uni)
    _save_table(cfg, "e5_surface_bias", bias)
    plotting.plot_advantage(bias[bias["decoder"] == "nn"], _fig(cfg, "fig_e5_advantage_vs_bias"), xcol="bias",
                            group="d", xlabel="bias r = p_Z / p_X", logx=True,
                            title=f"E5 - surface code d=3: NN advantage vs Z-bias (p={p_bias:g})")
    return dict(uniform=uni, bias=bias)


# ====================================================== device-derived noise (stretch)
def run_device(cfg: Config, scales: tuple = DEVICE_SCALES, device: str = DEFAULT_FAKE_BACKEND,
               aer_d: int = 3, aer_train: int = 100_000, aer_test: int = 40_000) -> dict[str, pd.DataFrame]:
    """Stretch: decoders for *device-derived* noise.

    (a) Stim approximation of the calibration of ``device``: MWPM from its DEM vs a network
        trained on it, at several noise multipliers.
    (b) Transfer to Qiskit Aer (backend noise model, a more faithful device simulation than the
        Pauli approximation): decoders built from the approximation vs a network trained
        directly on Aer-collected data (the "hardware-noise-trained decoder").
    """
    cfg.log(f"== DEVICE: decoders for noise derived from {device}")
    specs = {(d, s): NoiseSpec.from_device(s, device) for d in cfg.distances for s in scales}
    prefetch_models(cfg, [("rep", d, d, sp, "mlp", True) for (d, s), sp in specs.items()])
    rows = []
    for (d, s), spec in specs.items():
        decs = {"mwpm_matched": mwpm_for("rep", d, d, spec), "nn": get_nn(cfg, "rep", d, d, spec),
                "greedy": GreedyDecoder(d, d)}
        det, obs = fresh_test_data(cfg, "rep", d, d, spec)
        for r in evaluate_decoders(decs, det, obs, rounds=d, reference="mwpm_matched"):
            rows.append({**_point("rep", d, d, spec), **r})
        cfg.log(f"   d={d} x{s:g}: " + "  ".join(f"{r['decoder']}={r['ler']:.5f}" for r in rows[-3:]))
    stim_df = pd.DataFrame(rows)
    _save_table(cfg, "e4b_device_stim", stim_df)
    plot_df = stim_df.assign(p=stim_df["scale"])
    plotting.plot_ler_vs_p(plot_df, ["mwpm_matched", "nn", "greedy"], _fig(cfg, "fig_e4b_device_ler_vs_scale"),
                           title=f"Device-derived noise ({device} calibration, Pauli approximation)",
                           xlabel="noise multiplier (x calibrated error rates)")

    # ---- (b) transfer to Aer ------------------------------------------------------------
    from .common import RAW_DIR
    from .data import SyndromeDataset
    from .hardware import prepare_circuit, run_aer
    from .noise import get_fake_backend
    from .training import train_on_dataset

    d = aer_d
    backend = get_fake_backend(device)
    _, _, chain, cal = prepare_circuit(d, d, backend, logical_state=1)
    spec = NoiseSpec.from_device(1.0, device)

    def aer_set(role: str, n: int) -> SyndromeDataset:
        path = RAW_DIR / f"aer_{device}_d{d}_r{d}_{role}_{n}.npz"
        if path.exists():
            return SyndromeDataset.load(path)
        cfg.log(f"   [aer] collecting {n} {role} shots (~{n / 1000:.0f}s)")
        det, obs = run_aer(d, d, n, backend, chain, logical_state=1, seed=derive_seed("hardware", f"{device}|{d}|{role}"))
        ds = SyndromeDataset(det, obs, dict(source="aer", backend=device, chain=chain, d=d, rounds=d,
                                            role=role, shots=n, code="rep"))
        ds.save(path)
        return ds
    tr, va, te = aer_set("train", aer_train), aer_set("val", aer_train // 5), aer_set("test", aer_test)
    nn_aer_path = cfg.models_dir / f"rep_aer-{device}_d{d}_r{d}_mlp.pt"
    if nn_aer_path.exists() and not cfg.retrain:
        nn_aer = NeuralDecoder.load(nn_aer_path)
    else:
        cfg.log("   [aer] training the network on Aer data")
        nn_aer, _ = train_on_dataset(tr, va, d=d, steps=cfg.steps_for(d), batch=2048, lr=cfg.lr,
                                     lr_schedule=cfg.lr_schedule, eval_every=cfg.eval_every, patience=cfg.patience,
                                     seeds=dict(train=derive_seed("train", f"aer|{device}|{d}"),
                                                torch=derive_seed("torch", f"aer|{device}|{d}")),
                                     meta=dict(noise=spec.to_dict(), source="Aer data"), log=None)
        nn_aer.save(nn_aer_path)
    decs = {"mwpm_matched": mwpm_for("rep", d, d, spec), "nn": get_nn(cfg, "rep", d, d, spec),
            "nn_aer": nn_aer, "greedy": GreedyDecoder(d, d)}
    rows = []
    for r in evaluate_decoders(decs, te.detectors, te.observables, rounds=d, reference="mwpm_matched"):
        rows.append(dict(source=f"Aer ({device})", d=d, rounds=d, train_shots=len(tr), **r))
    aer_df = pd.DataFrame(rows)
    _save_table(cfg, "e4b_aer_transfer", aer_df)
    plotting.plot_hardware(aer_df, _fig(cfg, "fig_e4b_aer_transfer"),
                           title=f"Decoders on Aer data ({device}), d={d}: approximation vs trained-on-Aer",
                           subtitle="MWPM / NN use the Stim approximation of the calibration; 'NN trained on Aer data' "
                                    "learned from simulated device data only.  Simulation, not hardware.")
    cfg.log("   Aer: " + "  ".join(f"{r['decoder']}={r['ler']:.5f}" for r in rows))
    return dict(stim=stim_df, aer=aer_df)


# ================================================================ architecture study
def run_arch_comparison(cfg: Config, d: int = 5, ps: tuple = (0.005, 0.01, 0.02),
                        archs: tuple = ("mlp", "cnn", "gru"), gru_ps: tuple = (0.01,)) -> pd.DataFrame:
    """MLP vs 1-D CNN vs GRU (all with the record feature) on uniform and biased noise.
    The GRU is ~5x slower to train on CPU, so it is only run at the ``gru_ps`` error rates."""
    cfg.log(f"== ARCH: architecture comparison at d={d}")
    rows = []
    use = lambda a, p: a != "gru" or p in gru_ps          # noqa: E731
    prefetch_models(cfg, [("rep", d, d, NoiseSpec.uniform(p) if kind == "uniform" else _biased(cfg, p), a, True)
                          for kind in ("uniform", "biased") for p in ps for a in archs if use(a, p)])
    for kind in ("uniform", "biased"):
        for p in ps:
            spec = NoiseSpec.uniform(p) if kind == "uniform" else _biased(cfg, p)
            decs = {"mwpm_matched": mwpm_for("rep", d, d, spec)}
            if kind == "biased":
                decs["mwpm_uniform"] = mwpm_for("rep", d, d, NoiseSpec.uniform(p))
            for a in archs:
                if use(a, p):
                    decs[a if a != "mlp" else "nn"] = get_nn(cfg, "rep", d, d, spec, arch=a, record=True)
            det, obs = fresh_test_data(cfg, "rep", d, d, spec)
            cands = {k: v for k, v in decs.items() if k.startswith("mwpm")}
            ref = best_mwpm(cfg, "rep", d, d, spec, cands)
            for r in evaluate_decoders(decs, det, obs, rounds=d, reference=ref):
                rows.append({**_point("rep", d, d, spec), "ref_decoder": ref, **r})
            cfg.log(f"   {kind} p={p}: " + "  ".join(f"{r['decoder']}={r['ler']:.5f}" for r in rows[-len(decs):]))
    df = pd.DataFrame(rows)
    _save_table(cfg, "arch_comparison", df)
    for kind in ("uniform", "biased"):
        sub = df[df["noise"] == kind]
        plotting.plot_ler_vs_p(sub, ["mwpm_matched", "nn", "cnn", "gru"],
                               _fig(cfg, f"fig_arch_{kind}"), title=f"Architectures, {kind} noise, d={d}",
                               distances=[d])
    return df


# ============================================================== manifest / driver
def write_manifest(cfg: Config, extra: dict | None = None) -> Path:
    """Record library versions + the config of this run in ``run_manifest.json``.

    The file accumulates one entry per invocation under ``runs`` so that experiments run
    separately (or on different days) all stay documented."""
    import pymatching
    import qiskit
    path = cfg.results_dir / "run_manifest.json"
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(path.read_text()) if path.exists() else {}
    manifest["versions"] = dict(
        python=platform.python_version(), platform=platform.platform(), stim=stim.__version__,
        pymatching=pymatching.__version__, torch=str(torch.__version__), qiskit=qiskit.__version__,
        numpy=np.__version__, pandas=pd.__version__)
    manifest.setdefault("runs", []).append(dict(
        created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        config={k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.__dict__.items()},
        **(extra or {})))
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path


RUNNERS: dict[str, Callable[[Config], object]] = {
    "e1": run_e1, "e2": run_e2, "e3": run_e3, "e3b": run_e3_corr_sweep, "e3c": run_e3_bias_check,
    "e5": run_e5, "arch": run_arch_comparison, "device": run_device,
}
