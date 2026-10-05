"""Check that this machine is ready to run / train / test the project.

    python -m scripts.check_env              # standard check (~30 s)
    python -m scripts.check_env --quick      # packages + the fastest function checks (~10 s)
    python -m scripts.check_env --full       # also drives the game app headlessly + a short learning check
    python -m scripts.check_env --json out.json --strict

What it does
------------
1. **System**   Python version, OS / CPU, virtual environment, CPU threads.
2. **Packages** every package in ``requirements.txt`` is installed and new enough.
3. **Functions ("mock tests")**  each building block runs on tiny inputs: Stim sampling, the three
   noise models, MWPM (PyMatching, checked against the exact formula), the greedy decoder, a tiny
   PyTorch training + save/load round trip, dataset files, plotting, Qiskit circuit + transpile,
   Qiskit Aer, the offline IBM fake-backend job path (the same code as a real PINQ2 job), the game
   engine, and write permission for the output folders.
4. **Optional** things that are *not* required to run offline (IBM token, pretrained models, notebook tools).

Only the standard library is imported at start-up, so the script still runs -- and tells you what
to install -- when packages are missing.  Nothing is trained for real and nothing in the repo is
modified (all scratch files go to a temporary folder).  The IBM token is never printed.

Exit code: 0 = ready (warnings allowed), 1 = something required is broken (or any warning with ``--strict``).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import json
import os
import platform
import re
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PASS, FAIL, WARN, SKIP, INFO = "PASS", "FAIL", "WARN", "SKIP", "INFO"
MIN_PYTHON = (3, 10)

#: distribution name -> importable module name (they differ for a few packages)
IMPORT_NAME = {
    "numpy": "numpy", "scipy": "scipy", "pandas": "pandas", "matplotlib": "matplotlib", "tqdm": "tqdm",
    "stim": "stim", "pymatching": "pymatching", "torch": "torch", "qiskit": "qiskit",
    "qiskit-aer": "qiskit_aer", "qiskit-ibm-runtime": "qiskit_ibm_runtime", "python-dotenv": "dotenv",
    "streamlit": "streamlit", "pytest": "pytest", "nbformat": "nbformat", "nbconvert": "nbconvert",
    "ipykernel": "ipykernel",
}
#: packages only needed for optional features -> WARN instead of FAIL when missing
OPTIONAL = {"nbformat", "nbconvert", "ipykernel", "tqdm"}
PURPOSE = {
    "stim": "fast syndrome simulation", "pymatching": "MWPM baseline decoder", "torch": "neural decoder",
    "qiskit": "circuits", "qiskit-aer": "noisy simulation", "qiskit-ibm-runtime": "IBM Runtime / fake backends",
    "streamlit": "the game", "pytest": "tests", "matplotlib": "figures", "pandas": "result tables",
    "python-dotenv": "reads .env (IBM token)", "numpy": "arrays", "scipy": "numerics",
    "nbformat": "notebooks", "nbconvert": "notebooks", "ipykernel": "notebooks", "tqdm": "progress bars",
}


@dataclass
class Result:
    section: str
    name: str
    status: str
    detail: str = ""
    hint: str = ""
    seconds: float = 0.0


# ------------------------------------------------------------------------- version helpers
def parse_version(v: str) -> tuple[int, ...]:
    """'2.14.1+cpu' -> (2, 14, 1); tolerant of local/pre-release suffixes."""
    nums = []
    for part in re.split(r"[.+\-]", v):
        m = re.match(r"\d+", part)
        if not m:
            break
        nums.append(int(m.group()))
    return tuple(nums)


def version_ok(installed: str, minimum: str) -> bool:
    a, b = parse_version(installed), parse_version(minimum)
    n = max(len(a), len(b))
    return a + (0,) * (n - len(a)) >= b + (0,) * (n - len(b))


def read_requirements(path: Path = ROOT / "requirements.txt") -> dict[str, str]:
    """``{distribution: minimum version}`` from requirements.txt ('' when unpinned)."""
    out = {}
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*(?:>=\s*([\w.]+))?", line)
        if m:
            out[m.group(1).lower()] = m.group(2) or ""
    return out


def _installed_version(dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


# ------------------------------------------------------------------------ section 1: system
def check_system() -> list[Result]:
    res = []
    v = sys.version_info
    ver = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) < MIN_PYTHON:
        res.append(Result("system", "Python version", FAIL, f"{ver} is too old",
                          f"install Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ (3.11 or 3.12 recommended; tested on 3.11)"))
    elif (v.major, v.minor) >= (3, 14):
        res.append(Result("system", "Python version", WARN, f"{ver} (newer than the tested 3.11-3.12)",
                          "if a package fails to install, create the venv with Python 3.11 or 3.12"))
    else:
        res.append(Result("system", "Python version", PASS, ver))
    res.append(Result("system", "Platform", INFO, f"{platform.system()} {platform.release()} / {platform.machine()}"
                      f" / {os.cpu_count()} CPU threads"))
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix) or bool(os.environ.get("CONDA_PREFIX"))
    res.append(Result("system", "Virtual environment", PASS if in_venv else WARN,
                      sys.prefix if in_venv else "not inside a venv / conda env",
                      "" if in_venv else "recommended: python3 -m venv .venv && source .venv/bin/activate"))
    cwd_ok = (ROOT / "src").is_dir() and (ROOT / "game").is_dir()
    res.append(Result("system", "Repository layout", PASS if cwd_ok else FAIL, str(ROOT),
                      "" if cwd_ok else "run the script from a complete clone of the repository"))
    return res


# ----------------------------------------------------------------------- section 2: packages
def check_packages(requirements: dict[str, str] | None = None) -> list[Result]:
    reqs = requirements if requirements is not None else read_requirements()
    res = []
    for dist, minimum in reqs.items():
        optional = dist in OPTIONAL
        bad = WARN if optional else FAIL
        why = PURPOSE.get(dist, "")
        installed = _installed_version(dist)
        if installed is None:
            res.append(Result("packages", dist, bad, f"not installed ({why})",
                              "pip install -r requirements.txt" if not optional else f"pip install {dist}  (only for {why})"))
            continue
        if minimum and not version_ok(installed, minimum):
            res.append(Result("packages", dist, bad, f"{installed} < required {minimum}",
                              f"pip install --upgrade '{dist}>={minimum}'"))
            continue
        mod = IMPORT_NAME.get(dist, dist.replace("-", "_"))
        try:
            importlib.import_module(mod)
        except Exception as exc:                      # installed but broken (e.g. wrong CPU arch / missing DLL)
            res.append(Result("packages", dist, bad, f"{installed} installed but `import {mod}` failed: "
                              f"{type(exc).__name__}: {str(exc).splitlines()[0][:100] if str(exc) else ''}",
                              f"pip install --force-reinstall --no-cache-dir {dist}"))
            continue
        res.append(Result("packages", dist, PASS, f"{installed}" + (f" (>= {minimum})" if minimum else "")))
    return res


# ------------------------------------------------------------ section 3: function (mock) checks
def _need(*dists: str) -> str | None:
    """Name of the first missing/broken package among ``dists`` (None if all importable)."""
    for d in dists:
        try:
            importlib.import_module(IMPORT_NAME.get(d, d))
        except Exception:
            return d
    return None


def _tmp() -> Path:
    return Path(tempfile.mkdtemp(prefix="qec_env_check_"))


def f_stim_sampling():
    from src.circuits import build_repetition_circuit
    from src.data import sample_syndromes
    from src.noise import NoiseSpec

    c = build_repetition_circuit(3, 3, NoiseSpec.uniform(0.01))
    det, obs = sample_syndromes(c, 1000, seed=0)
    assert det.shape == (1000, 8) and obs.shape == (1000,), (det.shape, obs.shape)
    return f"sampled 1000 shots, {c.num_detectors} detectors, detection rate {det.mean():.3f}"


def f_noise_models():
    from src.circuits import build_repetition_circuit, build_surface_circuit
    from src.noise import NoiseSpec, load_device_calibration

    uni = build_repetition_circuit(3, 3, NoiseSpec.uniform(0.01))
    bia = build_repetition_circuit(3, 3, NoiseSpec.biased(0.01, 5.0, 0.5))
    cal = load_device_calibration("fake_quebec", 3)
    dev = build_repetition_circuit(3, 3, NoiseSpec.from_device(1.0), calibration=cal)
    srf = build_surface_circuit(3, 3, NoiseSpec.uniform(0.005))
    assert uni.num_detectors == bia.num_detectors == dev.num_detectors == 8 and srf.num_detectors > 0
    return f"uniform, biased/correlated, device ({cal.backend} chain {cal.chain}) and surface-code circuits build"


def f_mwpm():
    import numpy as np

    from src.circuits import build_repetition_circuit
    from src.data import sample_syndromes
    from src.decoders import MWPMDecoder
    from src.evaluate import code_capacity_flip_prob, majority_vote_failure
    from src.noise import NoiseSpec

    d, p, n = 3, 0.1, 200_000
    c = build_repetition_circuit(d, 0, NoiseSpec.uniform(p))
    det, obs = sample_syndromes(c, n, seed=1)
    ler = MWPMDecoder.from_circuit(c).logical_error_rate(det, obs)
    exact = majority_vote_failure(d, code_capacity_flip_prob(p))
    z = abs(ler - exact) / np.sqrt(exact * (1 - exact) / n)
    assert z < 4.5, f"MWPM LER {ler:.5f} differs from the exact value {exact:.5f} (z={z:.1f})"
    return f"MWPM LER {ler:.4f} matches the exact formula {exact:.4f} (z={z:.1f})"


def f_greedy():
    import numpy as np

    from src.decoders import GreedyDecoder

    g = GreedyDecoder(5, 0)
    assert g.predict(np.zeros(4, dtype=np.uint8))[0] == 0 and g.predict(np.array([0, 0, 0, 1]))[0] == 1
    return "greedy (human-like) decoder behaves as specified"


def f_torch_train(steps: int = 30):
    import numpy as np
    import torch

    from src.circuits import build_repetition_circuit
    from src.decoders import NeuralDecoder
    from src.noise import NoiseSpec
    from src.training import train_on_circuit

    c = build_repetition_circuit(3, 3, NoiseSpec.uniform(0.03))
    nd, res = train_on_circuit(c, d=3, steps=steps, batch=512, eval_every=10, val_shots=2000, hidden=32)
    assert np.isfinite(res.best_val_loss), "non-finite loss"
    path = nd.save(_tmp() / "m.pt")
    back = NeuralDecoder.load(path)
    x = np.random.default_rng(0).integers(0, 2, (16, nd.n_det), dtype=np.uint8)
    assert np.allclose(nd.predict_logits(x), back.predict_logits(x)), "save/load round trip changed the weights"
    dev = "cpu" + (", MPS (Apple GPU) available" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "") \
        + (", CUDA available" if torch.cuda.is_available() else "")
    return f"{steps}-step training ran (val BCE {res.best_val_loss:.3f}), checkpoint saved + reloaded; torch threads={torch.get_num_threads()} [{dev}]"


def f_dataset_io():
    import numpy as np

    from src.circuits import build_repetition_circuit
    from src.data import SyndromeDataset, make_dataset
    from src.noise import NoiseSpec

    c = build_repetition_circuit(5, 4, NoiseSpec.uniform(0.02))
    ds = make_dataset(c, 500, 3, {"d": 5})
    back = SyndromeDataset.load(ds.save(_tmp() / "ds.npz"))
    assert np.array_equal(back.detectors, ds.detectors) and np.array_equal(back.observables, ds.observables)
    return "dataset .npz save/load round trip is lossless"


def f_plotting():
    import pandas as pd

    from src import plotting
    from src.evaluate import wilson_interval

    rows = []
    for d in (3, 5):
        for p, k in ((0.01, 30), (0.02, 90)):
            lo, hi = wilson_interval(k, 10000)
            rows.append(dict(d=d, p=p, decoder="mwpm_matched", errors=k, ler=k / 10000, ler_lo=lo, ler_hi=hi))
    out = _tmp() / "fig.png"
    plotting.plot_ler_vs_p(pd.DataFrame(rows), ["mwpm_matched"], out, title="env check")
    assert out.stat().st_size > 3000
    return "matplotlib (Agg backend) wrote a PNG figure"


def f_qiskit_transpile():
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    from src.circuits import build_qiskit_circuit
    from src.noise import best_linear_chain, get_fake_backend

    backend = get_fake_backend("fake_quebec")
    chain = best_linear_chain(backend, 5)
    isa = generate_preset_pass_manager(optimization_level=1, backend=backend, initial_layout=chain).run(
        build_qiskit_circuit(3, 2, 1))
    ops = dict(isa.count_ops())
    assert ops.get("ecr", 0) + ops.get("cz", 0) > 0 and ops.get("reset", 0) > 0 and ops.get("measure", 0) > 0
    return f"repetition-code circuit transpiled for {backend.name} on chain {chain} (depth {isa.depth()})"


def f_aer_noiseless():
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2

    from src.circuits import build_qiskit_circuit, measurements_to_syndromes, sampler_result_to_measurements

    qc = transpile(build_qiskit_circuit(3, 2, 1), AerSimulator(), optimization_level=0)
    pub = SamplerV2().run([qc], shots=64).result()[0]
    det, obs = measurements_to_syndromes(sampler_result_to_measurements(pub.data, 3, 2), 3, 2, 1)
    assert not det.any() and not obs.any(), "a noiseless simulation must produce no detection events"
    return "Qiskit Aer runs mid-circuit measure/reset; noiseless run gives zero detection events"


def f_fake_backend_job():
    from src import hardware as hw
    from src.noise import get_fake_backend

    out = _tmp()
    backend = get_fake_backend("fake_quebec")
    run_dir, job = hw.submit(3, 2, 100, backend=backend, simulated=True, out_root=out, seed_simulator=1,
                             log=lambda *_: None)
    assert (run_dir / "meta.json").exists(), "job id was not saved"
    assert hw.fetch(run_dir, job=job, wait=True, log=lambda *_: None), "job did not finish"
    det, obs = hw.load_run(run_dir).syndromes()
    assert det.shape == (100, 6)
    return (f"submit -> fetch -> decode path works on {backend.name} (the same code a real PINQ2 job uses); "
            f"detection rate {det.mean():.3f}")


def f_game_engine():
    from game import engine as eng

    eps = eng.load_episodes("uniform", 0.02, 3, 20, seed=1, skip_trivial=True)
    players = eng.load_players("uniform", 0.02, 3)
    det, obs = eps.detectors[0], int(eps.observables[0])
    e = eng.true_error(det, obs, 3, 3)
    assert eng.score_correction(det, obs, e, 3, 3).success
    succ = eng.play_decoder(players["mwpm"], eps).mean()
    return f"episodes load, scoring works, MWPM player solved {succ:.0%} of 20 test episodes"


def f_game_files():
    from game import engine as eng

    missing = [str(eng.bank_path(n, lv, d).name) for n in eng.NOISE_LABELS for lv in eng.levels_for(n)
               for d in eng.DISTANCES if not eng.bank_path(n, lv, d).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} game-bank files missing (e.g. {missing[0]})")
    return "all pre-generated game syndrome files are present"


def f_streamlit_app():
    import logging

    from streamlit.testing.v1 import AppTest

    for name in list(logging.root.manager.loggerDict):          # headless runs log harmless warnings
        if name.startswith("streamlit"):
            logging.getLogger(name).setLevel(logging.ERROR)
    os.environ.setdefault("QEC_LEADERBOARD", str(_tmp() / "lb.jsonl"))
    at = AppTest.from_file(str(ROOT / "game" / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    [b for b in at.sidebar.button if "New game" in b.label][0].click().run()
    assert not at.exception, at.exception
    at.checkbox[0].check().run()
    [b for b in at.button if b.label == "Submit correction"][0].click().run()
    assert not at.exception, at.exception
    return "the Streamlit game starts, a game is created and a correction is scored (headless)"


def f_learning():
    import numpy as np

    from src.circuits import build_repetition_circuit
    from src.data import sample_syndromes
    from src.decoders import MWPMDecoder
    from src.noise import NoiseSpec
    from src.training import train_on_circuit

    c = build_repetition_circuit(3, 3, NoiseSpec.uniform(0.04))
    nd, _ = train_on_circuit(c, d=3, steps=400, batch=2048, eval_every=100, val_shots=20000)
    det, obs = sample_syndromes(c, 50_000, seed=9)
    nn, mw, none = nd.logical_error_rate(det, obs), MWPMDecoder.from_circuit(c).logical_error_rate(det, obs), float(obs.mean())
    assert nn < none, f"network ({nn:.4f}) is not better than never correcting ({none:.4f})"
    return f"a 400-step network learned to decode: LER {nn:.4f} (MWPM {mw:.4f}, no correction {none:.4f})"


def f_benchmark():
    import torch

    from src.circuits import build_repetition_circuit
    from src.data import SyndromeSampler, sample_syndromes
    from src.decoders.neural import train
    from src.noise import NoiseSpec
    from src.training import make_network

    d = 5
    c = build_repetition_circuit(d, d, NoiseSpec.uniform(0.01))
    net = make_network("mlp", c.num_detectors, d=d, record=True)
    val = sample_syndromes(c, 2000, 1)
    t = time.time()
    train(net, SyndromeSampler(c, 2), val_data=val, steps=20, batch=4096, eval_every=20)
    per = (time.time() - t) / 20
    return (f"~{per * 1000:.0f} ms per training step (d=5, batch 4096) => a default 4000-step model takes about "
            f"{per * 4000 / 60:.1f} min on this machine ({torch.get_num_threads()} threads)")


def f_write_permissions():
    for name in ("results", "data"):
        d = ROOT / name
        d.mkdir(exist_ok=True)
        probe = d / ".write_probe"
        probe.write_text("ok")
        probe.unlink()
    return "results/ and data/ are writable"


# (section, name, function, required distributions, tier)  tier: 'quick' | 'std' | 'full'
FUNCTION_CHECKS = [
    ("functions", "Stim syndrome sampling", f_stim_sampling, ("stim", "numpy"), "quick",
     "pip install stim numpy"),
    ("functions", "Noise models + circuits", f_noise_models, ("stim", "qiskit", "qiskit-ibm-runtime"), "quick",
     "pip install -r requirements.txt  (the device model reads data/calibration/*.json)"),
    ("functions", "MWPM decoder (PyMatching)", f_mwpm, ("stim", "pymatching"), "quick",
     "pip install --upgrade pymatching stim"),
    ("functions", "Greedy decoder", f_greedy, ("numpy",), "quick", "pip install numpy"),
    ("functions", "Neural decoder (PyTorch train/save/load)", f_torch_train, ("torch", "stim"), "quick",
     "pip install torch   (CPU build is enough)"),
    ("functions", "Dataset files", f_dataset_io, ("stim", "numpy"), "quick", "pip install -r requirements.txt"),
    ("functions", "Plotting", f_plotting, ("matplotlib", "pandas"), "quick", "pip install matplotlib pandas"),
    ("functions", "Qiskit circuit + transpile", f_qiskit_transpile, ("qiskit", "qiskit-ibm-runtime"), "quick",
     "pip install --upgrade qiskit qiskit-ibm-runtime"),
    ("functions", "Qiskit Aer simulation", f_aer_noiseless, ("qiskit", "qiskit-aer"), "std",
     "pip install --upgrade qiskit-aer"),
    ("functions", "Offline IBM job path (fake backend)", f_fake_backend_job,
     ("qiskit", "qiskit-aer", "qiskit-ibm-runtime", "stim"), "std", "pip install --upgrade qiskit-aer qiskit-ibm-runtime"),
    ("functions", "Game engine + decoders on game episodes", f_game_engine, ("stim", "pymatching"), "std",
     "run from the repository root; python -m scripts.make_game_bank"),
    ("functions", "Game syndrome bank present", f_game_files, ("stim",), "std",
     "python -m scripts.make_game_bank"),
    ("functions", "Output folders writable", f_write_permissions, (), "quick",
     "check folder permissions of the repository"),
    ("functions", "Training speed estimate", f_benchmark, ("torch", "stim"), "std", ""),
    ("functions", "Streamlit game (headless run)", f_streamlit_app, ("streamlit", "stim", "pymatching"), "full",
     "pip install --upgrade 'streamlit>=1.50'"),
    ("functions", "End-to-end learning check", f_learning, ("torch", "stim", "pymatching"), "full", ""),
]
TIERS = {"quick": 0, "std": 1, "full": 2}


def run_function_checks(level: str, filter_missing: bool = True) -> list[Result]:
    res = []
    for section, name, fn, needs, tier, hint in FUNCTION_CHECKS:
        if TIERS[tier] > TIERS[level]:
            res.append(Result(section, name, SKIP, f"only in --{tier} mode" if tier == "full" else "skipped by --quick"))
            continue
        missing = _need(*needs) if filter_missing else None
        if missing:
            res.append(Result(section, name, SKIP, f"needs '{missing}' (see the package section)", hint))
            continue
        t0 = time.time()
        try:
            detail = fn()
            res.append(Result(section, name, PASS, detail, seconds=time.time() - t0))
        except Exception as exc:
            tb = traceback.extract_tb(exc.__traceback__)
            where = f" [{Path(tb[-1].filename).name}:{tb[-1].lineno}]" if tb else ""
            res.append(Result(section, name, FAIL, f"{type(exc).__name__}: {str(exc)[:160]}{where}", hint,
                              time.time() - t0))
    return res


# ------------------------------------------------------------------- section 4: optional items
def check_optional() -> list[Result]:
    res = []
    env_file = ROOT / ".env"
    ignored = False
    gi = ROOT / ".gitignore"
    if gi.exists():
        ignored = ".env" in gi.read_text().splitlines()
    res.append(Result("optional", ".env is git-ignored", PASS if ignored else FAIL,
                      "yes" if ignored else "`.env` is NOT in .gitignore",
                      "" if ignored else "add a line `.env` to .gitignore before putting a token in it"))
    try:                                   # load .env if python-dotenv exists, without printing anything
        from dotenv import dotenv_values
        file_vals = dotenv_values(env_file) if env_file.exists() else {}
    except Exception:
        file_vals = {}
    have = bool(os.environ.get("PINQ2_TOKEN") or file_vals.get("PINQ2_TOKEN"))
    res.append(Result("optional", "IBM / PINQ2 token", PASS if have else INFO,
                      "PINQ2_TOKEN is set (value not shown)" if have else
                      "not set - fine for offline use; real hardware needs it (copy .env.example to .env)"))
    models = list((ROOT / "results" / "models").glob("*.pt")) if (ROOT / "results" / "models").exists() else []
    res.append(Result("optional", "Pretrained networks (results/models)", PASS if models else INFO,
                      f"{len(models)} found" if models else
                      "none yet - the game's neural-net player stays disabled until you train one "
                      "(python -m scripts.train --d 5 --noise uniform --p 0.01)"))
    return res


# ------------------------------------------------------------------------------- reporting
SYMBOL = {PASS: "[ OK ]", FAIL: "[FAIL]", WARN: "[WARN]", SKIP: "[SKIP]", INFO: "[INFO]"}
COLOR = {PASS: "32", FAIL: "31", WARN: "33", SKIP: "90", INFO: "36"}


def print_report(results: list[Result], color: bool) -> None:
    section = None
    titles = {"system": "1. System", "packages": "2. Python packages", "functions": "3. Function checks (mock runs)",
              "optional": "4. Optional"}
    for r in results:
        if r.section != section:
            section = r.section
            print(f"\n{titles.get(section, section)}")
        tag = SYMBOL[r.status]
        if color:
            tag = f"\033[{COLOR[r.status]}m{tag}\033[0m"
        secs = f"  ({r.seconds:.1f}s)" if r.seconds >= 0.5 else ""
        print(f"  {tag} {r.name:<42s} {r.detail}{secs}")
        if r.hint and r.status in (FAIL, WARN, SKIP):
            print(f"         -> {r.hint}")


def summarize(results: list[Result], strict: bool) -> int:
    n = {s: sum(1 for r in results if r.status == s) for s in (PASS, FAIL, WARN, SKIP, INFO)}
    bad = n[FAIL] + (n[WARN] if strict else 0)
    print("\n" + "-" * 78)
    print(f"{n[PASS]} passed, {n[FAIL]} failed, {n[WARN]} warnings, {n[SKIP]} skipped, {n[INFO]} notes")
    if bad == 0:
        print("READY: the environment can run, train and test the project."
              + ("  (warnings above are optional improvements)" if n[WARN] else ""))
        print("Next:  python -m pytest -q      |  streamlit run game/app.py  |  see README.md")
        return 0
    print("NOT READY. Fix the items marked [FAIL]" + (" / [WARN]" if strict else "") + " above, then re-run this script.")
    fails = [r for r in results if r.status == FAIL]
    pkgs = [r.name for r in fails if r.section == "packages"]
    if pkgs:
        print("Most likely fix (inside your virtual environment):\n    pip install --upgrade pip && pip install -r requirements.txt")
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true", help="packages + fastest function checks only")
    g.add_argument("--full", action="store_true", help="also run the headless game app and a short learning check")
    ap.add_argument("--strict", action="store_true", help="treat warnings as failures")
    ap.add_argument("--json", type=Path, default=None, help="also write the results to this JSON file")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args(argv)
    level = "quick" if args.quick else "full" if args.full else "std"

    print(f"Quantum neural decoder game - environment check  (mode: {level})")
    t0 = time.time()
    results: list[Result] = []
    results += check_system()
    results += check_packages()
    results += run_function_checks(level)
    results += check_optional()
    print_report(results, color=sys.stdout.isatty() and not args.no_color and os.name != "nt")
    code = summarize(results, args.strict)
    print(f"(took {time.time() - t0:.1f}s)")
    if args.json:
        args.json.write_text(json.dumps([asdict(r) for r in results], indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
