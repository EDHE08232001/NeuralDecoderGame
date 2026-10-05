"""Build (and optionally execute) the three notebooks of the plan (notebooks/01..03).

    python -m scripts.build_notebooks            # write + execute (needs nbformat, nbconvert, ipykernel)
    python -m scripts.build_notebooks --no-run   # only write the .ipynb files

All notebooks have a QUICK switch in their first code cell (default True: runs in about a minute).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"

SETUP = '''import sys, pathlib, warnings
warnings.filterwarnings("ignore")
ROOT = pathlib.Path.cwd().parent if pathlib.Path.cwd().name == "notebooks" else pathlib.Path.cwd()
sys.path.insert(0, str(ROOT))
QUICK = True   # False = bigger budgets (slower, tighter error bars)
'''

NB1 = [
("md", "# 01 - Baseline: noisy qubits, syndromes and MWPM\n\n**In plain words.** We store one bit of quantum information in `d` qubits (a *repetition code*). Noise flips some of them. "
       "Parity checks (*syndromes*) tell us *where neighbours disagree* without reading the data. A *decoder* looks at the lit-up checks and guesses whether the stored bit was flipped. "
       "MWPM (minimum-weight perfect matching) is the classic decoder: it explains the lit checks with the fewest, most likely errors."),
("code", SETUP + '''import numpy as np
from src.noise import NoiseSpec
from src.circuits import build_repetition_circuit
from src.data import sample_syndromes
from src.decoders import MWPMDecoder
from src.evaluate import wilson_interval, majority_vote_failure, code_capacity_flip_prob, fit_loglog_slope'''),
("md", "## 1. Sanity check against an exact formula\nWith no repeated measurement (`rounds=0`) the best decoder is a majority vote, so MWPM must match the binomial formula. If it does not, there is a bug."),
("code", '''shots = 100_000 if QUICK else 1_000_000
for d in (3, 5, 7):
    c = build_repetition_circuit(d, 0, NoiseSpec.uniform(0.1))
    det, obs = sample_syndromes(c, shots, seed=1)
    ler = MWPMDecoder.from_circuit(c).logical_error_rate(det, obs)
    print(f"d={d}: MWPM {ler:.5f}   exact {majority_vote_failure(d, code_capacity_flip_prob(0.1)):.5f}")'''),
("md", "## 2. Does a bigger code help? (E1)\nBelow a threshold error rate, larger `d` should give a *lower* logical error rate, scaling like `p^((d+1)/2)`."),
("code", '''import matplotlib.pyplot as plt
ps = [0.005, 0.01, 0.02, 0.05]
fig, ax = plt.subplots(figsize=(5, 3.6))
for d in (3, 5, 7):
    lers = []
    for p in ps:
        c = build_repetition_circuit(d, d, NoiseSpec.uniform(p))
        det, obs = sample_syndromes(c, shots, seed=2)
        lers.append(MWPMDecoder.from_circuit(c).logical_error_rate(det, obs))
    ax.loglog(ps, np.maximum(lers, 1e-7), "o-", label=f"d={d}")
    print(d, [f"{x:.5f}" for x in lers], "slope", round(fit_loglog_slope(ps[:3], lers[:3]), 2), "expected", (d + 1) / 2)
ax.set_xlabel("physical error rate p"); ax.set_ylabel("logical error rate"); ax.legend(); plt.show()'''),
]

NB2 = [
("md", "# 02 - Neural decoder vs MWPM\n\n**In plain words.** Instead of a hand-written rule, we *train* a small neural network on millions of simulated syndromes (the right answer is free in simulation). "
       "Question: *when does learning beat the classic decoder?* Our hypothesis: tie on simple independent noise, win when errors come in correlated pairs that MWPM assumes away."),
("code", SETUP + '''import numpy as np, tempfile
from src.experiments import Config, get_nn, mwpm_for, fresh_test_data, best_mwpm
from src.noise import NoiseSpec
from src.decoders import GreedyDecoder
from src.evaluate import evaluate_decoders
cfg = Config.quick(tempfile.mkdtemp(), verbose=True) if QUICK else Config(results_dir=pathlib.Path(tempfile.mkdtemp()), verbose=True)
if QUICK: cfg.steps = {3: 800, 5: 800, 7: 800}; cfg.shots_test = 100_000
d, p = 3, 0.02'''),
("md", "## Uniform noise (control) - expect a tie"),
("code", '''spec = NoiseSpec.uniform(p)
decs = {"mwpm_matched": mwpm_for("rep", d, d, spec), "nn": get_nn(cfg, "rep", d, d, spec), "greedy": GreedyDecoder(d, d)}
det, obs = fresh_test_data(cfg, "rep", d, d, spec)
import pandas as pd
pd.DataFrame(evaluate_decoders(decs, det, obs, rounds=d, reference="mwpm_matched"))[["decoder","ler","ler_lo","ler_hi","adv","adv_lo","adv_hi"]]'''),
("md", "## Correlated noise - the interesting case\n`adv = LER(best MWPM) - LER(NN)`; positive and with a confidence interval above 0 means the network is genuinely better."),
("code", '''spec_b = NoiseSpec.biased(p, bias=5.0, corr=0.5)
decs = {"mwpm_matched": mwpm_for("rep", d, d, spec_b), "mwpm_uniform": mwpm_for("rep", d, d, spec),
        "nn": get_nn(cfg, "rep", d, d, spec_b), "nn_transfer": get_nn(cfg, "rep", d, d, spec)}
det, obs = fresh_test_data(cfg, "rep", d, d, spec_b)
ref = best_mwpm(cfg, "rep", d, d, spec_b, {k: decs[k] for k in ("mwpm_matched", "mwpm_uniform")})
print("reference MWPM weighting (chosen on validation data):", ref)
pd.DataFrame(evaluate_decoders(decs, det, obs, rounds=d, reference=ref))[["decoder","ler","ler_lo","ler_hi","adv","adv_lo","adv_hi"]]'''),
]

NB3 = [
("md", "# 03 - Hardware experiment (offline, SIMULATED)\n\n**Important.** This notebook runs the *same code path* as a real IBM/PINQ2 job, but against an IBM **fake backend** (a local noisy simulation). Results are simulated, not from a quantum computer. "
       "For real hardware see the README (`python -m scripts.run_hardware submit ...` with your token in `.env`)."),
("code", SETUP + '''import tempfile
from pathlib import Path
from src import hardware as hw
from src.noise import get_fake_backend
backend = get_fake_backend("fake_quebec")
out = Path(tempfile.mkdtemp())
shots = 2000 if QUICK else 8192
run_dir, job = hw.submit(3, 3, shots, backend=backend, simulated=True, out_root=out, seed_simulator=1)
hw.fetch(run_dir, job=job, wait=True)'''),
("code", '''df = hw.analyze_run(run_dir, nn_steps=300 if QUICK else 3000, stim_shots=50_000, log=print)
df[["source", "decoder", "shots", "errors", "ler", "ler_lo", "ler_hi"]]'''),
("code", '''from IPython.display import Image
hw.plot_run(run_dir, df)
Image(filename=str(run_dir / "fig_hardware_vs_sim.png"))'''),
("md", "**Reading the figure.** Each dot is a logical error rate with a 95% interval. `Aer` = simulation with the backend's noise model; `Stim approx.` = a simpler model built only from the calibration numbers. A gap between them is the *approximation error*, and on real hardware a further gap appears between simulation and reality."),
]


def build(run: bool) -> None:
    import nbformat as nbf

    NB_DIR.mkdir(exist_ok=True)
    for name, cells in (("01_baseline", NB1), ("02_neural", NB2), ("03_hardware", NB3)):
        nb = nbf.v4.new_notebook()
        nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src) for kind, src in cells]
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        path = NB_DIR / f"{name}.ipynb"
        if run:
            from nbconvert.preprocessors import ExecutePreprocessor
            ExecutePreprocessor(timeout=900, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(NB_DIR)}})
        nbf.write(nb, path)
        print("wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-run", action="store_true")
    build(not ap.parse_args().no_run)
