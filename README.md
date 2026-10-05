# Learning to Decode: a Neural QEC Decoder Game

*Qiskit Fall Fest 2026 hackathon project - "Gamifying Error Detection, Mitigation and Correction".*

Quantum computers make mistakes all the time. This project asks a simple question and builds the tools to answer it:

> **When can a small, trained neural network fix a quantum computer's mistakes better than the classic textbook method - and when can it not?**

It contains a **simulator**, three **decoders** (a human-like rule, the classic algorithm, a neural network), an **evaluation pipeline**, a **hardware path** for real IBM quantum computers (with an offline fallback), and a **playable game** where you compete against the decoders.

**Contents:** [The idea in plain words](#1-the-idea-in-plain-words) - [Quick start](#2-quick-start) - [Check your setup](#3-check-your-setup-before-anything-else) - [Play the game](#4-play-the-game) - [Train a decoder](#5-train-a-decoder-on-your-machine) - [Collect a dataset](#6-collect-a-dataset) - [Evaluate and interpret results](#7-evaluate-and-interpret-results) - [Real IBM hardware](#8-run-on-real-ibm-hardware-pinq2-or-simulate-it) - [Tests](#9-tests) - [Repository map](#10-repository-map) - [Honest limits](#11-what-is-and-is-not-claimed) - [Troubleshooting](#12-troubleshooting)

---

## 1. The idea in plain words

**No quantum knowledge needed.** Think of it like this.

* **The problem.** A quantum bit (qubit) is fragile: heat, noise and imperfect hardware randomly *flip* it. A computer that cannot protect its bits cannot compute anything useful.
* **The trick (error correction).** Do what you do when you shout a message across a noisy room: **repeat it**. We store *one* bit of information in several qubits (3, 5 or 7 - "the code distance `d`"). If one flips, the others still "remember" the truth.
* **Checks without peeking.** We cannot look at the qubits directly (looking destroys quantum information). Instead we ask *"do these two neighbours agree?"* over and over. Whenever two neighbours disagree, a **defect** lights up. A pattern of lit defects is called the **syndrome**.
* **The decoder.** A *decoder* looks at the defects and must decide: *"was the stored bit flipped or not?"* Get it right and the information survives; get it wrong and we have a **logical error**. The fraction of attempts that fail is the **logical error rate (LER)** - *lower is better*. This is the number we plot everywhere.
* **Why it is a puzzle.** One flipped qubit lights two neighbouring defects: easy, pair them up. But several flips in a row ("a chain") leave defects far apart, and a rule like "pair the closest ones" goes wrong. Measurements are noisy too, so some defects are lies.

The three players:

| Player | How it thinks | Plain-words analogy |
|---|---|---|
| **Greedy** (stand-in for a human) | pair the nearest defects first | "connect the dots that look close" |
| **MWPM** (minimum-weight perfect matching) | find the *cheapest overall* explanation of all defects, using known error probabilities | a planner who sees the whole map |
| **Neural network** | learned from millions of simulated examples | a student who has seen a million exams |

**The research question.** If noise is *simple* (every qubit fails independently), MWPM is almost perfect, so the network can at best tie. If noise is *messy* - for example two qubits tend to fail **together** - MWPM's assumptions break, and a network that has *seen* such patterns might do better. A negative result ("it never wins") is also a valid finding and is reported honestly.

**Three kinds of noise are used** (plan section 4): (1) *uniform* independent noise (the control), (2) *biased / correlated* noise, where pairs of qubits flip together, and (3) *device-derived* noise built from the published calibration numbers of a real IBM chip.

**Where Qiskit fits.** Qiskit builds the actual circuits, simulates them with realistic device noise (Aer), and sends them to real IBM hardware through `qiskit-ibm-runtime` (the PINQ2 allocation). Stim does the fast bulk simulation, PyMatching is MWPM, PyTorch is the network.

### Mini glossary
| Word | Meaning |
|---|---|
| qubit | the quantum version of a bit |
| repetition code | store one bit in `d` qubits, like repeating a message `d` times |
| distance `d` | how many qubits; bigger = more protection (when noise is low enough) |
| round | one full set of neighbour-agreement checks (we do `d` rounds) |
| defect / detection event | a check whose answer *changed* |
| syndrome | the whole pattern of defects |
| logical error | the decoder's final answer about the stored bit is wrong |
| LER | logical error rate = fraction of runs ending in a logical error |
| `p` | physical error probability: how often a basic operation goes wrong (0.01 = 1%) |
| threshold | noise level below which a bigger code is better, above which it is worse |
| MWPM | classic decoder (minimum-weight perfect matching) |
| Aer | Qiskit's noisy simulator; **Stim** = very fast simulator for this kind of noise |
| fake backend | an offline copy of a real IBM chip's calibration data (no account needed) |

---

## 2. Quick start

Works on macOS (your M1 MacBook, `zsh`), Linux, and Windows. Needs **Python 3.10 - 3.12** (3.11 is what was tested).

```zsh
git clone https://github.com/EDHE08232001/QuantumNeuralQecDecoderGame.git
cd QuantumNeuralQecDecoderGame

python3 -m venv .venv
source .venv/bin/activate            # Windows PowerShell:  .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt      # a few minutes (PyTorch is the big one)

python -m scripts.check_env          # <- verifies everything, see next section
```

Shortcut with `make` (macOS/Linux): `make setup && make check`.

*Course environment (`qff26`)?* Activate it and run `pip install -r requirements.txt` inside it. Stim and PyMatching are **not** part of the course environment and are installed from `requirements.txt`; the exact versions this repo was developed with are recorded in `results/run_manifest.json`.

*Apple Silicon note:* everything runs on the CPU; the networks are tiny, and the CPU is fast enough. You do not need a GPU.

---

## 3. Check your setup before anything else

```zsh
python -m scripts.check_env            # standard check, ~30 s
python -m scripts.check_env --quick    # ~10 s
python -m scripts.check_env --full     # + headless game run + short learning check
```

It checks, in order: **(1)** Python version / virtual env, **(2)** that every package in `requirements.txt` is installed *and* new enough *and* importable, **(3)** "mock runs" of every building block on tiny inputs - Stim sampling, the noise models, MWPM vs. an exact formula, a 30-step PyTorch training with save/load, dataset files, plotting, Qiskit transpile, Qiskit Aer, the offline IBM-job path, the game engine - and **(4)** optional extras (IBM token present?, pretrained models?). It also estimates **how long training will take on your machine**.

Every problem is printed with the command that fixes it, and the exit code is `0` (ready) or `1` (not ready), so it can also be used in scripts. It never prints your token and never changes the repository. Missing packages do not crash it - it tells you what to install. `--strict` treats warnings as failures; `--json out.json` saves the report.

Sample (healthy machine):
```
32 passed, 0 failed, 0 warnings, 2 skipped, 3 notes
READY: the environment can run, train and test the project.
```

---

## 4. Play the game

```zsh
streamlit run game/app.py            # or: make game
```
A browser tab opens - **Syndrome Hunter**:

1. Choose a **noise model** (uniform / biased-correlated / "real IBM device"), a **difficulty** (`p`) and a **distance** `d` in the sidebar, press **New game**.
2. You see a grid of rounds; orange circles are defects. Tick the **data qubits you think were flipped** and press **Submit**. Red diamonds show defects your answer would leave behind.
3. The truth is revealed, you get a point if your answer is exactly right, and you see what **Greedy, MWPM and the neural net** answered on the *same* episode.
4. The **Arena** tab compares everyone's success rate (95% confidence bars) over your episodes. The **Why decoders?** tab shows the "chain" example where local rules fail, and success rate by error pattern.

Nothing is trained live: the game reads pre-generated syndromes (`data/game_bank/`) and saved models (`results/models/`). The neural-net player appears once a model for that setting exists - the sidebar prints the exact command to create one (e.g. `python -m scripts.train --d 5 --noise uniform --p 0.01`, about 2 minutes on a laptop). Regenerate the syndrome bank any time with `python -m scripts.make_game_bank`.

---

## 5. Train a decoder on your machine

### One model (recommended first step)
```zsh
python -m scripts.train --d 5 --noise uniform --p 0.01
```
This follows the plan's recipe: **fresh simulated data at every step** (so overfitting is impossible), Adam optimizer, cross-entropy loss, a fixed validation set for early stopping, and the best weights are kept. At the end it prints a **fair comparison on brand-new test shots** (same shots for every decoder):

```
decoder               LER                95% CI   LER(MWPM)-LER(dec) [95% paired CI]
mwpm_matched      0.00xxx  [0.00xxx, 0.00xxx]
nn                0.00xxx  [...]                  +0.0000x [...]
greedy            ...
```

More examples:
```zsh
python -m scripts.train --d 5 --noise biased --p 0.01 --bias 5 --corr 0.5 --steps 6000   # correlated noise
python -m scripts.train --d 3 --noise device --scale 1          # noise derived from an IBM chip's calibration
python -m scripts.train --d 5 --noise uniform --p 0.01 --arch cnn    # or --arch gru
python -m scripts.train --d 5 --noise uniform --p 0.01 --no-record   # the plan-literal "events only" input
python -m scripts.train --code surface --d 3 --noise uniform --p 0.005   # stretch: surface code
python -m scripts.train --help
```
Models are saved to `results/models/` (naming: `custom_<code>_<noise>_d<d>_r<rounds>_<arch>.pt`). **Training time:** ~25 ms/step on a 4-core CPU for `d=5` (see `check_env`), so the default 3000 steps is about 1-2 minutes. Larger `d` and lower `p` need more steps (`--steps 6000+`).

### The whole study from the plan (optional, slow)
```zsh
python -m scripts.run_experiments --quick                  # 2-minute smoke test, tiny budget (numbers meaningless)
python -m scripts.run_experiments e1                       # MWPM sanity check only (30 s, no training)
python -m scripts.run_experiments e2 e3 --workers 4        # neural vs MWPM, uniform and correlated noise
python -m scripts.run_experiments all --workers 4          # everything (several hours on a laptop)
```
| Experiment | What it does |
|---|---|
| `e1` | MWPM only: does a bigger code help? Matches an exact formula and the `p^((d+1)/2)` scaling law |
| `e2` | neural net vs MWPM on uniform noise (+ an input-feature ablation) |
| `e3`, `e3b`, `e3c` | **main result**: biased/correlated noise, advantage vs correlation strength, "Z-bias is invisible" check |
| `e5` | stretch: rotated surface code `d=3` |
| `arch` | MLP vs CNN vs GRU |
| `device` | decoders for IBM-calibration noise, and transfer to Qiskit Aer data |

Training is cached: re-running only trains what is missing (`--retrain` forces it). Everything is seeded; the same command gives bit-identical networks. Outputs: tables in `results/tables/*.csv`, figures in `results/figures/*.png`, a run record in `results/run_manifest.json`.

> **Status of this repository:** only `e1` (cheap, no training) was run in full and its results are committed in `results/`. The other experiments are implemented, unit-tested end to end on tiny budgets, and exercised by the notebooks, but **their full-budget results have not been generated here** - run them on your machine.

### The notebooks
`notebooks/01_baseline.ipynb`, `02_neural.ipynb`, `03_hardware.ipynb` (plain-language, each runs in about a minute with `QUICK = True`). Open with `jupyter lab` (`pip install jupyterlab` if needed) or regenerate with `python -m scripts.build_notebooks`.

---

## 6. Collect a dataset

A dataset is a table of independent experiments. Each row = one run of the memory experiment:
* **`detectors`** - which checks lit up (a row of 0/1, length `(d-1)*(rounds+1)`), and
* **`observables`** - the *label*: was the stored bit really flipped (0/1)?

Data comes from three sources:

| Source | What it is | Speed | Use |
|---|---|---|---|
| `stim` | fast Pauli-noise simulation | millions of shots/s | training (labels are free) |
| `aer` | Qiskit Aer with an IBM backend's noise model, running the real transpiled circuit | ~1 ms/shot | device-like data |
| `hardware` | shots saved from a real IBM job | queue-limited | the real thing |

```zsh
# 1M training shots, d=5, uniform noise (plan: 1e5-1e6 per (d, p))
python -m scripts.collect_dataset --source stim --d 5 --noise uniform --p 0.01 --shots 1000000 --role train
# the held-out test set - a DIFFERENT seed, never used for training
python -m scripts.collect_dataset --source stim --d 5 --noise uniform --p 0.01 --shots 200000 --role test
# correlated noise / device-derived noise
python -m scripts.collect_dataset --source stim --d 5 --noise biased --p 0.01 --bias 5 --corr 0.5 --shots 500000
python -m scripts.collect_dataset --source stim --d 3 --noise device --scale 1 --shots 500000
# Qiskit Aer with the FakeQuebec noise model
python -m scripts.collect_dataset --source aer --d 3 --rounds 3 --shots 100000 --fake-backend fake_quebec
# a fetched hardware run -> dataset
python -m scripts.collect_dataset --source hardware --run-dir results/hardware/<run folder>
```
Files go to `data/raw/` (git-ignored) as compressed `.npz` with a JSON description of exactly how they were made. `--role train|val|test|bank` derives a reproducible seed from the role *and* the configuration, so splits never overlap. Load one in Python:

```python
from src.data import SyndromeDataset
ds = SyndromeDataset.load("data/raw/stim_rep_uniform_p0.01_d5_r5_train.npz")
ds.detectors, ds.observables, ds.meta
```
Train from stored data instead of live simulation: `python -m scripts.train --d 5 --p 0.01 --dataset data/raw/train.npz --val-dataset data/raw/val.npz`.

> You do **not** need to collect data to train: the default training samples fresh data on the fly. Collect datasets to inspect data, to train from Aer/hardware measurements, or to share a fixed benchmark.

---

## 7. Evaluate and interpret results

### What is measured
* **LER (logical error rate)** per shot; lower is better. A decoder that never corrects has LER = "how often the bit flipped"; every real decoder must beat that.
* **95% Wilson confidence interval** on every LER (error bars). Few failures = wide interval. A hollow `v` marker on a plot means *zero* failures were seen, and the point shows the 95% **upper bound**.
* **NN advantage** = `LER(best MWPM) - LER(NN)`, computed as a **paired** difference on the same test shots with its own 95% interval. **Positive = the network is better. Only believe it if the whole interval is above 0.**

### Fairness rules built in
Same test shots for every decoder; test data is never used for training/validation (separate seeded streams); at least 10^5 test shots by default; the MWPM baseline gets the **best available weights**: both the noise-matched weights *and* uniform weights are tried and the one that is better on a *separate validation set* is used as the reference. (This matters: for correlated noise the "matched" weights can be *worse* than ignoring the correlation, because splitting a correlated event into independent pieces mis-teaches the matcher. We saw this at `d=3`.)

### How to read each figure
| Figure (`results/figures/`) | Look for |
|---|---|
| `fig_e1_baseline.png` | curves for bigger `d` sit **lower** at small `p` and **cross** near the threshold; measured low-`p` slopes ~ `(d+1)/2` |
| `fig_e2_uniform_ler_vs_p.png` | NN ~ MWPM (tie) on simple noise; greedy clearly worse and *not improving with d* |
| `fig_e2_ablation_input.png` | effect of giving the net the "syndrome record" feature (see below) |
| `fig_e3_biased_ler_vs_p.png` | **main result**: do the orange (NN) curves drop below the best blue (MWPM) ones? |
| `fig_e3_advantage_*.png` | advantage with confidence bars; above 0 = learning helps; grows with correlation? |
| `fig_e4*_*.png` | hardware/Aer vs the Stim approximation (see section 8) |
| `fig_e5_*.png` | surface code: where the *bias* knob matters |

Every plot has a CSV twin in `results/tables/` with the exact numbers.

### What outcomes mean (a cheat sheet)
* **NN ~ MWPM on uniform noise** - expected: MWPM is near-optimal there. Not a failure.
* **NN clearly worse than MWPM** - usually *under-training* (more `--steps`, larger `d`, or lower `p` needs more data, since failures are rare). Do not conclude "neural nets cannot decode" from a short run; check the training curve and try 2x steps.
* **NN better with a positive, significant advantage on correlated noise** - supports the hypothesis: learning exploits structure matching ignores.
* **No advantage anywhere** - report it; it is a valid answer ("MWPM is near-optimal for this noise").
* **Stim vs Aer vs hardware gaps** - Stim is an *approximation* built from calibration numbers; gaps measure how much that approximation misses (T1 decay is symmetrised, etc.). Hardware vs Aer measures how much the simulator misses.

### Reference numbers you can reproduce
* `python -m scripts.run_experiments e1` (committed in `results/tables/`): measured low-`p` slopes **1.92 / 2.95 / 4.05** for `d = 3 / 5 / 7` against the theory **2 / 3 / 4**, and MWPM matches the exact majority-vote formula within statistical error (|z| < 2.4). That confirms the whole simulate-decode-count pipeline.
* `notebooks/02_neural.ipynb` (tiny 800-step budget, `d=3`, `p=0.02`): uniform noise, NN LER 0.0248 vs MWPM 0.0264 vs greedy 0.0320. Correlated noise: NN 0.0952 vs best MWPM 0.0969 (paired advantage +0.0017, CI [+0.0012, +0.0022]) - and the noise-*matched* MWPM was much worse (0.134) than the uniform-weights one. Treat these as illustrations from a small run, not final results; your numbers will differ with budget.

### Design note: the "syndrome record" input
The network always receives the raw detection events. By default it also computes, *inside the model*, the running parity of those events over the rounds (which is simply the raw check outcomes). It adds **no information**, but integrating events over time is a parity task that small ReLU networks learn slowly; in a development test at `d=5, p=0.02` and 3000 steps this took the network from about 1.5x MWPM's error rate to about 1.04x. The plan-literal events-only input remains available (`--no-record`) and is compared in experiment `e2`.

---

## 8. Run on real IBM hardware (PINQ2) - or simulate it

**Offline demo (no account, no internet to IBM):** runs the *same code path* as a real job on an IBM **fake backend** and labels everything *SIMULATED*.
```zsh
python -m scripts.run_hardware simulate --d 3 --rounds 3 --shots 4096
```
Output: a run folder in `results/hardware/`, a table `analysis.csv`, and a figure comparing the saved shots, the Stim approximation, and (for real runs) Aer.

**Real hardware** (needs your PINQ2 token):
```zsh
cp .env.example .env          # then edit .env: PINQ2_TOKEN=...   (.env is git-ignored; never commit/screenshot it)
python -m scripts.run_hardware submit --d 3 --rounds 3 --shots 4096                  # least-busy device
python -m scripts.run_hardware submit --d 3 --rounds 3 --shots 4096 --backend ibm_quebec
python -m scripts.run_hardware fetch   results/hardware/<run folder> --wait          # when the job is done
python -m scripts.run_hardware analyze results/hardware/<run folder>
python -m scripts.run_hardware list
```
* **Submit early** - queues are long. The job id, backend, calibration snapshot and transpiled circuit are written to disk *immediately* after submission.
* The circuit uses a well-calibrated chain of qubits, mid-circuit measurement and reset. If a backend lacks them, use `--rounds 0` (final-readout-only fallback).
* `--logical-state 1` (default) prepares the logical `|1..1>` state, the harder one under qubit decay.
* Expect a short repetition code on today's devices to **not necessarily beat an unencoded qubit** - the point is the measured data and the simulation-vs-hardware gap.
* If you have an organizer-provided instance CRN, set `PINQ2_INSTANCE` in `.env`.

> **Honesty note:** the authenticated IBM path (`get_service`, real `ibm_*` backends) could not be run in the development environment (no token). It is covered by tests with stand-ins; every other step (circuit, transpile, `SamplerV2`, saving, fetching, decoding, analysis) is exercised for real against fake backends. Expect to fix small account-specific details (instance/plan/backend names) on your first real submission.

---

## 9. Tests

```zsh
python -m pytest                    # everything (~3-5 min)
python -m pytest -m "not slow"      # faster
make test
```
About 130 tests cover: equivalence of our circuit builder with Stim's own generator, detector conversion, MWPM vs an exact formula, greedy behaviour, networks (shapes, save/load, **bit-reproducible training**), Wilson/paired confidence intervals (with regression tests for bugs found along the way), datasets, noise/calibration, the whole hardware workflow (job id saved first, fetch of unfinished jobs, token never written to disk), the game engine and a **headless run of the Streamlit app**, all command-line scripts, and the environment checker itself.

---

## 10. Repository map

```
README.md  requirements.txt  Makefile  pytest.ini  .env.example  .gitignore
src/
  noise.py        NoiseSpec (uniform / biased-correlated / device) + IBM calibration -> Stim noise
  circuits.py     Stim + Qiskit repetition-code circuits, surface code, syndrome conversion
  data.py         sampling, dataset save/load           training.py   shared training helpers
  decoders/       mwpm.py (PyMatching)  greedy.py  neural.py (MLP / CNN / GRU + training loop)
  evaluate.py     LER, Wilson & paired CIs, analytic checks
  plotting.py     the evaluation figures                experiments.py  E1-E5 pipelines
  hardware.py     IBM submit / fetch / analyze + offline fake-backend path
game/             app.py (Streamlit)  engine.py (testable logic)  viz.py (drawings)
scripts/          check_env  train  collect_dataset  run_experiments  run_hardware
                  make_game_bank  build_notebooks
notebooks/        01_baseline  02_neural  03_hardware
data/             game_bank/ (pre-generated syndromes)  calibration/ (device snapshots)  raw/ (git-ignored)
results/          figures/  tables/  models/  hardware/  run_manifest.json
tests/
```
Plan coverage: E1-E3 and E5 (`src/experiments.py`), E4 (`src/hardware.py`), game (`game/`), the three notebooks, the repository layout of plan section 7, the fairness protocol of section 9, and the risk mitigations (Aer fallback, noise-matched MWPM, token hygiene).

---

## 11. What is and is not claimed

* **Deviations from the plan, and why.** (a) The detector vector has `(d-1)(rounds+1)` entries, not `rounds*(d-1)`: the final data readout adds one more layer. (b) In a repetition code **Z-type ("phase") errors are invisible**, so the plan's bias knob `r` changes nothing for it (verified by a test). The correlated pair flips are the knob that matters for the repetition code; the bias knob matters for the surface code (`e5`). (c) A "syndrome record" input feature and a best-of-two MWPM reference are added (explained above). (d) A greedy decoder stands in for the "human" on leaderboards.
* **Approximations.** Stim supports only Pauli noise, so the "device" model is a Pauli approximation of calibration data (T1 decay symmetrised, readout symmetrised, reset error set equal to readout error - which matched Aer's average detection rate within 1% in our check). It is **not** exact device fidelity.
* **Scale.** Small codes only. We do not claim to beat state-of-the-art decoders (AlphaQubit etc.) or to scale to large codes. Neural decoders get harder to train as `d` grows and `p` shrinks.
* **Limits of the repetition code.** It protects against bit flips only; it is a teaching/benchmark code, not a full quantum code.

---

## 12. Troubleshooting

| Symptom | Fix |
|---|---|
| `check_env` says a package is missing/old | `pip install -r requirements.txt` inside the activated venv |
| `pip install torch` very slow | CPU build: `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| `No module named src` | run commands from the repository root, with `python -m scripts.<name>` |
| game: "No saved neural network" | train one with the command shown in the sidebar; the other players still work |
| training NN is far worse than MWPM | more `--steps`, especially for `d=7` or small `p`; look at the validation loss curve |
| `streamlit` complains about `width="stretch"` | `pip install --upgrade "streamlit>=1.50"` |
| real-hardware submit fails | check `.env` (`PINQ2_TOKEN`, optional `PINQ2_INSTANCE`), the backend name, and try `--rounds 0` |
| Windows: `make` not found | use the plain commands shown in this README |

*References:* IBM Quantum Learning "Foundations of QEC"; Qiskit Global Summer School 2025 Lab 4; the Stim and PyMatching documentation; Google Quantum AI, "Quantum error correction below the surface code threshold"; DeepMind's AlphaQubit.
