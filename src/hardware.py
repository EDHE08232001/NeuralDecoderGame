"""E4 -- repetition-code memory experiment on IBM hardware (PINQ2) and its offline fallback.

Workflow (plan section 5.4; submit EARLY, queues are long)::

    submit  -> build the Qiskit circuit, transpile onto a well-calibrated chain,
               run it with ``SamplerV2``, **save the job id immediately**
    fetch   -> later: retrieve the finished job, save raw shot-by-shot data
    analyze -> decode with MWPM and the neural decoder and compare with the Aer
               prediction and the Stim approximation

Everything a run produces is written to ``results/hardware/<run>/``::

    meta.json         job id, backend, shots, d, rounds, chain, status, versions
    calibration.json  calibration snapshot of the chain at submission time
    circuit.qpy       the transpiled (ISA) circuit that was executed
    raw_record.npz    packed shot-by-shot measurement record  (after fetch)
    counts.json       counts of the full measurement record   (after fetch)
    analysis.csv      decoder results per source              (after analyze)

Credentials: the IBM/PINQ2 token is read from the environment variable ``PINQ2_TOKEN``
(loaded from the git-ignored ``.env``); it is never logged, never written to disk and
never part of any file produced here.

Offline fallback (plan section 10, checkpoint): with ``simulated=True`` the *same*
code path runs against an IBM **fake backend** through ``qiskit_ibm_runtime.SamplerV2``
(a local Aer simulation with the backend's noise model).  Those runs are flagged
``"simulated": true`` in ``meta.json`` and must be presented as simulated.

.. note::
   The authenticated path (``get_service`` / real backends) cannot be exercised
   without a token and network access to IBM, so it is covered by tests that inject a
   local sampler, not by a live job.  Everything else is identical for real and
   simulated runs.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .circuits import (build_qiskit_circuit, build_repetition_circuit, measurements_to_syndromes,
                       num_qubits, sampler_result_to_measurements)
from .common import DEFAULT_FAKE_BACKEND, HARDWARE_DIR, ROOT, derive_seed
from .data import sample_syndromes
from .decoders import GreedyDecoder, MWPMDecoder, NeuralDecoder
from .evaluate import evaluate_decoders
from .noise import (DeviceCalibration, NoiseSpec, best_linear_chain, calibration_from_backend,
                    get_fake_backend)

STATUS_FINAL_OK = "DONE"


# ================================================================== credentials
def load_env() -> None:
    """Load ``.env`` (git-ignored) into the process environment, if python-dotenv exists."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return
    load_dotenv(ROOT / ".env", override=False)


def get_service():
    """Authenticated ``QiskitRuntimeService`` using ``PINQ2_TOKEN`` (plan section 8.3).

    ``PINQ2_INSTANCE`` (the CRN supplied by the organizers) and ``QISKIT_CHANNEL``
    (default ``ibm_cloud``) are optional.  The token is never printed or stored.
    """
    load_env()
    token = os.environ.get("PINQ2_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "PINQ2_TOKEN is not set.  Copy .env.example to .env, put your token there "
            "(the file is git-ignored), or export PINQ2_TOKEN in your shell.  "
            "Use --simulate to run offline against an IBM fake backend instead.")
    from qiskit_ibm_runtime import QiskitRuntimeService

    kwargs = dict(channel=os.environ.get("QISKIT_CHANNEL", "ibm_cloud").strip() or "ibm_cloud", token=token)
    instance = os.environ.get("PINQ2_INSTANCE", "").strip()
    if instance:
        kwargs["instance"] = instance
    return QiskitRuntimeService(**kwargs)


def fake_name_for(backend_name: str) -> str | None:
    """``ibm_quebec`` -> ``fake_quebec`` (offline noise model for the same device family)."""
    m = re.match(r"^(?:ibm_)?(\w+)$", backend_name)
    cand = f"fake_{m.group(1)}" if m else None
    try:
        get_fake_backend(cand) if cand else None
        return cand
    except Exception:
        return None


def resolve_backend(service, name: str | None, n_qubits: int):
    """Plan section 8.3: ``service.least_busy(operational=True, simulator=False)``, or a named backend."""
    if name:
        return service.backend(name)
    return service.least_busy(operational=True, simulator=False, min_num_qubits=max(n_qubits, 5))


def dynamic_circuit_report(backend) -> dict:
    """Plan ``[verify]``: does the backend support mid-circuit measurement and reset?"""
    ops = set(backend.target.operation_names)
    cfg_flag = None
    try:
        cfg_flag = bool(backend.configuration().supports_mid_circuit_measurement)
    except Exception:
        pass
    return dict(measure="measure" in ops, reset="reset" in ops, if_else="if_else" in ops,
                supports_mid_circuit_measurement_flag=cfg_flag,
                ok=("measure" in ops and "reset" in ops and cfg_flag is not False))


# ====================================================================== run files
@dataclass
class RunMeta:
    job_id: str
    backend: str
    simulated: bool
    d: int
    rounds: int
    shots: int
    logical_state: int
    chain: list[int]
    optimization_level: int
    status: str = "SUBMITTED"
    submitted_utc: str = ""
    completed_utc: str = ""
    note: str = ""
    isa_ops: dict = field(default_factory=dict)
    isa_depth: int = 0
    dynamic_circuits: dict = field(default_factory=dict)
    versions: dict = field(default_factory=dict)
    label: str = ""

    def save(self, run_dir: Path) -> None:
        (run_dir / "meta.json").write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, run_dir: Path) -> "RunMeta":
        return cls(**json.loads((Path(run_dir) / "meta.json").read_text()))


@dataclass
class Run:
    path: Path
    meta: RunMeta
    calibration: DeviceCalibration
    measurements: np.ndarray | None = None

    def syndromes(self) -> tuple[np.ndarray, np.ndarray]:
        if self.measurements is None:
            raise RuntimeError("run has no data yet -- fetch it first")
        return measurements_to_syndromes(self.measurements, self.meta.d, self.meta.rounds,
                                         self.meta.logical_state)


def _versions() -> dict:
    import qiskit
    import qiskit_aer
    import qiskit_ibm_runtime
    return dict(qiskit=qiskit.__version__, qiskit_aer=qiskit_aer.__version__,
                qiskit_ibm_runtime=qiskit_ibm_runtime.__version__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def prepare_circuit(d: int, rounds: int, backend, *, logical_state: int = 1,
                    chain: list[int] | None = None, optimization_level: int = 1):
    """Build + transpile the memory circuit onto a good chain.

    Returns ``(logical_circuit, isa_circuit, chain, calibration)``.  Data qubits sit on
    even chain positions, ancillas on odd ones, so every CX is between neighbours on the
    chip and the transpiler needs no SWAPs.
    """
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    chain = list(chain) if chain else best_linear_chain(backend, 2 * d - 1)
    if len(chain) != 2 * d - 1:
        raise ValueError(f"need a chain of {2 * d - 1} qubits for d={d}, got {len(chain)}")
    qc = build_qiskit_circuit(d, rounds, logical_state)
    layout = chain if rounds > 0 else chain[::2]
    pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend,
                                      initial_layout=layout, seed_transpiler=0)
    isa = pm.run(qc)
    return qc, isa, chain, calibration_from_backend(backend, chain)


def _default_sampler(backend, seed_simulator: int | None):
    from qiskit_ibm_runtime import SamplerV2

    if seed_simulator is not None:
        return SamplerV2(mode=backend, options={"simulator": {"seed_simulator": int(seed_simulator)}})
    return SamplerV2(mode=backend)


def submit(d: int, rounds: int, shots: int, *, backend, simulated: bool, logical_state: int = 1,
           optimization_level: int = 1, chain: list[int] | None = None, out_root: Path = HARDWARE_DIR,
           sampler_factory: Callable | None = None, seed_simulator: int | None = None,
           label: str = "", log: Callable[[str], None] = print) -> tuple[Path, object]:
    """Submit one job; the job id is written to disk **before anything else can fail**.

    Returns ``(run_dir, job)``.  ``sampler_factory(backend) -> sampler`` is injectable for
    tests; the default is ``qiskit_ibm_runtime.SamplerV2(mode=backend)`` as in plan 8.3.
    """
    dyn = dynamic_circuit_report(backend)
    if rounds > 0 and not dyn["ok"]:
        raise RuntimeError(
            f"backend {backend.name} does not report mid-circuit measure/reset support ({dyn}). "
            "Re-run with --rounds 0 for the final-readout-only (code-capacity) fallback.")
    qc, isa, chain, cal = prepare_circuit(d, rounds, backend, logical_state=logical_state, chain=chain,
                                          optimization_level=optimization_level)
    log(f"[submit] backend={backend.name} d={d} rounds={rounds} shots={shots} chain={chain}")
    log(f"[submit] {cal.summary()}")
    log(f"[submit] transpiled: depth={isa.depth()} ops={dict(isa.count_ops())}")

    sampler = (sampler_factory or (lambda b: _default_sampler(b, seed_simulator)))(backend)
    job = sampler.run([isa], shots=shots)
    job_id = str(job.job_id())
    log(f"[submit] JOB ID: {job_id}   <-- saved to disk now")

    run_dir = Path(out_root) / f"{backend.name}_d{d}_r{rounds}_{job_id[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = RunMeta(
        job_id=job_id, backend=backend.name, simulated=simulated, d=d, rounds=rounds, shots=shots,
        logical_state=logical_state, chain=chain, optimization_level=optimization_level,
        submitted_utc=_utcnow(), isa_ops={k: int(v) for k, v in isa.count_ops().items()},
        isa_depth=int(isa.depth()), dynamic_circuits=dyn, versions=_versions(), label=label,
        note=("SIMULATED: local Aer simulation with the noise model of IBM fake backend "
              f"'{backend.name}'. NOT data from a real quantum computer." if simulated else
              "Real hardware job (IBM Runtime / PINQ2)."))
    meta.save(run_dir)                                   # job id on disk immediately
    cal.to_json(run_dir / "calibration.json")
    try:
        from qiskit import qpy
        with open(run_dir / "circuit.qpy", "wb") as fh:
            qpy.dump(isa, fh)
    except Exception as exc:                            # pragma: no cover - cosmetic artifact
        log(f"[submit] could not write circuit.qpy: {exc}")
    return run_dir, job


def fetch(run_dir: Path, *, service=None, job=None, wait: bool = False, timeout: float | None = None,
          log: Callable[[str], None] = print) -> bool:
    """Retrieve a finished job and store its raw shot-by-shot record.  ``True`` when data was saved."""
    run_dir = Path(run_dir)
    meta = RunMeta.load(run_dir)
    if meta.status == STATUS_FINAL_OK and (run_dir / "raw_record.npz").exists():
        return True
    if job is None:
        if service is None:
            service = get_service()
        job = service.job(meta.job_id)
    if wait:
        if hasattr(job, "wait_for_final_state"):
            job.wait_for_final_state(timeout=timeout)
        else:                                  # local (fake-backend) jobs: result() blocks
            job.result()
    status = getattr(job.status(), "name", str(job.status())).upper()
    log(f"[fetch] job {meta.job_id}: {status}")
    if status != STATUS_FINAL_OK:
        meta.status = status
        meta.save(run_dir)
        return False
    pub = job.result()[0]
    meas = sampler_result_to_measurements(pub.data, meta.d, meta.rounds)
    np.savez_compressed(run_dir / "raw_record.npz", packed=np.packbits(meas, axis=1),
                        n_bits=np.int64(meas.shape[1]), shots=np.int64(meas.shape[0]))
    rows, cnt = np.unique(meas, axis=0, return_counts=True)
    counts = {"".join(map(str, r)): int(c) for r, c in zip(rows, cnt)}
    (run_dir / "counts.json").write_text(json.dumps(
        dict(bit_order="record order: round-0 ancillas ... last-round ancillas | data qubits", counts=counts)))
    meta.status, meta.completed_utc = STATUS_FINAL_OK, _utcnow()
    meta.save(run_dir)
    log(f"[fetch] saved {meas.shape[0]} shots x {meas.shape[1]} bits to {run_dir}")
    return True


def load_run(run_dir: Path | str) -> Run:
    run_dir = Path(run_dir)
    meta = RunMeta.load(run_dir)
    cal = DeviceCalibration.from_json(run_dir / "calibration.json")
    meas = None
    raw = run_dir / "raw_record.npz"
    if raw.exists():
        z = np.load(raw)
        meas = np.unpackbits(z["packed"], axis=1)[:, : int(z["n_bits"])]
    return Run(run_dir, meta, cal, meas)


def list_runs(root: Path = HARDWARE_DIR) -> list[Path]:
    return sorted(p.parent for p in Path(root).glob("*/meta.json"))


# ===================================================================== Aer + Stim
def run_aer(d: int, rounds: int, shots: int, backend, chain: list[int], *, logical_state: int = 1,
            seed: int = 0, optimization_level: int = 1, chunk: int = 5000,
            log: Callable[[str], None] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Detection events + observable flips from a noisy Aer simulation of the *transpiled*
    circuit with the backend's noise model (``AerSimulator.from_backend`` under the hood).
    Slow (~1 ms/shot); use Stim for bulk data."""
    _, isa, _, _ = prepare_circuit(d, rounds, backend, logical_state=logical_state, chain=chain,
                                   optimization_level=optimization_level)
    dets, obss, done, i = [], [], 0, 0
    while done < shots:
        n = min(chunk, shots - done)
        sampler = _default_sampler(backend, seed + i)
        pub = sampler.run([isa], shots=n).result()[0]
        det, obs = measurements_to_syndromes(sampler_result_to_measurements(pub.data, d, rounds), d, rounds,
                                             logical_state)
        dets.append(det)
        obss.append(obs)
        done += n
        i += 1
        if log:
            log(f"[aer] {done}/{shots} shots")
    return np.concatenate(dets), np.concatenate(obss)


def stim_device_circuit(cal: DeviceCalibration, d: int, rounds: int, logical_state: int = 1, scale: float = 1.0):
    """Approximate per-qubit Pauli (Stim) model of the device -- see :mod:`src.noise`."""
    spec = NoiseSpec.from_device(scale, device=cal.backend)
    return build_repetition_circuit(d, rounds, spec, calibration=cal, logical_state=logical_state)


def approx_duration(cal: DeviceCalibration, rounds: int) -> float:
    """Rough wall-clock duration of the memory circuit (for the unencoded reference line)."""
    return rounds * (2 * cal.t_2q + cal.t_meas + cal.t_reset) + cal.t_meas


# ===================================================================== analysis
def analyze_run(run_dir: Path | str, *, nn_steps: int = 3000, stim_shots: int = 200_000,
                aer_shots: int = 20_000, backend=None, retrain: bool = False,
                log: Callable[[str], None] = print) -> pd.DataFrame:
    """Decode a run with MWPM / neural / greedy and compare it with the simulations.

    Sources (columns ``source``):

    * ``IBM hardware`` (or ``Aer (this run)`` for a simulated run): the saved shots
    * ``Aer noise model``: fresh Aer shots with the backend noise model (real runs only;
      needs ``backend`` or an offline fake backend of the same family)
    * ``Stim approx.``: Pauli approximation built from the calibration snapshot

    Decoders are built from the *Stim approximation of the calibration snapshot* (MWPM from
    its detector error model, the network trained on it): that is all that is known about
    a real device a priori.  The result is written to ``analysis.csv``.
    """
    from .training import train_on_circuit

    run = load_run(run_dir)
    meta, cal = run.meta, run.calibration
    d, rounds, ls = meta.d, meta.rounds, meta.logical_state
    det_run, obs_run = run.syndromes()
    approx = stim_device_circuit(cal, d, rounds, ls)

    nn_path = run.path / "nn_device_approx.pt"
    if nn_path.exists() and not retrain:
        nn = NeuralDecoder.load(nn_path, name="NN")
    else:
        log(f"[analyze] training the neural decoder on the Stim approximation ({nn_steps} steps)")
        nn, _ = train_on_circuit(approx, d=d, steps=nn_steps, val_shots=50_000,
                                 seeds=dict(val=derive_seed("val", f"hw|{meta.job_id}"),
                                            train=derive_seed("train", f"hw|{meta.job_id}"),
                                            torch=derive_seed("torch", f"hw|{meta.job_id}")),
                                 meta=dict(trained_for_job=meta.job_id), log=None)
        nn.save(nn_path)
    decs = {"mwpm_matched": MWPMDecoder.from_circuit(approx), "nn": nn, "greedy": GreedyDecoder(d, rounds)}

    rows = []

    def add(source: str, det: np.ndarray, obs: np.ndarray) -> None:
        for r in evaluate_decoders(decs, det, obs, rounds=rounds, reference="mwpm_matched"):
            rows.append(dict(source=source, undecoded=float(obs.mean()), **r))

    first = "Aer (this run)" if meta.simulated else "IBM hardware"
    add(first, det_run, obs_run)
    if not meta.simulated:
        be = backend
        if be is None:
            fake = fake_name_for(meta.backend)
            be = get_fake_backend(fake) if fake else None
        if be is not None:
            log(f"[analyze] Aer reference with the noise model of {be.name} ({aer_shots} shots)")
            adet, aobs = run_aer(d, rounds, aer_shots, be, meta.chain, logical_state=ls,
                                 seed=derive_seed("hardware", meta.job_id))
            add("Aer noise model", adet, aobs)
        else:
            log("[analyze] no backend / fake backend available for an Aer reference -- skipped")
    sdet, sobs = sample_syndromes(approx, stim_shots, derive_seed("test", f"hw-stim|{meta.job_id}"))
    add("Stim approx.", sdet, sobs)

    df = pd.DataFrame(rows)
    df.insert(0, "job_id", meta.job_id)
    df.insert(1, "backend", meta.backend)
    df.insert(2, "simulated", meta.simulated)
    df.to_csv(run.path / "analysis.csv", index=False)
    return df


def plot_run(run_dir: Path | str, df: pd.DataFrame | None = None, path: Path | str | None = None):
    """Figure (3) of the evaluation protocol: hardware vs Aer prediction vs Stim approximation."""
    from . import plotting

    run = load_run(run_dir)
    df = df if df is not None else pd.read_csv(Path(run_dir) / "analysis.csv")
    meta = run.meta
    kind = "SIMULATED (Aer, fake backend)" if meta.simulated else "real hardware"
    bare = run.calibration.bare_qubit_error(approx_duration(run.calibration, meta.rounds))
    return plotting.plot_hardware(
        df, path or (Path(run_dir) / "fig_hardware_vs_sim.png"),
        title=f"E4 - {meta.backend}, d={meta.d}, {meta.rounds} rounds, {meta.shots} shots [{kind}]",
        bare_qubit=bare,
        subtitle=("Stim points use only the calibration snapshot (Pauli approximation); the line is a rough "
                  "unencoded-qubit error (T1 decay + readout) -- not a rigorous break-even claim."
                  + ("  No real hardware data in this figure." if meta.simulated else "")))
