"""Noise models (plan section 4) and IBM device calibration.

Three noise models are supported, all expressed as a frozen :class:`NoiseSpec`:

``uniform``  (control)
    Independent depolarizing noise ``p`` after every CX gate and on every data
    qubit each round, plus measurement flips ``p`` and reset flips ``p``.
    MWPM is expected to be near-optimal here, so the neural decoder should tie.

``biased``   (biased / correlated; the main result)
    * ``bias`` r : single-qubit Pauli channel on the data qubits with
      ``p_Z = r * p_X`` and ``p_X = p_Y = p / 3``.  ``r = 1`` is exactly depolarizing.
    * ``corr`` c : a *correlated two-qubit bit-flip channel*.  Every round, each pair
      of data qubits at separation 1 or 2 suffers a simultaneous X flip with
      probability ``c * p`` (Stim ``CORRELATED_ERROR``).  These events create
      correlated defects that violate the "independent edges" assumption of MWPM.

    Honest physics note: in a *repetition* code only X-type flips are visible, so at
    fixed ``p_X`` the logical error rate is independent of ``bias`` (Z errors are
    invisible to a Z-basis repetition code).  The bias knob matters for the surface
    code (stretch experiment E5, where Y errors flip both X- and Z-type checks);
    the correlated channel is the knob that matters for the repetition code.

``device``   (device-derived)
    Per-qubit / per-gate rates read from an IBM backend's calibration data
    (two-qubit gate error, readout error, T1/T2) turned into an *approximate*
    per-qubit Pauli noise model for Stim.  Stim only supports Pauli noise, so
    amplitude damping is Pauli-twirled and readout asymmetry is symmetrised.  This
    is an approximation (plan section 4, "Honesty note") and is never presented as
    exact device fidelity.  The full-fidelity route is Qiskit Aer
    (``NoiseModel.from_backend``) in :mod:`src.hardware`.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .common import CALIBRATION_DIR, DEFAULT_FAKE_BACKEND, fmt_p

KINDS = ("uniform", "biased", "device")


# =========================================================================== NoiseSpec
@dataclass(frozen=True)
class NoiseSpec:
    """Immutable description of a noise setting; hashable and JSON-serialisable."""

    kind: str = "uniform"
    p: float = 0.01          # physical error rate (uniform / biased)
    bias: float = 1.0        # r = p_Z / p_X          (biased)
    corr: float = 0.0        # correlated pair-flip probability in units of p (biased)
    scale: float = 1.0       # multiplier on calibrated error rates (device)
    device: str = DEFAULT_FAKE_BACKEND  # calibration source name (device)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if not 0.0 <= self.p <= 0.5:
            raise ValueError(f"p must lie in [0, 0.5], got {self.p}")
        if self.bias < 0 or self.corr < 0 or self.scale < 0:
            raise ValueError("bias, corr and scale must be non-negative")
        if self.kind == "biased":
            if self.p * (2.0 + self.bias) / 3.0 > 1.0:
                raise ValueError("p * (2 + bias) / 3 must be <= 1 (Pauli probabilities)")
            if self.corr * self.p > 0.5:
                raise ValueError("corr * p must be <= 0.5")

    # ------------------------------------------------------------ constructors
    @classmethod
    def uniform(cls, p: float) -> "NoiseSpec":
        return cls(kind="uniform", p=p)

    @classmethod
    def biased(cls, p: float, bias: float = 5.0, corr: float = 0.5) -> "NoiseSpec":
        return cls(kind="biased", p=p, bias=bias, corr=corr)

    @classmethod
    def from_device(cls, scale: float = 1.0, device: str = DEFAULT_FAKE_BACKEND) -> "NoiseSpec":
        return cls(kind="device", p=0.0, scale=scale, device=device)

    # --------------------------------------------------------------- identity
    @property
    def level(self) -> float:
        """The 1-D difficulty knob: ``p`` for uniform/biased, ``scale`` for device."""
        return self.scale if self.kind == "device" else self.p

    @property
    def key(self) -> str:
        """Filesystem-safe unique identifier."""
        if self.kind == "uniform":
            return f"uniform_p{fmt_p(self.p)}"
        if self.kind == "biased":
            return f"biased_p{fmt_p(self.p)}_r{self.bias:g}_c{self.corr:g}"
        return f"device_{self.device}_x{self.scale:g}"

    @property
    def label(self) -> str:
        if self.kind == "uniform":
            return f"uniform p={self.p:g}"
        if self.kind == "biased":
            return f"biased p={self.p:g}, r={self.bias:g}, corr={self.corr:g}"
        return f"device {self.device} x{self.scale:g}"

    def with_p(self, p: float) -> "NoiseSpec":
        return NoiseSpec(**{**asdict(self), "p": p})

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "NoiseSpec":
        return cls(**{k: d[k] for k in ("kind", "p", "bias", "corr", "scale", "device") if k in d})


# ============================================================ twirled idle noise
def idle_pauli_probs(t: float, t1: float, t2: float) -> tuple[float, float, float]:
    """Pauli-twirled amplitude + phase damping over an idle time ``t`` (seconds).

    Returns ``(p_x, p_y, p_z)`` with ``p_x = p_y = (1 - e^{-t/T1}) / 4`` and
    ``p_z = (1 - e^{-t/T2}) / 2 - p_x`` (Ghosh et al. 2012).  Only the X/Y part is
    visible to a Z-basis repetition code.
    """
    if t <= 0:
        return 0.0, 0.0, 0.0
    t2 = min(t2, 2.0 * t1)  # physical constraint T2 <= 2 T1
    px = (1.0 - math.exp(-t / t1)) / 4.0
    pz = max(0.0, (1.0 - math.exp(-t / t2)) / 2.0 - px)
    return px, px, pz


# ============================================================ device calibration
@dataclass
class DeviceCalibration:
    """Calibration snapshot for a linear chain of ``len(chain)`` physical qubits.

    Chain position ``k`` plays the role of repetition-code qubit ``k`` (even ``k`` =
    data qubit, odd ``k`` = syndrome ancilla), so ``cx_error[k]`` is the native
    two-qubit error of the physical edge ``(chain[k], chain[k+1])``.
    """

    backend: str
    chain: list[int]
    t1: list[float]               # seconds, per chain qubit
    t2: list[float]               # seconds
    readout_error: list[float]    # assignment error, per chain qubit
    cx_error: list[float]         # native 2q gate error, per adjacent chain pair
    sx_error: list[float]         # single-qubit gate error
    t_2q: float = 6.0e-7          # native two-qubit gate duration (s)
    t_meas: float = 8.4e-7        # measurement duration (s)
    t_reset: float = 9.0e-7       # reset duration (s)
    reset_error_factor: float = 1.0   # reset flip prob = factor * readout error (approx.)
    gate_name: str = "ecr"
    captured: str = ""
    source: str = ""
    extra: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- helpers
    @property
    def n_qubits(self) -> int:
        return len(self.chain)

    def prefix(self, n: int) -> "DeviceCalibration":
        """Calibration restricted to the first ``n`` chain qubits (a shorter chain)."""
        if n > self.n_qubits:
            raise ValueError(f"chain has only {self.n_qubits} qubits, asked for {n}")
        return DeviceCalibration(
            backend=self.backend, chain=self.chain[:n], t1=self.t1[:n], t2=self.t2[:n],
            readout_error=self.readout_error[:n], cx_error=self.cx_error[: max(n - 1, 0)],
            sx_error=self.sx_error[:n], t_2q=self.t_2q, t_meas=self.t_meas,
            t_reset=self.t_reset, reset_error_factor=self.reset_error_factor,
            gate_name=self.gate_name, captured=self.captured, source=self.source,
            extra=dict(self.extra),
        )

    def idle_probs(self, k: int, scale: float = 1.0) -> tuple[float, float, float]:
        """Twirled idle Pauli probabilities of chain qubit ``k`` while its neighbours
        are being measured and reset (one syndrome-extraction window)."""
        t = self.t_meas + self.t_reset
        px, py, pz = idle_pauli_probs(t, self.t1[k], self.t2[k])
        return min(px * scale, 0.25), min(py * scale, 0.25), min(pz * scale, 0.5)

    def bare_qubit_error(self, duration: float) -> float:
        """Rough error of an *unencoded* qubit prepared in |1>, idled for ``duration``
        and measured: T1 decay + readout error, averaged over the chain.  Used only as
        a dashed reference line in the hardware figure (not a rigorous break-even)."""
        errs = []
        for t1, ro in zip(self.t1, self.readout_error):
            decay = 1.0 - math.exp(-duration / t1)
            errs.append(decay * (1 - ro) + (1 - decay) * ro)
        return float(sum(errs) / len(errs))

    # --------------------------------------------------------------- JSON I/O
    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "DeviceCalibration":
        return cls(**json.loads(Path(path).read_text()))

    def summary(self) -> str:
        med = lambda v: sorted(v)[len(v) // 2]
        return (f"{self.backend}: chain={self.chain} median 2q err={med(self.cx_error):.4f} "
                f"median readout err={med(self.readout_error):.4f} "
                f"median T1={med(self.t1) * 1e6:.0f}us T2={med(self.t2) * 1e6:.0f}us")


def get_fake_backend(name: str = DEFAULT_FAKE_BACKEND):
    """Instantiate an offline IBM fake backend by snake_case name (``fake_quebec``)."""
    import qiskit_ibm_runtime.fake_provider as fp

    cls_name = "".join(part.capitalize() for part in name.split("_"))
    cls = getattr(fp, cls_name, None)
    if cls is None:
        avail = sorted(n for n in dir(fp) if n.startswith("Fake") and "V2" not in n)
        raise ValueError(f"unknown fake backend {name!r} ({cls_name}); available: {avail}")
    return cls()


# ---- reading properties out of a BackendV2 target (robust to missing entries) ------
_DEFAULTS = dict(t1=100e-6, t2=80e-6, readout=0.02, e2=0.02, e1=5e-4,
                 t_2q=6.0e-7, t_meas=1.0e-6, t_reset=1.0e-6)


def _two_qubit_gate(target) -> str:
    for name in ("ecr", "cz", "cx"):
        if name in target.operation_names:
            return name
    raise ValueError("backend has no native two-qubit gate among ecr / cz / cx")


def _edge_props(target, gate: str, a: int, b: int):
    props = target[gate]
    for key in ((a, b), (b, a)):
        if key in props:
            return props[key]
    return None


def _coupling_adjacency(backend) -> dict[int, set[int]]:
    adj: dict[int, set[int]] = {q: set() for q in range(backend.num_qubits)}
    for a, b in backend.coupling_map.get_edges():
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _qubit_stats(backend, q: int) -> dict:
    t = backend.target
    qp = t.qubit_properties[q] if t.qubit_properties else None
    t1 = getattr(qp, "t1", None) or _DEFAULTS["t1"]
    t2 = getattr(qp, "t2", None) or _DEFAULTS["t2"]
    m = t["measure"].get((q,)) if "measure" in t.operation_names else None
    ro = getattr(m, "error", None)
    t_meas = getattr(m, "duration", None)
    r = t["reset"].get((q,)) if "reset" in t.operation_names else None
    t_reset = getattr(r, "duration", None)
    sx = t["sx"].get((q,)) if "sx" in t.operation_names else None
    return dict(
        t1=t1, t2=t2,
        ro=_DEFAULTS["readout"] if ro is None else ro,
        t_meas=t_meas or _DEFAULTS["t_meas"], t_reset=t_reset or _DEFAULTS["t_reset"],
        e1=getattr(sx, "error", None) or _DEFAULTS["e1"],
    )


def _edge_error(backend, gate: str, a: int, b: int) -> tuple[float, float]:
    ip = _edge_props(backend.target, gate, a, b)
    err = getattr(ip, "error", None)
    dur = getattr(ip, "duration", None)
    return (_DEFAULTS["e2"] if err is None else err), (dur or _DEFAULTS["t_2q"])


def best_linear_chain(backend, n: int, beam: int = 400, ancilla_weight: float = 2.0) -> list[int]:
    """Choose a well-calibrated path of ``n`` physical qubits (plan E4: "Choose a
    linear chain of qubits with good calibration").

    Beam search over simple paths in the coupling graph minimising
    ``sum(2q gate errors) + sum(role-weighted readout / idle errors)``.  Odd positions
    are syndrome ancillas (measured and reset every round) and are weighted by
    ``ancilla_weight``.  Deterministic.
    """
    if n < 1 or n > backend.num_qubits:
        raise ValueError("invalid chain length")
    gate = _two_qubit_gate(backend.target)
    adj = _coupling_adjacency(backend)
    stats = {q: _qubit_stats(backend, q) for q in adj}

    def node_cost(q: int, pos: int) -> float:
        s = stats[q]
        idle = (s["t_meas"] + s["t_reset"]) / s["t1"] / 2.0
        w = ancilla_weight if pos % 2 == 1 else 1.0
        return w * s["ro"] + (0.0 if pos % 2 == 1 else idle)

    edge_cache: dict[tuple[int, int], float] = {}

    def edge_cost(a: int, b: int) -> float:
        key = (min(a, b), max(a, b))
        if key not in edge_cache:
            edge_cache[key] = _edge_error(backend, gate, a, b)[0]
        return edge_cache[key]

    frontier = [(node_cost(q, 0), (q,)) for q in sorted(adj)]
    frontier.sort(key=lambda x: (x[0], x[1]))
    frontier = frontier[:beam]
    for pos in range(1, n):
        grown = []
        for cost, path in frontier:
            for nb in sorted(adj[path[-1]]):
                if nb in path:
                    continue
                grown.append((cost + edge_cost(path[-1], nb) + node_cost(nb, pos), path + (nb,)))
        if not grown:
            raise RuntimeError(f"no simple path of length {n} found in the coupling map")
        grown.sort(key=lambda x: (x[0], x[1]))
        frontier = grown[:beam]
    return list(frontier[0][1])


def calibration_from_backend(backend, chain: Iterable[int], source: str = "") -> DeviceCalibration:
    """Snapshot the calibration of ``chain`` (physical qubit indices, in chain order)."""
    chain = list(chain)
    gate = _two_qubit_gate(backend.target)
    stats = [_qubit_stats(backend, q) for q in chain]
    cx_err, durs = [], []
    for a, b in zip(chain[:-1], chain[1:]):
        e, dur = _edge_error(backend, gate, a, b)
        cx_err.append(float(e))
        durs.append(float(dur))
    med = lambda v: sorted(v)[len(v) // 2]
    return DeviceCalibration(
        backend=backend.name,
        chain=chain,
        t1=[float(s["t1"]) for s in stats],
        t2=[float(s["t2"]) for s in stats],
        readout_error=[float(s["ro"]) for s in stats],
        cx_error=cx_err,
        sx_error=[float(s["e1"]) for s in stats],
        t_2q=float(med(durs)) if durs else _DEFAULTS["t_2q"],
        t_meas=float(med([s["t_meas"] for s in stats])),
        t_reset=float(med([s["t_reset"] for s in stats])),
        gate_name=gate,
        captured=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source=source or f"{backend.name} (target properties)",
    )


def calibration_path(device: str, d: int) -> Path:
    return CALIBRATION_DIR / f"{device}_d{d}.json"


def load_device_calibration(device: str, d: int, *, refresh: bool = False) -> DeviceCalibration:
    """Calibration for a distance-``d`` repetition code (``2d - 1`` chain qubits).

    Looks for the committed snapshot ``data/calibration/<device>_d<d>.json`` first so
    that results are reproducible even if a newer qiskit-ibm-runtime ships different
    fake-backend data; otherwise (or with ``refresh``) it builds one from the offline
    fake backend and caches it.  Calibrations saved by :mod:`src.hardware` for a real
    run are loaded with :func:`load_calibration_file` and passed to the circuit
    builder explicitly.
    """
    path = calibration_path(device, d)
    if path.exists() and not refresh:
        return DeviceCalibration.from_json(path)
    backend = get_fake_backend(device)
    chain = best_linear_chain(backend, 2 * d - 1)
    cal = calibration_from_backend(
        backend, chain, source=f"offline fake backend '{device}' (qiskit-ibm-runtime snapshot)")
    cal.to_json(path)
    return cal


def load_calibration_file(path: str | Path, n_qubits: int) -> DeviceCalibration:
    cal = DeviceCalibration.from_json(path)
    return cal if cal.n_qubits == n_qubits else cal.prefix(n_qubits)
