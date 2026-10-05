"""Circuit builders: Stim (bulk data) and Qiskit (Aer + IBM hardware).

Repetition-code memory experiment (``d`` data qubits, ``d - 1`` syndrome ancillas)
===================================================================================
Qubit layout on a line (identical in Stim and Qiskit, and identical to
``stim.Circuit.generated("repetition_code:memory")``)::

    data  : 0   2   4   ...   2(d-1)        (qubit 2i     = data qubit i)
    ancilla:  1   3   ...   2d-3            (qubit 2j + 1 = ancilla j, checks Z_j Z_{j+1})

One round = two CX layers (data_j -> ancilla_j, then data_{j+1} -> ancilla_j), then
measure + reset the ancillas.  After ``rounds`` rounds all data qubits are measured.

Measurement record order (both Stim and Qiskit)::

    [ round 0 ancillas (d-1 bits) ... round R-1 ancillas | data qubits (d bits) ]

Detectors (parity checks on the record that are deterministic without noise)::

    D(0, j)   = a(0, j)
    D(t, j)   = a(t, j) xor a(t-1, j)               t = 1 .. R-1
    D(R, j)   = a(R-1, j) xor m(data_j) xor m(data_{j+1})      (final layer)

so there are ``(d - 1) * (R + 1)`` detectors, ordered round-major.  Note: the plan
quotes ``rounds * (d - 1)``; the extra layer comes from the final data readout.
The logical observable is the final measurement of the *last* data qubit (Stim's
convention); the decoder's job is to predict whether it was flipped.

``rounds == 0`` builds the *code-capacity / final-readout-only* variant (no ancillas,
no mid-circuit measurement): the plan's fallback when dynamic circuits are not
available on the chosen backend.
"""

from __future__ import annotations

import re

import numpy as np
import stim

from .noise import DeviceCalibration, NoiseSpec

# ============================================================== layout helpers


def num_detectors(d: int, rounds: int) -> int:
    return (d - 1) * (rounds + 1) if rounds > 0 else (d - 1)


def detector_layout(d: int, rounds: int) -> tuple[int, int]:
    """``(layers, ancillas)`` such that detector index = layer * (d - 1) + ancilla."""
    return (rounds + 1 if rounds > 0 else 1), d - 1


def data_qubit(i: int, rounds: int) -> int:
    return 2 * i if rounds > 0 else i


def ancilla_qubit(j: int) -> int:
    return 2 * j + 1


def num_qubits(d: int, rounds: int) -> int:
    return 2 * d - 1 if rounds > 0 else d


# ================================================================== Stim builder


def _fmt(p: float) -> str:
    return repr(float(p))


def _targets(qs) -> str:
    return " ".join(str(q) for q in qs)


def correlated_pairs(d: int, rounds: int, max_sep: int = 2) -> list[tuple[int, int]]:
    """Pairs of data qubits (in Stim qubit indices) that suffer correlated flips."""
    pairs = []
    for sep in range(1, max_sep + 1):
        for i in range(d - sep):
            pairs.append((data_qubit(i, rounds), data_qubit(i + sep, rounds)))
    return pairs


def build_repetition_circuit(
    d: int,
    rounds: int,
    spec: NoiseSpec,
    *,
    calibration: DeviceCalibration | None = None,
    logical_state: int = 0,
) -> stim.Circuit:
    """Stim repetition-code memory circuit under the given noise specification.

    ``calibration`` is required for ``spec.kind == "device"`` (it must describe at
    least ``2d - 1`` chain qubits; only the first ``2d - 1`` are used).
    """
    if d < 2:
        raise ValueError("distance must be >= 2")
    if rounds < 0:
        raise ValueError("rounds must be >= 0")
    if logical_state not in (0, 1):
        raise ValueError("logical_state must be 0 or 1")
    if spec.kind == "device":
        if calibration is None:
            raise ValueError("device noise requires a DeviceCalibration")
        if calibration.n_qubits < 2 * d - 1:
            raise ValueError(f"calibration has {calibration.n_qubits} qubits, need {2 * d - 1}")
    if rounds == 0:
        return _build_code_capacity(d, spec, calibration, logical_state)

    nq = 2 * d - 1
    data = [2 * i for i in range(d)]
    anc = [2 * j + 1 for j in range(d - 1)]
    n = d - 1
    lines: list[str] = []
    dev = spec.kind == "device"
    s = spec.scale

    # -- noise rates --------------------------------------------------------
    if dev:
        cal = calibration
        cx_err = [min(0.75, cal.cx_error[k] * s) for k in range(nq - 1)]   # edge k = (k, k+1)
        ro = [min(0.5, cal.readout_error[k] * s) for k in range(nq)]
        rst = [min(0.5, cal.readout_error[k] * cal.reset_error_factor * s) for k in range(nq)]
    else:
        p = spec.p

    # -- reset ---------------------------------------------------------------
    lines.append(f"R {_targets(range(nq))}")
    if not dev and p > 0:
        lines.append(f"X_ERROR({_fmt(p)}) {_targets(range(nq))}")   # == after_reset_flip_probability
    if logical_state == 1:
        lines.append(f"X {_targets(data)}")
    lines.append("TICK")

    for t in range(rounds):
        # -- per-round data noise (uniform / biased) ------------------------
        if not dev:
            if spec.kind == "uniform":
                if p > 0:
                    lines.append(f"DEPOLARIZE1({_fmt(p)}) {_targets(data)}")
            else:  # biased / correlated
                px = p / 3.0
                pz = spec.bias * p / 3.0
                if p > 0:
                    lines.append(f"PAULI_CHANNEL_1({_fmt(px)}, {_fmt(px)}, {_fmt(pz)}) {_targets(data)}")
                q = spec.corr * p
                if q > 0:
                    for a, b in correlated_pairs(d, rounds):
                        lines.append(f"CORRELATED_ERROR({_fmt(q)}) X{a} X{b}")
        # -- CX layer 1: data_j -> ancilla_j -------------------------------
        for layer in (0, 1):
            pairs = [(2 * j + layer * 2, 2 * j + 1) for j in range(n)]
            lines.append("CX " + " ".join(f"{a} {b}" for a, b in pairs))
            if dev:
                for a, b in pairs:
                    e = cx_err[min(a, b)]
                    if e > 0:
                        lines.append(f"DEPOLARIZE2({_fmt(e)}) {a} {b}")
            elif p > 0:
                lines.append(f"DEPOLARIZE2({_fmt(p)}) " + " ".join(f"{a} {b}" for a, b in pairs))
            lines.append("TICK")
        # -- device: data qubits idle while ancillas are measured + reset ----
        if dev:
            for i, q in enumerate(data):
                px, py, pz = cal.idle_probs(q, s)
                if px + py + pz > 0:
                    lines.append(f"PAULI_CHANNEL_1({_fmt(px)}, {_fmt(py)}, {_fmt(pz)}) {q}")
        # -- measure + reset ancillas ---------------------------------------
        if dev:
            for q in anc:
                if ro[q] > 0:
                    lines.append(f"X_ERROR({_fmt(ro[q])}) {q}")
        elif p > 0:
            lines.append(f"X_ERROR({_fmt(p)}) {_targets(anc)}")        # before_measure_flip
        lines.append(f"MR {_targets(anc)}")
        if dev:
            for q in anc:
                if rst[q] > 0:
                    lines.append(f"X_ERROR({_fmt(rst[q])}) {q}")
        elif p > 0:
            lines.append(f"X_ERROR({_fmt(p)}) {_targets(anc)}")        # after_reset_flip
        # -- detectors -------------------------------------------------------
        for j in range(n):
            recs = [f"rec[{-n + j}]"] + ([f"rec[{-2 * n + j}]"] if t > 0 else [])
            lines.append(f"DETECTOR({anc[j]}, {t}) " + " ".join(recs))
        if t < rounds - 1:
            lines.append("TICK")

    # -- final data readout -------------------------------------------------------
    if dev:
        for q in data:
            if ro[q] > 0:
                lines.append(f"X_ERROR({_fmt(ro[q])}) {q}")
    elif p > 0:
        lines.append(f"X_ERROR({_fmt(p)}) {_targets(data)}")
    lines.append(f"M {_targets(data)}")
    for j in range(n):
        lines.append(
            f"DETECTOR({anc[j]}, {rounds}) rec[{-d - n + j}] rec[{-d + j}] rec[{-d + j + 1}]")
    lines.append(f"OBSERVABLE_INCLUDE(0) rec[{-1}]")
    return stim.Circuit("\n".join(lines))


def _build_code_capacity(d: int, spec: NoiseSpec, cal: DeviceCalibration | None,
                         logical_state: int) -> stim.Circuit:
    """Final-readout-only experiment: data noise + readout flips, no ancillas."""
    data = list(range(d))
    lines = [f"R {_targets(data)}"]
    if logical_state == 1:
        lines.append(f"X {_targets(data)}")
    lines.append("TICK")
    if spec.kind == "device":
        ro = [min(0.5, cal.readout_error[2 * i] * spec.scale) for i in range(d)]
        for i in data:
            px, py, pz = cal.idle_probs(2 * i, spec.scale)
            if px + py + pz > 0:
                lines.append(f"PAULI_CHANNEL_1({_fmt(px)}, {_fmt(py)}, {_fmt(pz)}) {i}")
        for i in data:
            if ro[i] > 0:
                lines.append(f"X_ERROR({_fmt(ro[i])}) {i}")
    else:
        p = spec.p
        if spec.kind == "uniform":
            if p > 0:
                lines.append(f"DEPOLARIZE1({_fmt(p)}) {_targets(data)}")
        else:
            if p > 0:
                lines.append(f"PAULI_CHANNEL_1({_fmt(p / 3)}, {_fmt(p / 3)}, {_fmt(spec.bias * p / 3)}) "
                             f"{_targets(data)}")
            q = spec.corr * p
            if q > 0:
                for a, b in correlated_pairs(d, 0):
                    lines.append(f"CORRELATED_ERROR({_fmt(q)}) X{a} X{b}")
        if p > 0:
            lines.append(f"X_ERROR({_fmt(p)}) {_targets(data)}")
    lines.append(f"M {_targets(data)}")
    for j in range(d - 1):
        lines.append(f"DETECTOR({2 * j + 1}, 0) rec[{-d + j}] rec[{-d + j + 1}]")
    lines.append("OBSERVABLE_INCLUDE(0) rec[-1]")
    return stim.Circuit("\n".join(lines))


# ============================================================ surface code (E5)
_DEP1 = re.compile(r"^(\s*)DEPOLARIZE1\(([^)]*)\)\s+(.*)$")


def build_surface_circuit(d: int, rounds: int, spec: NoiseSpec) -> stim.Circuit:
    """Rotated surface code Z-memory (stretch experiment E5) via Stim's generator.

    ``uniform``: standard circuit-level depolarizing noise with strength ``p``.
    ``biased`` : the same, but the per-round data-qubit depolarizing channel is
    replaced by ``PAULI_CHANNEL_1(p/3, p/3, r p/3)`` (``p_Z = r p_X``).  Here the bias
    *is* visible: Y errors flip both X- and Z-type checks, creating correlated
    defects that matching treats as independent.
    """
    if spec.kind == "device":
        raise ValueError("the surface-code experiment supports uniform and biased noise only")
    if spec.kind == "biased" and spec.corr > 0:
        raise ValueError("correlated pair flips are only implemented for the repetition code")
    p = spec.p
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=rounds,
        after_clifford_depolarization=p, before_round_data_depolarization=p,
        before_measure_flip_probability=p, after_reset_flip_probability=p)
    if spec.kind == "uniform" or spec.bias == 1.0:
        return circ
    # data qubits = targets of the final data-measurement instruction(s)
    data = _final_measured_qubits(circ)
    out, replaced = [], 0
    for line in str(circ).splitlines():
        m = _DEP1.match(line)
        if m and {int(x) for x in m.group(3).split()} == data:
            indent, _, tgt = m.groups()
            out.append(f"{indent}PAULI_CHANNEL_1({_fmt(p / 3)}, {_fmt(p / 3)}, "
                       f"{_fmt(spec.bias * p / 3)}) {tgt}")
            replaced += 1
        else:
            out.append(line)
    if replaced == 0 and p > 0:
        raise RuntimeError("could not locate the data-qubit depolarizing channel to bias")
    return stim.Circuit("\n".join(out))


def _final_measured_qubits(circ: stim.Circuit) -> set[int]:
    last = None
    for inst in circ.flattened():
        if inst.name in ("M", "MZ"):
            last = {t.value for t in inst.targets_copy()}
    if last is None:
        raise RuntimeError("circuit has no final data measurement")
    return last


# ======================================================= Qiskit circuit (hardware)


def build_qiskit_circuit(d: int, rounds: int, logical_state: int = 1):
    """Qiskit version of :func:`build_repetition_circuit` (mid-circuit measure + reset).

    One classical register per round (``anc0`` .. ``anc{R-1}``, ``d - 1`` bits, bit j =
    ancilla j) plus a ``data`` register (``d`` bits, bit i = data qubit i), so a
    ``SamplerV2`` result exposes the record in exactly the order assumed by
    :func:`measurements_to_syndromes`.  ``logical_state = 1`` prepares logical |1...1>,
    the harder state under T1 decay (|0...0> is the relaxation fixed point).
    """
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    if rounds < 0:
        raise ValueError("rounds must be >= 0")
    nq = num_qubits(d, rounds)
    qc = QuantumCircuit(QuantumRegister(nq, "q"), name=f"repcode_d{d}_r{rounds}_L{logical_state}")
    anc_regs = [ClassicalRegister(d - 1, f"anc{t}") for t in range(rounds)]
    data_reg = ClassicalRegister(d, "data")
    for r in anc_regs:
        qc.add_register(r)
    qc.add_register(data_reg)
    dq = [data_qubit(i, rounds) for i in range(d)]
    if logical_state == 1:
        for q in dq:
            qc.x(q)
        qc.barrier()
    for t in range(rounds):
        for layer in (0, 1):
            for j in range(d - 1):
                qc.cx(2 * j + 2 * layer, 2 * j + 1)
            qc.barrier()
        for j in range(d - 1):
            qc.measure(2 * j + 1, anc_regs[t][j])
        if t < rounds - 1:                      # no reset needed after the final round
            for j in range(d - 1):
                qc.reset(2 * j + 1)
        qc.barrier()
    for i in range(d):
        qc.measure(dq[i], data_reg[i])
    return qc


def bitarray_to_matrix(ba) -> np.ndarray:
    """``qiskit.primitives.BitArray`` -> uint8 matrix ``(shots, num_bits)`` where column
    j is classical bit j of that register (Qiskit stores bit 0 as the least
    significant bit of the last byte)."""
    arr = np.asarray(ba.array, dtype=np.uint8)
    bits = np.unpackbits(arr, axis=-1)            # big-endian within each byte
    return bits[:, bits.shape[1] - ba.num_bits:][:, ::-1].copy()


def sampler_result_to_measurements(data_bin, d: int, rounds: int) -> np.ndarray:
    """Assemble the measurement record from a ``SamplerV2`` ``PubResult.data`` bin."""
    cols = [bitarray_to_matrix(getattr(data_bin, f"anc{t}")) for t in range(rounds)]
    cols.append(bitarray_to_matrix(data_bin.data))
    return np.concatenate(cols, axis=1)


def measurements_to_syndromes(meas: np.ndarray, d: int, rounds: int,
                              logical_state: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Measurement record -> ``(detection events, observable flips)``, uint8.

    Pure-numpy re-implementation of the detector definitions above; unit-tested
    against Stim's own ``compile_m2d_converter``.
    """
    meas = np.asarray(meas, dtype=np.uint8)
    n = d - 1
    shots = meas.shape[0]
    if rounds == 0:
        data = meas[:, :d]
        det = data[:, :-1] ^ data[:, 1:]
    else:
        if meas.shape[1] != rounds * n + d:
            raise ValueError(f"expected {rounds * n + d} measurement bits, got {meas.shape[1]}")
        anc = meas[:, : rounds * n].reshape(shots, rounds, n)
        data = meas[:, rounds * n:]
        layers = np.empty((shots, rounds + 1, n), dtype=np.uint8)
        layers[:, 0] = anc[:, 0]
        layers[:, 1:rounds] = anc[:, 1:] ^ anc[:, :-1]
        layers[:, rounds] = anc[:, rounds - 1] ^ data[:, :-1] ^ data[:, 1:]
        det = layers.reshape(shots, -1)
    obs = data[:, d - 1] ^ np.uint8(logical_state)
    return np.ascontiguousarray(det), np.ascontiguousarray(obs)


# ============================================ geometry helpers shared by greedy/game


def final_syndrome(det: np.ndarray, d: int, rounds: int) -> np.ndarray:
    """XOR of every detector layer per ancilla column == parity pattern of the final
    data readout (telescoping sum).  ``det`` has shape ``(shots, n_det)`` or ``(n_det,)``."""
    layers, n = detector_layout(d, rounds)
    arr = np.asarray(det, dtype=np.uint8)
    arr = arr.reshape(*arr.shape[:-1], layers, n)
    return np.bitwise_xor.reduce(arr, axis=-2)


def error_estimate(s_final: np.ndarray, obs_flip: np.ndarray | int, d: int) -> np.ndarray:
    """Data-qubit flip pattern ``e`` (length ``d``) with ``e_i xor e_{i+1} = s_i`` and
    ``e_{d-1} = obs_flip`` (the observable is the last data qubit)."""
    s = np.asarray(s_final, dtype=np.uint8)
    obs = np.asarray(obs_flip, dtype=np.uint8)
    e = np.zeros(s.shape[:-1] + (d,), dtype=np.uint8)
    e[..., d - 1] = obs
    for i in range(d - 2, -1, -1):
        e[..., i] = e[..., i + 1] ^ s[..., i]
    return e
