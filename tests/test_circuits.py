"""Circuit builders: equivalence with Stim's generator, layouts, detector conversion."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest
import stim

from src.circuits import (bitarray_to_matrix, build_qiskit_circuit, build_repetition_circuit,
                          build_surface_circuit, detector_layout, error_estimate, final_syndrome,
                          measurements_to_syndromes, num_detectors)
from src.noise import NoiseSpec, load_device_calibration


def _merged_dem(circuit: stim.Circuit) -> dict:
    """Undecomposed DEM with duplicate mechanisms merged (p (+) q = p + q - 2pq)."""
    out = defaultdict(float)
    for inst in circuit.detector_error_model(decompose_errors=False).flattened():
        if inst.type == "error":
            key = tuple(sorted(str(t) for t in inst.targets_copy()))
            p, q = inst.args_copy()[0], out[key]
            out[key] = p + q - 2 * p * q
    return out


@pytest.mark.parametrize("d,rounds,p", [(3, 3, 0.01), (5, 4, 0.003), (7, 7, 0.02), (3, 1, 0.05)])
def test_uniform_builder_matches_stim_generated(d, rounds, p):
    """My builder (needed for the Qiskit layout + per-qubit noise) == stim.Circuit.generated."""
    mine = build_repetition_circuit(d, rounds, NoiseSpec.uniform(p))
    gen = stim.Circuit.generated(
        "repetition_code:memory", distance=d, rounds=rounds, before_round_data_depolarization=p,
        before_measure_flip_probability=p, after_reset_flip_probability=p, after_clifford_depolarization=p)
    a, b = _merged_dem(mine), _merged_dem(gen)
    assert set(a) == set(b)
    assert max(abs(a[k] - b[k]) for k in a) < 1e-12
    assert mine.num_detectors == gen.num_detectors == num_detectors(d, rounds)
    assert mine.num_observables == 1


@pytest.mark.parametrize("d,rounds,logical", [(3, 3, 0), (5, 2, 1), (3, 1, 1), (7, 5, 0), (3, 0, 0), (5, 0, 1)])
def test_numpy_detector_conversion_matches_stim_m2d(d, rounds, logical):
    c = build_repetition_circuit(d, rounds, NoiseSpec.uniform(0.05), logical_state=logical)
    meas = c.compile_sampler(seed=1).sample(2000).astype(np.uint8)
    det_s, obs_s = c.compile_m2d_converter().convert(measurements=meas.astype(bool), separate_observables=True)
    det_n, obs_n = measurements_to_syndromes(meas, d, rounds, logical)
    assert np.array_equal(det_s.astype(np.uint8), det_n)
    assert np.array_equal(obs_s[:, 0].astype(np.uint8), obs_n)


def test_noiseless_circuit_has_no_events():
    for rounds in (0, 1, 4):
        c = build_repetition_circuit(5, rounds, NoiseSpec.uniform(0.0), logical_state=1)
        det, obs = c.compile_detector_sampler(seed=0).sample(200, separate_observables=True)
        assert not det.any() and not obs.any()


def test_final_syndrome_and_error_estimate_recover_data_flips():
    d, rounds = 5, 3
    c = build_repetition_circuit(d, rounds, NoiseSpec.uniform(0.05))
    meas = c.compile_sampler(seed=2).sample(3000).astype(np.uint8)
    det, obs = measurements_to_syndromes(meas, d, rounds, 0)
    data = meas[:, -d:]
    s = final_syndrome(det, d, rounds)
    assert np.array_equal(s, data[:, :-1] ^ data[:, 1:])
    assert np.array_equal(error_estimate(s, obs, d), data)


def test_detector_layout():
    assert detector_layout(5, 4) == (5, 4)
    assert detector_layout(5, 0) == (1, 4)
    assert num_detectors(7, 7) == 48


def test_biased_noise_is_blind_to_z_bias_for_repetition_code():
    """Z errors are invisible to a Z-basis repetition code: the DEM must not depend on r."""
    base = _merged_dem(build_repetition_circuit(5, 5, NoiseSpec.biased(0.01, bias=1.0, corr=0.5)))
    for r in (5.0, 20.0):
        other = _merged_dem(build_repetition_circuit(5, 5, NoiseSpec.biased(0.01, bias=r, corr=0.5)))
        assert set(base) == set(other)
        assert max(abs(base[k] - other[k]) for k in base) < 1e-12


def test_correlated_noise_adds_mechanisms_and_decomposes_for_matching():
    plain = build_repetition_circuit(5, 5, NoiseSpec.biased(0.01, 5.0, 0.0))
    corr = build_repetition_circuit(5, 5, NoiseSpec.biased(0.01, 5.0, 0.5))
    assert len(_merged_dem(corr)) > len(_merged_dem(plain))
    import pymatching
    pymatching.Matching.from_detector_error_model(corr.detector_error_model(decompose_errors=True))


def test_device_circuit_needs_calibration_and_builds():
    with pytest.raises(ValueError):
        build_repetition_circuit(3, 3, NoiseSpec.from_device(1.0))
    cal = load_device_calibration("fake_quebec", 3)
    c = build_repetition_circuit(3, 3, NoiseSpec.from_device(1.0), calibration=cal, logical_state=1)
    det, obs = c.compile_detector_sampler(seed=0).sample(20000, separate_observables=True)
    assert 0.0 < det.mean() < 0.2 and 0.0 < obs.mean() < 0.2
    hi = build_repetition_circuit(3, 3, NoiseSpec.from_device(4.0), calibration=cal)
    det4, _ = hi.compile_detector_sampler(seed=0).sample(20000, separate_observables=True)
    assert det4.mean() > det.mean()


def test_surface_code_builder_and_bias():
    uni = build_surface_circuit(3, 3, NoiseSpec.uniform(0.005))
    assert uni.num_observables == 1 and uni.num_detectors > 0
    biased = build_surface_circuit(3, 3, NoiseSpec.biased(0.005, 5.0, 0.0))
    assert "PAULI_CHANNEL_1" in str(biased) and biased.num_detectors == uni.num_detectors
    with pytest.raises(ValueError):
        build_surface_circuit(3, 3, NoiseSpec.biased(0.005, 5.0, 0.5))


# ------------------------------------------------------------------------ Qiskit side
def test_qiskit_circuit_layout_and_registers():
    qc = build_qiskit_circuit(3, 3, logical_state=1)
    assert qc.num_qubits == 5
    assert [r.name for r in qc.cregs] == ["anc0", "anc1", "anc2", "data"]
    assert [r.size for r in qc.cregs] == [2, 2, 2, 3]
    ops = qc.count_ops()
    assert ops["cx"] == 12 and ops["measure"] == 9 and ops["reset"] == 4   # no reset after the last round
    assert build_qiskit_circuit(4, 0).num_qubits == 4                       # final-readout-only fallback


def test_bitarray_bit_order():
    """Column j of the matrix must be classical bit j (a known circuit: X on clbit 1 only)."""
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit_aer.primitives import SamplerV2

    qc = QuantumCircuit(QuantumRegister(3), ClassicalRegister(3, "c"))
    qc.x(1)
    qc.measure([0, 1, 2], [0, 1, 2])
    m = bitarray_to_matrix(SamplerV2().run([qc], shots=5).result()[0].data.c)
    assert np.array_equal(m, np.tile([0, 1, 0], (5, 1)))


def test_noiseless_qiskit_circuit_yields_no_events():
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2

    from src.circuits import sampler_result_to_measurements

    for rounds, logical in [(3, 0), (3, 1), (2, 1)]:
        qc = build_qiskit_circuit(3, rounds, logical)
        from qiskit import transpile
        isa = transpile(qc, AerSimulator(), optimization_level=0)
        pub = SamplerV2().run([isa], shots=64).result()[0]
        meas = sampler_result_to_measurements(pub.data, 3, rounds)
        det, obs = measurements_to_syndromes(meas, 3, rounds, logical)
        assert not det.any() and not obs.any()
