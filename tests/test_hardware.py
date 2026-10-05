"""Hardware module (E4).  Everything except the network call to IBM is exercised here:
the real workflow runs against an IBM *fake backend* through qiskit_ibm_runtime.SamplerV2,
and the authenticated path is tested with injected stand-ins (no token, no network)."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import hardware as hw
from src.noise import get_fake_backend


@pytest.fixture(scope="module")
def backend():
    return get_fake_backend("fake_quebec")


class StubJob:
    """Looks like a Runtime job whose state we control."""

    def __init__(self, job_id="job-abc123", status="QUEUED", result=None):
        self._id, self._status, self._result = job_id, status, result

    def job_id(self):
        return self._id

    def status(self):
        return self._status

    def result(self):
        if self._result is None:
            raise AssertionError("result() must not be called before the job is DONE")
        return self._result


class StubSampler:
    def __init__(self, job):
        self.job, self.calls = job, []

    def run(self, pubs, shots=None):
        self.calls.append((pubs, shots))
        return self.job


def _simulate(tmp_path, backend, d=3, rounds=2, shots=300, **kw):
    run_dir, job = hw.submit(d, rounds, shots, backend=backend, simulated=True, out_root=tmp_path,
                             seed_simulator=7, log=lambda *_: None, **kw)
    assert hw.fetch(run_dir, job=job, wait=True, log=lambda *_: None)
    return run_dir


def test_job_id_is_saved_before_the_result_is_requested(tmp_path, backend):
    job = StubJob("job-abc123", "QUEUED", result=None)         # result() would raise
    sampler = StubSampler(job)
    run_dir, returned = hw.submit(3, 2, 512, backend=backend, simulated=False, out_root=tmp_path,
                                  sampler_factory=lambda b: sampler, log=lambda *_: None)
    assert returned is job and sampler.calls[0][1] == 512
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["job_id"] == "job-abc123" and meta["status"] == "SUBMITTED" and meta["simulated"] is False
    assert "Real hardware" in meta["note"]
    assert (run_dir / "calibration.json").exists() and (run_dir / "circuit.qpy").exists()
    assert meta["chain"] and meta["isa_ops"]["measure"] > 0 and meta["dynamic_circuits"]["ok"]


def test_fetch_returns_false_until_done_then_saves_raw_data(tmp_path, backend):
    job = StubJob("job-xyz", "RUNNING")
    run_dir, _ = hw.submit(3, 2, 200, backend=backend, simulated=False, out_root=tmp_path,
                           sampler_factory=lambda b: StubSampler(job), log=lambda *_: None)
    assert hw.fetch(run_dir, job=job, log=lambda *_: None) is False
    assert hw.RunMeta.load(run_dir).status == "RUNNING"
    assert not (run_dir / "raw_record.npz").exists()
    # the same job now finishes: build a genuine result with the local runtime sampler
    real_dir = _simulate(tmp_path / "real", backend, shots=200)
    real = hw.load_run(real_dir)
    done = StubJob("job-xyz", "DONE", result=_FakeResult(real.measurements, 3, 2))
    assert hw.fetch(run_dir, job=done, log=lambda *_: None) is True
    run = hw.load_run(run_dir)
    assert run.meta.status == "DONE" and run.measurements.shape == (200, 2 * 2 + 3)


class _FakeResult:
    """PrimitiveResult-like object assembled from a saved measurement record."""

    def __init__(self, meas, d, rounds):
        from qiskit.primitives.containers.bit_array import BitArray

        n = d - 1
        data = types.SimpleNamespace()
        for t in range(rounds):
            setattr(data, f"anc{t}", BitArray.from_samples(_to_ints(meas[:, t * n:(t + 1) * n]), num_bits=n))
        data.data = BitArray.from_samples(_to_ints(meas[:, rounds * n:]), num_bits=d)
        self._pub = types.SimpleNamespace(data=data)

    def __getitem__(self, i):
        return self._pub


def _to_ints(bits):
    return [int("".join(map(str, row[::-1])), 2) for row in bits]


def test_simulated_run_is_flagged_and_data_roundtrips(tmp_path, backend):
    run_dir = _simulate(tmp_path, backend, d=3, rounds=3, shots=400)
    meta = hw.RunMeta.load(run_dir)
    assert meta.simulated and "SIMULATED" in meta.note and meta.status == "DONE"
    run = hw.load_run(run_dir)
    det, obs = run.syndromes()
    assert det.shape == (400, 8) and obs.shape == (400,)
    assert 0.0 < det.mean() < 0.2                                  # noisy but alive
    counts = json.loads((run_dir / "counts.json").read_text())["counts"]
    assert sum(counts.values()) == 400
    assert run_dir in hw.list_runs(tmp_path)


def test_aer_is_reproducible_with_a_seed(backend):
    chain = hw.best_linear_chain(backend, 5)
    a = hw.run_aer(3, 2, 300, backend, chain, seed=11)
    b = hw.run_aer(3, 2, 300, backend, chain, seed=11)
    c = hw.run_aer(3, 2, 300, backend, chain, seed=12)
    assert np.array_equal(a[0], b[0]) and not np.array_equal(a[0], c[0])


def test_readout_only_fallback_without_mid_circuit_operations(tmp_path, backend):
    run_dir = _simulate(tmp_path, backend, d=3, rounds=0, shots=300)
    det, obs = hw.load_run(run_dir).syndromes()
    assert det.shape == (300, 2)                                    # code-capacity syndrome of the readout
    ops = hw.RunMeta.load(run_dir).isa_ops
    assert ops.get("reset", 0) == 0 and ops["measure"] == 3


def test_submit_refuses_backends_without_mid_circuit_support(tmp_path, backend):
    class NoReset:
        name = "no_reset_backend"
        target = types.SimpleNamespace(operation_names={"measure", "x", "sx", "rz", "ecr"})

        def configuration(self):
            raise AttributeError

    assert hw.dynamic_circuit_report(NoReset())["ok"] is False
    with pytest.raises(RuntimeError, match="--rounds 0"):
        hw.submit(3, 2, 10, backend=NoReset(), simulated=False, out_root=tmp_path, log=lambda *_: None)
    assert hw.dynamic_circuit_report(backend)["ok"] is True


def test_analyze_run_compares_sources_and_plots(tmp_path, backend):
    run_dir = _simulate(tmp_path, backend, d=3, rounds=2, shots=600)
    df = hw.analyze_run(run_dir, nn_steps=40, stim_shots=3000, log=lambda *_: None)
    assert set(df["source"]) == {"Aer (this run)", "Stim approx."}
    assert set(df["decoder"]) == {"mwpm_matched", "nn", "greedy"}
    assert (df["errors"] <= df["shots"]).all() and (run_dir / "analysis.csv").exists()
    assert (run_dir / "nn_device_approx.pt").exists()
    hw.plot_run(run_dir, df)
    assert (run_dir / "fig_hardware_vs_sim.png").stat().st_size > 5000
    assert isinstance(pd.read_csv(run_dir / "analysis.csv"), pd.DataFrame)


def test_token_is_never_written_to_disk(tmp_path, backend, monkeypatch):
    secret = "SUPER-SECRET-PINQ2-TOKEN-0123456789"
    monkeypatch.setenv("PINQ2_TOKEN", secret)
    monkeypatch.setenv("PINQ2_INSTANCE", "crn:v1:secret-instance")
    run_dir = _simulate(tmp_path, backend, d=3, rounds=1, shots=100)
    hw.analyze_run(run_dir, nn_steps=20, stim_shots=1000, log=lambda *_: None)
    for f in run_dir.iterdir():
        blob = f.read_bytes()
        assert secret.encode() not in blob and b"crn:v1:secret-instance" not in blob, f.name


def test_get_service_requires_a_token_and_never_echoes_it(monkeypatch):
    monkeypatch.setattr(hw, "load_env", lambda: None)
    monkeypatch.delenv("PINQ2_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="PINQ2_TOKEN"):
        hw.get_service()


def test_get_service_passes_credentials_from_the_environment(monkeypatch):
    captured = {}

    class FakeService:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import qiskit_ibm_runtime
    monkeypatch.setattr(qiskit_ibm_runtime, "QiskitRuntimeService", FakeService)
    monkeypatch.setattr(hw, "load_env", lambda: None)
    monkeypatch.setenv("PINQ2_TOKEN", " tok ")
    monkeypatch.setenv("PINQ2_INSTANCE", "crn:abc")
    monkeypatch.delenv("QISKIT_CHANNEL", raising=False)
    hw.get_service()
    assert captured == dict(channel="ibm_cloud", token="tok", instance="crn:abc")
    captured.clear()
    monkeypatch.setenv("PINQ2_INSTANCE", "")
    monkeypatch.setenv("QISKIT_CHANNEL", "ibm_quantum_platform")
    hw.get_service()
    assert captured == dict(channel="ibm_quantum_platform", token="tok")


def test_resolve_backend_uses_named_or_least_busy():
    calls = []

    class Svc:
        def backend(self, name):
            calls.append(("backend", name))
            return "named"

        def least_busy(self, **kw):
            calls.append(("least_busy", kw))
            return "lb"

    assert hw.resolve_backend(Svc(), "ibm_quebec", 5) == "named"
    assert hw.resolve_backend(Svc(), None, 9) == "lb"
    assert calls[1][1] == dict(operational=True, simulator=False, min_num_qubits=9)


def test_fake_name_for_maps_real_names():
    assert hw.fake_name_for("ibm_quebec") == "fake_quebec"
    assert hw.fake_name_for("ibm_not_a_device") is None


def test_gitignore_protects_the_token_file():
    root = Path(__file__).resolve().parent.parent
    ignored = (root / ".gitignore").read_text().splitlines()
    assert ".env" in ignored
    example = (root / ".env.example").read_text()
    assert "PINQ2_TOKEN=" in example and "PINQ2_TOKEN=\n" in example     # the template holds no value
