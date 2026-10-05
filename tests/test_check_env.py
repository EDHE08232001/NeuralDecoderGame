"""The environment checker must (a) pass on a healthy environment and (b) actually detect problems."""

from __future__ import annotations

import json

import pytest

from scripts import check_env as ce


def test_version_helpers():
    assert ce.parse_version("2.14.1+cpu") == (2, 14, 1)
    assert ce.parse_version("1.65.0") == (1, 65, 0)
    assert ce.parse_version("0.50.0.dev3") == (0, 50, 0)
    assert ce.version_ok("2.4.6", "1.26") and ce.version_ok("1.26", "1.26.0") and ce.version_ok("2.0", "2")
    assert not ce.version_ok("1.25.9", "1.26") and not ce.version_ok("0.9", "1.0")


def test_requirements_are_parsed_with_minimum_versions():
    reqs = ce.read_requirements()
    assert reqs["stim"] and reqs["pymatching"] and reqs["torch"] and reqs["qiskit-ibm-runtime"] and reqs["streamlit"]
    assert set(reqs) <= set(ce.IMPORT_NAME), "every requirement needs an import-name mapping"


def test_healthy_environment_passes(capsys, tmp_path):
    out = tmp_path / "r.json"
    code = ce.main(["--quick", "--no-color", "--json", str(out)])
    text = capsys.readouterr().out
    assert code == 0 and "READY" in text
    data = json.loads(out.read_text())
    assert not [r for r in data if r["status"] == "FAIL"]
    assert {r["section"] for r in data} == {"system", "packages", "functions", "optional"}


def test_missing_package_is_reported_and_dependent_checks_are_skipped(monkeypatch, capsys):
    real = ce._installed_version
    monkeypatch.setattr(ce, "_installed_version", lambda d: None if d == "stim" else real(d))
    pk = {r.name: r for r in ce.check_packages()}
    assert pk["stim"].status == ce.FAIL and "pip install" in pk["stim"].hint
    real_need = ce._need
    monkeypatch.setattr(ce, "_need", lambda *d: "stim" if "stim" in d else real_need(*d))
    fn = {r.name: r for r in ce.run_function_checks("quick")}
    assert fn["Stim syndrome sampling"].status == ce.SKIP and "stim" in fn["Stim syndrome sampling"].detail
    assert ce.main(["--quick", "--no-color"]) == 1
    text = capsys.readouterr().out
    assert "NOT READY" in text and "pip install" in text


def test_outdated_package_is_reported(monkeypatch):
    real = ce._installed_version
    monkeypatch.setattr(ce, "_installed_version", lambda d: "0.0.1" if d == "numpy" else real(d))
    r = {x.name: x for x in ce.check_packages()}["numpy"]
    assert r.status == ce.FAIL and "required" in r.detail and "--upgrade" in r.hint


def test_optional_packages_only_warn(monkeypatch):
    real = ce._installed_version
    monkeypatch.setattr(ce, "_installed_version", lambda d: None if d == "nbformat" else real(d))
    r = {x.name: x for x in ce.check_packages()}["nbformat"]
    assert r.status == ce.WARN


def test_a_broken_function_is_caught_with_its_hint(monkeypatch):
    def boom():
        raise RuntimeError("simulated breakage")

    monkeypatch.setattr(ce, "FUNCTION_CHECKS", [("functions", "Broken thing", boom, (), "quick", "reinstall the thing")])
    (r,) = ce.run_function_checks("quick")
    assert r.status == ce.FAIL and "simulated breakage" in r.detail and r.hint == "reinstall the thing"
    assert ce.summarize([r], strict=False) == 1


def test_strict_mode_turns_warnings_into_failures():
    warn = ce.Result("packages", "x", ce.WARN, "meh")
    ok = ce.Result("packages", "y", ce.PASS, "fine")
    assert ce.summarize([ok, warn], strict=False) == 0
    assert ce.summarize([ok, warn], strict=True) == 1


def test_python_version_gate(monkeypatch):
    import sys
    import types
    monkeypatch.setattr(sys, "version_info", types.SimpleNamespace(major=3, minor=8, micro=10))
    r = ce.check_system()[0]
    assert r.status == ce.FAIL and "3.10" in r.hint


@pytest.mark.slow
def test_full_mode_passes(capsys):
    assert ce.main(["--full", "--no-color"]) == 0
