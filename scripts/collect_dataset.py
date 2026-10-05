"""Collect a syndrome dataset (plan section 3 "Syndrome datasets").

A dataset = detection events + logical-observable flips for ``--shots`` independent runs of
the memory experiment, stored as a compressed .npz (see src/data.py) together with a JSON
description of exactly how it was produced.

Sources
-------
``stim``      fast bulk sampling of the Pauli-noise circuit (millions of shots per second);
              this is what the networks are trained on (labels are free).
``aer``       Qiskit Aer with an IBM backend's noise model, running the *transpiled*
              Qiskit circuit (slow, ~1 ms/shot; more faithful to a device than Stim).
``hardware``  convert a fetched IBM Runtime run (results/hardware/<run>) into a dataset.

Seeds: ``--role {train,val,test,bank}`` derives a reproducible seed from the role and the
configuration, so train / validation / test splits of the same configuration never
overlap; ``--seed`` overrides it.

Examples
--------
    # 1M training shots of a d=5 uniform-noise experiment (plan: 10^5 - 10^6 per (d, p))
    python -m scripts.collect_dataset --source stim --d 5 --noise uniform --p 0.01 --shots 1000000 --role train
    # the matching held-out test set (different seed, never used for training)
    python -m scripts.collect_dataset --source stim --d 5 --noise uniform --p 0.01 --shots 200000 --role test
    # biased / correlated noise
    python -m scripts.collect_dataset --source stim --d 5 --noise biased --p 0.01 --bias 5 --corr 0.5 --shots 500000
    # Qiskit Aer with the FakeQuebec noise model (device-like data)
    python -m scripts.collect_dataset --source aer --d 3 --rounds 3 --shots 100000 --fake-backend fake_quebec
    # hardware run -> dataset
    python -m scripts.collect_dataset --source hardware --run-dir results/hardware/<run>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from scripts._common import add_noise_args, rounds_from_args, spec_from_args


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["stim", "aer", "hardware"], default="stim")
    add_noise_args(ap)
    ap.add_argument("--shots", type=int, default=100_000)
    ap.add_argument("--role", choices=["train", "val", "test", "bank"], default="train",
                    help="seed role (keeps train/val/test disjoint); ignored if --seed is given")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--logical-state", type=int, choices=[0, 1], default=0,
                    help="logical state prepared (aer/hardware; stim noise is symmetric)")
    ap.add_argument("--fake-backend", default="fake_quebec", help="[aer] IBM fake backend providing the noise model")
    ap.add_argument("--run-dir", type=Path, default=None, help="[hardware] fetched run directory")
    ap.add_argument("--out", type=Path, default=None, help="output .npz (default: data/raw/<auto>.npz)")
    args = ap.parse_args(argv)

    import numpy as np

    from src.common import RAW_DIR, derive_seed, ensure_dirs
    from src.data import SyndromeDataset, make_dataset

    ensure_dirs()
    t0 = time.time()

    if args.source == "hardware":
        if args.run_dir is None:
            ap.error("--source hardware needs --run-dir")
        from src.hardware import load_run

        run = load_run(args.run_dir)
        det, obs = run.syndromes()
        m = run.meta
        ds = SyndromeDataset(det, obs, dict(
            source="hardware" if not m.simulated else "simulated (Aer, fake backend)",
            backend=m.backend, job_id=m.job_id, d=m.d, rounds=m.rounds, shots=m.shots,
            logical_state=m.logical_state, chain=m.chain, code="rep"))
        out = args.out or RAW_DIR / f"{m.backend}_d{m.d}_r{m.rounds}_{m.job_id[:8]}.npz"
    else:
        spec = spec_from_args(args)
        rounds = rounds_from_args(args)
        key = f"{args.code}|d{args.d}|r{rounds}|{spec.key}"
        seed = args.seed if args.seed is not None else derive_seed(args.role, key)
        meta = dict(code=args.code, d=args.d, rounds=rounds, noise=spec.to_dict(), role=args.role,
                    logical_state=args.logical_state)
        if args.source == "stim":
            from src.experiments import make_circuit

            circuit = make_circuit(args.code, args.d, rounds, spec)
            ds = make_dataset(circuit, args.shots, seed, dict(meta, source="stim"))
        else:  # aer
            if args.code != "rep":
                ap.error("--source aer is implemented for the repetition code")
            from src.hardware import prepare_circuit, run_aer
            from src.noise import get_fake_backend

            backend = get_fake_backend(args.fake_backend)
            _, _, chain, cal = prepare_circuit(args.d, rounds, backend, logical_state=args.logical_state)
            print(f"[aer] {cal.summary()}")
            det, obs = run_aer(args.d, rounds, args.shots, backend, chain, logical_state=args.logical_state,
                               seed=seed, log=print)
            ds = SyndromeDataset(det, obs, dict(meta, source="aer", backend=backend.name, chain=chain,
                                                shots=args.shots, seed=seed, n_det=det.shape[1]))
        suffix = f"{args.source}_{args.code}_{spec.key}_d{args.d}_r{rounds}_{args.role}"
        out = args.out or RAW_DIR / f"{suffix}.npz"

    path = ds.save(out)
    print(f"saved {len(ds):,} shots  (n_det={ds.n_det}, undecoded logical flip rate "
          f"{ds.logical_error_rate_undecoded:.4f}, mean detection rate {float(np.mean(ds.detectors)):.4f}) "
          f"-> {path}  [{time.time() - t0:.1f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
