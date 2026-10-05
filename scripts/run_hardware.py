"""Experiment E4: repetition-code memory on IBM hardware (PINQ2) -- or its offline stand-in.

Real hardware (needs PINQ2_TOKEN in .env; submit EARLY, queues are long)::

    python -m scripts.run_hardware submit --d 3 --rounds 3 --shots 4096            # least-busy backend
    python -m scripts.run_hardware submit --d 3 --rounds 3 --shots 4096 --backend ibm_quebec
    python -m scripts.run_hardware fetch   results/hardware/<run_dir> --wait       # later
    python -m scripts.run_hardware analyze results/hardware/<run_dir>

Offline (no account, no network): everything runs against an IBM *fake backend* and
is flagged as SIMULATED in meta.json and on the figure::

    python -m scripts.run_hardware simulate --d 3 --rounds 3 --shots 8192

Other:  ``list`` shows the runs on disk.  ``--rounds 0`` is the final-readout-only
(code-capacity) fallback for backends without mid-circuit measurement support.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def _common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--d", type=int, default=3, help="repetition-code distance (3 or 5 recommended)")
    ap.add_argument("--rounds", type=int, default=3, help="syndrome rounds (1-3; 0 = readout-only fallback)")
    ap.add_argument("--shots", type=int, default=4096, help="shots (plan: 4096-8192)")
    ap.add_argument("--logical-state", type=int, choices=[0, 1], default=1,
                    help="prepare logical |1..1> (default; the harder state under T1 decay) or |0..0>")
    ap.add_argument("--opt-level", type=int, default=1, help="transpiler optimization level (plan: 1)")
    ap.add_argument("--chain", default=None, help="comma-separated physical qubits (default: best-calibrated chain)")
    ap.add_argument("--label", default="", help="free-text label stored in meta.json")
    ap.add_argument("--out-dir", type=Path, default=None, help="where run folders are created (default results/hardware)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("submit", help="submit a job to IBM Runtime (saves the job id immediately)")
    _common(s)
    s.add_argument("--backend", default=None, help="backend name (default: least busy operational device)")
    s.add_argument("--simulate", action="store_true", help="run on a fake backend locally instead")
    s.add_argument("--fake-backend", default="fake_quebec", help="fake backend for --simulate")

    f = sub.add_parser("fetch", help="download results of a submitted job")
    f.add_argument("run_dir", type=Path)
    f.add_argument("--wait", action="store_true", help="block until the job finishes")
    f.add_argument("--timeout", type=float, default=None)

    a = sub.add_parser("analyze", help="decode a fetched run and compare with simulations")
    a.add_argument("run_dir", type=Path)
    a.add_argument("--nn-steps", type=int, default=3000)
    a.add_argument("--aer-shots", type=int, default=20000)
    a.add_argument("--stim-shots", type=int, default=200000)
    a.add_argument("--retrain", action="store_true")
    a.add_argument("--figure", type=Path, default=None, help="where to copy the figure (default results/figures)")

    m = sub.add_parser("simulate", help="offline end-to-end demo: submit + fetch + analyze on a fake backend")
    _common(m)
    m.add_argument("--fake-backend", default="fake_quebec")
    m.add_argument("--nn-steps", type=int, default=3000)
    m.add_argument("--aer-shots", type=int, default=20000)
    m.add_argument("--stim-shots", type=int, default=200000)
    m.add_argument("--figure", type=Path, default=None, help="where to copy the figure (default results/figures)")

    sub.add_parser("list", help="list runs in results/hardware")
    args = ap.parse_args(argv)

    from src import hardware as hw
    from src.common import FIGURES_DIR, HARDWARE_DIR, ensure_dirs
    from src.noise import get_fake_backend

    ensure_dirs()
    chain = [int(x) for x in args.chain.split(",")] if getattr(args, "chain", None) else None

    if args.cmd == "list":
        for p in hw.list_runs():
            m_ = hw.RunMeta.load(p)
            print(f"{p.name:55s} {m_.status:10s} {'SIMULATED' if m_.simulated else 'hardware '} "
                  f"d={m_.d} rounds={m_.rounds} shots={m_.shots}")
        return 0

    if args.cmd == "submit":
        if args.simulate:
            backend = get_fake_backend(args.fake_backend)
            run_dir, job = hw.submit(args.d, args.rounds, args.shots, backend=backend, simulated=True,
                                     logical_state=args.logical_state, optimization_level=args.opt_level,
                                     chain=chain, label=args.label, seed_simulator=1234,
                                     out_root=args.out_dir or HARDWARE_DIR)
            hw.fetch(run_dir, job=job, wait=True)
            print(f"\nSimulated run saved to {run_dir}\nNext: python -m scripts.run_hardware analyze {run_dir}")
            return 0
        service = hw.get_service()
        backend = hw.resolve_backend(service, args.backend, 2 * args.d - 1)
        run_dir, job = hw.submit(args.d, args.rounds, args.shots, backend=backend, simulated=False,
                                 logical_state=args.logical_state, optimization_level=args.opt_level,
                                 chain=chain, label=args.label, out_root=args.out_dir or HARDWARE_DIR)
        print(f"\nSubmitted. Run directory: {run_dir}\n"
              f"Later:  python -m scripts.run_hardware fetch {run_dir} --wait\n"
              f"        python -m scripts.run_hardware analyze {run_dir}")
        return 0

    if args.cmd == "fetch":
        ok = hw.fetch(args.run_dir, wait=args.wait, timeout=args.timeout)
        return 0 if ok else 2

    if args.cmd in ("analyze", "simulate"):
        if args.cmd == "simulate":
            backend = get_fake_backend(args.fake_backend)
            run_dir, job = hw.submit(args.d, args.rounds, args.shots, backend=backend, simulated=True,
                                     logical_state=args.logical_state, optimization_level=args.opt_level,
                                     chain=chain, label=args.label or "offline demo", seed_simulator=1234,
                                     out_root=args.out_dir or HARDWARE_DIR)
            hw.fetch(run_dir, job=job, wait=True)
            kw = dict(nn_steps=args.nn_steps, stim_shots=args.stim_shots, aer_shots=args.aer_shots)
            figure = args.figure
        else:
            run_dir = args.run_dir
            kw = dict(nn_steps=args.nn_steps, stim_shots=args.stim_shots, aer_shots=args.aer_shots,
                      retrain=args.retrain)
            figure = args.figure
        df = hw.analyze_run(run_dir, **kw)
        meta = hw.RunMeta.load(run_dir)
        print("\n" + df[["source", "decoder", "shots", "errors", "ler", "ler_lo", "ler_hi"]].to_string(index=False))
        hw.plot_run(run_dir, df)
        tag = "simulated_demo" if meta.simulated else "hardware_vs_sim"
        dest = figure or (FIGURES_DIR / f"fig_e4_{tag}_d{meta.d}_r{meta.rounds}.png")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(run_dir) / "fig_hardware_vs_sim.png", dest)
        print(f"\nFigure: {dest}\nTable : {Path(run_dir) / 'analysis.csv'}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
