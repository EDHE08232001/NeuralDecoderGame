"""Run the experiments of the plan (E1-E3, E5, architecture study) end to end.

Examples
--------
    python -m scripts.run_experiments --quick                 # ~2 min smoke run (tiny budget)
    python -m scripts.run_experiments e1 e2 --workers 4       # selected experiments
    python -m scripts.run_experiments all --workers 4         # full run (about an hour on 4 cores)

Outputs: CSV tables in results/tables, PNG figures in results/figures, trained networks
in results/models (cached: re-running only trains what is missing; use --retrain to
force), and results/run_manifest.json (versions, config, seeds).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ALL = ["e1", "e2", "e3", "e3b", "e3c", "e5", "arch", "device"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiments", nargs="*", default=["all"],
                    help=f"any of {ALL} or 'all' (default)")
    ap.add_argument("--quick", action="store_true",
                    help="tiny budget (smoke test; written to results_quick/ unless --results-dir is given)")
    ap.add_argument("--workers", type=int, default=1, help="parallel training processes (default 1)")
    ap.add_argument("--results-dir", type=Path, default=None, help="where to write results (default: results/)")
    ap.add_argument("--shots", type=int, default=None, help="test shots per point (default 200000)")
    ap.add_argument("--steps-scale", type=float, default=1.0,
                    help="multiply every training budget by this factor (e.g. 0.25 for a fast run)")
    ap.add_argument("--retrain", action="store_true", help="ignore cached networks and retrain")
    ap.add_argument("--bias", type=float, default=None, help="Z:X bias r of noise model 2 (default 5)")
    ap.add_argument("--corr", type=float, default=None, help="correlated-flip strength of noise model 2 (default 0.5)")
    ap.add_argument("--no-ablation", action="store_true", help="skip the events-only input ablation in E2")
    args = ap.parse_args(argv)

    from src.common import RESULTS_DIR, ensure_dirs
    from src.experiments import Config, RUNNERS, write_manifest

    wanted = ALL if "all" in args.experiments else args.experiments
    bad = [w for w in wanted if w not in RUNNERS]
    if bad:
        ap.error(f"unknown experiment(s) {bad}; choose from {ALL}")

    if args.quick:
        cfg = Config.quick(args.results_dir or Path("results_quick"), verbose=True)
    else:
        ensure_dirs()
        cfg = Config(results_dir=args.results_dir or RESULTS_DIR)
    cfg.workers, cfg.retrain = args.workers, args.retrain
    if args.shots:
        cfg.shots_test = args.shots
    if args.bias is not None:
        cfg.bias = args.bias
    if args.corr is not None:
        cfg.corr = args.corr
    if args.no_ablation:
        cfg.ablation = False
    if args.steps_scale != 1.0:
        cfg.steps = {d: max(50, int(s * args.steps_scale)) for d, s in cfg.steps.items()}
        cfg.surface_steps = max(50, int(cfg.surface_steps * args.steps_scale))

    t0 = time.time()
    for name in wanted:
        RUNNERS[name](cfg)
    path = write_manifest(cfg, extra=dict(experiments=wanted, wall_seconds=round(time.time() - t0, 1)))
    print(f"\nDone in {time.time() - t0:.0f}s.  Tables: {cfg.tables_dir}  Figures: {cfg.figures_dir}  Manifest: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
