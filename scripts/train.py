"""Train ONE neural decoder and compare it with MWPM.

Two ways to feed the network:

``circuit mode`` (default; the plan's recipe): fresh syndromes are sampled from the Stim
    circuit of the chosen code / noise model at *every* step -- unlimited free data, so
    overfitting is impossible.  Early stopping on a fixed validation set.

``dataset mode`` (``--dataset``): supervised training from a stored dataset (e.g. data
    collected with Qiskit Aer or on hardware by scripts/collect_dataset.py).  Samples are
    drawn with replacement; ``--val-dataset`` is required for early stopping.

Examples
--------
    python -m scripts.train --d 5 --noise uniform --p 0.01                       # plan defaults
    python -m scripts.train --d 5 --noise biased --p 0.01 --bias 5 --corr 0.5 --steps 6000
    python -m scripts.train --d 3 --noise device --scale 1                        # IBM-calibration noise
    python -m scripts.train --d 5 --noise uniform --p 0.01 --arch cnn             # or gru
    python -m scripts.train --d 5 --noise uniform --p 0.01 --no-record            # events-only input (plan literal)
    python -m scripts.train --code surface --d 3 --noise uniform --p 0.005        # stretch: surface code
    python -m scripts.train --dataset data/raw/train.npz --val-dataset data/raw/val.npz --d 3 --rounds 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts._common import add_noise_args, rounds_from_args, spec_from_args


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_noise_args(ap)
    ap.add_argument("--arch", choices=["mlp", "cnn", "gru"], default="mlp")
    ap.add_argument("--no-record", action="store_true",
                    help="events-only input (no cumulative-parity 'syndrome record' feature)")
    ap.add_argument("--steps", type=int, default=3000, help="training steps (plan: 3000)")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-schedule", choices=["constant", "cosine"], default="cosine",
                    help="'constant' is the plan's recipe; 'cosine' usually gets closer to MWPM per step")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=3, help="hidden layers of the MLP (plan: 2-3)")
    ap.add_argument("--val-shots", type=int, default=100_000)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--patience", type=int, default=8, help="early stopping patience, in evaluations")
    ap.add_argument("--test-shots", type=int, default=200_000, help="fresh shots for the final comparison")
    ap.add_argument("--accel", default=None, metavar="auto|cpu|cuda|mps",
                    help="compute device for training/inference (default: $QEC_DEVICE or auto = CUDA > MPS > CPU)")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    ap.add_argument("--dataset", type=Path, default=None, help="train from this stored dataset instead")
    ap.add_argument("--val-dataset", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="checkpoint path (default results/models/custom_*.pt)")
    args = ap.parse_args(argv)

    import numpy as np
    import torch

    from src.common import MODELS_DIR, derive_seed, ensure_dirs
    from src.data import SyndromeDataset, sample_syndromes
    from src.decoders import GreedyDecoder, MWPMDecoder
    from src.evaluate import evaluate_decoders
    from src.experiments import make_circuit
    from src.training import train_on_circuit, train_on_dataset

    if args.threads:
        torch.set_num_threads(args.threads)
    ensure_dirs()
    spec, rounds = spec_from_args(args), rounds_from_args(args)
    record = (not args.no_record) and args.code == "rep"
    key = f"{args.code}|d{args.d}|r{rounds}|{spec.key}"
    seeds = dict(val=derive_seed("val", key), train=derive_seed("train", f"{key}|{args.arch}|{record}"),
                 torch=derive_seed("torch", key))
    circuit = make_circuit(args.code, args.d, rounds, spec)
    common = dict(d=args.d, code=args.code, arch=args.arch, record=record, steps=args.steps, batch=args.batch,
                  lr=args.lr, lr_schedule=args.lr_schedule, hidden=args.hidden, depth=args.depth,
                  eval_every=args.eval_every, patience=args.patience, seeds=seeds,
                  meta=dict(code=args.code, d=args.d, rounds=rounds, noise=spec.to_dict()), log=print,
                  device=args.accel)
    from src.decoders.neural import resolve_device
    print(f"compute device: {resolve_device(args.accel)}")
    print(f"training {args.arch}{'+record' if record else ''} on {args.code} d={args.d} rounds={rounds} "
          f"[{spec.label}]  n_det={circuit.num_detectors}")
    if args.dataset:
        if not args.val_dataset:
            ap.error("--dataset needs --val-dataset (used for early stopping)")
        tr, va = SyndromeDataset.load(args.dataset), SyndromeDataset.load(args.val_dataset)
        common.pop("meta")
        common["batch"] = min(args.batch, len(tr))
        nn, res = train_on_dataset(tr, va, meta=dict(code=args.code, d=args.d, rounds=rounds), **common)
    else:
        nn, res = train_on_circuit(circuit, val_shots=args.val_shots, **common)
    print(f"done: {res.steps_run} steps in {res.seconds:.0f}s, best val BCE {res.best_val_loss:.5f} "
          f"@ step {res.best_step}{' (stopped early)' if res.stopped_early else ''}")

    out = args.out or MODELS_DIR / f"custom_{args.code}_{spec.key}_d{args.d}_r{rounds}_{args.arch}{'' if record else '-ev'}.pt"
    nn.save(out)
    print(f"saved {out}")

    # ---- fair comparison on FRESH test shots drawn from the true noise model ----
    det, obs = sample_syndromes(circuit, args.test_shots, derive_seed("test", key))
    decs = {"mwpm_matched": MWPMDecoder.from_circuit(circuit), "nn": nn}
    if args.code == "rep":
        decs["greedy"] = GreedyDecoder(args.d, rounds)
    rows = evaluate_decoders(decs, det, obs, rounds=max(rounds, 1), reference="mwpm_matched")
    print(f"\ntest set: {args.test_shots:,} fresh shots (same shots for every decoder), 95% Wilson intervals")
    print(f"{'decoder':14s} {'LER':>10s}  {'95% CI':>22s}   LER(MWPM)-LER(dec) [95% paired CI]")
    for r in rows:
        adv = "" if r["decoder"] == "mwpm_matched" else f"{r['adv']:+.5f} [{r['adv_lo']:+.5f}, {r['adv_hi']:+.5f}]"
        print(f"{r['decoder']:14s} {r['ler']:10.5f}  [{r['ler_lo']:.5f}, {r['ler_hi']:.5f}]   {adv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
