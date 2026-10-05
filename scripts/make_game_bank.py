"""Pre-generate the game's syndrome bank (plan section 6: "the game consumes pre-generated
syndromes and saved models; it never trains anything live").

For every (noise model, difficulty level, distance) this writes ``data/game_bank/<setting>.npz``
with ``--episodes`` fresh syndromes (seed role ``bank``: disjoint from the train / val / test
streams).  For the device-derived noise model it also snapshots the calibration of the
best-calibrated chain of the IBM fake backend into ``data/calibration/`` so the numbers do not
change when a newer qiskit-ibm-runtime ships different fake-backend data.

    python -m scripts.make_game_bank                       # everything (a few seconds)
    python -m scripts.make_game_bank --noise uniform --d 3 5
"""

from __future__ import annotations

import argparse
import sys
import time


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--noise", nargs="+", default=["uniform", "biased", "device"],
                    choices=["uniform", "biased", "device"])
    ap.add_argument("--d", nargs="+", type=int, default=[3, 5, 7])
    ap.add_argument("--episodes", type=int, default=1000)
    args = ap.parse_args(argv)

    from game import engine as eng
    from src.common import derive_seed, ensure_dirs
    from src.data import make_dataset

    ensure_dirs()
    t0 = time.time()
    n_files = 0
    for noise in args.noise:
        for level in eng.levels_for(noise):
            for d in args.d:
                spec = eng.spec_for(noise, level)
                circuit = eng._circuit(noise, level, d)
                seed = derive_seed("bank", f"{spec.key}|d{d}")
                ds = make_dataset(circuit, args.episodes, seed,
                                  dict(code="rep", d=d, rounds=d, noise=spec.to_dict(), role="bank"))
                ds.save(eng.bank_path(noise, level, d))
                n_files += 1
        print(f"  {noise}: done")
    print(f"wrote {n_files} bank files to data/game_bank in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
