"""Greedy nearest-neighbour matching: the "human-like" player.

The plan compares "a human/greedy player, MWPM, and a trained neural decoder".  Humans
playing the game naturally pair up the closest lit defects first and send leftovers to
the nearest edge of the code.  This decoder implements exactly that heuristic, which
(a) gives the leaderboard a reproducible stand-in for a human and (b) shows *why*
decoders are needed: greedy pairing fails on error chains, where the globally cheapest
explanation differs from the locally cheapest one.

Geometry (repetition code, ``n = d - 1`` ancilla columns, detector index
``layer * n + column``):

* defect <-> defect cost  = max(|layer difference|, |column difference|)
  (diagonal neighbours count as one step: a data error that strikes *between* two
  ancilla measurements lights up a diagonal pair in circuit-level noise)
* defect -> left  boundary = column + 1       (flips data qubits 0 .. column)
* defect -> right boundary = n - column       (flips data qubits column+1 .. d-1)
* ties: pairing two defects beats sending them to the boundary ("pair neighbours first")

The logical observable is the last data qubit, so it flips iff an odd number of
defects were sent to the **right** boundary.
"""

from __future__ import annotations

import numpy as np

from .base import Decoder, decode_unique

LEFT, RIGHT = -1, -2


class GreedyDecoder(Decoder):
    def __init__(self, d: int, rounds: int, name: str = "Greedy"):
        self.d = d
        self.rounds = rounds
        self.n = d - 1
        self.layers = rounds + 1 if rounds > 0 else 1
        self.n_det = self.n * self.layers
        self.name = name

    # ---------------------------------------------------------------- matching
    def match(self, detector_row: np.ndarray) -> list[tuple[int, int]]:
        """Greedy matching of one syndrome.

        Returns ``[(a, b), ...]`` with detector indices; ``b`` is ``LEFT`` / ``RIGHT``
        when ``a`` is matched to a boundary.  Deterministic: candidates are ordered by
        cost, then defect-defect before defect-boundary, then by detector index.
        """
        defects = np.flatnonzero(np.asarray(detector_row))
        k = len(defects)
        if k == 0:
            return []
        layer, col = defects // self.n, defects % self.n
        # (cost, is_boundary, i, j): j >= 0 -> partner defect index, else LEFT / RIGHT
        cands: list[tuple[int, int, int, int]] = []
        for i in range(k):
            left, right = int(col[i]) + 1, self.n - int(col[i])
            cands.append((left, 1, i, LEFT) if left <= right else (right, 1, i, RIGHT))
            for j in range(i + 1, k):
                cost = max(abs(int(layer[i]) - int(layer[j])), abs(int(col[i]) - int(col[j])))
                cands.append((cost, 0, i, j))
        cands.sort()
        done = np.zeros(k, dtype=bool)
        out: list[tuple[int, int]] = []
        for _, _, i, j in cands:
            if done[i] or (j >= 0 and done[j]):
                continue
            done[i] = True
            if j >= 0:
                done[j] = True
                out.append((int(defects[i]), int(defects[j])))
            else:
                out.append((int(defects[i]), j))
        return out

    def _predict_rows(self, rows: np.ndarray) -> np.ndarray:
        pred = np.zeros(len(rows), dtype=np.uint8)
        for r, row in enumerate(rows):
            pred[r] = sum(1 for _, b in self.match(row) if b == RIGHT) & 1
        return pred

    def predict(self, detectors: np.ndarray) -> np.ndarray:
        det = np.asarray(detectors, dtype=np.uint8)
        if det.ndim == 1:
            det = det[None, :]
        if det.shape[1] != self.n_det:
            raise ValueError(f"expected {self.n_det} detectors, got {det.shape[1]}")
        return decode_unique(self._predict_rows, det)
