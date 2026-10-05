"""Common decoder interface.

Every decoder exposes ``name`` and ``predict(detectors) -> uint8[shots]`` (predicted
observable flip).  The evaluation code only relies on this, so MWPM, the greedy
"human-like" player and the neural networks are interchangeable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Decoder(ABC):
    name: str = "decoder"

    @abstractmethod
    def predict(self, detectors: np.ndarray) -> np.ndarray:
        """``detectors``: uint8 ``(shots, n_det)`` -> uint8 ``(shots,)`` logical flip."""

    def logical_error_rate(self, detectors: np.ndarray, observables: np.ndarray) -> float:
        return float(np.mean(self.predict(detectors) != np.asarray(observables)))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} {self.name!r}>"


def decode_unique(fn, detectors: np.ndarray) -> np.ndarray:
    """Apply ``fn`` (rows -> predictions) only to the *distinct* syndromes.

    At low physical error rates most shots share a handful of syndromes (typically the
    all-zero one), so this speeds up slow per-shot decoders by orders of magnitude
    without changing any result.
    """
    det = np.ascontiguousarray(detectors, dtype=np.uint8)
    uniq, inverse = np.unique(det, axis=0, return_inverse=True)
    return np.asarray(fn(uniq), dtype=np.uint8)[inverse.reshape(-1)]
