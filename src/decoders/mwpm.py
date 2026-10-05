"""Minimum-weight perfect matching baseline (PyMatching).

MWPM treats the decoding problem as matching defects on a graph whose edge weights
are ``log((1 - p_e) / p_e)`` for *independent* error mechanisms ``e``.  The graph is
built from the circuit's detector error model (DEM), so "noise-matched" weights come
for free: build the decoder from the **same** circuit that generated the data.

Two flavours are used in the experiments (plan sections 5.3 and 9):

* ``MWPMDecoder.from_circuit(noisy_circuit)``      noise-matched weights ("best available")
* ``MWPMDecoder.from_circuit(uniform_circuit)``    mis-specified weights (assumes the
  uniform model while the data comes from the biased / correlated model)

Verified for the plan's ``[verify: ...]`` item: ``Matching.from_detector_error_model``
requires *graph-like* errors (at most two detectors).  Correlated events that flip four
detectors are accepted via ``decompose_errors=True``: Stim splits them into two
graph-like components that already exist as independent mechanisms, and PyMatching adds
each component as an ordinary edge.  This is exactly the information MWPM loses -- the
two halves are treated as independent although they always occur together.
"""

from __future__ import annotations

import numpy as np
import pymatching
import stim

from .base import Decoder


class MWPMDecoder(Decoder):
    def __init__(self, matching: pymatching.Matching, name: str = "MWPM", n_det: int | None = None):
        self.matching = matching
        self.name = name
        self.n_det = n_det if n_det is not None else matching.num_detectors

    # ------------------------------------------------------------- constructors
    @classmethod
    def from_dem(cls, dem: stim.DetectorErrorModel, name: str = "MWPM") -> "MWPMDecoder":
        return cls(pymatching.Matching.from_detector_error_model(dem), name, dem.num_detectors)

    @classmethod
    def from_circuit(cls, circuit: stim.Circuit, name: str = "MWPM",
                     ignore_decomposition_failures: bool = False) -> "MWPMDecoder":
        """Build from the circuit's own DEM (noise-matched weights).

        ``ignore_decomposition_failures=True`` silently drops errors that cannot be
        split into graph-like pieces (needed e.g. for some surface-code noise); the
        default raises so that a mis-built baseline can never go unnoticed.
        """
        dem = circuit.detector_error_model(
            decompose_errors=True, ignore_decomposition_failures=ignore_decomposition_failures)
        return cls.from_dem(dem, name)

    # --------------------------------------------------------------- inference
    def predict(self, detectors: np.ndarray) -> np.ndarray:
        det = np.asarray(detectors)
        if det.ndim == 1:
            det = det[None, :]
        pred = self.matching.decode_batch(det.astype(np.uint8))
        return np.asarray(pred[:, 0], dtype=np.uint8)
