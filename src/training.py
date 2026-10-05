"""Training helpers shared by the CLI, the experiments and the hardware analysis.

* :func:`train_on_circuit`  -- the plan's recipe: fresh syndromes from a Stim circuit
  every step, early stopping on a fixed validation set.
* :func:`train_on_dataset`  -- supervised training from a *stored* dataset (e.g. data
  collected with Qiskit Aer or on hardware); sampling with replacement, so overfitting
  is possible and early stopping on a separate validation set matters.
"""

from __future__ import annotations

import platform
from typing import Callable

import numpy as np
import stim
import torch

from .data import SyndromeDataset, SyndromeSampler, sample_syndromes
from .decoders.neural import NeuralDecoder, TrainResult, build_network, train


class DatasetSampler:
    """Random mini-batches (with replacement) from a stored :class:`SyndromeDataset`."""

    def __init__(self, ds: SyndromeDataset, seed: int):
        self.ds = ds
        self.rng = np.random.default_rng(seed)

    def __call__(self, batch: int) -> tuple[np.ndarray, np.ndarray]:
        idx = self.rng.integers(0, len(self.ds), size=batch)
        return self.ds.detectors[idx], self.ds.observables[idx]


def make_network(arch: str, n_det: int, *, d: int, code: str = "rep", record: bool = True,
                 hidden: int = 256, depth: int = 3) -> torch.nn.Module:
    """Network factory that knows the repetition-code layout (``n_anc = d - 1``)."""
    if code != "rep":
        if arch != "mlp":
            raise ValueError(f"{arch} is only implemented for the repetition code")
        return build_network("mlp", n_det, None, hidden=hidden, depth=depth, record=False)
    if arch == "mlp":
        return build_network("mlp", n_det, d - 1 if record else None, hidden=hidden, depth=depth, record=record)
    return build_network(arch, n_det, d - 1, record=record)


def _meta(net, res: TrainResult, extra: dict) -> dict:
    return dict(
        arch=net.arch, record=bool(getattr(net, "record", False)), steps_run=res.steps_run,
        best_step=res.best_step, best_val_loss=res.best_val_loss, train_seconds=res.seconds,
        stopped_early=res.stopped_early, history=res.history,
        versions=dict(stim=str(stim.__version__), torch=str(torch.__version__),
                      python=platform.python_version(), device=str(next(net.parameters()).device)), **extra)


def train_on_circuit(circuit: stim.Circuit, *, d: int, code: str = "rep", arch: str = "mlp",
                     record: bool = True, steps: int = 3000, batch: int = 4096, lr: float = 1e-3,
                     lr_schedule: str = "cosine", hidden: int = 256, depth: int = 3,
                     val_shots: int = 100_000, eval_every: int = 250, patience: int = 8,
                     seeds: dict | None = None, meta: dict | None = None,
                     log: Callable[[str], None] | None = None,
                     device: str | None = None) -> tuple[NeuralDecoder, TrainResult]:
    """Train a network on fresh samples of ``circuit`` (plan section 5.2)."""
    seeds = {"val": 11, "train": 13, "torch": 0, **(seeds or {})}
    val = sample_syndromes(circuit, val_shots, seeds["val"])
    torch.manual_seed(seeds["torch"])               # seed BEFORE construction: initial weights are reproducible too
    net = make_network(arch, circuit.num_detectors, d=d, code=code, record=record, hidden=hidden, depth=depth)
    res = train(net, SyndromeSampler(circuit, seeds["train"]), val_data=val, steps=steps, batch=batch,
                lr=lr, lr_schedule=lr_schedule, eval_every=eval_every, patience=patience,
                seed=seeds["torch"], log=log, device=device)
    info = {**dict(source="stim circuit", seeds=seeds, batch=batch, lr=lr, lr_schedule=lr_schedule,
                   steps_requested=steps, shots_val=val_shots), **(meta or {})}      # caller metadata may override
    return NeuralDecoder(net, _meta(net, res, info), name=arch.upper()), res


def train_on_dataset(train_ds: SyndromeDataset, val_ds: SyndromeDataset, *, d: int, code: str = "rep",
                     arch: str = "mlp", record: bool = True, steps: int = 3000, batch: int = 2048,
                     lr: float = 1e-3, lr_schedule: str = "cosine", hidden: int = 256, depth: int = 3,
                     eval_every: int = 250, patience: int = 8, seeds: dict | None = None,
                     meta: dict | None = None,
                     log: Callable[[str], None] | None = None,
                     device: str | None = None) -> tuple[NeuralDecoder, TrainResult]:
    """Train on a stored dataset; early stopping on ``val_ds`` (never on test data)."""
    seeds = {"train": 13, "torch": 0, **(seeds or {})}
    torch.manual_seed(seeds["torch"])               # seed BEFORE construction: initial weights are reproducible too
    net = make_network(arch, train_ds.n_det, d=d, code=code, record=record, hidden=hidden, depth=depth)
    res = train(net, DatasetSampler(train_ds, seeds["train"]),
                val_data=(val_ds.detectors, val_ds.observables), steps=steps, batch=batch, lr=lr,
                lr_schedule=lr_schedule, eval_every=eval_every, patience=patience, seed=seeds["torch"], log=log, device=device)
    info = {**dict(source="stored dataset", seeds=seeds, batch=batch, lr=lr, lr_schedule=lr_schedule,
                   steps_requested=steps, train_shots=len(train_ds), val_shots=len(val_ds),
                   dataset_meta=train_ds.meta), **(meta or {})}                      # caller metadata may override
    return NeuralDecoder(net, _meta(net, res, info), name=arch.upper()), res
