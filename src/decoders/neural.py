"""PyTorch neural decoders and their training loop (plan section 5.2 / 8.2).

Task: given the flattened detection events (binary vector of length ``n_det``),
predict whether the logical observable was flipped (one logit, binary cross-entropy).
Training labels are free because they come from simulation.

Recipe (plan E2): Adam, lr 1e-3, batch 4096, fresh syndromes sampled every step (no
overfitting is possible with free data), early stopping on a *fixed* held-out
validation set, best-validation weights restored at the end.

Architectures
-------------
``MLPDecoder``  2-3 hidden layers of 128-256 units (the plan's default model)
``GRUDecoder``  recurrent over the syndrome rounds (stretch goal)
``CNNDecoder``  1-D convolution along the ancilla axis, rounds as channels (optional)

Input features.  The network interface is always "raw detection events in, one logit
out".  With ``record=True`` (needs the repetition-code layout ``n_anc = d - 1``) the
network additionally computes, inside ``forward``, the cumulative parity of the events
over the rounds (the raw ancilla record) and feeds ``[events, record]`` to the model.
This is a fixed function of the events, i.e. no information is added; it only spares a
small ReLU network from learning a parity computation from scratch (see README,
"Neural decoder design notes", and the ``record=False`` ablation in experiment E2).
The GRU / CNN reshape the detector vector to ``(layers, n_anc)`` (layer-major, as
produced by :mod:`src.circuits`), so they apply to the repetition code only.
"""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from .base import Decoder


# ==================================================================== device
def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Pick the compute device: an explicit name, else ``$QEC_DEVICE``, else the best
    available accelerator (CUDA, then Apple MPS, then CPU).

    ``"auto"`` (or ``None``) auto-detects; ``"cuda"``/``"cuda:1"``/``"mps"``/``"cpu"`` are
    honoured and rejected with a clear error if the backend is not available.
    """
    if isinstance(device, torch.device):
        return device
    name = (device or os.environ.get("QEC_DEVICE") or "auto").strip().lower()
    mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "mps" if mps_ok else "cpu")
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if dev.type == "mps" and not mps_ok:
        raise RuntimeError("MPS requested but torch.backends.mps.is_available() is False")
    return dev


# =================================================================== networks
def _events_and_record(x: torch.Tensor, layers: int, n_anc: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the flat detector vector into ``(batch, layers, n_anc)`` events and the
    cumulative parity over layers.

    The cumulative parity of the detection events is the raw syndrome record (the
    ancilla outcomes themselves; in the last layer it is the parity pattern of the
    final data readout).  It carries *no extra information* -- it is a fixed function
    of the events -- but integrating events over time is a parity computation that
    ReLU networks learn slowly, so providing it explicitly (``record=True``) lets a
    small network reach MWPM-level accuracy with far fewer samples.
    """
    ev = x.reshape(x.shape[0], layers, n_anc)
    return ev, torch.remainder(torch.cumsum(ev, dim=1), 2.0)


class _Net(nn.Module):
    """Shared plumbing: layout bookkeeping + optional record features."""

    arch = "base"

    def _setup(self, n_det: int, n_anc: int | None, record: bool) -> None:
        if record and n_anc is None:
            raise ValueError("record features need n_anc (= d - 1, repetition code layout)")
        if n_anc is not None and n_det % n_anc:
            raise ValueError("n_det must be a multiple of n_anc")
        self.n_det, self.n_anc, self.record = n_det, n_anc, bool(record)
        self.layers_ = n_det // n_anc if n_anc else 1

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        """``(batch, layers, features)`` input tensor (features = n_anc or 2 * n_anc)."""
        ev, rec = _events_and_record(x, self.layers_, self.n_anc)
        return torch.cat([ev, rec], dim=2) if self.record else ev


class MLPDecoder(_Net):
    """Plan section 8.2 network: 2-3 hidden layers of 128-256 units, one logit."""

    arch = "mlp"

    def __init__(self, n_det: int, hidden: int = 256, depth: int = 3,
                 n_anc: int | None = None, record: bool = False):
        super().__init__()
        self._setup(n_det, n_anc, record)
        n_in = n_det * (2 if self.record else 1)
        layers: list[nn.Module] = [nn.Linear(n_in, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
        self.kwargs = dict(n_det=n_det, hidden=hidden, depth=depth, n_anc=n_anc, record=self.record)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._features(x).flatten(1) if self.record else x
        return self.net(h).squeeze(-1)


class GRUDecoder(_Net):
    """Recurrent over the syndrome rounds (stretch goal)."""

    arch = "gru"

    def __init__(self, n_det: int, n_anc: int, hidden: int = 128, num_layers: int = 1,
                 record: bool = False):
        super().__init__()
        self._setup(n_det, n_anc, record)
        self.gru = nn.GRU(n_anc * (2 if self.record else 1), hidden, num_layers=num_layers,
                          batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.kwargs = dict(n_det=n_det, n_anc=n_anc, hidden=hidden, num_layers=num_layers,
                           record=self.record)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(self._features(x))
        return self.head(out[:, -1]).squeeze(-1)


class CNNDecoder(_Net):
    """1-D convolution along the ancilla axis with the syndrome rounds as channels."""

    arch = "cnn"

    def __init__(self, n_det: int, n_anc: int, channels: int = 64, hidden: int = 128,
                 record: bool = False):
        super().__init__()
        self._setup(n_det, n_anc, record)
        c_in = self.layers_ * (2 if self.record else 1)
        self.conv = nn.Sequential(
            nn.Conv1d(c_in, channels, 3, padding=1), nn.ReLU(),
            nn.Conv1d(channels, channels, 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(channels * n_anc, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))
        self.kwargs = dict(n_det=n_det, n_anc=n_anc, channels=channels, hidden=hidden,
                           record=self.record)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self._features(x)                          # (B, layers, F)
        if self.record:                                # events | record -> channels
            f = torch.cat([f[:, :, : self.n_anc], f[:, :, self.n_anc:]], dim=1)
        return self.head(self.conv(f)).squeeze(-1)


ARCHS = {"mlp": MLPDecoder, "gru": GRUDecoder, "cnn": CNNDecoder}


def build_network(arch: str, n_det: int, n_anc: int | None = None, **kwargs) -> nn.Module:
    """Factory.  ``n_anc`` (= d - 1 for the repetition code) is required by gru / cnn and by
    ``record=True``; leave it ``None`` for an events-only MLP (e.g. the surface code)."""
    if arch not in ARCHS:
        raise ValueError(f"unknown architecture {arch!r}; choose from {sorted(ARCHS)}")
    if arch == "mlp":
        return MLPDecoder(n_det, n_anc=n_anc, **kwargs)
    if n_anc is None:
        raise ValueError(f"{arch} needs n_anc (number of ancilla columns, d - 1)")
    return ARCHS[arch](n_det, n_anc, **kwargs)


# ================================================================== training
@dataclass
class TrainResult:
    best_val_loss: float
    best_step: int
    steps_run: int
    stopped_early: bool
    seconds: float
    history: list[dict] = field(default_factory=list)   # [{"step", "train_loss", "val_loss"}]


@torch.no_grad()
def _val_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor, chunk: int = 65536) -> float:
    model.eval()
    lossf = nn.BCEWithLogitsLoss(reduction="sum")
    total = 0.0
    for i in range(0, len(x), chunk):
        total += float(lossf(model(x[i:i + chunk]), y[i:i + chunk]))
    model.train()
    return total / len(x)


def train(
    model: nn.Module,
    sampler: Callable[[int], tuple[np.ndarray, np.ndarray]],
    *,
    val_data: tuple[np.ndarray, np.ndarray],
    steps: int = 3000,
    batch: int = 4096,
    lr: float = 1e-3,
    eval_every: int = 200,
    patience: int = 8,
    min_delta: float = 1e-5,
    lr_schedule: str = "constant",
    seed: int = 0,
    log: Callable[[str], None] | None = None,
    device: str | torch.device | None = None,
) -> TrainResult:
    """Train ``model`` on fresh syndromes from ``sampler(batch) -> (det, obs)``.

    Early stopping: every ``eval_every`` steps the BCE on the fixed validation set is
    computed; training stops after ``patience`` evaluations without an improvement of at
    least ``min_delta``.  The best-validation weights are restored before returning.

    Reproducibility: ``seed`` only controls randomness *during* training.  The network's
    initial weights are drawn when it is constructed, so seed torch before building it --
    :func:`src.training.train_on_circuit` / ``train_on_dataset`` do this for you.

    ``lr_schedule="constant"`` is the plan's recipe (Adam, lr 1e-3); ``"cosine"``
    anneals the learning rate to ``0.01 * lr`` over ``steps`` (optional, usually lets a
    small network get noticeably closer to the optimal decoder for the same budget).

    ``device`` selects CPU / CUDA / Apple MPS (see :func:`resolve_device`; default auto).
    The model is left on that device; samples are generated on the CPU and copied over.
    """
    if lr_schedule not in ("constant", "cosine"):
        raise ValueError("lr_schedule must be 'constant' or 'cosine'")
    dev = resolve_device(device)
    torch.manual_seed(seed)
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=0.01 * lr)
             if lr_schedule == "cosine" else None)
    lossf = nn.BCEWithLogitsLoss()
    xv = torch.from_numpy(np.ascontiguousarray(val_data[0])).float().to(dev)
    yv = torch.from_numpy(np.ascontiguousarray(val_data[1])).float().to(dev)

    best = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_step, bad, stopped = 0, 0, False
    hist: list[dict] = []
    run_loss, run_n = 0.0, 0
    t0 = time.time()
    model.train()
    step = 0
    for step in range(1, steps + 1):
        det, obs = sampler(batch)                       # fresh data each step
        x = torch.from_numpy(det).float().to(dev)
        y = torch.from_numpy(obs).float().to(dev)
        loss = lossf(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        run_loss += loss.item()
        run_n += 1
        if step % eval_every == 0 or step == steps:
            vl = _val_loss(model, xv, yv)
            hist.append(dict(step=step, train_loss=run_loss / run_n, val_loss=vl))
            if log:
                log(f"  step {step:5d}  train {run_loss / run_n:.5f}  val {vl:.5f}")
            run_loss, run_n = 0.0, 0
            if vl < best - min_delta:
                best, best_step, bad = vl, step, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
                if bad >= patience:
                    stopped = True
                    break
    model.load_state_dict(best_state)
    model.eval()
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    return TrainResult(best_val_loss=best, best_step=best_step, steps_run=step,
                       stopped_early=stopped, seconds=time.time() - t0, history=hist)


# ============================================================ inference wrapper
class NeuralDecoder(Decoder):
    """Wraps a trained network + metadata behind the common ``Decoder`` interface."""

    def __init__(self, net: nn.Module, meta: dict | None = None, name: str = "NN",
                 device: str | torch.device | None = None):
        # ``device=None`` keeps the net wherever it already is (e.g. where it was trained)
        self.net = (net if device is None else net.to(resolve_device(device))).eval()
        self.meta = dict(meta or {})
        self.name = name
        self.n_det = net.kwargs["n_det"]

    @property
    def device(self) -> torch.device:
        return next(self.net.parameters()).device

    def to(self, device: str | torch.device | None) -> "NeuralDecoder":
        """Move the network to ``device`` (``None``/``"auto"`` = best available)."""
        self.net.to(resolve_device(device))
        return self

    @torch.no_grad()
    def predict_logits(self, detectors: np.ndarray, chunk: int = 131072) -> np.ndarray:
        det = np.asarray(detectors, dtype=np.uint8)
        if det.ndim == 1:
            det = det[None, :]
        if det.shape[1] != self.n_det:
            raise ValueError(f"network expects {self.n_det} detectors, got {det.shape[1]}")
        dev = self.device
        out = [self.net(torch.from_numpy(det[i:i + chunk]).float().to(dev)).cpu().numpy()
               for i in range(0, len(det), chunk)]
        return np.concatenate(out)

    def predict_proba(self, detectors: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.predict_logits(detectors)))

    def predict(self, detectors: np.ndarray) -> np.ndarray:
        return (self.predict_logits(detectors) > 0).astype(np.uint8)

    # ----------------------------------------------------------------- saving
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # metadata is round-tripped through JSON so the checkpoint only ever contains plain
        # python types and can be read back with torch.load(weights_only=True) (no pickle code)
        meta = json.loads(json.dumps(self.meta, default=str))
        torch.save({"arch": self.net.arch, "kwargs": self.net.kwargs,
                    "state_dict": {k: v.detach().cpu() for k, v in self.net.state_dict().items()},
                    "meta": meta}, path)
        return path

    @classmethod
    def load(cls, path: str | Path, name: str = "NN",
             device: str | torch.device | None = "cpu") -> "NeuralDecoder":
        """Load a checkpoint (always device-independent); ``device`` is where to put the net."""
        ck = torch.load(path, map_location="cpu", weights_only=True)
        kw = dict(ck["kwargs"])
        n_det = kw.pop("n_det")
        n_anc = kw.pop("n_anc", None)
        net = build_network(ck["arch"], n_det, n_anc, **kw)
        net.load_state_dict(ck["state_dict"])
        return cls(net, ck.get("meta", {}), name, device=device)
