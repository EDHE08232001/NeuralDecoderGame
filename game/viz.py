"""Matplotlib drawings for the game (independent of Streamlit, so they can be tested)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from src.circuits import detector_layout  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
DEFECT, CORRECTION, TRUTH, OK, BAD = "#eda100", "#2a78d6", "#e34948", "#1a7f37", "#c62828"


def defect_figure(det_row: np.ndarray, d: int, rounds: int, *, correction: np.ndarray | None = None,
                  true_err: np.ndarray | None = None, residual: np.ndarray | None = None,
                  title: str | None = None):
    """Space-time defect map + data-qubit row.

    Rows = syndrome rounds (top = round 0) and the final-readout row; lit circles are defects
    (detection events).  Bottom row = data qubits: blue squares = the player's correction,
    red rings = the qubits that were *really* flipped (revealed after submitting).
    ``residual`` = defects left after applying the correction (red diamonds between qubits).
    """
    layers, n = detector_layout(d, rounds)
    grid = np.asarray(det_row, dtype=np.uint8).reshape(layers, n)
    data_y = layers + 0.6
    fig, ax = plt.subplots(figsize=(max(4.6, 0.95 * d + 1.8), 0.46 * (layers + 4) + 0.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for t in range(layers):                                   # every detector position (faint)
        ax.scatter([2 * j + 1 for j in range(n)], [t] * n, s=70, color=GRID, zorder=1)
    ts, js = np.nonzero(grid)
    ax.scatter(2 * js + 1, ts, s=330, color=DEFECT, edgecolor=INK, linewidth=1.4, zorder=3)
    for i in range(d):                                        # data qubits
        ax.scatter([2 * i], [data_y], s=300, facecolor=SURFACE, edgecolor=INK2, linewidth=1.6, zorder=2)
        ax.text(2 * i, data_y + 0.62, f"q{i}", ha="center", va="center", fontsize=9, color=INK2)
    if true_err is not None:
        for i in np.flatnonzero(true_err):
            ax.scatter([2 * i], [data_y], s=700, facecolor="none", edgecolor=TRUTH, linewidth=2.6, zorder=4)
    if correction is not None:
        for i in np.flatnonzero(correction):
            ax.scatter([2 * i], [data_y], s=200, marker="s", color=CORRECTION, zorder=5)
    if residual is not None:
        for j in np.flatnonzero(residual):
            ax.scatter([2 * j + 1], [data_y - 0.55], s=140, marker="D", facecolor="none", edgecolor=BAD,
                       linewidth=2.0, zorder=4)
    labels = [f"round {t}" for t in range(layers - 1)] + ["final readout"]
    ax.set_yticks(list(range(layers)) + [data_y])
    ax.set_yticklabels(labels + ["data qubits"], fontsize=8, color=INK2)
    ax.set_xticks([])
    ax.set_xlim(-1, 2 * d - 1)
    ax.set_ylim(data_y + 1.1, -0.6)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    handles = [Line2D([], [], marker="o", ls="", color=DEFECT, mec=INK, ms=10, label="defect (parity check changed)")]
    if correction is not None:
        handles.append(Line2D([], [], marker="s", ls="", color=CORRECTION, ms=8, label="your correction (flip back)"))
    if true_err is not None:
        handles.append(Line2D([], [], marker="o", ls="", mfc="none", mec=TRUTH, mew=2.4, ms=12, label="really flipped"))
    if residual is not None and np.any(residual):
        handles.append(Line2D([], [], marker="D", ls="", mfc="none", mec=BAD, mew=2, ms=8, label="defect left by your correction"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False, fontsize=8)
    if title:
        ax.set_title(title, fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    return fig


def chain_example_figure(d: int = 7):
    """Worked example of why local rules fail on chains (computed with the real decoders).

    A chain of three adjacent flipped qubits leaves two defects that are each *closer to an edge
    than to each other*; greedy pairing sends them to opposite edges (a logical flip), while the
    globally cheapest explanation -- which MWPM finds -- pairs them.
    """
    from game import engine as eng
    from src.circuits import build_repetition_circuit, error_estimate, final_syndrome
    from src.decoders import GreedyDecoder, MWPMDecoder
    from src.noise import NoiseSpec

    truth = np.zeros(d, dtype=np.uint8)
    truth[[d // 2 - 1, d // 2, d // 2 + 1]] = 1
    det = (truth[:-1] ^ truth[1:]).astype(np.uint8)         # code-capacity syndrome (rounds = 0)
    obs = int(truth[d - 1])
    greedy = GreedyDecoder(d, 0)
    mwpm = MWPMDecoder.from_circuit(build_repetition_circuit(d, 0, NoiseSpec.uniform(0.05)))
    rows = [("What really happened", truth, None)]
    for name, dec in (("Greedy: pair / nearest edge", greedy), ("MWPM: cheapest global explanation", mwpm)):
        pred = int(dec.predict(det[None, :])[0])
        c = error_estimate(final_syndrome(det, d, 0), pred, d)
        rows.append((name, c, bool(np.array_equal(c, truth))))
    fig, axes = plt.subplots(len(rows), 1, figsize=(6.4, 1.15 * len(rows) + 0.4), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (name, flips, ok) in zip(axes, rows):
        ax.set_facecolor(SURFACE)
        for i in range(d):
            fill = (TRUTH if ok is None else CORRECTION) if flips[i] else SURFACE
            ax.scatter([2 * i], [0], s=420, facecolor=fill, edgecolor=INK2,
                       linewidth=1.6, zorder=2, marker="s" if flips[i] else "o")
        for j in np.flatnonzero(det):
            ax.scatter([2 * j + 1], [0.75], s=260, color=DEFECT, edgecolor=INK, linewidth=1.3, zorder=3)
        verdict = "" if ok is None else ("   ✓ correct" if ok else "   ✗ logical error")
        ax.set_title(name + verdict, loc="left", fontsize=10, color=(INK if ok is None else (OK if ok else BAD)))
        ax.set_ylim(-0.6, 1.2)
        ax.axis("off")
    axes[0].text(2 * d - 1.2, 0.75, "defects", ha="right", va="center", fontsize=8, color=INK2)
    fig.tight_layout()
    return fig
