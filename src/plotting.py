"""Figures for the evaluation protocol (plan section 9) -- matplotlib, static PNG.

Design rules (validated with the dataviz palette checker):

* one hue per *decoder family*, fixed across every figure -- MWPM = blue, neural net =
  orange, greedy = aqua, human player = violet; variants inside a family (e.g. MWPM with
  uniform weights, NN trained on another noise model) differ by line style + marker,
  never by an extra hue.  Aqua is < 3:1 on the light surface, so every figure carries a
  legend and the numbers are also written to CSV (results/tables/).
* distance ``d`` is ordinal -> one-hue blue ramp (steps 250 / 450 / 650).
* thin 2 px lines, hairline solid grid, recessive spines, error bars = 95 % Wilson
  intervals (paired CIs for the advantage plots).
* a hollow ``v`` marks a point with **zero** observed failures; its height is the 95 %
  upper confidence bound (a log axis cannot show 0).
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .evaluate import wilson_interval  # noqa: E402

SURFACE, INK, INK2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#c9c8c3"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
D_RAMP = {3: "#86b6ef", 5: "#2a78d6", 7: "#104281"}     # blue ramp steps 250 / 450 / 650

#: decoder key -> legend label + fixed style (color follows the entity, never its rank)
STYLE: dict[str, dict] = {
    "mwpm_matched": dict(label="MWPM (noise-matched weights)", color=BLUE, marker="o", ls="-"),
    "mwpm_uniform": dict(label="MWPM (uniform weights)", color=BLUE, marker="o", ls="--", hollow=True),
    "nn": dict(label="Neural net (MLP)", color=ORANGE, marker="s", ls="-"),
    "nn_transfer": dict(label="NN trained on uniform noise", color=ORANGE, marker="D", ls="-.", hollow=True),
    "nn_aer": dict(label="NN trained on Aer data", color=ORANGE, marker="h", ls="-.", hollow=True),
    "nn_events": dict(label="NN, events-only input", color=ORANGE, marker="s", ls="--", hollow=True),
    "cnn": dict(label="CNN", color=ORANGE, marker="X", ls=":"),
    "gru": dict(label="GRU", color=ORANGE, marker="P", ls="-."),
    "greedy": dict(label="Greedy (human-like)", color=AQUA, marker="^", ls=":"),
    "human": dict(label="Human player", color=VIOLET, marker="o", ls="-"),
}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2.0,
        "lines.markersize": 7, "figure.dpi": 110, "savefig.dpi": 160,
    })


def _style(key: str) -> dict:
    return STYLE.get(key, dict(label=key, color=INK2, marker="o", ls="-"))


def _save(fig, path: str | Path | None) -> None:
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)


def _plot_series(ax, x, y, lo, hi, errors, key, label=True):
    st = _style(key)
    x, y, lo, hi, errors = (np.asarray(v, float) for v in (x, y, lo, hi, errors))
    nz = errors > 0
    kw = dict(color=st["color"], marker=st["marker"], ls=st["ls"], zorder=3,
              mfc=SURFACE if st.get("hollow") else st["color"], mec=st["color"], mew=1.6)
    if nz.any():
        ax.errorbar(x[nz], y[nz], yerr=[y[nz] - lo[nz], hi[nz] - y[nz]], capsize=2.5, elinewidth=1.0,
                    label=st["label"] if label else None, **kw)
    if (~nz).any():                                   # zero failures: show the upper bound
        ax.plot(x[~nz], hi[~nz], ls="none", marker="v", mfc="none", mec=st["color"], mew=1.6,
                ms=7, zorder=3, label=None if nz.any() or not label else st["label"])


def plot_ler_vs_p(df: pd.DataFrame, decoders: Sequence[str], path: str | Path | None, *,
                  title: str, distances: Sequence[int] | None = None,
                  xlabel: str = "physical error rate p", ylabel: str = "logical error rate (per shot)",
                  ymin: float | None = None):
    """Primary figure: LER vs p, one panel per distance, one line per decoder."""
    apply_style()
    ds = list(distances) if distances else sorted(df["d"].unique())
    fig, axes = plt.subplots(1, len(ds), figsize=(4.4 * len(ds), 4.0), sharey=True, squeeze=False)
    for ax, d in zip(axes[0], ds):
        sub = df[df["d"] == d]
        for key in decoders:
            s = sub[sub["decoder"] == key].sort_values("p")
            if len(s):
                _plot_series(ax, s["p"], s["ler"], s["ler_lo"], s["ler_hi"], s["errors"], key,
                             label=(d == ds[0]))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"distance d = {d}")
        ax.set_xlabel(xlabel)
        if ymin:
            ax.set_ylim(bottom=ymin)
    axes[0][0].set_ylabel(ylabel)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 3), bbox_to_anchor=(0.5, -0.12))
    fig.suptitle(title, y=1.03, fontsize=12, fontweight="bold")
    fig.text(0.5, -0.17 if len(l) > 3 else -0.10,
             "error bars: 95% Wilson intervals; hollow v = zero failures observed (95% upper bound)",
             ha="center", fontsize=8, color=INK2)
    _save(fig, path)
    return fig


def plot_advantage(df: pd.DataFrame, path: str | Path | None, *, xcol: str, xlabel: str, title: str,
                   group: str = "d", logx: bool = False, note: str = ""):
    """NN advantage ``LER_MWPM - LER_NN`` (paired difference, 95 % CI) vs ``xcol``."""
    apply_style()
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.axhline(0, color=INK2, lw=1.0, zorder=1)
    for g, sub in sorted(df.groupby(group)):
        sub = sub.sort_values(xcol)
        color = D_RAMP.get(g, BLUE) if group == "d" else BLUE
        ax.errorbar(sub[xcol], sub["adv"], yerr=[sub["adv"] - sub["adv_lo"], sub["adv_hi"] - sub["adv"]],
                    color=color, marker="o", capsize=2.5, elinewidth=1.0, mfc=color,
                    label=f"d = {g}" if group == "d" else str(g), zorder=3)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("NN advantage  =  LER(best MWPM) − LER(NN)")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.text(0.5, -0.04, "above 0: the network decodes better.  error bars: 95% paired CI on shared test shots"
             + (f".  {note}" if note else ""), ha="center", fontsize=8, color=INK2)
    _save(fig, path)
    return fig


def plot_e1(df: pd.DataFrame, slopes: pd.DataFrame, path: str | Path | None):
    """E1 sanity check: MWPM LER vs p per distance (with threshold crossing) and the
    measured low-p slope against the analytic expectation (d+1)/2."""
    apply_style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 4.0), gridspec_kw=dict(width_ratios=[1.5, 1]))
    for d, sub in sorted(df.groupby("d")):
        sub = sub.sort_values("p")
        nz = sub["errors"] > 0
        c = D_RAMP.get(d, BLUE)
        a.errorbar(sub["p"][nz], sub["ler"][nz],
                   yerr=[sub["ler"][nz] - sub["ler_lo"][nz], sub["ler_hi"][nz] - sub["ler"][nz]],
                   color=c, marker="o", capsize=2.5, elinewidth=1.0, label=f"d = {d}", zorder=3)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("physical error rate p")
    a.set_ylabel("MWPM logical error rate")
    a.set_title("E1: LER falls with d below threshold")
    a.legend(loc="lower right")
    sl = slopes.sort_values("d")
    xs = np.arange(len(sl))
    b.bar(xs - 0.18, sl["slope_measured"], width=0.34, color=BLUE, label="measured slope")
    b.bar(xs + 0.18, sl["slope_expected"], width=0.34, color=AXIS, label="(d+1)/2")
    b.set_xticks(xs)
    b.set_xticklabels([f"d={d}" for d in sl["d"]])
    b.set_ylabel("slope of log LER vs log p")
    b.set_title("low-p scaling  LER ~ p^((d+1)/2)")
    b.legend(loc="upper left")
    _save(fig, path)
    return fig


def plot_hardware(rows: pd.DataFrame, path: str | Path | None, *, title: str,
                  bare_qubit: float | None = None, subtitle: str = ""):
    """Simulation-vs-hardware gap: dot-and-whisker plot (source x decoder, log axis).

    ``rows`` needs columns ``source``, ``decoder``, ``errors``, ``shots``; whiskers are
    95 % Wilson intervals; a hollow ``v`` is the upper bound of a zero-failure point.
    """
    apply_style()
    sources = list(dict.fromkeys(rows["source"]))
    decs = list(dict.fromkeys(rows["decoder"]))
    fig, ax = plt.subplots(figsize=(1.9 * len(sources) + 3.6, 4.4))
    w = 0.22
    lows, highs = [], []
    for j, dec in enumerate(decs):
        st = _style(dec)
        xs, ys, los, his, zero_x, zero_y = [], [], [], [], [], []
        for i, src in enumerate(sources):
            r = rows[(rows["source"] == src) & (rows["decoder"] == dec)]
            if r.empty:
                continue
            k, n = int(r["errors"].iloc[0]), int(r["shots"].iloc[0])
            lo, hi = wilson_interval(k, n)
            x = i + (j - (len(decs) - 1) / 2) * w
            if k > 0:
                xs.append(x), ys.append(k / n), los.append(k / n - lo), his.append(hi - k / n)
                lows.append(lo)
            else:
                zero_x.append(x), zero_y.append(hi)
            highs.append(hi)
        if xs:
            ax.errorbar(xs, ys, yerr=[los, his], color=st["color"], marker=st["marker"], ls="none",
                        capsize=3, elinewidth=1.3, mfc=st["color"], mec=st["color"], zorder=3, label=st["label"])
        if zero_x:
            ax.plot(zero_x, zero_y, ls="none", marker="v", mfc="none", mec=st["color"], mew=1.6, zorder=3,
                    label=None if xs else st["label"])
    ax.set_yscale("log")
    top = max(highs + ([bare_qubit] if bare_qubit else [])) * 2.0
    bottom = min(lows) / 2.5 if lows else top / 1000
    ax.set_ylim(bottom, top)
    if bare_qubit is not None:
        ax.axhline(bare_qubit, color=INK2, lw=1.2, zorder=2)
        ax.text(-0.45, bare_qubit * 1.12, "unencoded qubit (rough estimate)", va="bottom", ha="left",
                fontsize=8, color=INK2)
    ax.set_xlim(-0.55, len(sources) - 0.45)
    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels([s.replace(" | ", "\n") for s in sources])
    ax.set_ylabel("logical error rate (per shot)")
    ax.set_title(title, fontsize=10)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), bbox_to_anchor=(0.5, -0.02))
    if subtitle:
        fig.text(0.5, -0.10, subtitle, ha="center", fontsize=8, color=INK2, wrap=True)
    _save(fig, path)
    return fig


def plot_leaderboard(rows: Sequence[Mapping], path: str | Path | None, *, title: str = "Syndrome Hunter leaderboard"):
    """Game leaderboard: logical *success* rate per player with 95 % Wilson intervals.

    ``rows``: dicts with ``player`` (decoder key), ``successes``, ``episodes``.
    """
    apply_style()
    rows = sorted(rows, key=lambda r: r["successes"] / max(r["episodes"], 1))
    fig, ax = plt.subplots(figsize=(6.4, 0.9 + 0.7 * len(rows)))
    for i, r in enumerate(rows):
        st = _style(r["player"])
        k, n = int(r["successes"]), int(r["episodes"])
        lo, hi = wilson_interval(k, n)
        ax.barh(i, k / max(n, 1), height=0.5, color=st["color"], zorder=3)
        ax.errorbar(k / max(n, 1), i, xerr=[[k / max(n, 1) - lo], [hi - k / max(n, 1)]], color=INK, capsize=3,
                    lw=1.2, zorder=4)
        ax.text(1.02, i, f"{100 * k / max(n, 1):.1f}%  ({k}/{n})", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([_style(r["player"])["label"] for r in rows])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("logical success rate")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    _save(fig, path)
    return fig


def plot_training(histories: Mapping[str, list[dict]], path: str | Path | None, *, title: str = "Training curves"):
    """Validation BCE vs step for one or several training runs."""
    apply_style()
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for name, h in histories.items():
        ax.plot([r["step"] for r in h], [r["val_loss"] for r in h], label=name, color=ORANGE)
    ax.set_xlabel("training step")
    ax.set_ylabel("validation BCE")
    ax.set_yscale("log")
    ax.set_title(title)
    if len(histories) > 1:
        ax.legend()
    _save(fig, path)
    return fig
