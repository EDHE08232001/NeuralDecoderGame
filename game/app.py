"""Syndrome Hunter -- the playable decoder arena (Streamlit).

Run:   streamlit run game/app.py

You see the *defects* (parity checks that changed) of a noisy repetition-code memory
experiment and must say which data qubits were flipped.  Then the true error is revealed
and you are scored exactly like the decoders -- Greedy (a human-like heuristic), MWPM and
a trained neural network -- which play the very same syndromes.  Nothing is trained live:
the game only reads pre-generated syndromes (data/game_bank) and saved models (results/models).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from game import engine as eng  # noqa: E402
from game import viz  # noqa: E402
from src import plotting  # noqa: E402

st.set_page_config(page_title="Syndrome Hunter", page_icon="\U0001F50D", layout="wide")


# ------------------------------------------------------------------------ cached loaders
@st.cache_resource(show_spinner="Loading decoders ...")
def get_players(noise: str, level: float, d: int):
    return eng.load_players(noise, level, d)


@st.cache_data(show_spinner=False)
def get_pool(noise: str, level: float, d: int, n: int, seed: int, skip_trivial: bool = False):
    return eng.load_episodes(noise, level, d, n, seed, skip_trivial)


def bot_results(game: dict) -> dict[str, np.ndarray | None]:
    """Boolean success arrays of every bot on THIS game's episodes (cached in the game)."""
    if "bots" not in game:
        players = get_players(game["noise"], game["level"], game["d"])
        game["bots"] = {k: (eng.play_decoder(v, game["eps"]) if v is not None else None)
                        for k, v in players.items()}
    return game["bots"]


# --------------------------------------------------------------------------- state helpers
def new_game(noise: str, level: float, d: int, n: int, seed: int, name: str, skip: bool) -> None:
    st.session_state["gid"] = st.session_state.get("gid", 0) + 1
    st.session_state["game"] = dict(
        noise=noise, level=level, d=d, n=n, seed=seed, name=name, skip=skip,
        eps=get_pool(noise, level, d, n, seed, skip), i=0, submitted=False, outcome=None,
        human=[], finished=False)


def _qkeys(game: dict) -> list[str]:
    return [f"c_{st.session_state['gid']}_{game['i']}_{k}" for k in range(game["d"])]


def _flip_all() -> None:
    for k in _qkeys(st.session_state["game"]):
        st.session_state[k] = not st.session_state.get(k, False)


def _clear() -> None:
    for k in _qkeys(st.session_state["game"]):
        st.session_state[k] = False


def _next() -> None:
    g = st.session_state["game"]
    g["i"] += 1
    g["submitted"], g["outcome"] = False, None


def _finish() -> None:
    g = st.session_state["game"]
    g["finished"] = True
    ok = int(sum(h["success"] for h in g["human"]))
    eng.append_leaderboard(eng.Entry(g["name"], "human", g["noise"], g["level"], g["d"], ok, len(g["human"]), g["seed"]))
    for kind, succ in bot_results(g).items():
        if succ is not None:
            eng.append_leaderboard(eng.Entry(eng.PLAYER_LABEL[kind], kind, g["noise"], g["level"], g["d"],
                                             int(succ.sum()), len(succ), g["seed"]))


# ------------------------------------------------------------------------------- sidebar
st.sidebar.title("\U0001F50D Syndrome Hunter")
st.sidebar.caption("Can you decode a noisy qubit better than MWPM or a neural net?")
noise = st.sidebar.radio("Noise model", list(eng.NOISE_LABELS), format_func=eng.NOISE_LABELS.get,
                         help="Uniform = independent errors (MWPM is near-optimal). Biased/correlated = errors that "
                              "come in pairs, which matching ignores. Device = per-qubit rates derived from an IBM "
                              "backend's calibration (an approximation; Stim only supports Pauli noise).")
levels = eng.levels_for(noise)
level = st.sidebar.select_slider(eng.level_label(noise), options=list(levels),
                                 value=levels[3] if noise != "device" else levels[1],
                                 help="Higher = harder.")
d = st.sidebar.select_slider("Code distance d", options=list(eng.DISTANCES), value=5,
                             help="d data qubits, d-1 parity checks, d rounds of syndrome extraction.")
n = st.sidebar.select_slider("Episodes per game", options=list(eng.EPISODE_CHOICES), value=eng.EPISODE_CHOICES[-1])
seed = int(st.sidebar.number_input("Seed", min_value=0, value=1, step=1,
                                   help="Same seed = same syndromes for you and every decoder."))
skip = st.sidebar.toggle("Skip trivial episodes (no defects)", value=True,
                         help="Episodes without any defect have nothing to decode. Everyone is scored on the same episodes.")
name = st.sidebar.text_input("Player name", value="Player")
if st.sidebar.button("▶ New game", type="primary", width="stretch"):
    new_game(noise, level, d, n, seed, name, skip)
game = st.session_state.get("game")
if game and (game["noise"], game["level"], game["d"], game["n"], game["seed"], game["skip"]) != (noise, level, d, n, seed, skip):
    st.sidebar.info("Settings changed - press **New game** to apply them.")

players = get_players(noise, level, d)
if players["nn"] is None:
    hint = (f"python -m scripts.train --d {d} --noise device --scale {level:g}" if noise == "device" else
            f"python -m scripts.train --d {d} --noise {noise} --p {level:g}")
    st.sidebar.warning("No saved neural network for this setting. Train one with\n\n`" + hint + "`")

tab_play, tab_arena, tab_why, tab_about = st.tabs(["\U0001F3AE Play", "\U0001F3C6 Arena / leaderboard",
                                                 "\U0001F9E0 Why decoders?", "ℹ️ About"])

# ---------------------------------------------------------------------------------- play
with tab_play:
    if not game:
        st.markdown("### Welcome, Syndrome Hunter")
        st.write("A qubit stored in a repetition code is hit by noise. The **orange circles** are *defects*: parity "
                 "checks that changed. Decide **which data qubits were flipped**, tick them, and press **Submit**. "
                 "Then the truth is revealed and you are scored exactly like the decoders.")
        st.info("Pick a noise model and difficulty in the sidebar, then press **▶ New game**.")
    else:
        g = game
        eps: eng.Episodes = g["eps"]
        i = min(g["i"], len(eps) - 1)
        done = len(g["human"])
        score = sum(h["success"] for h in g["human"])
        head = st.columns([3, 1, 1])
        head[0].markdown(f"#### Episode {i + 1} / {len(eps)}  —  {eng.NOISE_LABELS[g['noise']].split(' (')[0]}, "
                         f"d={g['d']}, {eng.level_label(g['noise']).split(' (')[0]} = {g['level']:g}")
        head[1].metric("Your score", f"{score} / {done}")
        head[2].progress(done / len(eps), text=f"{done}/{len(eps)} played")

        if g["finished"]:
            st.success(f"Game over! You decoded **{score} of {len(eps)}** episodes correctly "
                       f"({100 * score / len(eps):.0f}%). See the **Arena** tab for the comparison with the decoders.")
        else:
            det, obs = eps.detectors[i], int(eps.observables[i])
            keys = _qkeys(g)
            for k in keys:
                st.session_state.setdefault(k, False)
            left, right = st.columns([3, 2])
            with right:
                st.markdown("**Which data qubits were flipped?**")
                cols = st.columns(g["d"])
                corr = np.array([int(cols[k].checkbox(f"q{k}", key=keys[k], disabled=g["submitted"]))
                                 for k in range(g["d"])], dtype=np.uint8)
                s_final = eng.final_syndrome(det, g["d"], g["d"])
                residual = s_final ^ eng.syndrome_of(corr)
                if not g["submitted"]:
                    b1, b2 = st.columns(2)
                    b1.button("⇅ Flip all qubits", on_click=_flip_all, width="stretch",
                              help="Complement your pattern - the other way to clear the same defects.")
                    b2.button("Clear", on_click=_clear, width="stretch")
                    if residual.any():
                        st.warning(f"Your correction leaves **{int(residual.sum())}** defect(s) at the readout "
                                   "(red diamonds). A valid answer must clear every defect.")
                    else:
                        st.info("All readout defects cleared. There are exactly **two** ways to do that "
                                "(they differ by flipping every qubit) - history decides which one is right.")
                    if st.button("Submit correction", type="primary", width="stretch"):
                        g["outcome"] = eng.score_correction(det, obs, corr, g["d"], g["d"])
                        g["submitted"] = True
                        g["human"].append(dict(ep=i, success=g["outcome"].success, correction=corr.copy(),
                                               bucket=eng.chain_bucket(g["outcome"].true_error)))
                        st.rerun()
                else:
                    out: eng.Outcome = g["outcome"]
                    (st.success if out.success else st.error)(
                        "✅ Correct!" if out.success else
                        f"❌ Not quite — {out.reason}. "
                        f"The true flip pattern was `{''.join(map(str, out.true_error))}`.")
                    bots = []
                    for kind in ("greedy", "mwpm", "nn"):
                        dec = players[kind]
                        if dec is None:
                            continue
                        pred = int(dec.predict(det[None, :])[0])
                        c = eng.correction_from_prediction(det, pred, g["d"], g["d"])
                        bots.append({"player": eng.PLAYER_LABEL[kind], "their correction": "".join(map(str, c)),
                                     "result": "✅" if pred == obs else "❌"})
                    st.markdown("**The decoders on this same episode**")
                    st.dataframe(pd.DataFrame(bots), hide_index=True, width="stretch")
                    if g["i"] + 1 < len(eps):
                        st.button("Next episode ▶", on_click=_next, type="primary", width="stretch")
                    else:
                        st.button("Finish game \U0001F3C1", on_click=_finish, type="primary", width="stretch")
            with left:
                out = g["outcome"] if g["submitted"] else None
                fig = viz.defect_figure(
                    det, g["d"], g["d"], correction=corr if corr.any() else None,
                    true_err=out.true_error if out else None,
                    residual=residual if residual.any() else None)
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)
                st.caption("Rows are syndrome rounds (top = first); the last row compares the final data readout with "
                           "the last round.")

# --------------------------------------------------------------------------------- arena
with tab_arena:
    if not game:
        st.info("Start a game to see how you compare with the decoders on the same syndromes.")
    else:
        g = game
        bots = bot_results(g)
        rows = []
        if g["human"]:
            rows.append(dict(player="human", successes=int(sum(h["success"] for h in g["human"])), episodes=len(g["human"])))
        for kind, succ in bots.items():
            if succ is not None:
                # bots play all episodes; to compare like with like, restrict to those the human played
                sel = [h["ep"] for h in g["human"]] or list(range(len(succ)))
                rows.append(dict(player=kind, successes=int(succ[sel].sum()), episodes=len(sel)))
        st.markdown(f"#### Same syndromes, four players  (seed {g['seed']}, {len(rows) and rows[0]['episodes']} episodes)")
        fig = plotting.plot_leaderboard(rows, None, title="Logical success rate on your episodes")
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
        tbl = pd.DataFrame([dict(Player=eng.PLAYER_LABEL[r["player"]], Successes=r["successes"], Episodes=r["episodes"],
                                 **{"Success rate": f"{100 * r['successes'] / max(r['episodes'], 1):.1f}%"}) for r in rows])
        st.dataframe(tbl.sort_values("Success rate", ascending=False), hide_index=True, width="stretch")
        if not g["human"]:
            st.caption("Play some episodes first - the bots are scored on the episodes you have played.")
        board = [e for e in eng.read_leaderboard() if (e.noise, e.level, e.d) == (g["noise"], g["level"], g["d"])]
        if board:
            st.markdown("**All finished games on this machine for this setting**")
            st.dataframe(pd.DataFrame([dict(Player=e.player, Kind=e.kind, Success=f"{100 * e.rate:.0f}%",
                                            Episodes=e.episodes, Seed=e.seed, When=e.when) for e in
                                       sorted(board, key=lambda e: -e.rate)]), hide_index=True, width="stretch")

# ------------------------------------------------------------------------------------ why
with tab_why:
    st.markdown("### Why do we need decoders?")
    st.write("Humans (and greedy rules) are good at **isolated errors**: one flipped qubit lights two neighbouring "
             "defects, so you pair them up. They fail on **chains**: two or three adjacent flips leave defects that are "
             "each *closer to an edge than to each other*, so the locally cheapest answer is globally wrong. "
             "Matching (MWPM) and a trained network look at the whole picture.")
    fig = viz.chain_example_figure(7)
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
    st.markdown("#### Success rate by error pattern")
    pool_n = 1000
    pool = get_pool(noise, level, d, pool_n, 12345, False)
    pl = get_players(noise, level, d)
    series = {}
    for kind in ("greedy", "mwpm", "nn"):
        if pl[kind] is not None:
            series[eng.PLAYER_LABEL[kind]] = eng.success_by_bucket(pool, eng.play_decoder(pl[kind], pool))
    if game and game["human"]:
        hb = {b: [0, 0] for b in eng.BUCKETS}
        for h in game["human"]:
            hb[h["bucket"]][0] += int(h["success"])
            hb[h["bucket"]][1] += 1
        series["You (this game)"] = {b: tuple(v) for b, v in hb.items()}
    df = pd.DataFrame({k: {b: (v[b][0] / v[b][1] if v[b][1] else np.nan) for b in eng.BUCKETS} for k, v in series.items()})
    st.caption(f"Bots: {pool_n} fresh episodes of the current setting.  'You': your own episodes (few per bucket - read with care).")
    st.dataframe(df.style.format("{:.0%}", na_rep="–"), width="stretch")
    counts = {b: sum(1 for det, obs in zip(pool.detectors, pool.observables)
                     if eng.chain_bucket(eng.true_error(det, int(obs), d, d)) == b) for b in eng.BUCKETS}
    st.caption("Pattern frequency in the pool: " + ", ".join(f"{b}: {c}" for b, c in counts.items()))

# ---------------------------------------------------------------------------------- about
with tab_about:
    st.markdown("""
### How it works
* **Task.** A distance-*d* repetition code (*d* data qubits, *d-1* parity checks) runs for *d* rounds under noise.
  You see the *detection events*; you say which data qubits were flipped.
* **Scoring.** Identical for everyone: success iff your correction equals the true flip pattern - i.e. it clears every
  defect **and** picks the right one of the two patterns that do (they differ by a logical flip).
* **Players.** *Greedy*: pair the closest defects first (what people instinctively do). *MWPM*: minimum-weight
  perfect matching (PyMatching) with weights from the noise model. *Neural net*: a small MLP trained offline on simulated syndromes.
* **Noise models.** Uniform; biased/correlated (Z-bias is invisible to a repetition code - the correlated pair flips are what
  matter); and "real IBM device": per-qubit rates (two-qubit gate error, readout error, T1/T2) from an IBM backend's
  calibration, converted to a **Pauli approximation** - not exact device fidelity.
* **Nothing is trained live.** The game reads pre-generated syndromes (`data/game_bank`) and saved models (`results/models`).
* **Limits.** Small codes; the neural net is a research toy, not a claim to beat state-of-the-art decoders.
""")
