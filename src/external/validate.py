"""Score the pipeline on Metrica's public sample games with the official metric code.

Other games, event-derived labels -- allowed by RULES.md for validation and tuning. The sample
tracking is cleaner than the challenge match's raw tracking in one way that matters: its ball sits
at the nearest player's feet (median 0.29 m, against 1.86 m on the challenge match). Settings tuned
on it trust ball proximity far too much, so every game is made realistic first:

  noisy       the ball moved by temporally correlated noise (BALL_NOISE m per axis, correlated
              over BALL_NOISE_TAU frames), calibrated so the nearest-player distance matches the
              challenge match's -- a statistic of its tracking, not of any label
  noisy-hard  the same, plus 10 s ball gaps punched into live play, because the sample ball is
              rarely missing while live

The official baseline (nearest player, +/-50 frame majority vote) is scored alongside, so numbers
can be read relative to it.

    python src/external/validate.py
"""

import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

from metrica_sample import GAMES, load_game  # noqa: E402
from model import PARAMS  # noqa: E402
from possession_metrics import evaluate as official_evaluate  # noqa: E402
from possession_metrics import label_codes  # noqa: E402
from run import build_inputs_from, predict  # noqa: E402
from tracking import CACHE_DIR  # noqa: E402

VARIANTS = ("noisy", "noisy-hard")
BALL_NOISE = 1.25           # m per axis
BALL_NOISE_TAU = 12         # frames
HARD_GAP = 250              # frames per punched gap
HARD_DROP = 0.20            # live-ball visibility removed in the hard variant
BASELINE_WINDOW = 50


def degrade(ball, labels, seed):
    """Copy of the ball with 10 s gaps in live play until HARD_DROP of live visibility is gone."""
    ball = ball.copy()
    live = np.isin(labels, ["H", "A"]) & ~np.isnan(ball[:, 0])
    target = int(HARD_DROP * np.sum(np.isin(labels, ["H", "A"])))
    rng = np.random.RandomState(seed)
    starts = rng.permutation(np.where(live)[0])
    removed = 0
    for s in starts:
        if removed >= target:
            break
        seg = slice(s, s + HARD_GAP)
        removed += int(np.sum(~np.isnan(ball[seg, 0]) & live[seg]))
        ball[seg] = np.nan
    return ball


def ball_noise(n, seed, sigma=BALL_NOISE, tau=BALL_NOISE_TAU):
    """AR(1) noise, (n, 2), stationary standard deviation sigma, correlation time tau frames."""
    rng = np.random.RandomState(seed)
    a = np.exp(-1.0 / tau)
    shocks = rng.randn(n, 2) * sigma * np.sqrt(1 - a * a)
    out = np.empty((n, 2))
    out[0] = rng.randn(2) * sigma
    for t in range(1, n):
        out[t] = a * out[t - 1] + shocks[t]
    return out


def game_inputs(g, variant, refresh=False):
    path = os.path.join(CACHE_DIR, "sample{}_{}.pkl".format(g, variant))
    if os.path.isfile(path) and not refresh:
        with open(path, "rb") as handle:
            return pickle.load(handle)
    game = load_game(g)
    ball = game["ball"].copy()
    ball[:, :2] += ball_noise(len(ball), g)
    if variant.endswith("hard"):
        ball = degrade(ball, game["labels"], g)
    half = np.where(game["labels"] != "X", game["half"], 0).astype(np.int8)
    inputs = build_inputs_from(game["players"], ball, game["team"], half, game["sign"],
                               game["pitch"])
    inputs["labels"] = game["labels"]
    inputs["baseline"] = baseline(inputs, half)
    with open(path, "wb") as handle:
        pickle.dump(inputs, handle)
    return inputs


def baseline(inputs, half):
    """The official baseline: nearest player's team, majority vote over +/-50 frames, else D."""
    bf = inputs["bf"]
    seen = bf["visible"]
    home = seen & (bf["d_home"] <= bf["d_away"])
    away = seen & (bf["d_away"] < bf["d_home"])
    k = np.ones(2 * BASELINE_WINDOW + 1)
    h, a = np.convolve(home, k, "same"), np.convolve(away, k, "same")
    out = np.where(h + a == 0, "D", np.where(h >= a, "H", "A"))
    return np.where(half > 0, out, "X")


def score(labels, pred):
    r = official_evaluate(label_codes(labels), label_codes(pred))
    return r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]


def run(p=PARAMS, verbose=True, refresh=False):
    rows = []
    for g in GAMES:
        for v in VARIANTS:
            inputs = game_inputs(g, v, refresh)
            ours = score(inputs["labels"], predict(inputs, p))
            base = score(inputs["labels"], inputs["baseline"])
            rows.append((g, v, ours, base))
            if verbose:
                print("game {} {:<6} ours spellF1 {:.3f} macroF1 {:.3f} poss {:>3}/{:<3} | "
                      "baseline spellF1 {:.3f} macroF1 {:.3f}".format(
                          g, v, ours[0], ours[1], ours[2], ours[3], base[0], base[1]))
    mean = float(np.mean([r[2][0] for r in rows]))
    if verbose:
        print("mean spellF1 ours {:.3f}  baseline {:.3f}".format(
            mean, np.mean([r[3][0] for r in rows])))
    return mean, rows


if __name__ == "__main__":
    run(refresh="--refresh" in sys.argv)
