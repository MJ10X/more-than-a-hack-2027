"""Proxy validation on the full match, without any ground truth.

The public 5 minutes may not be used for tuning, and the full-match labels are private, so model
changes are judged here instead:

  masked   hide the ball in 10 s windows where it is actually visible, decode again, and score
           the masked run against the pseudo-labels inside those windows. This is the case that
           decides the match: 41% of graded frames have no ball.
  pseudo   agreement of the decoded labels with the confident pseudo-labels.
  stats    dead-ball share and possession count / duration, to compare with football norms
           (ball in play is typically ~55-60 of ~100 minutes, so D should be well above 25%).

    python src/diagnostics.py
"""

import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from run import build_inputs, predict, spells  # noqa: E402
from tracking import CACHE_DIR  # noqa: E402

MASK_LEN = 250          # 10 s, the median real ball gap
N_MASKS = 300
INPUTS_CACHE = os.path.join(CACHE_DIR, "inputs.pkl")


def cached_inputs(refresh=False):
    if os.path.isfile(INPUTS_CACHE) and not refresh:
        with open(INPUTS_CACHE, "rb") as handle:
            return pickle.load(handle)
    inputs = build_inputs()
    with open(INPUTS_CACHE, "wb") as handle:
        pickle.dump(inputs, handle)
    return inputs


def mask_windows(bf, half, seed=0):
    """Non-overlapping windows of MASK_LEN frames where the ball is visible throughout."""
    visible = bf["visible"] & (half > 0)
    run = np.convolve(visible, np.ones(MASK_LEN, dtype=int), mode="valid") == MASK_LEN
    candidates = np.where(run)[0]
    rng = np.random.RandomState(seed)
    rng.shuffle(candidates)
    taken = np.zeros(len(visible), dtype=bool)
    starts = []
    for s in candidates:
        if not taken[max(0, s - MASK_LEN):s + 2 * MASK_LEN].any():
            taken[s:s + MASK_LEN] = True
            starts.append(s)
            if len(starts) == N_MASKS:
                break
    return taken


def masked_bf(bf, mask):
    out = dict(bf)
    out["visible"] = bf["visible"] & ~mask
    out["out"] = bf["out"] & ~mask
    for key in ("d_home", "d_away"):
        out[key] = np.where(mask, np.nan, bf[key])
    return out


def macro_f1(ref, pred):
    f1s = []
    for c in "HAD":
        tp = np.sum((ref == c) & (pred == c))
        denom = np.sum(ref == c) + np.sum(pred == c)
        if np.sum(ref == c):
            f1s.append(2.0 * tp / denom if denom else 0.0)
    return float(np.mean(f1s))


def evaluate(inputs, p, mask=None, verbose=True):
    half = inputs["half"]
    graded = half > 0
    pred = predict(inputs, p)
    if mask is None:
        mask = mask_windows(inputs["bf"], half)
    pred_masked = predict(inputs, p, bf=masked_bf(inputs["bf"], mask))

    pseudo = inputs["pseudo"]
    lab = np.array(["H", "A", "D"])
    has = graded & (pseudo >= 0)
    agree_pseudo = float(np.mean(pred[has] == lab[pseudo[has]]))
    # Judged against the pseudo-labels, not the unmasked run: those do not depend on any decoding
    # parameter, so heavier smoothing cannot score well just by agreeing with itself.
    ref = mask & has
    agree_masked = float(np.mean(pred_masked[ref] == lab[pseudo[ref]]))
    f1_masked = macro_f1(lab[pseudo[ref]], pred_masked[ref])

    g = pred[graded]
    count, mean_s = spells(g)
    result = {"masked_acc": agree_masked, "masked_f1": f1_masked, "pseudo_acc": agree_pseudo,
              "D": float(np.mean(g == "D")), "poss": count, "dur": mean_s}
    if verbose:
        print("masked acc {masked_acc:.3f}  masked macroF1 {masked_f1:.3f}  pseudo acc "
              "{pseudo_acc:.3f}  D {D:.1%}  poss {poss}  dur {dur:.1f}s".format(**result))
    return result


if __name__ == "__main__":
    from model import PARAMS
    evaluate(cached_inputs(refresh="--refresh" in sys.argv), PARAMS)
