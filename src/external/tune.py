"""Tune the decoding parameters on the sample games (other matches), never on the challenge match.

Coordinate ascent over a coarse multiplicative grid, maximising mean spellF1 across every sample
game and variant (see validate.py: the sample ball is noised to match the challenge match's
tracking first). Coarse on purpose: two matches cannot support fine-grained choices.

    python src/external/tune.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from model import PARAMS  # noqa: E402
from validate import run  # noqa: E402

TUNED = ("switch_cost", "dead_cost", "w_shape", "w_ctrl", "w_moving", "w_still",
         "w_slow_missing", "w_out", "snap_window", "snap_radius", "ctrl_radius")
FACTORS = (0.5, 0.7, 1.4, 2.0)
ROUNDS = 3


# Where the search starts: the hand-set values from before any tuning, with touch snapping on.
START = dict(PARAMS, w_shape=0.15, w_moving=0.5, w_still=0.5, w_slow_missing=0.5,
             switch_cost=6.0, dead_cost=8.0, snap_window=37, snap_radius=1.0)


def tune(start=START):
    best = dict(start)
    best_score, _ = run(best, verbose=False)
    print("start {:.4f}".format(best_score))
    for r in range(ROUNDS):
        improved = False
        for key in TUNED:
            for f in FACTORS:
                trial = dict(best, **{key: best[key] * f})
                s, _ = run(trial, verbose=False)
                if s > best_score + 0.002:
                    best, best_score, improved = trial, s, True
                    print("round {} {} = {:.3g}  -> {:.4f}".format(r + 1, key, trial[key], s))
        if not improved:
            break
    return best, best_score


if __name__ == "__main__":
    best, score = tune()
    print("\nbest mean spellF1 {:.4f}".format(score))
    for k in TUNED:
        print("  {:<16} {:.4g}   (start {:.4g})".format(k, best[k], START[k]))
