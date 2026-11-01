"""Data in, submission out.

    python src/run.py                       # writes evaluation/results/possession_hmm.csv
    python src/run.py --out my_team.csv
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from features import ball_features, half_index, shape_features, smart_tag_priors  # noqa: E402
from model import decode, emissions, pseudo_labels, shape_log_probs  # noqa: E402
from tracking import load  # noqa: E402

DEFAULT_OUT = os.path.join(ROOT, "evaluation", "results", "possession_hmm.csv")


def spells(labels):
    """(count, mean seconds) of maximal H / A runs."""
    change = np.r_[True, labels[1:] != labels[:-1]]
    starts = np.where(change)[0]
    lengths = np.diff(np.r_[starts, len(labels)])
    poss = np.isin(labels[starts], ["H", "A"])
    return int(poss.sum()), float(lengths[poss].mean() / 25.0) if poss.any() else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    t0 = time.time()

    players, ball, team = load()
    n = len(ball)
    half = half_index(n)
    graded = half > 0
    print("features ...")
    bf = ball_features(players, ball, team)
    f_home, f_away, _ = shape_features(players, team, half)
    tag_vote, setpiece = smart_tag_priors(n)

    labels = pseudo_labels(bf, graded)
    print("pseudo-labels on graded frames: H {}  A {}  D {}  none {}".format(
        *(int(np.sum(labels[graded] == k)) for k in (0, 1, 2, -1))))

    print("shape model ...")
    shape_lp = shape_log_probs(f_home, f_away, labels, graded)
    pred = decode(emissions(shape_lp, bf, tag_vote, setpiece), half)

    with open(args.out, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(pred))

    g = pred[graded]
    count, mean_s = spells(g)
    print("graded: H {:.1%}  A {:.1%}  D {:.1%}   {} possessions, mean {:.1f} s".format(
        np.mean(g == "H"), np.mean(g == "A"), np.mean(g == "D"), count, mean_s))
    print("wrote {} in {:.0f} s".format(os.path.relpath(args.out, ROOT), time.time() - t0))


if __name__ == "__main__":
    main()
