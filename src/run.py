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
from model import PARAMS, decode, emissions, pseudo_labels, shape_log_probs, snap_transitions  # noqa: E402
from tracking import load  # noqa: E402

DEFAULT_OUT = os.path.join(ROOT, "evaluation", "results", "possession_hmm.csv")


def build_inputs():
    """Everything decoding needs for the challenge match: tracking plus its Smart Tagging."""
    players, ball, team = load()
    half = half_index(len(ball))
    return build_inputs_from(players, ball, team, half, tags=smart_tag_priors(len(ball)))


def build_inputs_from(players, ball, team, half, sign=None, pitch=None, tags=None):
    """Ball evidence, shape-model scores, tag priors and halves for any game.

    `tags` is (team_vote, setpiece_team, pre_kickoff); a game without Smart Tagging gets none.
    """
    n = len(ball)
    graded = half > 0
    print("features ...")
    bf = ball_features(players, ball, team, pitch)
    f_home, f_away, _ = shape_features(players, team, half, sign, pitch)
    if tags is None:
        tags = (np.zeros(n, np.float32), np.full(n, -1, np.int8), np.zeros(n, bool))
    tag_vote, setpiece, pre_kickoff = tags

    labels = pseudo_labels(bf, graded, pre_kickoff)
    print("pseudo-labels on graded frames: H {}  A {}  D {}  none {}".format(
        *(int(np.sum(labels[graded] == k)) for k in (0, 1, 2, -1))))

    print("shape model ...")
    shape_lp = shape_log_probs(f_home, f_away, labels, graded)
    return {"bf": bf, "shape_lp": shape_lp, "tag_vote": tag_vote, "setpiece": setpiece,
            "pre_kickoff": pre_kickoff, "half": half, "pseudo": labels}


def predict(inputs, p=PARAMS, bf=None):
    """Labels for every frame. `bf` overrides the ball evidence (used by the masked-ball test)."""
    score = emissions(inputs["shape_lp"], bf if bf is not None else inputs["bf"],
                      inputs["tag_vote"], inputs["setpiece"], inputs["pre_kickoff"], p)
    bf = bf if bf is not None else inputs["bf"]
    labels = decode(score, inputs["half"], p)
    return snap_transitions(labels, bf, p)


def spells(labels):
    """(count, mean seconds) of maximal H / A runs."""
    change = np.r_[True, labels[1:] != labels[:-1]]
    starts = np.where(change)[0]
    lengths = np.diff(np.r_[starts, len(labels)])
    poss = np.isin(labels[starts], ["H", "A"])
    return int(poss.sum()), float(lengths[poss].mean() / 25.0) if poss.any() else 0.0


def write_submission(path, labels):
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    t0 = time.time()

    inputs = build_inputs()
    pred = predict(inputs)
    write_submission(args.out, pred)

    g = pred[inputs["half"] > 0]
    count, mean_s = spells(g)
    print("graded: H {:.1%}  A {:.1%}  D {:.1%}   {} possessions, mean {:.1f} s".format(
        np.mean(g == "H"), np.mean(g == "A"), np.mean(g == "D"), count, mean_s))
    print("wrote {} in {:.0f} s".format(os.path.relpath(args.out, ROOT), time.time() - t0))


if __name__ == "__main__":
    main()
