"""Possession model: pseudo-labelled shape classifier + ball evidence, decoded with Viterbi.

No human labels and no ground truth go into it. Training targets are pseudo-labels read off the
tracking itself, on frames where the ball is unambiguous:

  team in possession  ball visible and in play, one player within CTRL_RADIUS of it, the nearest
                      opponent clearly further away, and that stays true for most of a second
  dead ball           ball visible and outside the pitch

A gradient-boosted classifier learns from those frames what the 22 players look like when each
side has the ball, or when the ball is dead, and is then applied everywhere, including inside
ball gaps. Its probabilities, the direct ball evidence and the Smart Tagging prior are summed as
per-frame log-scores for H / A / D, and Viterbi picks the label sequence with switching penalties,
so possessions come out as whole spells rather than flickering frames.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from features import AWAY, FPS, HOME, rolling_mean

# Pseudo-labels
CTRL_RADIUS = 1.5        # m, a player this close to the ball is treated as on it
CTRL_MARGIN = 1.5        # m, the nearest opponent must be at least this much further away
CONSISTENCY = 0.8        # share of the surrounding +/-0.5 s that must agree
N_FOLDS = 5              # contiguous time blocks, so a frame is never scored by a model that saw it
RESTART_SPEED = 6.0      # m/s, ball speed that counts as a restart kick or throw
MIN_STOPPAGE = 3 * FPS   # a stoppage lasts at least this long
MAX_STOPPAGE = 60 * FPS  # no restart seen within this -> do not guess, leave unlabelled
GLITCH_SPEED = 2.0       # m/s, players running this fast through an "out" episode -> tracking glitch
QUIET_SPEED = 0.8        # m/s, median player speed (+/-2 s) of a stoppage taken as certain
QUIET_MIN = 5 * FPS      # ... sustained this long, with the ball missing or still

# Emission weights and Viterbi costs. One dict, so a sweep can override any of them.
PARAMS = {
    "w_shape": 0.15,          # shape-model log-probabilities are highly correlated frame to frame
    "w_ctrl": 1.0,            # a player on the ball, per frame
    "ctrl_radius": 2.5,       # m, how close counts as on the ball for that evidence
    "w_out": 2.0,             # ball outside the pitch, per frame
    "w_moving": 0.5,          # ball moving in play, against D, per frame
    "moving_speed": 3.0,      # m/s, +/-0.5 s average
    "w_still": 0.5,           # ball still in play, for D, per frame
    "still_speed": 0.7,       # m/s, +/-0.5 s average
    "w_slow_missing": 0.5,    # ball not seen and players walking, per frame at zero speed
    "slow_speed": 1.2,        # m/s, median player speed (+/-2 s) below which play looks stopped
    "w_kickoff": 3.0,         # certain stoppage before a kick-off, per frame
    "w_tag": 0.02,            # Smart Tagging phase vote, per frame per tag
    "w_setpiece": 0.03,       # Smart Tagging set-piece team, per frame
    "switch_cost": 6.0,       # H <-> A
    "dead_cost": 8.0,         # live <-> D
}

STATES = np.array(["H", "A", "D"])
S_H, S_A, S_D = 0, 1, 2
# Shape-model classes, in a team's own view
C_DEAD, C_OWN, C_OPP = 0, 1, 2


def pseudo_labels(bf, graded, pre_kickoff):
    """Per-frame pseudo-label: HOME, AWAY, 2 for dead, -1 for no label."""
    d_home, d_away = bf["d_home"], bf["d_away"]
    live = bf["visible"] & ~bf["out"]
    with np.errstate(invalid="ignore"):
        home = live & (d_home < CTRL_RADIUS) & (d_away - d_home > CTRL_MARGIN)
        away = live & (d_away < CTRL_RADIUS) & (d_home - d_away > CTRL_MARGIN)
    win = FPS // 2
    kernel = np.ones(2 * win + 1) / (2 * win + 1)
    home_share = np.convolve(home, kernel, mode="same")
    away_share = np.convolve(away, kernel, mode="same")

    labels = np.full(len(d_home), -1, dtype=np.int8)
    labels[home & (home_share >= CONSISTENCY * (home_share + away_share))] = HOME
    labels[away & (away_share >= CONSISTENCY * (home_share + away_share))] = AWAY
    labels[dead_episodes(bf, graded) | quiet_stoppages(bf) | pre_kickoff] = 2
    labels[~graded] = -1
    return labels


def dead_episodes(bf, graded):
    """Frames from the ball leaving the pitch until it is visibly kicked back into play.

    A restart is the ball visible inside the pitch and moving faster than RESTART_SPEED for three
    frames running, no sooner than MIN_STOPPAGE after the ball went out -- the ball crossing back
    over the line on its own momentum is not a restart.
    """
    live = bf["visible"] & ~bf["out"]
    kick = live & (bf["ball_speed"] > RESTART_SPEED)
    kick = kick & np.roll(kick, 1) & np.roll(kick, 2)
    dead = np.zeros(len(live), dtype=bool)
    out = np.where(bf["out"] & graded)[0]
    if not len(out):
        return dead
    starts = out[np.r_[True, np.diff(out) > MIN_STOPPAGE]]
    for s in starts:
        window = kick[s + MIN_STOPPAGE:s + MAX_STOPPAGE]
        hit = np.where(window)[0]
        if not len(hit):
            continue
        e = s + MIN_STOPPAGE + hit[0]
        if np.nanmean(bf["players_speed"][s:e]) < GLITCH_SPEED:
            dead[s:e] = True
    return dead & graded


def quiet_stoppages(bf):
    """Stretches where everybody walks for QUIET_MIN or longer and the ball is missing or still.

    Live play does not stand still for five seconds, so these are stoppages for fouls, injuries,
    substitutions and the like, which never send the ball out of the pitch.
    """
    ball_speed = rolling_mean(np.where(bf["visible"], bf["ball_speed"], np.nan), FPS // 2)
    with np.errstate(invalid="ignore"):
        quiet = (bf["players_speed"] < QUIET_SPEED) & (~bf["visible"] | (ball_speed < 1.0))
    edges = np.diff(np.r_[0, quiet.astype(np.int8), 0])
    out = np.zeros(len(quiet), dtype=bool)
    for s, e in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
        if e - s >= QUIET_MIN:
            out[s:e] = True
    return out


def _view_targets(labels, own):
    """Shape-model class for the view of team `own`."""
    y = np.full(len(labels), -1, dtype=np.int8)
    y[labels == 2] = C_DEAD
    y[labels == own] = C_OWN
    y[labels == 1 - own] = C_OPP
    return y


def _new_classifier():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=0)


def shape_log_probs(f_home, f_away, labels, graded):
    """(n_frames, 3) log-probabilities for H, A, D from the symmetric shape model.

    Trained on both team views stacked. Contiguous-block cross-fitting: frames inside a block are
    predicted by the model trained without that block, so pseudo-labelled frames do not just get
    their own label back.
    """
    n = len(labels)
    y_home, y_away = _view_targets(labels, HOME), _view_targets(labels, AWAY)
    p_home = np.zeros((n, 3))
    p_away = np.zeros((n, 3))
    idx = np.where(graded)[0]
    blocks = np.array_split(idx, N_FOLDS)
    for k, block in enumerate(blocks):
        train = np.concatenate([b for j, b in enumerate(blocks) if j != k])
        train = train[(y_home[train] >= 0)]
        X = np.concatenate([f_home[train], f_away[train]])
        y = np.concatenate([y_home[train], y_away[train]])
        clf = _new_classifier().fit(X, y)
        p_home[block] = clf.predict_proba(f_home[block])
        p_away[block] = clf.predict_proba(f_away[block])
        held = block[y_home[block] >= 0]
        acc = np.mean(clf.predict(f_home[held]) == y_home[held]) if len(held) else float("nan")
        print("  fold {}: trained on {} rows, held-out pseudo-label accuracy {:.3f}".format(
            k + 1, len(y), acc))

    eps = 1e-4
    lh, la = np.log(p_home + eps), np.log(p_away + eps)
    out = np.empty((n, 3))
    out[:, S_H] = 0.5 * (lh[:, C_OWN] + la[:, C_OPP])
    out[:, S_A] = 0.5 * (lh[:, C_OPP] + la[:, C_OWN])
    out[:, S_D] = 0.5 * (lh[:, C_DEAD] + la[:, C_DEAD])
    return out


def emissions(shape_lp, bf, tag_vote, setpiece_team, pre_kickoff, p=PARAMS):
    """(n_frames, 3) per-frame log-scores for H, A, D."""
    score = p["w_shape"] * shape_lp

    live = bf["visible"] & ~bf["out"]
    d_home, d_away = bf["d_home"], bf["d_away"]
    with np.errstate(invalid="ignore"):
        nearest = np.fmin(d_home, d_away)
        on_ball = live & (nearest < p["ctrl_radius"])
        margin = np.clip(np.abs(d_home - d_away), 0, 3) / 3.0
    home_on = on_ball & (d_home < d_away)
    away_on = on_ball & (d_away <= d_home)
    # Who is nearer decides H against A only. A player beside a still ball is also what a free
    # kick or goal kick looks like, so nearness says nothing about live against dead by itself.
    half_ctrl = 0.5 * p["w_ctrl"] * margin
    score[home_on, S_H] += half_ctrl[home_on]
    score[home_on, S_A] -= half_ctrl[home_on]
    score[away_on, S_A] += half_ctrl[away_on]
    score[away_on, S_H] -= half_ctrl[away_on]

    # A ball moving in play is live; a ball sitting still in play is being set up for a restart.
    with np.errstate(invalid="ignore"):
        ball_speed = rolling_mean(np.where(live, bf["ball_speed"], np.nan), FPS // 2)
        moving = live & (ball_speed > p["moving_speed"])
        still = live & (ball_speed < p["still_speed"])
    score[moving, S_D] -= p["w_moving"]
    score[still, S_D] += p["w_still"]

    score[bf["out"], S_D] += p["w_out"]
    score[pre_kickoff, S_D] += p["w_kickoff"]
    # Players walking while the tracker has no ball: the commonest look of a stoppage.
    slow = np.clip((p["slow_speed"] - bf["players_speed"]) / p["slow_speed"], 0, 1)
    score[~bf["visible"], S_D] += p["w_slow_missing"] * slow[~bf["visible"]]

    score[:, S_H] += p["w_tag"] * tag_vote
    score[:, S_A] -= p["w_tag"] * tag_vote
    score[setpiece_team == HOME, S_H] += p["w_setpiece"]
    score[setpiece_team == AWAY, S_A] += p["w_setpiece"]
    return score


def viterbi(score, switch_cost, dead_cost):
    """Best state path under per-frame scores and fixed transition penalties."""
    n = len(score)
    trans = np.array([[0, -switch_cost, -dead_cost],
                      [-switch_cost, 0, -dead_cost],
                      [-dead_cost, -dead_cost, 0]])
    back = np.zeros((n, 3), dtype=np.int8)
    acc = score[0].copy()
    for t in range(1, n):
        cand = acc[:, None] + trans
        back[t] = np.argmax(cand, axis=0)
        acc = cand[back[t], np.arange(3)] + score[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = np.argmax(acc)
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(score, half, p=PARAMS):
    """Labels for every frame: Viterbi inside each half, X outside the match."""
    labels = np.full(len(score), "X", dtype="<U1")
    for h in (1, 2):
        idx = np.where(half == h)[0]
        labels[idx] = STATES[viterbi(score[idx], p["switch_cost"], p["dead_cost"])]
    return labels
