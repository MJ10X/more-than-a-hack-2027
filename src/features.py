"""Per-frame features from the tactical ATD and the Smart Tagging.

Two families, because the ball is missing on ~41% of graded frames, mostly in long gaps:

  ball   -- distances from the ball to the nearest home / away player, ball in or out of the pitch.
            Only defined where the tracking sees the ball.
  shape  -- what the 22 players are doing: centroids, spread, forward movement, speed. Defined on
            every frame, and the only evidence available inside a ball gap.

Shape features are expressed per team in that team's own attacking direction, so a model can be
trained on (team, opponent) pairs and applied symmetrically to both sides.
"""

import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, half_video_blocks  # noqa: E402

HOME, AWAY = 0, 1
FPS = 25

# Ball gaps up to this length are bridged by linear interpolation.
MAX_INTERP_GAP = 12
# Margin (m) beyond the lines before the ball counts as out of play; absorbs tracking jitter.
OUT_MARGIN = 1.0

# Smart Tagging codes whose Team tag is the team in possession, and those whose Team tag is the
# defending side.
ATTACKING_CODES = {"BUILD UP", "PROGRESSION", "ATTACKING TRANSITION", "CREATING CHANCES",
                   "LONG BALL", "BALL IN FINAL THIRD", "BALL IN THE BOX"}
DEFENDING_CODES = {"DEFENDING IN DEFENSIVE THIRD", "DEFENDING IN MIDDLE THIRD",
                   "DEFENDING IN ATTACKING THIRD", "DEFENSIVE TRANSITION"}
TEAM_TAGS = {"barcelona": HOME, "atletic club": AWAY}
# A tagged kick-off interval starts ~3.4 s before the ball is played (checked against both
# half-opening kick-offs, 11185 and 113742).
KICKOFF_DELAY = 85
KICKOFF_GUARD = FPS              # stop a second short of the kick-off itself
AFTER_GOAL_DEAD = 40 * FPS       # celebration and walk back after a goal is at least this long
STILL_PLAYER = 0.5               # m/s, a player slower than this is standing
RUN_SPEED = 1.5                  # m/s, forward or backward speed that counts as running that way


def half_index(n_frames, game=GAME):
    """0 outside the match, 1 or 2 inside a half."""
    half = np.zeros(n_frames, dtype=np.int8)
    for h, first, last in half_video_blocks(game):
        half[first:last + 1] = h
    return half


def attack_sign(half):
    """+1 where home attacks towards x = length, -1 where it attacks towards x = 0.

    Barcelona attack left to right in the first half (ATD metadata, attack_direction_first_half).
    """
    return np.where(half == 2, -1.0, 1.0)


def interpolate_ball(ball, max_gap=MAX_INTERP_GAP):
    """Linear interpolation across short ball gaps only; long gaps stay NaN."""
    out = ball[:, :2].copy()
    missing = np.isnan(out[:, 0])
    edges = np.diff(np.r_[0, missing.astype(np.int8), 0])
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    for s, e in zip(starts, ends):
        if e - s > max_gap or s == 0 or e >= len(out):
            continue
        t = (np.arange(s, e) - (s - 1)) / float(e - (s - 1))
        out[s:e] = out[s - 1] + t[:, None] * (out[e] - out[s - 1])
    return out


def rolling_mean(x, half_window):
    """Centred NaN-aware moving average along axis 0."""
    if half_window <= 0:
        return x
    valid = ~np.isnan(x)
    filled = np.where(valid, x, 0.0)
    pad = [(1, 0)] + [(0, 0)] * (x.ndim - 1)
    cs = np.pad(np.cumsum(filled, axis=0), pad)
    cn = np.pad(np.cumsum(valid, axis=0), pad)
    idx = np.arange(len(x))
    lo = np.maximum(0, idx - half_window)
    hi = np.minimum(len(x), idx + half_window + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (cs[hi] - cs[lo]) / (cn[hi] - cn[lo])


def ball_features(players, ball, team, pitch=None):
    length, width = pitch or GAMES[GAME]["pitch"]
    b = interpolate_ball(ball)
    dist = np.linalg.norm(players[..., :2] - b[:, None, :], axis=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d_home = np.nanmin(dist[:, team == HOME], axis=1)
        d_away = np.nanmin(dist[:, team == AWAY], axis=1)
        players_speed = rolling_mean(np.nanmedian(players[:, :, 2], axis=1), 2 * FPS)
    visible = ~np.isnan(b[:, 0])
    out = visible & ((b[:, 0] < -OUT_MARGIN) | (b[:, 0] > length + OUT_MARGIN)
                     | (b[:, 1] < -OUT_MARGIN) | (b[:, 1] > width + OUT_MARGIN))
    step = np.linalg.norm(np.diff(b, axis=0, prepend=b[:1]), axis=1) * FPS
    return {"ball": b, "visible": visible, "out": out, "d_home": d_home, "d_away": d_away,
            "ball_speed": step, "players_speed": players_speed}


def _team_stats(players, mask, sign, length):
    """Per-frame stats of one team, x measured in that team's attacking direction."""
    xy = players[:, mask, :2]
    x = np.where(sign[:, None] > 0, xy[..., 0], length - xy[..., 0])
    y = xy[..., 1]
    speed = players[:, mask, 2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        stats = {
            "cx": np.nanmean(x, axis=1),
            "cy": np.abs(np.nanmean(y, axis=1) - 34.0),
            "sx": np.nanstd(x, axis=1),
            "sy": np.nanstd(y, axis=1),
            "xmax": np.nanmax(x, axis=1),
            "xmin": np.nanmin(x, axis=1),
            "speed": np.nanmedian(speed, axis=1),
            "speed_hi": np.nanpercentile(speed, 90, axis=1),
            "n": np.sum(~np.isnan(x), axis=1).astype(np.float32),
        }
    # Forward velocity of the centroid over one second, from smoothed centroids.
    cx = rolling_mean(stats["cx"], 6)
    stats["vx"] = (np.roll(cx, -FPS // 2) - np.roll(cx, FPS // 2))

    # Where the team stands: player counts in thirds (own attacking direction) x lanes (distance
    # from the centre line, so left and right are the same lane).
    present = ~np.isnan(x)
    lane = np.abs(y - 34.0)
    for i, (x0, x1) in enumerate(((-99, 35), (35, 70), (70, 999))):
        for j, (l0, l1) in enumerate(((0, 11), (11, 22), (22, 999))):
            with np.errstate(invalid="ignore"):
                inside = present & (x >= x0) & (x < x1) & (lane >= l0) & (lane < l1)
            stats["zone{}{}".format(i, j)] = inside.sum(axis=1).astype(np.float32)

    # Share of players running forward / back, per-player velocity over half a second.
    step = FPS // 4
    vx = (np.roll(x, -step, axis=0) - np.roll(x, step, axis=0)) * FPS / (2.0 * step)
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        n = np.maximum(present.sum(axis=1), 1)
        stats["fwd_share"] = np.sum(vx > RUN_SPEED, axis=1) / n
        stats["back_share"] = np.sum(vx < -RUN_SPEED, axis=1) / n
    return stats


def _free_space(players, own_mask, opp_mask, chunk=20000):
    """Median distance from each player of one team to the nearest opponent.

    The side in possession spreads out to get free; the side without it closes players down.
    """
    out = np.full(len(players), np.nan, dtype=np.float32)
    for s in range(0, len(players), chunk):
        own = players[s:s + chunk, own_mask, :2]
        opp = players[s:s + chunk, opp_mask, :2]
        d = np.linalg.norm(own[:, :, None, :] - opp[:, None, :, :], axis=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[s:s + chunk] = np.nanmedian(np.nanmin(d, axis=2), axis=1)
    return out


def shape_features(players, team, half, sign=None, pitch=None):
    """(features_home_view, features_away_view, names).

    Each view is [own team stats, opponent stats, pairwise], so the row for "home has the ball"
    and the row for "away has the ball" share one model.
    """
    length = (pitch or GAMES[GAME]["pitch"])[0]
    if sign is None:
        sign = attack_sign(half)
    home = _team_stats(players, team == HOME, sign, length)
    away = _team_stats(players, team == AWAY, -sign, length)
    home["free"] = _free_space(players, team == HOME, team == AWAY)
    away["free"] = _free_space(players, team == AWAY, team == HOME)
    keys = sorted(home)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        overall_speed = np.nanmedian(players[:, :, 2], axis=1)
        still_share = np.nanmean(np.where(np.isnan(players[:, :, 2]), np.nan,
                                          players[:, :, 2] < STILL_PLAYER), axis=1)
    # Speeding up or slowing down: a restart is a burst after a stretch of walking.
    speed_trend = rolling_mean(overall_speed, FPS) - rolling_mean(overall_speed, 6 * FPS)

    def view(own, opp):
        cols, names = [], []
        for k in keys:
            cols.append(own[k]); names.append("own_" + k)
            cols.append(opp[k]); names.append("opp_" + k)
        # Gap between the two blocks, in own attacking direction (opp x is mirrored).
        cols.append(own["cx"] - (length - opp["cx"])); names.append("cx_gap")
        cols.append(overall_speed); names.append("all_speed")
        cols.append(still_share); names.append("still_share")
        cols.append(speed_trend); names.append("speed_trend")
        base = np.stack(cols, axis=1).astype(np.float32)
        # Temporal context: the same features averaged over +/-1 s, +/-4 s and +/-10 s.
        windows = (("_1s", FPS), ("_4s", 4 * FPS), ("_10s", 10 * FPS))
        ctx = [rolling_mean(base, w) for _, w in windows]
        names = names + [n + suffix for suffix, _ in windows for n in names]
        return np.concatenate([base] + ctx, axis=1), names

    fh, names = view(home, away)
    fa, _ = view(away, home)
    return fh, fa, names


def smart_tag_priors(n_frames, game=GAME):
    """(team_vote, setpiece_team, pre_kickoff) per frame.

    team_vote: +1 per phase tag saying home is in possession, -1 per tag saying away is.
    setpiece_team: HOME / AWAY / -1, the side tagged as taking a set piece covering the frame.
    pre_kickoff: True on frames that are certainly dead because a kick-off is coming -- the run-up
        to each half's kick-off inside the graded half, and the celebration before each kick-off
        after a goal.
    """
    from load_smart_tagging import load as load_tags
    vote = np.zeros(n_frames, dtype=np.float32)
    setpiece = np.full(n_frames, -1, dtype=np.int8)
    pre_kickoff = np.zeros(n_frames, dtype=bool)
    half = half_index(n_frames, game)
    for ev in load_tags(game):
        code = ev["code"].strip().upper()
        kind = ev["tags"].get("Type", "").strip().lower()
        if kind.startswith("kick off"):
            kickoff = ev["start_frame"] + KICKOFF_DELAY
            if "after goal" in kind:
                first = kickoff - AFTER_GOAL_DEAD
            else:
                # From the start of the half the kick-off opens.
                first = max((f for _, f, _ in half_video_blocks(game) if f <= kickoff), default=0)
            pre_kickoff[max(0, first):kickoff - KICKOFF_GUARD] = True
        side = TEAM_TAGS.get(ev["tags"].get("Team", "").strip().lower())
        s, e = max(0, ev["start_frame"]), min(n_frames, ev["end_frame"])
        if side is None:
            continue
        if code == "SET PIECES":
            setpiece[s:e] = side
            continue
        if code in DEFENDING_CODES:
            side = 1 - side
        elif code not in ATTACKING_CODES:
            continue
        vote[s:e] += 1.0 if side == HOME else -1.0
    # Only graded frames matter.
    pre_kickoff &= half > 0
    return vote, setpiece, pre_kickoff
