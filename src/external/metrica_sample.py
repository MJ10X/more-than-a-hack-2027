"""Metrica's public sample games as a validation set: tracking in, H/A/D labels from event data.

These are OTHER games (anonymised, not Barcelona - Athletic), with human-annotated event data.
RULES.md allows labelled data about other games for training and validation; the source is public
(https://github.com/metrica-sports/sample-data), fetched by download_metrica_sample.sh.

Labels follow CHALLENGE.md's possession rules, read off the events:

  possession  the team of the last PASS / RECOVERY / SET PIECE / SHOT / BALL LOST -- events where
              that team had the ball under control. Challenges and loose balls do not change it.
  dead        from the ball leaving the pitch (BALL OUT end frame, or the end of a shot that went
              out or in), or from a foul (FAULT RECEIVED), until the next event -- the restart.
  ungraded    before each period's first event, and after its last.

Event data is not the same as the corrected tracking the challenge ground truth comes from, so
transition frames differ by a few frames from what the official labels would say. It is used to
compare model variants against each other, not to predict the leaderboard number.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
from tracking import DATA_DIR  # noqa: E402

SAMPLE_DIR = os.path.join(DATA_DIR, "external", "metrica-sample")
PITCH = (105.0, 68.0)
FPS = 25
GAMES = (1, 2)

POSSESSION_TYPES = {"PASS", "RECOVERY", "SET PIECE", "SHOT", "BALL LOST"}
# Events that can happen while the ball is dead without restarting play.
DEAD_TIME_TYPES = {"CARD", "FAULT RECEIVED", "BALL OUT"}
TAIL = 2 * FPS          # frames after a period's last event still graded
SPEED_STEP = 2          # frames either side for player speed


def _read_team(path):
    """(frames, periods, xy) with xy shaped (n_frames, n_players, 2), plus the ball."""
    df = pd.read_csv(path, skiprows=2)
    names = [c for c in df.columns if c.startswith("Player")]
    xy = np.stack([df[[n, df.columns[df.columns.get_loc(n) + 1]]].to_numpy(np.float32)
                   for n in names], axis=1)
    ball = df[["Ball", df.columns[df.columns.get_loc("Ball") + 1]]].to_numpy(np.float32)
    return df["Frame"].to_numpy(), df["Period"].to_numpy(), xy, ball


def _speed(xy):
    """m/s from a centred difference; NaN where a position is missing."""
    k = SPEED_STEP
    d = np.full(xy.shape[:2], np.nan, dtype=np.float32)
    d[k:-k] = np.linalg.norm(xy[2 * k:] - xy[:-2 * k], axis=2) * FPS / (2.0 * k)
    return d


def labels_from_events(events, n_frames, period):
    """H / A / D / X per frame (index = Frame - 1)."""
    labels = np.full(n_frames, "X", dtype="<U1")
    events = events.sort_values("Start Frame", kind="stable").reset_index(drop=True)
    for p in np.unique(events["Period"]):
        ev = events[events["Period"] == p].reset_index(drop=True)
        first = int(ev["Start Frame"].min()) - 1
        last = min(int(np.max(np.where(period == p)[0])) + 1,
                   int(ev[["Start Frame", "End Frame"]].max().max()) + TAIL)

        state = np.full(n_frames, "", dtype="<U1")
        owner = None
        dead_from = None
        for i, row in ev.iterrows():
            s = int(row["Start Frame"]) - 1
            kind = row["Type"]
            sub = row["Subtype"] if isinstance(row["Subtype"], str) else ""
            team = "H" if row["Team"] == "Home" else "A"

            if dead_from is not None and kind not in DEAD_TIME_TYPES:
                state[dead_from:s] = "D"      # the restart ends the stoppage
                dead_from = None
            if kind in POSSESSION_TYPES:
                owner = team
                state[s] = owner
            if kind == "BALL OUT":
                dead_from = int(row["End Frame"]) - 1
            elif kind == "FAULT RECEIVED":
                dead_from = s
            elif kind == "SHOT" and ("GOAL" in sub or sub.endswith("OUT")) and "WOODWORK" not in sub:
                dead_from = int(row["End Frame"]) - 1
        if dead_from is not None:
            state[dead_from:last] = "D"

        # Forward-fill possession between events; D stays D.
        current = ""
        for f in range(first, last):
            if state[f] == "D":
                labels[f] = "D"
                current = ""
                continue
            if state[f]:
                current = state[f]
            # After a stoppage, nobody has the ball until the restart event names a team.
            labels[f] = current or "X"
    return labels


def load_game(g):
    """Everything the pipeline needs for one sample game, plus its labels."""
    folder = os.path.join(SAMPLE_DIR, "Sample_Game_{}".format(g))
    stem = os.path.join(folder, "Sample_Game_{}_".format(g))
    frames, period, home_xy, ball = _read_team(stem + "RawTrackingData_Home_Team.csv")
    frames_a, _, away_xy, _ = _read_team(stem + "RawTrackingData_Away_Team.csv")
    if not np.array_equal(frames, frames_a) or frames[0] != 1:
        raise ValueError("game {}: home and away tracking are not frame-aligned".format(g))

    length, width = PITCH
    xy = np.concatenate([home_xy, away_xy], axis=1)
    xy[..., 0] *= length
    xy[..., 1] *= width
    ball = ball.copy()
    ball[:, 0] *= length
    ball[:, 1] *= width
    players = np.concatenate([xy, _speed(xy)[..., None]], axis=2)
    ball3 = np.concatenate([ball, np.full((len(ball), 1), np.nan, np.float32)], axis=1)
    team = np.r_[np.zeros(home_xy.shape[1]), np.ones(away_xy.shape[1])].astype(np.int8)

    half = period.astype(np.int8)
    # Home attacks away from wherever its players stand on average in that period.
    sign = np.ones(len(half))
    for p in np.unique(half):
        home_x = np.nanmean(home_xy[half == p, :, 0])
        sign[half == p] = 1.0 if home_x < 0.5 else -1.0

    events = pd.read_csv(stem + "RawEventsData.csv")
    labels = labels_from_events(events, len(frames), period)
    return {"players": players, "ball": ball3, "team": team, "half": half, "sign": sign,
            "labels": labels, "pitch": PITCH}


if __name__ == "__main__":
    for g in GAMES:
        game = load_game(g)
        lab = game["labels"]
        graded = lab != "X"
        vis = ~np.isnan(game["ball"][:, 0])
        print("game {}: {} frames, graded {}  H {:.1%} A {:.1%} D {:.1%}".format(
            g, len(lab), graded.sum(), *(np.mean(lab[graded] == c) for c in "HAD")))
        print("  ball visible: live {:.1%}  dead {:.1%}".format(
            np.mean(vis[np.isin(lab, ["H", "A"])]), np.mean(vis[lab == "D"])))
        change = np.r_[True, lab[1:] != lab[:-1]]
        starts = np.where(change & np.isin(lab, ["H", "A"]))[0]
        print("  possessions {}".format(len(starts)))
