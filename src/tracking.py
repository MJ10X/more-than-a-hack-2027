"""Load the raw ATD into numpy arrays, cached under data/cache/ (gitignored with the rest of data/).

    players  float32 (n_frames, n_slots, 3)  x, y in metres, speed
    ball     float32 (n_frames, 3)           x, y in metres, speed
    team     int8    (n_slots,)              0 home, 1 away

Row i is video frame i (the ATD counts from 1, so the first record is row 0).
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import DATA_DIR, GAME, GAMES, require, source_frames  # noqa: E402

CACHE_DIR = os.path.join(DATA_DIR, "cache")


def slot_teams(meta_path):
    root = ET.parse(meta_path).getroot()
    home_id = next(root.iter("Team")).get("id")
    return np.array([0 if p.get("teamId") == home_id else 1 for p in root.iter("Player")],
                    dtype=np.int8)


def parse_raw(raw_path, n_slots, n_frames):
    """Every separator becomes a comma, so the whole file is one flat float array."""
    with open(raw_path) as handle:
        text = handle.read()
    text = text.replace(":", ",").replace(";", ",").replace("\n", ",")
    flat = np.array(text.rstrip(",").split(","), dtype=np.float32)
    width = 1 + 3 * n_slots + 3
    rows = flat.reshape(-1, width)
    if len(rows) != n_frames or rows[0, 0] != 1:
        raise ValueError("unexpected ATD shape {} / first id {}".format(rows.shape, rows[0, 0]))
    players = rows[:, 1:1 + 3 * n_slots].reshape(-1, n_slots, 3)
    ball = rows[:, 1 + 3 * n_slots:]
    return players, ball


def load(game=GAME, source="tactical", refresh=False):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, "{}_{}_atd.npz".format(game, source))
    if os.path.isfile(cache) and not refresh:
        data = np.load(cache)
        return data["players"], data["ball"], data["team"]

    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, source=source)
    team = slot_teams(meta_path)
    players, ball = parse_raw(raw_path, len(team), source_frames(source, game))
    length, width = GAMES[game]["pitch"]
    for arr in (players[..., 0], ball[:, 0]):
        arr *= length
    for arr in (players[..., 1], ball[:, 1]):
        arr *= width
    np.savez(cache, players=players, ball=ball, team=team)
    return players, ball, team


if __name__ == "__main__":
    players, ball, team = load(refresh=True)
    print("players", players.shape, "ball", ball.shape, "home slots", int((team == 0).sum()))
    print("ball visible {:.1f}%".format(100 * np.mean(~np.isnan(ball[:, 0]))))
