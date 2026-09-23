"""Pool scoring: converts raw stats into the fantasy points we predict.

A skater's fantasy points are goals + assists, plus a bonus point for each
shorthanded goal (SHG is a season-total column, so this is exact). Keeping
scoring behind a function so further rule changes stay a one-file edit
rather than a hunt through every model script.

NOTE: a hat-trick bonus was requested but is NOT implemented. This dataset
is season-total stats (Hockey-Reference), not per-game logs, so there is no
way to tell whether a player's goals in a season came 1-per-game or 3 in one
night. Computing it needs a per-game log (e.g. NHL API gamelogs) -- see
CLAUDE.md open question.
"""

from __future__ import annotations

import pandas as pd

TARGET = "fantasy_points"

# Stat column -> points per unit. Add PIM/HIT/BLK/etc. here if the pool
# ever moves to multi-category scoring.
SKATER_WEIGHTS = {
    "G": 1.0,
    "A": 1.0,
    "SHG": 1.0,  # bonus on top of the point already earned via G
}


# Goalies score on results and their own (rare) offense. T/O is Hockey-
# Reference's "ties plus OT/SO losses" -- ties don't exist post-2005, so it's
# the overtime/shootout-loss column. W already includes OT/SO wins. A
# shutout win is therefore worth W (2) + SO (2) = 4.
GOALIE_WEIGHTS = {
    "W": 2.0,
    "T/O": 1.0,
    "SO": 2.0,
    "G": 5.0,
    "A": 2.0,
}

# The pool's "1 team" slot scores NHL standings points: 2 per win, 1 per
# OT/SO loss.
TEAM_WEIGHTS = {
    "W": 2.0,
    "OTL": 1.0,
}


def skater_points(df: pd.DataFrame) -> pd.Series:
    """Fantasy points for each skater row."""
    total = pd.Series(0.0, index=df.index)
    for col, weight in SKATER_WEIGHTS.items():
        total += df[col].fillna(0) * weight
    return total


def goalie_points(df: pd.DataFrame) -> pd.Series:
    """Fantasy points for each goalie row."""
    total = pd.Series(0.0, index=df.index)
    for col, weight in GOALIE_WEIGHTS.items():
        total += df[col].fillna(0) * weight
    return total


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with the fantasy-points target column attached.

    Goalie rows (``pos_group == "G"``, when present) use the goalie weights;
    everything else scores as a skater.
    """
    out = df.copy()
    is_g = (out["pos_group"] == "G") if "pos_group" in out.columns else pd.Series(False, index=out.index)
    out[TARGET] = 0.0
    if (~is_g).any():
        out.loc[~is_g, TARGET] = skater_points(out[~is_g])
    if is_g.any() and "W" in out.columns:
        out.loc[is_g, TARGET] = goalie_points(out[is_g])
    return out
