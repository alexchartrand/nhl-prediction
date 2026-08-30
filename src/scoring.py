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


def skater_points(df: pd.DataFrame) -> pd.Series:
    """Fantasy points for each skater row."""
    total = pd.Series(0.0, index=df.index)
    for col, weight in SKATER_WEIGHTS.items():
        total += df[col].fillna(0) * weight
    return total


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with the fantasy-points target column attached."""
    out = df.copy()
    out[TARGET] = skater_points(out)
    return out
