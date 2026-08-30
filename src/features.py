"""Feature set shared by the forward and defense models.

Rate stats (per game) rather than raw totals, so a 60-game injury-shortened
season isn't penalized relative to an 82-game one -- GP itself is kept as a
separate durability signal. FOW/FOL/CF/CA/FF/FA are dropped in favor of
their percentage forms to avoid redundancy with GP/TOI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import loading
import scoring

# Games-played floor below which a season's per-game rates are too noisy to
# use directly -- shared by training-pair construction and prediction-time
# feature lookup so both fall back to a player's last healthy season the
# same way.
MIN_GP = 10

FEATURE_COLS = [
    "Age",
    "GP",
    "ATOI",
    "G_pg",
    "A_pg",
    "PTS_pg",
    "SOG_pg",
    "BLK_pg",
    "HIT_pg",
    "PP_pts_pg",
    "SPCT",
    "FO%",
    "CF%",
    "CF% rel",
    "FF%",
    "FF% rel",
    "oZS%",
    "dZS%",
    "PDO",
    "oiSH%",
    "oiSV%",
    "TOI/60",
    "prior_fantasy_points_pg",
]


def add_prior_points_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Attach each row's own-season fantasy points per game as a feature.

    Requires ``scoring.add_target`` to have run already. Named distinctly
    from ``scoring.TARGET`` so it survives ``loading.make_training_pairs``,
    which overwrites the ``TARGET`` column with the *next* season's value.
    """
    out = df.copy()
    gp = out["GP"].replace(0, np.nan)
    out["prior_fantasy_points_pg"] = out[scoring.TARGET] / gp
    return out


def load_scored_seasons(data_dir: Path = loading.DATA_DIR) -> pd.DataFrame:
    """Every season, scored and with the prior-points-rate feature attached."""
    df = loading.load_all_seasons(data_dir)
    df = scoring.add_target(df)
    return add_prior_points_rate(df)
