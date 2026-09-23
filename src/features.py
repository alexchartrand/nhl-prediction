"""Feature set shared by the forward and defense models.

Rate stats (per game) rather than raw totals, so a 60-game injury-shortened
season isn't penalized relative to an 82-game one -- GP itself is kept as a
separate durability signal. FOW/FOL/CF/CA/FF/FA are dropped in favor of
their percentage forms to avoid redundancy with GP/TOI.

Each row also looks back at the player's two previous seasons (see
add_multi_season_features): a single season's points rate is noisy, and the
goalie model already gained from 2-season averages.
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
    # Multi-season history -- add_multi_season_features.
    "GP_share",
    "pts_pg_lag1",
    "pts_pg_lag2",
    "GP_share_lag1",
    "GP_share_lag2",
    "ATOI_lag1",
    "ATOI_lag2",
    "marcel_pts_pg",
    "n_prior_seasons",
]

# Weights of this season / 1 back / 2 back in marcel_pts_pg (Marcel-style).
MARCEL_WEIGHTS = (5, 4, 3)


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


def add_multi_season_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach each row's view of the player's previous two seasons.

    Requires ``add_prior_points_rate`` to have run. Lags are joined by
    calendar year, so ``_lag1`` is the season directly before the row's own
    season (NaN if the player didn't play in it) -- for a row picked by the
    injury fallback (``loading.latest_healthy_row``), that's relative to the
    older season the features actually come from, so the history stays
    consistent with the rest of the row.

    * ``GP_share`` -- GP / that season's league max, so the COVID-shortened
      seasons read as full rather than as ~70% availability.
    * ``pts_pg_lag{1,2}``, ``GP_share_lag{1,2}``, ``ATOI_lag{1,2}``.
    * ``marcel_pts_pg`` -- fantasy points per game over this + 2 prior
      seasons, each season weighted ``MARCEL_WEIGHTS`` x its games, so a
      short season counts for less.
    * ``n_prior_seasons`` -- NHL seasons before this one, capped at 3; NaN
      when the data doesn't reach 3 seasons back (a veteran in 2016-17 isn't
      a rookie), left to the model's median imputation.

    Missing lags stay NaN (median-imputed in the ElasticNet pipeline) --
    validated against filling them with the current season's value, which
    was slightly worse. Age² was also tried and dropped (no gain).
    Rolling holdouts, backlog.md #3: MAE improved in all 6 F/D tests.
    """
    out = df.copy()
    year = out["season"].map(loading._season_start_year)
    out["GP_share"] = out["GP"] / out.groupby("season")["GP"].transform("max")

    own = pd.DataFrame({
        loading.ID_COL: out[loading.ID_COL],
        "_year": year,
        "pts_pg": out["prior_fantasy_points_pg"],
        "GP_share": out["GP_share"],
        "ATOI": out["ATOI"],
        "_gp": out["GP"],
        "_pts": out[scoring.TARGET],
    })
    out["_year"] = year
    for k in (1, 2):
        lag = own.assign(_year=own["_year"] + k)
        out = out.merge(
            lag.rename(columns=lambda c: c if c in (loading.ID_COL, "_year") else f"{c}_lag{k}"),
            on=[loading.ID_COL, "_year"], how="left",
        )

    w0, w1, w2 = MARCEL_WEIGHTS
    pts = w0 * out[scoring.TARGET] + w1 * out["_pts_lag1"].fillna(0) + w2 * out["_pts_lag2"].fillna(0)
    gp = w0 * out["GP"] + w1 * out["_gp_lag1"].fillna(0) + w2 * out["_gp_lag2"].fillna(0)
    out["marcel_pts_pg"] = pts / gp.replace(0, np.nan)

    n_prior = out.groupby(loading.ID_COL)["_year"].rank(method="first") - 1
    observable = out["_year"] - year.min() >= 3
    out["n_prior_seasons"] = n_prior.clip(upper=3).where(observable)

    return out.drop(columns=["_year", "_gp_lag1", "_gp_lag2", "_pts_lag1", "_pts_lag2"])


def load_scored_seasons(data_dir: Path = loading.DATA_DIR) -> pd.DataFrame:
    """Every season, scored, with the prior-points-rate and multi-season
    features attached."""
    df = loading.load_all_seasons(data_dir)
    df = scoring.add_target(df)
    return add_multi_season_features(add_prior_points_rate(df))
