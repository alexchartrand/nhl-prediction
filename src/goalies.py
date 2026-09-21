"""Goalie feature set and data assembly for the goalie point-prediction model.

Goalie scoring (see scoring.GOALIE_WEIGHTS) is dominated by wins and
workload, so besides quality rates (SV%, GSAA, ...) the features carry
games/starts/minutes -- the model has to learn how many games a goalie is
likely to be handed as well as how well he plays them. Rates are per game
played, like the skater features.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loading
import scoring
from train import make_elasticnet

# Same floor as the skater models: below this a goalie's rates are a
# handful of starts and too noisy to use directly.
MIN_GP = 10

GOALIE_FEATURE_COLS = [
    "Age",
    "GP",
    "GS",
    "GS_share",
    "W_pg",
    "OTL_pg",
    "SO_pg",
    "SV%",
    "GAA",
    "QS%",
    "RBS_pg",
    "GA%-",
    "GSAA_pg",
    "GPS_pg",
    "Shots_pg",
    "MIN_pg",
    "prior_fantasy_points_pg",
    "pts_pg_2y",
    "SV_2y",
    "GSAA_2y",
    "GP_2y",
]

# Latest completed season, and how far the injury fallback may reach for a
# live prediction -- same reasoning as rank.MAX_SEASONS_BACK.
LATEST_SEASON = "2025_2026"
MAX_SEASONS_BACK = 1


def add_goalie_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    gp = out["GP"].replace(0, np.nan)
    out["GS_share"] = out["GS"] / gp
    out["W_pg"] = out["W"] / gp
    out["OTL_pg"] = out["T/O"] / gp
    out["SO_pg"] = out["SO"] / gp
    out["RBS_pg"] = out["RBS"] / gp
    out["GSAA_pg"] = out["GSAA"] / gp
    out["GPS_pg"] = out["GPS"] / gp
    out["Shots_pg"] = out["Shots"] / gp
    out["MIN_pg"] = out["MIN"] / gp
    out["prior_fantasy_points_pg"] = out[scoring.TARGET] / gp

    # Goalie seasons are noisy (a .915 year is often luck), so blend in the
    # season before: the mean of this and the previous consecutive season,
    # or just this season for a goalie with no prior one. Validated against
    # single-season-only features -- lower holdout MAE (15.9 vs 16.7).
    year = out["season"].str.split("_").str[0].astype(int)
    lag = pd.DataFrame({
        loading.ID_COL: out[loading.ID_COL],
        "year": year + 1,
        "pts_lag": out["prior_fantasy_points_pg"],
        "sv_lag": out["SV%"],
        "gsaa_lag": out["GSAA_pg"],
        "gp_lag": out["GP"],
    })
    out = out.assign(year=year).merge(lag, on=[loading.ID_COL, "year"], how="left")
    for name, cur, prev in (
        ("pts_pg_2y", "prior_fantasy_points_pg", "pts_lag"),
        ("SV_2y", "SV%", "sv_lag"),
        ("GSAA_2y", "GSAA_pg", "gsaa_lag"),
        ("GP_2y", "GP", "gp_lag"),
    ):
        out[name] = out[[cur, prev]].mean(axis=1)
    return out.drop(columns=["year", "pts_lag", "sv_lag", "gsaa_lag", "gp_lag"])


def load_scored_goalies(data_dir: Path = loading.DATA_DIR) -> pd.DataFrame:
    """Every goalie season, scored, with features attached."""
    df = scoring.add_target(loading.load_all_goalie_seasons(data_dir))
    return add_goalie_features(df)


def fit_model(df_all: pd.DataFrame):
    """ElasticNet fit on every available goalie season-pair."""
    pairs = loading.make_training_pairs(
        df_all, feature_cols=GOALIE_FEATURE_COLS + ["Player", "pos_group"], min_feature_gp=MIN_GP
    )
    return make_elasticnet().fit(pairs[GOALIE_FEATURE_COLS], pairs[scoring.TARGET])


def predict_upcoming(model, df_all: pd.DataFrame, as_of_season: str = LATEST_SEASON) -> pd.DataFrame:
    """Predicted next-season fantasy points per goalie, falling back to the
    last season with ``MIN_GP`` games (at most ``MAX_SEASONS_BACK`` back)."""
    df = loading.latest_healthy_row(df_all, as_of_season, MIN_GP, max_seasons_back=MAX_SEASONS_BACK)
    df = df.copy()
    df["predicted_points"] = model.predict(df[GOALIE_FEATURE_COLS])
    return df


if __name__ == "__main__":
    from train import HOLDOUT_TARGET_SEASON, cv_predictions, eval_metrics

    df = load_scored_goalies()
    pairs = loading.make_training_pairs(
        df, feature_cols=GOALIE_FEATURE_COLS + ["Player", "pos_group"], min_feature_gp=MIN_GP
    )
    train = pairs[pairs["target_season"] != HOLDOUT_TARGET_SEASON].reset_index(drop=True)
    test = pairs[pairs["target_season"] == HOLDOUT_TARGET_SEASON]
    X, y = train[GOALIE_FEATURE_COLS], train[scoring.TARGET]
    cv = eval_metrics(y, cv_predictions(make_elasticnet, X, y, train["player_id"]))
    model = make_elasticnet().fit(X, y)
    ho = eval_metrics(test[scoring.TARGET], model.predict(test[GOALIE_FEATURE_COLS]))
    print(f"G ElasticNet: train={len(train)} rows, holdout={len(test)} rows")
    print(f"  CV      MAE {cv['MAE']:.2f}  R2 {cv['R2']:.3f}  Spearman {cv['Spearman']:.3f}")
    print(f"  holdout MAE {ho['MAE']:.2f}  R2 {ho['R2']:.3f}  Spearman {ho['Spearman']:.3f}")
