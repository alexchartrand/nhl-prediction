"""Turn point predictions into a draft board via value over replacement.

Predicted points alone aren't a draft order -- a top forward and a top
defenseman aren't worth comparing on raw points, since forwards fill 9 roster
slots per team and defense fills 5. VORP puts every position on the same
scale by measuring each player against the last starter-quality player at
his position (roster slots x number of teams in the pool).

Uses ElasticNet, the model that won both position groups in src/train.py's
holdout evaluation. Refit here on *all* historical season-pairs (train and
holdout) rather than the train-only models train.py saves, since for an
actual draft board there's no reason to hold data back.

Each player's features come from ``loading.latest_healthy_row``, which falls
back to his last season with at least ``features.MIN_GP`` games if his most
recent season was injury-shortened -- so a player who missed most of
2025-26 is still ranked off his last healthy year rather than dropped or
judged on a handful of noisy games. ``seasons_back``/``feature_season`` in
the output show when that fallback kicked in.

True rookies with zero NHL history in any of the loaded seasons have no
fallback season to use and are absent from the board -- there is no stat
history to build a prediction from. That's a real gap for draft prep, not
a bug; covering it would need external prospect data (junior/AHL stats,
draft rankings), which isn't part of this dataset.

Goalies are out of scope until goalie stats are added (see CLAUDE.md). The
"1 team" roster slot isn't modeled at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loading
import nhl_api
import notable
import scoring
from features import FEATURE_COLS, MIN_GP, load_scored_seasons
from train import make_elasticnet

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "output" / "draft_board.csv"
LATEST_SEASON = "2025_2026"
TEAMS_IN_POOL = 12
ROSTER_SLOTS = {"F": 9, "D": 5}
# How far the injury fallback may reach for a live prediction (see
# loading.latest_healthy_row) -- 1 means "last season's stats if this
# season was a washout." Kept small since there's no way to tell an
# injured-but-active player from a retired one beyond recency.
MAX_SEASONS_BACK = 1


def fit_production_models(pairs: pd.DataFrame) -> dict:
    """ElasticNet per position, fit on every available season-pair."""
    models = {}
    for pos in ROSTER_SLOTS:
        sub = pairs[pairs["pos_group"] == pos]
        models[pos] = make_elasticnet().fit(sub[FEATURE_COLS], sub[scoring.TARGET])
    return models


def predict_upcoming(models: dict, df_all: pd.DataFrame, as_of_season: str = LATEST_SEASON) -> pd.DataFrame:
    """Predict next season's fantasy points, falling back to each player's
    last healthy season when ``as_of_season`` was injury-shortened."""
    df = loading.latest_healthy_row(df_all, as_of_season, MIN_GP, max_seasons_back=MAX_SEASONS_BACK)
    df = df[df["pos_group"].isin(ROSTER_SLOTS)].copy()

    df["predicted_points"] = float("nan")
    for pos, model in models.items():
        mask = df["pos_group"] == pos
        df.loc[mask, "predicted_points"] = model.predict(df.loc[mask, FEATURE_COLS])
    return df


def add_vorp(df: pd.DataFrame, teams: int = TEAMS_IN_POOL, roster: dict = ROSTER_SLOTS) -> pd.DataFrame:
    ranked = []
    for pos, slots in roster.items():
        sub = df[df["pos_group"] == pos].sort_values("predicted_points", ascending=False).reset_index(drop=True)
        cutoff = min(teams * slots, len(sub))
        replacement = sub.loc[cutoff - 1, "predicted_points"]
        sub["pos_rank"] = sub.index + 1
        sub["replacement_level"] = replacement
        sub["VORP"] = sub["predicted_points"] - replacement
        ranked.append(sub)
    return pd.concat(ranked, ignore_index=True)


def build_draft_board(fetch_live_team_changes: bool = True) -> pd.DataFrame:
    """``fetch_live_team_changes=False`` skips the live NHL API roster check
    (see nhl_api.py) for a fully deterministic, network-free board -- used
    by tests. Live data only ever adds the 'New Team' note; a failed or
    skipped fetch degrades to no team-change tags, nothing else changes."""
    df_all = load_scored_seasons()
    pairs = loading.make_training_pairs(
        df_all, feature_cols=FEATURE_COLS + ["Player", "pos_group"], min_feature_gp=MIN_GP
    )
    models = fit_production_models(pairs)
    predicted = predict_upcoming(models, df_all)
    ranked = add_vorp(predicted)
    ranked = notable.add_notable_flags(ranked, df_all, LATEST_SEASON)

    team_map = {}
    if fetch_live_team_changes:
        try:
            team_map = nhl_api.current_team_map()
        except Exception:
            team_map = {}
    ranked["team_change"] = nhl_api.detect_team_changes(ranked, team_map)
    ranked["Notes"] = [
        notable.combine_notes(f, t, c)
        for f, t, c in zip(ranked["fragile"], ranked["trend"], ranked["team_change"])
    ]

    cols = [
        loading.ID_COL, "Player", "Team", "Pos", "pos_group", "Age", "GP",
        "feature_season", "seasons_back",
        "predicted_points", "pos_rank", "VORP",
        "fragile", "trend", "team_change", "Notes",
    ]
    return ranked.sort_values("VORP", ascending=False)[cols].reset_index(drop=True)


if __name__ == "__main__":
    board = build_draft_board()
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    board.to_csv(OUTPUT_PATH, index=False)
    print(f"wrote {len(board)} players to {OUTPUT_PATH}\n")

    stale = board[board["seasons_back"] > 0]
    print(f"{len(stale)} players ranked off a season other than {LATEST_SEASON} (injury/limited GP fallback):")
    print(stale.head(15)[["Player", "feature_season", "seasons_back", "predicted_points", "VORP"]].to_string(index=False))

    pd.set_option("display.width", 120)
    print("\ntop of the board:")
    print(board.head(30).to_string(index=False))
