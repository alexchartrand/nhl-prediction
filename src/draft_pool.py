"""Live draft-day view over the static draft board: undrafted players only,
with VORP/replacement-level recomputed against what's actually left, plus an
unranked goalie listing (no model exists for goalies -- see CLAUDE.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loading
import rank


def undrafted_board(board: pd.DataFrame, drafted_ids: set[str]) -> pd.DataFrame:
    """``board`` (from ``rank.build_draft_board()``) minus drafted players,
    with pos_rank/VORP/replacement_level recomputed on who's left -- so
    replacement level shifts as a position gets drafted down."""
    remaining = board[~board["player_id"].isin(drafted_ids)].copy()
    remaining = remaining.drop(columns=["pos_rank", "VORP", "replacement_level"], errors="ignore")
    return rank.add_vorp(remaining).sort_values("VORP", ascending=False).reset_index(drop=True)


def goalie_pool(df_all: pd.DataFrame, as_of_season: str = rank.LATEST_SEASON) -> pd.DataFrame:
    """One row per current goalie: name/team/GP only -- no predicted_points
    or VORP, since goalie stats/scoring aren't modeled yet. min_gp=1 (rather
    than features.MIN_GP) so a backup goalie is still listed and pickable."""
    goalies = loading.latest_healthy_row(df_all, as_of_season, min_gp=1, max_seasons_back=rank.MAX_SEASONS_BACK)
    goalies = goalies[goalies["pos_group"] == "G"]
    return (
        goalies[["player_id", "Player", "Team", "GP", "feature_season", "seasons_back"]]
        .sort_values("Player")
        .reset_index(drop=True)
    )
