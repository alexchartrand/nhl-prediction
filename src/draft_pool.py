"""Live draft-day view over the static draft board: undrafted players only,
with VORP/replacement-level recomputed against what's actually left, plus an
unranked goalie listing (no model exists for goalies -- see CLAUDE.md).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loading
import nhl_api
import rank

_MULTI_TEAM = re.compile(r"^\d+TM$")


def undrafted_board(
    board: pd.DataFrame,
    drafted_ids: set[str],
    teams: int = rank.TEAMS_IN_POOL,
    roster: dict = rank.ROSTER_SLOTS,
) -> pd.DataFrame:
    """``board`` (from ``rank.build_draft_board()``) minus drafted players,
    with pos_rank/VORP/replacement_level recomputed on who's left -- so
    replacement level shifts as a position gets drafted down. ``teams``/
    ``roster`` must match what the board was built with (see draft_state
    settings) or the replacement level drifts from the one the board's own
    VORP column was computed against."""
    remaining = board[~board["player_id"].isin(drafted_ids)].copy()
    remaining = remaining.drop(columns=["pos_rank", "VORP", "replacement_level"], errors="ignore")
    return rank.add_vorp(remaining, teams=teams, roster=roster).sort_values("VORP", ascending=False).reset_index(drop=True)


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


def team_pool(df_all: pd.DataFrame, as_of_season: str = rank.LATEST_SEASON) -> pd.DataFrame:
    """One row per current NHL team (for the "1 team" roster slot -- see
    CLAUDE.md, unmodeled beyond being pickable). Live from the NHL API
    (code + full name); if unreachable, falls back to the team codes found
    in the latest loaded season so the slot stays pickable offline (no full
    names available in that case -- Code doubles as Team)."""
    teams = nhl_api.current_teams()
    if teams:
        pool = pd.DataFrame(teams).rename(columns={"code": "player_id", "name": "Team"})
    else:
        codes = df_all.loc[df_all["season"] == as_of_season, "Team"].astype(str)
        codes = sorted(c for c in codes.unique() if not _MULTI_TEAM.match(c))
        pool = pd.DataFrame({"player_id": codes, "Team": codes})
    pool["Code"] = pool["player_id"]
    return pool.sort_values("Team").reset_index(drop=True)
