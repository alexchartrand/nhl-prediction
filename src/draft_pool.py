"""Live draft-day view over the static draft board: undrafted players only,
with VORP/replacement-level recomputed against what's actually left, plus an
goalie listing ranked by the goalie model (see goalies.py).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import goalies
import loading
import nhl_api
import rank

_MULTI_TEAM = re.compile(r"^\d+TM$")


def undrafted_board(
    board: pd.DataFrame,
    drafted_ids: set[str],
    teams: int,
    roster: dict,
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


def goalie_pool(goalie_df: pd.DataFrame, as_of_season: str = rank.LATEST_SEASON) -> pd.DataFrame:
    """One row per current goalie, with ``predicted_points`` from the goalie
    model (see goalies.py). ``goalie_df`` is ``goalies.load_scored_goalies()``.

    Every goalie with a game in the window is listed and pickable
    (min_gp=1), but only those with at least ``goalies.MIN_GP`` games in
    their feature season get a prediction -- a call-up's handful of starts
    is too noisy to model -- so a backup can show ``predicted_points`` NaN.
    Team is the live NHL API team where found (else last season's)."""
    listed = loading.latest_healthy_row(goalie_df, as_of_season, min_gp=1, max_seasons_back=rank.MAX_SEASONS_BACK)
    model = goalies.fit_model(goalie_df)
    predicted = goalies.predict_upcoming(model, goalie_df, as_of_season)
    listed = listed.merge(predicted[["player_id", "predicted_points"]], on="player_id", how="left")

    team_map, complete = nhl_api.current_team_map()
    listed = nhl_api.apply_live_team(listed, team_map)
    if team_map and complete:
        no_team = nhl_api.detect_no_team(listed, team_map)
        listed["Notes"] = no_team.map({True: "No Team", False: ""})
    else:
        listed["Notes"] = ""
    return listed[
        ["player_id", "Player", "Team", "pos_group", "GP", "predicted_points", "Notes", "feature_season", "seasons_back"]
    ].reset_index(drop=True)


def undrafted_goalies(pool: pd.DataFrame, drafted_ids: set[str], teams: int, slots: int) -> pd.DataFrame:
    """``goalie_pool`` minus drafted goalies, ranked by predicted points with
    VORP against the ``teams * slots``-th best goalie left (same replacement
    logic as the F/D board). Goalies without a prediction sort last, no VORP."""
    remaining = pool[~pool["player_id"].isin(drafted_ids)].copy()
    modeled = remaining[remaining["predicted_points"].notna()]
    unmodeled = remaining[remaining["predicted_points"].isna()].sort_values("Player")
    if slots >= 1 and not modeled.empty:
        modeled = rank.add_vorp(modeled, teams=teams, roster={"G": slots})
    else:
        modeled = modeled.sort_values("predicted_points", ascending=False)
    out = pd.concat([modeled, unmodeled], ignore_index=True)
    for col in ("pos_rank", "VORP", "replacement_level"):
        if col not in out.columns:
            out[col] = float("nan")
    return out


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
