"""Live draft-day view over the static draft board: undrafted players only,
with VORP/replacement-level recomputed against what's actually left, plus
goalie and team listings ranked directly off NHL.com's own projections (see
nhl_projections.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loading
import nhl_api
import nhl_projections
import rank


def undrafted_board(
    board: pd.DataFrame,
    drafted_ids: set[str],
    teams: int,
    roster: dict,
) -> pd.DataFrame:
    """``board`` (from ``rank.build_draft_board()``) minus drafted players
    (live picks and keepers), with pos_rank/VORP/replacement_level
    recomputed on who's left. The replacement level is the (open slots
    left)-th best remaining player at each position, so it only moves when
    the draft departs from projection order (a reach, a keeper, a run on a
    position) -- not simply because players are being drafted."""
    remaining = board[~board["player_id"].isin(drafted_ids)].copy()
    remaining = remaining.drop(columns=["pos_rank", "VORP", "replacement_level"], errors="ignore")
    return rank.add_vorp(
        remaining, teams=teams, roster=roster, filled=_filled_slots(board, drafted_ids)
    ).sort_values("VORP", ascending=False).reset_index(drop=True)


def _filled_slots(pool: pd.DataFrame, drafted_ids: set[str]) -> dict:
    """Roster slots already taken per pos_group, from drafted rows of ``pool``."""
    return pool.loc[pool["player_id"].isin(drafted_ids), "pos_group"].value_counts().to_dict()


def goalie_pool(goalie_df: pd.DataFrame, as_of_season: str = rank.LATEST_SEASON) -> pd.DataFrame:
    """One row per goalie NHL.com projects for the season, ranked directly
    off NHL.com's projected win totals (see nhl_projections.py) rather than
    the in-repo model -- those projections bake in this season's
    starter/backup depth chart, which the historical-stats model has no way
    to see (see CLAUDE.md).

    ``goalie_df`` (``goalies.load_scored_goalies()``) supplies GP for
    display only, joined to the projection by normalized name -- a goalie
    NHL.com projects with no Hockey-Reference history (e.g. a rookie) still
    gets a row, just without that column. Team is the live NHL API team
    where found, else NHL.com's own (which can be blank for a name-only
    "X or Y" committee line -- see nhl_projections.py)."""
    projections = nhl_projections.load_goalie_projections()
    history = loading.latest_healthy_row(goalie_df, as_of_season, min_gp=1, max_seasons_back=rank.MAX_SEASONS_BACK)

    pool = projections.assign(_key=projections["Player"].map(nhl_api.normalize_name))
    history = history.assign(_key=history["Player"].map(nhl_api.normalize_name))
    pool = pool.merge(history[["_key", "player_id", "GP"]], on="_key", how="left").drop(columns="_key")
    pool["player_id"] = pool["player_id"].fillna(
        "proj_" + pool["Player"].map(nhl_api.normalize_name).str.replace(" ", "_")
    )
    pool["pos_group"] = "G"
    pool["predicted_points"] = pool["projected_wins"]

    team_map, complete = nhl_api.current_team_map()
    pool = nhl_api.apply_live_team(pool, team_map)
    no_team = nhl_api.detect_no_team(pool, team_map) if (team_map and complete) else pd.Series(False, index=pool.index)
    pool["Notes"] = [
        " • ".join(tag for tag, on in (("Injured", inj), ("No Team", nt)) if on)
        for inj, nt in zip(pool["injured"], no_team)
    ]
    return pool[
        ["player_id", "Player", "Team", "pos_group", "GP", "predicted_points", "Notes"]
    ].reset_index(drop=True)


def undrafted_goalies(pool: pd.DataFrame, drafted_ids: set[str], teams: int, slots: int) -> pd.DataFrame:
    """``goalie_pool`` minus drafted goalies, VORP'd against the
    (open goalie slots left)-th best goalie left (same replacement logic as
    the F/D board)."""
    remaining = pool[~pool["player_id"].isin(drafted_ids)].copy()
    if slots >= 1 and not remaining.empty:
        remaining = rank.add_vorp(remaining, teams=teams, roster={"G": slots}, filled=_filled_slots(pool, drafted_ids))
    else:
        remaining = remaining.sort_values("predicted_points", ascending=False)
        for col in ("pos_rank", "VORP", "replacement_level"):
            remaining[col] = float("nan")
    return remaining.reset_index(drop=True)


def team_pool() -> pd.DataFrame:
    """One row per NHL team, ranked by NHL.com's projected win total for the
    season (nhl_projections.load_team_projections) -- the pool's "1 team"
    roster slot otherwise has no model at all (see CLAUDE.md). Team is the
    live NHL API full name where available, else NHL.com's own team code."""
    pool = nhl_projections.load_team_projections()
    name_by_code = {t["code"]: t["name"] for t in nhl_api.current_teams()}
    pool["Team"] = pool["Code"].map(name_by_code).fillna(pool["Code"])
    pool["player_id"] = pool["Code"]
    pool["pos_group"] = "TEAM"
    pool["predicted_points"] = pool["projected_wins"]
    return pool.sort_values("Team").reset_index(drop=True)


def undrafted_teams(pool: pd.DataFrame, drafted_ids: set[str], teams: int, slots: int) -> pd.DataFrame:
    """``team_pool`` minus drafted teams, VORP'd against the
    (open team slots left)-th best team left (same replacement logic as the
    F/D board)."""
    remaining = pool[~pool["player_id"].isin(drafted_ids)].copy()
    if slots >= 1 and not remaining.empty:
        remaining = rank.add_vorp(remaining, teams=teams, roster={"TEAM": slots}, filled=_filled_slots(pool, drafted_ids))
    else:
        remaining = remaining.sort_values("predicted_points", ascending=False)
        for col in ("pos_rank", "VORP", "replacement_level"):
            remaining[col] = float("nan")
    return remaining.reset_index(drop=True)
