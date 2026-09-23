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
import espn_injuries
import loading
import nhl_api
import nhl_projections
import rank
import scoring


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


# NHL.com projects goalie *wins*; pool points also come from OT/SO losses,
# shutouts and (rarely) assists, all of which scale with wins. The league
# ratio (fantasy points per win) is pooled over recent seasons' regular
# starters -- 2.51 for 2022-23..2025-26 goalies with 40+ GP.
PPW_RECENT_SEASONS = 4
PPW_MIN_GP = 40
# A goalie's own ratio (last PPW_GOALIE_SEASONS seasons) is shrunk toward
# the league ratio as if he had PPW_PRIOR_WINS extra wins at the league
# rate. Most of the spread in individual ratios is shutout/OTL luck: 3-season
# ratios of 40+ win goalies have SD 0.12 around 2.52, and binomial noise
# alone accounts for about half of that at ~70 wins -- so a real difference
# needs ~80 wins of evidence to count for half.
PPW_GOALIE_SEASONS = 3
PPW_PRIOR_WINS = 80


def _recent_seasons(df: pd.DataFrame, as_of_season: str, n: int) -> list[str]:
    return sorted(s for s in df["season"].unique() if s <= as_of_season)[-n:]


def goalie_points_per_win(goalie_df: pd.DataFrame, as_of_season: str = rank.LATEST_SEASON) -> tuple[pd.Series, float]:
    """(fantasy points per win by player_id, league ratio) -- see the
    PPW_* constants. Goalies absent from the Series get the league ratio."""
    recent = goalie_df[goalie_df["season"].isin(_recent_seasons(goalie_df, as_of_season, PPW_RECENT_SEASONS))]
    starters = recent[recent["GP"] >= PPW_MIN_GP]
    league = starters[scoring.TARGET].sum() / starters["W"].sum()

    own = goalie_df[goalie_df["season"].isin(_recent_seasons(goalie_df, as_of_season, PPW_GOALIE_SEASONS))]
    totals = own.groupby(loading.ID_COL)[[scoring.TARGET, "W"]].sum()
    ppw = (totals[scoring.TARGET] + PPW_PRIOR_WINS * league) / (totals["W"] + PPW_PRIOR_WINS)
    return ppw, league


def league_otl_per_team_game(goalie_df: pd.DataFrame, as_of_season: str = rank.LATEST_SEASON) -> float:
    """OT/SO losses per team-game over the recent full-length seasons
    (league-wide goalie T/O totals; every OT/SO loss is charged to a goalie).
    ~0.11 in 2022-23..2025-26, i.e. ~9 per team over 82 games."""
    recent = goalie_df[goalie_df["season"].isin(_recent_seasons(goalie_df, as_of_season, PPW_RECENT_SEASONS))]
    # Every game has exactly one winning goalie, so wins = games, and each
    # game is two team-games.
    return float(recent["T/O"].sum() / (2 * recent["W"].sum()))


def goalie_pool(goalie_df: pd.DataFrame, as_of_season: str = rank.LATEST_SEASON) -> pd.DataFrame:
    """One row per goalie NHL.com projects for the season, ranked directly
    off NHL.com's projected win totals (see nhl_projections.py) rather than
    the in-repo model -- those projections bake in this season's
    starter/backup depth chart, which the historical-stats model has no way
    to see (see CLAUDE.md). Points are scaled down for games a goalie is
    expected to miss per ESPN's live injury list (espn_injuries.py). Wins are converted to fantasy points
    (``predicted_points``) with each goalie's shrunk points-per-win ratio
    (goalie_points_per_win), so goalie VORP is on the same scale as skater
    VORP.

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
    ppw, league_ppw = goalie_points_per_win(goalie_df, as_of_season)
    pool["pts_per_win"] = pool["player_id"].map(ppw).fillna(league_ppw).round(2)
    pool["predicted_points"] = (pool["projected_wins"] * pool["pts_per_win"]).round(1)
    pool, _ = espn_injuries.apply_live_injuries(pool)

    team_map, complete = nhl_api.current_team_map()
    pool = nhl_api.apply_live_team(pool, team_map)
    no_team = nhl_api.detect_no_team(pool, team_map) if (team_map and complete) else pd.Series(False, index=pool.index)
    pool["Notes"] = [
        " • ".join(tag for tag in (inj, "No Team" if nt else None) if tag)
        for inj, nt in zip(pool["injury_tag"], no_team)
    ]
    return pool[
        ["player_id", "Player", "Team", "pos_group", "GP", "projected_wins", "pts_per_win", "predicted_points",
         "games_missed", "Notes", "injury"]
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


def team_pool(goalie_df: pd.DataFrame, as_of_season: str = rank.LATEST_SEASON) -> pd.DataFrame:
    """One row per NHL team, ranked by projected standings points (the
    team slot's scoring, scoring.TEAM_WEIGHTS) built from NHL.com's projected
    win total (nhl_projections.load_team_projections) -- the pool's "1 team"
    roster slot otherwise has no model at all (see CLAUDE.md). Team is the
    live NHL API full name where available, else NHL.com's own team code.

    NHL.com's win totals leave out OT/SO losses (see teams.txt), so every
    team gets the league-average OTL count (league_otl_per_team_game, from
    ``goalie_df``) -- a constant, so it shifts the displayed points but not
    ranks or VORP. Season length comes from the projections themselves: one
    win per game, so games per team = 2 x total wins / teams (84 for
    2026-27)."""
    pool = nhl_projections.load_team_projections()
    name_by_code = {t["code"]: t["name"] for t in nhl_api.current_teams()}
    pool["Team"] = pool["Code"].map(name_by_code).fillna(pool["Code"])
    pool["player_id"] = pool["Code"]
    pool["pos_group"] = "TEAM"
    games_per_team = nhl_projections.games_per_team()
    pool["projected_otl"] = round(league_otl_per_team_game(goalie_df, as_of_season) * games_per_team, 1)
    pool["predicted_points"] = (
        scoring.TEAM_WEIGHTS["W"] * pool["projected_wins"] + scoring.TEAM_WEIGHTS["OTL"] * pool["projected_otl"]
    )
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
