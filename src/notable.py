"""Draft-day "Notes" tags computed from data already loaded by the pipeline.

Two signals, both purely local (no network): ``fragile`` (a multi-season
games-played shortfall pattern -- a proxy for injury-proneness, since the
dataset has no injury log) and ``trend`` (a fantasy-point-rate move sharp
enough, combined with age, to read as decline or a breakout). A third signal,
whether a player changed teams this offseason, needs live data and lives in
``nhl_api.py`` -- ``combine_notes`` is where all three come together into the
board's single "Notes" string.
"""

from __future__ import annotations

import pandas as pd

import loading
from features import MIN_GP

# How many of a player's last FRAGILE_LOOKBACK_SEASONS *qualifying* seasons
# (GP >= MIN_GP, i.e. an actual NHL roster player that year, not a brief
# call-up) need to look injury-shortened -- GP under FRAGILE_GP_PCT of that
# season's league-wide max GP -- to call him fragile.
FRAGILE_LOOKBACK_SEASONS = 4
FRAGILE_MIN_FLAGGED_SEASONS = 2
FRAGILE_GP_PCT = 0.75

# A player's latest qualifying season's fantasy-point rate vs. the mean of
# his earlier qualifying seasons (within TREND_LOOKBACK_SEASONS) has to move
# by this much, combined with an age cutoff, to read as decline or a
# breakout rather than normal year-to-year noise.
TREND_LOOKBACK_SEASONS = 4
DECLINE_PCT = 0.20
DECLINE_MIN_AGE = 30
RISING_PCT = 0.25
RISING_MAX_AGE = 23


def _season_year(season: str) -> int:
    return int(season.split("_")[0])


def add_notable_flags(board: pd.DataFrame, df_all: pd.DataFrame, as_of_season: str) -> pd.DataFrame:
    """Returns a copy of ``board`` with 'fragile' (bool) and 'trend'
    (str | None: 'Declining' / 'Rising' / None) columns added.

    Computed per player from ``df_all``'s multi-season history at or before
    ``as_of_season``. A player with too little history simply gets
    fragile=False, trend=None -- not an error.
    """
    seasons = sorted(df_all["season"].unique(), key=_season_year)
    if as_of_season not in seasons:
        raise ValueError(f"{as_of_season!r} not in available seasons {seasons}")
    as_of_idx = seasons.index(as_of_season)

    scored = df_all.assign(_season_max_gp=df_all.groupby("season")["GP"].transform("max"))
    board_ids = set(board[loading.ID_COL])
    history = scored[
        scored[loading.ID_COL].isin(board_ids)
        & (scored["season"].map(_season_year) <= _season_year(as_of_season))
    ]

    fragile_window = set(seasons[max(0, as_of_idx - FRAGILE_LOOKBACK_SEASONS + 1) : as_of_idx + 1])
    trend_window = set(seasons[max(0, as_of_idx - TREND_LOOKBACK_SEASONS + 1) : as_of_idx + 1])
    age_by_id = board.set_index(loading.ID_COL)["Age"]

    fragile: dict[str, bool] = {}
    trend: dict[str, str | None] = {}
    for player_id, g in history.groupby(loading.ID_COL):
        qualifying = g[g["GP"] >= MIN_GP]

        f_seasons = qualifying[qualifying["season"].isin(fragile_window)]
        shortened = (f_seasons["GP"] < f_seasons["_season_max_gp"] * FRAGILE_GP_PCT).sum()
        fragile[player_id] = bool(shortened >= FRAGILE_MIN_FLAGGED_SEASONS)

        t_seasons = qualifying[qualifying["season"].isin(trend_window)].sort_values(
            "season", key=lambda s: s.map(_season_year)
        )
        trend[player_id] = None
        if len(t_seasons) >= 2:
            latest = t_seasons.iloc[-1]["prior_fantasy_points_pg"]
            baseline = t_seasons.iloc[:-1]["prior_fantasy_points_pg"].mean()
            age = age_by_id.get(player_id)
            if pd.notna(latest) and pd.notna(baseline) and baseline > 0 and age is not None:
                pct_change = (latest - baseline) / baseline
                if pct_change <= -DECLINE_PCT and age >= DECLINE_MIN_AGE:
                    trend[player_id] = "Declining"
                elif pct_change >= RISING_PCT and age <= RISING_MAX_AGE:
                    trend[player_id] = "Rising"

    out = board.copy()
    out["fragile"] = out[loading.ID_COL].map(fragile).fillna(False)
    out["trend"] = out[loading.ID_COL].map(trend)
    return out


def combine_notes(fragile: bool, trend: str | None, team_change: bool | None) -> str:
    """Single source of truth for the 'Notes' string's wording and order.

    team_change=True means a live roster check found the player on a
    different team than his last loaded season; False/None (not sure, or no
    live data available) renders no tag -- this never asserts a change it
    isn't sure of.
    """
    tags = []
    if team_change:
        tags.append("New Team")
    if fragile:
        tags.append("Fragile")
    if isinstance(trend, str) and trend:
        tags.append(trend)
    return " • ".join(tags)
