"""ESPN Fantasy's own season projections -- a second outside opinion next to
NHL.com's (nhl_projections.py), shown as the app's "ESPN Projection" /
"ESPN Wins" columns, and averaged with NHL.com's number for the rookies the
model can't rank (rank.add_projection_only_players).

Pulled from the JSON API behind fantasy.espn.com/hockey/players/projections
(``lm-api-reads.fantasy.espn.com``, unauthenticated, unofficial). ESPN only
projects its ~380 draft-ranked players, so a player with no ESPN number is
unranked there, not projected at zero. Like espn_injuries.py every entry
point degrades to "no data" (None) rather than raising.

Skaters get projected points (G + A -- ESPN doesn't project shorthanded
goals, so this is a hair under the pool's scoring for PK specialists);
goalies get projected wins, same unit as NHL.com's goalies.txt.
"""

from __future__ import annotations

import json
import time

import pandas as pd
import requests

URL = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/fhl/seasons/{year}/segments/0/leaguedefaults/1"
REQUEST_TIMEOUT = 15.0
# Comfortably past the ~380 players ESPN projects; they come first when
# sorted by draft rank, the rest have no projection.
PLAYER_LIMIT = 600

# ESPN defaultPositionId -> pos_group (1 C, 2 LW, 3 RW, 4 D, 5 G).
POSITION_TO_GROUP = {1: "F", 2: "F", 3: "F", 4: "D", 5: "G"}
# ESPN stat ids: 16 = points, 1 = goalie wins, 30 = games played.
STAT_POINTS, STAT_WINS, STAT_GP = "16", "1", "30"
PROJECTION_SOURCE_ID = 1  # statSourceId: 0 = actual, 1 = projected
FULL_SEASON_SPLIT = 0

PROJECTION_COLS = ["Player", "pos_group", "espn_projection", "espn_gp"]

# The F/D board and the goalie pool are rebuilt back to back -- share one fetch.
CACHE_TTL = 300.0
_CACHE: dict[int, tuple[float, pd.DataFrame]] = {}


def parse_projections(payload: dict, year: int) -> pd.DataFrame:
    """One row per projected player (see PROJECTION_COLS): points for F/D,
    wins for G. Players without a full-season projection for ``year`` are
    skipped."""
    rows = []
    for entry in payload.get("players", []):
        player = entry.get("player") or {}
        pos_group = POSITION_TO_GROUP.get(player.get("defaultPositionId"))
        name = player.get("fullName")
        if not name or pos_group is None:
            continue
        stats = next(
            (
                s.get("stats") or {}
                for s in player.get("stats") or []
                if s.get("seasonId") == year
                and s.get("statSourceId") == PROJECTION_SOURCE_ID
                and s.get("statSplitTypeId") == FULL_SEASON_SPLIT
            ),
            None,
        )
        value = (stats or {}).get(STAT_WINS if pos_group == "G" else STAT_POINTS)
        if value is None:
            continue
        rows.append({"Player": name, "pos_group": pos_group, "espn_projection": float(value), "espn_gp": stats.get(STAT_GP)})
    return pd.DataFrame(rows, columns=PROJECTION_COLS)


def fetch_projections(year: int, timeout: float = REQUEST_TIMEOUT) -> pd.DataFrame | None:
    """ESPN's projections for the season ending in ``year`` (2027 for
    2026-27), or None on any failure -- or if ESPN hasn't published any
    yet, since an empty list would read as "nobody projected"."""
    hit = _CACHE.get(year)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return hit[1]
    # Server-side filters trim the response to projections only (~1 MB
    # instead of ~30 MB with every past season's weekly splits).
    fantasy_filter = {"players": {
        "limit": PLAYER_LIMIT,
        "sortDraftRanks": {"sortPriority": 100, "sortAsc": True, "value": "STANDARD"},
        "filterStatsForSourceIds": {"value": [PROJECTION_SOURCE_ID]},
        "filterStatsForSplitTypeIds": {"value": [FULL_SEASON_SPLIT]},
        "filterStatsForExternalIds": {"value": [year]},
    }}
    try:
        resp = requests.get(
            URL.format(year=year),
            params={"view": "kona_player_info"},
            headers={"X-Fantasy-Filter": json.dumps(fantasy_filter)},
            timeout=timeout,
        )
        resp.raise_for_status()
        projections = parse_projections(resp.json(), year)
    except Exception:
        return None
    if projections.empty:
        return None
    _CACHE[year] = (time.time(), projections)
    return projections
