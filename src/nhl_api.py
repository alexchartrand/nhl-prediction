"""Live NHL API lookup for one signal the historical CSVs can't see: whether
a player has changed teams since the last loaded season (this offseason's
trades/UFA signings). Unauthenticated, unofficial endpoints
(api-web.nhle.com) -- no key, but no guarantees either, so every function
here is built to degrade to "no data" rather than raise or hang a live
draft session.

Bridges Hockey-Reference's ``Player`` name to the NHL API's own roster data
by normalized name, keyed together with position group -- confirmed live
that name alone isn't unique (two active NHL players are both named
"Sebastian Aho": a Carolina forward and an NY Islanders defenseman), but the
roster endpoint already reports each player's position, so no second data
source is needed to disambiguate.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd
import requests

STANDINGS_URL = "https://api-web.nhle.com/v1/standings/now"
ROSTER_URL = "https://api-web.nhle.com/v1/roster/{team}/current"

# The one Hockey-Reference team code confirmed to differ from the live API's
# style (data/skaters-2025_2026.csv uses "VEG", the live API uses "VGK").
# Every other HR code already matches (checked directly against the data).
TEAM_CODE_FIXES = {"VEG": "VGK"}

# NHL API positionCode -> this project's pos_group (loading._POS_GROUP uses
# the same F/D/G buckets but a different source vocabulary, C/LW/RW vs
# C/L/R here, so kept as a separate small map rather than shared).
POS_CODE_TO_GROUP = {"C": "F", "L": "F", "R": "F", "D": "D", "G": "G"}

REQUEST_TIMEOUT = 3.0
MAX_CONSECUTIVE_FAILURES = 3

_SUFFIXES = re.compile(r"\b(jr|sr|ii|iii|iv)\.?$")
_PUNCTUATION = re.compile(r"[^\w\s]")
_WHITESPACE = re.compile(r"\s+")
_MULTI_TEAM = re.compile(r"^\d+TM$")


def normalize_name(name: str) -> str:
    """Casefold, strip accents/punctuation/suffixes -- bridges Hockey-
    Reference's 'Player' spelling to the NHL API's firstName/lastName."""
    stripped = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    stripped = _PUNCTUATION.sub(" ", stripped).casefold().strip()
    stripped = _SUFFIXES.sub("", stripped).strip()
    return _WHITESPACE.sub(" ", stripped)


def current_team_codes(timeout: float = REQUEST_TIMEOUT) -> list[str]:
    """The NHL's current team abbreviations, straight from the standings
    endpoint rather than a hardcoded list -- survives a future relocation,
    rename, or expansion team with no code change. [] on any failure."""
    try:
        resp = requests.get(STANDINGS_URL, timeout=timeout)
        resp.raise_for_status()
        rows = resp.json()["standings"]
        return [row["teamAbbrev"]["default"] for row in rows]
    except Exception:
        return []


def fetch_current_rosters(timeout: float = REQUEST_TIMEOUT) -> dict[tuple[str, str], str | None]:
    """Best-effort {(normalized_name, pos_group): team_code}, one request per
    current team. A (name, pos_group) pair seen on more than one roster maps
    to None (ambiguous -- never guessed). Individual team failures are
    skipped; if the first MAX_CONSECUTIVE_FAILURES all fail, aborts early
    rather than burning a timeout per team when there's no connectivity."""
    team_codes = current_team_codes(timeout)
    result: dict[tuple[str, str], str | None] = {}
    seen_once: set[tuple[str, str]] = set()
    consecutive_failures = 0

    with requests.Session() as session:
        for team in team_codes:
            try:
                resp = session.get(ROSTER_URL.format(team=team), timeout=timeout)
                resp.raise_for_status()
                payload = resp.json()
            except Exception:
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES and not result:
                    break
                continue

            consecutive_failures = 0
            for group_key in ("forwards", "defensemen", "goalies"):
                for player in payload.get(group_key, []):
                    pos_group = POS_CODE_TO_GROUP.get(player.get("positionCode"))
                    if pos_group is None:
                        continue
                    full_name = f"{player['firstName']['default']} {player['lastName']['default']}"
                    key = (normalize_name(full_name), pos_group)
                    if key in seen_once:
                        result[key] = None
                    else:
                        seen_once.add(key)
                        result[key] = team

    return result


def current_team_map(timeout: float = REQUEST_TIMEOUT) -> dict[tuple[str, str], str | None]:
    """Guaranteed-safe entry point: never raises, returns {} on any failure
    including an unanticipated API response shape. Callers must treat {} as
    'no live data available', not 'no changes'."""
    try:
        return fetch_current_rosters(timeout)
    except Exception:
        return {}


def detect_team_changes(board: pd.DataFrame, team_map: dict[tuple[str, str], str | None]) -> pd.Series:
    """True/False/None per board row: whether a live roster check found the
    player on a different team than his last loaded season. None (name not
    found, ambiguous match, or team_map == {}) is the safe default and is
    never rendered as a tag by notable.combine_notes."""

    def _check(row: pd.Series) -> bool | None:
        last_team = row["Team"]
        if _MULTI_TEAM.match(str(last_team)):
            # Mid-season-trade marker -- the actual team he ended last
            # season on isn't preserved (see loading._drop_traded_splits),
            # so there's nothing reliable to compare the live team against.
            return None
        key = (normalize_name(row["Player"]), row["pos_group"])
        live_team = team_map.get(key)
        if live_team is None:
            return None
        return live_team != TEAM_CODE_FIXES.get(last_team, last_team)

    return board.apply(_check, axis=1)
