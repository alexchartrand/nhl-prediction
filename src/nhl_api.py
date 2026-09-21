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
import time
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
# One request per team (~32) fired back-to-back was enough to trip
# api-web.nhle.com's rate limiter (confirmed live: HTTP 429 on the standings
# endpoint after a handful of full fetches within a minute) -- this delay
# keeps a single fetch well under that.
REQUEST_DELAY = 0.2

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


def current_teams(timeout: float = REQUEST_TIMEOUT) -> list[dict]:
    """The NHL's current teams (code + full name), straight from the
    standings endpoint rather than a hardcoded list -- survives a future
    relocation, rename, or expansion team with no code change. [] on any
    failure."""
    try:
        resp = requests.get(STANDINGS_URL, timeout=timeout)
        resp.raise_for_status()
        rows = resp.json()["standings"]
        return [
            {"code": row["teamAbbrev"]["default"], "name": row["teamName"]["default"]}
            for row in rows
        ]
    except Exception:
        return []


def current_team_codes(timeout: float = REQUEST_TIMEOUT) -> list[str]:
    return [team["code"] for team in current_teams(timeout)]


def fetch_current_rosters(
    timeout: float = REQUEST_TIMEOUT,
) -> tuple[dict[tuple[str, str], str | None], bool]:
    """Best-effort {(normalized_name, pos_group): team_code}, one request per
    current team, paced by REQUEST_DELAY between requests. A (name, pos_group)
    pair seen on more than one roster maps to None (ambiguous -- never
    guessed). Individual team failures are skipped; if the first
    MAX_CONSECUTIVE_FAILURES all fail, aborts early rather than burning a
    timeout per team when there's no connectivity.

    Returns (mapping, complete). complete=False means a 429 (rate limited)
    was hit partway through and the fetch was abandoned early rather than
    keep hammering an API that's already telling us to back off -- the
    mapping is then a partial snapshot (missing teams' players just aren't
    keys, which callers already treat as 'no live data', never 'no change').
    """
    team_codes = current_team_codes(timeout)
    result: dict[tuple[str, str], str | None] = {}
    seen_once: set[tuple[str, str]] = set()
    consecutive_failures = 0
    complete = bool(team_codes)

    with requests.Session() as session:
        for i, team in enumerate(team_codes):
            if i:
                time.sleep(REQUEST_DELAY)
            try:
                resp = session.get(ROSTER_URL.format(team=team), timeout=timeout)
                if resp.status_code == 429:
                    complete = False
                    break
                resp.raise_for_status()
                payload = resp.json()
            except Exception:
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES and not result:
                    complete = False
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

    return result, complete


def current_team_map(timeout: float = REQUEST_TIMEOUT) -> tuple[dict[tuple[str, str], str | None], bool]:
    """Guaranteed-safe entry point: never raises, returns ({}, False) on any
    failure including an unanticipated API response shape. Callers must treat
    an empty mapping as 'no live data available', not 'no changes' -- and the
    completeness flag as whether it's worth telling the user the check might
    be incomplete."""
    try:
        return fetch_current_rosters(timeout)
    except Exception:
        return {}, False


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


def detect_no_team(board: pd.DataFrame, team_map: dict[tuple[str, str], str | None]) -> pd.Series:
    """True where the player is on no current NHL roster (retired/unsigned).
    Only call with a complete, non-empty team_map -- a partial or empty map
    can't distinguish 'no team' from 'not fetched'. Ambiguous names (mapped to
    None) are present in the map, so they are not flagged."""
    keys = [(normalize_name(n), g) for n, g in zip(board["Player"], board["pos_group"])]
    return pd.Series([k not in team_map for k in keys], index=board.index)


def apply_live_team(df: pd.DataFrame, team_map: dict[tuple[str, str], str | None]) -> pd.DataFrame:
    """Copy of ``df`` with ``Team`` replaced by the player's current NHL team
    wherever the live roster lookup found an unambiguous match; everyone else
    keeps their last-loaded-season team. Call after detect_team_changes, which
    needs the old Team to compare against."""
    out = df.copy()
    live = [team_map.get((normalize_name(n), g)) for n, g in zip(out["Player"], out["pos_group"])]
    out["Team"] = [t if t else old for t, old in zip(live, out["Team"])]
    return out
