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
# On a 429 the request is retried after a cooldown rather than abandoned, so
# a rate limit slows the board build down instead of leaving it partial. The
# server's Retry-After is honored when present, else exponential backoff
# (RATE_LIMIT_COOLDOWN * 2**attempt); both capped at MAX_COOLDOWN. Each 429
# also permanently widens the pacing delay for the rest of the fetch.
RATE_LIMIT_COOLDOWN = 5.0
MAX_COOLDOWN = 60.0
DELAY_BACKOFF_STEP = 0.3
# Total time one fetch may spend in 429 cooldowns before giving up.
COOLDOWN_BUDGET = 600.0

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


def _cooldown_seconds(resp: requests.Response, attempt: int) -> float:
    retry_after = resp.headers.get("Retry-After", "")
    try:
        wait = float(retry_after)
    except ValueError:
        wait = RATE_LIMIT_COOLDOWN * 2**attempt
    return min(max(wait, 1.0), MAX_COOLDOWN)


def _get_with_cooldown(getter, url: str, timeout: float, budget: float = COOLDOWN_BUDGET) -> tuple[requests.Response, float]:
    """``getter(url, timeout=...)`` retried through 429s, sleeping a cooldown
    between attempts until ``budget`` seconds of waiting are spent. Returns
    (response, seconds_waited); the response is still a 429 if the budget ran
    out. Non-429 errors are the caller's to handle (exceptions propagate)."""
    waited = 0.0
    resp = getter(url, timeout=timeout)
    attempt = 0
    while resp.status_code == 429:
        wait = _cooldown_seconds(resp, attempt)
        if waited + wait > budget:
            break
        time.sleep(wait)
        waited += wait
        attempt += 1
        resp = getter(url, timeout=timeout)
    return resp, waited


def current_teams(timeout: float = REQUEST_TIMEOUT) -> list[dict]:
    """The NHL's current teams (code + full name), straight from the
    standings endpoint rather than a hardcoded list -- survives a future
    relocation, rename, or expansion team with no code change. [] on any
    failure."""
    try:
        resp, _ = _get_with_cooldown(requests.get, STANDINGS_URL, timeout)
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


def _parse_roster(payload: dict) -> list[tuple[tuple[str, str], str | None]]:
    """[((normalized_name, pos_group), birth_date 'YYYY-MM-DD' or None)]."""
    players = []
    for group_key in ("forwards", "defensemen", "goalies"):
        for player in payload.get(group_key, []):
            pos_group = POS_CODE_TO_GROUP.get(player.get("positionCode"))
            if pos_group is None:
                continue
            full_name = f"{player['firstName']['default']} {player['lastName']['default']}"
            players.append(((normalize_name(full_name), pos_group), player.get("birthDate")))
    return players


# team_code -> (fetched_at, [(name_key, birth_date)]) for every roster fetched so
# far in this process. Lets a rate-limited fetch resume with only the teams
# still missing instead of restarting from team 1 (which is what kept
# re-tripping the limiter), and lets the board, goalie and team views share
# one fetch instead of each hammering the API in turn.
_ROSTER_CACHE: dict[str, tuple[float, list]] = {}
ROSTER_CACHE_TTL = 1800.0


def fetch_current_rosters(
    timeout: float = REQUEST_TIMEOUT,
) -> tuple[dict[tuple[str, str], str | None], bool]:
    """Best-effort {(normalized_name, pos_group): team_code}, one request per
    current team, paced by REQUEST_DELAY between requests. A (name, pos_group)
    pair seen on more than one roster maps to None (ambiguous -- never
    guessed). Individual team failures are skipped; if the first
    MAX_CONSECUTIVE_FAILURES all fail, aborts early rather than burning a
    timeout per team when there's no connectivity.

    A 429 (rate limited) triggers a cooldown and a retry of the same team
    (see _get_with_cooldown), widens the delay between later requests, and
    keeps going for up to COOLDOWN_BUDGET seconds of waiting. Rosters already
    fetched (within ROSTER_CACHE_TTL) are reused, so a retry after a failed
    attempt only requests the teams still missing.

    Returns (mapping, complete). complete=False means some team never
    succeeded (budget exhausted or no connectivity) -- the mapping is then a
    partial snapshot (missing teams' players just aren't keys, which callers
    already treat as 'no live data', never 'no change').
    """
    team_codes = current_team_codes(timeout)
    now = time.time()
    fresh = {t: v for t, v in _ROSTER_CACHE.items() if now - v[0] < ROSTER_CACHE_TTL}
    _ROSTER_CACHE.clear()
    _ROSTER_CACHE.update(fresh)

    consecutive_failures = 0
    delay = REQUEST_DELAY
    budget = COOLDOWN_BUDGET
    made_request = False

    with requests.Session() as session:
        for team in team_codes:
            if team in _ROSTER_CACHE:
                continue
            if made_request:
                time.sleep(delay)
            made_request = True
            try:
                resp, waited = _get_with_cooldown(session.get, ROSTER_URL.format(team=team), timeout, budget)
                budget -= waited
                if waited:
                    delay += DELAY_BACKOFF_STEP
                if resp.status_code == 429:
                    break
                resp.raise_for_status()
                _ROSTER_CACHE[team] = (time.time(), _parse_roster(resp.json()))
            except Exception:
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES and not _ROSTER_CACHE:
                    break
                continue
            consecutive_failures = 0

    result: dict[tuple[str, str], str | None] = {}
    seen_once: set[tuple[str, str]] = set()
    for team in team_codes:
        for key, _ in _ROSTER_CACHE.get(team, (0, []))[1]:
            if key in seen_once:
                result[key] = None
            else:
                seen_once.add(key)
                result[key] = team
    complete = bool(team_codes) and all(t in _ROSTER_CACHE for t in team_codes)
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


def current_birth_dates(timeout: float = REQUEST_TIMEOUT) -> dict[tuple[str, str], str]:
    """{(normalized_name, pos_group): 'YYYY-MM-DD'} for every current roster
    player, from the same roster fetch as current_team_map -- cached, so
    right after it this costs one standings request, not 32 roster ones.
    Names on more than one roster are left out (ambiguous). Never raises;
    {} on any failure."""
    try:
        fetch_current_rosters(timeout)  # (re)fills _ROSTER_CACHE with current, fresh rosters
        dates: dict[tuple[str, str], str] = {}
        dupes: set[tuple[str, str]] = set()
        for _, players in _ROSTER_CACHE.values():
            for key, born in players:
                if key in dates:
                    dupes.add(key)
                if born:
                    dates[key] = born
        return {k: v for k, v in dates.items() if k not in dupes}
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
