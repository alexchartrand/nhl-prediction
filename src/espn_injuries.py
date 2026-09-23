"""Live injury/suspension feed from ESPN, the one current-availability signal
none of the other sources have (the NHL's public API has no injury endpoint,
and NHL.com's "(INJ.)" markers in the pasted projections are a static
snapshot -- deliberately not used).

``https://site.api.espn.com/.../nhl/injuries`` is unauthenticated and
unofficial, so like ``nhl_api.py`` every entry point degrades to "no data"
rather than raising. Each listed player comes with a status (Out / Injured
Reserve / Day-To-Day / Suspension), an injury type and an estimated
``returnDate``. The return date is turned into a share of the regular season
missed -- days out / season length, with the season's start/end dates from
the NHL API's schedule endpoint -- and predicted points are scaled down by
that share, so an injured player's VORP reflects the games he'll actually
play. If the schedule can't be fetched, players are still tagged but their
points are left alone.

Matched to the board by (normalized name, pos_group), with the same nickname
fallback as the NHL.com projections (nhl_projections.match_projection_rows).
"""

from __future__ import annotations

import time
from datetime import date

import pandas as pd
import requests

import nhl_api
import nhl_projections

INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/injuries"
SCHEDULE_URL = "https://api-web.nhle.com/v1/schedule/now"
REQUEST_TIMEOUT = 5.0

# ESPN position abbreviations -> pos_group.
POS_TO_GROUP = {"C": "F", "LW": "F", "RW": "F", "F": "F", "D": "D", "G": "G"}

# The board and goalie pool are rebuilt back to back by the app's
# "Recompute" button -- share one fetch between them, but keep the window
# short so the next recompute sees fresh news.
CACHE_TTL = 300.0
_CACHE: dict[str, tuple[float, object]] = {}

INJURY_COLS = ["Player", "pos_group", "espn_status", "injury_type", "return_date", "comment"]


def _cached(key: str, fetch):
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return hit[1]
    value = fetch()
    if value is not None:
        _CACHE[key] = (time.time(), value)
    return value


def parse_injuries(payload: dict) -> pd.DataFrame:
    """One row per listed player (see INJURY_COLS). Players whose position
    isn't F/D/G are skipped; a missing/garbled return date is NaT."""
    rows = []
    for team in payload.get("injuries", []):
        for item in team.get("injuries", []):
            athlete = item.get("athlete") or {}
            pos_group = POS_TO_GROUP.get((athlete.get("position") or {}).get("abbreviation"))
            name = athlete.get("displayName")
            if not name or pos_group is None:
                continue
            details = item.get("details") or {}
            rows.append({
                "Player": name,
                "pos_group": pos_group,
                "espn_status": item.get("status") or "",
                "injury_type": details.get("type") or "",
                "return_date": pd.to_datetime(details.get("returnDate"), errors="coerce"),
                "comment": item.get("shortComment") or "",
            })
    return pd.DataFrame(rows, columns=INJURY_COLS)


def fetch_injuries(timeout: float = REQUEST_TIMEOUT) -> pd.DataFrame | None:
    """ESPN's current injury list, or None on any failure (network, bad
    status, unexpected shape) -- None means 'no data', never 'nobody hurt'."""
    def _fetch():
        try:
            resp = requests.get(INJURIES_URL, timeout=timeout)
            resp.raise_for_status()
            return parse_injuries(resp.json())
        except Exception:
            return None

    return _cached("injuries", _fetch)


def fetch_season_dates(timeout: float = REQUEST_TIMEOUT) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """(regular-season start, end) from the NHL schedule, or None."""
    def _fetch():
        try:
            resp, _ = nhl_api._get_with_cooldown(requests.get, SCHEDULE_URL, timeout, budget=30.0)
            resp.raise_for_status()
            data = resp.json()
            start = pd.Timestamp(data["regularSeasonStartDate"])
            end = pd.Timestamp(data["regularSeasonEndDate"])
            return (start, end) if end > start else None
        except Exception:
            return None

    return _cached("season_dates", _fetch)


def status_label(espn_status: str, injury_type: str) -> str:
    """Short 'Notes' tag for one ESPN entry."""
    status = espn_status.casefold()
    if status == "suspension" or injury_type == "Suspension":
        return "Suspended"
    if status == "day-to-day":
        return "Day-to-day"
    if injury_type == "Contract Dispute":
        return "Holdout"
    if injury_type == "Personal":
        return "Out"
    return "Injured"


def season_share_missed(
    return_date: pd.Timestamp, season: tuple[pd.Timestamp, pd.Timestamp] | None, today: date | None = None
) -> float:
    """Share (0-1) of the full regular season a player misses when he's out
    until ``return_date`` -- counted from the later of today and opening
    night. NaN when either date is unknown."""
    if season is None or pd.isna(return_date):
        return float("nan")
    start, end = season
    today = pd.Timestamp(today or date.today())
    days_out = (return_date - max(today, start)).days
    return min(max(days_out, 0) / (end - start).days, 1.0)


def _detail(label: str, injury_type: str, return_date, games_missed, comment: str, healthy: float | None) -> str:
    parts = [label]
    if injury_type and injury_type not in ("Suspension", "Undisclosed"):
        parts.append(injury_type)
    if pd.notna(return_date):
        parts.append(f"back ~{return_date:%b %d}")
    if pd.notna(games_missed) and games_missed > 0:
        miss = f"~{int(games_missed)} GP missed"
        if healthy is not None and pd.notna(healthy):
            miss += f" ({healthy:.0f} pts if healthy)"
        parts.append(miss)
    text = " · ".join(parts)
    # ESPN's placeholder comments ("out", "ir", "day-to-day") add nothing.
    if len(comment.split()) > 2:
        text += f" -- {comment}"
    return text


def apply_injuries(
    df: pd.DataFrame,
    injuries: pd.DataFrame | None,
    season: tuple[pd.Timestamp, pd.Timestamp] | None,
    games_per_team: float,
    today: date | None = None,
) -> pd.DataFrame:
    """Copy of ``df`` (needs Player, pos_group, predicted_points) with:

    - ``healthy_points``: predicted_points before any injury adjustment
    - ``predicted_points``: scaled by (1 - share of the season missed)
    - ``games_missed``: that share x ``games_per_team``, rounded (NaN if unknown)
    - ``injury_tag``: short Notes tag ("Injured", "Day-to-day", ...), or None
    - ``injury``: one-line detail (type, return date, games missed, ESPN's note)

    ``injuries=None`` (feed unavailable) leaves every player untagged and
    unadjusted."""
    out = df.copy()
    out["healthy_points"] = out["predicted_points"]
    out["games_missed"] = float("nan")
    out["injury_tag"] = None
    out["injury"] = ""
    if injuries is None or injuries.empty or out.empty:
        return out

    matched = nhl_projections.match_projection_rows(out, injuries)
    for idx, inj_idx in matched.dropna().items():
        inj = injuries.loc[int(inj_idx)]
        label = status_label(inj["espn_status"], inj["injury_type"])
        share = season_share_missed(inj["return_date"], season, today)
        healthy = out.at[idx, "predicted_points"]
        if pd.notna(share):
            out.at[idx, "games_missed"] = round(share * games_per_team)
            if pd.notna(healthy):
                out.at[idx, "predicted_points"] = healthy * (1 - share)
        games = out.at[idx, "games_missed"]
        if pd.notna(games) and games > 0:
            out.at[idx, "injury_tag"] = f"{label} ~{int(games)} GP"
        elif games == 0 and label == "Injured":
            out.at[idx, "injury_tag"] = "Minor injury"  # expected back by opening night
        else:
            out.at[idx, "injury_tag"] = label
        out.at[idx, "injury"] = _detail(
            label, inj["injury_type"], inj["return_date"], games, inj["comment"],
            healthy if pd.notna(share) and share > 0 else None,
        )
    return out


def apply_live_injuries(df: pd.DataFrame, today: date | None = None) -> tuple[pd.DataFrame, bool]:
    """``apply_injuries`` with live ESPN + NHL schedule data. Never raises.
    Returns (frame, feed_ok) -- feed_ok=False means ESPN couldn't be reached
    and nobody is tagged."""
    try:
        injuries = fetch_injuries()
        season = fetch_season_dates()
        games = nhl_projections.games_per_team()
        return apply_injuries(df, injuries, season, games, today), injuries is not None
    except Exception:
        return apply_injuries(df, None, None, 0.0, today), False
