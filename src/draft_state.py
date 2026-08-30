"""Draft-day pick and manager persistence.

Single local user, no concurrency -- plain JSON files under ``state/`` so the
draft survives an app restart (a live draft runs for hours) and stays
hand-editable if a pick needs correcting. Picks are keyed by ``player_id``
(``loading.ID_COL``), matching the rest of the codebase's join-key
discipline, rather than by player name.

State is scoped **per season** (one pool year's draft, e.g. "2026-2027") so a
new draft doesn't inherit last year's picks/managers, and last year's draft
stays around to look back on. Each season gets its own directory under
``state/seasons/<season>/`` holding that season's ``picks.json`` and
``managers.json``. A season name is free text (the user types it, e.g. when
starting a new pool year) but sanitized into a filesystem-safe directory
name via ``slugify``.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

STATE_DIR = Path(__file__).resolve().parent.parent / "state"
SEASONS_DIR = STATE_DIR / "seasons"
# Pre-season-state layout, kept only so an existing draft in progress isn't
# stranded when this version first runs -- see _migrate_legacy_state.
LEGACY_PICKS_PATH = STATE_DIR / "picks.json"
LEGACY_MANAGERS_PATH = STATE_DIR / "managers.json"
LEGACY_MIGRATION_SEASON = "2026-2027"

PICK_COLUMNS = ["player_id", "player_name", "pos_group", "manager", "pick_number", "timestamp"]

# League shape a season's VORP/roster tracking is computed against -- pool
# size (num_managers) drives the VORP replacement-level cutoff (see
# rank.add_vorp), the rest drive both that cutoff (forwards/defense) and the
# "My Pool" roster-progress display (all four). Defaults match the values
# that were hardcoded pre-season-scoping (rank.TEAMS_IN_POOL/ROSTER_SLOTS,
# app.py's old ROSTER_TARGETS).
DEFAULT_SETTINGS = {
    "num_managers": 12,
    "forwards": 9,
    "defense": 5,
    "goalies": 1,
    "team_slots": 1,
}


def slugify(season: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", season.strip()).strip("-")


def _season_dir(season: str) -> Path:
    slug = slugify(season)
    if not slug:
        raise ValueError(f"invalid season name: {season!r}")
    return SEASONS_DIR / slug


def _picks_path(season: str) -> Path:
    return _season_dir(season) / "picks.json"


def _managers_path(season: str) -> Path:
    return _season_dir(season) / "managers.json"


def _settings_path(season: str) -> Path:
    return _season_dir(season) / "settings.json"


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(exist_ok=True, parents=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _migrate_legacy_state() -> None:
    """One-time move of the pre-season-scoped state/picks.json and
    state/managers.json into a season directory, so upgrading doesn't lose
    an in-progress draft. No-ops once state/seasons/ exists."""
    if SEASONS_DIR.exists() or not (LEGACY_PICKS_PATH.exists() or LEGACY_MANAGERS_PATH.exists()):
        return
    target = _season_dir(LEGACY_MIGRATION_SEASON)
    target.mkdir(exist_ok=True, parents=True)
    if LEGACY_PICKS_PATH.exists():
        os.replace(LEGACY_PICKS_PATH, target / "picks.json")
    if LEGACY_MANAGERS_PATH.exists():
        os.replace(LEGACY_MANAGERS_PATH, target / "managers.json")


def list_seasons() -> list[str]:
    """Existing seasons, most recently created first."""
    _migrate_legacy_state()
    if not SEASONS_DIR.exists():
        return []
    dirs = [d for d in SEASONS_DIR.iterdir() if d.is_dir()]
    dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return [d.name for d in dirs]


def season_exists(season: str) -> bool:
    return _season_dir(season).exists()


def create_season(season: str) -> str:
    """Creates a new, empty season and returns its (slugified) name. If the
    season already exists, just returns its name -- no data is reset."""
    _season_dir(season).mkdir(exist_ok=True, parents=True)
    return slugify(season)


def load_picks(season: str) -> pd.DataFrame:
    path = _picks_path(season)
    if not path.exists():
        return pd.DataFrame(columns=PICK_COLUMNS)
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    return pd.DataFrame(rows, columns=PICK_COLUMNS) if rows else pd.DataFrame(columns=PICK_COLUMNS)


def save_picks(season: str, picks: pd.DataFrame) -> None:
    _write_json(_picks_path(season), picks.to_dict(orient="records"))


def add_pick(season: str, player_id: str, player_name: str, pos_group: str, manager: str) -> pd.DataFrame:
    picks = load_picks(season)
    if player_id in set(picks["player_id"]):
        return picks
    row = {
        "player_id": player_id,
        "player_name": player_name,
        "pos_group": pos_group,
        "manager": manager,
        "pick_number": len(picks) + 1,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    picks = pd.concat([picks, pd.DataFrame([row])], ignore_index=True)
    save_picks(season, picks)
    return picks


def undo_last_pick(season: str) -> pd.DataFrame:
    picks = load_picks(season)
    if picks.empty:
        return picks
    picks = picks[picks["pick_number"] != picks["pick_number"].max()].reset_index(drop=True)
    save_picks(season, picks)
    return picks


def drafted_player_ids(season: str) -> set[str]:
    return set(load_picks(season)["player_id"])


def load_managers(season: str) -> dict:
    path = _managers_path(season)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_managers(season: str, my_team_name: str, other_managers: list[str]) -> None:
    _write_json(_managers_path(season), {"my_team_name": my_team_name, "other_managers": other_managers})


def load_settings(season: str) -> dict:
    """League shape for this season (pool size + roster slots), falling
    back to DEFAULT_SETTINGS for any key not yet saved (e.g. a season
    created before a new setting existed)."""
    path = _settings_path(season)
    saved = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)
    return {**DEFAULT_SETTINGS, **saved}


def save_settings(season: str, settings: dict) -> None:
    _write_json(_settings_path(season), settings)
