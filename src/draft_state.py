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
``managers.json`` (plus ``settings.json``, ``keepers.json`` -- see below --
and the computed boards under ``draft_board/``, see ``board_path``). A season name is free text (the user types it, e.g. when
starting a new pool year) but sanitized into a filesystem-safe directory
name via ``slugify``.

Keepers: each manager's last pick of a draft is his "keeper", kept into the
next season, where it counts as his last-round pick -- so a manager holding
one skips the final round. They're entered before the draft and stored apart
from live picks (``keepers.json``, no pick_number, never undone by "undo last
pick"), but count everywhere a roster is counted (``roster_picks``,
``drafted_player_ids``). Their final-round slots aren't stored; they're
derived from the current draft order and roster size (``reserved_slots``) so
editing the order or settings afterwards can't leave them stale.
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
# Where the per-season boards lived before they moved into the season
# directory (output/<kind>_board_<season>.csv) -- see board_path.
LEGACY_OUTPUT_DIR = STATE_DIR.parent / "output"
# Board kind -> file name under the season's draft_board/ directory.
BOARD_FILES = {"draft": "skaters.csv", "goalie": "goalies.csv", "team": "teams.csv"}

# ``slot`` is the pick's position in the snake sequence (0-based; see
# snake_manager) -- normally pick_number - 1, but it differs after a manual
# out-of-turn override so the order continues from the overridden manager.
# Missing (NaN) on picks made before a draft order was set.
PICK_COLUMNS = ["player_id", "player_name", "pos_group", "manager", "pick_number", "timestamp", "slot"]
KEEPER_COLUMNS = ["player_id", "player_name", "pos_group", "manager"]

# League shape a season's VORP/roster tracking is computed against -- pool
# size (num_managers) drives the VORP replacement-level cutoff (see
# rank.add_vorp), the rest drive both that cutoff (forwards/defense) and the
# "My Pool" roster-progress display (all four). Defaults only fill keys a
# season's settings.json doesn't have yet; rank/draft_pool take no defaults.
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


def _keepers_path(season: str) -> Path:
    return _season_dir(season) / "keepers.json"


def _settings_path(season: str) -> Path:
    return _season_dir(season) / "settings.json"


def board_path(season: str, kind: str) -> Path:
    """Where a season's computed board is saved (``kind`` is "draft" for
    F/D, "goalie" or "team"): ``state/seasons/<season>/draft_board/``. Lives
    with the season's other state since its VORP/pos_rank columns bake in
    that season's settings. A board saved at the old output/ location is
    moved here on first access, so upgrading doesn't force a rebuild (which
    hits the live APIs)."""
    if kind not in BOARD_FILES:
        raise ValueError(f"unknown board kind: {kind!r}")
    path = _season_dir(season) / "draft_board" / BOARD_FILES[kind]
    legacy = LEGACY_OUTPUT_DIR / f"{kind}_board_{slugify(season)}.csv"
    if not path.exists() and legacy.exists():
        path.parent.mkdir(exist_ok=True, parents=True)
        os.replace(legacy, path)
    return path


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
    records = picks.astype(object).where(picks.notna(), None)  # NaN slot -> JSON null
    _write_json(_picks_path(season), records.to_dict(orient="records"))


def snake_manager(order: list[str], slot: int) -> str:
    """Manager on the clock at ``slot`` (0-based) of a snake draft over
    ``order``: round 1 goes first-to-last, round 2 last-to-first, and so on
    alternating."""
    n = len(order)
    rnd, i = divmod(slot, n)
    return order[i if rnd % 2 == 0 else n - 1 - i]


def total_rounds(settings: dict) -> int:
    """Draft length in rounds: one pick per roster slot."""
    return sum(settings[k] for k in ("forwards", "defense", "goalies", "team_slots"))


def keeper_slot(order: list[str], manager: str, rounds: int) -> int:
    """``manager``'s turn in the final round -- the slot his keeper fills."""
    n = len(order)
    rnd = rounds - 1
    pos = order.index(manager)
    return rnd * n + (pos if rnd % 2 == 0 else n - 1 - pos)


def reserved_slots(season: str, order: list[str] | None = None) -> set[int]:
    """Final-round slots already filled by keepers, which the snake skips."""
    order = load_draft_order(season) if order is None else order
    if not order:
        return set()
    rounds = total_rounds(load_settings(season))
    return {keeper_slot(order, m, rounds) for m in load_keepers(season)["manager"] if m in order}


def slot_after(slot: int, reserved: set[int]) -> int:
    """First slot at or after ``slot`` that isn't taken by a keeper."""
    while slot in reserved:
        slot += 1
    return slot


def next_slot(picks: pd.DataFrame, reserved: set[int] = frozenset()) -> int:
    """Snake slot of the next pick: one past the last pick's slot, skipping
    slots filled by keepers. Picks made before a draft order was set have no
    slot, so those count by pick number."""
    if picks.empty:
        return slot_after(0, reserved)
    last = picks.sort_values("pick_number").iloc[-1]
    if pd.isna(last["slot"]):
        return slot_after(int(last["pick_number"]), reserved)
    return slot_after(int(last["slot"]) + 1, reserved)


def _pick_slot(picks: pd.DataFrame, order: list[str], manager: str, reserved: set[int]) -> int | None:
    """Slot to record for a pick by ``manager``. On turn, the expected slot.
    Off turn (manual override), the overridden manager's own slot in the
    current round, so the sequence continues from him rather than snapping
    back to where it was."""
    if not order or manager not in order:
        return None
    expected = next_slot(picks, reserved)
    if snake_manager(order, expected) == manager:
        return expected
    n = len(order)
    rnd = expected // n
    pos = order.index(manager)
    return rnd * n + (pos if rnd % 2 == 0 else n - 1 - pos)


def active_manager(season: str) -> str | None:
    """Manager on the clock, or None if no draft order is set for the season."""
    order = load_draft_order(season)
    if not order:
        return None
    return snake_manager(order, next_slot(load_picks(season), reserved_slots(season, order)))


def add_pick(season: str, player_id: str, player_name: str, pos_group: str, manager: str) -> pd.DataFrame:
    picks = load_picks(season)
    if player_id in drafted_player_ids(season):
        return picks
    order = load_draft_order(season)
    row = {
        "player_id": player_id,
        "player_name": player_name,
        "pos_group": pos_group,
        "manager": manager,
        "pick_number": len(picks) + 1,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "slot": _pick_slot(picks, order, manager, reserved_slots(season, order)),
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
    """Live picks and keepers -- everyone off the board."""
    return set(load_picks(season)["player_id"]) | set(load_keepers(season)["player_id"])


def load_keepers(season: str) -> pd.DataFrame:
    path = _keepers_path(season)
    if not path.exists():
        return pd.DataFrame(columns=KEEPER_COLUMNS)
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    return pd.DataFrame(rows, columns=KEEPER_COLUMNS)


def set_keeper(season: str, player_id: str, player_name: str, pos_group: str, manager: str) -> None:
    """One keeper per manager -- setting another replaces his previous one.
    Raises ValueError for a non-skater or a player already picked/kept."""
    if pos_group not in ("F", "D"):
        raise ValueError("Keepers must be forwards or defensemen.")
    keepers = load_keepers(season)
    keepers = keepers[keepers["manager"] != manager]
    if player_id in set(load_picks(season)["player_id"]) | set(keepers["player_id"]):
        raise ValueError(f"{player_name} is already drafted or kept by another manager.")
    row = {"player_id": player_id, "player_name": player_name, "pos_group": pos_group, "manager": manager}
    keepers = pd.concat([keepers, pd.DataFrame([row])], ignore_index=True)
    _write_json(_keepers_path(season), keepers.to_dict(orient="records"))


def remove_keeper(season: str, manager: str) -> None:
    keepers = load_keepers(season)
    _write_json(_keepers_path(season), keepers[keepers["manager"] != manager].to_dict(orient="records"))


def roster_picks(season: str) -> pd.DataFrame:
    """Every player on a roster: live picks plus keepers (``keeper`` True,
    no pick_number). Use for roster counts/views; use ``load_picks`` for
    anything about the order of live picks (snake, undo, last pick)."""
    picks = load_picks(season).assign(keeper=False)
    keepers = load_keepers(season).assign(keeper=True)
    if keepers.empty:
        return picks
    return pd.concat([picks, keepers], ignore_index=True)


def load_managers(season: str) -> dict:
    path = _managers_path(season)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_draft_order(season: str) -> list[str]:
    """Manager ids ("me" or an other-manager name) in draw order; empty if
    none set. Dropped down to ids that still exist, so renaming/removing a
    manager can't leave the snake pointing at nobody -- an incomplete order
    is treated as no order rather than silently skipping someone."""
    managers = load_managers(season)
    order = managers.get("draft_order", [])
    valid = {"me", *managers.get("other_managers", [])}
    return order if order and set(order) == valid and len(order) == len(valid) else []


def save_managers(
    season: str, my_team_name: str, other_managers: list[str], draft_order: list[str] | None = None
) -> None:
    data = {"my_team_name": my_team_name, "other_managers": other_managers}
    if draft_order:
        data["draft_order"] = draft_order
    _write_json(_managers_path(season), data)


def load_settings(season: str) -> dict:
    """League shape for this season (pool size + roster slots), falling
    back to DEFAULT_SETTINGS for any key not yet saved (e.g. a season
    created before a new setting existed). Once a draft order is set, its
    length *is* the manager count -- a stale saved ``num_managers`` would
    put every VORP replacement level at the wrong depth."""
    path = _settings_path(season)
    saved = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)
    settings = {**DEFAULT_SETTINGS, **saved}
    order = load_draft_order(season)
    if order:
        settings["num_managers"] = len(order)
    return settings


def save_settings(season: str, settings: dict) -> None:
    _write_json(_settings_path(season), settings)
