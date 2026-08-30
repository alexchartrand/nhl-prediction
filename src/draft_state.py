"""Draft-day pick and manager persistence.

Single local user, no concurrency -- plain JSON files under ``state/`` so the
draft survives an app restart (a live draft runs for hours) and stays
hand-editable if a pick needs correcting. Picks are keyed by ``player_id``
(``loading.ID_COL``), matching the rest of the codebase's join-key
discipline, rather than by player name.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

STATE_DIR = Path(__file__).resolve().parent.parent / "state"
PICKS_PATH = STATE_DIR / "picks.json"
MANAGERS_PATH = STATE_DIR / "managers.json"

PICK_COLUMNS = ["player_id", "player_name", "pos_group", "manager", "pick_number", "timestamp"]


def _write_json(path: Path, data) -> None:
    STATE_DIR.mkdir(exist_ok=True, parents=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def load_picks() -> pd.DataFrame:
    if not PICKS_PATH.exists():
        return pd.DataFrame(columns=PICK_COLUMNS)
    with open(PICKS_PATH, encoding="utf-8") as f:
        rows = json.load(f)
    return pd.DataFrame(rows, columns=PICK_COLUMNS) if rows else pd.DataFrame(columns=PICK_COLUMNS)


def save_picks(picks: pd.DataFrame) -> None:
    _write_json(PICKS_PATH, picks.to_dict(orient="records"))


def add_pick(player_id: str, player_name: str, pos_group: str, manager: str) -> pd.DataFrame:
    picks = load_picks()
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
    save_picks(picks)
    return picks


def undo_last_pick() -> pd.DataFrame:
    picks = load_picks()
    if picks.empty:
        return picks
    picks = picks[picks["pick_number"] != picks["pick_number"].max()].reset_index(drop=True)
    save_picks(picks)
    return picks


def drafted_player_ids() -> set[str]:
    return set(load_picks()["player_id"])


def load_managers() -> dict:
    if not MANAGERS_PATH.exists():
        return {}
    with open(MANAGERS_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_managers(my_team_name: str, other_managers: list[str]) -> None:
    _write_json(MANAGERS_PATH, {"my_team_name": my_team_name, "other_managers": other_managers})
