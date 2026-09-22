"""Parses NHL.com's 2026-27 season projections (``data/nhl 2026-2027
projections/*.txt``, pasted from the site's fantasy hub) for goalies and
teams. These drive goalie and team ranking directly (see CLAUDE.md) instead
of a trained model: NHL.com's numbers bake in this season's depth-chart role
(starter vs. backup) and team-building moves that the historical Hockey-
Reference stats have no way to see. Forwards/defense keep using the in-repo
model -- ``fowards.txt``/``defense.txt`` are not read anywhere.

Each numbered goalie line is "Name, POS, TEAM: wins", sometimes with an
"(INJ.)" tag or a committee of two goalies sharing one line ("Alex Lyon or
Colten Ellis, G, BUF: 18") -- that expands to one row per goalie, sharing the
line's projected win total and whichever team appears on either name. The
number is NHL.com's projected win count, not a fantasy point projection.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PROJECTIONS_DIR = Path(__file__).resolve().parent.parent / "data" / "nhl 2026-2027 projections"
GOALIES_FILE = PROJECTIONS_DIR / "goalies.txt"
TEAMS_FILE = PROJECTIONS_DIR / "teams.txt"

_INJ = re.compile(r"\s*\(INJ\.\)\s*")
_NAME_SUFFIX = re.compile(r",\s*[FDG](?:,\s*([A-Z]{2,3}))?\s*$")
_TEAM_LINE = re.compile(r"^([A-Z]{2,3}):\s*(\d+)\s*\((?:(plus|minus)-(\d+)|same)\)\s*$")


def _strip_line_number(raw: str) -> str:
    """Lines come as "<n>\\t<content>" -- drop the leading rank/index."""
    _, _, rest = raw.partition("\t")
    return rest.strip() if rest else raw.strip()


def _parse_name_segment(segment: str) -> tuple[str, str | None]:
    """Strips a trailing ", POS[, TEAM]" off one name, if present."""
    segment = segment.strip()
    m = _NAME_SUFFIX.search(segment)
    if not m:
        return segment, None
    return segment[: m.start()].strip(), m.group(1)


def _parse_goalie_line(line: str) -> list[dict]:
    name_part, _, wins_part = line.rpartition(":")
    injured = bool(_INJ.search(name_part))
    name_part = _INJ.sub("", name_part).strip()
    wins = int(wins_part.strip())

    parsed = [_parse_name_segment(s) for s in name_part.split(" or ")]
    team = next((t for _, t in parsed if t), None)
    return [
        {"Player": name, "Team": team, "projected_wins": wins, "injured": injured}
        for name, _ in parsed
        if name
    ]


def load_goalie_projections(path: Path = GOALIES_FILE) -> pd.DataFrame:
    """NHL.com's projected win totals for the season, one row per goalie
    (committee lines split into one row each, see module docstring). No
    player_id here -- callers join to Hockey-Reference history by name."""
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = _strip_line_number(raw)
        if not line or ":" not in line:
            continue
        rows.extend(_parse_goalie_line(line))
    df = pd.DataFrame(rows, columns=["Player", "Team", "projected_wins", "injured"])
    return df.sort_values("projected_wins", ascending=False).reset_index(drop=True)


def load_team_projections(path: Path = TEAMS_FILE) -> pd.DataFrame:
    """NHL.com's projected win total per team for the season, plus the
    change from last season as printed on the page (``win_delta``, e.g. +10,
    -1, 0 for "same"). Used to rank the pool's "1 team" roster slot, which
    otherwise has no model (see CLAUDE.md)."""
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = _strip_line_number(raw)
        m = _TEAM_LINE.match(line)
        if not m:
            continue
        code, wins, sign, delta = m.groups()
        win_delta = 0 if sign is None else int(delta) * (1 if sign == "plus" else -1)
        rows.append({"Code": code, "projected_wins": int(wins), "win_delta": win_delta})
    df = pd.DataFrame(rows, columns=["Code", "projected_wins", "win_delta"])
    return df.sort_values("projected_wins", ascending=False).reset_index(drop=True)
