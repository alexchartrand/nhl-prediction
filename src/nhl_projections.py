"""Parses NHL.com's 2026-27 season projections (``data/nhl 2026-2027
projections/*.txt``, pasted from the site's fantasy hub) for goalies and
teams, plus forwards/defense for display. Goalie and team projections drive
ranking directly (see CLAUDE.md) instead of a trained model: NHL.com's numbers bake in this season's depth-chart role
(starter vs. backup) and team-building moves that the historical Hockey-
Reference stats have no way to see. Forwards/defense are still ranked by the
in-repo model; ``fowards.txt``/``defense.txt`` (fantasy points, not wins) are
shown alongside as a reference column, and stand in as ``predicted_points``
only for players the model can't rank (rookies with no usable history).

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

import nhl_api

PROJECTIONS_DIR = Path(__file__).resolve().parent.parent / "data" / "nhl 2026-2027 projections"
GOALIES_FILE = PROJECTIONS_DIR / "goalies.txt"
TEAMS_FILE = PROJECTIONS_DIR / "teams.txt"
FORWARDS_FILE = PROJECTIONS_DIR / "fowards.txt"
DEFENSE_FILE = PROJECTIONS_DIR / "defense.txt"

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


def load_skater_projections(
    forwards_path: Path = FORWARDS_FILE, defense_path: Path = DEFENSE_FILE
) -> pd.DataFrame:
    """NHL.com's projected fantasy points per forward/defenseman, one row per
    "Name, F|D, TEAM[ (INJ.)]: points" line. ``pos_group`` is included so
    callers can join by (normalized name, pos_group) -- same-name collisions
    across positions exist (see CLAUDE.md)."""
    rows = []
    for path in (forwards_path, defense_path):
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or ":" not in line:
                continue
            name_part, _, points_part = line.rpartition(":")
            name_part = _INJ.sub("", name_part).strip()
            fields = [f.strip() for f in name_part.split(",")]
            if len(fields) < 2 or fields[1] not in ("F", "D"):
                continue
            team = fields[2] if len(fields) > 2 else None
            rows.append(
                {"Player": fields[0], "pos_group": fields[1], "Team": team, "nhl_projection": int(points_part.strip())}
            )
    return pd.DataFrame(rows, columns=["Player", "pos_group", "Team", "nhl_projection"])


# First-name spellings that differ between NHL.com and Hockey-Reference and
# aren't caught by the 3-letter prefix fallback below (Josh/Joshua is).
_FIRST_NAME_ALIASES = {"tommy": "thomas"}


def _loose_name_key(name: str) -> str:
    """Last name + first 3 letters of the (alias-resolved) first name --
    bridges nicknames like "Matt Coronato" vs. "Matthew Coronato"."""
    first, _, last = nhl_api.normalize_name(name).partition(" ")
    return f"{_FIRST_NAME_ALIASES.get(first, first)[:3]} {last}"


def match_projection_rows(players: pd.DataFrame, proj: pd.DataFrame) -> pd.Series:
    """For each row of ``players`` (needs Player, pos_group), the index of
    its row in ``proj`` or NA. Matches by (exact normalized name, pos_group),
    falling back to the loose nickname key only when that key is unique on
    both sides and the projection row has no exact match elsewhere -- so it
    can't attach one brother's projection to another (Ilya vs. Aliaksei
    Protas)."""
    key = players["Player"].map(nhl_api.normalize_name)
    loose = players["Player"].map(_loose_name_key)
    proj_key = proj["Player"].map(nhl_api.normalize_name)
    proj_loose = proj["Player"].map(_loose_name_key)

    exact = {}
    for i, k, pos in zip(proj.index, proj_key, proj["pos_group"]):
        exact.setdefault((k, pos), i)
    player_keys = set(zip(key, players["pos_group"]))
    loose_pairs = list(zip(proj_loose, proj["pos_group"]))
    loose_counts = pd.Series(loose_pairs).value_counts()
    loose_map = {
        lp: i
        for i, lp, k in zip(proj.index, loose_pairs, proj_key)
        if loose_counts[lp] == 1 and (k, lp[1]) not in player_keys
    }
    player_loose_counts = pd.Series(list(zip(loose, players["pos_group"]))).value_counts()

    out = []
    for k, lk, pos in zip(key, loose, players["pos_group"]):
        if (k, pos) in exact:
            out.append(exact[(k, pos)])
        elif player_loose_counts[(lk, pos)] == 1:
            out.append(loose_map.get((lk, pos)))
        else:
            out.append(None)
    return pd.Series(out, index=players.index, dtype="Int64")


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
