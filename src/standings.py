"""Live projected pool standings and roster outlook for the "My Pool" tab.

Everything here works off the three boards the app already has (F/D from
rank.build_draft_board, goalies/teams from draft_pool) and the rostered
picks + keepers (draft_state.roster_picks) -- pure pandas, no Streamlit, so
it can be checked outside the app.

Mid-draft, raw drafted totals favor whoever has picked most, so each
manager's open slots are filled at the *expected fill value* of that
position: the average of the best remaining players who will fill the
league's open slots there (open slots league-wide = managers x slots -
rostered). Every manager gets the same value per open slot, so the
standings differences come only from picks already made.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from draft_state import snake_manager

POSITIONS = ["F", "D", "G", "TEAM"]
SETTING_KEY = {"F": "forwards", "D": "defense", "G": "goalies", "TEAM": "team_slots"}
# Pool pays the top 2 (see backlog.md, league facts).
PAID_PLACES = 2
# "Drop if you wait" (fantasy points) at or above which a position gets a
# "Big drop" flag. In a simulated 12-manager draft (others taking the best
# player at a position weighted by their open slots), one-round drops were
# mostly 0-5 pts; 6+ caught the few real tier breaks (~5% of cases, mostly
# F/D in the first rounds).
BIG_DROP_PTS = 6.0


def points_pool(board: pd.DataFrame, goalies: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """One (player_id, pos_group, predicted_points) row per draftable
    player/team, across the three boards."""
    return pd.concat(
        [df[["player_id", "pos_group", "predicted_points"]] for df in (board, goalies, teams)],
        ignore_index=True,
    ).drop_duplicates("player_id")


def slot_targets(settings: dict) -> dict[str, int]:
    return {pos: settings[SETTING_KEY[pos]] for pos in POSITIONS}


def expected_fill_values(pool: pd.DataFrame, rostered: pd.DataFrame, settings: dict) -> dict[str, float]:
    """Expected points of a still-open slot at each position: the mean of
    the top (open slots league-wide) remaining players there. NaN when the
    position has no open slots or nobody left."""
    targets = slot_targets(settings)
    remaining = pool[~pool["player_id"].isin(rostered["player_id"])]
    fill = {}
    for pos in POSITIONS:
        open_slots = settings["num_managers"] * targets[pos] - (rostered["pos_group"] == pos).sum()
        pts = remaining.loc[remaining["pos_group"] == pos, "predicted_points"].nlargest(max(int(open_slots), 0))
        fill[pos] = float(pts.mean()) if len(pts) else float("nan")
    return fill


def projected_standings(
    pool: pd.DataFrame, rostered: pd.DataFrame, managers: list[str], settings: dict
) -> pd.DataFrame:
    """One row per manager, best projected total first: drafted points
    (``rostered`` = draft_state.roster_picks, keepers included), open slots
    left, and projected final total (drafted + open slots at the expected
    fill value). ``unmatched`` counts picks with no row in ``pool`` (scored
    as 0) -- should stay 0."""
    targets = slot_targets(settings)
    fill = expected_fill_values(pool, rostered, settings)
    picks = rostered.merge(pool[["player_id", "predicted_points"]], on="player_id", how="left")
    rows = []
    for m in managers:
        mine = picks[picks["manager"] == m]
        have = mine["pos_group"].value_counts()
        open_by_pos = {pos: max(targets[pos] - int(have.get(pos, 0)), 0) for pos in POSITIONS}
        drafted = float(mine["predicted_points"].sum())
        future = sum(n * fill[pos] for pos, n in open_by_pos.items() if n)
        rows.append({
            "manager": m,
            "picks": len(mine),
            "drafted_points": drafted,
            "open_slots": sum(open_by_pos.values()),
            "projected_total": drafted + future,
            "unmatched": int(mine["predicted_points"].isna().sum()),
        })
    out = pd.DataFrame(rows).sort_values("projected_total", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", out.index + 1)
    return out


def my_turns(order: list[str], start_slot: int, reserved: set[int], rounds: int, me: str = "me", n: int = 2) -> list[int]:
    """The next ``n`` snake slots (at or after ``start_slot``) that are
    ``me``'s and not already filled by a keeper."""
    turns = []
    for slot in range(start_slot, rounds * len(order)):
        if slot not in reserved and snake_manager(order, slot) == me:
            turns.append(slot)
            if len(turns) == n:
                break
    return turns


def _expected_taken(
    order: list[str], slots: range, reserved: set[int], open_counts: dict[str, dict[str, float]]
) -> dict[str, float]:
    """Expected players taken per position over ``slots``: each pick by
    manager X is position P with probability X's open P slots / X's open
    slots, and X's open counts are decremented by those probabilities as the
    snake is walked (so a manager picking twice in a row at the turn doesn't
    count the same need twice). Mutates ``open_counts``."""
    taken = {pos: 0.0 for pos in POSITIONS}
    for slot in slots:
        if slot in reserved:
            continue
        counts = open_counts[snake_manager(order, slot)]
        total = sum(counts.values())
        if total <= 0:
            continue
        for pos in POSITIONS:
            p = counts[pos] / total
            taken[pos] += p
            counts[pos] -= p
    return taken


def _value_at(sorted_pts: np.ndarray, k: float) -> float:
    """Points of the best player left after ``k`` (possibly fractional)
    players are taken from the top of ``sorted_pts`` (descending)."""
    if len(sorted_pts) == 0 or k > len(sorted_pts) - 1:
        return float("nan")
    return float(np.interp(k, np.arange(len(sorted_pts)), sorted_pts))


def position_outlook(
    pool: pd.DataFrame,
    rostered: pd.DataFrame,
    managers: list[str],
    settings: dict,
    order: list[str],
    next_slot: int,
    reserved: set[int],
    me: str = "me",
) -> tuple[pd.DataFrame, dict]:
    """For each position ``me`` still needs: best available now, expected
    best at my next pick, and at the pick after that -- the gap between the
    last two is what waiting a round at that position costs. Assumes the
    other managers take the best remaining player at whichever position they
    pick (by their own open slots, see _expected_taken).

    Returns (table, info) where info has ``picks_before_next`` and
    ``picks_between`` (other managers' picks before my next turn / between
    my next two turns; None when there's no such turn)."""
    targets = slot_targets(settings)
    rounds = sum(targets.values())
    counts = {
        m: {pos: float(max(targets[pos] - ((rostered["manager"] == m) & (rostered["pos_group"] == pos)).sum(), 0))
            for pos in POSITIONS}
        for m in managers
    }
    my_open = dict(counts[me])

    turns = my_turns(order, next_slot, reserved, rounds, me=me) if order else []
    before = {pos: float("nan") for pos in POSITIONS}
    after = dict(before)
    info = {"picks_before_next": None, "picks_between": None}
    if turns:
        before = _expected_taken(order, range(next_slot, turns[0]), reserved, counts)
        info["picks_before_next"] = sum(1 for s in range(next_slot, turns[0]) if s not in reserved)
    if len(turns) == 2:
        between = _expected_taken(order, range(turns[0] + 1, turns[1]), reserved, counts)
        after = {pos: before[pos] + between[pos] for pos in POSITIONS}
        info["picks_between"] = sum(1 for s in range(turns[0] + 1, turns[1]) if s not in reserved)

    remaining = pool[~pool["player_id"].isin(rostered["player_id"])]
    rows = []
    for pos in POSITIONS:
        if my_open[pos] <= 0:
            continue
        pts = np.sort(remaining.loc[remaining["pos_group"] == pos, "predicted_points"].to_numpy())[::-1]
        at_next = _value_at(pts, before[pos])
        at_following = _value_at(pts, after[pos])
        rows.append({
            "pos_group": pos,
            "need": int(my_open[pos]),
            "best_now": _value_at(pts, 0),
            "at_next_pick": at_next,
            "taken_before_next": before[pos],
            "at_following_pick": at_following,
            "taken_by_following": after[pos],
            "drop_if_wait": at_next - at_following,
        })
    return pd.DataFrame(rows), info
