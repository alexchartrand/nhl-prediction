"""Draft-day smoke test: a full 10-manager snake draft through the app's real
code paths, in a throwaway state directory (the real ``state/`` is never
touched).

Run:
    .venv/Scripts/python.exe tests/smoke_full_draft.py            # live feeds (NHL API, ESPN)
    .venv/Scripts/python.exe tests/smoke_full_draft.py --no-app   # skip the headless Streamlit runs

What it does:
  1. Creates a season with 10 managers in draft order (you at --my-pos) and
     lets the headless app (streamlit.testing AppTest) build the F/D, goalie
     and team boards exactly as "Recompute draft board" would.
  2. Enters keepers for 7 of the 9 other managers (you hold none), and checks
     keeper validation (goalie keeper / duplicate keeper are refused).
  3. Drafts all 16 rounds: each manager picks per a different style (VORP,
     raw points, NHL.com rank, random-ish, D-heavy); you pick by VORP with the
     keeper logic (goalie/team before your last F/D slot, keeper value last).
     Along the way: undo/redo, an out-of-turn pick then undo, a duplicate
     pick, and several picks made through the app's own "Draft" button.
  4. Checks invariants after every pick (snake order, keeper slots skipped,
     no player twice, VORP/standings/roster-needs computable) and at the end
     (every roster exactly 9F/5D/1G/1TEAM, your last pick is your keeper).
  5. Renders every app tab headlessly at the start, mid-draft, at your final
     pick and after the draft, failing on any exception or st.error.
  6. Replays the draft logic (no app) with you at each of the 10 draft slots.
"""

from __future__ import annotations

import argparse
import random
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import pandas as pd

import draft_pool
import draft_state
import keeper
import rank
import standings

SEASON = "smoke-2026-2027"
MY_TEAM = "Alex"
OTHERS = ["Lacoste", "Louis", "Martin", "Julie", "Sam", "Gab", "Max", "Vince", "Phil"]
# Drafting style per other manager (see choose_pick).
STYLES = {
    "Lacoste": "vorp", "Louis": "raw_points", "Martin": "nhl_rank", "Julie": "random_top", "Sam": "d_heavy",
    "Gab": "vorp", "Max": "raw_points", "Vince": "nhl_rank", "Phil": "random_top",
}
# Managers holding a keeper from last season (the rest, and you, don't).
KEEPER_HOLDERS = ["Lacoste", "Louis", "Martin", "Julie", "Sam", "Gab", "Max"]
# Live picks (1-based, the pick about to be made) made by pressing the app's
# own "Draft" button rather than calling draft_state directly.
UI_PICKS = {3, 58}


class Checks:
    def __init__(self) -> None:
        self.passed = 0
        self.failures: list[str] = []

    def check(self, cond, msg: str) -> bool:
        if cond:
            self.passed += 1
        else:
            self.failures.append(msg)
            print(f"  FAIL: {msg}")
        return bool(cond)


C = Checks()


def isolate(tmp: Path) -> None:
    """Point every file write (draft state JSON, board CSVs) at ``tmp``."""
    draft_state.STATE_DIR = tmp / "state"
    draft_state.SEASONS_DIR = tmp / "state" / "seasons"
    draft_state.LEGACY_PICKS_PATH = tmp / "state" / "picks.json"
    draft_state.LEGACY_MANAGERS_PATH = tmp / "state" / "managers.json"
    draft_state.LEGACY_OUTPUT_DIR = tmp / "output"


def board_paths(season: str) -> dict[str, Path]:
    return {k: draft_state.board_path(season, k) for k in draft_state.BOARD_FILES}


def setup_season(season: str, my_pos: int) -> list[str]:
    draft_state.create_season(season)
    names = OTHERS[:my_pos] + [MY_TEAM] + OTHERS[my_pos:]
    order = ["me" if n == MY_TEAM else n for n in names]
    draft_state.save_managers(season, MY_TEAM, OTHERS, order)
    return order


def settings_for(season: str) -> dict:
    return draft_state.load_settings(season)


def roster(settings: dict) -> dict:
    return {"F": settings["forwards"], "D": settings["defense"]}


# --------------------------------------------------------------------------
# Boards
# --------------------------------------------------------------------------

def build_boards_directly(season: str, settings: dict, live: bool) -> None:
    """Same calls as app._rebuild_board/_rebuild_goalies/_rebuild_teams, for
    runs without the headless app."""
    import goalies as goalies_module

    paths = board_paths(season)
    paths["draft"].parent.mkdir(parents=True, exist_ok=True)
    board = rank.build_draft_board(
        teams=settings["num_managers"], roster=roster(settings),
        fetch_live_team_changes=live, fetch_live_injuries=live, fetch_espn_projections=live,
    )
    board.to_csv(paths["draft"], index=False)
    g = goalies_module.load_scored_goalies()
    draft_pool.goalie_pool(g).to_csv(paths["goalie"], index=False)
    draft_pool.team_pool(g).to_csv(paths["team"], index=False)


def load_boards(season: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Boards as the app reads them back (CSV round-trip included)."""
    p = board_paths(season)
    return pd.read_csv(p["draft"]), pd.read_csv(p["goalie"]), pd.read_csv(p["team"])


def check_boards(board: pd.DataFrame, goalies: pd.DataFrame, teams: pd.DataFrame, settings: dict) -> None:
    n = settings["num_managers"]
    print(f"  boards: {len(board)} F/D ({(board.pos_group == 'F').sum()} F, {(board.pos_group == 'D').sum()} D), "
          f"{len(goalies)} G, {len(teams)} teams")
    for name, df in (("F/D", board), ("goalie", goalies), ("team", teams)):
        C.check(df["player_id"].is_unique, f"{name} board has duplicate player_ids")
        C.check(df["player_id"].notna().all(), f"{name} board has missing player_ids")
        C.check(df["predicted_points"].notna().all(), f"{name} board has NaN predicted_points")
        # The linear F/D model isn't clipped at 0, so fringe depth players
        # (enforcers etc., far below replacement) can come out slightly
        # negative. Harmless unless it reaches anyone draftable (top 2x the
        # league's roster slots at the position).
        draftable = df["pos_rank"] <= 2 * n * df["pos_group"].map(roster(settings)) if "pos_rank" in df else True
        neg = df[(df["predicted_points"] < 0) & draftable]
        C.check(neg.empty, f"{name} board: negative predicted_points near draftable range: {neg['Player'].tolist()[:5]}"
                if "Player" in df else f"{name} board has negative predicted_points")
    ids = pd.concat([board.player_id, goalies.player_id, teams.player_id])
    C.check(ids.is_unique, f"player_id shared across boards: {ids[ids.duplicated()].tolist()[:5]}")
    for pos, slots in roster(settings).items():
        C.check((board.pos_group == pos).sum() >= n * slots + 20, f"not enough {pos} on the board for {n} managers")
    C.check(len(goalies) >= n * settings["goalies"] + 3, f"only {len(goalies)} goalies for {n} managers")
    C.check(len(teams) == 32, f"expected 32 teams, got {len(teams)}")
    for col in ("nhl_projection", "injury", "future_value", "next_season_points", "healthy_points", "Notes"):
        C.check(col in board.columns, f"F/D board missing column {col}")
    for col in ("pts_per_win", "injury", "Notes"):
        C.check(col in goalies.columns, f"goalie board missing column {col}")
    C.check("projected_otl" in teams.columns, "team board missing projected_otl")
    top = board.head(20)
    C.check(top["Player"].notna().all() and top["VORP"].gt(0).all(), "top of F/D board has blank players or VORP <= 0")
    C.check(board["future_value"].ge(0).all(), "negative future_value on the F/D board")
    teams_named = teams["Team"].str.len().gt(3).mean()
    C.check(teams_named > 0.9, f"only {teams_named:.0%} of teams have a full name (live NHL API names missing?)")


# --------------------------------------------------------------------------
# Live availability, exactly as the app computes it
# --------------------------------------------------------------------------

def available(season: str, boards, settings: dict) -> dict[str, pd.DataFrame]:
    board, goalies, teams = boards
    drafted = draft_state.drafted_player_ids(season)
    n = settings["num_managers"]
    fd = draft_pool.undrafted_board(board, drafted, teams=n, roster=roster(settings))
    fd["keeper_value"] = keeper.keeper_value(fd).round(1)
    g = draft_pool.undrafted_goalies(goalies, drafted, teams=n, slots=settings["goalies"])
    t = draft_pool.undrafted_teams(teams, drafted, teams=n, slots=settings["team_slots"])
    return {"FD": fd, "G": g, "TEAM": t}


def open_slots(season: str, manager: str, settings: dict) -> dict[str, int]:
    picks = draft_state.roster_picks(season)
    have = picks.loc[picks["manager"] == manager, "pos_group"].value_counts()
    return {pos: settings[key] - int(have.get(pos, 0)) for pos, key in standings.SETTING_KEY.items()}


def candidates(av: dict[str, pd.DataFrame], open_: dict[str, int]) -> pd.DataFrame:
    """Every available player/team at a position the manager still needs."""
    fd = av["FD"].assign(name=av["FD"]["Player"])
    g = av["G"].assign(name=av["G"]["Player"], nhl_projection=float("nan"), keeper_value=float("nan"))
    t = av["TEAM"].assign(name=av["TEAM"]["Team"], nhl_projection=float("nan"), keeper_value=float("nan"))
    cols = ["player_id", "name", "pos_group", "VORP", "predicted_points", "nhl_projection", "keeper_value"]
    allc = pd.concat([fd[cols], g[cols], t[cols]], ignore_index=True)
    return allc[allc["pos_group"].map(lambda p: open_.get(p, 0) > 0)]


def choose_pick(season: str, manager: str, style: str, av, settings: dict, rng: random.Random) -> pd.Series:
    open_ = open_slots(season, manager, settings)
    cand = candidates(av, open_)
    assert not cand.empty, f"{manager} has open slots {open_} but no candidates"
    if manager == "me":
        status = keeper.roster_status(draft_state.roster_picks(season), draft_state.load_keepers(season), settings)
        if status["final_pick"]:
            return cand[cand["pos_group"].isin(keeper.KEEPER_POSITIONS)].sort_values("keeper_value").iloc[-1]
        # Keep an F/D slot for last so the final pick can be a keeper.
        if status["open_skater"] == 1 and status["open_other"] >= 1:
            cand = cand[~cand["pos_group"].isin(keeper.KEEPER_POSITIONS)]
        return cand.sort_values("VORP").iloc[-1]
    if style == "vorp":
        return cand.sort_values("VORP").iloc[-1]
    if style == "raw_points":  # ignores position scarcity -> teams/goalies go early
        return cand.sort_values("predicted_points").iloc[-1]
    if style == "nhl_rank":  # NHL.com's F/D rank, G/T only when F/D are full or late
        fd = cand[cand["pos_group"].isin(["F", "D"])]
        rounds_left = sum(open_.values())
        if not fd.empty and rounds_left > open_["G"] + open_["TEAM"] + 1:
            return fd.assign(k=fd["nhl_projection"].fillna(fd["predicted_points"])).sort_values("k").iloc[-1]
        return cand.sort_values("VORP").iloc[-1]
    if style == "random_top":
        return cand.sort_values("VORP", ascending=False).iloc[rng.randrange(min(8, len(cand)))]
    if style == "d_heavy":
        d = cand[cand["pos_group"] == "D"]
        return (d if not d.empty else cand).sort_values("VORP").iloc[-1]
    raise ValueError(style)


# --------------------------------------------------------------------------
# Keepers
# --------------------------------------------------------------------------

def enter_keepers(season: str, boards, holders: list[str]) -> None:
    board, goalies, _ = boards
    fd = board.assign(keeper_value=keeper.keeper_value(board))
    # Realistic keepers: young players from the middle of the board.
    pool = fd[(fd["Age"] <= 25) & fd["VORP"].between(-5, 25)].sort_values("keeper_value", ascending=False)
    for m, (_, row) in zip(holders, pool.iterrows()):
        draft_state.set_keeper(season, row["player_id"], row["Player"], row["pos_group"], m)
    keepers = draft_state.load_keepers(season)
    C.check(len(keepers) == len(holders), f"expected {len(holders)} keepers, stored {len(keepers)}")

    # Validation the sidebar form relies on.
    g = goalies.iloc[0]
    try:
        draft_state.set_keeper(season, g["player_id"], g["Player"], "G", "Phil")
        C.check(False, "a goalie keeper was accepted")
    except ValueError:
        C.check(True, "")
    k = keepers.iloc[0]
    try:
        draft_state.set_keeper(season, k["player_id"], k["player_name"], k["pos_group"], "Phil")
        C.check(False, "the same player was accepted as keeper for two managers")
    except ValueError:
        C.check(True, "")
    # Replacing and removing a keeper.
    spare = pool.iloc[len(holders)]
    draft_state.set_keeper(season, spare["player_id"], spare["Player"], spare["pos_group"], "Phil")
    C.check(len(draft_state.load_keepers(season)) == len(holders) + 1, "adding Phil's keeper failed")
    draft_state.remove_keeper(season, "Phil")
    C.check(len(draft_state.load_keepers(season)) == len(holders), "removing Phil's keeper failed")


# --------------------------------------------------------------------------
# Headless app
# --------------------------------------------------------------------------

def run_app(label: str, at=None, timeout: int = 900):
    from streamlit.testing.v1 import AppTest

    t0 = time.time()
    if at is None:
        at = AppTest.from_file(str(SRC / "app.py"), default_timeout=timeout)
    at.run(timeout=timeout)
    ok = C.check(not at.exception, f"[app {label}] exception: {[e.value for e in at.exception]}")
    C.check(not at.error, f"[app {label}] st.error: {[e.value for e in at.error]}")
    if not ok:
        for e in at.exception:
            print("\n".join(e.stack_trace))
    print(f"  app render '{label}': {time.time() - t0:.1f}s, {len(at.warning)} warning(s)")
    return at


def check_app_banner(at, season: str, order: list[str], settings: dict, label: str) -> None:
    infos = [i.value for i in at.info]
    picks = draft_state.load_picks(season)
    reserved = draft_state.reserved_slots(season, order)
    slot = draft_state.next_slot(picks, reserved)
    total = len(order) * draft_state.total_rounds(settings)
    if slot >= total:
        return
    on_clock = draft_state.snake_manager(order, slot)
    label_of = MY_TEAM if on_clock == "me" else on_clock
    banner = [i for i in infos if i.startswith("On the clock")]
    C.check(banner and f"**{label_of}**" in banner[0] and f"pick #{len(picks) + 1}," in banner[0]
            and f"round {slot // len(order) + 1} of {draft_state.total_rounds(settings)}" in banner[0],
            f"[app {label}] banner wrong: {banner} (expected {label_of}, pick #{len(picks) + 1})")


def ui_pick(at, season: str, player_id: str, pos_group: str, label: str):
    """Checks ``player_id``'s row in the tab's table and presses the side
    panel's "Draft ..." button -- the real draft-day path."""
    prefix, tab_df = {"F": ("fd", "FD"), "D": ("fd", "FD"), "G": ("g", "G"), "TEAM": ("team", "TEAM")}[pos_group]
    tbl_base = {"fd": "fd_table", "g": "g_table", "team": "team_table"}[prefix]
    version_key = f"{tbl_base}_version"
    key = f"{tbl_base}_v{at.session_state[version_key] if version_key in at.session_state else 0}"
    # Row index in the table as the app displays it (default filter/sort).
    settings = settings_for(season)
    av = available(season, _BOARDS, settings)[tab_df]
    if prefix == "fd" and keeper.roster_status(
        draft_state.roster_picks(season), draft_state.load_keepers(season), settings
    )["final_pick"] and draft_state.active_manager(season) == "me":
        av = av.sort_values("keeper_value", ascending=False)
    row = av.reset_index(drop=True).index[av["player_id"].reset_index(drop=True) == player_id]
    assert len(row) == 1, f"{player_id} not in the {prefix} table"
    from streamlit.util import AttributeDictionary

    def select_row() -> None:
        # AppTest doesn't carry a dataframe's selection over between runs
        # (it isn't one of its widgets), so it's set again before each run.
        at.session_state[key] = AttributeDictionary(
            {"selection": AttributeDictionary({"rows": [int(row[0])], "columns": [], "cells": []})}
        )

    select_row()
    at = run_app(f"{label}: select", at)
    buttons = [b for b in at.button if b.label.startswith("Draft ") and prefix + "_pick_form" in str(b.form_id)]
    if not C.check(len(buttons) == 1, f"[app {label}] expected one enabled Draft button in {prefix}, got "
                                      f"{[b.label for b in at.button if b.label.startswith('Draft')]}"):
        return at
    before = len(draft_state.load_picks(season))
    buttons[0].click()
    select_row()
    at = run_app(f"{label}: draft", at)
    picks = draft_state.load_picks(season)
    C.check(len(picks) == before + 1 and picks.iloc[-1]["player_id"] == player_id,
            f"[app {label}] Draft button didn't record {player_id}")
    return at


_BOARDS = None


# --------------------------------------------------------------------------
# The draft
# --------------------------------------------------------------------------

def check_after_pick(season: str, order: list[str], settings: dict, boards, pool: pd.DataFrame) -> None:
    picks = draft_state.load_picks(season)
    rostered = draft_state.roster_picks(season)
    C.check(rostered["player_id"].is_unique, f"a player is rostered twice after pick {len(picks)}")
    C.check(list(picks["pick_number"]) == list(range(1, len(picks) + 1)), "pick numbers not 1..n")
    reserved = draft_state.reserved_slots(season, order)
    C.check(not set(picks["slot"].dropna().astype(int)) & reserved, "a live pick landed on a keeper slot")
    for m in order:
        o = open_slots(season, m, settings)
        C.check(min(o.values()) >= 0, f"{m} is over a roster limit: {o}")
    table = standings.projected_standings(pool, rostered, order, settings)
    C.check(table["unmatched"].sum() == 0, f"rostered players missing from every board: pick {len(picks)}")
    C.check(table["projected_total"].notna().all(), f"NaN projected total after pick {len(picks)}")


def run_draft(season: str, order: list[str], boards, seed: int = 7, at=None, app_checks: bool = False,
              verbose: bool = True) -> None:
    settings = settings_for(season)
    rng = random.Random(seed)
    n = len(order)
    rounds = draft_state.total_rounds(settings)
    reserved = draft_state.reserved_slots(season, order)
    keepers = draft_state.load_keepers(season)
    C.check(settings["num_managers"] == n, f"num_managers {settings['num_managers']} != draft order length {n}")
    C.check(len(reserved) == len(keepers), f"{len(keepers)} keepers but {len(reserved)} reserved slots")
    for _, k in keepers.iterrows():
        s = draft_state.keeper_slot(order, k["manager"], rounds)
        C.check(s // n == rounds - 1 and draft_state.snake_manager(order, s) == k["manager"],
                f"{k['manager']}'s keeper slot {s} isn't his final-round turn")
    expected_live = n * rounds - len(keepers)
    my_expected_turns = standings.my_turns(order, 0, reserved, rounds, n=rounds)
    pool = standings.points_pool(*boards)
    my_turns_taken: list[int] = []
    my_final_seen = False
    mid_app_done = final_app_done = False
    t0 = time.time()

    while True:
        picks = draft_state.load_picks(season)
        slot = draft_state.next_slot(picks, reserved)
        if slot >= n * rounds:
            break
        pick_no = len(picks) + 1
        manager = draft_state.active_manager(season)
        C.check(manager == draft_state.snake_manager(order, slot), f"pick {pick_no}: active manager mismatch")
        C.check(slot not in reserved, f"pick {pick_no}: on a keeper slot")
        C.check(sum(open_slots(season, manager, settings).values()) > 0,
                f"pick {pick_no}: {manager} on the clock with a full roster")

        av = available(season, boards, settings)
        drafted = draft_state.drafted_player_ids(season)
        for k, df in av.items():
            C.check(not df["player_id"].isin(drafted).any(), f"pick {pick_no}: drafted player still in {k} table")
            if not df.empty:
                C.check(df["VORP"].notna().all(), f"pick {pick_no}: NaN VORP in {k} table")

        if manager == "me":
            my_turns_taken.append(slot)
            status = keeper.roster_status(draft_state.roster_picks(season), draft_state.load_keepers(season), settings)
            outlook, info = standings.position_outlook(
                pool, draft_state.roster_picks(season), order, settings, order, slot, reserved
            )
            C.check(not outlook.empty and info["picks_before_next"] == 0,
                    f"pick {pick_no}: roster-needs table wrong on my turn: {info}")
            C.check(outlook["best_now"].notna().all(), f"pick {pick_no}: roster-needs has NaN best_now")
            if status["final_pick"]:
                my_final_seen = True
                if at is not None and not final_app_done:
                    at = run_app("my final pick (keeper)", at)
                    check_app_banner(at, season, order, settings, "final pick")
                    C.check(any("becomes your keeper" in i.value for i in at.info),
                            "[app final pick] no 'becomes your keeper' notice")
                    final_app_done = True

        choice = choose_pick(season, manager, STYLES.get(manager, "vorp"), av, settings, rng)

        if at is not None and pick_no in UI_PICKS:
            at = ui_pick(at, season, choice["player_id"], choice["pos_group"], f"pick {pick_no}")
            if len(draft_state.load_picks(season)) == pick_no:
                C.check(draft_state.load_picks(season).iloc[-1]["manager"] == manager,
                        f"pick {pick_no}: UI pick recorded for the wrong manager")
            else:  # UI pick failed (already reported) -- make it directly so the draft goes on
                draft_state.add_pick(season, choice["player_id"], choice["name"], choice["pos_group"], manager)
        else:
            draft_state.add_pick(season, choice["player_id"], choice["name"], choice["pos_group"], manager)

        # --- scripted mishaps ----------------------------------------------
        if pick_no == 5:  # duplicate: a player already drafted
            first = draft_state.load_picks(season).iloc[0]
            before = len(draft_state.load_picks(season))
            draft_state.add_pick(season, first["player_id"], first["player_name"], first["pos_group"], order[0])
            C.check(len(draft_state.load_picks(season)) == before, "a player was drafted twice")
        if pick_no == 23:  # undo then redo the same pick
            undone = draft_state.load_picks(season).iloc[-1]
            draft_state.undo_last_pick(season)
            C.check(len(draft_state.load_picks(season)) == pick_no - 1, "undo didn't remove the pick")
            C.check(draft_state.active_manager(season) == manager, "undo didn't restore the on-clock manager")
            C.check(undone["player_id"] not in draft_state.drafted_player_ids(season), "undone player still drafted")
            draft_state.add_pick(season, undone["player_id"], undone["player_name"], undone["pos_group"], manager)
        if pick_no == 37:  # the next manager's pick gets entered by mistake, then undone
            nxt = draft_state.snake_manager(order, draft_state.slot_after(draft_state.next_slot(
                draft_state.load_picks(season), reserved), reserved))
            wrong = draft_state.snake_manager(order, draft_state.slot_after(draft_state.next_slot(
                draft_state.load_picks(season), reserved) + 1, reserved))
            if wrong != nxt:
                av2 = available(season, boards, settings)
                c2 = choose_pick(season, wrong, "vorp", av2, settings, rng)
                draft_state.add_pick(season, c2["player_id"], c2["name"], c2["pos_group"], wrong)
                after_wrong = draft_state.active_manager(season)
                draft_state.undo_last_pick(season)
                C.check(draft_state.active_manager(season) == nxt,
                        f"after out-of-turn pick + undo, on clock is {draft_state.active_manager(season)} not {nxt}")
                C.check(after_wrong is not None, "no active manager after an out-of-turn pick")

        check_after_pick(season, order, settings, boards, pool)

        if at is not None and not mid_app_done and pick_no >= expected_live // 2:
            at = run_app("mid-draft", at)
            check_app_banner(at, season, order, settings, "mid-draft")
            # Undo from the sidebar button, then put the pick back.
            last = draft_state.load_picks(season).iloc[-1]
            undo = [b for b in at.button if b.label == "Undo last pick"]
            if C.check(len(undo) == 1, "[app] no 'Undo last pick' button"):
                undo[0].click()
                at = run_app("mid-draft: undo", at)
                C.check(len(draft_state.load_picks(season)) == pick_no - 1, "[app] Undo button didn't undo")
                check_app_banner(at, season, order, settings, "after undo")
                draft_state.add_pick(season, last["player_id"], last["player_name"], last["pos_group"], last["manager"])
            mid_app_done = True

    # --- end of draft ------------------------------------------------------
    picks = draft_state.load_picks(season)
    rostered = draft_state.roster_picks(season)
    C.check(len(picks) == expected_live, f"{len(picks)} live picks, expected {expected_live}")
    C.check(draft_state.active_manager(season) == draft_state.snake_manager(order, n * rounds),
            "active manager after the draft")  # just must not raise
    for m in order:
        o = open_slots(season, m, settings)
        C.check(all(v == 0 for v in o.values()), f"{m}'s roster isn't exactly full: open {o}")
    C.check(my_turns_taken == my_expected_turns, f"my picks came at {my_turns_taken}, predicted {my_expected_turns}")
    my_last = picks[picks["manager"] == "me"].iloc[-1]
    C.check(my_final_seen, "the final-pick (keeper) state never triggered for me")
    C.check(my_last["pos_group"] in keeper.KEEPER_POSITIONS, f"my last pick {my_last['player_name']} isn't a skater")
    for m in keepers["manager"]:
        C.check(int(picks.loc[picks["manager"] == m, "slot"].max()) < (rounds - 1) * n,
                f"{m} holds a keeper but picked in the final round")
    status = keeper.roster_status(rostered, draft_state.load_keepers(season), settings)
    C.check(status["picks_left"] == 0, "My Pool still shows picks left after the draft")
    table = standings.projected_standings(pool, rostered, order, settings)
    C.check((table["open_slots"] == 0).all(), "open slots left in final standings")
    C.check(abs(table["projected_total"] - table["drafted_points"]).max() < 1e-6,
            "final projected totals include unfilled slots")
    outlook, _ = standings.position_outlook(pool, rostered, order, settings, order, n * rounds, reserved)
    C.check(outlook.empty, "roster needs not empty after the draft")

    if verbose:
        print(f"  {len(picks)} live picks + {len(keepers)} keepers in {time.time() - t0:.1f}s")
        disp = table.assign(manager=table["manager"].replace("me", MY_TEAM))
        print(disp[["rank", "manager", "picks", "projected_total"]].round(0).to_string(index=False))
        print(f"  my keeper: {my_last['player_name']} ({my_last['pos_group']})")
    if at is not None:
        at = run_app("draft complete", at)
        C.check(any("Your keeper for next season" in s.value for s in at.success),
                "[app end] My Pool doesn't show my keeper")
    return at


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--my-pos", type=int, default=4, help="your 0-based draft slot for the main run (default 4)")
    ap.add_argument("--no-app", action="store_true", help="skip the headless Streamlit app checks")
    ap.add_argument("--offline", action="store_true", help="no NHL API / ESPN calls (implies --no-app)")
    ap.add_argument("--keep", action="store_true", help="keep the temp directory for inspection")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    use_app = not (args.no_app or args.offline)

    tmp = Path(tempfile.mkdtemp(prefix="draft_smoke_"))
    isolate(tmp)
    print(f"temp state dir: {tmp}")
    global _BOARDS
    try:
        print(f"\n== main draft: 10 managers, you at slot {args.my_pos + 1} ==")
        order = setup_season(SEASON, args.my_pos)
        settings = settings_for(SEASON)
        C.check(settings["num_managers"] == 10, f"num_managers is {settings['num_managers']}, not 10")
        at = None
        t0 = time.time()
        if use_app:
            at = run_app("start (builds boards)")
            print(f"  (board build + first render {time.time() - t0:.0f}s)")
            C.check(not any("rate-limited" in w.value or "couldn't be reached" in w.value for w in at.sidebar.warning),
                    f"live feed warning: {[w.value for w in at.sidebar.warning]}")
        else:
            build_boards_directly(SEASON, settings, live=not args.offline)
            print(f"  (board build {time.time() - t0:.0f}s)")
        C.check(all(p.exists() for p in board_paths(SEASON).values()), "board CSVs weren't saved")
        _BOARDS = boards = load_boards(SEASON)
        check_boards(*boards, settings)

        enter_keepers(SEASON, boards, KEEPER_HOLDERS)
        if at is not None:
            at = run_app("keepers entered", at)
            check_app_banner(at, SEASON, order, settings, "keepers entered")
            exp = [e for e in at.sidebar.expander if e.label.startswith("Keepers from last season")]
            C.check(exp and f"({len(KEEPER_HOLDERS)})" in exp[0].label, "[app] keeper count wrong in the sidebar")
        at = run_draft(SEASON, order, boards, at=at, app_checks=use_app)

        print("\n== logic-only replays: you at each draft slot ==")
        for pos in range(10):
            season = f"smoke-slot-{pos + 1}"
            order = setup_season(season, pos)
            # Share the main season's boards (same settings).
            for k, p in board_paths(SEASON).items():
                board_paths(season)[k].parent.mkdir(parents=True, exist_ok=True)
                board_paths(season)[k].write_bytes(p.read_bytes())
            enter_keepers(season, boards, KEEPER_HOLDERS)
            fails = len(C.failures)
            run_draft(season, order, boards, seed=pos, verbose=False)
            print(f"  slot {pos + 1:2d}: {'ok' if len(C.failures) == fails else 'FAILED'}")
    except Exception:
        traceback.print_exc()
        C.failures.append("unhandled exception (see traceback)")
    finally:
        if args.keep:
            print(f"\nkept {tmp}")
        else:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{C.passed} checks passed, {len(C.failures)} failed")
    for f in C.failures[:40]:
        print(f"  - {f}")
    return 1 if C.failures else 0


if __name__ == "__main__":
    sys.exit(main())
