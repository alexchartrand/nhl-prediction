"""Local Streamlit app for live draft-day use.

Run with:
    .venv/Scripts/python.exe -m streamlit run src/app.py

Lets the user browse the ranked F/D draft board and the unranked goalie/team
lists, filter by position/name, mark players (or a team) as drafted (by
themselves or another manager) with VORP recomputed on who's left, undo a
pick, and look up any player's season history. "My Pool" shows the user's
own roster against the 9F/5D/1G/1TEAM targets. Picks and manager names
persist to ``state/`` so they survive an app restart mid-draft.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import draft_pool
import draft_state
import explore
import features
import goalies as goalies_module
import rank

# First three slots of the validated categorical palette (dataviz skill) --
# these three pass the CVD/contrast floors together in both light and dark.
COMPARE_LINE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
GOALIE_HISTORY_COLS = ["season", "Team", "GP", "GS", "W", "L", "T/O", "SV%", "GAA", "SO", "fantasy_points"]
HISTORY_COLS = ["season", "Team", "Pos", "GP", "G", "A", "PTS", "SOG", "PPG", "PP"]

st.set_page_config(page_title="Draft Assistant", layout="wide")


def board_path(season: str) -> Path:
    """Each season's board is persisted separately -- its VORP/pos_rank
    columns bake in that season's num_managers/forwards/defense settings
    (see draft_state.load_settings), so one season's board can't double as
    another's."""
    return rank.OUTPUT_PATH.parent / f"draft_board_{draft_state.slugify(season)}.csv"


def _rebuild_board(season: str, settings: dict) -> pd.DataFrame:
    board = rank.build_draft_board(
        teams=settings["num_managers"],
        roster={"F": settings["forwards"], "D": settings["defense"]},
    )
    path = board_path(season)
    path.parent.mkdir(exist_ok=True)
    board.to_csv(path, index=False)
    st.session_state["team_fetch_complete"] = board.attrs.get("team_fetch_complete", True)
    return board


@st.cache_data
def get_board(season: str, settings: dict) -> pd.DataFrame:
    path = board_path(season)
    if path.exists():
        return pd.read_csv(path)
    return _rebuild_board(season, settings)


@st.cache_data
def get_all_seasons() -> pd.DataFrame:
    return features.load_scored_seasons()


@st.cache_data
def get_goalie_seasons() -> pd.DataFrame:
    return goalies_module.load_scored_goalies()


@st.cache_data
def get_goalies() -> pd.DataFrame:
    return draft_pool.goalie_pool(get_goalie_seasons())


@st.cache_data
def get_teams() -> pd.DataFrame:
    return draft_pool.team_pool()


NEW_SEASON_OPTION = "+ New season..."


def season_picker() -> str | None:
    """Sidebar control to create a new season or switch to an existing one.
    All draft state (picks, managers) is scoped to the returned season, so
    a fresh pool year starts clean while past drafts stay intact and
    reloadable. Selection is stuck in the URL query params so a browser
    refresh doesn't drop back to "no season selected".

    The selectbox's own widget state (key "season_select") is the single
    source of truth for the current selection. Streamlit forbids writing to
    a keyed widget's session_state after it's been instantiated in the same
    run, so switching the dropdown to a just-created season is staged via
    "_pending_season" and applied at the top of the *next* run, before the
    widget is instantiated.
    """
    seasons = draft_state.list_seasons()

    pending = st.session_state.pop("_pending_season", None)
    if pending is not None:
        st.session_state["season_select"] = pending

    st.sidebar.subheader("Season")
    options = seasons + [NEW_SEASON_OPTION]
    if "season_select" not in st.session_state:
        query_season = st.query_params.get("season")
        st.session_state["season_select"] = query_season if query_season in seasons else (
            seasons[0] if seasons else NEW_SEASON_OPTION
        )
    choice = st.sidebar.selectbox("Draft season", options, key="season_select")

    if choice == NEW_SEASON_OPTION:
        new_name = st.sidebar.text_input("New season name (e.g. 2026-2027)", key="new_season_name")
        if st.sidebar.button("Create season") and new_name.strip():
            season = draft_state.create_season(new_name.strip())
            st.session_state["_pending_season"] = season
            st.query_params["season"] = season
            st.rerun()
        return None

    st.query_params["season"] = choice
    return choice


def manager_options(managers: dict) -> tuple[list[str], dict[str, str]]:
    options = ["me"] + managers.get("other_managers", [])
    labels = {"me": managers.get("my_team_name", "me")}
    return options, labels


def manager_setup_form(season: str, existing: dict | None) -> None:
    container = st.sidebar.expander("Edit managers", expanded=False) if existing else st.sidebar
    with container:
        if not existing:
            st.subheader("Set up your league")
        with st.form("manager_setup"):
            my_team = st.text_input("My team name", value=(existing or {}).get("my_team_name", "Me"))
            # The draft order is the manager list: everyone, in draw order, including
            # your own team. Seasons saved before that (no order yet) prefill with
            # "me" first plus the old other-managers list.
            _, labels = manager_options(existing or {})
            ids = draft_state.load_draft_order(season) or (
                ["me", *existing.get("other_managers", [])] if existing else []
            )
            order_text = st.text_area(
                "Managers in draft order (one per line, including your own team)",
                value="\n".join(labels.get(m, m) for m in ids),
                height=260,
                help="Round 1 goes first to last, then the order reverses (snake). Each manager once.",
            )
            submitted = st.form_submit_button("Save")
        if submitted:
            names = [line.strip() for line in order_text.splitlines() if line.strip()]
            if my_team not in names:
                st.error("Your team name must appear in the list. Not saved.")
            elif len(set(names)) != len(names):
                st.error("Each manager must appear only once. Not saved.")
            else:
                order = ["me" if name == my_team else name for name in names]
                draft_state.save_managers(season, my_team, [n for n in names if n != my_team], order)
                st.rerun()


def settings_form(season: str, existing: dict) -> None:
    """League shape for this season: pool size (drives the VORP replacement
    level, see rank.add_vorp) and roster slots (drive both VORP and the "My
    Pool" progress tracker). Changing these doesn't retroactively rewrite an
    already-computed board -- hit "Recompute draft board" afterwards."""
    with st.sidebar.expander("League settings", expanded=False):
        with st.form("settings_form"):
            num_managers = st.number_input(
                "Number of managers", min_value=2, max_value=30, value=existing["num_managers"], step=1
            )
            forwards = st.number_input(
                "Forward roster slots", min_value=1, max_value=20, value=existing["forwards"], step=1
            )
            defense = st.number_input(
                "Defense roster slots", min_value=1, max_value=20, value=existing["defense"], step=1
            )
            goalies = st.number_input(
                "Goalie roster slots", min_value=0, max_value=10, value=existing["goalies"], step=1
            )
            team_slots = st.number_input(
                "Team roster slots", min_value=0, max_value=10, value=existing["team_slots"], step=1
            )
            submitted = st.form_submit_button("Save")
        if submitted:
            draft_state.save_settings(
                season,
                {
                    "num_managers": num_managers,
                    "forwards": forwards,
                    "defense": defense,
                    "goalies": goalies,
                    "team_slots": team_slots,
                },
            )
            st.rerun()


def player_history(player_id: str, pos_group: str, all_seasons: pd.DataFrame) -> pd.DataFrame:
    hist = all_seasons[all_seasons["player_id"] == player_id].sort_values("season", ascending=False)
    if hist.empty:
        return hist
    if pos_group == "G":
        return hist[GOALIE_HISTORY_COLS]  # needs the goalie frame (skater export has only G/A for goalies)
    return hist[HISTORY_COLS + ["fantasy_points"]]


def render_compare(rows: pd.DataFrame, all_seasons: pd.DataFrame, key_prefix: str) -> None:
    """Shows season history side by side for the players checked in the draft
    list above (up to 3), and optionally hits "Explore" to have Mistral
    web-search their current news/injury/form and summarize (1 player) or
    recommend a pick between them (2-3 players)."""
    cols = st.columns(len(rows))
    chart_series = []
    for col, (_, row) in zip(cols, rows.iterrows()):
        pos_group = row.get("pos_group", "G")
        hist = player_history(row["player_id"], pos_group, all_seasons)
        with col:
            st.write(f"**{row['Player']}**")
            if hist.empty:
                st.write("No season history found.")
                continue
            st.dataframe(hist, hide_index=True, width="stretch")
            value_col = "fantasy_points" if "fantasy_points" in hist.columns else "PTS"
            chart_series.append(hist.set_index("season")[value_col].rename(row["Player"]))

    if len(chart_series) >= 2:
        chart_df = pd.concat(chart_series, axis=1).sort_index()
        st.line_chart(chart_df, color=COMPARE_LINE_COLORS[: len(chart_series)])

    result_key = f"{key_prefix}_explore_result"
    if st.button("Explore", key=f"{key_prefix}_explore_btn"):
        contexts = [
            explore.build_player_context(row, player_history(row["player_id"], row.get("pos_group", "G"), all_seasons))
            for _, row in rows.iterrows()
        ]
        with st.spinner("Searching the web and summarizing..."):
            try:
                st.session_state[result_key] = explore.explore_players(contexts)
            except explore.ExploreError as e:
                st.session_state[result_key] = None
                st.error(str(e))

    if st.session_state.get(result_key):
        st.markdown(st.session_state[result_key])


ROSTER_SETTING_KEY = {"F": "forwards", "D": "defense", "G": "goalies", "TEAM": "team_slots"}


def pick_form(
    season: str,
    player_id: str | None,
    player_name: str | None,
    pos_group: str | None,
    options: list[str],
    labels: dict,
    key_prefix: str,
    settings: dict,
) -> None:
    enabled = player_id is not None
    # Defaults to whoever's on the clock (snake order) but stays changeable. The
    # widget key includes the pick count so the default re-applies after every
    # pick, while a manual override sticks until the pick is made.
    active = draft_state.active_manager(season)
    index = options.index(active) if active in options else 0
    picks = draft_state.load_picks(season)
    n_picks = len(picks)
    with st.form(f"{key_prefix}_pick_form"):
        manager = st.selectbox(
            "Drafted by",
            options,
            index=index,
            format_func=lambda m: labels.get(m, m),
            key=f"{key_prefix}_manager_{n_picks}",
            disabled=not enabled,
        )
        submitted = st.form_submit_button(f"Draft {player_name}" if enabled else "Draft", disabled=not enabled)
    if submitted and enabled:
        limit = settings[ROSTER_SETTING_KEY[pos_group]]
        have = ((picks["manager"] == manager) & (picks["pos_group"] == pos_group)).sum()
        if have >= limit:
            st.error(
                f"{labels.get(manager, manager)} already has {have} {POS_GROUP_LABELS[pos_group]} "
                f"(limit {limit}) -- undo a pick before drafting another."
            )
        else:
            draft_state.add_pick(season, player_id, player_name, pos_group, manager)
            st.rerun()


def render_selectable_table(df: pd.DataFrame, display_cols: list[str], key: str, selection_mode: str = "single-row"):
    """Renders df[display_cols] with row selection (checkboxes appear in the
    row selector column when selection_mode is "multi-row"). Returns the
    selected row(s) (full row(s), from df, not just the displayed columns) --
    a single row or None for "single-row", a DataFrame for "multi-row"."""
    event = st.dataframe(
        df[display_cols],
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode=selection_mode,
        key=key,
    )
    rows = event.selection.rows if event and event.selection else []
    if selection_mode == "multi-row":
        return df.iloc[rows]
    return df.iloc[rows[0]] if rows else None


def live_search_input(label: str, key: str, debounce_ms: int = 300) -> None:
    """A search-style text_input (``type="search"`` gets the browser's native
    "x" clear button for free) that filters live as the user types, instead
    of Streamlit's default text_input behavior which only commits a new
    value to session_state on Enter or losing focus. Injects a tiny script
    that watches the underlying <input> (found by its label, since Streamlit
    doesn't expose a stable id) and, ``debounce_ms`` after the user stops
    typing or clears it, dispatches a synthetic Enter keypress -- the same
    event Streamlit's own commit-on-Enter handler already listens for, so
    this doesn't fight it or duplicate the value it stores."""
    st.text_input(label, key=key, type="search")
    st.iframe(
        f"""
        <script>
        (function() {{
            const doc = window.parent.document;
            function attach() {{
                const el = doc.querySelector('input[aria-label="{label}"]');
                if (!el || el.dataset.liveSearchAttached) return;
                el.dataset.liveSearchAttached = "1";
                let timer = null;
                el.addEventListener("input", () => {{
                    clearTimeout(timer);
                    timer = setTimeout(() => {{
                        el.dispatchEvent(new KeyboardEvent("keydown", {{
                            key: "Enter", code: "Enter", keyCode: 13, which: 13,
                            bubbles: true, cancelable: true,
                        }}));
                    }}, {debounce_ms});
                }});
            }}
            attach();
            new MutationObserver(attach).observe(doc.body, {{childList: true, subtree: true}});
        }})();
        </script>
        """,
        height=1,
    )


def table_key(base_key: str) -> str:
    """The dataframe widget's key includes a version suffix bumped by
    uncheck_all_button -- popping the base key from session_state doesn't
    reset the frontend's own row-selection UI (it's a custom component that
    doesn't resync from a cleared backend value), so the only reliable way to
    actually clear checked rows is to remount the widget under a brand new
    key."""
    return f"{base_key}_v{st.session_state.get(f'{base_key}_version', 0)}"


def selection_rows(base_key: str) -> list[int]:
    """Reads a dataframe widget's current selection from session_state *before*
    the widget is (re-)instantiated later in the script -- Streamlit updates
    session_state for the key that triggered a rerun before the script starts
    running again, so this reflects the just-clicked checkbox even though the
    table itself renders further down the page."""
    state = st.session_state.get(table_key(base_key))
    return list(state.selection.rows) if state else []


def uncheck_all_button(base_key: str, label: str = "Uncheck all", disabled: bool = False) -> None:
    if st.button(label, key=f"{base_key}_uncheck_all", disabled=disabled):
        st.session_state[f"{base_key}_version"] = st.session_state.get(f"{base_key}_version", 0) + 1
        st.rerun()


def forwards_defense_tab(
    season: str, board: pd.DataFrame, all_seasons: pd.DataFrame, options: list[str], labels: dict, settings: dict
) -> None:
    drafted = draft_state.drafted_player_ids(season)
    available = draft_pool.undrafted_board(
        board, drafted, teams=settings["num_managers"], roster={"F": settings["forwards"], "D": settings["defense"]}
    )

    pos_filter = st.session_state.get("fd_pos_filter", "All")
    name_query = st.session_state.get("fd_name_query", "")
    filtered = available
    if pos_filter != "All":
        filtered = filtered[filtered["pos_group"] == pos_filter]
    if len(name_query) >= 2:
        filtered = filtered[filtered["Player"].str.contains(name_query, case=False, na=False)]
    filtered = filtered.reset_index(drop=True)

    rows = [r for r in selection_rows("fd_table") if r < len(filtered)]
    selected = filtered.iloc[rows]
    if len(selected) > 3:
        st.warning("Up to 3 players can be compared at once -- showing the first 3 checked.")
        selected = selected.iloc[:3]

    if len(selected) == 1:
        row = selected.iloc[0]
        pick_form(season, row["player_id"], row["Player"], row["pos_group"], options, labels, key_prefix="fd", settings=settings)
    else:
        pick_form(season, None, None, None, options, labels, key_prefix="fd", settings=settings)
    st.divider()

    col1, col2 = st.columns([1, 2])
    with col1:
        st.selectbox("Position", ["All", "F", "D"], key="fd_pos_filter")
    with col2:
        live_search_input("Search player name", key="fd_name_query")

    uncheck_all_button("fd_table", disabled=selected.empty)

    display_cols = ["Player", "Team", "Pos", "Age", "GP", "Notes", "predicted_points", "pos_rank", "VORP"]
    st.caption(f"{len(filtered)} available players shown -- check up to 3 to compare, or check exactly 1 to draft")
    render_selectable_table(filtered, display_cols, key=table_key("fd_table"), selection_mode="multi-row")

    if not selected.empty:
        st.divider()
        render_compare(selected, all_seasons, key_prefix="fd")

    with st.expander("Show drafted forwards/defense"):
        picks = draft_state.load_picks(season)
        drafted_board = board[board["player_id"].isin(drafted)].merge(
            picks[["player_id", "manager", "pick_number"]], on="player_id", how="left"
        )
        drafted_board["manager"] = drafted_board["manager"].map(lambda m: labels.get(m, m))
        st.dataframe(
            drafted_board[["Player", "Team", "Pos", "manager", "pick_number"]].sort_values("pick_number"),
            hide_index=True,
            width="stretch",
        )


def goalies_tab(season: str, all_seasons: pd.DataFrame, options: list[str], labels: dict, settings: dict) -> None:
    drafted = draft_state.drafted_player_ids(season)
    goalies = draft_pool.undrafted_goalies(
        get_goalies(), drafted, teams=settings["num_managers"], slots=settings["goalies"]
    )

    name_query = st.session_state.get("g_name_query", "")
    if len(name_query) >= 2:
        goalies = goalies[goalies["Player"].str.contains(name_query, case=False, na=False)]
    goalies = goalies.reset_index(drop=True)

    rows = [r for r in selection_rows("g_table") if r < len(goalies)]
    selected = goalies.iloc[rows]
    if len(selected) > 3:
        st.warning("Up to 3 players can be compared at once -- showing the first 3 checked.")
        selected = selected.iloc[:3]

    if len(selected) == 1:
        row = selected.iloc[0]
        pick_form(season, row["player_id"], row["Player"], "G", options, labels, key_prefix="g", settings=settings)
    else:
        pick_form(season, None, None, None, options, labels, key_prefix="g", settings=settings)
    st.divider()

    live_search_input("Search goalie name", key="g_name_query")

    uncheck_all_button("g_table", disabled=selected.empty)

    goalies = goalies.rename(columns={"predicted_points": "Projected Wins"})
    display_cols = ["Player", "Team", "GP", "Projected Wins", "pos_rank", "VORP", "Notes"]
    st.caption(f"{len(goalies)} available goalies shown -- check up to 3 to compare, or check exactly 1 to draft")
    render_selectable_table(goalies, display_cols, key=table_key("g_table"), selection_mode="multi-row")

    if not selected.empty:
        st.divider()
        render_compare(selected, get_goalie_seasons(), key_prefix="g")

    with st.expander("Show drafted goalies"):
        picks = draft_state.load_picks(season)
        drafted_goalies = picks[picks["pos_group"] == "G"].sort_values("pick_number").copy()
        drafted_goalies["manager"] = drafted_goalies["manager"].map(lambda m: labels.get(m, m))
        st.dataframe(
            drafted_goalies[["player_name", "manager", "pick_number"]],
            hide_index=True,
            width="stretch",
        )


def teams_tab(season: str, options: list[str], labels: dict, settings: dict) -> None:
    drafted = draft_state.drafted_player_ids(season)
    teams = draft_pool.undrafted_teams(
        get_teams(), drafted, teams=settings["num_managers"], slots=settings["team_slots"]
    )

    name_query = st.session_state.get("team_name_query", "")
    if len(name_query) >= 2:
        teams = teams[teams["Team"].str.contains(name_query, case=False, na=False)]
    teams = teams.reset_index(drop=True)

    rows = [r for r in selection_rows("team_table") if r < len(teams)]
    selected = teams.iloc[rows[0]] if rows else None

    if selected is not None:
        pick_form(season, selected["player_id"], selected["Team"], "TEAM", options, labels, key_prefix="team", settings=settings)
    else:
        pick_form(season, None, None, None, options, labels, key_prefix="team", settings=settings)
    st.divider()

    live_search_input("Search team name", key="team_name_query")

    uncheck_all_button("team_table", label="Clear selection", disabled=selected is None)

    display_cols = ["Team", "Code", "projected_wins", "win_delta", "pos_rank", "VORP"]
    st.caption(f"{len(teams)} available teams shown, ranked by NHL.com's projected win total")
    render_selectable_table(teams, display_cols, key=table_key("team_table"))

    with st.expander("Show drafted teams"):
        picks = draft_state.load_picks(season)
        drafted_teams = picks[picks["pos_group"] == "TEAM"].sort_values("pick_number").copy()
        drafted_teams["manager"] = drafted_teams["manager"].map(lambda m: labels.get(m, m))
        st.dataframe(
            drafted_teams[["player_name", "manager", "pick_number"]],
            hide_index=True,
            width="stretch",
        )


POS_GROUP_LABELS = {"F": "Forwards", "D": "Defense", "G": "Goalie", "TEAM": "Team"}


def my_pool_tab(season: str, labels: dict, settings: dict) -> None:
    roster_targets = {
        "F": settings["forwards"],
        "D": settings["defense"],
        "G": settings["goalies"],
        "TEAM": settings["team_slots"],
    }
    picks = draft_state.load_picks(season)
    mine = picks[picks["manager"] == "me"].sort_values("pick_number")

    st.subheader(labels.get("me", "me"))
    cols = st.columns(len(roster_targets))
    for col, (pos, target) in zip(cols, roster_targets.items()):
        have = (mine["pos_group"] == pos).sum()
        col.metric(POS_GROUP_LABELS[pos], f"{have}/{target}")

    if mine.empty:
        st.write("No players drafted yet.")
        return

    for pos, label in POS_GROUP_LABELS.items():
        sub = mine[mine["pos_group"] == pos]
        if sub.empty:
            continue
        st.write(f"**{label}**")
        st.dataframe(
            sub[["pick_number", "player_name"]].rename(columns={"pick_number": "Pick #", "player_name": "Player"}),
            hide_index=True,
            width="stretch",
        )


def draft_log_tab(season: str, labels: dict) -> None:
    picks = draft_state.load_picks(season).sort_values("pick_number", ascending=False).copy()
    if picks.empty:
        st.write("No picks yet.")
        return
    picks["manager"] = picks["manager"].map(lambda m: labels.get(m, m))
    st.dataframe(
        picks[["pick_number", "player_name", "pos_group", "manager", "timestamp"]],
        hide_index=True,
        width="stretch",
    )


def main() -> None:
    st.title("Draft Assistant")

    season = season_picker()
    if not season:
        st.info("Create or select a season in the sidebar to begin.")
        return
    st.caption(f"Season: {season}")

    managers = draft_state.load_managers(season)
    if not managers:
        manager_setup_form(season, None)
        st.info("Set up your league in the sidebar to begin.")
        return

    manager_setup_form(season, managers)
    settings = draft_state.load_settings(season)
    settings_form(season, settings)
    options, labels = manager_options(managers)

    st.sidebar.divider()
    if st.sidebar.button("Undo last pick"):
        draft_state.undo_last_pick(season)
        st.rerun()

    path = board_path(season)
    if path.exists():
        updated = pd.Timestamp(path.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M")
        st.sidebar.caption(f"Draft board last computed: {updated}")
    if st.sidebar.button("Recompute draft board"):
        with st.spinner("Recomputing draft board (refits models, hits the live NHL API)..."):
            _rebuild_board(season, settings)
        get_board.clear()
        st.rerun()
    if st.session_state.get("team_fetch_complete") is False:
        st.sidebar.warning(
            "Live NHL roster check was rate-limited or cut short -- some "
            "'New Team' tags may be missing. Recompute again in a bit."
        )

    picks = draft_state.load_picks(season)
    if not picks.empty:
        last = picks.sort_values("pick_number").iloc[-1]
        st.sidebar.caption(
            f"Last pick: #{int(last['pick_number'])} {last['player_name']} → {labels.get(last['manager'], last['manager'])}"
        )

    order = draft_state.load_draft_order(season)
    if order:
        slot = draft_state.next_slot(picks)
        on_clock = draft_state.snake_manager(order, slot)
        upcoming = draft_state.snake_manager(order, slot + 1)
        st.info(
            f"On the clock: **{labels.get(on_clock, on_clock)}** -- pick #{len(picks) + 1}, "
            f"round {slot // len(order) + 1}. Next: {labels.get(upcoming, upcoming)}"
        )

    board = get_board(season, settings)
    all_seasons = get_all_seasons()

    tab_fd, tab_g, tab_teams, tab_mypool, tab_log = st.tabs(
        ["Forwards & Defense", "Goalies", "Teams", "My Pool", "Draft Log"]
    )
    with tab_fd:
        forwards_defense_tab(season, board, all_seasons, options, labels, settings)
    with tab_g:
        goalies_tab(season, all_seasons, options, labels, settings)
    with tab_teams:
        teams_tab(season, options, labels, settings)
    with tab_mypool:
        my_pool_tab(season, labels, settings)
    with tab_log:
        draft_log_tab(season, labels)


if __name__ == "__main__":
    main()
