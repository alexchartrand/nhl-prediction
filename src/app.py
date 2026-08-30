"""Local Streamlit app for live draft-day use.

Run with:
    .venv/Scripts/python.exe -m streamlit run src/app.py

Lets the user browse the ranked F/D draft board and an unranked goalie list,
filter by position/name, mark players as drafted (by themselves or another
manager) with VORP recomputed on who's left, undo a pick, and look up any
player's season history. Picks and manager names persist to ``state/`` so
they survive an app restart mid-draft.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import draft_pool
import draft_state
import features
import rank

st.set_page_config(page_title="Draft Assistant", layout="wide")


BOARD_PATH = rank.OUTPUT_PATH


def _rebuild_board() -> pd.DataFrame:
    board = rank.build_draft_board()
    BOARD_PATH.parent.mkdir(exist_ok=True)
    board.to_csv(BOARD_PATH, index=False)
    return board


@st.cache_data
def get_board() -> pd.DataFrame:
    if BOARD_PATH.exists():
        return pd.read_csv(BOARD_PATH)
    return _rebuild_board()


@st.cache_data
def get_all_seasons() -> pd.DataFrame:
    return features.load_scored_seasons()


@st.cache_data
def get_goalies(_all_seasons: pd.DataFrame) -> pd.DataFrame:
    return draft_pool.goalie_pool(_all_seasons)


def manager_options(managers: dict) -> tuple[list[str], dict[str, str]]:
    options = ["me"] + managers.get("other_managers", [])
    labels = {"me": managers.get("my_team_name", "me")}
    return options, labels


def manager_setup_form(existing: dict | None) -> None:
    container = st.sidebar.expander("Edit managers", expanded=False) if existing else st.sidebar
    with container:
        if not existing:
            st.subheader("Set up your league")
        with st.form("manager_setup"):
            my_team = st.text_input("My team name", value=(existing or {}).get("my_team_name", "Me"))
            others_default = "\n".join((existing or {}).get("other_managers", []))
            others_text = st.text_area("Other managers (one per line)", value=others_default, height=180)
            submitted = st.form_submit_button("Save")
        if submitted:
            others = [line.strip() for line in others_text.splitlines() if line.strip()]
            draft_state.save_managers(my_team, others)
            st.rerun()


def render_history(player_id: str, player_name: str, pos_group: str, all_seasons: pd.DataFrame) -> None:
    with st.expander(f"{player_name} — season history"):
        hist = all_seasons[all_seasons["player_id"] == player_id].sort_values("season")
        if hist.empty:
            st.write("No season history found.")
            return
        cols = ["season", "Team", "Pos", "GP", "G", "A", "PTS", "SOG", "PPG", "PP"]
        if pos_group != "G":
            cols.append("fantasy_points")
        st.dataframe(hist[cols], hide_index=True, width="stretch")


def pick_form(player_id: str, player_name: str, pos_group: str, options: list[str], labels: dict, key_prefix: str) -> None:
    with st.form(f"{key_prefix}_pick_form"):
        manager = st.selectbox("Drafted by", options, format_func=lambda m: labels.get(m, m), key=f"{key_prefix}_manager")
        submitted = st.form_submit_button(f"Draft {player_name}")
    if submitted:
        draft_state.add_pick(player_id, player_name, pos_group, manager)
        st.rerun()


def render_selectable_table(df: pd.DataFrame, display_cols: list[str], key: str):
    """Renders df[display_cols] with single-row selection; returns the
    selected row (full row, from df, not just the displayed columns) or None."""
    event = st.dataframe(
        df[display_cols],
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key=key,
    )
    rows = event.selection.rows if event and event.selection else []
    return df.iloc[rows[0]] if rows else None


def forwards_defense_tab(board: pd.DataFrame, all_seasons: pd.DataFrame, options: list[str], labels: dict) -> None:
    drafted = draft_state.drafted_player_ids()
    available = draft_pool.undrafted_board(board, drafted)

    col1, col2 = st.columns([1, 2])
    with col1:
        pos_filter = st.selectbox("Position", ["All", "F", "D"], key="fd_pos_filter")
    with col2:
        name_query = st.text_input("Search player name", key="fd_name_query")

    filtered = available
    if pos_filter != "All":
        filtered = filtered[filtered["pos_group"] == pos_filter]
    if name_query:
        filtered = filtered[filtered["Player"].str.contains(name_query, case=False, na=False)]
    filtered = filtered.reset_index(drop=True)

    display_cols = ["Player", "Team", "Pos", "Age", "GP", "Notes", "predicted_points", "pos_rank", "VORP"]
    selected = render_selectable_table(filtered, display_cols, key="fd_table")
    st.caption(f"{len(filtered)} available players shown")

    if selected is not None:
        st.divider()
        render_history(selected["player_id"], selected["Player"], selected["pos_group"], all_seasons)
        pick_form(selected["player_id"], selected["Player"], selected["pos_group"], options, labels, key_prefix="fd")

    with st.expander("Show drafted forwards/defense"):
        picks = draft_state.load_picks()
        drafted_board = board[board["player_id"].isin(drafted)].merge(
            picks[["player_id", "manager", "pick_number"]], on="player_id", how="left"
        )
        drafted_board["manager"] = drafted_board["manager"].map(lambda m: labels.get(m, m))
        st.dataframe(
            drafted_board[["Player", "Team", "Pos", "manager", "pick_number"]].sort_values("pick_number"),
            hide_index=True,
            width="stretch",
        )


def goalies_tab(all_seasons: pd.DataFrame, options: list[str], labels: dict) -> None:
    drafted = draft_state.drafted_player_ids()
    goalies = get_goalies(all_seasons)
    goalies = goalies[~goalies["player_id"].isin(drafted)]

    name_query = st.text_input("Search goalie name", key="g_name_query")
    if name_query:
        goalies = goalies[goalies["Player"].str.contains(name_query, case=False, na=False)]
    goalies = goalies.reset_index(drop=True)

    display_cols = ["Player", "Team", "GP", "feature_season", "seasons_back"]
    selected = render_selectable_table(goalies, display_cols, key="g_table")
    st.caption(f"{len(goalies)} available goalies shown")

    if selected is not None:
        st.divider()
        render_history(selected["player_id"], selected["Player"], "G", all_seasons)
        pick_form(selected["player_id"], selected["Player"], "G", options, labels, key_prefix="g")

    with st.expander("Show drafted goalies"):
        picks = draft_state.load_picks()
        drafted_goalies = picks[picks["pos_group"] == "G"].sort_values("pick_number").copy()
        drafted_goalies["manager"] = drafted_goalies["manager"].map(lambda m: labels.get(m, m))
        st.dataframe(
            drafted_goalies[["player_name", "manager", "pick_number"]],
            hide_index=True,
            width="stretch",
        )


def draft_log_tab(labels: dict) -> None:
    picks = draft_state.load_picks().sort_values("pick_number", ascending=False).copy()
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

    managers = draft_state.load_managers()
    if not managers:
        manager_setup_form(None)
        st.info("Set up your league in the sidebar to begin.")
        return

    manager_setup_form(managers)
    options, labels = manager_options(managers)

    st.sidebar.divider()
    if st.sidebar.button("Undo last pick"):
        draft_state.undo_last_pick()
        st.rerun()

    if BOARD_PATH.exists():
        updated = pd.Timestamp(BOARD_PATH.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M")
        st.sidebar.caption(f"Draft board last computed: {updated}")
    if st.sidebar.button("Recompute draft board"):
        with st.spinner("Recomputing draft board (refits models, hits the live NHL API)..."):
            _rebuild_board()
        get_board.clear()
        st.rerun()

    picks = draft_state.load_picks()
    if not picks.empty:
        last = picks.sort_values("pick_number").iloc[-1]
        st.sidebar.caption(
            f"Last pick: #{int(last['pick_number'])} {last['player_name']} → {labels.get(last['manager'], last['manager'])}"
        )

    board = get_board()
    all_seasons = get_all_seasons()

    tab_fd, tab_g, tab_log = st.tabs(["Forwards & Defense", "Goalies", "Draft Log"])
    with tab_fd:
        forwards_defense_tab(board, all_seasons, options, labels)
    with tab_g:
        goalies_tab(all_seasons, options, labels)
    with tab_log:
        draft_log_tab(labels)


if __name__ == "__main__":
    main()
