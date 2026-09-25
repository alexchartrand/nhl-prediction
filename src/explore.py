"""On-demand player and team research via Mistral's web-search-grounded chat.

Manual, user-clicked feature (the "Explore" button in app.py) -- not part of the
board build pipeline, so unlike nhl_api.py's "never raise" convention, a failure
here is surfaced to the user as an error message rather than silently degraded,
since there's no sensible default to fall back to for "what does the web say
about this player right now."

Requires the MISTRAL_API_KEY environment variable to be set before launching
Streamlit; there's no other secrets convention in this repo to plug into.
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv()

MISTRAL_MODEL = "mistral-medium-latest"


class ExploreError(Exception):
    pass


@st.cache_resource
def _client() -> Mistral:
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ExploreError(
            "MISTRAL_API_KEY is not set. Set it as an environment variable before "
            "launching Streamlit to enable player exploration."
        )
    return Mistral(api_key=api_key)


def build_player_context(row: pd.Series, hist: pd.DataFrame) -> str:
    """Formats one player's known local data into a short grounding block for the
    prompt, so the model isn't relying on web search alone for baseline stats."""
    lines = [
        f"Player: {row['Player']}",
        f"Team: {row.get('Team', 'unknown')}",
        f"Position: {row.get('Pos', row.get('pos_group', 'unknown'))}",
        f"Age: {row.get('Age', 'unknown')}",
    ]
    notes = row.get("Notes")
    if isinstance(notes, str) and notes:
        lines.append(f"Local notes: {notes}")

    if not hist.empty:
        last = hist.sort_values("season", ascending=False).iloc[0]
        stat_bits = [f"GP {last.get('GP')}", f"G {last.get('G')}", f"A {last.get('A')}", f"PTS {last.get('PTS')}"]
        if "fantasy_points" in last.index and pd.notna(last["fantasy_points"]):
            stat_bits.append(f"fantasy_points {last['fantasy_points']:.0f}")
        lines.append(f"Last loaded season ({last['season']}): " + ", ".join(stat_bits))
    else:
        lines.append("No local season history available.")

    return "\n".join(lines)


def build_team_context(row: pd.Series) -> str:
    """Grounding block for one NHL team (a row of draft_pool.team_pool) --
    NHL.com's projection is the only local data there is for teams."""
    lines = [f"Team: {row['Team']} ({row.get('Code', '')})"]
    wins = row.get("projected_wins")
    if pd.notna(wins):
        delta = row.get("win_delta")
        change = f" ({int(delta):+d} vs last season)" if pd.notna(delta) else ""
        lines.append(f"NHL.com projected wins: {int(wins)}{change}")
    points = row.get("predicted_points")
    if pd.notna(points):
        lines.append(f"Projected standings points: {points:.0f}")
    return "\n".join(lines)


# subject -> (label, scoring note, what to search the web for, pronoun)
_SUBJECTS = {
    "player": (
        "Player",
        "This is for a fantasy hockey pool scored as goals + assists + 1 bonus "
        "point per short-handed goal (skaters); pool has 9F/5D/1G/1TEAM rosters.",
        "current status: recent form, any injury/IR status, and any notable recent news",
        "him",
    ),
    "team": (
        "Team",
        "This is for a fantasy hockey pool where each manager drafts one NHL team, "
        "scored on that team's regular-season standings points (2 per win, 1 per "
        "OT/shootout loss); pool has 9F/5D/1G/1TEAM rosters.",
        "outlook for this regular season: offseason roster changes, goaltending, "
        "key injuries, and any notable recent news",
        "it",
    ),
}


def explore_players(contexts: list[str], subject: str = "player") -> str:
    """Summarizes (1 player/team) or compares and recommends (2-3) using
    Mistral's web-search tool for current news/injury/form. ``subject`` is
    "player" or "team" (see _SUBJECTS). Returns markdown. Raises
    ExploreError on any failure (missing key, network, API error)."""
    label, scoring_note, search_for, pronoun = _SUBJECTS[subject]
    if not contexts:
        raise ExploreError(f"No {label.lower()}s selected.")

    block = "\n\n".join(f"{label} {i + 1}:\n{c}" for i, c in enumerate(contexts))

    if len(contexts) == 1:
        prompt = (
            f"{scoring_note}\n\n{block}\n\n"
            f"Search the web for this {label.lower()}'s {search_for}. Give a concise summary "
            f"(a few sentences) useful for deciding whether to draft {pronoun} now."
        )
    else:
        prompt = (
            f"{scoring_note}\n\n{block}\n\n"
            f"Search the web for each {label.lower()}'s {search_for}. Then recommend which "
            "one to draft first for this pool, with brief reasoning referencing "
            "both the local stats above and what you found on the web."
        )

    try:
        response = _client().beta.conversations.start(
            model=MISTRAL_MODEL,
            inputs=prompt,
            tools=[{"type": "web_search"}],
        )
    except ExploreError:
        raise
    except Exception as e:
        raise ExploreError(f"Mistral request failed: {e}") from e

    return _extract_text(response)


def _extract_text(response) -> str:
    parts = []
    for entry in response.outputs:
        if getattr(entry, "type", None) != "message.output":
            continue
        content = entry.content
        if isinstance(content, str):
            parts.append(content)
        else:
            for chunk in content:
                if getattr(chunk, "type", None) == "text":
                    parts.append(chunk.text)
    text = "".join(parts).strip()
    if not text:
        raise ExploreError("Mistral returned an empty response.")
    return text
