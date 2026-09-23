"""Turn point predictions into a draft board via value over replacement.

Predicted points alone aren't a draft order -- a top forward and a top
defenseman aren't worth comparing on raw points, since forwards fill 9 roster
slots per team and defense fills 5. VORP puts every position on the same
scale by measuring each player against the last starter-quality player at
his position (roster slots x number of teams in the pool).

Uses ElasticNet, the model that won both position groups in src/train.py's
holdout evaluation. Refit here on *all* historical season-pairs (train and
holdout) rather than the train-only models train.py saves, since for an
actual draft board there's no reason to hold data back.

Each player's features come from ``loading.latest_healthy_row``, which falls
back to his last season with at least ``features.MIN_GP`` games if his most
recent season was injury-shortened -- so a player who missed most of
2025-26 is still ranked off his last healthy year rather than dropped or
judged on a handful of noisy games. ``seasons_back``/``feature_season`` in
the output show when that fallback kicked in.

Rookies with no usable NHL history (none, or only a few call-up games below
``MIN_GP``) can't be predicted by the model. Those NHL.com projects anyway
(``data/nhl 2026-2027 projections/fowards.txt``/``defense.txt``) are added
with NHL.com's fantasy-point projection standing in as ``predicted_points``
(``source == "nhl.com"``), so they get a VORP and a board rank alongside
modeled players -- two different projection sources in one ranking. Every
player also carries ``nhl_projection`` for side-by-side reference.

Players on ESPN's live injury/suspension list (espn_injuries.py) have
``predicted_points`` scaled down by the share of the season they're expected
to miss before VORP is computed; the unadjusted number is kept as
``healthy_points``.

Each player also gets ``next_season_points``/``future_value`` -- what he'd
be worth as a keeper beyond this season (keeper.py).

F/D only -- goalies and teams are ranked separately off NHL.com's
projections (see draft_pool.goalie_pool / team_pool).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import espn_injuries
import keeper
import loading
import nhl_api
import nhl_projections
import notable
import scoring
from features import FEATURE_COLS, MIN_GP, load_scored_seasons
from train import fit_calibration, full_length_seasons, make_elasticnet

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "output" / "draft_board.csv"
LATEST_SEASON = "2025_2026"
# Positions we have a model for. League shape (pool size, roster slots) is not
# fixed here -- it lives in each season's settings.json (draft_state.load_settings).
MODELED_POSITIONS = ("F", "D")
# How far the injury fallback may reach for a live prediction (see
# loading.latest_healthy_row) -- 1 means "last season's stats if this
# season was a washout." Kept small since there's no way to tell an
# injured-but-active player from a retired one beyond recency.
MAX_SEASONS_BACK = 1


def fit_production_models(pairs: pd.DataFrame, df_all: pd.DataFrame) -> dict:
    """ElasticNet per position, fit on every available season-pair, plus its
    calibration (train.Calibration) back onto a full-season points scale."""
    full_seasons = full_length_seasons(df_all)
    models = {}
    for pos in MODELED_POSITIONS:
        sub = pairs[pairs["pos_group"] == pos]
        model = make_elasticnet().fit(sub[FEATURE_COLS], sub[scoring.TARGET])
        models[pos] = (model, fit_calibration(make_elasticnet, sub, full_seasons))
    return models


def predict_upcoming(models: dict, df_all: pd.DataFrame, as_of_season: str = LATEST_SEASON) -> pd.DataFrame:
    """Predict next season's fantasy points, falling back to each player's
    last healthy season when ``as_of_season`` was injury-shortened."""
    df = loading.latest_healthy_row(df_all, as_of_season, MIN_GP, max_seasons_back=MAX_SEASONS_BACK)
    df = df[df["pos_group"].isin(MODELED_POSITIONS)].copy()

    df["predicted_points"] = float("nan")
    for pos, (model, calibrate) in models.items():
        mask = df["pos_group"] == pos
        df.loc[mask, "predicted_points"] = calibrate(model.predict(df.loc[mask, FEATURE_COLS]))
    return df


def add_projection_only_players(
    predicted: pd.DataFrame, df_all: pd.DataFrame, as_of_season: str = LATEST_SEASON
) -> pd.DataFrame:
    """Adds ``nhl_projection`` to every modeled player, and appends one row
    per NHL.com-projected F/D the model couldn't rank, with that projection
    as ``predicted_points``. Such a player keeps his Hockey-Reference
    ``player_id``/Age/GP if he has any history (a few call-up games), so
    season history still works in the app; otherwise he gets a synthetic
    ``proj_`` id (same convention as draft_pool.goalie_pool). Team always
    comes from NHL.com, which already reflects where he's playing."""
    proj = nhl_projections.load_skater_projections()
    matched = nhl_projections.match_projection_rows(predicted, proj)
    predicted = predicted.assign(
        nhl_projection=proj["nhl_projection"].reindex(matched.astype("float").values).values,
        source="model",
    )

    missing = proj.drop(index=matched.dropna().astype(int).unique())
    history = df_all[df_all["season"].map(notable._season_year) <= notable._season_year(as_of_season)]
    latest = history.sort_values("season", key=lambda s: s.map(notable._season_year)).drop_duplicates(
        loading.ID_COL, keep="last"
    )
    latest = latest[latest["pos_group"].isin(MODELED_POSITIONS)].reset_index(drop=True)
    hist_idx = nhl_projections.match_projection_rows(missing, latest)

    rows = []
    for (_, p), h in zip(missing.iterrows(), hist_idx):
        row = {
            "Player": p["Player"], "pos_group": p["pos_group"], "Team": p["Team"], "Pos": p["pos_group"],
            "predicted_points": float(p["nhl_projection"]), "nhl_projection": p["nhl_projection"],
            "source": "nhl.com",
            loading.ID_COL: "proj_" + nhl_api.normalize_name(p["Player"]).replace(" ", "_"),
        }
        if pd.notna(h):
            hist = latest.loc[int(h)]
            row.update({loading.ID_COL: hist[loading.ID_COL], "Age": hist["Age"], "GP": hist["GP"], "Pos": hist["Pos"]})
        rows.append(row)
    return pd.concat([predicted, pd.DataFrame(rows)], ignore_index=True)


def add_vorp(df: pd.DataFrame, teams: int, roster: dict, filled: dict | None = None) -> pd.DataFrame:
    """Replacement level per position = the last player who'd still make a
    roster: the ``teams * slots``-th best, or, mid-draft, the (open slots
    left)-th best of ``df`` when ``filled`` gives how many of that position's
    slots are already taken (see draft_pool.undrafted_board)."""
    filled = filled or {}
    ranked = []
    for pos, slots in roster.items():
        sub = df[df["pos_group"] == pos].sort_values("predicted_points", ascending=False).reset_index(drop=True)
        cutoff = min(max(teams * slots - filled.get(pos, 0), 1), len(sub))
        replacement = sub.loc[cutoff - 1, "predicted_points"]
        sub["pos_rank"] = sub.index + 1
        sub["replacement_level"] = replacement
        sub["VORP"] = sub["predicted_points"] - replacement
        ranked.append(sub)
    return pd.concat(ranked, ignore_index=True)


def age_in_season(birth_dates: pd.Series, season: str) -> pd.Series:
    """Age by Hockey-Reference's convention (age on February 1 of the
    season's second year), from 'YYYY-MM-DD' birth dates; NaN if unknown."""
    ref = pd.Timestamp(year=loading._season_start_year(season) + 1, month=2, day=1)
    born = pd.to_datetime(birth_dates, errors="coerce")
    before_birthday = (born.dt.month > ref.month) | ((born.dt.month == ref.month) & (born.dt.day > ref.day))
    return ref.year - born.dt.year - before_birthday.astype(int)


def fill_live_ages(df: pd.DataFrame, birth_dates: dict, as_of_season: str = LATEST_SEASON) -> pd.DataFrame:
    """Age (during ``as_of_season``, like every modeled row) from the live
    NHL roster's birth date (nhl_api.current_birth_dates) for NHL.com-only
    players: those with no Hockey-Reference history have no Age at all, and
    those with a few call-up games carry the Age of their last HR season,
    not ``as_of_season``'s. Modeled players keep Hockey-Reference's Age. No
    match (or no live data) leaves Age as it was."""
    out = df.copy()
    target = out["Age"].isna() | (out["source"] == "nhl.com")
    keys = [(nhl_api.normalize_name(n), g) for n, g in zip(out.loc[target, "Player"], out.loc[target, "pos_group"])]
    live = age_in_season(pd.Series([birth_dates.get(k) for k in keys], index=out.index[target], dtype=object), as_of_season)
    out.loc[target, "Age"] = live.fillna(out.loc[target, "Age"])
    return out


def build_draft_board(
    teams: int,
    roster: dict,
    fetch_live_team_changes: bool = True,
    fetch_live_injuries: bool = True,
) -> pd.DataFrame:
    """``fetch_live_team_changes=False`` skips the live NHL API roster check
    (see nhl_api.py) and ``fetch_live_injuries=False`` the ESPN injury feed
    (espn_injuries.py), for a fully deterministic, network-free board. A
    failed or skipped fetch degrades to no team-change / injury tags (and no
    injury point adjustment), nothing else changes.

    ``teams``/``roster`` set the pool size and F/D roster slots that drive
    the VORP replacement level (see add_vorp) -- callers with a
    league settings (see draft_state.load_settings) -- no defaults here."""
    df_all = load_scored_seasons()
    pairs = loading.make_training_pairs(
        df_all, feature_cols=FEATURE_COLS + ["Player", "pos_group"], min_feature_gp=MIN_GP
    )
    models = fit_production_models(pairs, df_all)
    predicted = add_projection_only_players(predict_upcoming(models, df_all), df_all)
    if fetch_live_injuries:
        predicted, injury_feed_ok = espn_injuries.apply_live_injuries(predicted)
    else:
        predicted, injury_feed_ok = espn_injuries.apply_injuries(predicted, None, None, 0.0), True
    ranked = add_vorp(predicted, teams=teams, roster=roster)
    ranked = notable.add_notable_flags(ranked, df_all, LATEST_SEASON)

    team_map, team_fetch_complete, birth_dates = {}, True, {}
    if fetch_live_team_changes:
        try:
            team_map, team_fetch_complete = nhl_api.current_team_map()
        except Exception:
            team_map, team_fetch_complete = {}, False
        birth_dates = nhl_api.current_birth_dates()
    ranked["team_change"] = nhl_api.detect_team_changes(ranked, team_map)
    ranked = nhl_api.apply_live_team(ranked, team_map)
    ranked = fill_live_ages(ranked, birth_dates)
    ranked = keeper.add_future_value(ranked, keeper.age_curve(pairs, full_length_seasons(df_all)))
    if team_map and team_fetch_complete:
        # NHL.com-only prospects may not be on a live NHL roster yet, but
        # NHL.com projects them to play -- "No Team" would be misleading.
        ranked["no_team"] = nhl_api.detect_no_team(ranked, team_map) & (ranked["source"] != "nhl.com")
    else:
        ranked["no_team"] = False
    ranked["rookie"] = notable.is_rookie(ranked[loading.ID_COL], ranked["Age"], df_all, LATEST_SEASON)
    ranked["Notes"] = [
        notable.combine_notes(f, t, c, n, r, i)
        for f, t, c, n, r, i in zip(
            ranked["fragile"], ranked["trend"], ranked["team_change"], ranked["no_team"], ranked["rookie"],
            ranked["injury_tag"],
        )
    ]

    cols = [
        loading.ID_COL, "Player", "Team", "Pos", "pos_group", "Age", "GP",
        "feature_season", "seasons_back",
        "predicted_points", "healthy_points", "games_missed", "nhl_projection", "source", "pos_rank", "VORP",
        "next_season_points", "future_value",
        "fragile", "trend", "team_change", "rookie", "Notes", "injury",
    ]
    result = ranked.sort_values("VORP", ascending=False)[cols].reset_index(drop=True)
    # Not persisted to the CSV -- read by app.py right after a rebuild, in
    # the same process, to warn if the live team-change check was cut short.
    result.attrs["team_fetch_complete"] = team_fetch_complete
    result.attrs["injury_feed_ok"] = injury_feed_ok
    return result


if __name__ == "__main__":
    import draft_state

    seasons = draft_state.list_seasons()
    settings = draft_state.load_settings(seasons[0]) if seasons else draft_state.DEFAULT_SETTINGS
    board = build_draft_board(
        teams=settings["num_managers"],
        roster={"F": settings["forwards"], "D": settings["defense"]},
    )
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    board.to_csv(OUTPUT_PATH, index=False)
    print(f"wrote {len(board)} players to {OUTPUT_PATH}\n")

    stale = board[board["seasons_back"] > 0]
    print(f"{len(stale)} players ranked off a season other than {LATEST_SEASON} (injury/limited GP fallback):")
    print(stale.head(15)[["Player", "feature_season", "seasons_back", "predicted_points", "VORP"]].to_string(index=False))

    pd.set_option("display.width", 120)
    print("\ntop of the board:")
    print(board.head(30).to_string(index=False))
