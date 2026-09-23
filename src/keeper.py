"""Keeper value for the last pick (backlog.md K2).

Each manager's final-round pick becomes his keeper: he can hold him into
next season, and every season after, as that season's last-round pick. So
the last pick is worth this season's value *plus* whatever the player is
projected to beat a last-round pick by in later seasons:

    keeper value = VORP (this season, live)  +  future_value
    future_value = sum over k = 1..FUTURE_SEASONS of
                   DISCOUNT**k * max(0, points in season +k - replacement level)

A keeper who's no longer better than a last-round pick just gets released,
hence the max(0, ...). The replacement level is the board's full-draft one
at the player's position (``rank.add_vorp``: teams x slots-th best) --
next season's pool is assumed to look like this one's. Skaters only
(goalies/teams can't be kept), so this only touches the F/D board.

Future seasons are projected from this season's injury-free prediction
(``healthy_points``) by an age curve (``age_curve``), one season at a time.
Validated against a direct year N -> N+2 model in rolling holdouts
(``python src/keeper.py``, 2023-24/2024-25/2025-26 x F/D). Mean MAE:

    | method                    | F     | D    | bias, age <= 23 (F / D) |
    | N+1 prediction unchanged  | 12.14 | 8.26 | -1.9 / -1.9             |
    | N+1 x age curve (shipped) | 11.94 | 8.12 | +0.2 / -0.7             |
    | direct N -> N+2 model     | 12.32 | 8.82 | -3.3 / -2.2             |

Rank correlation was the same for all three. The full-strength ratio
over-projected young forwards by ~2 pts -- the N+1 model already has Age as
a feature, so the full ratio double-counts aging; hence AGE_CURVE_STRENGTH.

DISCOUNT and FUTURE_SEASONS are judgment calls, not fitted: projection error
grows each season out (N+2 MAE ~12 F / ~8 D vs ~10 / ~7.3 for N+1), and the
pool itself may change.

NHL.com-only rookies get their Age from the live NHL roster's birth date
(rank.fill_live_ages). One still without an Age (not on a roster, or the
NHL API was down) gets no age adjustment -- a flat projection. Assuming a
young age would overrate the older European signings among them (e.g.
Shabanov, 25).

Keeping is optional: a manager may just as well take his goalie/team last
and have no keeper. The app only warns when a pick would rule a keeper out
(blocks_keeper), never blocks it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scoring

# Age range of the curve; older/younger players use the end values.
AGE_MIN, AGE_MAX = 19, 37
# Each age's ratio pools the ages +-AGE_WINDOW around it -- single-age
# ratios are noisy (24: 0.98, 25: 1.04).
AGE_WINDOW = 1
# Fraction of the raw next-season/this-season ratio applied (see module doc).
AGE_CURVE_STRENGTH = 0.5
FUTURE_SEASONS = 3
DISCOUNT = 0.75
KEEPER_POSITIONS = ("F", "D")


def age_curve(pairs: pd.DataFrame, full_seasons: set[str]) -> pd.Series:
    """Multiplier from one season's points to the next, by age in the
    earlier season (index AGE_MIN..AGE_MAX). ``pairs`` are consecutive
    training pairs (``loading.make_training_pairs``, gap 1). Ratio of summed
    points, so it's weighted toward regulars; only non-fallback rows where
    both seasons are full length, so the COVID seasons don't read as decline.
    Shrunk toward 1 by AGE_CURVE_STRENGTH."""
    p = pairs[
        (pairs["seasons_back"] == 0)
        & pairs["feature_season"].isin(full_seasons)
        & pairs["target_season"].isin(full_seasons)
    ]
    sums = (
        pd.DataFrame({
            "age": p["Age"].round().clip(AGE_MIN, AGE_MAX).astype(int),
            "cur": p["prior_fantasy_points_pg"] * p["GP"],
            "nxt": p[scoring.TARGET],
        })
        .groupby("age")[["cur", "nxt"]].sum()
        .reindex(range(AGE_MIN, AGE_MAX + 1), fill_value=0.0)
    )
    pooled = sums.rolling(2 * AGE_WINDOW + 1, center=True, min_periods=1).sum()
    return 1 + AGE_CURVE_STRENGTH * (pooled["nxt"] / pooled["cur"] - 1)


def aging_multiplier(curve: pd.Series, ages) -> np.ndarray:
    """``curve`` at each age (clipped to its range); 1 where age is unknown."""
    ages = pd.Series(ages, dtype=float)
    mult = ages.round().clip(AGE_MIN, AGE_MAX).map(curve)
    return mult.fillna(1.0).to_numpy()


def future_points(base, age, curve: pd.Series, n: int = FUTURE_SEASONS) -> np.ndarray:
    """(players x n) projected points for the ``n`` seasons after the one
    ``base`` projects. ``age`` is each player's age in the season *before*
    ``base``'s (the board's Age, i.e. rank.LATEST_SEASON's), so the step
    into season +k uses the curve at age + k."""
    pts = np.asarray(base, dtype=float)
    age = pd.Series(age, dtype=float).reset_index(drop=True)
    out = np.empty((len(pts), n))
    for k in range(1, n + 1):
        pts = pts * aging_multiplier(curve, age + k)
        out[:, k - 1] = pts
    return out


def add_future_value(ranked: pd.DataFrame, curve: pd.Series) -> pd.DataFrame:
    """Adds ``next_season_points`` (season +1) and ``future_value`` (see
    module doc) to a VORP-ranked F/D board (needs healthy_points or
    predicted_points, Age, replacement_level). Injuries only dent this
    season, so the projection starts from ``healthy_points``."""
    out = ranked.copy()
    base = out["healthy_points"].fillna(out["predicted_points"]) if "healthy_points" in out else out["predicted_points"]
    proj = future_points(base, out["Age"], curve)
    gain = np.clip(proj - out["replacement_level"].to_numpy()[:, None], 0, None)
    out["next_season_points"] = proj[:, 0]
    out["future_value"] = (gain * DISCOUNT ** np.arange(1, proj.shape[1] + 1)).sum(axis=1)
    out.loc[~out["pos_group"].isin(KEEPER_POSITIONS), ["next_season_points", "future_value"]] = np.nan
    return out


def keeper_value(available: pd.DataFrame) -> pd.Series:
    """Live keeper value of each undrafted F/D: live VORP + future_value."""
    return available["VORP"] + available["future_value"]


def roster_status(rostered: pd.DataFrame, keepers: pd.DataFrame, settings: dict, me: str = "me") -> dict:
    """Keeper-related roster state for ``me``. ``rostered`` =
    draft_state.roster_picks (keepers included), ``keepers`` =
    draft_state.load_keepers.

    ``holds_keeper``: me already holds one, so skips the final round and
    this doesn't apply. ``picks_left``: live picks still to make.
    ``open_by_pos``: open slots per position; ``open_skater``/``open_other``:
    open F+D / G+TEAM slots. ``final_pick``: the next pick is my last one
    and it's an F/D slot, so it will be a keeper. (Leaving a goalie/team
    slot for last instead is allowed -- then there's no keeper.)"""
    slots = {"F": settings["forwards"], "D": settings["defense"], "G": settings["goalies"], "TEAM": settings["team_slots"]}
    have = rostered.loc[rostered["manager"] == me, "pos_group"].value_counts()
    open_by_pos = {pos: max(n - int(have.get(pos, 0)), 0) for pos, n in slots.items()}
    picks_left = sum(open_by_pos.values())
    holds = bool((keepers["manager"] == me).any())
    return {
        "holds_keeper": holds,
        "picks_left": picks_left,
        "open_by_pos": open_by_pos,
        "open_skater": open_by_pos["F"] + open_by_pos["D"],
        "open_other": open_by_pos["G"] + open_by_pos["TEAM"],
        "final_pick": not holds and picks_left == 1 and open_by_pos["F"] + open_by_pos["D"] > 0,
    }


def blocks_keeper(status: dict, pos_group: str) -> bool:
    """True if drafting a ``pos_group`` player now would fill ``me``'s last
    F/D slot while a goalie/team slot is still open -- the final pick would
    then have to be a goalie or team, and no keeper. A warning only: that
    can be exactly what the manager wants."""
    return (
        not status["holds_keeper"]
        and pos_group in KEEPER_POSITIONS
        and status["open_skater"] == 1
        and status["open_other"] >= 1
    )


def _eval() -> pd.DataFrame:
    """Rolling holdouts for the season-after-next (N+2): N+1 prediction
    unchanged vs x the age curve vs a direct N -> N+2 model. For target
    season T, everything is fit on pairs whose target is <= T-2 -- what's
    known when projecting T from T-2's stats."""
    from scipy.stats import spearmanr

    import loading
    from features import FEATURE_COLS, MIN_GP, load_scored_seasons
    from train import fit_calibration, full_length_seasons, make_elasticnet

    df = load_scored_seasons()
    full = full_length_seasons(df)
    cols = FEATURE_COLS + ["Player", "pos_group"]
    p1 = loading.make_training_pairs(df, feature_cols=cols, min_feature_gp=MIN_GP)
    p2 = loading.make_training_pairs(df, feature_cols=cols, min_feature_gp=MIN_GP, gap=2)
    year = loading._season_start_year

    rows = []
    for target in ("2023_2024", "2024_2025", "2025_2026"):
        known = year(target) - 2
        tr1 = p1[p1["target_season"].map(year) <= known]
        curve = age_curve(tr1, full)
        for pos in KEEPER_POSITIONS:
            s1 = tr1[tr1["pos_group"] == pos].reset_index(drop=True)
            s2 = p2[(p2["pos_group"] == pos) & (p2["target_season"].map(year) <= known)].reset_index(drop=True)
            test = p2[(p2["pos_group"] == pos) & (p2["target_season"] == target)].reset_index(drop=True)
            n1 = make_elasticnet().fit(s1[FEATURE_COLS], s1[scoring.TARGET])
            n1_pred = fit_calibration(make_elasticnet, s1, full)(n1.predict(test[FEATURE_COLS]))
            n2 = make_elasticnet().fit(s2[FEATURE_COLS], s2[scoring.TARGET])
            preds = {
                "N+1 flat": n1_pred,
                # test Age is the feature season's; the step N+1 -> N+2 uses age + 1.
                "N+1 x age curve": n1_pred * aging_multiplier(curve, test["Age"] + 1),
                "direct N+2": fit_calibration(make_elasticnet, s2, full)(n2.predict(test[FEATURE_COLS])),
            }
            y = test[scoring.TARGET].to_numpy()
            young = (test["Age"] <= 23).to_numpy()
            for method, pred in preds.items():
                top = np.argsort(-pred)[:30]
                rows.append({
                    "target": target, "pos": pos, "method": method,
                    "MAE": np.abs(pred - y).mean(), "bias": (pred - y).mean(),
                    "Spearman": spearmanr(pred, y).correlation,
                    "top30 MAE": np.abs(pred[top] - y[top]).mean(),
                    "<=23 MAE": np.abs(pred[young] - y[young]).mean(),
                    "<=23 bias": (pred[young] - y[young]).mean(),
                })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import draft_state
    import rank

    pd.set_option("display.width", 160)
    res = _eval()
    print(res.round(2).to_string(index=False))
    print("\nmean over holdouts:")
    print(res.groupby(["pos", "method"]).mean(numeric_only=True).round(2).to_string())

    settings = draft_state.DEFAULT_SETTINGS
    board = rank.build_draft_board(
        teams=settings["num_managers"], roster={"F": settings["forwards"], "D": settings["defense"]},
        fetch_live_team_changes=False, fetch_live_injuries=False,
    )
    board["keeper_value"] = keeper_value(board)
    print(f"\ntop keeper values at {settings['num_managers']} managers (full board, before any pick):")
    print(board.sort_values("keeper_value", ascending=False).head(25)[
        ["Player", "Pos", "Age", "predicted_points", "next_season_points", "VORP", "future_value", "keeper_value"]
    ].round(1).to_string(index=False))
