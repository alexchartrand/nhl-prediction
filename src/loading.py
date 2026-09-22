"""Load Hockey-Reference season exports into tidy per-player DataFrames.

The raw CSVs have a few quirks that every downstream script would otherwise
have to re-handle:

* two header rows (a column-group row above the real header)
* an unnamed trailing column holding the Hockey-Reference player slug,
  read by pandas as ``-9999`` -- this is the only reliable join/ID key,
  since player names collide and change spelling
* players traded mid-season get one "total" row (Team ``2TM``/``3TM``)
  plus one row per team; we keep the total and drop the splits
* a trailing "League Average" row
* clock-format times (``1884:48``, ``22:59``) instead of numbers
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

import scoring

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

ID_COL = "player_id"
_RAW_ID_COL = "-9999"
_MULTI_TEAM = re.compile(r"^\d+TM$")

# Hockey-Reference splits wingers into LW/RW and uses F/W for players it
# can't pin down; the pool only cares about forward vs defense vs goalie.
_POS_GROUP = {
    "C": "F", "LW": "F", "RW": "F", "F": "F", "W": "F",
    "D": "D",
    "G": "G",
}


def _to_minutes(value: object) -> float:
    """Convert ``MM:SS`` (or ``MMMM:SS``) to fractional minutes."""
    if not isinstance(value, str) or ":" not in value:
        return np.nan
    mins, _, secs = value.partition(":")
    try:
        return int(mins) + int(secs) / 60
    except ValueError:
        return np.nan


def available_seasons(data_dir: Path = DATA_DIR) -> list[str]:
    """Season strings (e.g. '2024_2025') with both a basic and advanced export."""
    basic = {
        p.stem.removeprefix("skaters-")
        for p in data_dir.glob("skaters-*.csv")
        if not p.stem.startswith("skaters-advance-")
    }
    adv = {p.stem.removeprefix("skaters-advance-") for p in data_dir.glob("skaters-advance-*.csv")}
    return sorted(basic & adv)


def _read_hr_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=1, encoding="utf-8")
    df = df.rename(columns={_RAW_ID_COL: ID_COL, "Tm": "Team"})
    # The "League Average" footer row carries a literal -9999 as its slug.
    df = df[df[ID_COL].notna() & (df[ID_COL].astype(str) != _RAW_ID_COL)]
    return df


def _drop_traded_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per player: the multi-team total when it exists."""
    is_total = df["Team"].astype(str).str.match(_MULTI_TEAM)
    has_total = df.groupby(ID_COL)["Team"].transform(
        lambda s: s.astype(str).str.match(_MULTI_TEAM).any()
    )
    keep = is_total | ~has_total
    return df[keep].drop_duplicates(subset=ID_COL, keep="first")


def load_basic(season: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    df = _read_hr_csv(data_dir / f"skaters-{season}.csv")
    df = _drop_traded_splits(df)
    df["TOI"] = df["TOI"].map(_to_minutes)
    df["ATOI"] = df["ATOI"].map(_to_minutes)
    df = df.drop(columns=["Rk", "Awards"], errors="ignore")
    return df.set_index(ID_COL)


def load_advanced(season: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    df = _read_hr_csv(data_dir / f"skaters-advance-{season}.csv")
    df = _drop_traded_splits(df)
    for col in ("TOI/60", "TOI(EV)"):
        df[col] = df[col].map(_to_minutes)
    # These duplicate the basic table; keep only the advanced-only columns.
    df = df.drop(
        columns=["Rk", "Player", "Age", "Team", "Pos", "GP", "TK", "GV"],
        errors="ignore",
    )
    return df.set_index(ID_COL)


def load_season(season: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """One row per player for ``season``, basic + advanced stats merged.

    Goalies are included but have no advanced-stat row (and no goalie
    stats at all -- the skater export only carries their assists), so
    their advanced columns are NaN.
    """
    basic = load_basic(season, data_dir)
    adv = load_advanced(season, data_dir)
    df = basic.join(adv, how="left")

    df.insert(0, "season", season)
    df.insert(3, "pos_group", df["Pos"].map(_POS_GROUP))

    df["PP_pts"] = df["PPG"] + df["PP"]

    gp = df["GP"].replace(0, np.nan)
    for col in ("G", "A", "PTS", "SOG", "BLK", "HIT", "PPG", "PP", "PP_pts"):
        df[f"{col}_pg"] = df[col] / gp

    return df.reset_index()


def load_all_seasons(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Every season with both exports present, stacked into one long frame."""
    seasons = available_seasons(data_dir)
    return pd.concat(
        [load_season(s, data_dir) for s in seasons], ignore_index=True
    )


def load_goalie_season(season: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """One row per goalie for ``season`` from ``goalies-{YYYY}-{YYYY}.csv``.

    Same slug/traded-splits/League-Average quirks as the skater exports but a
    single header row and hyphenated file names (``2024-2025``). ``season``
    uses the project's ``2024_2025`` spelling.
    """
    path = data_dir / f"goalies-{season.replace('_', '-')}.csv"
    df = pd.read_csv(path, encoding="utf-8")
    df = df.rename(columns={_RAW_ID_COL: ID_COL, "Tm": "Team"})
    df = df[df[ID_COL].notna() & (df[ID_COL].astype(str) != _RAW_ID_COL)]
    df = _drop_traded_splits(df)
    df["MIN"] = df["MIN"].map(_to_minutes)
    df = df.drop(columns=["Rk", "Awards"], errors="ignore")
    df.insert(0, "season", season)
    df["pos_group"] = "G"
    return df.reset_index(drop=True)


def available_goalie_seasons(data_dir: Path = DATA_DIR) -> list[str]:
    return sorted(
        p.stem.removeprefix("goalies-").replace("-", "_") for p in data_dir.glob("goalies-*.csv")
    )


def load_all_goalie_seasons(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.concat(
        [load_goalie_season(s, data_dir) for s in available_goalie_seasons(data_dir)],
        ignore_index=True,
    )


def _season_start_year(season: str) -> int:
    return int(season.split("_")[0])


def latest_healthy_row(
    df: pd.DataFrame, as_of_season: str, min_gp: int = 10, max_seasons_back: int | None = None
) -> pd.DataFrame:
    """One row per player: their most recent season at or before ``as_of_season``
    with at least ``min_gp`` games played, Age adjusted forward to ``as_of_season``.

    This is what lets an injury-shortened (or partial-call-up) season fall
    back to the player's last full season instead of either getting a noisy
    few-game sample as his "current" rates or being dropped outright. A
    player with no qualifying season anywhere in the window -- a true rookie
    with no NHL history yet -- is simply absent from the result; there is no
    stat history to build features from, and none is invented here.

    ``max_seasons_back`` bounds how far the fallback will reach (``None`` =
    unlimited). Unlimited is safe for training pairs -- a retired player has
    no future season to serve as the target, so they're filtered out there
    regardless -- but for a live prediction (no future season to check
    against) an unbounded fallback will happily resurrect a player who
    hasn't appeared in years, because the data can't distinguish "injured,
    still active" from "retired": both are just an absence in recent
    seasons. Callers predicting a not-yet-played season should pass a small
    cap (e.g. 1) to keep the fallback to recent-injury cases.

    Adds ``feature_season`` (which season's stats were actually used) and
    ``seasons_back`` (0 = as_of_season itself, 1 = one season earlier, ...)
    so callers can see how stale a given row's stats are.
    """
    seasons = sorted(df["season"].unique(), key=_season_start_year)
    if as_of_season not in seasons:
        raise ValueError(f"{as_of_season!r} not in available seasons {seasons}")
    window = seasons[: seasons.index(as_of_season) + 1]
    if max_seasons_back is not None:
        window = window[-(max_seasons_back + 1):]
    as_of_year = _season_start_year(as_of_season)

    rows = []
    for _player_id, g in df[df["season"].isin(window)].groupby(ID_COL):
        g = g.set_index("season")
        for seasons_back, season in enumerate(reversed(window)):
            if season not in g.index or g.loc[season, "GP"] < min_gp:
                continue
            row = g.loc[season].copy()
            row["feature_season"] = season
            row["seasons_back"] = seasons_back
            row["Age"] = row["Age"] + (as_of_year - _season_start_year(season))
            rows.append(row)
            break
    return pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame()


def make_training_pairs(
    df: pd.DataFrame, feature_cols: list[str], min_feature_gp: int = 10
) -> pd.DataFrame:
    """One row per player per season transition, fallback features -> that season's target.

    ``df`` should come from :func:`load_all_seasons` (already scored via
    :func:`scoring.add_target`). For each season S, features come from the
    player's most recent *qualifying* season strictly before S (see
    :func:`latest_healthy_row`) -- so a player who was hurt the season right
    before S still gets a training row, built from his last healthy season,
    instead of being dropped. The target is always season S's actual result.
    Only consecutive calendar-year transitions are used, to avoid silently
    pairing across a gap if the season files ever aren't contiguous.
    """
    seasons = sorted(df["season"].unique(), key=_season_start_year)
    pairs = []
    for prev_season, target_season in zip(seasons, seasons[1:]):
        if _season_start_year(target_season) != _season_start_year(prev_season) + 1:
            continue
        features = latest_healthy_row(df, prev_season, min_feature_gp)
        if features.empty:
            continue
        features = features.set_index(ID_COL)
        target_rows = df[df["season"] == target_season].set_index(ID_COL)
        common = features.index.intersection(target_rows.index)
        if common.empty:
            continue
        pair = features.loc[common, feature_cols + ["feature_season", "seasons_back"]].copy()
        pair[ID_COL] = common
        pair["target_season"] = target_season
        pair[scoring.TARGET] = target_rows.loc[common, scoring.TARGET].values
        pairs.append(pair)
    return pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame(columns=feature_cols)


if __name__ == "__main__":
    frame = load_all_seasons()
    print(frame.shape)
    print(frame["season"].value_counts().sort_index())
