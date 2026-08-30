# Hockey Pool Player Point Prediction

## Project Goal
Build a model to predict player fantasy points for a season-long hockey pool, to inform draft picks. Highest cumulative points at end of season wins.

## Pool Format
- Draft is turn-based (like a snake draft), no player picked twice
- Roster per manager: 9 forwards, 5 defense, 1 goalie, 1 team
- Winner = highest total points at end of season

## Data
Hockey-Reference season exports in `data/`, **10 seasons: 2016-17 through 2025-26**:
- `skaters-{season}.csv` — basic stats (~600 F / ~325 D / ~100 G per season)
- `skaters-advance-{season}.csv` — Corsi/Fenwick/PDO/zone starts (skaters only, no goalies)
- `allPlayersLookup.csv` — different source (NHL `playerId`, not HR slugs); only joins by
  name. Unique value is exact `birthDate` + handedness. Unused so far.

`src/loading.py` handles the export quirks: two header rows, the HR player slug living in
an unnamed trailing column (read as `-9999`) which is the only reliable join key, traded
players appearing as a `2TM`/`3TM` total row plus per-team splits, clock-format TOI, and a
`League Average` footer row. `load_all_seasons()` stacks every season found; `make_training_pairs()`
builds year-N-features → year-N+1-target rows for consecutive season pairs only (a player who
sat out a year isn't paired across the gap) — 3965 skater pairs across the 5 transitions currently
available.

Note: **2020-21 was COVID-shortened (56 GP max)**. It's used as a features-year without special
handling since the model runs on per-game rates, but keep an eye out if it behaves like an outlier.

### Data gaps blocking progress
- **No goalie stats.** Goalies appear in the skater export with only their assists — no W,
  SV%, GAA, SO, or starts. Deferred for now per the user — do not chase this until asked.
- **Goalie scoring undefined** — standard G+A doesn't describe how goalies earn points.

## Settled
- **Scoring system**: skater fantasy points = G + A + 1 bonus per SHG. Encoded in
  `src/scoring.py` as a weight dict so further rule changes are a one-file edit.
- **Hat-trick bonus (+1) requested but NOT implemented**: this dataset is Hockey-Reference
  season totals, no per-game log, so there's no way to detect actual 3-goal games. Skipped
  for now per user. Revisit if/when per-game data (e.g. NHL API gamelogs) is added — until
  then this is a real gap between modeled points and true pool scoring for streaky scorers.

## Modeling Approach
- **Separate models per position group**: forwards, defense, goalies (different stat drivers)
- **Features to consider**:
  - Skaters: prior season points, TOI/game, shot attempts, power-play time, age (aging curve), games played/durability, team offensive context
  - Goalies: save %, starts, team defensive context, workload
- **Model candidates**: start simple (ElasticNet/regularized linear) as baseline, compare against gradient boosting (LightGBM/XGBoost). Small dataset (few hundred skaters/season) → watch for overfitting.
- **Validation**: k-fold cross-validation given limited seasons; if multi-year data available, train on earlier years, test on most recent.

## Draft Strategy Layer (beyond raw point prediction)
- Compute **value over replacement** per position (predicted points − replacement-level player at that position) to actually drive draft order, not raw predicted points alone.

## Tech Stack
- Python, pandas, scikit-learn, lightgbm/xgboost
- Alexandre has prior experience with `requests`-only API integrations (see Intervals.icu project) — similar pattern likely reusable if pulling live NHL stats

## Status
- [x] Confirm scoring system and finalize target variable (standard G+A + SHG bonus)
- [x] Load and explore last year's stats (`src/loading.py`)
- [x] Multi-season data (10 seasons, 2016-17 to 2025-26 — expanded from 6 after confirming
      more data helps) + train/target pairing
- [x] Build position-specific baseline models (F, D — goalies deferred). `src/train.py`.
      ElasticNet still wins both on holdout Spearman (0.84 F / 0.78 D, 2024-25 features ->
      real 2025-26 results), but the gap over LightGBM narrowed a lot going from 6->10
      seasons (F holdout MAE 11.01->10.52) — LightGBM was data-starved before, not a fixed
      loser; recheck this comparison again if more seasons get added. Stuck to 2016-17+ to
      avoid mixing in the pre-3-on-3-OT scoring environment. Feature list in
      `src/features.py`. Trained models saved to `models/*.joblib`.
- [x] Add value-over-replacement draft ranking. `src/rank.py`, run directly to regenerate
      `output/draft_board.csv`. Pool size confirmed as **12 teams**; replacement cutoff =
      108 forwards / 60 defense (9F + 5D roster x 12 teams). Uses ElasticNet (won the
      holdout eval) refit on all season-pairs, not just train.py's train-only split.
- [x] Injury fallback: `loading.latest_healthy_row` -- a player under `features.MIN_GP`
      games in his most recent season falls back to his last season with enough games,
      instead of being dropped or judged on a handful of games. Used both when building
      training pairs (unlimited lookback -- safe, since a retired player has no future
      season to serve as a training target and gets filtered out anyway) and for live
      draft-board predictions (`rank.py`, capped at `MAX_SEASONS_BACK = 1` -- unlimited
      lookback there resurrected actually-retired players like Shea Weber/Ryan Ellis,
      since the data has no active-roster flag to tell "injured" apart from "retired,"
      just recency). 893 players on the board now (115 via the fallback), up from 778.
      True rookies with zero NHL history in any loaded season still aren't covered --
      no fallback season exists for them; would need external prospect data.
- [x] "Notes" draft-day tags: `src/notable.py` (fragile/declining/rising, local data only)
      + `src/nhl_api.py` (live NHL API team-change check) combine into a single `Notes`
      column (e.g. `"New Team • Fragile • Declining"`) shown in `app.py`'s F/D table.
      No live "currently injured" flag -- the NHL's public API has no injury/IR endpoint
      (checked; genuinely undocumented anywhere), out of scope entirely. Fragile/trend
      reuse data already loaded (`GP` history, `prior_fantasy_points_pg`), no network:
      fragile = GP short of `FRAGILE_GP_PCT` of that season's league-max in >=2 of the
      last 3 qualifying seasons; trend = >=20%/25% point-rate move combined with an age
      cutoff (30+/23-) to separate real decline/breakout from noise. Team-change hits
      `api-web.nhle.com` (unauthenticated, unofficial, does rate-limit -- confirmed live)
      for all current rosters, matched to Hockey-Reference names by normalized
      `(name, pos_group)` -- position is needed as a tiebreaker since real collisions
      exist (two active "Sebastian Aho," CAR forward vs. NYI defenseman). Every failure
      mode (network down, rate-limited, malformed response) degrades to no team-change
      tag rather than raising -- verified by pointing the client at an unreachable host
      and confirming the board still builds. `data/allPlayersLookup.csv` stays unused --
      evaluated it as a name-matching bridge for this feature and it would've added a
      second unreliable name join on top of the one already needed, no upside over
      matching the live roster response directly.
- [ ] (Later, per user) Add goalie stat export + define goalie scoring
- [x] Draft-day app: `src/app.py`, run via `.venv/Scripts/python.exe -m streamlit
      run src/app.py`. Live view over `rank.build_draft_board()` -- filter by
      position/name, mark a player picked (by you or another manager), undo,
      and pull up any player's season history. Picks persist to
      `state/picks.json` / `state/managers.json` (JSON, survives restarts) so
      a multi-hour live draft isn't lost on a browser/process restart. VORP
      recomputes live (`src/draft_pool.undrafted_board`) against only
      undrafted players, so replacement level shifts as a position gets
      drafted down -- reuses `rank.add_vorp` rather than a second
      implementation. Goalies get an unranked list (name/team/GP only, via
      `draft_pool.goalie_pool`, `min_gp=1` since it's just a pickable listing
      with no model behind it) since goalie scoring is still undefined (see
      gap above). The "1 team" roster slot isn't modeled -- tracked manually
      outside the app.

## Setup
```
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```
