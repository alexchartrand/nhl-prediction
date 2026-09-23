# NHL Predictor

Predicts player fantasy points for a season-long snake-draft hockey pool,
and turns those predictions into a live draft-day tool. See
[CLAUDE.md](CLAUDE.md) for the full decision log (scoring rules, modeling
choices, open gaps) and [backlog.md](backlog.md) for league rules and
remaining ideas; this file is the practical "how it works / how to run it /
how to maintain it" guide.

## Pool format

- Snake draft, no player picked twice. Pool size is set per season in the
  app (default 12 managers; once a draft order is entered, its length is
  the manager count).
- Roster per manager: 9 forwards, 5 defense, 1 goalie, 1 team (also
  editable per season).
- Each manager's last pick is their keeper into the next season, where it
  counts as their last-round pick (skaters only).
- Winner = highest cumulative points at season end.
- Scoring (`src/scoring.py`):
  - Skaters: **G + A + 1 bonus per SHG**.
  - Goalies: 2 per win, 1 per OT/SO loss, 2 per shutout, 5 per goal, 2 per
    assist.
  - Team: NHL standings points (2 per win, 1 per OT/SO loss).

## How it works

```
data/skaters-*.csv  → loading → features → train.py / rank.py (ElasticNet + VORP) ─┐
data/goalies-*.csv  → loading → goalies.py (history, points-per-win) ──────────────┤→ src/app.py
data/nhl 2026-2027 projections/*.txt → nhl_projections.py (goalie/team ranking) ────┘
```

1. **Data** — Hockey-Reference season exports in `data/`, 2016-17 through
   the latest completed season:
   - `skaters-{YYYY_YYYY}.csv` (basic stats)
   - `skaters-advance-{YYYY_YYYY}.csv` (Corsi/Fenwick/PDO/zone starts)
   - `goalies-{YYYY-YYYY}.csv` (goalie stats; note the hyphen, not
     underscore)

   Plus NHL.com's own projections for the upcoming season in
   `data/nhl 2026-2027 projections/` (`fowards.txt`/`defense.txt` =
   fantasy points, `goalies.txt`/`teams.txt` = projected wins).
2. **Loading** (`src/loading.py`) — normalizes the raw exports: strips the
   double header row, uses the Hockey-Reference player-slug column (read as
   `-9999`) as the real join key since names collide, collapses traded
   players' `2TM`/`3TM` + per-team split rows down to one total row, parses
   clock-format TOI, and drops the `League Average` footer. Also builds
   year-N → year-N+1 training pairs (`make_training_pairs`) and the
   injury-fallback lookup (`latest_healthy_row`, see below).
3. **Scoring** (`src/scoring.py`) — turns raw stats into fantasy points.
   All scoring-rule changes belong in this one file (`SKATER_WEIGHTS`,
   `GOALIE_WEIGHTS`, `TEAM_WEIGHTS`).
4. **Features** (`src/features.py`) — per-game rate stats (not raw totals,
   so a shortened season isn't penalized) plus age, GP, and prior-season
   fantasy points/game, and the two seasons before that (lagged points/game,
   GP share, ATOI, a Marcel-style 5/4/3 weighted points rate).
   `FEATURE_COLS` is the single list every skater model trains on.
5. **Modeling** (`src/train.py`) — separate ElasticNet and LightGBM models
   per position group (forwards, defense), grouped k-fold CV, most recent
   season held out for evaluation. ElasticNet wins on holdout Spearman for
   both groups and is what production uses, with a quadratic calibration
   (`train.Calibration`) so top players aren't under-predicted.
   `src/goalies.py` is a separate goalie ElasticNet — kept for its
   evaluation, but the app ranks goalies off NHL.com instead (see 7).
6. **Ranking** (`src/rank.py`) — refits ElasticNet on *all* season-pairs
   (no holdout, since a real draft board shouldn't waste data), predicts
   next-season points for every current F/D, and converts predictions into
   **value over replacement (VORP)**: predicted points minus the last
   starter-quality player at that position (managers × roster slots, e.g.
   108 F / 60 D at 12 × 9F/5D). Rookies the model can't rank but NHL.com
   projects are added with NHL.com's number (`source == "nhl.com"`); every
   row also shows NHL.com's projection for reference. Also attaches
   draft-day "Notes" tags (`src/notable.py` for fragile/declining/rising/
   rookie, `src/nhl_api.py` for live "New Team" / "No Team" checks).
   Players on ESPN's live injury list (`src/espn_injuries.py`) get an
   injury tag and have their points scaled down by the share of the season
   they're expected to miss (from ESPN's return date).
7. **Goalies and teams** (`src/draft_pool.py`) — ranked off NHL.com's
   projected wins, which bake in this season's depth charts. Goalie wins are
   converted to fantasy points with each goalie's (shrunk) historical
   points-per-win; team wins to standings points. Both get VORP on the same
   scale as skaters.
8. **Draft-day app** (`src/app.py`, Streamlit) — tabs for Forwards &
   Defense, Goalies, Teams, My Pool (your roster vs. the slot targets) and
   Draft Log. Filter by position/name, mark a player (or team) drafted by
   you or another manager, undo, and compare up to 3 players' season
   histories. With a draft order entered, the snake order auto-advances and
   a banner shows who's on the clock. Keepers are entered in the sidebar
   before the draft. VORP recomputes live against only undrafted players,
   with the replacement level at the last *open* roster slot. The
   **Explore** button in the compare view asks Mistral (web search) for
   current news/injury/form on the checked players.

All draft state is **per season**: pick or create a season in the sidebar,
and its picks, managers/draft order, settings and keepers live in
`state/seasons/<season>/*.json`, so a restart mid-draft loses nothing and
last year's draft stays intact.

## Running it

```
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Launch the live draft-day app (or double-click `run_app.bat`):
```
.venv/Scripts/python.exe -m streamlit run src/app.py
```

The app builds and saves its boards on first load
(`output/draft_board_<season>.csv`, `goalie_board_<season>.csv`,
`team_board_<season>.csv`); the sidebar's **Recompute draft board** button
rebuilds all three (refits the models, hits the live NHL API).

Retrain / re-evaluate the skater models (ElasticNet vs LightGBM, plus
calibration):
```
.venv/Scripts/python.exe src/train.py
```

Goalie model evaluation:
```
.venv/Scripts/python.exe src/goalies.py
```

`src/rank.py` run directly prints a board and writes
`output/draft_board.csv` — useful for a quick look, but the app doesn't read
that file.

The **Explore** button needs a `MISTRAL_API_KEY` environment variable
(`.env` file, loaded via `python-dotenv`). Everything else needs no secrets.

## Maintaining this for a new season

Each fall, once Hockey-Reference has a full prior-season export available:

1. **Add the new season's data.** Download that season's skater basic,
   skater advanced and goalie exports from Hockey-Reference
   (`data/data_link.txt` has the advanced-stats URL pattern — swap the
   year, and the page name for the other two) and save them as
   `data/skaters-{YYYY_YYYY}.csv`, `data/skaters-advance-{YYYY_YYYY}.csv`
   and `data/goalies-{YYYY-YYYY}.csv`, matching the existing filenames. No
   code changes needed to load them — `loading.load_all_seasons()` /
   `load_all_goalie_seasons()` pick up every season in `data/`.
2. **Add NHL.com's new projections.** Paste them into a new
   `data/nhl YYYY-YYYY projections/` folder (`fowards.txt`, `defense.txt`,
   `goalies.txt`, `teams.txt`, same line format as the current ones) and
   point `PROJECTIONS_DIR` in `src/nhl_projections.py` at it.
3. **Bump the season constants** to the newest data season:
   `LATEST_SEASON` in `src/rank.py` and `src/goalies.py`, and
   `HOLDOUT_TARGET_SEASON` in `src/train.py`.
4. **Sanity-check `src/train.py`'s holdout comparison** — rerun it and
   re-read the Spearman/MAE numbers for both ElasticNet and LightGBM. The
   gap between the two has been narrowing as seasons are added; if LightGBM
   ever overtakes ElasticNet on holdout, swap `rank.py`'s production model
   (`from train import make_elasticnet` → the LightGBM equivalent).
5. **Start a new season in the app** (sidebar → "+ New season..."), set up
   managers/draft order and settings, enter keepers, then hit **Recompute
   draft board**. Old seasons stay selectable; nothing needs deleting.
6. **Re-verify the NHL API still works** (`src/nhl_api.py`) — it hits an
   unauthenticated, unofficial endpoint (`api-web.nhle.com`) that could
   change shape season to season. It degrades to "no team-change tag" on
   any failure rather than breaking the board (the app warns if the check
   was cut short), but worth a spot-check before relying on it live.
7. If a season was COVID-shortened or otherwise anomalous (like 2020-21's
   56-game season), no special handling is required since the model runs on
   per-game rates — just keep an eye on it in the holdout numbers.

## Known gaps

- **No hat-trick bonus.** Requested but not implementable from
  season-total data — would need a per-game log (e.g. NHL API gamelogs).
- **Rookies depend on NHL.com.** The model can't rank players without
  enough NHL history; those NHL.com projects are added with NHL.com's
  number, so a rookie NHL.com leaves out isn't on the board.
- **Two projection sources in one ranking.** F/D are the in-repo model
  (plus NHL.com for rookies); goalies and teams are NHL.com's projections.
- **Injury data is ESPN's, and only current injuries.** Games missed come
  from ESPN's estimated return date, which is often a placeholder a few
  days out. The feed can't tell "injured all last season" apart from
  "retired", so the injury fallback (`loading.latest_healthy_row`) still
  caps how far back it will reach (`MAX_SEASONS_BACK = 1` in `rank.py`)
  to avoid resurrecting long-retired players. The Explore button is the
  manual workaround.

## Repo layout

```
data/            Hockey-Reference CSV exports (skaters, skaters-advance, goalies per season),
                 NHL.com projections folder, name-lookup CSV (unused)
models/          Trained skater model artifacts (*.joblib), from src/train.py
output/          Per-season draft/goalie/team boards saved by the app
state/seasons/   Per-season draft state (picks, managers, settings, keepers JSON), from src/app.py
backlog.md       League rules + remaining feature ideas
run_app.bat      Launches the Streamlit app
src/
  loading.py         CSV parsing, player-ID join key, training-pair construction, injury fallback
  scoring.py         Skater / goalie / team scoring rules
  features.py        Feature list + prior-season and multi-season features
  train.py           Skater model training + holdout evaluation (ElasticNet vs LightGBM) + calibration
  goalies.py         Goalie data/features + goalie model evaluation
  rank.py            Production F/D model refit + VORP draft board generation
  nhl_projections.py Parses NHL.com's projections, name-matches them to the board
  notable.py         Fragile/declining/rising/rookie "Notes" tags (local data only)
  nhl_api.py         Live NHL API rosters (team-change / no-team tags, team names)
  espn_injuries.py   Live ESPN injury feed: injury tags + games-missed point adjustment
  draft_pool.py      Live undrafted-only VORP boards; goalie and team pools
  draft_state.py     Per-season picks/managers/settings/keepers persistence, snake order
  explore.py         Mistral web-search player lookup for the Explore button
  app.py             Streamlit draft-day app
```
