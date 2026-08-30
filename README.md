# NHL Predictor

Predicts player fantasy points for a season-long, 12-team snake-draft hockey
pool, and turns those predictions into a live draft-day tool. See
[CLAUDE.md](CLAUDE.md) for the full decision log (scoring rules, modeling
choices, open gaps); this file is the practical "how it works / how to run
it / how to maintain it" guide.

## Pool format

- Snake draft, 12 teams, no player picked twice.
- Roster per manager: 9 forwards, 5 defense, 1 goalie, 1 team (the goalie
  and team slots aren't modeled — see [Known gaps](#known-gaps)).
- Winner = highest cumulative points at season end.
- Skater fantasy points = **G + A + 1 bonus per SHG** (`src/scoring.py`).

## How it works

```
data/*.csv  →  src/loading.py  →  src/features.py  →  src/train.py / src/rank.py  →  output/draft_board.csv  →  src/app.py
```

1. **Data** — Hockey-Reference season exports live in `data/`, one
   `skaters-{season}.csv` (basic stats) and one `skaters-advance-{season}.csv`
   (Corsi/Fenwick/PDO/zone starts) per season, 2016-17 through the current
   season. Goalies appear only with assist totals — no goalie model exists
   yet.
2. **Loading** (`src/loading.py`) — normalizes the raw exports: strips the
   double header row, uses the Hockey-Reference player-slug column (read as
   `-9999`) as the real join key since names collide, collapses traded
   players' `2TM`/`3TM` + per-team split rows down to one total row, parses
   clock-format TOI, and drops the `League Average` footer. Also builds
   year-N → year-N+1 training pairs (`make_training_pairs`) and the
   injury-fallback lookup (`latest_healthy_row`, see below).
3. **Scoring** (`src/scoring.py`) — turns raw stats into the fantasy-points
   target. All scoring-rule changes belong in this one file
   (`SKATER_WEIGHTS`).
4. **Features** (`src/features.py`) — per-game rate stats (not raw totals,
   so a shortened season isn't penalized) plus age, GP, and prior-season
   fantasy points/game. `FEATURE_COLS` is the single list every model
   trains on.
5. **Modeling** (`src/train.py`) — separate ElasticNet and LightGBM models
   per position group (forwards, defense), grouped k-fold CV, most recent
   season held out for evaluation. ElasticNet currently wins on holdout
   Spearman correlation for both groups and is what production uses.
6. **Ranking** (`src/rank.py`) — refits ElasticNet on *all* season-pairs
   (no holdout, since a real draft board shouldn't waste data), predicts
   next-season points for every current player, and converts predictions
   into **value over replacement (VORP)**: predicted points minus the
   last starter-quality player at that position (108 forwards / 60 defense
   = 9F/5D × 12 teams). Also attaches draft-day "Notes" tags
   (`src/notable.py` for fragile/declining/rising, `src/nhl_api.py` for a
   live "changed teams this offseason" check). Running this module directly
   regenerates `output/draft_board.csv`.
7. **Draft-day app** (`src/app.py`, Streamlit) — live view over the board:
   filter by position/name, mark a player (or team) drafted by you or
   another manager, undo, inspect a player's season history, compare
   multiple players, and a "My Pool" tab tracking your roster against
   9F/5D/1G/1TEAM. Picks/managers persist to `state/*.json` so a multi-hour
   draft survives a restart. VORP recomputes live against only undrafted
   players (`src/draft_pool.py`). Goalies and teams get simple unranked
   listings — no model behind them.

## Running it

```
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Regenerate the static draft board:
```
.venv/Scripts/python.exe src/rank.py
```

Launch the live draft-day app:
```
.venv/Scripts/python.exe -m streamlit run src/app.py
```

Retrain / re-evaluate the models (ElasticNet vs LightGBM comparison):
```
.venv/Scripts/python.exe src/train.py
```

The app's "Explore" tab (web-search player lookups) needs a `MISTRAL_API_KEY`
environment variable (`.env` file, loaded via `python-dotenv`) before
launching Streamlit. Everything else needs no secrets.

## Maintaining this for a new season

Each fall, once Hockey-Reference has a full prior-season export available:

1. **Add the new season's data.** Download that season's skater exports from
   Hockey-Reference (basic stats + advanced stats pages — see
   `data/data_link.txt` for the URL pattern, just swap the year) and save
   them as `data/skaters-{season}.csv` and
   `data/skaters-advance-{season}.csv`, matching the existing filename
   convention (`skaters-2025_2026.csv` etc.). No code changes needed —
   `loading.load_all_seasons()` picks up every season it finds in `data/`
   automatically.
2. **Update `LATEST_SEASON` / `HOLDOUT_TARGET_SEASON`.** Bump the season
   string constants in `src/rank.py` (`LATEST_SEASON`) and `src/train.py`
   (`HOLDOUT_TARGET_SEASON`) to the new season so predictions and the
   train/holdout split move forward with the data.
3. **Regenerate the board.** Run `src/rank.py` to refit on all season-pairs
   (including the new one) and rebuild `output/draft_board.csv`.
4. **Sanity-check `src/train.py`'s holdout comparison** — rerun it after
   adding the season and re-read the Spearman/MAE numbers for both
   ElasticNet and LightGBM. CLAUDE.md notes the gap between the two has been
   narrowing as more seasons are added; if LightGBM ever overtakes
   ElasticNet on holdout, swap `rank.py`'s production model
   (`from train import make_elasticnet` → the LightGBM equivalent).
5. **Reset draft-day state for the new draft.** `state/picks.json` and
   `state/managers.json` hold last season's picks — clear or archive them
   before a new live draft (the app has no "start new draft" button, this
   is a manual file operation).
6. **Re-verify the NHL API team-change check still works** (`src/nhl_api.py`)
   — it hits an unauthenticated, unofficial endpoint
   (`api-web.nhle.com`) that could change shape season to season. It's
   built to degrade to "no team-change tag" on any failure rather than
   break the board, but worth a manual spot-check before relying on it
   live.
7. If a full season was COVID-shortened or otherwise anomalous (like
   2020-21's 56-game season), no special handling is required since the
   model runs on per-game rates — just keep an eye on it as a potential
   outlier in the holdout numbers.

## Known gaps

- **No goalie model.** The Hockey-Reference skater export only gives
  goalies their assist totals (no W/SV%/GAA/SO/starts), and goalie fantasy
  scoring isn't even defined yet. The app lists goalies unranked
  (name/team/GP only) as a placeholder. Deferred until goalie stats are
  sourced and a scoring rule is picked.
- **No hat-trick bonus.** Requested but not implementable from
  season-total data — would need a per-game log (e.g. NHL API gamelogs).
- **True rookies aren't on the board.** Players with zero prior NHL season
  in the loaded data have no stats to predict from and no injury-fallback
  season to use; would need external prospect data (junior/AHL stats,
  draft rankings) to cover.
- **No live injury flag.** The NHL's public API has no injury/IR endpoint,
  so "injured" and "retired" are indistinguishable beyond recency — the
  injury fallback (`loading.latest_healthy_row`) caps how far back it will
  reach (`MAX_SEASONS_BACK = 1` in `rank.py`) specifically to avoid
  resurrecting long-retired players.
- **The "1 team" roster slot isn't modeled** — tracked manually outside the
  app.

## Repo layout

```
data/            Hockey-Reference CSV exports (one pair per season) + name-lookup CSV (unused)
models/          Trained model artifacts (*.joblib), from src/train.py
output/          output/draft_board.csv, from src/rank.py
state/           Live draft picks/managers (JSON), from src/app.py
src/
  loading.py     CSV parsing, player-ID join key, training-pair construction, injury fallback
  scoring.py     Fantasy-point scoring rules
  features.py    Feature list + prior-season rate features
  train.py       Model training + holdout evaluation (ElasticNet vs LightGBM)
  rank.py        Production model refit + VORP draft board generation
  notable.py     Fragile/declining/rising "Notes" tags (local data only)
  nhl_api.py     Live NHL API team-change check
  draft_pool.py  Live undrafted-only board + goalie listing for the app
  draft_state.py Picks/managers persistence
  explore.py     Web-search player lookup (Mistral API) for the app
  app.py         Streamlit draft-day app
```
