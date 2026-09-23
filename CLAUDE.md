# Hockey Pool Player Point Prediction

## Project Goal
Build a model to predict player fantasy points for a season-long hockey pool, to inform draft picks. Highest cumulative points at end of season wins.

## Pool Format
- Draft is turn-based (like a snake draft), no player picked twice
- Roster per manager: 9 forwards, 5 defense, 1 goalie, 1 team
- Winner = highest total points at end of season
- Pool size and roster slots are per-season settings (`draft_state.DEFAULT_SETTINGS`: 12
  managers, 9F/5D/1G/1TEAM), editable in the app. Once a draft order is set, its length is
  the manager count -- the real count isn't known until draft day.

## Data
Hockey-Reference season exports in `data/`, **10 seasons: 2016-17 through 2025-26**:
- `skaters-{season}.csv` — basic stats (~600 F / ~325 D / ~100 G per season)
- `skaters-advance-{season}.csv` — Corsi/Fenwick/PDO/zone starts (skaters only, no goalies)
- `goalies-{YYYY}-{YYYY}.csv` — goalie stats (W/L/T-O/SV%/GAA/SO/GSAA/...), one header row.
  Note the **hyphen** (`goalies-2025-2026.csv`) vs the skater files' underscore
  (`skaters-2025_2026.csv`); `loading.load_goalie_season` maps between the two.
- `allPlayersLookup.csv` — different source (NHL `playerId`, not HR slugs); only joins by
  name. Unique value is exact `birthDate` + handedness. Unused so far.

`src/loading.py` handles the export quirks: two header rows, the HR player slug living in
an unnamed trailing column (read as `-9999`) which is the only reliable join key, traded
players appearing as a `2TM`/`3TM` total row plus per-team splits, clock-format TOI, and a
`League Average` footer row. `load_all_seasons()` stacks every season found; `make_training_pairs()`
builds year-N-features → year-N+1-target rows for consecutive season pairs only (a player who
sat out a year isn't paired across the gap) — 6561 skater pairs (4269 F / 2292 D) across the 9
transitions currently available. Goalies load separately via `load_all_goalie_seasons()`.

Note: **2020-21 was COVID-shortened (56 GP max)**. It's used as a features-year without special
handling since the model runs on per-game rates, but keep an eye out if it behaves like an outlier.

### Data gaps
- Goalie stats come from the `goalies-*.csv` files above; the skater export's goalie rows
  (assists only) are ignored.
- No per-game logs (so no hat-trick bonus) and no injury/active-roster flag -- see Status.

`data/nhl 2026-2027 projections/` — NHL.com's own 2026-27 projections, pasted as
`goalies.txt`/`teams.txt`/`fowards.txt`/`defense.txt` (one numbered line per player/team,
`Name, POS, TEAM: <value>`). The trailing number is **fantasy points** for `fowards.txt`/
`defense.txt`, but **projected wins** for `goalies.txt` and `teams.txt` (per `teams.txt`'s
own notes: team win totals are NHL.com's per-goalie win projections summed by team).
`src/nhl_projections.py` parses all four; `goalies.txt`/`teams.txt` drive ranking, while
`fowards.txt`/`defense.txt` are shown as a reference "NHL.com Projection" column in the
app's F/D table and stand in as `predicted_points` only for rookies the model can't rank —
see Status below. The folder name is hardcoded in `nhl_projections.PROJECTIONS_DIR` -- a new
season's projections need a new folder and that constant bumped.

## Settled
- **Goalie scoring**: 2 per win (regular or OT/SO), 1 per OT/SO loss (HR's `T/O` column),
  2 per shutout, 5 per goal, 2 per assist. `scoring.GOALIE_WEIGHTS`. A shutout win is 4.
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
- Python, pandas, scikit-learn, lightgbm (xgboost never used), streamlit (draft-day app),
  requests (NHL API), mistralai + python-dotenv (the app's "Explore" button, see Status)
- Alexandre has prior experience with `requests`-only API integrations (see Intervals.icu project) — similar pattern likely reusable if pulling live NHL stats

## Status
- [x] Confirm scoring system and finalize target variable (standard G+A + SHG bonus)
- [x] Load and explore last year's stats (`src/loading.py`)
- [x] Multi-season data (10 seasons, 2016-17 to 2025-26 — expanded from 6 after confirming
      more data helps) + train/target pairing
- [x] Build position-specific baseline models (F, D — goalies came later, see below). `src/train.py`.
      ElasticNet still wins both on holdout Spearman (0.84 F / 0.78 D, 2024-25 features ->
      real 2025-26 results), but the gap over LightGBM narrowed a lot going from 6->10
      seasons (F holdout MAE 11.01->10.52) — LightGBM was data-starved before, not a fixed
      loser; recheck this comparison again if more seasons get added. Stuck to 2016-17+ to
      avoid mixing in the pre-3-on-3-OT scoring environment. Feature list in
      `src/features.py`. Trained models saved to `models/*.joblib`.
- [x] Calibration of F/D predictions (`train.Calibration`, applied in `rank.predict_upcoming`):
      raw ElasticNet under-predicted the best players badly (2025-26 holdout top-10: F predicted
      82.9 vs actual 98.6, D 52.7 vs 66.2). Two causes: the shortened 2019-20/2020-21 *target*
      seasons (top-30 residuals of -5 to -20 there vs +3 to +13 in every full season), and
      points being roughly rate x TOI, a curve the linear model can't bend to. Fix: quadratic
      map raw -> actual, fit on out-of-fold predictions for full-length target seasons only
      (league max GP >= `train.FULL_SEASON_GP`). Order-preserving (Spearman unchanged). Holdout:
      F MAE 10.65->10.05 (now beats LightGBM's 10.52), top-10 95.9 vs 98.6 actual; D MAE
      7.85->7.61, top-10 60.0 vs 66.2. Tried pro-rating short-season targets to 82 GP instead
      / in addition -- no better. `python src/train.py` prints the calibrated rows too.
      F/D mix of the board's top 50 barely moved (18 D before and after).
- [x] Add value-over-replacement draft ranking. `src/rank.py` (`build_draft_board`).
      Replacement cutoff = managers x roster slots per position (108 F / 60 D at the default
      12 x 9F/5D), taken from the season's settings (see Pool Format). Uses ElasticNet (won
      the holdout eval) refit on all season-pairs, not just train.py's train-only split.
      The app saves its board as `output/draft_board_<season>.csv` (rebuilt by the app's
      "Recompute draft board" button); running `rank.py` directly writes a separate
      `output/draft_board.csv` using the most recently created season's settings -- handy for
      inspection, but the app doesn't read it.
- [x] Injury fallback: `loading.latest_healthy_row` -- a player under `features.MIN_GP`
      games in his most recent season falls back to his last season with enough games,
      instead of being dropped or judged on a handful of games. Used both when building
      training pairs (unlimited lookback -- safe, since a retired player has no future
      season to serve as a training target and gets filtered out anyway) and for live
      draft-board predictions (`rank.py`, capped at `MAX_SEASONS_BACK = 1` -- unlimited
      lookback there resurrected actually-retired players like Shea Weber/Ryan Ellis,
      since the data has no active-roster flag to tell "injured" apart from "retired,"
      just recency). 893 modeled players on the board (115 via the fallback), up from 778.
      The model still can't rank true rookies (no fallback season exists for them); the ones
      NHL.com projects are now added from its projections instead -- see the NHL.com-only
      rookies item below (903 rows total).
- [x] "Notes" draft-day tags: `src/notable.py` (fragile/declining/rising, local data only)
      + `src/nhl_api.py` (live NHL API team-change check) combine into a single `Notes`
      column (e.g. `"New Team • Fragile • Declining"`) shown in `app.py`'s F/D table.
      No live "currently injured" flag -- the NHL's public API has no injury/IR endpoint
      (checked; genuinely undocumented anywhere), out of scope entirely. Fragile/trend
      reuse data already loaded (`GP` history, `prior_fantasy_points_pg`), no network:
      fragile = GP short of `FRAGILE_GP_PCT` (75%) of that season's league-max in >=2
      qualifying (GP >= `MIN_GP`) seasons within the last 4 league seasons; trend = >=20%/25% point-rate move combined with an age
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
- [x] Goalie model: `src/goalies.py` (`python src/goalies.py` prints the eval). ElasticNet on 21 features
      (rates per GP, GS/GP/MIN workload, SV%, GSAA, GPS, plus 2-season averages of pts/game, SV%, GSAA,
      GP). Holdout (2024-25 -> 2025-26): MAE 15.9, Spearman 0.60 (mean baseline MAE 20.4; last-season
      total MAE 18.6, Spearman 0.50). Much noisier than skaters -- goalie output hinges on workload/team,
      which the data only weakly reveals. Tried and dropped: team goals-for context (no gain), LightGBM
      (worse; ~570 train rows). 2-season averages helped modestly and are kept. Goalies with <10 GP in
      their feature season get no prediction. Not modeled: starter/backup role changes from trades
      or signings -- which is why the app's goalie ranking was later switched to NHL.com's
      projections (see below). The model is kept for its eval but no longer feeds the app; the app
      still uses `goalies.load_scored_goalies()` for goalie history and points-per-win.
- [x] Draft-day app: `src/app.py`, run via `.venv/Scripts/python.exe -m streamlit
      run src/app.py` (or `run_app.bat`). Live view over `rank.build_draft_board()` -- filter by
      position/name, mark a player picked (by you or another manager), undo,
      and pull up / compare (up to 3) players' season histories. Tabs: Forwards & Defense,
      Goalies, Teams, My Pool, Draft Log. All draft state is **per season**: the sidebar
      season picker ("+ New season..." creates one) scopes everything to
      `state/seasons/<season>/` -- `picks.json`, `managers.json`, `settings.json`,
      `keepers.json` (JSON, atomic writes, survives restarts) -- so a multi-hour live draft
      isn't lost on a restart and a new pool year starts clean without touching old drafts.
      Pre-season-scoped `state/picks.json`/`managers.json` are migrated once into
      `state/seasons/2026-2027/` (`draft_state._migrate_legacy_state`). VORP
      recomputes live (`src/draft_pool.undrafted_board`) against only
      undrafted players, with the replacement level at the (open slots left at that
      position)-th best remaining player (`add_vorp`'s `filled`, counting live picks +
      keepers) -- so it stays put when the draft follows projection order and moves only on
      reaches/keepers/position runs. (Until 2026-09-23 the cutoff stayed at teams x slots
      among the remaining players, which slid the replacement level down as players were
      drafted and inflated VORP at positions drafted faster.) Reuses `rank.add_vorp` rather
      than a second implementation. Goalies and teams: originally an unranked list / untracked
      slot; both are now ranked off NHL.com projections (see below).
- [x] "Explore" button (`src/explore.py`): in the compare view under any table, sends the
      checked players' context to Mistral's web-search-grounded chat (`MISTRAL_MODEL`) for
      current news/injury/form -- a summary for 1 player, a pick recommendation for 2-3.
      Needs `MISTRAL_API_KEY` (env var or `.env`, loaded by python-dotenv). Unlike `nhl_api.py`,
      failures surface as an error in the app rather than degrading silently -- it's a manual,
      user-clicked feature with no sensible fallback. Not part of the board build.

- [x] Snake-draft auto-advance: "Managers in draft order" box in the Edit managers form (the only manager list; one per line, every
      manager once incl. your own team; stored in `managers.json`). `draft_state.snake_manager` /
      `next_slot` / `active_manager`: round 1 first-to-last, then reversed, alternating. Each pick records
      a `slot`; the "Drafted by" box defaults to whoever's on the clock and is re-defaulted after every
      pick, but stays changeable. A manual out-of-turn pick continues the snake from the overridden manager
      (not snap-back), and undo restores the prior on-clock manager. No order set = old behavior. A banner
      at the top shows on the clock / next.
- [x] Goalie and team ranking now come from NHL.com's own 2026-27 projections instead of
      this repo's models. `src/nhl_projections.py` parses `data/nhl 2026-2027 projections/
      goalies.txt` and `teams.txt` (one numbered "Name, POS, TEAM: wins" line per player/
      team -- the number is a **projected win total**, not fantasy points; a shared
      "X or Y" committee line splits into one row per goalie, same win total/team).
      Deliberate choice over the ElasticNet goalie model (`src/goalies.py`, kept but
      no longer wired into the app): NHL.com's projections bake in this season's actual
      starter/backup depth chart and offseason moves, which historical stats alone can't
      see. `draft_pool.goalie_pool` now builds off the projection list directly (not
      Hockey-Reference history), joining in GP by normalized name for display only --
      a goalie with no HR history (true rookie) still gets ranked, just without that
      column (`feature_season`/`seasons_back` dropped from the goalie table entirely --
      they described the HR-history join, not the projection, and were confusing next
      to a projections-based ranking). `draft_pool.team_pool`/`undrafted_teams` add VORP
      ranking for the "1 team" slot (previously unranked, just an alphabetical pick list)
      off NHL.com's projected win totals, live team names from the NHL API. F/D ranking is
      unchanged (still the in-repo model trained on Hockey-Reference history);
      every F/D row carries NHL.com's number as `nhl_projection` ("NHL.com Projection" in the
      app), joined by (normalized name, pos_group) with a nickname fallback
      (`nhl_projections.match_projection_rows`: last name + first-name 3-letter prefix for
      Josh/Joshua, plus `_FIRST_NAME_ALIASES` for Tommy/Thomas), used only when unique on
      both sides so brothers (Ilya/Aliaksei Protas) never cross-match.
- [x] Goalie/team VORP on the same scale as skaters (backlog #1). Before this, goalie/team
      `predicted_points` were raw projected *wins*, so their VORP was ~2.5x too small next to
      skater fantasy points. Goalies: wins x points-per-win (`draft_pool.goalie_points_per_win`:
      last-3-season fantasy_points/W shrunk toward the league's 2.51 with an 80-win prior --
      most individual spread is shutout/OTL luck). Teams: standings points
      (`scoring.TEAM_WEIGHTS`) = 2 x W + league-average OTL (`league_otl_per_team_game` x games
      per team, derived from the projections: 2 x total wins / teams = 84 for 2026-27). OTL is a
      constant across teams, so it moves displayed points, not VORP. Saved goalie/team boards
      without the new columns rebuild on load. Also: once a draft order is set, its length is
      `num_managers` (`draft_state.load_settings`; settings input locked) -- a stale count put
      every replacement level at the wrong depth.
- [x] NHL.com-only rookies on the F/D board: `rank.add_projection_only_players` appends
      every NHL.com-projected F/D the model couldn't rank (no history, or only call-up games
      under `MIN_GP` -- McKenna, Martone, Stenberg, ... 10 players for 2026-27) with NHL.com's
      projection as `predicted_points` (`source == "nhl.com"`), so they get VORP/pos_rank.
      Deliberately mixes two projection sources in one ranking. They keep their HR
      `player_id` if they have any call-up games, else a synthetic `proj_` id; Team comes
      from NHL.com; never tagged "No Team". A "Rookie" Notes tag (`notable.is_rookie`)
      follows the NHL's Calder rule: never >25 GP in a season, never 6+ GP in two seasons,
      age <=26 -- so it also tags modeled players with thin call-up histories (Frondell,
      Cole Hutson), whose model predictions rest on 12-14 GP and run well under NHL.com's.
      `app.get_board` rebuilds a saved board that predates the `nhl_projection` column.
- [x] Goalie/team pools now persist per-season too (`output/goalie_board_<season>.csv`,
      `output/team_board_<season>.csv`), same pattern as `draft_board_<season>.csv`.
      Previously they were only `@st.cache_data`-memoized, so every app restart re-hit the
      live NHL API (rate-limited) to rebuild them. "Recompute draft board" rebuilds and
      re-saves all three now, not just the F/D board.
- [x] Keepers (K1 in `backlog.md`): a manager's last pick is his keeper, kept into the next
      season as his **last-round pick** (he skips the final round); skaters only, one per manager,
      no season limit. Entered before the draft in the sidebar "Keepers from last season" box,
      stored in `state/seasons/<season>/keepers.json` apart from live picks (no pick_number, not
      touched by undo). `draft_state.reserved_slots` derives each keeper's final-round snake slot
      from the current order + roster size (`total_rounds` = F+D+G+TEAM slots), and `next_slot`
      skips them. `drafted_player_ids`/`roster_picks` include keepers, so they're off the board and
      count toward roster limits and "My Pool". League rules and remaining ideas: `backlog.md`.

## Setup
```
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```
Launch the app with `run_app.bat` (or the streamlit command above). Optional: `MISTRAL_API_KEY`
in `.env` for the Explore button; nothing else needs secrets. New-season maintenance steps are
in `README.md`.
