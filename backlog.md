# Backlog

Proposed upgrades to improve draft results, ranked by expected impact on winning vs. effort.
Draft is **Friday 2026-09-25**; items are split into "before the draft" and "after the draft".

## League facts (confirmed 2026-09-23)

- Number of managers: **unknown until draft day**. Everything must use the count entered in the
  app (already the case -- `12` is only `draft_state.DEFAULT_SETTINGS`).
- Draft slot: known only a few minutes before the draft -- anything slot-dependent must work
  from the order typed in live.
- Team slot scoring: **same as NHL standings points** (2 per win, 1 per OT/SO loss).
- In-season moves: **3 swaps with undrafted free agents** for the whole season.
- Payout: **top 2**.
- Opponents: draft habits vary widely per manager (NHL.com ranks, Yahoo, gut feel...).
- **Keeper:** the last player each manager drafts is a "keeper" -- he can be kept in that
  manager's pool for the next season.
  - Other managers **already hold keepers** from last season → K1 is needed for Friday.
  - The keeper takes one of the 16 roster spots and scores like any other player.
  - Skaters only, and he must be playing in the NHL. If a manager's last pick is a goalie or
    a team, that manager simply has no keeper.
  - No limit on how many seasons a player can be kept. A held keeper counts as the manager's
    last-round pick, so that manager skips the final round.
  - You hold **no keeper** -- your final-round pick becomes your keeper.

## Before the draft

### 1. Goalies and teams on the fantasy-point scale + manager count from the draft order — ✅ DONE (2026-09-23)
- **Built:** goalies = projected wins × points-per-win (own last-3-season ratio shrunk toward
  the league's 2.51 with an 80-win prior; ratios end up 2.41–2.63). Teams = 2 × W + league-average
  OTL (0.11 per team-game × 84 games = 9.4, same for every team, so it doesn't change VORP).
  At 12 teams: Vasilevskiy 95.6 pts / VORP 19 (was 8), Colorado 115 pts / VORP 18 (was 9);
  the best goalie/team now rank ~60th/65th overall by VORP. Manager count = draft-order length
  whenever an order is set (`draft_state.load_settings`); the settings input is locked then.
  Betting-market team totals not done (optional).
- **Problem:** goalie/team VORP is in *wins*, skater VORP in *fantasy points*, so goalies and
  teams look far less valuable than they are and get drafted too late.
- **Goalies:** 2 W + 1 OTL + 2 SO. From 2022-23+ goalies with 40+ GP: OTL/W = 0.21,
  SO/W = 0.11, so **~2.5 fantasy pts per projected win** (use each goalie's own history
  ratio when available). Example: #1 goalie (39 W) ≈ 98 pts, 12th (31 W) ≈ 78 → real VORP
  ≈ 20, while the app shows 8.
- **Teams:** 2 × W + estimated OTL (NHL.com's team numbers exclude OTL -- see `teams.txt`
  notes). Optional upgrade: paste betting-market season point totals (over/unders), which
  are already in standings points and usually the most accurate team projection.
- **Manager count:** `num_managers` (League settings) and the draft-order list are stored
  separately. If they disagree, replacement levels are wrong. Set the manager count from the
  length of the draft-order list, or at least warn on a mismatch.

### 2. Live injury feed (ESPN) — ✅ DONE (2026-09-23)
- **Built:** `src/espn_injuries.py`. Games missed come from ESPN's structured `returnDate`
  (no comment parsing needed) vs. the regular-season dates from the NHL schedule endpoint;
  points are scaled by the share of the season missed before VORP, for F/D and goalies.
  Notes tag + "Injury" detail column in the app. On 2026-09-23: 56 F/D + 4 G tagged; Terry
  62 → 45 pts (~23 GP), Marchand 58 → 48 (~15 GP), Gustavsson ~14 GP.
- NHL.com's `(INJ.)` marker is **not** used (static snapshot) -- decided against it.
- Not done: the "FA replacement minus one swap" floor for long injuries, manual override.
- `https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/injuries` -- unauthenticated JSON,
  confirmed live (77 injured listed on 2026-09-23, preseason 2026-27). Has status
  (Out / IR / Day-to-day / Suspension) and a text comment with timelines.
- None of those players is flagged on the board now. Examples: Troy Terry (pred 62) out
  2-3 months (hip), Marchand IR through October, Fiala (broken leg) 3+ weeks.
- Also carry NHL.com's own `(INJ.)` marker for F/D -- `nhl_projections.load_skater_projections`
  currently strips it (Bedard, Marchand).
- Show an "Injured: <comment>" tag; reduce predicted points by estimated games missed
  (parse "N weeks/months" from the comment, or allow a manual games-missed override).
- With only 3 free-agent swaps per season, value an injured player by the games he'll actually
  play. A season-ending injury ≈ the free-agent replacement's value minus one swap.
- Same "never raise, degrade to no tag" convention as `nhl_api.py`. Matching by
  normalized name + position, as for the NHL API.

### 3. Multi-season skater features — ✅ DONE (2026-09-23)
- **Built:** `features.add_multi_season_features`, in `FEATURE_COLS` for both F and D. Shipped
  set = the list below minus Age² (ablation: no gain, and it would drift from Age when the injury
  fallback bumps Age forward). Missing lags stay NaN (median-imputed); filling them with the
  current season was slightly worse. Table below = shipped set (the Age² prototype was within ~0.1 MAE).
- Board effect: players coming off one off-year move up (Matthews 62 → 74, Makar 69 → 78,
  Point 52 → 60, Hedman 24 → 35), one-year spikes come down (Raddysh 64 → 52, Malkin 63 → 54).
  Correlation with the old predictions 0.994, mean shift 0. Top-50 F/D mix about the same (17 D vs 18).
- The skater model only sees one season (the goalie model already uses 2-season averages).
- Added: last 2 prior seasons' pts/GP, GP share of season max, ATOI, a Marcel-style
  5/4/3 GP-weighted pts/GP, age², number of prior seasons.
- Rolling holdouts (train on all earlier pairs, calibrated):

  | Holdout | F MAE | D MAE | Top-30 MAE |
  |---|---|---|---|
  | 2023-24 | 11.39 → 10.91 | 8.43 → 7.98 | F 16.1 → 14.9, D 16.6 → 14.4 |
  | 2024-25 | 10.37 → 10.31 | 6.89 → 6.83 | F 15.9 → 15.5, D 12.0 → 12.0 |
  | 2025-26 | 10.05 → 9.87 | 7.61 → 7.31 | F 13.8 → 13.7, D 11.7 → 10.4 |

  MAE improved in 6 of 6 tests and Spearman in 5 of 6 (D 2024-25 is essentially flat). The biggest
  gains are on the top players (early rounds).

### 4. Blend with NHL.com projections + disagreement column — S–M
- Correlation 0.94, but the model runs **~11 pts lower on average** than NHL.com (the model
  bakes in expected missed games; NHL.com seems to assume a near-full season), and a few
  early-round players are far apart:

  | Player | Model | NHL.com |
  |---|---|---|
  | Brayden Point | 51 | 85 |
  | Auston Matthews | 62 | 86 |
  | Lane Hutson | 57 | 80 |
  | Sidney Crosby | 65 | 90 |
  | Michael Misa | 28 | 53 |
  | Logan Cooley | 52 | 80 |

- Blend the two (weighted average, after putting both on the same scale). Weight toward
  NHL.com for players with few career NHL GP (thin samples like Misa, Frondell, Cole Hutson).
- Blend weight can't be validated without past NHL.com projections. Try the Wayback Machine
  for NHL.com's 2025-26 fantasy projections; otherwise start at 50/50.
- Disagreement column = where to spend research time (the "Explore" button).

### 5. Projected standings + roster needs in "My Pool" — S–M
- Projected total for every manager from their picks → live projected standings (top 2 paid).
- Your remaining needs vs. picks left (e.g. "3 D needed, 4 picks left"); alert when a
  position is about to run out (a big drop to the next tier).

### 6. Pick-timing assistant — M (riskiest to finish by Friday)
- For each player: probability he's still available at your next pick, computed from the
  live snake order (the app already knows who's on the clock).
- Recommend "take the player you'd lose now" rather than the top VORP player, when the
  top one is likely to still be there next turn.
- Draft-order estimate: NHL.com projection rank as a loose consensus, with wide noise since
  opponents draft very differently. Could also show a warning when many players at one
  position are going in a row.

### Late-round upside hint — S (light version of #9)
- With free-agent swaps, a late pick that flops can be replaced but a breakout is kept, so
  late picks should favor ceiling (young / "Rising" / new top-6 or PP role) over a steady
  veteran with the same projection. Show a hint in the late rounds.
- The very last pick is the keeper (below) and follows different logic.

### Keeper — S–M
**K1. Keepers held from last season — ✅ DONE (2026-09-23).**
- Confirmed: a manager's keeper **is** his last-round pick. He's on the roster before the
  draft starts, and that manager skips the final round.
- Built: sidebar "Keepers from last season" box (manager + searchable F/D player, one per
  manager, ✕ to remove), stored in `keepers.json` apart from live picks (undo never touches
  them). Keepers are off the board, count toward roster limits / "My Pool" / the drafted
  list, and are listed in the Draft Log. The snake skips each keeper holder's final-round
  turn (`draft_state.reserved_slots`, derived from the current order + roster size, so
  editing either afterwards stays correct). The banner shows "round X of 16".
- Verified: simulated 4-manager draft with 2 keepers → 62 live picks, the 2 keeper holders
  skipped in round 16, all 4 rosters end at exactly 16; headless app run shows no errors.

**K2. Keeper value for the last pick.**
- The last pick should be ranked on **this season + future seasons** value, not this season
  alone. With no limit on seasons, a keeper is a multi-year asset: young NHL skaters on the
  rise and rookies already in the NHL (no prospects, no goalies, no teams).
- Keeper value ≈ this season's points + discounted sum over future seasons of (keeper's
  projected points − points of a typical last-round pick), since keeping him uses up your
  last pick each year.
- **Roster planning:** your last open slot must be an F or D to have a keeper at all. Warn
  in the app if you're about to leave only a goalie or team slot for the last round.
- Other managers are hunting keepers in the last rounds too -- the best young candidates
  may need to be taken a round or two early (ties in with #6).
- You hold **no keeper** from last season, so you pick in the final round and that pick
  becomes your keeper -- K2 applies to you in full.
- Next-season projection: train a direct year N → year N+2 model (same pipeline as `train.py`,
  pairs two seasons apart), validated with the same rolling holdouts. Compare to a simpler
  age-curve multiplier on the current prediction (average N+1/N points ratio by age, fit
  from the data).
- VORP vs. next season's replacement level; show a "Keeper value" column / a keeper view
  in the app when your last pick comes up.
- Also useful for trades: if a keeper can be picked up as a free agent in-season, then #7
  should weigh future value too.

**Open question (minor):** can a manager release his keeper back to the pool before the
draft? Only matters if someone does it on Friday; then just don't enter that keeper in K1.

## After the draft

### 7. Free-agent swap tracker — M
- Recompute projections during the season with current-season stats (e.g. NHL API /
  MoneyPuck), rank undrafted players against your weakest roster player, and show the points
  gained over the rest of the season for each swap.
- Only 3 swaps per season, so timing matters; also monitor injuries on your roster (ESPN feed from #2).
- Open questions: do a dropped player's points stay with you? Is there a swap deadline?

### 8. MoneyPuck data upgrade — M
- `https://moneypuck.com/moneypuck/playerData/seasonSummary/{year}/regular/skaters.csv`
  (confirmed reachable, ~3.5 MB/season). Per-situation splits (5on4 ice time = power-play
  TOI, which CLAUDE.md lists as a feature but the model never had), individual xGoals
  (better luck regression than SPCT), primary vs. secondary assists.
- Keyed by NHL `playerId` → first real use for `data/allPlayersLookup.csv` (NHL playerId +
  birthDate) to bridge to Hockey-Reference slugs.
- Validate with the same rolling holdouts as #3.

### 9. Ceiling/floor projections — M
- Quantile models (P10/P90) or residual-based ranges per player. With the top 2 paid (not
  winner-take-all), this matters less; mainly useful for late rounds and swap decisions.

### 10. Rookie / prospect projections — S → L
- Quick version: covered by #4 (lean on NHL.com for thin NHL samples).
- Full version: NHLe translation of junior / AHL / European stats from the NHL API
  (`api-web.nhle.com/v1/player/{id}/landing` → `seasonTotals` across leagues).

### 11. Line and PP assignments, team-change context — M (fragile)
- Daily Faceoff line combos (PP1 / top-6) as features or tags; adjust for players on a
  new team. Scraping-based, so it could break.

### 12. Hat-trick bonus — M, low impact
- Needs per-game logs (`api-web.nhle.com/v1/player/{id}/game-log/{season}/2`) to count 3-goal
  games. Only about 0–3 pts/season for elite goal scorers. Tie-breaker between snipers at best.

## Smaller issues noticed

- ✅ **FIXED 2026-09-23 — Live VORP replacement level drifted during the draft** (`draft_pool.undrafted_board` →
  `rank.add_vorp`): the cutoff stays at `teams × slots` counted among the *remaining*
  players, so after k forwards are gone the replacement is the original (108 + k)-th forward
  instead of staying near the 108th. Positions drafted faster look inflated (e.g. F vs D late
  in the draft); keepers add to the drift. Fix: cutoff = open slots left at that position
  (`teams × slots − drafted/kept at that position`). Done for F/D, goalies and teams.

- `state/seasons/test` is set to 3 managers -- create the real season with the correct count.
- Gabriel Perreault (NYR, NHL.com 57) tagged "No Team", probably assigned to the AHL -- the
  "No Team" tag may mislead for AHL assignments; consider ignoring it when NHL.com projects
  the player.
- README still says "12 teams" and "goalie/team slots aren't modeled" -- out of date.
