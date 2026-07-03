# Prediction Model Improvement Plan

**Companion to:** `reports/OUTCOME_MODEL_RESULTS.md` (Phase B), `reports/OUTCOME_MODEL_PHASE_C.md`,
`reports/OUTCOME_MODEL_BASKETBALL.md`. Written 2026-07-03. Analysis only — no code changed.

## Where the models stand today

| Model | Data | OOS log loss | OOS accuracy | Ceiling |
|---|---|---:|---:|---|
| Football `outcome_v2` (elo_logistic) | 13 EPL seasons, 3 800 OOS matches | 0.972 | 0.539 | Bookmaker: 0.953 / 0.556 |
| Basketball `outcome_v2` (elo_logistic) | 18 NBA seasons, 17 889 OOS games | 0.614 | 0.660 | not measured (no odds in dataset) |
| v1 strength-logistic + heuristic | internal finished matches | — | — | fallback only |

Two structural facts drive everything below:

1. **Football headroom is small but real.** The gap to the de-margined Bet365 ceiling is
   +0.020 log loss. We will never beat the bookmaker (they price in lineups, injuries,
   market flow), but closing a third to a half of that gap is a realistic target
   (~0.960–0.965 pooled log loss).
2. **Basketball headroom is large and mostly about data.** `nba.sqlite` ends with the
   2022-23 season, so 2026 predictions run on ratings that are three seasons stale, and
   none of the well-documented NBA schedule effects (rest, back-to-backs) are used.

The v1 model trains on internally simulated matches — it cannot learn real-world skill and
should stay a fallback. All improvement effort belongs in the v2 pipeline.

## Ranked improvements — football

Ordered by expected impact per unit of effort. All are evaluated with the existing
walk-forward harness (`scripts/backtest_outcome.py` folds) so gains are apples-to-apples.

1. **Tune the Elo hyperparameters.** `team_ratings.py` hardcodes K=20, home_advantage=65,
   25% season reversion — never optimized. Grid-search (K ∈ 10..40, HA ∈ 40..90,
   reversion ∈ 0..0.4) on pooled walk-forward log loss. Effort: hours. Expected: small
   but free (~0.001–0.004).
2. **Margin-of-victory Elo.** Plain Elo ignores whether a win was 1-0 or 5-0. The
   FiveThirtyEight-style multiplier `ln(1 + goal_diff) × autocorrelation guard` is a
   standard, well-tested upgrade. Effort: ~half a day. Expected: 0.003–0.008.
3. **Add `abs(elo_diff)` to the logistic layer.** Draw probability peaks when teams are
   even; a multinomial logistic on `elo_diff` alone cannot represent that shape. One
   extra feature, known win for 3-class football.
4. **Ensemble the Phase C candidates.** elo_logistic (best LL), logistic_form (best
   accuracy), dixon_coles (best ECE) make decorrelated errors. Blend their probabilities
   with weights fit per fold on past data (or a logistic stacker). Classic result: the
   blend beats every member. Effort: ~half a day on top of existing code. Expected:
   0.005–0.010 — probably the single biggest football win.
5. **Shot-based rolling form.** Phase C's goals-based form features (ppg, gf, ga) added
   nothing — goals are too noisy. The historical CSVs already carry shots (HS/AS),
   shots on target (HST/AST) and corners; rolling SoT for/against is a much less noisy
   quality signal (the basis of xG-era models). Replace, don't add to, the failed
   goal-form features. Effort: ~a day.
6. **Pool more leagues.** football-data.co.uk serves the Championship (E1) and other
   countries in the identical format the loader already parses. More matches stabilize
   the mapping layer and calibration (train with a league dummy; Elo pools per league).
   Effort: mostly data plumbing.

Not worth doing: GBMs/neural nets on ~4 900 matches (Phase C already showed 7 features
overfit relative to 1); bookmaker odds as an input (not available at predict time in the
app — keep them as the evaluation ceiling only).

## Ranked improvements — basketball

1. **Refresh the dataset (highest-impact item in the whole plan).** Ratings frozen at
   2022-23 mean current-team predictions are near-noise. Source 2023-24 → 2025-26
   results (e.g. an updated Kaggle dump or the NBA stats API), append to the loader,
   rebuild the bundle.
2. **Rest and schedule features.** Back-to-back and 3-games-in-4-nights effects are worth
   1–3 accuracy points in the NBA literature, and game dates are already in the data.
   Add `rest_diff` and `is_b2b` flags to the logistic layer.
3. **Margin-of-victory Elo** — same change as football, shared code in
   `team_ratings.py`; FiveThirtyEight's NBA Elo is the reference implementation.
4. **Tune K / home advantage per sport, and let HA vary by era** — NBA home advantage has
   declined markedly; a fixed 100 overweights it for recent seasons.
5. **Get an odds baseline.** Without a bookmaker ceiling we can't tell how close to
   optimal 0.614 is. Any historical NBA odds source (even one season) calibrates
   expectations.

## Infrastructure that compounds

- **Score stored predictions.** `match_predictions` rows are written on every `/predict`
  but never evaluated. A small script (or `/predict/metrics` endpoint) joining stored
  probabilities to final results and reporting rolling log loss + calibration turns
  production into a continuous backtest and detects drift for free.
- **Keep Elo fresh after training.** The bundle's `elo_ratings` dict is frozen at build
  time; matches played afterwards never move ratings. Add either a periodic
  rebuild (cron on `finalize_outcome_v2.py`) or an online update step when real
  results arrive.
- **One harness for every candidate.** Any change above lands only if it beats the
  incumbent on the same expanding-window folds in `backtest_outcome.py` — the discipline
  that made Phase B/C trustworthy.

## Suggested order of work

| Step | Items | Effort | Expected outcome |
|---|---|---|---|
| 1 | ✅ **Done** (`reports/OUTCOME_MODEL_V3.md`): tuning shipped (SR 0.25→0.10, LL 0.972→0.971); abs(elo_diff) and the ensemble added nothing (errors too correlated) — single-source EPL team-level features are near-exhausted | ~1–2 days | LL 0.972 → ~0.962–0.966 *(actual: 0.971)* |
| 2 | ✅ **Done** (`reports/OUTCOME_MODEL_BASKETBALL_V3.md`): dataset refreshed through 2025-26 (`scripts/fetch_nba_recent.py`); MOV Elo + tuned SR=0.40 shipped (LL 0.614→0.609, acc 0.660→0.665); rest/b2b features proven in backtest (LL 0.607, acc 0.667) but need a real schedule feed to go live | ~2–3 days | acc 0.66 → ~0.67–0.69 *(actual: 0.665 shipped, 0.667 with rest)* |
| 3 | ✅ **Done** (`reports/OUTCOME_MODEL_V4.md`): MOV Elo (fte, K=40) + rolling SoT form shipped as `outcome_v4_elo_sot` (LL 0.971→0.966; gap to bookmaker +0.018→+0.013); bundle now carries `features` + `sot_form`, predict path assembles features from the bundle | ~1–2 days | LL → ~0.958–0.963 *(actual: 0.966)* |
| 4 | ✅ **Done**: `GET /predict/metrics` scores stored predictions vs results per model_version (accuracy/LL/Brier/ECE + calibration bins); `scripts/refresh_bundles.py` rebuilds both bundles on a cron and hot-swaps them via new `POST /models/reload` (no restart) | ~1 day | prevents silent decay |
| 5 | ✅ **Done**: multi-league pooling is a documented NEGATIVE result (`reports/OUTCOME_MODEL_V5.md` — even division-offset variants lose to E0-only v4, 0.967 vs 0.966; Championship data kept for revisits). NBA odds ceiling measured (`reports/OUTCOME_MODEL_NBA_ODDS.md`): market 0.593 vs elo_mov 0.615 log loss on 11,656 same games → +0.022 headroom, ~10% of which the backtest-proven rest features would close | as needed | better calibration, honest ceilings |
