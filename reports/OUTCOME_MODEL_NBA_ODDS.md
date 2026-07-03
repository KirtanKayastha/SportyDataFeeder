# Outcome Model — NBA Market (Odds) Baseline

**Companion to:** `reports/MODEL_IMPROVEMENT_PLAN.md` (step 5),
`reports/OUTCOME_MODEL_BASKETBALL_V3.md`.
**Data:** closing moneylines for 11656 of our regular-season games
(seasons 2011-12..2021-22; sportsbookreview archive via
`historical-datas/nba_odds_10y.json`). De-margined two-way probabilities.
Both columns are scored on exactly the same games, walk-forward, out-of-sample.

## Pooled out-of-sample metrics (odds-matched games only)

| Model | n | Accuracy | Log loss | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| elo_mov | 11656 | 0.658 | 0.615 | 0.427 | 0.016 |
| bookmaker | 11656 | 0.681 | 0.593 | 0.408 | 0.007 |

- **elo_mov** — the production basketball config (MOV Elo, K=20, HA=60, SR=0.40).
- **bookmaker** — de-margined closing moneylines: the practical ceiling.

**Gap to the market: +0.0221 log loss.** This is the number the basketball
backtests were missing — it bounds how much headroom is left for feature work
(rest/b2b, which backtests at ~-0.002, fits inside it; see the v3 report).

## Per-season out-of-sample accuracy (matched games)

| Season | n | elo_mov | bookmaker |
|---|---:|---:|---:|
| 2011-12 | 990 | 0.666 | 0.685 |
| 2013-14 | 1230 | 0.663 | 0.689 |
| 2014-15 | 1230 | 0.680 | 0.702 |
| 2015-16 | 1228 | 0.695 | 0.698 |
| 2016-17 | 1230 | 0.648 | 0.667 |
| 2017-18 | 1230 | 0.650 | 0.683 |
| 2018-19 | 1230 | 0.655 | 0.673 |
| 2019-20 | 979 | 0.658 | 0.672 |
| 2020-21 | 1079 | 0.618 | 0.656 |
| 2021-22 | 1230 | 0.648 | 0.677 |
