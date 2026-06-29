# Outcome Model — Basketball (NBA) Walk-Forward Backtest

**Companion to:** `reports/OUTCOME_MODEL_RESULTS.md` (football), `reports/MODEL_VALIDATION_REPORT.md`
**Data:** 21579 real NBA regular-season games (Kaggle nba.sqlite), 18 seasons. **Walk-forward, expanding window**, strictly causal.
**No draws** (2 classes). **No bookmaker odds** in this dataset, so there is no market ceiling baseline. No app model was trained or modified.

## Pooled out-of-sample metrics

| Baseline | n | Accuracy | Log loss | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| always_home | 17889 | 0.581 | 14.461 | 0.837 | 0.419 |
| base_rate | 17889 | 0.581 | 0.680 | 0.487 | 0.014 |
| elo_logistic | 17889 | 0.660 | 0.614 | 0.426 | 0.011 |

- **always_home** — predict Home every game. Accuracy 0.581 (NBA home-court is strong). Huge log loss = the penalty for hard 0/1 predictions.
- **base_rate** — training home/away frequencies.
- **elo_logistic** — multinomial logistic on causal Elo difference (1 feature). Accuracy 0.660, log loss 0.614, ECE 0.011.

**Headline:** Elo beats the strong always-home baseline (0.660 vs 0.581) and beats base-rate on log loss, out-of-sample over ~17889 games, while staying well-calibrated (ECE 0.011). NBA home advantage is large, so the accuracy gap over always-home is naturally smaller than in football — the real value shows up in calibrated probabilities (log loss / Brier).

## Per-season out-of-sample accuracy

| Season | always_home | base_rate | elo_logistic |
|---|---:|---:|---:|
| 2007-08 | 0.601 | 0.601 | 0.678 |
| 2008-09 | 0.608 | 0.608 | 0.688 |
| 2009-10 | 0.594 | 0.594 | 0.689 |
| 2010-11 | 0.604 | 0.604 | 0.685 |
| 2011-12 | 0.586 | 0.586 | 0.674 |
| 2013-14 | 0.580 | 0.580 | 0.654 |
| 2014-15 | 0.575 | 0.575 | 0.671 |
| 2015-16 | 0.589 | 0.589 | 0.680 |
| 2016-17 | 0.584 | 0.584 | 0.645 |
| 2017-18 | 0.579 | 0.579 | 0.661 |
| 2018-19 | 0.593 | 0.593 | 0.652 |
| 2019-20 | 0.551 | 0.551 | 0.647 |
| 2020-21 | 0.544 | 0.544 | 0.598 |
| 2021-22 | 0.544 | 0.544 | 0.649 |
| 2022-23 | 0.580 | 0.580 | 0.627 |

## Method notes

- Loader: `scripts/load_nba.py` (read-only). Regular season since 2004; Elo keyed by stable franchise team_id.
- Causal Elo: `app/services/team_ratings.py` (K=20, home advantage=100, 25% season mean-reversion).
- Note: the 2012-13 season is absent from this Kaggle dump (data gap); walk-forward is unaffected (Elo carries across the gap).
