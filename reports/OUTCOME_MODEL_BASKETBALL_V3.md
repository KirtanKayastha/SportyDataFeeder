# Outcome Model — Basketball v3 (Refreshed Data + MOV Elo + Rest)

**Companion to:** `reports/MODEL_IMPROVEMENT_PLAN.md` (step 2),
`reports/OUTCOME_MODEL_BASKETBALL.md` (baseline on the stale dump).
**Data:** 25269 real NBA regular-season games, 21 seasons
(2004-05..2025-26) — Kaggle dump + basketball-reference refresh
(`scripts/fetch_nba_recent.py`). Walk-forward, expanding window, strictly causal.
**No live app model was modified by this script.**

## Stage 1 — Elo hyperparameter grid search

Grid: mov ∈ {none, fte}, K ∈ [10.0, 15.0, 20.0, 25.0, 30.0, 40.0], HA ∈ [60.0, 100.0, 140.0], SR ∈ [0.0, 0.1, 0.25, 0.4];
objective = pooled walk-forward log loss of the 1-feature elo_logistic model.

- **Best classic Elo:** K=20, HA=60, SR=0.40 → LL 0.61311
- **Best MOV Elo (FiveThirtyEight multiplier):** K=20, HA=60, SR=0.40 → LL 0.60916

Top 10 configs:

```
mov    k  home_advantage  season_regression  log_loss
fte 20.0            60.0               0.40  0.609156
fte 20.0           100.0               0.40  0.609308
fte 15.0            60.0               0.40  0.609403
fte 20.0            60.0               0.25  0.609421
fte 20.0           100.0               0.25  0.609607
fte 20.0           140.0               0.40  0.609620
fte 15.0           100.0               0.40  0.609654
fte 25.0            60.0               0.25  0.609656
fte 25.0            60.0               0.40  0.609656
fte 25.0           100.0               0.40  0.609731
```

## Stage 2 — candidate comparison (pooled out-of-sample)

| Model | n | Accuracy | Log loss | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| elo_incumbent | 21579 | 0.660 | 0.614 | 0.426 | 0.010 |
| elo_tuned | 21579 | 0.661 | 0.613 | 0.425 | 0.010 |
| elo_mov | 21579 | 0.665 | 0.609 | 0.422 | 0.010 |
| elo_rest | 21579 | 0.663 | 0.611 | 0.423 | 0.009 |
| elo_mov_rest | 21579 | 0.667 | 0.607 | 0.420 | 0.009 |

- **elo_incumbent** — the production bundle config (K=20, HA=100, SR=0.25, classic Elo).
- **elo_tuned / elo_mov** — stage-1 winners (single elo_diff feature).
- **elo_rest / elo_mov_rest** — + causal `rest_diff`, `home_b2b`, `away_b2b`
  (`app/services/features_team.py:annotate_rest_days`).

**Winner:** `elo_mov_rest` (pooled OOS log loss 0.6071 vs incumbent
0.6139; accuracy 0.667 vs 0.660).

## Per-season out-of-sample accuracy

| Season | elo_incumbent | elo_tuned | elo_mov | elo_rest | elo_mov_rest |
|---|---:|---:|---:|---:|---:|
| 2007-08 | 0.678 | 0.674 | 0.685 | 0.677 | 0.683 |
| 2008-09 | 0.688 | 0.690 | 0.691 | 0.695 | 0.695 |
| 2009-10 | 0.689 | 0.690 | 0.693 | 0.689 | 0.691 |
| 2010-11 | 0.685 | 0.685 | 0.688 | 0.678 | 0.679 |
| 2011-12 | 0.674 | 0.674 | 0.666 | 0.671 | 0.668 |
| 2013-14 | 0.654 | 0.653 | 0.663 | 0.657 | 0.667 |
| 2014-15 | 0.671 | 0.678 | 0.680 | 0.678 | 0.678 |
| 2015-16 | 0.680 | 0.687 | 0.694 | 0.687 | 0.689 |
| 2016-17 | 0.645 | 0.646 | 0.648 | 0.644 | 0.653 |
| 2017-18 | 0.661 | 0.661 | 0.650 | 0.663 | 0.656 |
| 2018-19 | 0.652 | 0.656 | 0.655 | 0.653 | 0.659 |
| 2019-20 | 0.647 | 0.645 | 0.652 | 0.653 | 0.652 |
| 2020-21 | 0.598 | 0.599 | 0.619 | 0.606 | 0.620 |
| 2021-22 | 0.649 | 0.656 | 0.648 | 0.655 | 0.654 |
| 2022-23 | 0.627 | 0.632 | 0.631 | 0.633 | 0.633 |
| 2023-24 | 0.643 | 0.638 | 0.654 | 0.658 | 0.663 |
| 2024-25 | 0.650 | 0.651 | 0.661 | 0.663 | 0.669 |
| 2025-26 | 0.678 | 0.678 | 0.690 | 0.676 | 0.694 |

## Method notes

- MOV Elo: `EloModel(mov="fte")` — update multiplier ((margin+3)^0.8)/(7.5+0.006·winner_diff),
  the FiveThirtyEight NBA formulation (favourite-blowout dampening).
- Rest features are computed from each team's previous game date only (capped at
  5 days; season openers get the cap) — no schedule lookahead.
- 2012-13 is absent from the Kaggle dump (known gap); Elo carries across it.
