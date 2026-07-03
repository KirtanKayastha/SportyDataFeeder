# Outcome Model v3 — Tuned Elo + Draw Feature + Causal Ensemble

**Companion to:** `reports/MODEL_IMPROVEMENT_PLAN.md` (step 1), `reports/OUTCOME_MODEL_PHASE_C.md`.
**Data:** real EPL, 13 seasons. Same expanding-window walk-forward + metrics as Phase B/C.
**No live app model was modified by this script.** Integration happens in
`scripts/finalize_outcome_v2.py` once the winner is confirmed.

## Stage 1 — Elo hyperparameter grid search

Grid: K ∈ [10.0, 15.0, 20.0, 25.0, 30.0, 40.0], HA ∈ [40.0, 55.0, 65.0, 80.0, 95.0], SR ∈ [0.0, 0.1, 0.25, 0.4]; objective = pooled walk-forward
log loss of the 1-feature elo_logistic model.

**Best:** K=20, home_advantage=65,
season_regression=0.10 → LL 0.97094
(incumbent K=20/HA=65/SR=0.25 → LL 0.97245).

Top 10 configs:

```
   k  home_advantage  season_regression  log_loss
20.0            65.0                0.1  0.970937
20.0            55.0                0.1  0.970938
20.0            80.0                0.1  0.970941
20.0            40.0                0.1  0.970943
20.0            95.0                0.1  0.970953
25.0            95.0                0.1  0.971056
25.0            80.0                0.1  0.971099
25.0            65.0                0.1  0.971141
25.0            55.0                0.1  0.971169
25.0            40.0                0.1  0.971207
```

## Stage 2 — candidate comparison (pooled out-of-sample)

| Model | n | Accuracy | Log loss | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| elo_incumbent | 3800 | 0.539 | 0.972 | 0.577 | 0.019 |
| elo_tuned | 3800 | 0.544 | 0.971 | 0.576 | 0.013 |
| elo_absdiff | 3800 | 0.544 | 0.971 | 0.576 | 0.018 |
| logistic_form | 3800 | 0.546 | 0.973 | 0.577 | 0.014 |
| dixon_coles | 3800 | 0.531 | 0.985 | 0.586 | 0.017 |
| blend_ed | 3800 | 0.543 | 0.972 | 0.577 | 0.012 |
| blend_edf | 3800 | 0.544 | 0.971 | 0.576 | 0.013 |
| bookmaker | 3800 | 0.556 | 0.953 | 0.564 | 0.021 |

- **elo_incumbent** — the production outcome_v2 (K=20, HA=65, SR=0.25).
- **elo_tuned** — stage-1 params, same single elo_diff feature.
- **elo_absdiff** — tuned params + |elo_diff| (lets draw probability peak for even teams).
- **logistic_form / dixon_coles** — Phase C candidates (form model on tuned Elo).
- **blend_ed / blend_edf** — causal stacked blends; fold-s weights fit only on
  out-of-sample predictions from folds before s (equal weights for the first fold).
- **bookmaker** — de-margined Bet365 odds: the practical ceiling.

**Winner:** `elo_tuned` (pooled OOS log loss 0.9709); incumbent
0.9725; bookmaker ceiling 0.9527. Gap to ceiling:
**+0.0182** (was +0.0197).

Final-fold blend weights: {blend_ed: [0.774, 0.226], blend_edf: [0.642, 0.212, 0.146]}

## Per-season out-of-sample accuracy

| Season | elo_incumbent | elo_tuned | elo_absdiff | logistic_form | dixon_coles | blend_ed | blend_edf | bookmaker |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2016-17 | 0.574 | 0.579 | 0.576 | 0.587 | 0.553 | 0.582 | 0.587 | 0.611 |
| 2017-18 | 0.550 | 0.547 | 0.547 | 0.550 | 0.529 | 0.547 | 0.547 | 0.553 |
| 2018-19 | 0.563 | 0.576 | 0.579 | 0.597 | 0.582 | 0.584 | 0.589 | 0.587 |
| 2019-20 | 0.537 | 0.537 | 0.537 | 0.529 | 0.521 | 0.534 | 0.534 | 0.529 |
| 2020-21 | 0.495 | 0.500 | 0.505 | 0.508 | 0.526 | 0.508 | 0.508 | 0.518 |
| 2021-22 | 0.534 | 0.550 | 0.547 | 0.558 | 0.547 | 0.550 | 0.550 | 0.582 |
| 2022-23 | 0.550 | 0.553 | 0.553 | 0.547 | 0.508 | 0.542 | 0.542 | 0.558 |
| 2023-24 | 0.566 | 0.579 | 0.576 | 0.571 | 0.561 | 0.571 | 0.571 | 0.597 |
| 2024-25 | 0.539 | 0.539 | 0.539 | 0.532 | 0.511 | 0.532 | 0.529 | 0.539 |
| 2025-26 | 0.487 | 0.479 | 0.476 | 0.482 | 0.474 | 0.479 | 0.479 | 0.489 |

## Method notes

- Elo `season_regression` is now a tunable parameter of
  `app/services/team_ratings.py` (default unchanged at 0.25).
- Causal throughout: Elo/form features use only earlier matches; every model and
  every blend weight is fit strictly on data before the test season.
