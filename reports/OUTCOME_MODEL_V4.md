# Outcome Model v4 — Football MOV Elo + Shots-on-Target Form

**Companion to:** `reports/MODEL_IMPROVEMENT_PLAN.md` (step 3), `reports/OUTCOME_MODEL_V3.md`.
**Data:** real EPL, 13 seasons. Same expanding-window walk-forward + metrics as Phase B/C/v3.
**No live app model was modified by this script.**

## Stage 1 — MOV Elo grid search

Grid: mov ∈ {wfe, fte}, K ∈ [10.0, 15.0, 20.0, 25.0, 30.0, 40.0], SR ∈ [0.0, 0.1, 0.25, 0.4] (HA fixed at 65 — absorbed
by the logistic intercept, see v3). Objective = pooled walk-forward log loss.

**Best:** mov=fte, K=40, SR=0.10
→ LL 0.96981 (production v3 classic Elo → 0.97094).

Top 10 configs:

```
mov    k  season_regression  log_loss
fte 40.0               0.10  0.969809
wfe 20.0               0.10  0.969900
wfe 15.0               0.10  0.970221
wfe 25.0               0.10  0.970364
wfe 25.0               0.00  0.970401
wfe 20.0               0.00  0.970417
fte 40.0               0.00  0.970522
fte 30.0               0.10  0.970552
wfe 30.0               0.00  0.970966
wfe 15.0               0.25  0.971171
```

## Stage 2 — candidate comparison (pooled out-of-sample)

| Model | n | Accuracy | Log loss | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| elo_v3 | 3800 | 0.544 | 0.971 | 0.576 | 0.013 |
| elo_mov | 3800 | 0.541 | 0.970 | 0.575 | 0.018 |
| elo_shots_net | 3800 | 0.542 | 0.967 | 0.573 | 0.017 |
| elo_shots_full | 3800 | 0.543 | 0.968 | 0.574 | 0.016 |
| elo_mov_shots | 3800 | 0.543 | 0.966 | 0.573 | 0.015 |
| bookmaker | 3800 | 0.556 | 0.953 | 0.564 | 0.021 |

- **elo_v3** — the shipped production config (K=20, HA=65, SR=0.10, classic Elo).
- **elo_mov** — stage-1 best margin-of-victory Elo (wfe = World Football Elo
  goal-difference multiplier; fte = FiveThirtyEight favourite-dampened multiplier).
- **elo_shots_net / elo_shots_full** — v3 Elo + causal rolling shots-on-target form
  (`app/services/features_team.py:annotate_shot_form`, window 10, league prior 4.3).
- **elo_mov_shots** — best MOV Elo + compact SoT feature.
- **bookmaker** — de-margined Bet365 odds: the practical ceiling.

**Winner:** `elo_mov_shots` (pooled OOS log loss 0.9661); incumbent v3
0.9709; bookmaker 0.9527. Gap to ceiling: **+0.0134**
(v3 was +0.0182).

## Per-season out-of-sample accuracy

| Season | elo_v3 | elo_mov | elo_shots_net | elo_shots_full | elo_mov_shots | bookmaker |
|---|---:|---:|---:|---:|---:|---:|
| 2016-17 | 0.579 | 0.584 | 0.587 | 0.603 | 0.589 | 0.611 |
| 2017-18 | 0.547 | 0.553 | 0.542 | 0.547 | 0.537 | 0.553 |
| 2018-19 | 0.576 | 0.576 | 0.579 | 0.582 | 0.576 | 0.587 |
| 2019-20 | 0.537 | 0.537 | 0.521 | 0.518 | 0.532 | 0.529 |
| 2020-21 | 0.500 | 0.503 | 0.500 | 0.500 | 0.505 | 0.518 |
| 2021-22 | 0.550 | 0.542 | 0.561 | 0.547 | 0.558 | 0.582 |
| 2022-23 | 0.553 | 0.545 | 0.545 | 0.545 | 0.550 | 0.558 |
| 2023-24 | 0.579 | 0.571 | 0.582 | 0.589 | 0.579 | 0.597 |
| 2024-25 | 0.539 | 0.521 | 0.534 | 0.529 | 0.534 | 0.539 |
| 2025-26 | 0.479 | 0.479 | 0.474 | 0.466 | 0.471 | 0.489 |

## Method notes

- MOV variants live in `app/services/team_ratings.py:EloModel._mov_multiplier`.
- SoT form is any-venue, last-10-matches mean for/against, updated strictly after
  each row is emitted; cold starts get the league prior — no leakage.
