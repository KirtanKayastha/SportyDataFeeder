# Outcome Model — Phase C Candidate Results

**Companion to:** `reports/OUTCOME_MODEL_RESULTS.md` (Phase B), `reports/OUTCOME_MODEL_TRAINING_PLAN.md`
**Data:** real EPL, 13 seasons. Same expanding-window walk-forward + metrics as Phase B (≈ 3800 OOS matches).
**No live app model was modified.** Winner saved as a candidate bundle `models_pkl/outcome_v2.pkl` (integration is Phase E).

## Candidate comparison (pooled out-of-sample)

| Model | n | Accuracy | Log loss | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| elo_logistic | 3800 | 0.539 | 0.972 | 0.577 | 0.019 |
| logistic_form | 3800 | 0.543 | 0.974 | 0.578 | 0.020 |
| dixon_coles | 3800 | 0.531 | 0.985 | 0.586 | 0.017 |
| bookmaker | 3800 | 0.556 | 0.953 | 0.564 | 0.021 |

- **elo_logistic** — Phase B benchmark (1 feature).
- **logistic_form** — Elo diff + causal rolling home/away form & goals (7 features).
- **dixon_coles** — time-decayed bivariate-Poisson goal model (half-life 180d, L2=0.01).
- **bookmaker** — de-margined Bet365 odds: the practical ceiling.

**Winner:** `elo_logistic` (pooled OOS log loss 0.972). Gap to the bookmaker ceiling: **+0.020** log loss.

## Per-season out-of-sample accuracy

| Season | elo_logistic | logistic_form | dixon_coles | bookmaker |
|---|---:|---:|---:|---:|
| 2016-17 | 0.574 | 0.582 | 0.553 | 0.611 |
| 2017-18 | 0.550 | 0.553 | 0.529 | 0.553 |
| 2018-19 | 0.563 | 0.582 | 0.582 | 0.587 |
| 2019-20 | 0.537 | 0.526 | 0.521 | 0.529 |
| 2020-21 | 0.495 | 0.508 | 0.526 | 0.518 |
| 2021-22 | 0.534 | 0.547 | 0.547 | 0.582 |
| 2022-23 | 0.550 | 0.550 | 0.508 | 0.558 |
| 2023-24 | 0.566 | 0.568 | 0.561 | 0.597 |
| 2024-25 | 0.539 | 0.529 | 0.511 | 0.539 |
| 2025-26 | 0.487 | 0.484 | 0.474 | 0.489 |

## Method notes

- New, reusable modules: `app/services/dixon_coles.py`, `app/services/features_team.py`, `app/services/team_ratings.py`.
- Causal throughout: Elo + rolling form use only prior matches; Dixon-Coles refit per fold on past matches with recency weighting.
- Winner refit on all 13 seasons and pickled as a candidate; it is NOT loaded by the API (Phase E would version it behind `outcome_v2`).
