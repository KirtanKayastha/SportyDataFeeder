# Outcome Model — Real Walk-Forward Backtest Results (Phase B)

**Companion to:** `reports/MODEL_VALIDATION_REPORT.md`, `reports/OUTCOME_MODEL_TRAINING_PLAN.md`
**Data:** 4940 real EPL matches, 13 seasons. **Walk-forward, expanding window**, strictly causal.
**Tested out-of-sample on 10 seasons** (≈ 3800 matches); first 3 seasons warm up Elo only.
**No application model was trained or modified.** These are baselines establishing the real reference.

## Pooled out-of-sample metrics

| Baseline | n | Accuracy | Log loss | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| always_home | 3800 | 0.446 | 19.124 | 1.107 | 0.554 |
| base_rate | 3800 | 0.446 | 1.065 | 0.644 | 0.006 |
| elo_logistic | 3800 | 0.539 | 0.972 | 0.577 | 0.019 |
| bookmaker | 3800 | 0.556 | 0.953 | 0.564 | 0.021 |

Lower log loss / Brier / ECE = better; higher accuracy = better.

## How to read this

- **always_home** — predict Home every match. Accuracy 0.446. The bar the current production model *ties* (and never beats). Its huge log loss (19.1) is expected: a hard 0/1 prediction is punished severely whenever the result isn't Home — a reminder that *probabilities*, not hard calls, are what matter here.
- **base_rate** — training H/D/A frequencies. A model with no skill should not beat this on log loss.
- **elo_logistic** — multinomial logistic on causal Elo difference (one feature). Accuracy 0.539, log loss 0.972.
- **bookmaker** — de-margined Bet365 odds: the practical ceiling. Log loss 0.953.

**Headline:** a single causal Elo feature already beats always-home on accuracy and beats the base-rate on log loss, out-of-sample, over ~3800 matches — versus the current model's *zero* lift on 15 in-sample matches. The gap between elo_logistic and bookmaker (0.972 vs 0.953 log loss) is the headroom the Phase C model (Dixon-Coles + richer features) aims to close.

## Per-season out-of-sample accuracy

| Season | always_home | base_rate | elo_logistic | bookmaker |
|---|---:|---:|---:|---:|
| 2016-17 | 0.492 | 0.492 | 0.574 | 0.611 |
| 2017-18 | 0.455 | 0.455 | 0.550 | 0.553 |
| 2018-19 | 0.476 | 0.476 | 0.563 | 0.587 |
| 2019-20 | 0.453 | 0.453 | 0.537 | 0.529 |
| 2020-21 | 0.379 | 0.379 | 0.495 | 0.518 |
| 2021-22 | 0.429 | 0.429 | 0.534 | 0.582 |
| 2022-23 | 0.484 | 0.484 | 0.550 | 0.558 |
| 2023-24 | 0.461 | 0.461 | 0.566 | 0.597 |
| 2024-25 | 0.408 | 0.408 | 0.539 | 0.539 |
| 2025-26 | 0.426 | 0.426 | 0.487 | 0.489 |

## Method notes (reproducibility)

- Loader: `scripts/load_historical.py` (read-only, no DB).
- Causal Elo: `app/services/team_ratings.py` (K=20, home advantage=65, 25% season mean-reversion). Pre-match ratings only.
- Expanding window: train = all seasons before the test season; Elo updated forward in time; logistic mapping refit per fold on past data only.
- Metrics: accuracy, multiclass log loss, multiclass Brier, confidence-ECE (10 bins).
- Bookmaker metrics computed only over rows with valid Bet365 odds (n shown).
