# Outcome Model v5 — Multi-League Pooling (EPL + Championship)

**Companion to:** `reports/MODEL_IMPROVEMENT_PLAN.md` (step 5), `reports/OUTCOME_MODEL_V4.md`.
**Data:** 4940 EPL + 7176 Championship matches, 13 seasons
(`scripts/fetch_championship.py`). Test folds are EPL-only — pooling changes
only what the Elo/form/mapping layers learn from. Walk-forward, strictly causal.
**No live app model was modified by this script.**

## Stage 1 — pooled-Elo mini grid

Best pooled config: K=40, SR=0.05
(mov=fte, HA=65) → LL 0.97174.

## Stage 2 — candidate comparison (pooled out-of-sample, EPL test matches)

| Model | n | Accuracy | Log loss | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| v4_incumbent | 3800 | 0.543 | 0.966 | 0.573 | 0.015 |
| pooled_elo | 3800 | 0.538 | 0.972 | 0.577 | 0.011 |
| pooled_all | 3800 | 0.540 | 0.971 | 0.576 | 0.013 |
| bookmaker | 3800 | 0.556 | 0.953 | 0.564 | 0.021 |

- **v4_incumbent** — production `outcome_v4_elo_sot` (E0-only Elo + SoT form).
- **pooled_elo** — Elo/SoT computed over E0+E1 (promoted teams arrive with an
  earned rating; relegated teams keep theirs); logistic fit on prior E0 rows.
- **pooled_all** — as above, logistic fit on all prior rows (both divisions)
  with a top-flight dummy.
- **bookmaker** — de-margined Bet365 odds on the same EPL rows.

**Winner:** `v4_incumbent` (pooled OOS log loss 0.9661); v4 incumbent
0.9661; bookmaker 0.9527. Gap to ceiling: **+0.0134**
(v4 was +0.0134).

## Verdict: pooling is a documented NEGATIVE result — v4 stays in production

Naive pooling *hurts* (0.972 vs 0.966): with no cross-division matches to anchor
the scales, a promoted team's Championship-earned rating (~1550–1650 from
dominating E1) overstates its EPL strength, and Championship shot volumes
contaminate SoT form the same way. Follow-up experiments (division-crossing
rating offset applied at promotion/relegation):

| Variant | OOS log loss |
|---|---:|
| pooled, offset 0 | 0.9731 |
| pooled, offset 100 | 0.9695 |
| pooled, offset 200 | 0.9680 |
| pooled, offset **250** (optimum) | 0.9679 |
| pooled, offset 400 | 0.9694 |
| pooled offset 250 + **E0-only SoT form** | 0.9670 |
| **v4 incumbent (E0 only)** | **0.9661** |

The offset recovers most of the damage and isolating SoT to E0 recovers more,
but even the cleanest pooled variant stays behind the E0-only incumbent. The
lesson: the 1500 cold start + season regression is already a decent implicit
prior for promoted sides; second-tier results add more scale-mismatch noise
than signal at this data size. Championship data stays in the repo
(`historical-datas/championship/`, `scripts/fetch_championship.py`) for future
revisits (e.g. explicit inter-league strength model).

## Per-season out-of-sample accuracy (EPL)

| Season | v4_incumbent | pooled_elo | pooled_all | bookmaker |
|---|---:|---:|---:|---:|
| 2016-17 | 0.589 | 0.600 | 0.597 | 0.611 |
| 2017-18 | 0.537 | 0.532 | 0.534 | 0.553 |
| 2018-19 | 0.576 | 0.579 | 0.582 | 0.587 |
| 2019-20 | 0.532 | 0.524 | 0.526 | 0.529 |
| 2020-21 | 0.505 | 0.489 | 0.492 | 0.518 |
| 2021-22 | 0.558 | 0.550 | 0.547 | 0.582 |
| 2022-23 | 0.550 | 0.561 | 0.561 | 0.558 |
| 2023-24 | 0.579 | 0.555 | 0.566 | 0.597 |
| 2024-25 | 0.534 | 0.524 | 0.521 | 0.539 |
| 2025-26 | 0.471 | 0.468 | 0.471 | 0.489 |
