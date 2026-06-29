# Model Validation Report — Sporty Data Feeder

**Prepared for:** Engineering leads / stakeholders (pre-production gate)
**Scope:** Event-Rate model + Outcome model, Football & Basketball
**Date:** 2026-06-29
**Method:** Read-only evaluation. No models were retrained or modified (per constraint).
**Evaluation data:** The live database used by the application (Neon Postgres).

---

## ⚠️ Executive Summary — read this first

This is an **honest, evidence-based** evaluation. The headline conclusion is uncomfortable but important:

> **Neither model is production-ready, and — more fundamentally — the data required to validate them in a scientifically meaningful way does not exist in the system yet.** What can be measured shows the outcome model currently performs **no better than an "always predict home win" baseline**, and the event-rate "model" is **not a machine-learning model at all** — it is an empirical rate lookup table.

### The five facts that dominate every other finding

| # | Finding | Evidence |
|---|---------|----------|
| 1 | **There is no real match-result ground truth anywhere in the system.** The CSV importer only writes players + per-player stats; it creates **zero** matches and **zero** events. Every match/event in the DB was produced by the simulator itself. | `grep` of `importer.py`: 0 `Match(`/`Event(` constructions. |
| 2 | **Validating the outcome model on these matches is circular.** The model is trained on simulator-generated outcomes, then "tested" on simulator-generated outcomes. This measures self-consistency, not real-world accuracy. | Matches 1–15 all carry only simulator-written events. |
| 3 | **Basketball has never been run.** 0 finished matches, 0 events, 0 ratings. There is no basketball outcome model and no basketball simulation output to validate. | Inventory: basketball = 30 scheduled, 0 finished, 0 events. |
| 4 | **The outcome model has no measurable skill.** In-sample accuracy = **0.467**, identical to the majority-class baseline (always predict home). It never correctly predicts a draw or an away win. | Confusion matrix below; baseline = 0.467. |
| 5 | **n = 15, in-sample, single sport.** This is far below the threshold for any statistically meaningful conclusion. Every metric in Phases 2–10 must be read as *directional only*. | 15 finished matches, 0 held-out. |

### Model scorecard (1 = unusable, 10 = excellent)

| Dimension | Outcome Model | Event-Rate "Model" |
|---|:---:|:---:|
| Accuracy | **2** | **4** |
| Reliability | **2** | **4** |
| Calibration | **3** | **3** |
| Generalization | **1** | **4** |
| Interpretability | **8** | **9** |
| Maintainability | **6** | **7** |
| **Production Readiness** | **2** | **3** |

Interpretability and maintainability are genuine strengths (the code is clean, transparent, and well-factored). Everything related to *predictive validity* is currently unproven or actively poor.

---

## Phase 1 — System Analysis

### 1.1 What the two "models" actually are

**Model 1 — "Event-Rate Prediction Model" (`models_pkl/event_rates.pkl`)**

This is **not a trained ML model**. It is a Python `dict[player_id -> {event_type: per-minute probability}]` produced by `scripts/train_models.py:build_event_rates`, which simply reads each player's historical `player_stats` rows and divides totals by minutes (`app/services/features.py:compute_player_features`).

- **Algorithm:** arithmetic rate = `total_event / total_minutes`. No learning, no parameters, no loss function.
- **Football event types modeled:** `goal`, `assist`, `yellow_card`, `red_card` — **4 types only**.
- **Basketball event types modeled:** `point_2`, `point_3`, `free_throw`, `assist`, `rebound` — and the 2pt/3pt/FT split is **not data-driven**; total points are decomposed by a **hard-coded assumed mix of 60 % / 25 % / 15 %** (`BASKETBALL_POINT_MIX`).
- **Cold start:** players with no usable stats get hard-coded league-average rates (`FOOTBALL_FALLBACK_RATES` / `BASKETBALL_FALLBACK_RATES`).

**Model 2 — Outcome Prediction Model (`models_pkl/outcome_model.pkl`)**

- **Algorithm:** `sklearn.Pipeline([MinMaxScaler, LogisticRegression(solver="lbfgs", max_iter=500)])`. Multinomial, 3 classes (0 = away, 1 = draw, 2 = home).
- **Input features (3):** `[home_strength, away_strength, 1.0]`. The third feature is a **constant** (`HOME_BIAS = 1.0`); because it has zero variance, MinMaxScaler maps it to a dead column and its coefficients are forced to 0. **Effectively a 2-feature model.**
- **Feature definition:** `home_strength` / `away_strength` = `compute_team_strength` = mean of the team's players' `form_index`, `/15`, clamped to `[0, 1]`. `form_index` is an EWMA (α = 0.4) of recent per-match rating points (football) or points-per-36 (basketball).
- **Target:** match result label derived by **replaying the match's events** through `scoring_rules.score_events` and comparing scores.
- **Single model for all sports.** There is no sport-specific outcome model; the one pickle was trained only on football matches and would be applied to basketball unchanged.

### 1.2 Pipelines

- **Training** (`scripts/train_models.py`): builds the rate table for every player with stats; collects `(home_strength, away_strength, 1.0)` features + replay-derived labels for every `status='finished'` match; trains the logistic pipeline. **< 5 finished matches → skip; ≥ 20 → stratified 80/20 split; otherwise (the current case) trains on all rows with no held-out test set.**
- **Validation pipeline:** **none exists in code.** With 15 matches the script trains on all data and prints no held-out metrics. There is no cross-validation, no calibration step, no backtest.
- **Hyperparameters:** all sklearn defaults except `max_iter=500`. No tuning, no regularization search.
- **Prediction** (`ml_models.predict_outcome`): scales the 3 features, returns class probabilities mapped via `model.classes_`. Falls back to a hand-tuned heuristic if the pickle is missing.
- **Simulation** (`app/services/simulation.py`): per minute, Bernoulli-samples each lineup player's per-event-type probability from the rate table (league-average fallback for cold players), writes events, derives score via `scoring_rules`. This is a **Poisson/Bernoulli event generator**, independent of the outcome model. The outcome model does **not** drive the simulation; the two are decoupled.

---

## Phase 2 — Event-Rate Model Validation

**Feasibility: NOT POSSIBLE as specified. Reason stated explicitly below.**

The requested regression metrics (MAE, RMSE, MAPE, R², Poisson log-likelihood, residuals, predicted-vs-actual) require a dataset of **predicted rate vs. actually observed rate on held-out matches**. That dataset does not exist:

1. **The events used as "actuals" are generated *from* the rates being evaluated.** Measuring error between a Bernoulli draw and its own parameter is circular and only recovers sampling noise, not model error.
2. **Most requested statistics are not modeled or stored at all.** The system has no concept of: expected goals, shots, shots on target, corners, fouls, offsides, possession, saves (football); steals, blocks, turnovers, personal fouls, field goals, ORB/DRB split, quarter scores (basketball). Football tracks 4 event types; basketball tracks 5. **`stl`/`blk` exist in `player_stats` but are never converted into event rates or simulated.**
3. **The 2pt/3pt/FT decomposition is an assumption, not a prediction** (fixed 60/25/15), so it cannot be scored.

**What can be said:** the football rate table is a faithful empirical summary of real EPL per-player stats (1,043 stat rows, full coverage of goals/assists/cards/minutes), and the basketball table of real NBA stats (582 rows, full coverage of pts/ast/reb/stl/blk). As *descriptive* rates they are reasonable; as *predictive* models they are unvalidated. Note **105 / 1,152 players have all-zero rates** and will never generate an event.

> **Recommendation:** to validate event rates honestly, hold out real per-gameweek stats (e.g., train rates on GW 1–25, predict observed counts in GW 26–38) and compute Poisson deviance / MAE per stat. This is feasible **today** with the existing `player_stats` data and is the single highest-value missing evaluation.

---

## Phase 3 — Outcome Model Validation

**Football — in-sample, n = 15 (the only data that exists). Basketball — NOT POSSIBLE (0 matches).**

| Metric | Value | Plain-English meaning | Reference |
|---|---|---|---|
| Accuracy | **0.467** | Right 7 of 15 times. | Majority baseline = **0.467** (no lift) |
| Log loss | **0.994** | Confidence-weighted error. | Uniform 1/3 guess = 1.099 |
| Brier (multiclass) | **0.596** | Mean squared probability error. | Uniform guess = 0.667 |
| ROC AUC / PR AUC | *not reported* | Uninformative & unstable at n=15 with empty predicted classes. | — |

**Confusion matrix** (rows = true, cols = predicted; order away/draw/home):

```
            pred away   pred draw   pred home
true away       0           0           5
true draw       1           0           2
true home       0           0           7
```

**Per-class:** away → precision/recall = 0.00 / 0.00; draw → 0.00 / 0.00; home → 0.50 / 1.00.

**Reading:** the model predicts **home for 13/15 matches and away for 2**, and **never predicts a draw**. It correctly identifies *zero* away wins and *zero* draws. Its only "skill" is reproducing the base rate. Log loss and Brier are barely better than a uniform random guess — **and this is the optimistic in-sample number.** Out-of-sample performance would almost certainly be worse.

**Why:** the `home_strength`/`away_strength` features are **nearly constant** across football teams (range 0.433–0.465; see Phase 9). With essentially no signal, logistic regression correctly learns that the best it can do is predict the most common class.

**Basketball:** no finished matches exist, so accuracy/precision/recall/AUC/log-loss/Brier are all **undefined**. Additionally, the 3-class model includes a `draw` class that is essentially impossible in basketball, and the only trained pickle was fit on football — there is effectively **no basketball outcome model.**

---

## Phase 4 — Calibration

**Feasibility: aggregate only; per-bin reliability uninformative at n=15.**

- **Aggregate:** mean predicted home-win probability = **0.467**, actual home-win rate = **0.467**. Perfect *on average* — but only because the model never strays from the base rate. This is "calibrated by refusing to make a confident prediction," not genuine calibration.
- **Predicted home-win probabilities** ranged only **0.384 → 0.613** (spread 0.23). The model is structurally incapable of saying "this team will very likely win."
- **ECE / MCE / reliability diagram:** with 15 points, any binning yields 1–3 samples per bin and noise-dominated estimates. **Reporting an ECE here would be misleading.**

> Calibration methods (Platt/isotonic) are **not recommended yet** — there is nothing to calibrate until the model produces a usable probability spread on real, held-out data.

---

## Phase 5 — Simulation Validation

This validates the **simulation engine**, which is distinct from the prediction models.

**Football (15 simulated matches) vs real EPL reference:**

| Metric | Simulated | Real EPL (ref) | Verdict |
|---|---|---|---|
| Avg goals / match (total) | **6.13** | ~2.7 | ❌ **2.3× too high** |
| Avg goals / team | 3.07 | ~1.35 | ❌ too high |
| Home win % | 47 % | ~45 % | ✅ plausible |
| Draw % | **20 %** | ~26 % | ⚠️ a bit low |
| Away win % | 33 % | ~29 % | ✅ plausible |
| Events / match | 16.7 (min 10, max 21) | — | low absolute volume |

**Unrealistic behaviour identified:** goal scoring is more than double real life. Likely cause: per-minute goal probabilities are applied to 11 attackers every minute without accounting for the fact that real per-90 scoring is dominated by a few players and suppressed by defending/keeping. Draw rate is depressed as a direct consequence of high-variance high-scoring games.

**Other modeled stats** (shots, corners, cards beyond yellow/red counts, possession) are **not simulated**, so the Phase-5 football checklist cannot be fully completed.

**Basketball:** **cannot be evaluated — the simulator has never produced a basketball match.** All basketball checklist items (points, rebounds, assists, turnovers, fouls, steals, blocks, home win %) are unmeasurable. Worse, the basketball `compute_team_strength` is **broken**: points-per-36 form values divided by 15 saturate the clamp — 30 teams collapse to mean strength **0.989** with only 11 distinct values, several pinned at exactly 1.0.

---

## Phase 6 — Backtesting (walk-forward)

**Feasibility: NOT POSSIBLE.**

Walk-forward / season-based backtesting requires multiple seasons of **real** match results to train on early windows and test on later ones. The system has:

- **No real match results** (only simulated matches).
- **Effectively a single pseudo-season** of 15 matches with no temporal ordering tied to real fixtures.

No walk-forward validation can be performed. This is the cleanest example of the data limitation called out in the brief: **conclusions require a much larger set of unseen, real historical matches.**

---

## Phase 7 — Baseline Comparison

| Approach | Accuracy | Verdict |
|---|---|---|
| **Outcome model (logistic)** | **0.467** | — |
| Always predict Home Win | **0.467** | **Tie — model adds zero lift** |
| Uniform random (1/3 each) | ~0.33 (exp.) | model > random, but only via base-rate matching |

The machine-learning model provides **no statistically significant improvement** over the trivial "always predict home" rule. (With n=15 a McNemar test is not meaningfully powered, but the point estimates are *identical*, so there is nothing to test.) Elo / xG / recent-form baselines could not be computed because they, too, require real historical results.

---

## Phase 8 — Error Analysis

- **Away wins:** 0/5 correctly predicted — total blind spot.
- **Draws:** 0/3 correctly predicted — the model never outputs a draw.
- **Home wins:** 7/7 "correct" — but only because home is the default prediction.
- **Strong vs weak teams:** indistinguishable — the strength feature has almost no variance (Phase 9), so the model cannot separate favourites from underdogs.
- **Blowouts vs close games / upsets:** not separable for the same reason.

**Systematic weakness:** the model is a disguised base-rate predictor. Its errors are not random — they are 100 % concentrated in every non-home outcome.

---

## Phase 9 — Feature Importance

With a 2-effective-feature logistic model, coefficients *are* the importance:

| Feature | Home-class coef | Variance in data | Usefulness |
|---|---|---|---|
| `home_strength` | **+0.45** | **near-zero** (range 0.433–0.465) | directionally correct, practically inert |
| `away_strength` | **−0.47** | near-zero (range 0.433–0.465) | directionally correct, practically inert |
| `HOME_BIAS = 1.0` | 0.0 (forced) | **zero** | **dead feature — remove** |

- **Most influential:** away_strength / home_strength (signs are sensible: stronger home ↑ home win, stronger away ↓).
- **Least useful / redundant:** the constant `HOME_BIAS` feature is dead weight (its effect is already in the intercept).
- **Feature leakage:** none in the classic sense, **but** labels are derived from the same simulated events whose rates feed team strength — a soft circularity.
- **SHAP / permutation importance:** not informative on 2 near-constant features and 15 rows; omitted deliberately rather than reported misleadingly.

---

## Phase 10 — Overfitting / Bias-Variance

Train/validation/test comparison is **impossible** — the model was trained on all 15 rows with no held-out set.

Structurally, this is **not** classic overfitting; it is **high bias / underfitting**:

- 3 parameters over 15 points *could* overfit, but the features carry almost no signal, so the model degenerates to predicting the base rate (low variance, high bias).
- **Generalization is unproven and expected to be poor** — there is no unseen data, and the only feature is near-constant.
- **Soft data leakage** exists via the simulator→label loop (Phase 9).

---

## Phase 11 — Performance Report (consolidated)

**Architecture review:** clean, well-separated, fully interpretable. The engineering is good; the *modeling* and *evaluation* are not.

**Strengths**
- Transparent, debuggable, single-source-of-truth code (`scoring_rules`, `features`, `sport_resolver`).
- Event rates are grounded in real per-player stats.
- Logistic outcome model is fully interpretable.
- Cold-start fallbacks prevent runtime failures.

**Weaknesses**
- No real match-result ground truth → no honest validation possible.
- Outcome model = base-rate predictor, zero lift, never predicts away/draw.
- Team-strength feature has near-zero variance (football) and is saturated/broken (basketball).
- Simulator scores ~2.3× too many goals.
- Basketball is entirely unexercised; a `draw` class is meaningless there.
- One outcome model serves all sports.
- Dead constant feature; no held-out eval, no calibration, no backtest, no model registry/metrics.

**Production readiness: NOT READY.** Do not deploy the outcome model as a predictive component. The event-rate generator may be used to *drive a demo simulation* if (a) goal rates are recalibrated and (b) stakeholders understand it is a stochastic generator, not a validated forecaster.

---

## Phase 12 — Recommendations (prioritized)

Ordered by **impact ÷ difficulty**. Do **not** treat any of these as approved work yet — they are the recommended evaluation/remediation backlog.

| # | Recommendation | Impact | Difficulty | Expected gain |
|---|---|---|---|---|
| **P0** | **Get real match-result data** (ingest historical fixtures + final scores per season). Nothing below is meaningful without it. | 🔴 Critical | Med | Unblocks all validation |
| **P0** | **Hold-out event-rate validation** using existing `player_stats`: train rates on early gameweeks, score Poisson MAE/deviance on later ones. Feasible today. | 🔴 High | Low | First real accuracy number |
| **P1** | **Fix the simulator goal rate** (calibrate per-minute probabilities so total goals ≈ 2.7/match); re-check draw %. | 🔴 High | Low | Realistic simulation |
| **P1** | **Fix basketball `compute_team_strength` normalization** (it saturates to ~1.0). Use a sport-appropriate divisor or z-score. | 🟠 High | Low | Usable BB features |
| **P1** | **Replace near-constant strength feature** with discriminative features (recent goal diff, Elo, home/away splits). | 🔴 High | Med | The model's only path to real skill |
| **P2** | **Train separate, sport-specific outcome models;** drop the `draw` class for basketball; remove the dead `HOME_BIAS` feature. | 🟠 Med | Low | Correctness |
| **P2** | **Add a real validation pipeline:** walk-forward CV, calibration (only once probabilities spread), held-out metrics logged per training run. | 🟠 Med | Med | Trustworthy metrics |
| **P3** | **Actually simulate basketball** end-to-end before claiming multi-sport support. | 🟠 Med | Low | Coverage |
| **P3** | **Model the additional stats** stakeholders expect (shots, corners, fouls; steals/blocks/turnovers) — or explicitly de-scope them. | 🟡 Med | High | Feature completeness |

---

## Appendix — Reproducibility

- Inventory/metrics script: `scratchpad/evaluate.py` (read-only; no DB writes).
- Source of truth inspected: `app/services/{features,ml_models,scoring_rules,rater,simulation}.py`, `scripts/train_models.py`, `app/database.py`, `app/services/importer.py`.
- Artifacts inspected: `models_pkl/outcome_model.pkl`, `models_pkl/event_rates.pkl`.
- **Hard limitation:** every quantitative result derives from **15 in-sample, simulator-generated football matches**. They are **directional only** and must be re-run on a large set of unseen, real historical matches before any production decision.
