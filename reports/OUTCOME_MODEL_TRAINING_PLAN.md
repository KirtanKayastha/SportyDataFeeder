# Real Outcome Model — Training & Validation Plan (for review)

**Status:** PROPOSAL ONLY. No code, ingestion, or training has been performed.
**Companion to:** `reports/MODEL_VALIDATION_REPORT.md`
**Scope:** Football (EPL) outcome prediction trained on real historical data. Basketball is explicitly out of scope (no data exists).
**Date:** 2026-06-29

---

## 0. Why this plan exists

The validation report concluded the current outcome model has **no measurable skill** (in-sample accuracy 0.467 = always-predict-home), primarily because its only feature — player-form-derived team strength — is **near-constant** and it was trained on **15 circular, simulator-generated matches**.

We now have **4,940 real EPL matches over 13 seasons (2013‑14 → 2025‑26)** with results, shots, corners, fouls, cards, and bookmaker odds. This plan designs a **proper, leakage-free, walk-forward-validated** football outcome model on that data.

**Central design tension to acknowledge up front:** the historical data is **team-level** (no players), but the current pipeline is **player-level** (`event_rates` keyed by `player_id`, `team_strength` = mean player form). This plan therefore introduces a **parallel team-level feature/rating layer** rather than bending the new model into the player-centric one. The two stay decoupled by design (see §6).

---

## 1. Decisions that need your sign-off

These are the choices that change the work. My recommendation is in **bold**; please confirm or override.

| # | Decision | Options | Recommendation |
|---|---|---|---|
| D1 | **Primary model family** | (a) Multinomial logistic on Elo+form; (b) **Dixon‑Coles / bivariate Poisson on goals → derive H/D/A**; (c) Gradient boosting | **(b) Dixon‑Coles as primary** (principled for football, also yields a goals event-rate model), with (a) as an interpretable secondary/baseline |
| D2 | **Use bookmaker odds as a feature?** | (a) Yes — feature; (b) **No — use only as a baseline to beat**; (c) both, report separately | **(b)/(c): headline model excludes odds; we report "with odds" separately.** Odds are near-optimal, so including them makes the ML model just echo the bookie and hides whether *our* features have signal |
| D3 | **Where the data lives** | (a) Load CSVs directly at train time (no DB); (b) ingest into a new `historical_matches` table; (c) reuse `matches`/`events` with a `source` flag | **(a) for training/backtest** (keeps real data out of the simulated `matches` table and avoids schema churn); optionally (b) later if the API needs to serve it |
| D4 | **Integration depth now** | (a) Offline model + report only; (b) **also wire a sport-aware `outcome_v2` into `predict_outcome`**; (c) full pipeline incl. live rating updates | Start at **(a)**, then **(b)** once metrics justify it. Defer (c) |
| D5 | **Simulator recalibration** (the 6.13 vs 2.80 goals bug) | in-scope / separate workstream | **Separate workstream** — the simulator is decoupled from the outcome model; fixing goals rate is independent (tracked in the validation report's P1) |
| D6 | **Basketball** | block on data / de-scope | **De-scope until a real basketball dataset is provided** |

---

## 2. Data preparation

**Source:** `historical-datas/E0.csv … E12.csv` — all `Div=E0` (EPL), one season each, 380 matches/season.

Steps:
1. **Concatenate** all seasons; tag each row with a derived `season` label from the file/date span; sort strictly by **date+time** (causal ordering is mandatory — no shuffling, ever).
2. **Team-name normalization.** Teams enter/leave via promotion/relegation across 13 seasons. Build a canonical name map (football-data.co.uk names are already consistent, but verify e.g. "Man City"/"Man United"/"Nott'm Forest"). New teams get default ratings (Elo handles this).
3. **Clean** rows with missing `FTHG/FTAG/FTR` (one stray blank row exists in E11). Keep stat columns where present (`HS/AS/HST/AST/HC/AC/HF/AF/HY/AY/HR/AR`) and odds (`B365H/D/A`).
4. **Target:** `FTR ∈ {H, D, A}` → 3-class `{2=home, 1=draw, 0=away}` (consistent with existing label convention in `ml_models.py`).
5. **Leakage rule:** every feature for match *t* is computed **only from matches strictly before *t*** (and Elo updated only after *t* is scored).

---

## 3. Feature design (all strictly pre-match)

| Feature | Definition | Rationale |
|---|---|---|
| **Elo (home, away)** | Standard Elo, K≈20, home-field bump ≈ +60–100, updated chronologically; carry ratings across seasons with mean-reversion (e.g. regress 25% toward 1500 each new season) | Strongest single signal; gracefully handles promoted teams |
| **Elo difference** | `elo_home + home_adv − elo_away` | The model's main discriminator |
| **Rolling form (last N=5/10)** | points, goals for/against, **home/away split** (home team's home form, away team's away form), as-of-before | Captures momentum the rating smooths over |
| **Attack/defence rates** | rolling goals-scored & goals-conceded per match | Feeds both outcome and the Poisson goals model |
| **Rest days** (optional) | days since each team's previous match | Fatigue/fixture congestion |
| **(Optional, gated by D2) market features** | B365 implied probabilities (de-margined) | Benchmark / optional feature |

Note: shots/corners/fouls/cards become available **as targets** for team-level event-rate models (§7), and as *rolling* explanatory features — but not as same-match features (that would be leakage).

---

## 4. Models

- **Primary (D1b): Dixon‑Coles / bivariate Poisson.** Estimate per-team attack & defence strengths + home advantage from goals; the low-score correlation correction (Dixon‑Coles) fixes the well-known 0‑0/1‑0/0‑1/1‑1 underdispersion. Outcome probabilities (H/D/A) are derived by summing the score-line probability matrix. **Bonus:** this *is* a validated goals event-rate model, directly addressing Phase 2.
- **Secondary / interpretable baseline (D1a): multinomial logistic** on `[elo_diff, home_form, away_form, …]` — mirrors the current stack, easy to ship into `predict_outcome`.
- **Optional (D1c): gradient boosting** (e.g. `HistGradientBoostingClassifier`) — only if it beats the above on held-out log loss by a meaningful margin; lower interpretability.

All probabilistic outputs get a **calibration pass** (isotonic or Platt) fitted on a validation fold — *only if* the reliability diagram shows miscalibration.

---

## 5. Validation design (the point of the whole exercise)

**Walk-forward, expanding window** (the brief's requested approach):

```
Train seasons 1..3  → test season 4
Train seasons 1..4  → test season 5
...
Train seasons 1..12 → test season 13
```

- First ~3 seasons warm up Elo; report test results for seasons 4→13 (≈ 3,800 truly out-of-sample matches).
- **Causal only:** ratings/form computed forward in time; no future leakage.

**Metrics (per test season + pooled):**
- Accuracy, **multiclass log loss**, **Brier** — log loss is the headline (it's what bookmakers optimize).
- **Calibration:** reliability diagram + **ECE/MCE** (now meaningful at thousands of matches).
- Confusion matrix; per-class precision/recall/F1 (does it ever correctly call draws/aways? — the current model never does).

**Baselines to beat (Phase 7, for real this time):**
1. Always-predict-home (44.6% accuracy reference).
2. **Bookmaker implied probabilities** (de-margined B365) — the ceiling; if we can't beat the closing line, that's expected and fine.
3. Elo-only (no ML) — isolates whether the ML layer adds anything over the rating.

**Success criteria (proposed):** beat always-home on accuracy **and** beat Elo-only on log loss out-of-sample, with ECE < ~0.05. Matching (not beating) the bookmaker is a strong result.

---

## 6. Why the new model stays decoupled from the simulator

- The **simulator** samples per-player events from `event_rates` (player-level) and is **independent** of the outcome model today. This plan does **not** entangle them.
- The new outcome model is a **team-level forecaster** for the `/predict` path and stored predictions — it answers "who wins," not "which player scores."
- Player-level event simulation and its goal-rate recalibration (the 6.13→2.80 fix) are a **separate workstream** (D5). Keeping them separate avoids coupling two unrelated bugs.

---

## 7. Optional: team-level event-rate models (fulfills Phase 2 properly)

With real actuals for shots, corners, cards, goals, we can fit **Poisson attack/defence models per stat** and validate with **MAE / Poisson deviance on held-out seasons** — the honest Phase-2 evaluation that was impossible before. Recommended as a fast follow once the outcome model lands, since the Dixon‑Coles infrastructure already produces the goals version.

---

## 8. Proposed code changes (NOT yet implemented — for your approval)

Additive and version-gated; nothing existing is broken:

| New/changed | Purpose |
|---|---|
| `scripts/load_historical.py` | Read + normalize + concatenate the 13 CSVs into a clean DataFrame (no DB writes) |
| `app/services/team_ratings.py` | Elo store + chronological updater (causal) |
| `app/services/features_team.py` | Team-level pre-match feature builder (leakage-safe) |
| `scripts/backtest_outcome.py` | Walk-forward harness + baselines + metrics + calibration (delivers the real Phases 3/4/6/7) |
| `scripts/train_outcome_v2.py` | Fit final model on all seasons; save `models_pkl/outcome_v2.pkl` (+ Elo table) |
| `app/services/ml_models.py` (edit) | Add sport-aware `outcome_v2` path; **keep v1 + heuristic as fallback** (no breaking change) |
| `reports/OUTCOME_MODEL_RESULTS.md` | Final real-data results report |

---

## 9. Phasing & rough effort

| Phase | Work | Output | Effort |
|---|---|---|---|
| A | Data loader + name normalization + Elo | clean dataset + ratings | S |
| B | Feature builder + **walk-forward backtest harness + baselines** | first **real** out-of-sample numbers | M |
| C | Train Dixon‑Coles + logistic; pick winner by held-out log loss vs bookmaker | candidate model | M |
| D | Calibration + final results report | `OUTCOME_MODEL_RESULTS.md` | S |
| E | Wire `outcome_v2` into `predict_outcome` (version-gated) | live integration | M |
| F (opt) | Team-level event-rate Poisson models (Phase 2) | stat forecasts | M |

**Note:** Phase B alone already produces the real backtest + baseline comparison — i.e. the highest-value missing evidence — even before any new model is trained.

---

## 10. Open risks / things to confirm

- **Bookmaker ceiling:** beating closing odds is hard; the realistic win is *matching* them while gaining interpretability + coverage. Set expectations accordingly (D2).
- **Team identity** across promotion/relegation — handled by Elo defaults + name map, but worth a spot-check.
- **Football only.** Basketball remains unvalidated and unsupported until data arrives (D6).
- **Player vs team mismatch:** the new model improves `/predict`, not the player-event simulation; if stakeholders expect the *simulation* to look realistic, that's the separate goal-rate recalibration (D5).

---

**Awaiting your decisions on D1–D6 before any implementation.**
