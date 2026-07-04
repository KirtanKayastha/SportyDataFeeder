# Machine Learning & Statistical Models

See `GLOSSARY.md` for every abbreviation. This document covers every model that exists in
the codebase, including research-only candidates that were evaluated but never shipped
(each one says so explicitly). "Shipped" means the live `/predict` or `/simulate`
endpoints actually load and use it; "research" means it lives only in `scripts/` and
`reports/`.

## 0. Model inventory at a glance

| Model | Sport(s) | Status | File(s) that build it | File(s) that serve it |
|---|---|---|---|---|
| Elo rating engine (+ margin-of-victory variants) | Football & basketball | **Shipped** (core of `outcome_v2`) | `app/services/team_ratings.py`, `scripts/finalize_outcome_v2.py`, `scripts/finalize_outcome_basketball.py` | `app/services/ml_models.py:predict_outcome_v2` |
| Elo-diff (+ SoT form) multinomial logistic regression | Football & basketball | **Shipped** (`outcome_v2` / `outcome_v2_basketball` bundles) | `scripts/finalize_outcome_v2.py`, `scripts/finalize_outcome_basketball.py` | `app/services/ml_models.py:predict_outcome_v2` |
| Team-strength multinomial logistic regression (v1) | Any (uses internal simulated matches) | **Shipped as fallback only** | `scripts/train_models.py` | `app/services/ml_models.py:predict_outcome` |
| Heuristic outcome formula | Any | **Shipped as last-resort fallback** | n/a (hand-coded) | `app/services/ml_models.py:heuristic_outcome` |
| Per-player per-minute event-rate table | Football & basketball | **Shipped** (drives the simulation engine) | `scripts/train_models.py:build_event_rates`, `app/services/features.py` | `app/services/simulation.py` |
| Rule-based post-match rater | Football & basketball | **Shipped** | n/a (hand-coded weights) | `app/services/rater.py` |
| Dixon-Coles bivariate-Poisson goal model | Football | **Research only — not shipped** | `app/services/dixon_coles.py`, `scripts/train_outcome_v2.py` | n/a |
| Logistic-on-rolling-form model | Football | **Research only — not shipped** (lost to Elo+SoT) | `scripts/train_outcome_v2.py`, `app/services/features_team.py` | n/a |
| Causal stacked blend ensemble | Football | **Research only — not shipped** (no accuracy gain, errors too correlated) | `scripts/train_outcome_v3.py` | n/a |
| Multi-league (EPL+Championship) Elo pooling | Football | **Research only — explicitly a negative result** | `scripts/train_outcome_v5.py` | n/a |
| Prediction scoring / calibration engine | Any | **Shipped** (not a predictive model — an evaluation tool) | `app/services/prediction_metrics.py` | `GET /predict/metrics` |

The "winning" production models are the two **Elo + logistic regression bundles**
(`models_pkl/outcome_v2.pkl` for football, `models_pkl/outcome_v2_basketball.pkl` for
basketball) — everything else in `scripts/train_outcome_v3.py`/`v4.py`/`v5.py` and
`scripts/train_basketball_v3.py` was research that either got folded into those two
bundles' hyperparameters or was rejected. This is unusually well-documented in-repo: every
research script's docstring explicitly states *"NOTHING is wired into the live app here"*
or *"No live app model was modified by this script."*

---

## 1. Elo rating engine

**File:** `app/services/team_ratings.py`. **Full name:** Elo rating system, named for its
inventor Arpad Elo (originally designed for chess).

### Purpose
Give every team a single scalar "strength" number that updates after each match result,
usable to predict the outcome of a *future* match between any two rated teams — including
teams that have never played each other, which raw head-to-head statistics cannot do.

### Why this model was selected
- It requires only match results (no player-level data) and updates in O(1) per match,
  so it's trivial to keep "live" as new results arrive (`scripts/refresh_bundles.py`).
- It naturally handles new/promoted teams: an unseen team gets a configurable default
  rating (`base=1500.0`) rather than needing special-cased cold-start logic.
- `reports/MODEL_IMPROVEMENT_PLAN.md` explicitly rejects heavier alternatives (gradient
  boosting, neural nets) as overfitting risks given only ~4,900 EPL matches and a handful
  of features — Elo's single-number-per-team parameterization is appropriately
  low-capacity for the data volume available.

### Alternative models considered
- **Bradley-Terry / other paired-comparison models** — mathematically close cousins of
  Elo; not implemented, Elo was simpler to reason about and is the de facto standard in
  sports analytics (used by FiveThirtyEight, World Football Elo Ratings, chess).
- **Dixon-Coles** (evaluated, see §4) and **rolling-form logistic regression** (evaluated,
  see §5) were both tried as *alternatives to* Elo-diff as the outcome-model feature and
  lost on walk-forward log loss.

### Mathematical intuition
Two teams' ratings imply an expected match outcome via the logistic (Elo) curve:

```
expected_home = 1 / (1 + 10^(-((R_home + HOME_ADV) - R_away) / 400))
```

`EloModel.expected_home` (`team_ratings.py:69-72`) implements exactly this. After the
match is played, both ratings move toward the *actual* result by an amount proportional
to how surprising it was:

```
delta = K * mov_multiplier * (actual_score_home - expected_home)
R_home += delta
R_away -= delta
```

where `actual_score_home` is `1.0` for a home win, `0.5` for a draw, `0.0` for an away
win (`EloModel.update`, lines 101-113). `K` controls how fast ratings move (a large `K`
overreacts to a single result; a small `K` is slow to reflect real improvement).

**Margin-of-victory (MOV) multiplier** (`EloModel._mov_multiplier`, lines 81-99) — an
optional extra factor on `delta` so a 5-0 win moves ratings more than a 1-0 win:
- `mov="wfe"` (World Football Elo goal-difference multiplier): `1.0` for a margin ≤1,
  `1.5` for margin `2`, `(11 + margin) / 8` for margin ≥3.
- `mov="fte"` (FiveThirtyEight multiplier, the NBA-style formula):
  `((margin + 3)^0.8) / (7.5 + 0.006 * winner_diff)`, where `winner_diff` is the
  pre-match Elo edge *in favor of whoever actually won* — this is a "favourite-blowout
  dampening" guard: if the pre-match favourite wins by a lot, the multiplier shrinks, so
  a strong team can't keep inflating its own rating by beating weak teams badly (an
  autocorrelation problem the raw formula would otherwise have).
- `mov=None` (default): classic Elo, multiplier is always `1.0`.

**Season regression** (`EloModel._maybe_regress`, lines 74-79): at the first match of a
new season (tracked via `_last_season`), every team's rating is pulled `season_regression`
fraction of the way back toward `base` (`R += season_regression * (base - R)`). This
models the fact that squads change over the close season and a team's true strength
regresses toward the mean between seasons — implemented as mean-reversion, not a hard
reset, so the specific choice of `season_regression` trades off "carry forward earned
skill" against "don't over-trust a stale rating."

### Input features
Only match results: home team identifier, away team identifier, home goals/points,
away goals/points, and a season label (string, used only to detect season boundaries for
regression — order within a season doesn't matter to this trigger, only the season-label
*change*).

### Output
A `dict[str, float]` (`EloModel.ratings`) mapping team name/id → current rating.
`annotate_pre_match_elo` (lines 129-156) additionally produces, per historical match row,
the **pre-match** ratings and derived features used to train the logistic layer:
`elo_home`, `elo_away`, `elo_diff = (elo_home + home_advantage) - elo_away`,
`elo_exp_home` (the expected-score sigmoid value).

### Training / fitting process
There is no "loss function" being minimized for Elo itself — `fit_elo` (lines 116-126)
and `annotate_pre_match_elo` both simply **replay history forward, one match at a time,
in chronological order**, calling `.update()` after each. This is why every caller in the
codebase is careful to sort input frames causally
(`scripts/load_historical.py`/`load_nba.py`: `.sort_values(["date", "home"],
kind="mergesort")` — `mergesort` specifically because it's *stable*, so same-day matches
keep a deterministic order run to run).

`annotate_pre_match_elo` is the **leakage-free** variant used for feature engineering: it
records each match's pre-match ratings *before* calling `.update()` with that match's own
result — so a match's own outcome is never available to any feature derived from it (a
correctness property re-verified by every downstream backtest, since Elo/features feed a
supervised classifier that must never see the label baked into its own inputs).

### Hyperparameters (per shipped bundle)

| Parameter | Football (`outcome_v2.pkl`, `scripts/finalize_outcome_v2.py`) | Basketball (`outcome_v2_basketball.pkl`, `scripts/finalize_outcome_basketball.py`) |
|---|---|---|
| `k` (update step size) | 40.0 | 20.0 |
| `home_advantage` | 65.0 Elo points | 60.0 Elo points |
| `season_regression` | 0.10 | 0.40 |
| `mov` | `"fte"` | `"fte"` |
| `base` (unseen-team rating) | 1500.0 | 1500.0 |

These were arrived at by walk-forward grid search in `scripts/train_outcome_v4.py`
(football: `mov ∈ {wfe, fte}`, `K ∈ {10..40}`, `SR ∈ {0..0.40}`, `HA` fixed at 65 — shown
to be "absorbed by the logistic intercept" in `train_outcome_v3.py`'s findings) and
`scripts/train_basketball_v3.py` (basketball: same grid plus `HA ∈ {60,100,140}`).

### Assumptions
- Match outcomes are (approximately) a function of a single latent "strength" scalar per
  team — no explicit modelling of home-vs-away-specific skill, injuries, or matchup
  effects beyond the flat `home_advantage` constant.
- Team identity is stable over time — handled via `team_ratings.py:APP_TEAM_ALIASES`
  (football long names → football-data.co.uk short names) and stable franchise
  `team_id`s for basketball (`scripts/load_nba.py` explicitly notes this "survives
  relocations").
- A team's rating is meaningful even against opponents it has never played — the
  transitive-comparison assumption inherent to all Elo-family systems.

### Weaknesses / limitations
- Ignores any information beyond match results and score margin — no player
  availability, tactics, or injuries.
- Basketball's rating for a team is frozen at whatever the last training run saw;
  `reports/MODEL_IMPROVEMENT_PLAN.md` flags this explicitly: *"the bundle's `elo_ratings`
  dict is frozen at build time; matches played afterwards never move ratings"* — the
  mitigation shipped is `scripts/refresh_bundles.py` (a manually-triggered or cron-driven
  full rebuild), not an online/incremental update against live results.
- `season_regression` and `home_advantage` are single global constants — no
  team-specific or era-specific variation (the improvement plan explicitly flags NBA
  home-court advantage as having "declined markedly" over time, unaddressed).

### Strengths
- Cheap to compute, cheap to store, trivial to explain to a non-ML stakeholder ("this
  team is rated 1620, that one 1550").
- Naturally incremental — new results can update ratings without retraining anything else.
- Margin-of-victory variants are well-studied, standard techniques, not bespoke.

### Complexity
Time: O(1) per match update, O(M) to fit M historical matches. Space: O(T) for T teams.

### How it interacts with the rest of the system
Elo ratings are the **primary feature** fed into the logistic regression classifier (§3)
that actually produces win/draw/loss probabilities. The engine is entirely separate from,
and never touches, the live simulation engine (`SIMULATION.md`) — simulation event rates
come from `event_rates.pkl` (§6), a completely different artifact.

---

## 2. Dixon-Coles bivariate-Poisson goal model (research only, not shipped)

**File:** `app/services/dixon_coles.py`. **Full name:** the Dixon & Coles (1997) model
for association football scores.

### Purpose
Model the *full scoreline* distribution (not just win/draw/loss) by treating each team's
goal count as (nearly) an independent Poisson random variable, with team-specific attack
and defence strength parameters, then derive win/draw/loss probabilities by summing over
the resulting scoreline probability matrix.

### Why evaluated
Phase C of the model-improvement effort (`reports/OUTCOME_MODEL_TRAINING_PLAN.md`)
wanted a genuinely different feature family from Elo (a goal-generating process, not a
single strength scalar) to see if it could beat Elo-logistic on walk-forward log loss, or
be blended with it.

### Mathematical intuition
For a match between home team `h` and away team `a`:
```
log(lambda_home) = home_adv + attack[h] + defence[a]
log(mu_away)     =            attack[a] + defence[h]
```
`lambda_home`/`mu_away` are the Poisson rate parameters for home/away goals. Naively,
`P(home=x, away=y) = Poisson(x; lambda) * Poisson(y; mu)` — but real football data shows
systematically *more* 0-0, 1-0, 0-1, and 1-1 results than independent Poissons predict.
Dixon-Coles fixes this with a correction factor `tau(x, y, lambda, mu, rho)`
(`_tau`, lines 32-42) applied only to the four low-score cells:

```
tau(0,0) = 1 - lambda*mu*rho
tau(0,1) = 1 + lambda*rho
tau(1,0) = 1 + mu*rho
tau(1,1) = 1 - rho
tau(x,y) = 1   for all other (x,y)
```

`rho` (fitted, typically small and negative — the code initializes it at `-0.05`) governs
how strongly this correction pulls probability mass toward/away from those cells.

### Input features / training data
Raw match results (`home`, `away`, `fthg`, `ftag`, `date`) — no external features. Team
identity strings become dense integer indices via `self.index`.

### Training process
`DixonColes.fit(matches, as_of)` (lines 57-107) maximizes a **time-decay-weighted,
L2-regularised log-likelihood** via `scipy.optimize.minimize(method="L-BFGS-B")`:

- **Time decay:** each match's log-likelihood contribution is weighted by
  `w = exp(-xi * age_days)` where `xi = ln(2) / decay_half_life_days` (default half-life
  180 days) — a match 180 days old counts half as much as a fresh one, so the fitted
  attack/defence parameters reflect *recent* form more than a flat average of all history.
- **L2 regularisation:** `penalty = l2 * (sum(attack^2) + sum(defence^2))` (default
  `l2=0.01`) — stabilises parameter estimates for teams with few matches (e.g. newly
  promoted sides), preventing wild attack/defence values from a handful of extreme
  results.
- **Parameter vector:** `[attack(n teams), defence(n teams), home_adv, rho]`, bounded
  `[-3,3]` for attack/defence, `[-1,1]` for `home_adv`, `[-0.2,0.2]` for `rho`; capped at
  `maxiter=500`.
- The negative log-likelihood function (`nll`, lines 85-97) vectorises the Poisson
  log-pmf computation over all matches with NumPy, and only loops in Python over the
  small subset of "low-score" matches (`hg<=1 and ag<=1`) that need the `tau` correction
  — an explicit performance optimisation noted in the code comments.

### Output
`predict_proba(home, away)` returns
`{"H": p_home_win, "D": p_draw, "A": p_away_win, "lambda_home": ..., "mu_away": ...}`,
computed by building the full `(MAX_GOALS+1) × (MAX_GOALS+1)` scoreline probability
matrix (`MAX_GOALS = 10`, so 121 cells — *"P(>10 goals) is negligible"*), applying `tau`
to the 2×2 low-score block, renormalising, then summing the lower triangle for a home
win, upper triangle for an away win, and the trace (diagonal) for a draw.

### Evaluation result (why it was NOT shipped)
`reports/OUTCOME_MODEL_PHASE_C.md` (generated by `scripts/train_outcome_v2.py`) compared
`dixon_coles` against `elo_logistic` and `logistic_form` on the same walk-forward folds;
Elo-logistic had the best (lowest) pooled log loss. The improvement plan later notes
Dixon-Coles was strong specifically on **ECE** (calibration) but was folded into a
proposed — and itself ultimately abandoned — ensemble blend (`train_outcome_v3.py`)
rather than shipped standalone.

### Weaknesses
- More parameters than Elo (`2×n_teams + 2` vs. `n_teams`), which the improvement plan
  flags as an overfitting risk at ~4,900-match sample sizes.
- Refit from scratch per walk-forward fold (no incremental update), so it's the most
  computationally expensive candidate evaluated (L-BFGS-B over ~40+ parameters per fit).
- Never integrated with the SoT-form feature that ultimately won for the Elo path.

---

## 3. Multinomial logistic regression classifiers

Two distinct logistic regression models exist in the codebase, serving very different
roles. **Do not confuse them** — they use different feature sets, different training data,
and different code paths.

### 3a. `outcome_v2` / `outcome_v2_basketball` — the SHIPPED Elo-based classifier

**Files:** built by `scripts/finalize_outcome_v2.py` (football) and
`scripts/finalize_outcome_basketball.py` (basketball); served by
`app/services/ml_models.py:predict_outcome_v2`.

**Purpose:** map the Elo-derived numeric feature(s) to calibrated H/D/A (or H/A for
basketball) probabilities — Elo's `expected_home` sigmoid alone is not directly a
3-class (H/D/A) probability distribution, so a small classifier learns the actual
mapping from real historical outcomes.

**Input features:**
- Football (`outcome_v4_elo_sot`, `MODEL_VERSION_V4` in `ml_models.py`): 2 features —
  `elo_diff` and `sot_net_diff` (causal rolling shots-on-target net form, see §7).
- Basketball: 1 feature — `elo_diff` only (no shots-on-target equivalent tracked for
  basketball).

**Preprocessing:** `sklearn.preprocessing.StandardScaler` (zero-mean, unit-variance),
inside a `Pipeline` with the classifier — never a separately-tracked scaler file
(explicitly documented convention, `ml_models.py:1-6`).

**Model:** `sklearn.linear_model.LogisticRegression(max_iter=1000)` — default solver
(`lbfgs`), multinomial by default for >2 classes as of the sklearn version pinned
(`requirements.txt: scikit-learn>=1.5`); `finalize_outcome_v2.py` fits on
`model.classes_` being `['A','D','H']` (sklearn sorts string labels alphabetically) and
`predict_outcome_v2` maps back via `str(label)` keys — see "Assumptions" below for why
this matters.

**Output classes:** football 3-class (`H`/`D`/`A`); basketball 2-class (`H`/`A`, no
draws possible in basketball).

**Training process (exact, football):**
1. `df = load_matches()` — 13 seasons of real EPL results
   (`scripts/load_historical.py`).
2. `annotated = annotate_shot_form(annotate_pre_match_elo(df, k=40, home_advantage=65,
   season_regression=0.10, mov="fte"))` — causal Elo + causal rolling SoT form computed
   over the *entire* dataset in one forward pass (this is the **final production fit**,
   not a walk-forward fold — walk-forward is only used during the earlier research phase
   to *choose* these hyperparameters).
3. `model.fit(annotated[["elo_diff","sot_net_diff"]].values, annotated["ftr"].values)` —
   fit on ALL 13 seasons at once.
4. Separately, `elo = fit_elo(df, ...)` replays the *same* hyperparameters through the
   whole dataset again to capture the **final, current** Elo rating per team (as opposed
   to the per-row historical snapshots used for feature engineering) — this is what gets
   used at prediction time for a *future* match.
5. `sot_form = shot_form_state(df)` — likewise, the **current** end-of-data SoT net form
   per team.
6. Bundle assembled and pickled (schema below).

**Training process (basketball):** identical shape but only `elo_diff` (no SoT-equivalent
feature), fit over `scripts/load_nba.py`'s ~19-season dataset, with team keys re-mapped
from the Elo engine's internal franchise `team_id` to the **current abbreviation** (what
the live app's `players`/`teams` rows actually store) — `finalize_outcome_basketball.py:53-55`.

**Bundle (`.pkl`) schema** (both sports, per `finalize_outcome_v2.py:9-14` and
`finalize_outcome_basketball.py`):
```python
{
    "kind": "elo_logistic",
    "sport": "football" | "basketball",
    "model": <sklearn Pipeline>,
    "features": ["elo_diff"] or ["elo_diff", "sot_net_diff"],
    "elo_ratings": {canonical_team_name: rating, ...},
    "sot_form": {team: net_sot_form, ...},        # football only
    "aliases": {app_name: elo_key_name, ...},      # team-name normalisation map
    "home_advantage": 65.0, "base": 1500.0,
    "mov": "fte",                                  # football bundle only
    "classes": ["A", "D", "H"] (or ["A","H"]),
    "model_version": "outcome_v4_elo_sot" | (basketball has no explicit version key set,
                                              defaults to MODEL_VERSION_V2 at inference),
    "trained_on": "<n> EPL matches, seasons 2013-14..2025-26",
    "n_teams": 20,
}
```

**Inference (exact path, `ml_models.py:predict_outcome_v2`, lines 71-133):**
1. Guard: bundle must exist, `kind == "elo_logistic"`, and carry `elo_ratings` — else
   return `None` (caller falls back to v1).
2. Normalise both team names through the bundle's own `aliases` map (`normalize_team_name`
   in `team_ratings.py` — HTML-unescape + trim, then dict lookup with pass-through
   default) so the *app's* stored team name (e.g. "Manchester United") maps to the *Elo
   engine's* key (e.g. "Man United").
3. Look up each team's rating, defaulting to `base` (1500.0) if unknown —
   `home_known`/`away_known` booleans in the output flag whether this default fired.
4. `elo_diff = (rating_home + home_advantage) - rating_away`.
5. Assemble the feature row **in the order the bundle's own `features` list specifies**
   (not a hardcoded order) — this is what lets the same inference function serve both
   the 1-feature basketball bundle and the 2-feature football bundle without a branch;
   an unrecognised feature name in the list causes a `None` return (treated as an
   incompatible/newer bundle) rather than a guess.
6. `proba = model.predict_proba([row])[0]`; results are mapped back to named keys via
   `zip(model.classes_, proba)` — **never** assumes a fixed class order (a documented
   PRD requirement, R-3.4, re-verified here for the v2 path too).
7. Also computes a cheap `home_strength`/`away_strength` in `[0,1]` via
   `1 / (1 + exp(-(rating - base) / 400))` — a sigmoid of the raw Elo edge over the base
   rating, used to populate the `/predict` response without an expensive per-player
   feature query (explicitly noted as a performance shortcut in the code comment).

**Evaluation metrics** (from `reports/OUTCOME_MODEL_V4.md`/`OUTCOME_MODEL_BASKETBALL_V3.md`,
generated by the corresponding `train_outcome_v4.py`/`train_basketball_v3.py` walk-forward
harnesses): pooled out-of-sample **log loss**, **accuracy**, **Brier score**, **ECE**
(10-bin confidence-calibration error) — computed by `scripts/backtest_outcome.py`'s shared
metric functions (`accuracy`, `log_loss`, `brier`, `ece`). Football's shipped config: OOS
log loss ≈0.966 vs. a de-margined-bookmaker ceiling of ≈0.953 (gap +0.013). Basketball's
shipped config: OOS log loss ≈0.609 (bookmaker ceiling later measured in
`scripts/backtest_nba_odds.py` at ≈0.593, gap +0.022 over 11,656 odds-matched games —
notably **measured after** the model was already shipped, since no NBA odds dataset was
available during the original basketball model-selection work).

**Weaknesses / limitations:**
- Football's SoT feature requires `HST`/`AST` columns present in the historical CSV,
  which is guaranteed for the training data but has no live-refresh path beyond
  `scripts/refresh_bundles.py` re-running against updated CSVs.
- Basketball has no equivalent second feature — `reports/MODEL_IMPROVEMENT_PLAN.md`
  explicitly flags rest/back-to-back features as proven to help in backtest
  (`elo_rest`/`elo_mov_rest` in `train_basketball_v3.py`) but **not shipped** because "they
  need a real schedule feed at predict time" that the live app doesn't have.
- Ratings/form are frozen at build time (same limitation as Elo itself, §1).

### 3b. `outcome_model` (v1) — the SHIPPED but ONLY-FALLBACK strength-based classifier

**Files:** trained by `scripts/train_models.py`; served by `ml_models.py:predict_outcome`.

**Purpose:** the *original* outcome model, predating the Elo-bundle work, trained
entirely from the feeder's own internal `matches`/`player_stats` data rather than real
historical results. `PRD.md` and `reports/MODEL_IMPROVEMENT_PLAN.md` both explicitly
frame it as a fallback: *"The v1 model trains on internally simulated matches — it
cannot learn real-world skill and should stay a fallback. All improvement effort belongs
in the v2 pipeline."*

**Input features:** exactly 3 numbers per match — `[home_strength, away_strength,
HOME_BIAS]` where `HOME_BIAS` is a constant `1.0` (a bias/intercept-style feature) and
`home_strength`/`away_strength` come from `app/services/features.py:compute_team_strength`
(mean EWMA form-index of the team's players, normalised `/15`, clamped to `[0,1]`).

**Preprocessing:** `MinMaxScaler` inside the `Pipeline` (not `StandardScaler` — a
different choice from the v2 bundles, presumably because the 3 input features are already
roughly bounded in `[0,1]`).

**Model:** `LogisticRegression(solver="lbfgs", max_iter=500)`. The code comment notes the
PRD originally specified `multi_class="multinomial"` but that argument **was removed in
scikit-learn 1.7+**, and `lbfgs` is multinomial by default anyway, so it's simply omitted
— a documented, deliberate deviation from the PRD text, not an oversight.

**Training process** (`scripts/train_models.py:collect_outcome_training_data`,
`train_outcome_model`):
1. Query every `Match` with `status == "finished"` in the feeder's own database.
2. Label each via `_score_match` (the same score-derivation function every other part of
   the app uses) → `0=away win, 1=draw, 2=home win` (`LABEL_AWAY/DRAW/HOME` constants in
   `ml_models.py`).
3. If fewer than `MIN_MATCHES_TO_TRAIN = 5` finished matches exist, **skip training
   entirely** (logs a warning, returns `None` — `/predict` then falls through further to
   the heuristic).
4. If ≥`MIN_MATCHES_FOR_SPLIT = 20` matches exist, do an 80/20 **stratified** train/test
   split (`random_state=42`) and log a `sklearn.metrics.classification_report` on the
   held-out 20% (falls back to fitting on all data if stratification fails because a
   class is too rare — caught via `ValueError`).
5. Otherwise (5–19 matches) fit on 100% of the data with no held-out evaluation at all.

**Weaknesses:** by construction this model can only ever be as good as the feeder's own
simulated match history, which itself is generated by... a model. It has no access to
real-world results and is explicitly documented as never expected to beat the v2 bundles.
Given the feeder ships with essentially no finished internal matches by default, this path
is realistically rarely trained in practice.

### 3c. Heuristic fallback (not a model — hand-coded formula)

**File:** `ml_models.py:heuristic_outcome`. Used when `model is None` (no v1 model
loaded and no v2 bundle either). Purely a hand-tuned linear formula:
```python
diff = home_strength - away_strength
home = max(0.05, 0.40 + 0.35 * diff)
away = max(0.05, 0.32 - 0.35 * diff)
draw = 0.28
# then normalized so home+draw+away == 1.0
```
`model_version = "heuristic_v1"`. This guarantees `/predict` **never** hard-fails even on
a completely cold system with zero trained artifacts — the outermost ring of the
graceful-degradation chain described in `ARCHITECTURE.md`.

### The full `/predict` fallback chain (exact, `app/routers/predict.py:36-107`)
```
1. Resolve match's sport_type.
2. If FOOTBALL: try app.state.outcome_v2 bundle.
   If BASKETBALL: try app.state.outcome_v2_basketball bundle.
   (any other resolved sport, or a missing/incompatible bundle) -> result stays None
3. If a v2 result was produced -> use its home_strength/away_strength directly (cheap).
   Else -> compute_team_strength() for both teams (DB reads over player_stats),
           then run the v1 model (or heuristic if v1 is also absent).
4. Store a MatchPrediction row unconditionally.
5. Push to the Sporty backend IF the match has a "match" entity_link.
```

---

## 4. Prediction metrics / continuous backtest engine

**File:** `app/services/prediction_metrics.py`. **This is not a predictive model** — it's
an evaluation harness that runs *in production* against real stored predictions, turning
every `/predict` call into a data point for an ongoing accuracy audit.

### Purpose
`reports/MODEL_IMPROVEMENT_PLAN.md` step 4: *"`match_predictions` rows are written on
every `/predict` but never evaluated... turns production into a continuous backtest and
detects drift for free."*

### Exact computation (`compute_prediction_metrics`)
1. Join `match_predictions` to `matches` where `status == "finished"`.
2. For matches predicted more than once (even by the same model version — e.g. re-run
   after a lineup change), keep only the **latest** `MatchPrediction` row per
   `(match_id, model_version)` pair (rows are processed in creation order, so later
   inserts simply overwrite the dict entry for that key — a clean, index-free way to
   dedupe to "latest").
3. For each surviving row, compute the actual result label (`H`/`D`/`A`) via
   `_score_match` — reusing the match-scoring function, so the metrics can never disagree
   with what the app itself reports as the score.
4. Group by `model_version` and compute, per group: `n`, **accuracy**, **log loss**,
   **Brier score**, **ECE** (10 confidence bins), plus a full calibration table (per-bin
   `n`, average confidence, actual accuracy).

### Formulas (identical shapes to the offline backtest scripts, reimplemented in
production without a NumPy/pandas dependency — `prediction_metrics.py` is pure Python +
`math`, deliberately lightweight):
```
log_loss  = mean( -ln(p[true_class]) )                    # clipped at EPS=1e-15
brier     = mean( sum_c (p_c - onehot_c)^2 )               # multi-class Brier
ece       = sum_bins( |accuracy_bin - avg_confidence_bin| * n_bin / n_total )
```

### Why grouped by `model_version` rather than blended
So that shipping a new model bundle (e.g. `outcome_v2_elo` → `outcome_v4_elo_sot`) shows
up in the scorecard as **two separate, comparable rows**, letting an operator directly
verify a new bundle is actually performing better in the wild, not just in the offline
backtest — closing the loop between offline evaluation and real production behaviour.

### Where it's surfaced
`GET /predict/metrics` (on-demand), `POST /predict/metrics/push` (manual push to the
Sporty backend's `/api/v1/feed/model-metrics`), and automatically at the end of every
simulation (`app/services/simulation.py:run_simulation`, wrapped in its own
`try/except Exception` so a metrics-push failure can never affect the simulation's own
success/failure state).

---

## 5. Event-rate table — the model that actually drives the live simulation

**Files:** built by `scripts/train_models.py:build_event_rates` (calls
`app/services/features.py:compute_player_features` for every player with `player_stats`
rows); consumed by `app/services/simulation.py`.

### Purpose
Not a classifier — a **lookup table** of per-player, per-event-type, per-minute
probabilities, `{player_id: {event_type: probability}}`, pickled to
`models_pkl/event_rates.pkl`. This is the mechanism that replaced the original "every
player has identical odds" random simulator (`PRD.md`'s stated Phase 1 problem).

### Feature engineering (`app/services/features.py:compute_player_features`)
For each player, the last `MAX_FORM_ROWS = 20` `player_stats` rows (ordered by
`(season desc, gameweek desc)`, i.e. most recent first) are aggregated:

- **Football:** `event_rates = {goal: total_goals/total_minutes, assist:
  total_assists/total_minutes, yellow_card: total_yellows/total_minutes, red_card:
  total_reds/total_minutes}` — simple counts-over-minutes rates.
- **Basketball:** `player_stats` only stores *total points*, not the 2pt/3pt/free-throw
  split, so the code **decomposes** total points into an approximate event mix using a
  league-average split constant, `BASKETBALL_POINT_MIX = {point_2: 0.60, point_3: 0.25,
  free_throw: 0.15}` (i.e. assumes 60% of a player's scoring output comes via 2-pointers
  league-wide, applied uniformly to every player — a simplification, since real players
  vary widely in shot-selection mix): `rate[event] = (total_pts * share / point_value) /
  total_minutes`. `assist`/`rebound` rates come directly from `total_ast`/`total_reb`.
- **Form index:** an EWMA (§ below) over the same window, used elsewhere (team strength,
  v1 model) but not directly by the simulator.
- **Minutes ratio:** `min(1.0, (total_minutes / n_rows) / full_match_minutes)` — average
  minutes played per appearance, as a fraction of a full match; computed but, per current
  code inspection, **not actually consumed anywhere downstream** (`_prepare` in
  `simulation.py` reads only `event_rates` off this dict, not `minutes_ratio`) — a
  candidate improvement (§`IMPROVEMENTS.md`).

### EWMA (Exponentially Weighted Moving Average) — form index
`_ewma(values, alpha=0.4)` (`features.py:60-71`): given `values` ordered **newest
first**, weight `i`-th-newest value by `alpha * (1-alpha)^i`, normalise by the sum of
weights actually used (not a fixed denominator), so the most recent match contributes
40% of the weighted average, the one before that 24% (`0.4 * 0.6`), and so on — an
infinite-lookback exponential decay truncated to whatever's actually in `values`
(≤20 rows). This is what "form" means throughout the codebase — never a simple mean.

### Cold-start fallback (`_fallback_features`, league-average constants)
```python
FOOTBALL_FALLBACK_RATES  = {"goal": 0.003, "assist": 0.003, "yellow_card": 0.002, "red_card": 0.0002}
BASKETBALL_FALLBACK_RATES = {"point_2": 0.04, "point_3": 0.015, "free_throw": 0.02, "assist": 0.05, "rebound": 0.07}
```
Applied whenever a player has zero `player_stats` rows or zero total minutes across the
retrieved rows — **never raises**, per the explicit PRD requirement R-3.1. `form_index`
also gets a neutral fallback constant (`NEUTRAL_FORM_INDEX = 7.5`, chosen so `/15`
normalisation yields exactly `0.5` team strength — i.e. "average").

### How it's used at inference time
`app/services/simulation.py:_prepare` looks up each on-pitch/on-court player's rates by
id in the loaded `event_rates` dict; a miss (rather than an explicit call into
`compute_player_features`) falls back to the same *sport-level* constants directly
(`FOOTBALL_FALLBACK_RATES`/`BASKETBALL_FALLBACK_RATES`, imported straight from
`features.py`) — i.e. the simulator uses the **pre-baked pickle**, not a live DB query,
for performance (no per-minute database round-trip). See `SIMULATION.md` for exactly how
these per-minute probabilities become Bernoulli-sampled events.

### Evaluation
There is **no offline accuracy metric for this table** — it's not evaluated by
`scripts/backtest_*.py` (those only evaluate the *outcome* classifiers). Its "correctness"
is validated only indirectly, by simulation-level tests (`tests/test_simulation_calibration.py`)
checking that simulated aggregate scoring rates land near real league averages after the
calibration step (see `SIMULATION.md` §Calibration).

---

## 6. Causal team-form feature engineering (supporting features, not standalone models)

**File:** `app/services/features_team.py`. Three families of rolling, **strictly causal**
(no future-data leakage) team-level features, each computed by iterating a chronologically
sorted match frame row-by-row, recording the feature *before* updating any internal
state with that row's own result:

1. **Rest days** (`annotate_rest_days`): `home_rest`/`away_rest` = full days off since
   each team's previous game (capped at `REST_CAP_DAYS = 5`; season openers get the cap
   as if fully rested), `rest_diff`, and `home_b2b`/`away_b2b` (1.0 if playing on zero
   days' rest, i.e. a back-to-back). Built for basketball (step 2 of the improvement
   plan) but **only used in research scripts** (`train_basketball_v3.py`'s
   `elo_rest`/`elo_mov_rest` candidates) — not in the shipped basketball bundle, because
   "they need a real schedule feed at predict time" the live app doesn't have.
2. **Shots-on-target form** (`annotate_shot_form`, `shot_form_state`): rolling mean of
   shots-on-target for/against over the last `window=10` matches (any venue), cold-start
   default `SOT_PRIOR = 4.3` (the league-average EPL SoT/game). `sot_net_diff = (home_for
   - home_against) - (away_for - away_against)`. **This one IS shipped** — it's the
   second feature in the production football bundle. `shot_form_state` is the
   prediction-time counterpart: run the exact same causal loop over all history and
   return the *end-of-data* state per team, baked into the bundle the same way
   `elo_ratings` is.
3. **Rolling points-per-game / goals form** (`annotate_rolling_form`): `home_ppg`,
   `away_ppg` (last `window=5` matches, any venue), `home_gf`/`home_ga` (home-venue
   goals for/against), `away_gf`/`away_ga` (away-venue). Cold-start priors: 1.4 ppg,
   1.4 GF, 1.4 GA. **Evaluated and rejected** — `reports/MODEL_IMPROVEMENT_PLAN.md`:
   "Phase C's goals-based form features... added nothing — goals are too noisy,"
   directly motivating the switch to shots-on-target instead.

All three share the identical causal pattern: `deque(maxlen=window)` per team, read the
current average *before* appending this row's own values — the same "record-then-update"
discipline as `annotate_pre_match_elo`.

---

## 7. Rule-based rater (post-match player ratings)

**File:** `app/services/rater.py`. Not a trained model — a fixed linear scoring formula,
explicitly specified as such in `PRD.md` R-3.5.

### Formula
```
rating = clamp(1.0, 10.0,  BASE_RATING=6.0 + sum(weight[event_type] for each event the player recorded))
```
| Football weight | Value | Basketball weight | Value |
|---|---:|---|---:|
| goal | +2.0 | point_2 | +0.8 |
| assist | +1.2 | point_3 | +1.2 |
| yellow_card | −0.5 | free_throw | +0.3 |
| red_card | −2.5 | assist | +0.7 |
| | | rebound | +0.4 |
| | | steal | +0.6 |
| | | block | +0.5 |

Event types with no listed weight (e.g. a `substitution` event) contribute `0`.

### Man of the match
`find_man_of_match` = `argmax(rating)`; ties broken deterministically by **lowest player
id** (`min(ratings, key=lambda pid: (-rating[pid], pid))`) — a documented, reproducible
tie-break rather than arbitrary dict-iteration order.

### Why rule-based, not learned
No training data exists for "true" player quality per match — ratings are a
presentation/gamification feature (`PRD.md` G3: "Ratings correlate with goal+assist
counts (r > 0.7)" — a correlation target, not a supervised-learning target), so a
transparent, tunable formula was judged sufficient and is trivially explainable to
end users on the frontend.

---

## 8. Rejected/negative-result research (documented for completeness)

`reports/OUTCOME_MODEL_V5.md` (`scripts/train_outcome_v5.py`) tested pooling English
Championship (second-tier) matches into the Elo/SoT computation so promoted teams enter
the Premier League with an earned rating instead of a cold-start default. **Explicit
negative result:** even the best pooled variant scored worse pooled out-of-sample log
loss (0.967) than the shipped E0-only `outcome_v4_elo_sot` (0.966) — the Championship
data is retained in `historical-datas/championship/` "for revisits" but nothing was
changed in the live model. This is a useful signal for anyone tempted to re-attempt
league pooling: the walk-forward harness already tested it and it didn't help.

`scripts/train_outcome_v3.py`'s causal stacked blend (multiple candidate models'
probabilities combined via per-fold-fitted softmax weights) is a real, working
implementation of stacked generalisation, but was abandoned because the candidate models'
errors were "too correlated" — blending gains nothing when the models are wrong about the
same matches.

## 9. Explain Like I'm New

Two separate "brains" run this app. The **prediction brain** answers "who's going to
win?" before a match starts — it works by giving every team a single skill number (like a
video-game character's power level, called an Elo rating) that goes up when they win and
down when they lose, more so for big wins than narrow ones. A small second-stage model
then converts "team A is 80 points stronger than team B" into an actual percentage chance
of a home win, draw, or away win, because the raw skill gap alone doesn't map linearly to
odds. The **simulation brain** is completely different — it doesn't predict the outcome at
all, it generates a *plausible sequence of events* minute by minute using each individual
player's own historical scoring/assisting/card rates, like rolling a loaded die once per
player per minute where the die's weighting comes from real stats instead of being fair.
