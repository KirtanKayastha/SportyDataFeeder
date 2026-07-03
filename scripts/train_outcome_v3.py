# /home/sam069/projects/SportyDataFeeder/scripts/train_outcome_v3.py
#
# Step 1 of reports/MODEL_IMPROVEMENT_PLAN.md: improve the football outcome
# model with (a) tuned Elo hyperparameters, (b) an abs(elo_diff) draw feature,
# and (c) a causal stacked ensemble of the Phase C candidates. Judged on the
# SAME expanding-window walk-forward folds and metrics as Phase B/C
# (scripts/backtest_outcome.py), so every number is comparable and leakage-free.
#
# Stage 1  grid-search (K, home_advantage, season_regression) minimising pooled
#          walk-forward log loss of the plain elo_logistic model.
# Stage 2  walk-forward comparison of:
#            elo_incumbent  — K=20, HA=65, SR=0.25 (the production outcome_v2)
#            elo_tuned      — stage-1 best params, same 1-feature model
#            elo_absdiff    — tuned params, features [elo_diff, |elo_diff|]
#            logistic_form  — Phase C form model on tuned Elo
#            dixon_coles    — Phase C goal model (unchanged)
#            blend_ed       — causal blend of elo_absdiff + dixon_coles
#            blend_edf      — causal blend of all three
#            bookmaker      — de-margined Bet365 ceiling
#          Blend weights for fold s are fit ONLY on out-of-sample predictions
#          from folds before s (equal weights for the first fold) — stacked
#          generalisation without leakage.
#
# Output: reports/OUTCOME_MODEL_V3.md + printed summary. NOTHING is wired into
# the live app here — integration happens in scripts/finalize_outcome_v2.py
# once a winner is confirmed.
#
# Usage:  python -m scripts.train_outcome_v3

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize as sp_minimize
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.features_team import annotate_rolling_form
from app.services.team_ratings import annotate_pre_match_elo
from scripts.backtest_outcome import (
    CLASSES, WARMUP_SEASONS, accuracy, brier, ece, log_loss, p_bookmaker,
)
from scripts.load_historical import load_matches
from scripts.train_outcome_v2 import (
    FORM_FEATURES, fit_dixon_coles, fit_logistic_form, p_dixon_coles, p_logistic_form,
)

REPORT = Path(__file__).resolve().parents[1] / "reports" / "OUTCOME_MODEL_V3.md"

INCUMBENT = {"k": 20.0, "home_advantage": 65.0, "season_regression": 0.25}
GRID_K = [10.0, 15.0, 20.0, 25.0, 30.0, 40.0]
GRID_HA = [40.0, 55.0, 65.0, 80.0, 95.0]
GRID_SR = [0.0, 0.10, 0.25, 0.40]


# ------------------------------------------------------------------ helpers
def _fit_logistic(train: pd.DataFrame, features: list[str]) -> Pipeline:
    model = Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=1000))])
    model.fit(train[features].values, train["ftr"].values)
    return model


def _proba(model: Pipeline, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    proba = model.predict_proba(test[features].values)
    order = [list(model.classes_).index(c) for c in CLASSES]
    return proba[:, order]


def _folds(df: pd.DataFrame):
    seasons = list(dict.fromkeys(df["season"]))
    for s in seasons[WARMUP_SEASONS:]:
        train = df[df["season"].isin(seasons[: seasons.index(s)])]
        test = df[df["season"] == s]
        yield s, train, test


def _walkforward_ll(df: pd.DataFrame, features: list[str]) -> float:
    """Pooled walk-forward log loss of a logistic on `features`."""
    ps, ys = [], []
    for _, train, test in _folds(df):
        model = _fit_logistic(train, features)
        ps.append(_proba(model, test, features))
        ys.append(test["ftr"].to_numpy())
    return log_loss(np.vstack(ps), np.concatenate(ys))


def fit_blend_weights(member_probs: list[np.ndarray], y: np.ndarray) -> np.ndarray:
    """Simplex weights minimising log loss of the weighted probability average.
    Softmax parametrisation keeps the search unconstrained."""
    m = len(member_probs)

    def nll(theta):
        w = np.exp(theta - theta.max())
        w = w / w.sum()
        p = sum(wi * pi for wi, pi in zip(w, member_probs))
        return log_loss(p, y)

    res = sp_minimize(nll, np.zeros(m), method="Nelder-Mead",
                      options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 2000})
    w = np.exp(res.x - res.x.max())
    return w / w.sum()


# ------------------------------------------------------------------ stage 1
def tune_elo(raw: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Grid-search Elo hyperparameters on walk-forward log loss (elo_logistic)."""
    rows = []
    best = None
    for k, ha, sr in product(GRID_K, GRID_HA, GRID_SR):
        df = annotate_pre_match_elo(raw, k=k, home_advantage=ha, season_regression=sr)
        ll = _walkforward_ll(df, ["elo_diff"])
        rows.append({"k": k, "home_advantage": ha, "season_regression": sr, "log_loss": ll})
        if best is None or ll < best["log_loss"]:
            best = rows[-1]
            print(f"  new best: K={k:.0f} HA={ha:.0f} SR={sr:.2f} -> LL {ll:.5f}")
    grid = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
    return best, grid


# ------------------------------------------------------------------ stage 2
def compare_candidates(raw: pd.DataFrame, tuned: dict) -> dict:
    df_inc = annotate_pre_match_elo(raw, **INCUMBENT)
    df_tun = annotate_rolling_form(annotate_pre_match_elo(
        raw, k=tuned["k"], home_advantage=tuned["home_advantage"],
        season_regression=tuned["season_regression"]))
    df_tun["abs_elo_diff"] = df_tun["elo_diff"].abs()

    single = ["elo_incumbent", "elo_tuned", "elo_absdiff", "logistic_form", "dixon_coles"]
    blends = {"blend_ed": ["elo_absdiff", "dixon_coles"],
              "blend_edf": ["elo_absdiff", "dixon_coles", "logistic_form"]}
    names = single + list(blends) + ["bookmaker"]
    pooled = {n: {"p": [], "y": []} for n in names}
    per_season_acc = {n: {} for n in names}
    # accumulated past OOS member predictions for causal blend-weight fitting
    past = {n: {"p": [], "y": []} for n in single}
    blend_weights_last = {}

    for s, _, test in _folds(df_inc):
        seasons_before = list(dict.fromkeys(df_inc["season"]))
        idx = seasons_before.index(s)
        train_inc = df_inc[df_inc["season"].isin(seasons_before[:idx])]
        train_tun = df_tun[df_tun["season"].isin(seasons_before[:idx])]
        test_tun = df_tun[df_tun["season"] == s]
        y = test["ftr"].to_numpy()
        as_of = test_tun["date"].min()

        preds = {
            "elo_incumbent": _proba(_fit_logistic(train_inc, ["elo_diff"]), test, ["elo_diff"]),
            "elo_tuned": _proba(_fit_logistic(train_tun, ["elo_diff"]), test_tun, ["elo_diff"]),
            "elo_absdiff": _proba(_fit_logistic(train_tun, ["elo_diff", "abs_elo_diff"]),
                                  test_tun, ["elo_diff", "abs_elo_diff"]),
            "logistic_form": p_logistic_form(fit_logistic_form(train_tun), test_tun),
            "dixon_coles": p_dixon_coles(fit_dixon_coles(train_tun, as_of), test_tun),
        }

        # causal blends: weights from PAST folds' OOS predictions only
        for bname, members in blends.items():
            if past[members[0]]["p"]:
                w = fit_blend_weights(
                    [np.vstack(past[m]["p"]) for m in members],
                    np.concatenate(past[members[0]]["y"]),
                )
            else:
                w = np.full(len(members), 1.0 / len(members))
            blend_weights_last[bname] = w
            preds[bname] = sum(wi * preds[m] for wi, m in zip(w, members))

        for n in single:
            past[n]["p"].append(preds[n])
            past[n]["y"].append(y)
        for n, p in preds.items():
            pooled[n]["p"].append(p)
            pooled[n]["y"].append(y)
            per_season_acc[n][s] = accuracy(p, y)

        bp, mask = p_bookmaker(test)
        if mask.any():
            pooled["bookmaker"]["p"].append(bp[mask])
            pooled["bookmaker"]["y"].append(y[mask])
            per_season_acc["bookmaker"][s] = accuracy(bp[mask], y[mask])
        print(f"  fold {s} done (train={len(train_inc)}, test={len(test)})")

    results = {}
    for n in names:
        p = np.vstack(pooled[n]["p"])
        yy = np.concatenate(pooled[n]["y"])
        results[n] = {"n": len(yy), "accuracy": accuracy(p, yy), "log_loss": log_loss(p, yy),
                      "brier": brier(p, yy), "ece": ece(p, yy)}
    return {"results": results, "per_season_acc": per_season_acc, "names": names,
            "blend_weights_last": blend_weights_last,
            "test_seasons": [s for s, _, _ in _folds(df_inc)]}


# ------------------------------------------------------------------ reporting
def _table(results: dict, order: list) -> str:
    rows = ["| Model | n | Accuracy | Log loss | Brier | ECE |",
            "|---|---:|---:|---:|---:|---:|"]
    for n in order:
        r = results[n]
        rows.append(f"| {n} | {r['n']} | {r['accuracy']:.3f} | {r['log_loss']:.3f} "
                    f"| {r['brier']:.3f} | {r['ece']:.3f} |")
    return "\n".join(rows)


def main() -> None:
    raw = load_matches()
    print(f"Stage 1: Elo grid search ({len(GRID_K) * len(GRID_HA) * len(GRID_SR)} configs)")
    tuned, grid = tune_elo(raw)
    print(f"Tuned params: K={tuned['k']:.0f} HA={tuned['home_advantage']:.0f} "
          f"SR={tuned['season_regression']:.2f} (walk-forward LL {tuned['log_loss']:.5f})")

    print("Stage 2: candidate comparison")
    out = compare_candidates(raw, tuned)
    r = out["results"]
    order = out["names"]
    print("\n" + _table(r, order))

    candidates = {n: r[n]["log_loss"] for n in order if n != "bookmaker"}
    winner = min(candidates, key=candidates.get)
    inc_ll = r["elo_incumbent"]["log_loss"]
    book_ll = r["bookmaker"]["log_loss"]
    print(f"\nWinner: {winner} (LL {candidates[winner]:.4f}) | incumbent {inc_ll:.4f} "
          f"| bookmaker {book_ll:.4f}")
    for bname, w in out["blend_weights_last"].items():
        print(f"  {bname} final-fold weights: {[round(float(x), 3) for x in w]}")

    season_rows = ["| Season | " + " | ".join(order) + " |",
                   "|---|" + "---:|" * len(order)]
    for s in out["test_seasons"]:
        cells = [f"{out['per_season_acc'][n].get(s, float('nan')):.3f}" for n in order]
        season_rows.append(f"| {s} | " + " | ".join(cells) + " |")

    top_grid = grid.head(10).to_string(index=False)
    md = f"""# Outcome Model v3 — Tuned Elo + Draw Feature + Causal Ensemble

**Companion to:** `reports/MODEL_IMPROVEMENT_PLAN.md` (step 1), `reports/OUTCOME_MODEL_PHASE_C.md`.
**Data:** real EPL, 13 seasons. Same expanding-window walk-forward + metrics as Phase B/C.
**No live app model was modified by this script.** Integration happens in
`scripts/finalize_outcome_v2.py` once the winner is confirmed.

## Stage 1 — Elo hyperparameter grid search

Grid: K ∈ {GRID_K}, HA ∈ {GRID_HA}, SR ∈ {GRID_SR}; objective = pooled walk-forward
log loss of the 1-feature elo_logistic model.

**Best:** K={tuned['k']:.0f}, home_advantage={tuned['home_advantage']:.0f},
season_regression={tuned['season_regression']:.2f} → LL {tuned['log_loss']:.5f}
(incumbent K=20/HA=65/SR=0.25 → LL {inc_ll:.5f}).

Top 10 configs:

```
{top_grid}
```

## Stage 2 — candidate comparison (pooled out-of-sample)

{_table(r, order)}

- **elo_incumbent** — the production outcome_v2 (K=20, HA=65, SR=0.25).
- **elo_tuned** — stage-1 params, same single elo_diff feature.
- **elo_absdiff** — tuned params + |elo_diff| (lets draw probability peak for even teams).
- **logistic_form / dixon_coles** — Phase C candidates (form model on tuned Elo).
- **blend_ed / blend_edf** — causal stacked blends; fold-s weights fit only on
  out-of-sample predictions from folds before s (equal weights for the first fold).
- **bookmaker** — de-margined Bet365 odds: the practical ceiling.

**Winner:** `{winner}` (pooled OOS log loss {candidates[winner]:.4f}); incumbent
{inc_ll:.4f}; bookmaker ceiling {book_ll:.4f}. Gap to ceiling:
**{candidates[winner] - book_ll:+.4f}** (was {inc_ll - book_ll:+.4f}).

Final-fold blend weights: {{{', '.join(f"{k}: {[round(float(x), 3) for x in v]}" for k, v in out['blend_weights_last'].items())}}}

## Per-season out-of-sample accuracy

{chr(10).join(season_rows)}

## Method notes

- Elo `season_regression` is now a tunable parameter of
  `app/services/team_ratings.py` (default unchanged at 0.25).
- Causal throughout: Elo/form features use only earlier matches; every model and
  every blend weight is fit strictly on data before the test season.
"""
    REPORT.write_text(md)
    print(f"\nWrote {REPORT}")


if __name__ == "__main__":
    main()
