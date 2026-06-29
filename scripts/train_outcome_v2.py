# /home/sam069/projects/SportyDataFeeder/scripts/train_outcome_v2.py
#
# Phase C (reports/OUTCOME_MODEL_TRAINING_PLAN.md): train and walk-forward
# validate the two football outcome CANDIDATE models on real EPL data, judged
# against the Phase B benchmarks (Elo-logistic and the bookmaker ceiling):
#
#   - logistic_form : multinomial logistic on Elo diff + causal rolling form
#   - dixon_coles   : time-decayed bivariate-Poisson goal model -> H/D/A
#
# Same expanding-window folds and metrics as scripts/backtest_outcome.py (reused
# directly, so the comparison is apples-to-apples and leakage-free). The winner
# (lowest pooled out-of-sample log loss) is refit on ALL seasons and saved as a
# candidate bundle models_pkl/outcome_v2.pkl. NOTHING is wired into the live app
# here — integration is Phase E.
#
# Usage:  python -m scripts.train_outcome_v2

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.dixon_coles import DixonColes
from app.services.features_team import annotate_rolling_form
from app.services.team_ratings import annotate_pre_match_elo
from scripts.backtest_outcome import (
    CLASSES, WARMUP_SEASONS, accuracy, brier, ece, log_loss, p_bookmaker, p_elo_logistic,
)
from scripts.load_historical import load_matches

REPORT = Path(__file__).resolve().parents[1] / "reports" / "OUTCOME_MODEL_PHASE_C.md"
BUNDLE = Path(__file__).resolve().parents[1] / "models_pkl" / "outcome_v2.pkl"
FORM_FEATURES = ["elo_diff", "home_ppg", "away_ppg", "home_gf", "home_ga", "away_gf", "away_ga"]


# ----------------------------------------------------------------- candidates
def fit_logistic_form(train: pd.DataFrame):
    model = Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=2000))])
    model.fit(train[FORM_FEATURES].values, train["ftr"].values)
    return model


def p_logistic_form(model, test: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(test[FORM_FEATURES].values)
    order = [list(model.classes_).index(c) for c in CLASSES]
    return proba[:, order]


def fit_dixon_coles(train: pd.DataFrame, as_of) -> DixonColes:
    matches = [
        {"home": r.home, "away": r.away, "fthg": int(r.fthg), "ftag": int(r.ftag), "date": r.date}
        for r in train.itertuples(index=False)
    ]
    return DixonColes().fit(matches, as_of=as_of)


def p_dixon_coles(model: DixonColes, test: pd.DataFrame) -> np.ndarray:
    rows = []
    for r in test.itertuples(index=False):
        p = model.predict_proba(r.home, r.away)
        rows.append([p["H"], p["D"], p["A"]])
    return np.array(rows)


# ----------------------------------------------------------------- harness
def run() -> dict:
    df = annotate_rolling_form(annotate_pre_match_elo(load_matches()))
    seasons = list(dict.fromkeys(df["season"]))
    test_seasons = seasons[WARMUP_SEASONS:]

    names = ["elo_logistic", "logistic_form", "dixon_coles", "bookmaker"]
    pooled = {n: {"p": [], "y": []} for n in names}
    per_season_acc = {n: {} for n in names}

    for s in test_seasons:
        train = df[df["season"].isin(seasons[: seasons.index(s)])]
        test = df[df["season"] == s]
        y = test["ftr"].to_numpy()
        as_of = test["date"].min()

        preds = {
            "elo_logistic": p_elo_logistic(train, test),
            "logistic_form": p_logistic_form(fit_logistic_form(train), test),
            "dixon_coles": p_dixon_coles(fit_dixon_coles(train, as_of), test),
        }
        bp, mask = p_bookmaker(test)

        for n, p in preds.items():
            pooled[n]["p"].append(p); pooled[n]["y"].append(y)
            per_season_acc[n][s] = accuracy(p, y)
        if mask.any():
            pooled["bookmaker"]["p"].append(bp[mask]); pooled["bookmaker"]["y"].append(y[mask])
            per_season_acc["bookmaker"][s] = accuracy(bp[mask], y[mask])
        print(f"  fold {s} done (train={len(train)}, test={len(test)})")

    results = {}
    for n in names:
        p = np.vstack(pooled[n]["p"]); y = np.concatenate(pooled[n]["y"])
        results[n] = {"n": len(y), "accuracy": accuracy(p, y), "log_loss": log_loss(p, y),
                      "brier": brier(p, y), "ece": ece(p, y)}
    return {"results": results, "per_season_acc": per_season_acc,
            "test_seasons": test_seasons, "df": df, "seasons": seasons}


def _table(results: dict, order: list) -> str:
    rows = ["| Model | n | Accuracy | Log loss | Brier | ECE |",
            "|---|---:|---:|---:|---:|---:|"]
    for n in order:
        r = results[n]
        rows.append(f"| {n} | {r['n']} | {r['accuracy']:.3f} | {r['log_loss']:.3f} | {r['brier']:.3f} | {r['ece']:.3f} |")
    return "\n".join(rows)


def main() -> None:
    out = run()
    r = out["results"]
    order = ["elo_logistic", "logistic_form", "dixon_coles", "bookmaker"]

    # winner = lowest log loss among the trained candidates (exclude bookmaker ceiling)
    candidates = {n: r[n]["log_loss"] for n in ["elo_logistic", "logistic_form", "dixon_coles"]}
    winner = min(candidates, key=candidates.get)
    book_ll = r["bookmaker"]["log_loss"]

    print("\n" + _table(r, order))
    print(f"\nWinner (lowest OOS log loss): {winner} ({candidates[winner]:.3f})  | bookmaker ceiling {book_ll:.3f}")

    # Refit winner on ALL seasons and save a candidate bundle (not wired to app).
    df, seasons = out["df"], out["seasons"]
    if winner == "dixon_coles":
        final = fit_dixon_coles(df, as_of=df["date"].max())
        bundle = {"kind": "dixon_coles", "model": final, "features": None}
    elif winner == "logistic_form":
        final = fit_logistic_form(df)
        bundle = {"kind": "logistic_form", "model": final, "features": FORM_FEATURES}
    else:
        final = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
        final.fit(df[["elo_diff"]].values, df["ftr"].values)
        bundle = {"kind": "elo_logistic", "model": final, "features": ["elo_diff"]}
    bundle["classes"] = CLASSES
    bundle["trained_on"] = f"{len(df)} EPL matches, seasons {seasons[0]}..{seasons[-1]}"
    BUNDLE.parent.mkdir(exist_ok=True)
    with BUNDLE.open("wb") as fh:
        pickle.dump(bundle, fh)
    print(f"Saved candidate bundle -> {BUNDLE}")

    season_rows = ["| Season | " + " | ".join(order) + " |",
                   "|---|" + "---:|" * len(order)]
    for s in out["test_seasons"]:
        cells = [f"{out['per_season_acc'][n].get(s, float('nan')):.3f}" for n in order]
        season_rows.append(f"| {s} | " + " | ".join(cells) + " |")

    gap = candidates[winner] - book_ll
    md = f"""# Outcome Model — Phase C Candidate Results

**Companion to:** `reports/OUTCOME_MODEL_RESULTS.md` (Phase B), `reports/OUTCOME_MODEL_TRAINING_PLAN.md`
**Data:** real EPL, 13 seasons. Same expanding-window walk-forward + metrics as Phase B (≈ {r['dixon_coles']['n']} OOS matches).
**No live app model was modified.** Winner saved as a candidate bundle `models_pkl/outcome_v2.pkl` (integration is Phase E).

## Candidate comparison (pooled out-of-sample)

{_table(r, order)}

- **elo_logistic** — Phase B benchmark (1 feature).
- **logistic_form** — Elo diff + causal rolling home/away form & goals ({len(FORM_FEATURES)} features).
- **dixon_coles** — time-decayed bivariate-Poisson goal model (half-life 180d, L2=0.01).
- **bookmaker** — de-margined Bet365 odds: the practical ceiling.

**Winner:** `{winner}` (pooled OOS log loss {candidates[winner]:.3f}). Gap to the bookmaker ceiling: **{gap:+.3f}** log loss.

## Per-season out-of-sample accuracy

{chr(10).join(season_rows)}

## Method notes

- New, reusable modules: `app/services/dixon_coles.py`, `app/services/features_team.py`, `app/services/team_ratings.py`.
- Causal throughout: Elo + rolling form use only prior matches; Dixon-Coles refit per fold on past matches with recency weighting.
- Winner refit on all 13 seasons and pickled as a candidate; it is NOT loaded by the API (Phase E would version it behind `outcome_v2`).
"""
    REPORT.write_text(md)
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
