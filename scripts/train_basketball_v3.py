# /home/sam069/projects/SportyDataFeeder/scripts/train_basketball_v3.py
#
# Step 2 of reports/MODEL_IMPROVEMENT_PLAN.md: improve the NBA outcome model on
# the REFRESHED dataset (Kaggle dump + basketball-reference recent seasons via
# scripts/fetch_nba_recent.py) with (a) tuned Elo hyperparameters, (b) a
# margin-of-victory Elo update (FiveThirtyEight multiplier), and (c) causal
# rest/back-to-back schedule features. Judged on the same expanding-window
# walk-forward folds and 2-class metrics as scripts/backtest_basketball.py.
#
# Stage 1  grid-search (mov, K, home_advantage, season_regression) minimising
#          pooled walk-forward log loss of the 1-feature elo_logistic model.
# Stage 2  walk-forward comparison of:
#            elo_incumbent — K=20, HA=100, SR=0.25, classic Elo (production)
#            elo_tuned     — best classic-Elo config from stage 1
#            elo_mov       — best MOV-Elo config from stage 1
#            elo_rest      — elo_tuned features + rest_diff/b2b flags
#            elo_mov_rest  — elo_mov features + rest_diff/b2b flags
#
# Output: reports/OUTCOME_MODEL_BASKETBALL_V3.md. NOTHING is wired into the
# live app here — integration happens in scripts/finalize_outcome_basketball.py
# once a winner is confirmed.
#
# Usage:  python -m scripts.fetch_nba_recent && python -m scripts.train_basketball_v3

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.features_team import annotate_rest_days
from app.services.team_ratings import annotate_pre_match_elo
from scripts.backtest_basketball import (
    CLASSES, WARMUP_SEASONS, accuracy, brier, ece, log_loss,
)
from scripts.load_nba import load_nba_games

REPORT = Path(__file__).resolve().parents[1] / "reports" / "OUTCOME_MODEL_BASKETBALL_V3.md"

INCUMBENT = {"k": 20.0, "home_advantage": 100.0, "season_regression": 0.25, "mov": None}
GRID_MOV = [None, "fte"]
GRID_K = [10.0, 15.0, 20.0, 25.0, 30.0, 40.0]
GRID_HA = [60.0, 100.0, 140.0]
GRID_SR = [0.0, 0.10, 0.25, 0.40]
REST_FEATURES = ["rest_diff", "home_b2b", "away_b2b"]


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
        yield s, df[df["season"].isin(seasons[: seasons.index(s)])], df[df["season"] == s]


def _walkforward(df: pd.DataFrame, features: list[str]):
    ps, ys, acc = [], [], {}
    for s, train, test in _folds(df):
        p = _proba(_fit_logistic(train, features), test, features)
        y = test["ftr"].to_numpy()
        ps.append(p)
        ys.append(y)
        acc[s] = accuracy(p, y)
    p, y = np.vstack(ps), np.concatenate(ys)
    return {"n": len(y), "accuracy": accuracy(p, y), "log_loss": log_loss(p, y),
            "brier": brier(p, y), "ece": ece(p, y), "per_season_acc": acc}


# ------------------------------------------------------------------ stage 1
def tune(raw: pd.DataFrame) -> tuple[dict, dict, pd.DataFrame]:
    """Return (best classic config, best MOV config, full grid frame)."""
    rows = []
    best = {None: None, "fte": None}
    for mov, k, ha, sr in product(GRID_MOV, GRID_K, GRID_HA, GRID_SR):
        df = annotate_pre_match_elo(raw, k=k, home_advantage=ha,
                                    season_regression=sr, mov=mov)
        ll = _walkforward(df, ["elo_diff"])["log_loss"]
        row = {"mov": mov or "none", "k": k, "home_advantage": ha,
               "season_regression": sr, "log_loss": ll}
        rows.append(row)
        if best[mov] is None or ll < best[mov]["log_loss"]:
            best[mov] = {"k": k, "home_advantage": ha, "season_regression": sr,
                         "mov": mov, "log_loss": ll}
            print(f"  new best ({row['mov']}): K={k:.0f} HA={ha:.0f} SR={sr:.2f} -> LL {ll:.5f}")
    grid = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
    return best[None], best["fte"], grid


# ------------------------------------------------------------------ stage 2
def compare(raw: pd.DataFrame, tuned: dict, tuned_mov: dict) -> dict:
    def annotated(cfg):
        df = annotate_pre_match_elo(raw, k=cfg["k"], home_advantage=cfg["home_advantage"],
                                    season_regression=cfg["season_regression"], mov=cfg["mov"])
        return annotate_rest_days(df)

    df_inc = annotated(INCUMBENT)
    df_tun = annotated(tuned)
    df_mov = annotated(tuned_mov)

    runs = {
        "elo_incumbent": (df_inc, ["elo_diff"]),
        "elo_tuned": (df_tun, ["elo_diff"]),
        "elo_mov": (df_mov, ["elo_diff"]),
        "elo_rest": (df_tun, ["elo_diff"] + REST_FEATURES),
        "elo_mov_rest": (df_mov, ["elo_diff"] + REST_FEATURES),
    }
    results, per_season_acc = {}, {}
    for name, (df, feats) in runs.items():
        r = _walkforward(df, feats)
        per_season_acc[name] = r.pop("per_season_acc")
        results[name] = r
        print(f"  {name}: LL {r['log_loss']:.5f}  acc {r['accuracy']:.4f}")
    return {"results": results, "per_season_acc": per_season_acc,
            "names": list(runs), "test_seasons": [s for s, _, _ in _folds(df_inc)]}


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
    raw = load_nba_games()
    seasons = list(dict.fromkeys(raw["season"]))
    print(f"Loaded {len(raw)} games, {len(seasons)} seasons "
          f"({seasons[0]}..{seasons[-1]})")

    n_cfg = len(GRID_MOV) * len(GRID_K) * len(GRID_HA) * len(GRID_SR)
    print(f"Stage 1: Elo grid search ({n_cfg} configs)")
    tuned, tuned_mov, grid = tune(raw)
    print(f"Best classic: K={tuned['k']:.0f} HA={tuned['home_advantage']:.0f} "
          f"SR={tuned['season_regression']:.2f} -> LL {tuned['log_loss']:.5f}")
    print(f"Best MOV:     K={tuned_mov['k']:.0f} HA={tuned_mov['home_advantage']:.0f} "
          f"SR={tuned_mov['season_regression']:.2f} -> LL {tuned_mov['log_loss']:.5f}")

    print("Stage 2: candidate comparison")
    out = compare(raw, tuned, tuned_mov)
    r, order = out["results"], out["names"]
    print("\n" + _table(r, order))

    winner = min(order, key=lambda n: r[n]["log_loss"])
    inc = r["elo_incumbent"]
    print(f"\nWinner: {winner} (LL {r[winner]['log_loss']:.4f}) | incumbent {inc['log_loss']:.4f}")

    season_rows = ["| Season | " + " | ".join(order) + " |",
                   "|---|" + "---:|" * len(order)]
    for s in out["test_seasons"]:
        cells = [f"{out['per_season_acc'][n].get(s, float('nan')):.3f}" for n in order]
        season_rows.append(f"| {s} | " + " | ".join(cells) + " |")

    md = f"""# Outcome Model — Basketball v3 (Refreshed Data + MOV Elo + Rest)

**Companion to:** `reports/MODEL_IMPROVEMENT_PLAN.md` (step 2),
`reports/OUTCOME_MODEL_BASKETBALL.md` (baseline on the stale dump).
**Data:** {len(raw)} real NBA regular-season games, {len(seasons)} seasons
({seasons[0]}..{seasons[-1]}) — Kaggle dump + basketball-reference refresh
(`scripts/fetch_nba_recent.py`). Walk-forward, expanding window, strictly causal.
**No live app model was modified by this script.**

## Stage 1 — Elo hyperparameter grid search

Grid: mov ∈ {{none, fte}}, K ∈ {GRID_K}, HA ∈ {GRID_HA}, SR ∈ {GRID_SR};
objective = pooled walk-forward log loss of the 1-feature elo_logistic model.

- **Best classic Elo:** K={tuned['k']:.0f}, HA={tuned['home_advantage']:.0f}, SR={tuned['season_regression']:.2f} → LL {tuned['log_loss']:.5f}
- **Best MOV Elo (FiveThirtyEight multiplier):** K={tuned_mov['k']:.0f}, HA={tuned_mov['home_advantage']:.0f}, SR={tuned_mov['season_regression']:.2f} → LL {tuned_mov['log_loss']:.5f}

Top 10 configs:

```
{grid.head(10).to_string(index=False)}
```

## Stage 2 — candidate comparison (pooled out-of-sample)

{_table(r, order)}

- **elo_incumbent** — the production bundle config (K=20, HA=100, SR=0.25, classic Elo).
- **elo_tuned / elo_mov** — stage-1 winners (single elo_diff feature).
- **elo_rest / elo_mov_rest** — + causal `rest_diff`, `home_b2b`, `away_b2b`
  (`app/services/features_team.py:annotate_rest_days`).

**Winner:** `{winner}` (pooled OOS log loss {r[winner]['log_loss']:.4f} vs incumbent
{inc['log_loss']:.4f}; accuracy {r[winner]['accuracy']:.3f} vs {inc['accuracy']:.3f}).

## Per-season out-of-sample accuracy

{chr(10).join(season_rows)}

## Method notes

- MOV Elo: `EloModel(mov="fte")` — update multiplier ((margin+3)^0.8)/(7.5+0.006·winner_diff),
  the FiveThirtyEight NBA formulation (favourite-blowout dampening).
- Rest features are computed from each team's previous game date only (capped at
  {5} days; season openers get the cap) — no schedule lookahead.
- 2012-13 is absent from the Kaggle dump (known gap); Elo carries across it.
"""
    REPORT.write_text(md)
    print(f"\nWrote {REPORT}")


if __name__ == "__main__":
    main()
