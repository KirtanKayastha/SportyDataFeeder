# /home/sam069/projects/SportyDataFeeder/scripts/train_outcome_v4.py
#
# Step 3 of reports/MODEL_IMPROVEMENT_PLAN.md: football margin-of-victory Elo
# and shots-on-target rolling form. Same expanding-window walk-forward folds
# and metrics as Phase B/C and v3 (scripts/backtest_outcome.py), so results are
# comparable and leakage-free.
#
# Stage 1  grid-search the MOV Elo update (mov in {wfe, fte} x K x SR; HA fixed
#          at 65 — v3 showed it is absorbed by the logistic intercept).
# Stage 2  walk-forward comparison of:
#            elo_v3          — production config (K=20, HA=65, SR=0.10, classic)
#            elo_mov         — stage-1 best MOV config, single elo_diff feature
#            elo_shots_net   — v3 Elo + compact SoT form (sot_net_diff)
#            elo_shots_full  — v3 Elo + 4 raw SoT for/against features
#            elo_mov_shots   — best MOV Elo + sot_net_diff
#            bookmaker       — de-margined Bet365 ceiling
#
# Output: reports/OUTCOME_MODEL_V4.md. NOTHING is wired into the live app here —
# integration happens in scripts/finalize_outcome_v2.py once a winner is
# confirmed.
#
# Usage:  python -m scripts.train_outcome_v4

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.features_team import annotate_shot_form
from app.services.team_ratings import annotate_pre_match_elo
from scripts.backtest_outcome import accuracy, brier, ece, log_loss, p_bookmaker
from scripts.train_outcome_v3 import _fit_logistic, _folds, _proba

REPORT = Path(__file__).resolve().parents[1] / "reports" / "OUTCOME_MODEL_V4.md"

# The shipped v3 production config (see scripts/finalize_outcome_v2.py).
INCUMBENT = {"k": 20.0, "home_advantage": 65.0, "season_regression": 0.10, "mov": None}
HA = 65.0
GRID_MOV = ["wfe", "fte"]
GRID_K = [10.0, 15.0, 20.0, 25.0, 30.0, 40.0]
GRID_SR = [0.0, 0.10, 0.25, 0.40]

SOT_NET = ["sot_net_diff"]
SOT_FULL = ["home_sot_for", "home_sot_against", "away_sot_for", "away_sot_against"]


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
def tune_mov(raw: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    rows, best = [], None
    for mov, k, sr in product(GRID_MOV, GRID_K, GRID_SR):
        df = annotate_pre_match_elo(raw, k=k, home_advantage=HA,
                                    season_regression=sr, mov=mov)
        ll = _walkforward(df, ["elo_diff"])["log_loss"]
        rows.append({"mov": mov, "k": k, "season_regression": sr, "log_loss": ll})
        if best is None or ll < best["log_loss"]:
            best = {"k": k, "home_advantage": HA, "season_regression": sr,
                    "mov": mov, "log_loss": ll}
            print(f"  new best: mov={mov} K={k:.0f} SR={sr:.2f} -> LL {ll:.5f}")
    grid = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
    return best, grid


# ------------------------------------------------------------------ stage 2
def compare(raw: pd.DataFrame, tuned_mov: dict) -> dict:
    def annotated(cfg):
        df = annotate_pre_match_elo(raw, k=cfg["k"], home_advantage=cfg["home_advantage"],
                                    season_regression=cfg["season_regression"], mov=cfg["mov"])
        return annotate_shot_form(df)

    df_inc = annotated(INCUMBENT)
    df_mov = annotated(tuned_mov)

    runs = {
        "elo_v3": (df_inc, ["elo_diff"]),
        "elo_mov": (df_mov, ["elo_diff"]),
        "elo_shots_net": (df_inc, ["elo_diff"] + SOT_NET),
        "elo_shots_full": (df_inc, ["elo_diff"] + SOT_FULL),
        "elo_mov_shots": (df_mov, ["elo_diff"] + SOT_NET),
    }
    results, per_season_acc = {}, {}
    for name, (df, feats) in runs.items():
        r = _walkforward(df, feats)
        per_season_acc[name] = r.pop("per_season_acc")
        results[name] = r
        print(f"  {name}: LL {r['log_loss']:.5f}  acc {r['accuracy']:.4f}")

    # bookmaker ceiling on the same folds
    bp_all, by_all, book_acc = [], [], {}
    for s, _, test in _folds(df_inc):
        bp, mask = p_bookmaker(test)
        if mask.any():
            y = test["ftr"].to_numpy()
            bp_all.append(bp[mask])
            by_all.append(y[mask])
            book_acc[s] = accuracy(bp[mask], y[mask])
    bp, by = np.vstack(bp_all), np.concatenate(by_all)
    results["bookmaker"] = {"n": len(by), "accuracy": accuracy(bp, by),
                            "log_loss": log_loss(bp, by), "brier": brier(bp, by),
                            "ece": ece(bp, by)}
    per_season_acc["bookmaker"] = book_acc

    return {"results": results, "per_season_acc": per_season_acc,
            "names": list(runs) + ["bookmaker"],
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
    from scripts.load_historical import load_matches
    raw = load_matches()

    n_cfg = len(GRID_MOV) * len(GRID_K) * len(GRID_SR)
    print(f"Stage 1: football MOV Elo grid search ({n_cfg} configs)")
    tuned_mov, grid = tune_mov(raw)
    print(f"Best MOV: mov={tuned_mov['mov']} K={tuned_mov['k']:.0f} "
          f"SR={tuned_mov['season_regression']:.2f} -> LL {tuned_mov['log_loss']:.5f}")

    print("Stage 2: candidate comparison")
    out = compare(raw, tuned_mov)
    r, order = out["results"], out["names"]
    print("\n" + _table(r, order))

    candidates = {n: r[n]["log_loss"] for n in order if n != "bookmaker"}
    winner = min(candidates, key=candidates.get)
    inc_ll = r["elo_v3"]["log_loss"]
    book_ll = r["bookmaker"]["log_loss"]
    print(f"\nWinner: {winner} (LL {candidates[winner]:.4f}) | incumbent v3 {inc_ll:.4f} "
          f"| bookmaker {book_ll:.4f}")

    season_rows = ["| Season | " + " | ".join(order) + " |",
                   "|---|" + "---:|" * len(order)]
    for s in out["test_seasons"]:
        cells = [f"{out['per_season_acc'][n].get(s, float('nan')):.3f}" for n in order]
        season_rows.append(f"| {s} | " + " | ".join(cells) + " |")

    md = f"""# Outcome Model v4 — Football MOV Elo + Shots-on-Target Form

**Companion to:** `reports/MODEL_IMPROVEMENT_PLAN.md` (step 3), `reports/OUTCOME_MODEL_V3.md`.
**Data:** real EPL, 13 seasons. Same expanding-window walk-forward + metrics as Phase B/C/v3.
**No live app model was modified by this script.**

## Stage 1 — MOV Elo grid search

Grid: mov ∈ {{wfe, fte}}, K ∈ {GRID_K}, SR ∈ {GRID_SR} (HA fixed at {HA:.0f} — absorbed
by the logistic intercept, see v3). Objective = pooled walk-forward log loss.

**Best:** mov={tuned_mov['mov']}, K={tuned_mov['k']:.0f}, SR={tuned_mov['season_regression']:.2f}
→ LL {tuned_mov['log_loss']:.5f} (production v3 classic Elo → {inc_ll:.5f}).

Top 10 configs:

```
{grid.head(10).to_string(index=False)}
```

## Stage 2 — candidate comparison (pooled out-of-sample)

{_table(r, order)}

- **elo_v3** — the shipped production config (K=20, HA=65, SR=0.10, classic Elo).
- **elo_mov** — stage-1 best margin-of-victory Elo (wfe = World Football Elo
  goal-difference multiplier; fte = FiveThirtyEight favourite-dampened multiplier).
- **elo_shots_net / elo_shots_full** — v3 Elo + causal rolling shots-on-target form
  (`app/services/features_team.py:annotate_shot_form`, window 10, league prior {4.3}).
- **elo_mov_shots** — best MOV Elo + compact SoT feature.
- **bookmaker** — de-margined Bet365 odds: the practical ceiling.

**Winner:** `{winner}` (pooled OOS log loss {candidates[winner]:.4f}); incumbent v3
{inc_ll:.4f}; bookmaker {book_ll:.4f}. Gap to ceiling: **{candidates[winner] - book_ll:+.4f}**
(v3 was {inc_ll - book_ll:+.4f}).

## Per-season out-of-sample accuracy

{chr(10).join(season_rows)}

## Method notes

- MOV variants live in `app/services/team_ratings.py:EloModel._mov_multiplier`.
- SoT form is any-venue, last-10-matches mean for/against, updated strictly after
  each row is emitted; cold starts get the league prior — no leakage.
"""
    REPORT.write_text(md)
    print(f"\nWrote {REPORT}")


if __name__ == "__main__":
    main()
