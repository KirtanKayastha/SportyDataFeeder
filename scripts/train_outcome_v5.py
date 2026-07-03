# /home/sam069/projects/SportyDataFeeder/scripts/train_outcome_v5.py
#
# Step 5 of reports/MODEL_IMPROVEMENT_PLAN.md (multi-league pooling): pool the
# English Championship (E1) into the Elo/SoT pass so promoted teams enter the
# EPL with a real earned rating instead of the 1500 cold start. Evaluation is
# unchanged: same expanding-window folds, same metrics, and the TEST set is
# always EPL matches only (the app predicts EPL) — pooling only changes what
# the ratings/form/mapping layers get to learn from.
#
# Stage 1  mini grid around the v4 optimum on the pooled data (pooling adds
#          ~7k matches, which can shift the best K/SR).
# Stage 2  walk-forward comparison of:
#            v4_incumbent    — E0-only Elo+SoT (production outcome_v4_elo_sot)
#            pooled_elo      — Elo/SoT over E0+E1; logistic fit on prior E0 rows
#            pooled_all      — as above + logistic fit on ALL prior rows with a
#                              top-flight dummy feature
#            bookmaker       — de-margined Bet365 ceiling (E0 rows)
#
# Output: reports/OUTCOME_MODEL_V5.md. NOTHING is wired into the live app here.
#
# Usage:  python -m scripts.fetch_championship && python -m scripts.train_outcome_v5

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.features_team import annotate_shot_form
from app.services.team_ratings import annotate_pre_match_elo
from scripts.backtest_outcome import WARMUP_SEASONS, accuracy, brier, ece, log_loss, p_bookmaker
from scripts.load_historical import load_matches
from scripts.train_outcome_v3 import _fit_logistic, _proba

REPORT = Path(__file__).resolve().parents[1] / "reports" / "OUTCOME_MODEL_V5.md"

# The shipped v4 production config (scripts/finalize_outcome_v2.py).
V4 = {"k": 40.0, "home_advantage": 65.0, "season_regression": 0.10, "mov": "fte"}
FEATURES = ["elo_diff", "sot_net_diff"]
GRID_K = [30.0, 40.0, 50.0]
GRID_SR = [0.05, 0.10, 0.20]


def _annotate(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    return annotate_shot_form(annotate_pre_match_elo(
        raw, k=cfg["k"], home_advantage=cfg["home_advantage"],
        season_regression=cfg["season_regression"], mov=cfg["mov"]))


def _epl_folds(df: pd.DataFrame):
    """Expanding-window folds; test = EPL rows of the season, train = all prior
    rows (caller filters divisions as its candidate requires)."""
    seasons = list(dict.fromkeys(df[df["Div"] == "E0"]["season"]))
    for s in seasons[WARMUP_SEASONS:]:
        prior = df[df["season"].isin(seasons[: seasons.index(s)])]
        test = df[(df["season"] == s) & (df["Div"] == "E0")]
        yield s, prior, test


def _walkforward(df: pd.DataFrame, features: list[str], train_divs: list[str]):
    ps, ys, acc = [], [], {}
    for s, prior, test in _epl_folds(df):
        train = prior[prior["Div"].isin(train_divs)]
        p = _proba(_fit_logistic(train, features), test, features)
        y = test["ftr"].to_numpy()
        ps.append(p)
        ys.append(y)
        acc[s] = accuracy(p, y)
    p, y = np.vstack(ps), np.concatenate(ys)
    return {"n": len(y), "accuracy": accuracy(p, y), "log_loss": log_loss(p, y),
            "brier": brier(p, y), "ece": ece(p, y), "per_season_acc": acc}


def main() -> None:
    raw_epl = load_matches()
    raw_all = load_matches(include_championship=True)
    n_e1 = int((raw_all["Div"] == "E1").sum())
    print(f"EPL matches: {len(raw_epl)} | pooled with Championship: {len(raw_all)} (+{n_e1} E1)")

    # Stage 1: mini grid on the pooled data (EPL-only test, E0-trained logistic).
    print(f"Stage 1: pooled-Elo mini grid ({len(GRID_K) * len(GRID_SR)} configs)")
    best = None
    for k, sr in product(GRID_K, GRID_SR):
        cfg = {**V4, "k": k, "season_regression": sr}
        r = _walkforward(_annotate(raw_all, cfg), FEATURES, ["E0"])
        marker = ""
        if best is None or r["log_loss"] < best["log_loss"]:
            best = {"k": k, "season_regression": sr, "log_loss": r["log_loss"]}
            marker = "  <- best"
        print(f"  K={k:.0f} SR={sr:.2f} -> LL {r['log_loss']:.5f}{marker}")
    pooled_cfg = {**V4, "k": best["k"], "season_regression": best["season_regression"]}

    # Stage 2: candidates.
    print("Stage 2: candidate comparison")
    df_v4 = _annotate(raw_epl, V4)
    df_pool = _annotate(raw_all, pooled_cfg)
    df_pool["div_e0"] = (df_pool["Div"] == "E0").astype(float)

    results, per_season_acc = {}, {}
    runs = {
        "v4_incumbent": (df_v4, FEATURES, ["E0"]),
        "pooled_elo": (df_pool, FEATURES, ["E0"]),
        "pooled_all": (df_pool, FEATURES + ["div_e0"], ["E0", "E1"]),
    }
    for name, (df, feats, train_divs) in runs.items():
        r = _walkforward(df, feats, train_divs)
        per_season_acc[name] = r.pop("per_season_acc")
        results[name] = r
        print(f"  {name}: LL {r['log_loss']:.5f}  acc {r['accuracy']:.4f}")

    # Bookmaker ceiling over the same EPL test rows.
    bp_all, by_all, book_acc = [], [], {}
    for s, _, test in _epl_folds(df_v4):
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
    order = list(runs) + ["bookmaker"]

    rows = ["| Model | n | Accuracy | Log loss | Brier | ECE |",
            "|---|---:|---:|---:|---:|---:|"]
    for n in order:
        r = results[n]
        rows.append(f"| {n} | {r['n']} | {r['accuracy']:.3f} | {r['log_loss']:.3f} "
                    f"| {r['brier']:.3f} | {r['ece']:.3f} |")
    table = "\n".join(rows)
    print("\n" + table)

    candidates = {n: results[n]["log_loss"] for n in runs}
    winner = min(candidates, key=candidates.get)
    inc_ll = results["v4_incumbent"]["log_loss"]
    book_ll = results["bookmaker"]["log_loss"]
    print(f"\nWinner: {winner} (LL {candidates[winner]:.4f}) | v4 {inc_ll:.4f} | bookmaker {book_ll:.4f}")

    test_seasons = [s for s, _, _ in _epl_folds(df_v4)]
    season_rows = ["| Season | " + " | ".join(order) + " |", "|---|" + "---:|" * len(order)]
    for s in test_seasons:
        cells = [f"{per_season_acc[n].get(s, float('nan')):.3f}" for n in order]
        season_rows.append(f"| {s} | " + " | ".join(cells) + " |")

    md = f"""# Outcome Model v5 — Multi-League Pooling (EPL + Championship)

**Companion to:** `reports/MODEL_IMPROVEMENT_PLAN.md` (step 5), `reports/OUTCOME_MODEL_V4.md`.
**Data:** {len(raw_epl)} EPL + {n_e1} Championship matches, 13 seasons
(`scripts/fetch_championship.py`). Test folds are EPL-only — pooling changes
only what the Elo/form/mapping layers learn from. Walk-forward, strictly causal.
**No live app model was modified by this script.**

## Stage 1 — pooled-Elo mini grid

Best pooled config: K={pooled_cfg['k']:.0f}, SR={pooled_cfg['season_regression']:.2f}
(mov=fte, HA=65) → LL {best['log_loss']:.5f}.

## Stage 2 — candidate comparison (pooled out-of-sample, EPL test matches)

{table}

- **v4_incumbent** — production `outcome_v4_elo_sot` (E0-only Elo + SoT form).
- **pooled_elo** — Elo/SoT computed over E0+E1 (promoted teams arrive with an
  earned rating; relegated teams keep theirs); logistic fit on prior E0 rows.
- **pooled_all** — as above, logistic fit on all prior rows (both divisions)
  with a top-flight dummy.
- **bookmaker** — de-margined Bet365 odds on the same EPL rows.

**Winner:** `{winner}` (pooled OOS log loss {candidates[winner]:.4f}); v4 incumbent
{inc_ll:.4f}; bookmaker {book_ll:.4f}. Gap to ceiling: **{candidates[winner] - book_ll:+.4f}**
(v4 was {inc_ll - book_ll:+.4f}).

## Per-season out-of-sample accuracy (EPL)

{chr(10).join(season_rows)}
"""
    REPORT.write_text(md)
    print(f"\nWrote {REPORT}")


if __name__ == "__main__":
    main()
