# /home/sam069/projects/SportyDataFeeder/scripts/backtest_outcome.py
#
# Phase B (reports/OUTCOME_MODEL_TRAINING_PLAN.md): expanding-window walk-forward
# backtest of football outcome BASELINES on real EPL data. No application model
# is trained or modified here — this establishes the honest out-of-sample
# reference (what "good" looks like) that the validation report said was missing.
#
# Baselines, each producing leakage-free P(H), P(D), P(A) per test match:
#   - always_home   : predict Home every time (accuracy reference only)
#   - base_rate     : training-window H/D/A frequencies (a real probabilistic baseline)
#   - elo_logistic  : multinomial logistic on causal Elo difference
#   - bookmaker     : de-margined Bet365 implied probabilities (the ceiling)
#
# Validation: train on seasons[:k], test on season k, for k >= WARMUP. Elo is
# causal regardless of fold; the logistic mapping is refit per fold on past only.
#
# Usage:  python -m scripts.backtest_outcome

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.team_ratings import annotate_pre_match_elo
from scripts.load_historical import load_matches

CLASSES = ["H", "D", "A"]          # fixed probability-vector order
CIDX = {c: i for i, c in enumerate(CLASSES)}
WARMUP_SEASONS = 3                  # seasons used only to warm up before testing
REPORT = Path(__file__).resolve().parents[1] / "reports" / "OUTCOME_MODEL_RESULTS.md"
EPS = 1e-15


# ----------------------------------------------------------------------------- metrics
def _onehot(y: np.ndarray) -> np.ndarray:
    m = np.zeros((len(y), 3))
    for i, c in enumerate(y):
        m[i, CIDX[c]] = 1.0
    return m


def accuracy(proba: np.ndarray, y: np.ndarray) -> float:
    pred = np.array([CLASSES[i] for i in proba.argmax(1)])
    return float((pred == y).mean())


def log_loss(proba: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(proba, EPS, 1.0)
    true_p = p[np.arange(len(y)), [CIDX[c] for c in y]]
    return float(-np.log(true_p).mean())


def brier(proba: np.ndarray, y: np.ndarray) -> float:
    return float(((proba - _onehot(y)) ** 2).sum(1).mean())


def ece(proba: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Confidence ECE: gap between top-class confidence and its accuracy."""
    conf = proba.max(1)
    pred = np.array([CLASSES[i] for i in proba.argmax(1)])
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            total += abs(correct[m].mean() - conf[m].mean()) * m.sum() / len(y)
    return float(total)


# ----------------------------------------------------------------------------- baselines
def p_base_rate(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    rate = train["ftr"].value_counts(normalize=True)
    vec = np.array([rate.get(c, 0.0) for c in CLASSES])
    vec = vec / vec.sum()
    return np.tile(vec, (len(test), 1))


def p_always_home(test: pd.DataFrame) -> np.ndarray:
    vec = np.zeros((len(test), 3))
    vec[:, CIDX["H"]] = 1.0  # degenerate; for accuracy reference
    return vec


def p_elo_logistic(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    model = Pipeline(
        [("scaler", StandardScaler()),
         ("clf", LogisticRegression(max_iter=1000))]
    )
    model.fit(train[["elo_diff"]].values, train["ftr"].values)
    proba = model.predict_proba(test[["elo_diff"]].values)
    # reorder model columns -> fixed CLASSES order
    order = [list(model.classes_).index(c) for c in CLASSES]
    return proba[:, order]


def p_bookmaker(test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """De-margined Bet365 implied probabilities. Returns (proba, valid_mask)."""
    cols = ["B365H", "B365D", "B365A"]
    if not all(c in test.columns for c in cols):
        return np.zeros((len(test), 3)), np.zeros(len(test), bool)
    odds = test[cols].apply(pd.to_numeric, errors="coerce")
    valid = odds.notna().all(axis=1) & (odds > 1.0).all(axis=1)
    inv = 1.0 / odds.where(valid)
    proba = inv.div(inv.sum(axis=1), axis=0).to_numpy()
    proba = np.nan_to_num(proba)
    return proba, valid.to_numpy()


# ----------------------------------------------------------------------------- harness
def run() -> dict:
    df = annotate_pre_match_elo(load_matches())
    seasons = list(dict.fromkeys(df["season"]))  # chronological, de-duped
    test_seasons = seasons[WARMUP_SEASONS:]

    # accumulate pooled predictions per baseline
    pooled = {b: {"p": [], "y": []} for b in ["always_home", "base_rate", "elo_logistic"]}
    book = {"p": [], "y": []}
    per_season_acc = {b: {} for b in ["always_home", "base_rate", "elo_logistic", "bookmaker"]}

    for s in test_seasons:
        train = df[df["season"].isin(seasons[: seasons.index(s)])]
        test = df[df["season"] == s]
        y = test["ftr"].to_numpy()

        preds = {
            "always_home": p_always_home(test),
            "base_rate": p_base_rate(train, test),
            "elo_logistic": p_elo_logistic(train, test),
        }
        for b, p in preds.items():
            pooled[b]["p"].append(p)
            pooled[b]["y"].append(y)
            per_season_acc[b][s] = accuracy(p, y)

        bp, mask = p_bookmaker(test)
        if mask.any():
            book["p"].append(bp[mask])
            book["y"].append(y[mask])
            per_season_acc["bookmaker"][s] = accuracy(bp[mask], y[mask])

    # pooled metrics
    results = {}
    for b in pooled:
        p = np.vstack(pooled[b]["p"])
        y = np.concatenate(pooled[b]["y"])
        results[b] = {
            "n": len(y),
            "accuracy": accuracy(p, y),
            "log_loss": log_loss(p, y),
            "brier": brier(p, y),
            "ece": ece(p, y),
        }
    bp = np.vstack(book["p"])
    by = np.concatenate(book["y"])
    results["bookmaker"] = {
        "n": len(by),
        "accuracy": accuracy(bp, by),
        "log_loss": log_loss(bp, by),
        "brier": brier(bp, by),
        "ece": ece(bp, by),
    }
    return {
        "results": results,
        "per_season_acc": per_season_acc,
        "test_seasons": test_seasons,
        "n_train_warmup": int(df["season"].isin(seasons[:WARMUP_SEASONS]).sum()),
        "n_total": len(df),
    }


def _fmt_table(results: dict) -> str:
    order = ["always_home", "base_rate", "elo_logistic", "bookmaker"]
    rows = ["| Baseline | n | Accuracy | Log loss | Brier | ECE |",
            "|---|---:|---:|---:|---:|---:|"]
    for b in order:
        r = results[b]
        rows.append(f"| {b} | {r['n']} | {r['accuracy']:.3f} | {r['log_loss']:.3f} "
                    f"| {r['brier']:.3f} | {r['ece']:.3f} |")
    return "\n".join(rows)


def _fmt_season_acc(per_season_acc: dict, seasons: list) -> str:
    order = ["always_home", "base_rate", "elo_logistic", "bookmaker"]
    head = "| Season | " + " | ".join(order) + " |"
    sep = "|---|" + "---:|" * len(order)
    rows = [head, sep]
    for s in seasons:
        cells = [f"{per_season_acc[b].get(s, float('nan')):.3f}" for b in order]
        rows.append(f"| {s} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main() -> None:
    out = run()
    r = out["results"]
    print(f"Total matches: {out['n_total']}  |  warm-up matches: {out['n_train_warmup']}")
    print(f"Test seasons ({len(out['test_seasons'])}): {out['test_seasons']}\n")
    print(_fmt_table(r))

    elo_acc = r["elo_logistic"]["accuracy"]
    home_acc = r["always_home"]["accuracy"]
    book_ll = r["bookmaker"]["log_loss"]
    elo_ll = r["elo_logistic"]["log_loss"]

    md = f"""# Outcome Model — Real Walk-Forward Backtest Results (Phase B)

**Companion to:** `reports/MODEL_VALIDATION_REPORT.md`, `reports/OUTCOME_MODEL_TRAINING_PLAN.md`
**Data:** {out['n_total']} real EPL matches, 13 seasons. **Walk-forward, expanding window**, strictly causal.
**Tested out-of-sample on {len(out['test_seasons'])} seasons** (≈ {r['elo_logistic']['n']} matches); first {WARMUP_SEASONS} seasons warm up Elo only.
**No application model was trained or modified.** These are baselines establishing the real reference.

## Pooled out-of-sample metrics

{_fmt_table(r)}

Lower log loss / Brier / ECE = better; higher accuracy = better.

## How to read this

- **always_home** — predict Home every match. Accuracy {home_acc:.3f}. The bar the current production model *ties* (and never beats). Its huge log loss ({r['always_home']['log_loss']:.1f}) is expected: a hard 0/1 prediction is punished severely whenever the result isn't Home — a reminder that *probabilities*, not hard calls, are what matter here.
- **base_rate** — training H/D/A frequencies. A model with no skill should not beat this on log loss.
- **elo_logistic** — multinomial logistic on causal Elo difference (one feature). Accuracy {elo_acc:.3f}, log loss {elo_ll:.3f}.
- **bookmaker** — de-margined Bet365 odds: the practical ceiling. Log loss {book_ll:.3f}.

**Headline:** a single causal Elo feature already {'beats' if elo_acc > home_acc + 1e-9 else 'matches'} always-home on accuracy and {'beats' if elo_ll < r['base_rate']['log_loss'] else 'does not beat'} the base-rate on log loss, out-of-sample, over ~{r['elo_logistic']['n']} matches — versus the current model's *zero* lift on 15 in-sample matches. The gap between elo_logistic and bookmaker ({elo_ll:.3f} vs {book_ll:.3f} log loss) is the headroom the Phase C model (Dixon-Coles + richer features) aims to close.

## Per-season out-of-sample accuracy

{_fmt_season_acc(out['per_season_acc'], out['test_seasons'])}

## Method notes (reproducibility)

- Loader: `scripts/load_historical.py` (read-only, no DB).
- Causal Elo: `app/services/team_ratings.py` (K=20, home advantage=65, 25% season mean-reversion). Pre-match ratings only.
- Expanding window: train = all seasons before the test season; Elo updated forward in time; logistic mapping refit per fold on past data only.
- Metrics: accuracy, multiclass log loss, multiclass Brier, confidence-ECE (10 bins).
- Bookmaker metrics computed only over rows with valid Bet365 odds (n shown).
"""
    REPORT.write_text(md)
    print(f"\nWrote {REPORT}")


if __name__ == "__main__":
    main()
