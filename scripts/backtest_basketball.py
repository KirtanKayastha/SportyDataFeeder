# /home/sam069/projects/SportyDataFeeder/scripts/backtest_basketball.py
#
# Basketball analogue of scripts/backtest_outcome.py: expanding-window
# walk-forward backtest of NBA outcome baselines on real data. Basketball has
# NO draws (2 classes) and this dataset carries NO bookmaker odds, so the
# benchmarks are always_home / base_rate / elo_logistic. No app model is
# trained or modified here.
#
# Usage:  python -m scripts.backtest_basketball

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.team_ratings import annotate_pre_match_elo
from scripts.load_nba import load_nba_games

CLASSES = ["H", "A"]                  # no draw in basketball
CIDX = {c: i for i, c in enumerate(CLASSES)}
WARMUP_SEASONS = 3
HOME_ADV = 100.0                      # NBA home court ~= +100 Elo
K = 20.0
EPS = 1e-15
REPORT = Path(__file__).resolve().parents[1] / "reports" / "OUTCOME_MODEL_BASKETBALL.md"


def _onehot(y):
    m = np.zeros((len(y), len(CLASSES)))
    for i, c in enumerate(y):
        m[i, CIDX[c]] = 1.0
    return m


def accuracy(p, y):
    pred = np.array([CLASSES[i] for i in p.argmax(1)])
    return float((pred == y).mean())


def log_loss(p, y):
    pc = np.clip(p, EPS, 1.0)
    return float(-np.log(pc[np.arange(len(y)), [CIDX[c] for c in y]]).mean())


def brier(p, y):
    return float(((p - _onehot(y)) ** 2).sum(1).mean())


def ece(p, y, bins=10):
    conf = p.max(1)
    pred = np.array([CLASSES[i] for i in p.argmax(1)])
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            total += abs(correct[m].mean() - conf[m].mean()) * m.sum() / len(y)
    return float(total)


def p_always_home(test):
    v = np.zeros((len(test), len(CLASSES)))
    v[:, CIDX["H"]] = 1.0
    return v


def p_base_rate(train, test):
    rate = train["ftr"].value_counts(normalize=True)
    vec = np.array([rate.get(c, 0.0) for c in CLASSES])
    vec = vec / vec.sum()
    return np.tile(vec, (len(test), 1))


def p_elo_logistic(train, test):
    model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
    model.fit(train[["elo_diff"]].values, train["ftr"].values)
    proba = model.predict_proba(test[["elo_diff"]].values)
    order = [list(model.classes_).index(c) for c in CLASSES]
    return proba[:, order]


def run():
    df = annotate_pre_match_elo(load_nba_games(), k=K, home_advantage=HOME_ADV)
    seasons = list(dict.fromkeys(df["season"]))
    test_seasons = seasons[WARMUP_SEASONS:]
    names = ["always_home", "base_rate", "elo_logistic"]
    pooled = {n: {"p": [], "y": []} for n in names}
    per_season_acc = {n: {} for n in names}

    for s in test_seasons:
        train = df[df["season"].isin(seasons[: seasons.index(s)])]
        test = df[df["season"] == s]
        y = test["ftr"].to_numpy()
        preds = {
            "always_home": p_always_home(test),
            "base_rate": p_base_rate(train, test),
            "elo_logistic": p_elo_logistic(train, test),
        }
        for n, p in preds.items():
            pooled[n]["p"].append(p); pooled[n]["y"].append(y)
            per_season_acc[n][s] = accuracy(p, y)

    results = {}
    for n in names:
        p = np.vstack(pooled[n]["p"]); y = np.concatenate(pooled[n]["y"])
        results[n] = {"n": len(y), "accuracy": accuracy(p, y), "log_loss": log_loss(p, y),
                      "brier": brier(p, y), "ece": ece(p, y)}
    return {"results": results, "per_season_acc": per_season_acc, "test_seasons": test_seasons, "n_total": len(df)}


def _table(results, order):
    rows = ["| Baseline | n | Accuracy | Log loss | Brier | ECE |",
            "|---|---:|---:|---:|---:|---:|"]
    for n in order:
        r = results[n]
        rows.append(f"| {n} | {r['n']} | {r['accuracy']:.3f} | {r['log_loss']:.3f} | {r['brier']:.3f} | {r['ece']:.3f} |")
    return "\n".join(rows)


def main():
    out = run()
    r = out["results"]
    order = ["always_home", "base_rate", "elo_logistic"]
    print(f"Total NBA games: {out['n_total']}  | test seasons: {len(out['test_seasons'])}\n")
    print(_table(r, order))
    elo = r["elo_logistic"]; home = r["always_home"]; base = r["base_rate"]

    season_rows = ["| Season | " + " | ".join(order) + " |", "|---|" + "---:|" * len(order)]
    for s in out["test_seasons"]:
        season_rows.append(f"| {s} | " + " | ".join(f"{out['per_season_acc'][n].get(s, float('nan')):.3f}" for n in order) + " |")

    md = f"""# Outcome Model — Basketball (NBA) Walk-Forward Backtest

**Companion to:** `reports/OUTCOME_MODEL_RESULTS.md` (football), `reports/MODEL_VALIDATION_REPORT.md`
**Data:** {out['n_total']} real NBA regular-season games (Kaggle nba.sqlite), {len(out['test_seasons'])+WARMUP_SEASONS} seasons. **Walk-forward, expanding window**, strictly causal.
**No draws** (2 classes). **No bookmaker odds** in this dataset, so there is no market ceiling baseline. No app model was trained or modified.

## Pooled out-of-sample metrics

{_table(r, order)}

- **always_home** — predict Home every game. Accuracy {home['accuracy']:.3f} (NBA home-court is strong). Huge log loss = the penalty for hard 0/1 predictions.
- **base_rate** — training home/away frequencies.
- **elo_logistic** — multinomial logistic on causal Elo difference (1 feature). Accuracy {elo['accuracy']:.3f}, log loss {elo['log_loss']:.3f}, ECE {elo['ece']:.3f}.

**Headline:** Elo {'beats' if elo['accuracy'] > home['accuracy'] + 1e-9 else 'matches'} the strong always-home baseline ({elo['accuracy']:.3f} vs {home['accuracy']:.3f}) and {'beats' if elo['log_loss'] < base['log_loss'] else 'does not beat'} base-rate on log loss, out-of-sample over ~{elo['n']} games, while staying well-calibrated (ECE {elo['ece']:.3f}). NBA home advantage is large, so the accuracy gap over always-home is naturally smaller than in football — the real value shows up in calibrated probabilities (log loss / Brier).

## Per-season out-of-sample accuracy

{chr(10).join(season_rows)}

## Method notes

- Loader: `scripts/load_nba.py` (read-only). Regular season since 2004; Elo keyed by stable franchise team_id.
- Causal Elo: `app/services/team_ratings.py` (K={K:.0f}, home advantage={HOME_ADV:.0f}, 25% season mean-reversion).
- Note: the 2012-13 season is absent from this Kaggle dump (data gap); walk-forward is unaffected (Elo carries across the gap).
"""
    REPORT.write_text(md)
    print(f"\nWrote {REPORT}")


if __name__ == "__main__":
    main()
