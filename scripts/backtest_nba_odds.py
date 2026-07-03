# /home/sam069/projects/SportyDataFeeder/scripts/backtest_nba_odds.py
#
# Step 5 of reports/MODEL_IMPROVEMENT_PLAN.md (NBA odds baseline): the
# basketball backtests had no market ceiling, so 0.609 pooled log loss floated
# without context. This joins historical closing MONEYLINES to our games and
# scores de-margined market probabilities against the production elo_mov model
# on exactly the same games — the NBA analogue of football's Bet365 column.
#
# Odds source: historical-datas/nba_odds_10y.json — closing lines for 13,903
# regular-season games, seasons 2011-12..2021-22, mirrored from
# https://github.com/flancast90/sportsbookreview-scraper (sportsbookreview
# archive). Games are matched by date + both franchises (nicknames like
# "Hornets" are era-ambiguous, so a match requires BOTH teams to agree).
#
# Usage:  python -m scripts.backtest_nba_odds

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.team_ratings import annotate_pre_match_elo
from scripts.backtest_basketball import CLASSES, WARMUP_SEASONS, accuracy, brier, ece, log_loss
from scripts.load_nba import load_nba_games
from scripts.train_basketball_v3 import _fit_logistic, _folds, _proba

ODDS_JSON = Path(__file__).resolve().parents[1] / "historical-datas" / "nba_odds_10y.json"
REPORT = Path(__file__).resolve().parents[1] / "reports" / "OUTCOME_MODEL_NBA_ODDS.md"

# Production basketball config (scripts/finalize_outcome_basketball.py).
PROD = {"k": 20.0, "home_advantage": 60.0, "season_regression": 0.40, "mov": "fte"}

# Odds-archive nickname -> candidate franchise abbreviation(s). "Hornets" is
# New Orleans before 2013-14 and Charlotte after; the both-teams date match
# resolves it, so both candidates are listed.
NICK_TO_ABBRS: dict[str, tuple[str, ...]] = {
    "Bucks": ("MIL",), "Bulls": ("CHI",), "Cavaliers": ("CLE",), "Celtics": ("BOS",),
    "Clippers": ("LAC",), "Grizzlies": ("MEM",), "Hawks": ("ATL",), "Heat": ("MIA",),
    "Jazz": ("UTA",), "Kings": ("SAC",), "Knicks": ("NYK",), "Lakers": ("LAL",),
    "Magic": ("ORL",), "Mavericks": ("DAL",), "Nets": ("BKN",), "NewJersey": ("BKN",),
    "Nuggets": ("DEN",), "Pacers": ("IND",), "Pelicans": ("NOP",), "Pistons": ("DET",),
    "Raptors": ("TOR",), "Rockets": ("HOU",), "Seventysixers": ("PHI",), "Spurs": ("SAS",),
    "Suns": ("PHX",), "Thunder": ("OKC",), "Timberwolves": ("MIN",),
    "Trailblazers": ("POR",), "Warriors": ("GSW",), "Golden State": ("GSW",),
    "Wizards": ("WAS",), "Hornets": ("NOP", "CHA"), "Bobcats": ("CHA",),
}


def ml_to_prob(ml: float) -> float:
    """American moneyline -> raw implied probability (margin still in)."""
    return (-ml) / (-ml + 100.0) if ml < 0 else 100.0 / (ml + 100.0)


def load_odds() -> pd.DataFrame:
    rows = []
    for g in json.load(ODDS_JSON.open()):
        home, away = str(g.get("home_team")), str(g.get("away_team"))
        hml, aml = g.get("home_close_ml"), g.get("away_close_ml")
        if home not in NICK_TO_ABBRS or away not in NICK_TO_ABBRS:
            continue
        if not isinstance(hml, (int, float)) or not isinstance(aml, (int, float)):
            continue
        if hml == 0 or aml == 0:
            continue
        ph, pa = ml_to_prob(float(hml)), ml_to_prob(float(aml))
        total = ph + pa  # de-margin
        rows.append({
            "date": pd.to_datetime(str(int(g["date"])), format="%Y%m%d"),
            "home_cands": NICK_TO_ABBRS[home],
            "away_cands": NICK_TO_ABBRS[away],
            "p_home": ph / total,
            "p_away": pa / total,
        })
    return pd.DataFrame(rows)


def attach_odds(games: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Left-join market probabilities onto our games by date + both teams."""
    by_date: dict = {}
    for r in odds.itertuples(index=False):
        by_date.setdefault(r.date, []).append(r)

    p_home, p_away = [], []
    for g in games.itertuples(index=False):
        hit = None
        for o in by_date.get(g.date, []):
            if g.home_abbr in o.home_cands and g.away_abbr in o.away_cands:
                hit = o
                break
        p_home.append(hit.p_home if hit else np.nan)
        p_away.append(hit.p_away if hit else np.nan)
    out = games.copy()
    out["book_p_home"] = p_home
    out["book_p_away"] = p_away
    return out


def main() -> None:
    odds = load_odds()
    print(f"Odds rows usable: {len(odds)}")

    games = load_nba_games()
    df = annotate_pre_match_elo(games, k=PROD["k"], home_advantage=PROD["home_advantage"],
                                season_regression=PROD["season_regression"], mov=PROD["mov"])
    df = attach_odds(df, odds)
    matched = int(df["book_p_home"].notna().sum())
    print(f"Games matched to odds: {matched} / {len(df)}")

    # Walk-forward exactly as the v3 harness, but score BOTH the model and the
    # market only on odds-matched test rows so the comparison is same-games.
    pooled = {"elo_mov": {"p": [], "y": []}, "bookmaker": {"p": [], "y": []}}
    per_season = []
    for s, train, test in _folds(df):
        sub = test[test["book_p_home"].notna()]
        if sub.empty:
            continue
        model = _fit_logistic(train, ["elo_diff"])
        p_model = _proba(model, sub, ["elo_diff"])
        p_book = sub[["book_p_home", "book_p_away"]].to_numpy()  # CLASSES order H, A
        y = sub["ftr"].to_numpy()
        pooled["elo_mov"]["p"].append(p_model)
        pooled["elo_mov"]["y"].append(y)
        pooled["bookmaker"]["p"].append(p_book)
        pooled["bookmaker"]["y"].append(y)
        per_season.append((s, len(sub), accuracy(p_model, y), accuracy(p_book, y)))
        print(f"  {s}: n={len(sub)}  model acc {accuracy(p_model, y):.3f}  book acc {accuracy(p_book, y):.3f}")

    results = {}
    for name, acc in pooled.items():
        p, y = np.vstack(acc["p"]), np.concatenate(acc["y"])
        results[name] = {"n": len(y), "accuracy": accuracy(p, y), "log_loss": log_loss(p, y),
                         "brier": brier(p, y), "ece": ece(p, y)}

    rows = ["| Model | n | Accuracy | Log loss | Brier | ECE |",
            "|---|---:|---:|---:|---:|---:|"]
    for name in ("elo_mov", "bookmaker"):
        r = results[name]
        rows.append(f"| {name} | {r['n']} | {r['accuracy']:.3f} | {r['log_loss']:.3f} "
                    f"| {r['brier']:.3f} | {r['ece']:.3f} |")
    table = "\n".join(rows)
    print("\n" + table)
    gap = results["elo_mov"]["log_loss"] - results["bookmaker"]["log_loss"]
    print(f"\nGap to market: {gap:+.4f} log loss")

    season_rows = ["| Season | n | elo_mov | bookmaker |", "|---|---:|---:|---:|"]
    for s, n, am, ab in per_season:
        season_rows.append(f"| {s} | {n} | {am:.3f} | {ab:.3f} |")

    md = f"""# Outcome Model — NBA Market (Odds) Baseline

**Companion to:** `reports/MODEL_IMPROVEMENT_PLAN.md` (step 5),
`reports/OUTCOME_MODEL_BASKETBALL_V3.md`.
**Data:** closing moneylines for {matched} of our regular-season games
(seasons 2011-12..2021-22; sportsbookreview archive via
`historical-datas/nba_odds_10y.json`). De-margined two-way probabilities.
Both columns are scored on exactly the same games, walk-forward, out-of-sample.

## Pooled out-of-sample metrics (odds-matched games only)

{table}

- **elo_mov** — the production basketball config (MOV Elo, K=20, HA=60, SR=0.40).
- **bookmaker** — de-margined closing moneylines: the practical ceiling.

**Gap to the market: {gap:+.4f} log loss.** This is the number the basketball
backtests were missing — it bounds how much headroom is left for feature work
(rest/b2b, which backtests at ~-0.002, fits inside it; see the v3 report).

## Per-season out-of-sample accuracy (matched games)

{chr(10).join(season_rows)}
"""
    REPORT.write_text(md)
    print(f"\nWrote {REPORT}")


if __name__ == "__main__":
    main()
