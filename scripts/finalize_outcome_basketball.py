# /home/sam069/projects/SportyDataFeeder/scripts/finalize_outcome_basketball.py
#
# Build the PRODUCTION basketball outcome bundle used by the live /predict path
# (the NBA analogue of scripts/finalize_outcome_v2.py). Fits the Elo-logistic
# winner on ALL real NBA regular-season games and bakes in the final Elo
# ratings keyed by current team ABBREVIATION (what the app stores). 2-class
# (no draw).
#
# Usage:  python -m scripts.finalize_outcome_basketball

import pickle
import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.team_ratings import EloModel, annotate_pre_match_elo, fit_elo
from scripts.load_nba import load_nba_games

BUNDLE = Path(__file__).resolve().parents[1] / "models_pkl" / "outcome_v2_basketball.pkl"
HOME_ADV = 100.0
K = 20.0


def main() -> None:
    df = load_nba_games()
    seasons = list(dict.fromkeys(df["season"]))

    # Fit logistic on causal pre-match elo_diff over all games...
    annotated = annotate_pre_match_elo(df, k=K, home_advantage=HOME_ADV)
    model = Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=1000))])
    model.fit(annotated[["elo_diff"]].values, annotated["ftr"].values)

    # ...and capture CURRENT Elo ratings (keyed by team_id), then re-key by the
    # franchise's current abbreviation, which is what the app stores.
    elo: EloModel = fit_elo(df, k=K, home_advantage=HOME_ADV)
    abbr_by_id = dict(zip(df["home"], df["home_abbr"]))
    abbr_by_id.update(dict(zip(df["away"], df["away_abbr"])))
    elo_ratings = {abbr_by_id[tid]: r for tid, r in elo.ratings.items() if tid in abbr_by_id}

    bundle = {
        "kind": "elo_logistic",
        "sport": "basketball",
        "model": model,
        "features": ["elo_diff"],
        "elo_ratings": elo_ratings,
        # App basketball teams are stored as abbreviations == the rating keys, so
        # no alias remapping is needed (identity after HTML-unescape/trim).
        "aliases": {},
        "home_advantage": HOME_ADV,
        "base": elo.base,
        "classes": list(model.classes_),
        "trained_on": f"{len(df)} NBA regular-season games, seasons {seasons[0]}..{seasons[-1]}",
        "n_teams": len(elo_ratings),
    }
    BUNDLE.parent.mkdir(exist_ok=True)
    with BUNDLE.open("wb") as fh:
        pickle.dump(bundle, fh)

    top = sorted(elo_ratings.items(), key=lambda kv: -kv[1])[:6]
    print(f"Saved production bundle -> {BUNDLE}")
    print(f"  trained_on: {bundle['trained_on']}")
    print(f"  classes: {bundle['classes']}  n_teams: {bundle['n_teams']}")
    print(f"  top Elo: {[(t, round(r)) for t, r in top]}")


if __name__ == "__main__":
    main()
