# /home/sam069/projects/SportyDataFeeder/scripts/finalize_outcome_v2.py
#
# Phase E: build the PRODUCTION outcome_v2 bundle used by the live /predict path.
# The Phase C winner was `elo_logistic` (lowest out-of-sample log loss), so this
# fits that model on ALL real EPL seasons and bakes in the final Elo ratings
# table (each team's current rating) needed at prediction time.
#
# Bundle schema (consumed by app.services.ml_models.predict_outcome_v2):
#   kind='elo_logistic', model (Pipeline on [elo_diff]),
#   elo_ratings {canonical_team_name: rating}, home_advantage, base, classes,
#   trained_on, n_teams.
#
# Usage:  python -m scripts.finalize_outcome_v2

import pickle
import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.team_ratings import APP_TEAM_ALIASES, EloModel, annotate_pre_match_elo, fit_elo
from scripts.load_historical import load_matches

BUNDLE = Path(__file__).resolve().parents[1] / "models_pkl" / "outcome_v2.pkl"
HOME_ADV = 65.0
K = 20.0


def main() -> None:
    df = load_matches()
    seasons = list(dict.fromkeys(df["season"]))

    # Fit the logistic on causal pre-match elo_diff over all matches...
    annotated = annotate_pre_match_elo(df, k=K, home_advantage=HOME_ADV)
    model = Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=1000))])
    model.fit(annotated[["elo_diff"]].values, annotated["ftr"].values)

    # ...and capture the CURRENT (post-last-match) Elo ratings for prediction.
    elo: EloModel = fit_elo(df, k=K, home_advantage=HOME_ADV)

    bundle = {
        "kind": "elo_logistic",
        "sport": "football",
        "model": model,
        "features": ["elo_diff"],
        "elo_ratings": dict(elo.ratings),
        # App football teams use long/official names; map them to the Elo keys.
        "aliases": dict(APP_TEAM_ALIASES),
        "home_advantage": HOME_ADV,
        "base": elo.base,
        "classes": list(model.classes_),
        "trained_on": f"{len(df)} EPL matches, seasons {seasons[0]}..{seasons[-1]}",
        "n_teams": len(elo.ratings),
    }
    BUNDLE.parent.mkdir(exist_ok=True)
    with BUNDLE.open("wb") as fh:
        pickle.dump(bundle, fh)

    top = sorted(elo.ratings.items(), key=lambda kv: -kv[1])[:6]
    print(f"Saved production bundle -> {BUNDLE}")
    print(f"  trained_on: {bundle['trained_on']}")
    print(f"  classes: {bundle['classes']}  n_teams: {bundle['n_teams']}")
    print(f"  top Elo: {[(t, round(r)) for t, r in top]}")


if __name__ == "__main__":
    main()
