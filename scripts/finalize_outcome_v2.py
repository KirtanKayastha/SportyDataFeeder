# /home/sam069/projects/SportyDataFeeder/scripts/finalize_outcome_v2.py
#
# Build the PRODUCTION football outcome bundle used by the live /predict path.
# Current winner (reports/OUTCOME_MODEL_V4.md, step 3 of the improvement plan):
# margin-of-victory Elo (FiveThirtyEight multiplier, K=40, SR=0.10) + causal
# rolling shots-on-target form — pooled OOS log loss 0.9661 vs 0.9709 for the
# v3 classic-Elo config and 0.9527 for the bookmaker ceiling.
#
# Bundle schema (consumed by app.services.ml_models.predict_outcome_v2):
#   kind='elo_logistic', model (Pipeline on `features`),
#   features=['elo_diff', 'sot_net_diff'],
#   elo_ratings {canonical_team_name: rating}, sot_form {team: net SoT form},
#   aliases, home_advantage, base, classes, model_version, trained_on, n_teams.
#
# Usage:  python -m scripts.finalize_outcome_v2

import pickle
import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.features_team import annotate_shot_form, shot_form_state
from app.services.ml_models import MODEL_VERSION_V4
from app.services.team_ratings import APP_TEAM_ALIASES, EloModel, annotate_pre_match_elo, fit_elo
from scripts.load_historical import load_matches

BUNDLE = Path(__file__).resolve().parents[1] / "models_pkl" / "outcome_v2.pkl"
# Walk-forward tuned across reports/OUTCOME_MODEL_V3.md and _V4.md. K=40 pairs
# with the "fte" MOV multiplier (which scales updates down by ~0.4-0.5 for
# typical football margins); HA is absorbed by the logistic intercept.
HOME_ADV = 65.0
K = 40.0
SEASON_REGRESSION = 0.10
MOV = "fte"
FEATURES = ["elo_diff", "sot_net_diff"]


def main() -> None:
    df = load_matches()
    seasons = list(dict.fromkeys(df["season"]))

    # Fit the logistic on causal pre-match features over all matches...
    annotated = annotate_shot_form(annotate_pre_match_elo(
        df, k=K, home_advantage=HOME_ADV, season_regression=SEASON_REGRESSION, mov=MOV
    ))
    model = Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=1000))])
    model.fit(annotated[FEATURES].values, annotated["ftr"].values)

    # ...and capture the CURRENT (post-last-match) Elo ratings and per-team
    # shots-on-target form for prediction.
    elo: EloModel = fit_elo(
        df, k=K, home_advantage=HOME_ADV, season_regression=SEASON_REGRESSION, mov=MOV
    )
    sot_form = shot_form_state(df)

    bundle = {
        "kind": "elo_logistic",
        "sport": "football",
        "model": model,
        "features": list(FEATURES),
        "elo_ratings": dict(elo.ratings),
        "sot_form": sot_form,
        # App football teams use long/official names; map them to the Elo keys.
        "aliases": dict(APP_TEAM_ALIASES),
        "home_advantage": HOME_ADV,
        "base": elo.base,
        "mov": MOV,
        "classes": list(model.classes_),
        "model_version": MODEL_VERSION_V4,
        "trained_on": f"{len(df)} EPL matches, seasons {seasons[0]}..{seasons[-1]}",
        "n_teams": len(elo.ratings),
    }
    BUNDLE.parent.mkdir(exist_ok=True)
    with BUNDLE.open("wb") as fh:
        pickle.dump(bundle, fh)

    top = sorted(elo.ratings.items(), key=lambda kv: -kv[1])[:6]
    print(f"Saved production bundle -> {BUNDLE}")
    print(f"  trained_on: {bundle['trained_on']}  version: {bundle['model_version']}")
    print(f"  classes: {bundle['classes']}  n_teams: {bundle['n_teams']}")
    print(f"  top Elo: {[(t, round(r)) for t, r in top]}")
    top_sot = sorted(sot_form.items(), key=lambda kv: -kv[1])[:4]
    print(f"  top SoT form: {[(t, round(v, 2)) for t, v in top_sot]}")


if __name__ == "__main__":
    main()
