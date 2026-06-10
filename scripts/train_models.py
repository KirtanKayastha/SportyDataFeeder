# /home/sam069/projects/SportyDataFeeder/scripts/train_models.py
#
# Offline training (PRD R-3.3):
#   - event_rates.pkl: {player_id: {event_type: per-minute probability}} for
#     every player with player_stats rows.
#   - outcome_model.pkl: sklearn Pipeline(MinMaxScaler, LogisticRegression)
#     trained on finished matches. The scaler is serialised WITH the model —
#     never as a separate scaler.pkl.
#
# Usage: alembic upgrade head && python -m scripts.train_models

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from app.database import Match, PlayerStat, Session
from app.routers.matches import _score_match
from app.services.features import compute_player_features, compute_team_strength
from app.services.ml_models import (
    EVENT_RATES_FILE,
    LABEL_AWAY,
    LABEL_DRAW,
    LABEL_HOME,
    OUTCOME_MODEL_FILE,
    save_model,
)

logger = logging.getLogger(__name__)

MIN_MATCHES_TO_TRAIN = 5
MIN_MATCHES_FOR_SPLIT = 20
HOME_BIAS = 1.0


def build_event_rates(db) -> dict[int, dict[str, float]]:
    player_ids = [row[0] for row in db.query(PlayerStat.player_id).distinct().all()]
    rates = {
        player_id: compute_player_features(player_id, db)["event_rates"]
        for player_id in player_ids
    }
    logger.info("Built event rates for %s players", len(rates))
    return rates


def collect_outcome_training_data(db) -> tuple[list[list[float]], list[int]]:
    matches = db.query(Match).filter(Match.status == "finished").all()
    features: list[list[float]] = []
    labels: list[int] = []
    for match in matches:
        home_score, away_score = _score_match(db, match)
        if home_score > away_score:
            label = LABEL_HOME
        elif away_score > home_score:
            label = LABEL_AWAY
        else:
            label = LABEL_DRAW
        home_strength = compute_team_strength(match.home_team_id, db)
        away_strength = compute_team_strength(match.away_team_id, db)
        features.append([home_strength, away_strength, HOME_BIAS])
        labels.append(label)
    return features, labels


def _make_pipeline() -> Pipeline:
    # PRD specifies multi_class="multinomial"; the argument was removed in
    # sklearn 1.7+ and lbfgs is multinomial by default, so it is omitted.
    return Pipeline(
        [
            ("scaler", MinMaxScaler()),
            ("clf", LogisticRegression(solver="lbfgs", max_iter=500)),
        ]
    )  


def train_outcome_model(features: list[list[float]], labels: list[int]) -> Pipeline | None:
    if len(labels) < MIN_MATCHES_TO_TRAIN:
        logger.warning(
            "Only %s finished matches (< %s); skipping outcome model training",
            len(labels),
            MIN_MATCHES_TO_TRAIN,
        )
        return None

    model = _make_pipeline()
    if len(labels) >= MIN_MATCHES_FOR_SPLIT:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, stratify=labels, random_state=42
            )
        except ValueError:
            logger.warning("Stratified split failed (a class is too rare); training on all data")
            model.fit(features, labels)
            return model
        model.fit(X_train, y_train)
        report = classification_report(y_test, model.predict(X_test), zero_division=0)
        logger.info("Outcome model held-out report:\n%s", report)
    else:
        logger.info("Training outcome model on all %s matches (too few for a split)", len(labels))
        model.fit(features, labels)
    return model


def main(models_dir: Path | None = None) -> dict:
    db = Session()
    try:
        event_rates = build_event_rates(db)
        save_model(event_rates, EVENT_RATES_FILE, models_dir=models_dir)

        features, labels = collect_outcome_training_data(db)
        logger.info("Found %s finished matches for outcome training", len(labels))
        outcome_model = train_outcome_model(features, labels)
        if outcome_model is not None:
            save_model(outcome_model, OUTCOME_MODEL_FILE, models_dir=models_dir)

        return {
            "event_rates_players": len(event_rates),
            "finished_matches": len(labels),
            "outcome_model_trained": outcome_model is not None,
        }
    finally:
        db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    summary = main()
    logger.info("Training summary: %s", summary)
