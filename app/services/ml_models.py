# /home/sam069/projects/SportyDataFeeder/app/services/ml_models.py
#
# Load/save helpers for the pkl models (PRD R-3.4). The scaler is always
# serialised inside the sklearn Pipeline — never as a separate pkl. A missing
# model is never fatal: loaders return None with a WARNING and callers fall
# back to heuristics.

import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_VERSION = "outcome_v1_logistic"
MODELS_DIR = Path("models_pkl")
OUTCOME_MODEL_FILE = "outcome_model.pkl"
EVENT_RATES_FILE = "event_rates.pkl"

# Outcome class labels (PRD R-3.3): 0=away win, 1=draw, 2=home win.
LABEL_AWAY, LABEL_DRAW, LABEL_HOME = 0, 1, 2


def save_model(obj, filename: str, models_dir: Path | None = None) -> Path:
    models_dir = Path(models_dir) if models_dir else MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / filename
    with path.open("wb") as fh:
        pickle.dump(obj, fh)
    logger.info("Saved model to %s", path)
    return path


def load_model(filename: str, models_dir: Path | None = None):
    models_dir = Path(models_dir) if models_dir else MODELS_DIR
    path = models_dir / filename
    if not path.exists():
        logger.warning("Model file %s not found; falling back to heuristics", path)
        return None
    try:
        with path.open("rb") as fh:
            return pickle.load(fh)
    except Exception:
        logger.exception("Failed to load model %s; falling back to heuristics", path)
        return None


def load_all_models(models_dir: Path | None = None) -> dict:
    return {
        "outcome_model": load_model(OUTCOME_MODEL_FILE, models_dir),
        "event_rates": load_model(EVENT_RATES_FILE, models_dir),
    }


def heuristic_outcome(home_strength: float, away_strength: float) -> dict:
    """Strength-difference heuristic used when no trained model is available."""
    diff = home_strength - away_strength
    home = max(0.05, 0.40 + 0.35 * diff)
    away = max(0.05, 0.32 - 0.35 * diff)
    draw = 0.28
    total = home + draw + away
    return {
        "home_win_prob": home / total,
        "draw_prob": draw / total,
        "away_win_prob": away / total,
        "model_version": "heuristic_v1",
    }


def predict_outcome(model, home_strength: float, away_strength: float) -> dict:
    """3-class outcome probabilities. Maps probabilities through
    ``model.classes_`` — never assumes class order (PRD R-3.4)."""
    if model is None:
        return heuristic_outcome(home_strength, away_strength)

    features = [[home_strength, away_strength, 1.0]]
    probabilities = model.predict_proba(features)[0]
    by_label = {int(label): float(prob) for label, prob in zip(model.classes_, probabilities)}
    return {
        "home_win_prob": by_label.get(LABEL_HOME, 0.0),
        "draw_prob": by_label.get(LABEL_DRAW, 0.0),
        "away_win_prob": by_label.get(LABEL_AWAY, 0.0),
        "model_version": MODEL_VERSION,
    }
