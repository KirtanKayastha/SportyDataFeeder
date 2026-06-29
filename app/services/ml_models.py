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
MODEL_VERSION_V2 = "outcome_v2_elo"
MODELS_DIR = Path("models_pkl")
OUTCOME_MODEL_FILE = "outcome_model.pkl"
OUTCOME_V2_FILE = "outcome_v2.pkl"                       # football Elo bundle
OUTCOME_V2_BASKETBALL_FILE = "outcome_v2_basketball.pkl"  # basketball Elo bundle
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


def load_outcome_v2(models_dir: Path | None = None):
    """Load the real-data FOOTBALL outcome_v2 bundle (Elo + logistic). Loaded
    separately from load_all_models so existing callers/tests are unaffected.
    Returns None (with a WARNING) when the bundle is absent."""
    return load_model(OUTCOME_V2_FILE, models_dir)


def load_outcome_v2_basketball(models_dir: Path | None = None):
    """Load the real-data BASKETBALL (NBA) outcome_v2 bundle. Same Elo+logistic
    shape as football but 2-class (no draw) and keyed by team abbreviation."""
    return load_model(OUTCOME_V2_BASKETBALL_FILE, models_dir)


def predict_outcome_v2(bundle, home_team: str | None, away_team: str | None) -> dict | None:
    """Outcome probabilities from the Elo-based outcome_v2 bundle. Returns None
    if the bundle is missing/incompatible so the caller can fall back to v1.

    The bundle (built by scripts/finalize_outcome_v2.py) holds:
      kind='elo_logistic', model (sklearn Pipeline on [elo_diff]),
      elo_ratings {canonical_team_name: rating}, home_advantage, base.
    """
    if not bundle or bundle.get("kind") != "elo_logistic" or "elo_ratings" not in bundle:
        return None

    # Imported here to keep ml_models import-light and avoid cycles.
    from app.services.team_ratings import APP_TEAM_ALIASES, normalize_team_name

    import math

    ratings = bundle["elo_ratings"]
    home_adv = bundle.get("home_advantage", 65.0)
    base = bundle.get("base", 1500.0)
    # Each bundle carries its own app-name -> Elo-key alias map (football long
    # names; basketball abbreviations). Older football bundles without one fall
    # back to the built-in football aliases.
    aliases = bundle.get("aliases", APP_TEAM_ALIASES)
    home = normalize_team_name(home_team, aliases)
    away = normalize_team_name(away_team, aliases)
    rh = ratings.get(home, base)
    ra = ratings.get(away, base)
    elo_diff = (rh + home_adv) - ra

    model = bundle["model"]
    proba = model.predict_proba([[elo_diff]])[0]
    by_class = {str(label): float(prob) for label, prob in zip(model.classes_, proba)}
    return {
        "home_win_prob": by_class.get("H", 0.0),
        "draw_prob": by_class.get("D", 0.0),
        "away_win_prob": by_class.get("A", 0.0),
        "model_version": MODEL_VERSION_V2,
        "elo_diff": elo_diff,
        "home_known": home in ratings,
        "away_known": away in ratings,
        # Cheap Elo-derived strength in [0,1] (0.5 == league-average rating) so
        # the /predict response stays populated without expensive player-form
        # queries. A 400-pt Elo edge maps to ~0.73.
        "home_strength": 1.0 / (1.0 + math.exp(-(rh - base) / 400.0)),
        "away_strength": 1.0 / (1.0 + math.exp(-(ra - base) / 400.0)),
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
