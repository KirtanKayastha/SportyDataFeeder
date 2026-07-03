# /home/sam069/projects/SportyDataFeeder/app/services/prediction_metrics.py
#
# Step 4 of reports/MODEL_IMPROVEMENT_PLAN.md: score the predictions we stored.
# Every /predict call writes a MatchPrediction row; once the match finishes, the
# stored probabilities can be judged against the actual result — turning
# production into a continuous backtest and exposing calibration drift that
# offline walk-forward numbers can't see.
#
# Scores are derived from events via matches._score_match (never stored), same
# as everywhere else. When a match was predicted several times by the same
# model version, only the LATEST prediction counts (the earlier ones were
# superseded). Different model versions are scored separately so a bundle
# upgrade shows up as two comparable rows, not a blended average.

import math
from datetime import datetime, timezone

from app.database import Match, MatchPrediction

CLASSES = ("H", "D", "A")
EPS = 1e-15
CALIBRATION_BINS = 10


def _label(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "H"
    if away_score > home_score:
        return "A"
    return "D"


def _proba(row: MatchPrediction) -> tuple[float, float, float]:
    return (row.home_win_prob, row.draw_prob, row.away_win_prob)


def _metrics(scored: list[tuple[tuple[float, float, float], str]]) -> dict:
    """Accuracy / log loss / Brier / confidence-ECE + calibration bins for a
    list of (probability vector in CLASSES order, actual label) pairs."""
    n = len(scored)
    correct_total = 0
    log_loss_sum = 0.0
    brier_sum = 0.0
    bins = [
        {"n": 0, "confidence_sum": 0.0, "correct": 0}
        for _ in range(CALIBRATION_BINS)
    ]

    for proba, actual in scored:
        actual_idx = CLASSES.index(actual)
        log_loss_sum += -math.log(max(proba[actual_idx], EPS))
        brier_sum += sum(
            (p - (1.0 if i == actual_idx else 0.0)) ** 2 for i, p in enumerate(proba)
        )
        conf = max(proba)
        pred_idx = proba.index(conf)
        correct = pred_idx == actual_idx
        correct_total += correct
        b = min(int(conf * CALIBRATION_BINS), CALIBRATION_BINS - 1)
        bins[b]["n"] += 1
        bins[b]["confidence_sum"] += conf
        bins[b]["correct"] += correct

    ece = 0.0
    calibration = []
    for i, b in enumerate(bins):
        if not b["n"]:
            continue
        avg_conf = b["confidence_sum"] / b["n"]
        acc = b["correct"] / b["n"]
        ece += abs(acc - avg_conf) * b["n"] / n
        calibration.append({
            "bin": f"{i / CALIBRATION_BINS:.1f}-{(i + 1) / CALIBRATION_BINS:.1f}",
            "n": b["n"],
            "avg_confidence": round(avg_conf, 4),
            "accuracy": round(acc, 4),
        })

    return {
        "n": n,
        "accuracy": round(correct_total / n, 4),
        "log_loss": round(log_loss_sum / n, 4),
        "brier": round(brier_sum / n, 4),
        "ece": round(ece, 4),
        "calibration": calibration,
    }


def compute_prediction_metrics(db) -> dict:
    """Join stored match_predictions to finished matches and report accuracy,
    log loss, Brier and calibration per model_version (see module header)."""
    # matches.py imports services, never this module, so the import is acyclic
    # (same precedent as scripts/train_models.py).
    from app.routers.matches import _score_match

    rows = (
        db.query(MatchPrediction, Match)
        .join(Match, MatchPrediction.match_id == Match.id)
        .filter(Match.status == "finished")
        .order_by(MatchPrediction.created_at.asc(), MatchPrediction.id.asc())
        .all()
    )

    # Latest prediction per (match, model_version); rows are in creation order
    # so later inserts overwrite earlier ones.
    latest: dict[tuple[int, str], tuple[MatchPrediction, Match]] = {}
    for pred, match in rows:
        latest[(match.id, pred.model_version)] = (pred, match)

    labels: dict[int, str] = {}
    by_version: dict[str, list] = {}
    for (match_id, version), (pred, match) in latest.items():
        if match_id not in labels:
            labels[match_id] = _label(*_score_match(db, match))
        by_version.setdefault(version, []).append((_proba(pred), labels[match_id]))

    return {
        "finished_matches_scored": len(labels),
        "predictions_scored": sum(len(v) for v in by_version.values()),
        "by_model_version": {
            version: _metrics(scored) for version, scored in sorted(by_version.items())
        },
    }


def build_metrics_push_payload(db) -> dict:
    """The /predict/metrics result stamped for the Sporty backend feed
    (POST /api/v1/feed/model-metrics)."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **compute_prediction_metrics(db),
    }
