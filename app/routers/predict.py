# /home/sam069/projects/SportyDataFeeder/app/routers/predict.py

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.database import Match, MatchPrediction, Sport, Team, get_db
from app.schemas import PredictRequest, PredictResponse
from app.services.backend_client import get_backend_client
from app.services.features import compute_team_strength
from app.services.links import get_sporty_uuid
from app.services.ml_models import predict_outcome, predict_outcome_v2
from app.services.prediction_metrics import build_metrics_push_payload, compute_prediction_metrics
from app.services.sport_resolver import SportType, resolve_sport_type

router = APIRouter()


@router.get("/predict/metrics")
def prediction_metrics(db=Depends(get_db)):
    """Score every stored prediction whose match has since finished: accuracy,
    log loss, Brier and a calibration table per model_version. Production's
    continuous backtest (step 4 of reports/MODEL_IMPROVEMENT_PLAN.md)."""
    return compute_prediction_metrics(db)


@router.post("/predict/metrics/push")
async def push_prediction_metrics(db=Depends(get_db)):
    """Compute the metrics scorecard and push it to the Sporty backend
    (POST /api/v1/feed/model-metrics) for the frontend's model panel. The
    simulation also pushes automatically when a match finishes; this endpoint
    is the manual/replay trigger."""
    payload = build_metrics_push_payload(db)
    pushed = await get_backend_client().push_model_metrics(payload)
    return {"pushed": pushed, "metrics": payload}


@router.post("/predict", response_model=PredictResponse)
async def predict_match_outcome(payload: PredictRequest, request: Request, db=Depends(get_db)):
    match = db.query(Match).filter_by(id=payload.match_id).first()
    if not match:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Match {payload.match_id} not found")

    # Prefer the real-data Elo model (outcome_v2) per sport (football +
    # basketball); fall back to the v1 strength model / heuristic for other
    # sports or when a sport's bundle is unavailable. The v2 path is cheap (Elo
    # lookup); only the fallback needs the expensive player-form team strength
    # (many DB reads), so compute it lazily.
    result = None
    sport = db.query(Sport).filter_by(id=match.sport_id).first()
    sport_type = resolve_sport_type(sport.name if sport else None)
    bundle = None
    if sport_type is SportType.FOOTBALL:
        bundle = getattr(request.app.state, "outcome_v2", None)
    elif sport_type is SportType.BASKETBALL:
        bundle = getattr(request.app.state, "outcome_v2_basketball", None)

    if bundle is not None:
        home_team = db.query(Team).filter_by(id=match.home_team_id).first()
        away_team = db.query(Team).filter_by(id=match.away_team_id).first()
        result = predict_outcome_v2(
            bundle,
            home_team.name if home_team else None,
            away_team.name if away_team else None,
        )

    if result is not None:
        home_strength = result["home_strength"]
        away_strength = result["away_strength"]
    else:
        home_strength = compute_team_strength(match.home_team_id, db)
        away_strength = compute_team_strength(match.away_team_id, db)
        model = getattr(request.app.state, "outcome_model", None)
        result = predict_outcome(model, home_strength, away_strength)

    db.add(
        MatchPrediction(
            match_id=match.id,
            home_win_prob=result["home_win_prob"],
            draw_prob=result["draw_prob"],
            away_win_prob=result["away_win_prob"],
            model_version=result["model_version"],
        )
    )
    db.commit()

    pushed = False
    sporty_match_id = get_sporty_uuid(db, "match", match.id)
    if sporty_match_id is not None:
        pushed = await get_backend_client().push_prediction(
            {
                "sporty_match_id": sporty_match_id,
                "home_win_prob": result["home_win_prob"],
                "draw_prob": result["draw_prob"],
                "away_win_prob": result["away_win_prob"],
                "model_version": result["model_version"],
            }
        )

    return PredictResponse(
        match_id=match.id,
        home_win_prob=result["home_win_prob"],
        draw_prob=result["draw_prob"],
        away_win_prob=result["away_win_prob"],
        model_version=result["model_version"],
        home_strength=home_strength,
        away_strength=away_strength,
        pushed=pushed,
    )
