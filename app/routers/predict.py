# /home/sam069/projects/SportyDataFeeder/app/routers/predict.py

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.database import Match, MatchPrediction, get_db
from app.schemas import PredictRequest, PredictResponse
from app.services.backend_client import get_backend_client
from app.services.features import compute_team_strength
from app.services.links import get_sporty_uuid
from app.services.ml_models import predict_outcome

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
async def predict_match_outcome(payload: PredictRequest, request: Request, db=Depends(get_db)):
    match = db.query(Match).filter_by(id=payload.match_id).first()
    if not match:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Match {payload.match_id} not found")

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
