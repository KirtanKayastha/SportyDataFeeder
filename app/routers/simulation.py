# /home/sam069/projects/SportyDataFeeder/app/routers/simulation.py

from fastapi import APIRouter, Depends, HTTPException, status

from app.database import Match, get_db
from app.schemas import SimulationRequest, SimulationResult
from app.services.simulation import simulate_match_live

router = APIRouter()


@router.post("/{match_id}/simulate", response_model=SimulationResult)
def trigger_simulation(match_id: int, request: SimulationRequest | None = None, db=Depends(get_db)):
    if request is not None and request.match_id != match_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="match_id in the body must match the path parameter")

    match = db.query(Match).filter_by(id=match_id).first()
    if not match:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Match {match_id} not found")

    try:
        events_inserted = simulate_match_live(db, match_id)
        db.refresh(match)
        return SimulationResult(match_id=match_id, events_inserted=events_inserted, status=match.status)
    except ValueError as exc:
        message = str(exc)
        if "not found" in message:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=message) from exc
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message) from exc
    except HTTPException:
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Simulation failed") from exc
