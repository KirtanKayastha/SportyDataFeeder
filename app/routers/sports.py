# /home/sam069/projects/SportyDataFeeder/app/routers/sports.py

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError

from app.database import Event, Match, Player, Sport, Team, get_db
from app.schemas import SportCreate, SportRead

router = APIRouter()


def _not_found(entity: str, entity_id: int):
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{entity} {entity_id} not found")


@router.get("", response_model=list[SportRead])
def list_sports(db=Depends(get_db)):
    return db.query(Sport).order_by(Sport.name.asc()).all()


@router.post("", response_model=SportRead, status_code=status.HTTP_201_CREATED)
def create_sport(payload: SportCreate, db=Depends(get_db)):
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Sport name is required")

    existing = db.query(Sport).filter(Sport.name == name).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Sport '{name}' already exists")

    sport = Sport(name=name)
    db.add(sport)
    try:
        db.commit()
        db.refresh(sport)
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not create sport") from exc
    return sport


@router.delete("/{sport_id}")
def delete_sport(sport_id: int, db=Depends(get_db)):
    sport = db.query(Sport).filter_by(id=sport_id).first()
    if not sport:
        _not_found("Sport", sport_id)

    team_count = db.query(Team).filter_by(sport_id=sport_id).count()
    player_count = db.query(Player).filter_by(sport_id=sport_id).count()
    match_count = db.query(Match).filter_by(sport_id=sport_id).count()
    if team_count or player_count or match_count:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Sport {sport_id} cannot be deleted because dependent records exist: "
                f"teams={team_count}, players={player_count}, matches={match_count}"
            ),
        )

    db.delete(sport)
    db.commit()
    return {"detail": f"Sport {sport_id} deleted"}
