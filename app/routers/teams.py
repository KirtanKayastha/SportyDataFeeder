# /home/sam069/projects/SportyDataFeeder/app/routers/teams.py

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import or_
from sqlalchemy.exc import IntegrityError

from app.database import Match, Player, Sport, Team, get_db
from app.schemas import TeamCreate, TeamRead

router = APIRouter()


def _not_found(entity: str, entity_id: int):
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{entity} {entity_id} not found")


@router.get("", response_model=list[TeamRead])
def list_teams(sport_id: int | None = Query(default=None), db=Depends(get_db)):
    query = db.query(Team)
    if sport_id is not None:
        query = query.filter(Team.sport_id == sport_id)
    return query.order_by(Team.name.asc()).all()


@router.post("", response_model=TeamRead, status_code=status.HTTP_201_CREATED)
def create_team(payload: TeamCreate, db=Depends(get_db)):
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Team name is required")

    sport = db.query(Sport).filter_by(id=payload.sport_id).first()
    if not sport:
        _not_found("Sport", payload.sport_id)

    existing = db.query(Team).filter(Team.sport_id == payload.sport_id, Team.name == name).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Team '{name}' already exists for this sport")

    team = Team(name=name, sport_id=payload.sport_id)
    db.add(team)
    try:
        db.commit()
        db.refresh(team)
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not create team") from exc
    return team


@router.delete("/{team_id}")
def delete_team(team_id: int, db=Depends(get_db)):
    team = db.query(Team).filter_by(id=team_id).first()
    if not team:
        _not_found("Team", team_id)

    player_count = db.query(Player).filter_by(team_id=team_id).count()
    match_count = db.query(Match).filter(or_(Match.home_team_id == team_id, Match.away_team_id == team_id)).count()
    if player_count or match_count:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Team {team_id} cannot be deleted because dependent records exist: "
                f"players={player_count}, matches={match_count}"
            ),
        )

    db.delete(team)
    db.commit()
    return {"detail": f"Team {team_id} deleted"}
