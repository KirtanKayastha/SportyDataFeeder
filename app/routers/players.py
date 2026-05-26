# /home/sam069/projects/SportyDataFeeder/app/routers/players.py

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.exc import IntegrityError

from app.database import Event, Player, Sport, Team, get_db
from app.schemas import PlayerCreate, PlayerRead

router = APIRouter()


def _not_found(entity: str, entity_id: int):
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{entity} {entity_id} not found")


@router.get("", response_model=list[PlayerRead])
def list_players(
    team_id: int | None = Query(default=None),
    sport_id: int | None = Query(default=None),
    db=Depends(get_db),
):
    query = db.query(Player)
    if team_id is not None:
        query = query.filter(Player.team_id == team_id)
    if sport_id is not None:
        query = query.filter(Player.sport_id == sport_id)
    return query.order_by(Player.name.asc()).all()


@router.post("", response_model=PlayerRead, status_code=status.HTTP_201_CREATED)
def create_player(payload: PlayerCreate, db=Depends(get_db)):
    name = payload.name.strip()
    position = payload.position.strip() if payload.position else None
    if not name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Player name is required")

    team = db.query(Team).filter_by(id=payload.team_id).first()
    if not team:
        _not_found("Team", payload.team_id)

    sport = db.query(Sport).filter_by(id=payload.sport_id).first()
    if not sport:
        _not_found("Sport", payload.sport_id)

    if team.sport_id != payload.sport_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Player sport_id must match the selected team")

    existing = db.query(Player).filter(Player.team_id == payload.team_id, Player.name == name).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Player '{name}' already exists on this team")

    player = Player(name=name, team_id=payload.team_id, position=position, sport_id=payload.sport_id)
    db.add(player)
    try:
        db.commit()
        db.refresh(player)
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not create player") from exc
    return player


@router.delete("/{player_id}")
def delete_player(player_id: int, db=Depends(get_db)):
    player = db.query(Player).filter_by(id=player_id).first()
    if not player:
        _not_found("Player", player_id)

    event_count = db.query(Event).filter_by(player_id=player_id).count()
    if event_count:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Player {player_id} cannot be deleted because dependent events exist: events={event_count}",
        )

    db.delete(player)
    db.commit()
    return {"detail": f"Player {player_id} deleted"}
