# /home/sam069/projects/SportyDataFeeder/app/routers/matches.py

import json

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.exc import IntegrityError

from app.database import Event, Match, Player, Sport, Team, get_db
from app.schemas import MatchCreate, MatchDetailRead, MatchRead

router = APIRouter()


def _not_found(entity: str, entity_id: int):
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{entity} {entity_id} not found")


def _score_match(db, match: Match) -> tuple[int, int]:
    sport = db.query(Sport).filter_by(id=match.sport_id).first()
    sport_name = sport.name if sport else ""
    is_basketball = "basket" in sport_name.lower()

    events = (
        db.query(Event)
        .filter_by(match_id=match.id)
        .order_by(Event.minute.asc(), Event.created_at.asc(), Event.id.asc())
        .all()
    )
    player_team_map = {
        player.id: player.team_id
        for player in db.query(Player).filter(Player.sport_id == match.sport_id).all()
    }

    home_score = 0
    away_score = 0
    for event in events:
        if event.player_id is None:
            continue
        event_team_id = player_team_map.get(event.player_id)
        if event_team_id not in {match.home_team_id, match.away_team_id}:
            continue

        if is_basketball and event.event_type in {"point_2", "point_3", "free_throw"}:
            extra_data = {}
            if event.extra:
                try:
                    extra_data = json.loads(event.extra)
                except json.JSONDecodeError:
                    extra_data = {}
            points = extra_data.get("points")
            if points is None:
                points = {"point_2": 2, "point_3": 3, "free_throw": 1}.get(event.event_type, 0)
            if event_team_id == match.home_team_id:
                home_score += int(points or 0)
            else:
                away_score += int(points or 0)
        elif not is_basketball and event.event_type == "goal":
            if event_team_id == match.home_team_id:
                home_score += 1
            else:
                away_score += 1

    return home_score, away_score


@router.get("", response_model=list[MatchRead])
def list_matches(
    status_filter: str | None = Query(default=None, alias="status"),
    sport_id: int | None = Query(default=None),
    db=Depends(get_db),
):
    query = db.query(Match)
    if status_filter is not None:
        query = query.filter(Match.status == status_filter)
    if sport_id is not None:
        query = query.filter(Match.sport_id == sport_id)
    return query.order_by(Match.match_date.desc(), Match.id.desc()).all()


@router.post("", response_model=MatchRead, status_code=status.HTTP_201_CREATED)
def create_match(payload: MatchCreate, db=Depends(get_db)):
    if payload.home_team_id == payload.away_team_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Home and away teams must be different")

    sport = db.query(Sport).filter_by(id=payload.sport_id).first()
    if not sport:
        _not_found("Sport", payload.sport_id)

    home_team = db.query(Team).filter_by(id=payload.home_team_id).first()
    if not home_team:
        _not_found("Team", payload.home_team_id)

    away_team = db.query(Team).filter_by(id=payload.away_team_id).first()
    if not away_team:
        _not_found("Team", payload.away_team_id)

    if home_team.sport_id != payload.sport_id or away_team.sport_id != payload.sport_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Both teams must belong to the selected sport")

    match = Match(
        home_team_id=payload.home_team_id,
        away_team_id=payload.away_team_id,
        match_date=payload.match_date,
        sport_id=payload.sport_id,
        status="scheduled",
    )
    db.add(match)
    try:
        db.commit()
        db.refresh(match)
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not create match") from exc
    return match


@router.get("/{match_id}", response_model=MatchDetailRead)
def get_match(match_id: int, db=Depends(get_db)):
    match = db.query(Match).filter_by(id=match_id).first()
    if not match:
        _not_found("Match", match_id)

    home_score, away_score = _score_match(db, match)
    return MatchDetailRead.model_validate(
        {
            "id": match.id,
            "home_team_id": match.home_team_id,
            "away_team_id": match.away_team_id,
            "match_date": match.match_date,
            "status": match.status,
            "sport_id": match.sport_id,
            "home_score": home_score,
            "away_score": away_score,
        }
    )
