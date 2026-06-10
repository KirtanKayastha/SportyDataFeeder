# /home/sam069/projects/SportyDataFeeder/app/routers/events.py

import json
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError

from app.database import Event, Match, Player, get_db
from app.schemas import EventCreate, EventRead

router = APIRouter()


def _not_found(entity: str, entity_id: int):
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{entity} {entity_id} not found")


def _serialize_event(event: Event) -> EventRead:
    extra_data = None
    if event.extra:
        try:
            extra_data = json.loads(event.extra)
        except json.JSONDecodeError:
            extra_data = None

    return EventRead.model_validate(
        {
            "id": event.id,
            "event_id": event.event_id,
            "match_id": event.match_id,
            "event_type": event.event_type,
            "player_id": event.player_id,
            "minute": event.minute,
            "extra": extra_data,
            "created_at": event.created_at,
        }
    )


@router.get("/matches/{match_id}/events", response_model=list[EventRead])
def list_match_events(match_id: int, db=Depends(get_db)):
    match = db.query(Match).filter_by(id=match_id).first()
    if not match:
        _not_found("Match", match_id)

    events = (
        db.query(Event)
        .filter_by(match_id=match_id)
        .order_by(Event.minute.asc(), Event.created_at.asc(), Event.id.asc())
        .all()
    )
    return [_serialize_event(event) for event in events]


@router.post("/matches/{match_id}/events", response_model=EventRead, status_code=status.HTTP_201_CREATED)
def create_match_event(match_id: int, payload: EventCreate, db=Depends(get_db)):
    match = db.query(Match).filter_by(id=match_id).first()
    if not match:
        _not_found("Match", match_id)

    if payload.player_id is not None:
        player = db.query(Player).filter_by(id=payload.player_id).first()
        if not player:
            _not_found("Player", payload.player_id)

    event = Event(
        event_id=str(uuid.uuid4()),
        match_id=match_id,
        event_type=payload.event_type.strip(),
        player_id=payload.player_id,
        minute=payload.minute,
        extra=json.dumps(payload.extra) if payload.extra is not None else None,
    )
    db.add(event)
    try:
        db.commit()
        db.refresh(event)
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not create event") from exc
    return _serialize_event(event)
