# /home/sam069/projects/SportyDataFeeder/app/routers/simulation.py

import uuid as uuid_module

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.database import Event, Match, Player, Sport, Team, get_db
from app.routers.matches import _score_match
from app.schemas import ReplayPushResult, SimulateStartRequest, SimulateStartResponse, SimulationStatusRead
from app.services.backend_client import get_backend_client
from app.services.links import upsert_link
from app.services.simulation import (
    TOTAL_MINUTES,
    SimulationState,
    _load_entity_uuid_map,
    build_match_result_payload,
    get_simulation_state,
    is_running,
    list_simulations,
    request_stop,
    start_simulation,
)
from app.services.sport_resolver import SportType, resolve_sport_type

router = APIRouter()


def _not_found(entity: str, entity_id: int):
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{entity} {entity_id} not found")


def _resolve_match_sport_type(db, match: Match) -> SportType:
    sport = db.query(Sport).filter_by(id=match.sport_id).first()
    sport_type = resolve_sport_type(sport.name if sport else None)
    if sport_type is SportType.UNKNOWN:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Unknown sport '{sport.name if sport else None}' — seed the sports table with a known alias",
        )
    return sport_type


def _get_or_create_match(db, payload: SimulateStartRequest) -> Match:
    if payload.match_id is not None:
        match = db.query(Match).filter_by(id=payload.match_id).first()
        if not match:
            _not_found("Match", payload.match_id)
        return match

    if payload.home_team_id is None or payload.away_team_id is None or payload.sport_id is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Provide either match_id or home_team_id + away_team_id + sport_id",
        )
    if payload.home_team_id == payload.away_team_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Home and away teams must be different")

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
        sport_id=payload.sport_id,
        status="scheduled",
    )
    db.add(match)
    db.commit()
    db.refresh(match)
    return match


@router.post("/simulate", response_model=SimulateStartResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_match_simulation(payload: SimulateStartRequest, request: Request, db=Depends(get_db)):
    match = _get_or_create_match(db, payload)

    if is_running(match.id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Match {match.id} already has a running simulation; stop it first",
        )

    sport_type = _resolve_match_sport_type(db, match)

    if payload.sporty_match_id:
        upsert_link(db, "match", match.id, payload.sporty_match_id)
    if payload.sporty_home_team_id:
        upsert_link(db, "team", match.home_team_id, payload.sporty_home_team_id)
    if payload.sporty_away_team_id:
        upsert_link(db, "team", match.away_team_id, payload.sporty_away_team_id)

    event_rates = getattr(request.app.state, "event_rates", None)
    start_simulation(match.id, sport_type, event_rates, get_backend_client())

    return SimulateStartResponse(
        match_id=match.id,
        status="running",
        status_url=f"/simulate/{match.id}/status",
    )


@router.get("/simulate", response_model=list[SimulationStatusRead])
def list_match_simulations():
    """All simulations this process knows about (running + recently ended).
    Drives the admin panel's live monitor without polling each match."""
    return [
        SimulationStatusRead(
            match_id=state.match_id,
            status=state.status,
            current_minute=state.current_minute,
            total_minutes=state.total_minutes,
            home_score=state.home_score,
            away_score=state.away_score,
            events_inserted=state.events_inserted,
            push_failures=state.push_failures,
            error=state.error,
        )
        for state in list_simulations()
    ]


@router.get("/simulate/{match_id}/status", response_model=SimulationStatusRead)
def simulation_status(match_id: int):
    state = get_simulation_state(match_id)
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No simulation found for match {match_id}",
        )
    return SimulationStatusRead(
        match_id=state.match_id,
        status=state.status,
        current_minute=state.current_minute,
        total_minutes=state.total_minutes,
        home_score=state.home_score,
        away_score=state.away_score,
        events_inserted=state.events_inserted,
        push_failures=state.push_failures,
        error=state.error,
    )


@router.post("/simulate/{match_id}/stop")
def stop_simulation(match_id: int):
    if not request_stop(match_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"No running simulation for match {match_id}",
        )
    return {"detail": f"Stop requested for match {match_id}; simulation halts after the current minute"}


@router.post("/matches/{match_id}/replay-push", response_model=ReplayPushResult)
async def replay_push(match_id: int, db=Depends(get_db)):
    """Re-send every stored event for a match to the Sporty backend (recovery
    after an outage). The backend dedupes on event_id, so replays are safe."""
    match = db.query(Match).filter_by(id=match_id).first()
    if not match:
        _not_found("Match", match_id)

    sport_type = _resolve_match_sport_type(db, match)

    sporty_match_id = _load_entity_uuid_map(db, "match", [match_id]).get(match_id)
    if sporty_match_id is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Match {match_id} has no sporty_match_id entity link; create one via POST /links first",
        )

    events = (
        db.query(Event)
        .filter_by(match_id=match_id)
        .order_by(Event.minute.asc(), Event.created_at.asc(), Event.id.asc())
        .all()
    )

    # Backfill idempotency keys on legacy rows so the backend can dedupe.
    backfilled = False
    for event in events:
        if event.event_id is None:
            event.event_id = str(uuid_module.uuid4())
            backfilled = True
    if backfilled:
        db.commit()

    player_team_map = {
        player.id: player.team_id
        for player in db.query(Player).filter(Player.sport_id == match.sport_id).all()
    }
    home_score, away_score = _score_match(db, match)

    state = SimulationState(
        match_id=match_id,
        sport_type=sport_type,
        total_minutes=TOTAL_MINUTES[sport_type],
        current_minute=max((event.minute or 0 for event in events), default=0),
        home_score=home_score,
        away_score=away_score,
    )
    mappings = {
        "match": sporty_match_id,
        "teams": _load_entity_uuid_map(db, "team", [match.home_team_id, match.away_team_id]),
        "players": _load_entity_uuid_map(db, "player", list(player_team_map)),
    }
    event_dicts = [
        {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "player_id": event.player_id,
            "team_id": player_team_map.get(event.player_id),
            "minute": event.minute,
        }
        for event in events
    ]
    payload = build_match_result_payload(state, mappings, match.status, event_dicts)
    delivered = await get_backend_client().push_match_result(payload)
    return ReplayPushResult(match_id=match_id, delivered=delivered, events_sent=len(event_dicts))
