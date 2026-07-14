# /home/sam069/projects/SportyDataFeeder/app/routers/matches.py

import json

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.exc import IntegrityError

from app.database import (
    Event,
    Match,
    MatchPrediction,
    Player,
    PlayerMatchRating,
    Sport,
    Team,
    get_db,
)
from app.schemas import MatchCreate, MatchDetailRead, MatchRead
from app.services.backend_client import feeder_match_external_ref, get_backend_client
from app.services.links import delete_link, get_sporty_uuid, upsert_link
from app.services.scoring_rules import score_events
from app.services.simulation import LINEUP_SIZE, _select_lineup, is_running
from app.services.sport_resolver import SportType, resolve_sport_type

router = APIRouter()


def _not_found(entity: str, entity_id: int):
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{entity} {entity_id} not found")


def _score_match(db, match: Match) -> tuple[int, int]:
    sport = db.query(Sport).filter_by(id=match.sport_id).first()
    sport_type = resolve_sport_type(sport.name if sport else None)
    if sport_type is SportType.UNKNOWN:
        # Pre-resolver behaviour: anything that is not basketball scored as football.
        sport_type = SportType.FOOTBALL

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
    match_team_ids = {match.home_team_id, match.away_team_id}

    scorable_events = []
    for event in events:
        if event.player_id is None:
            continue
        event_team_id = player_team_map.get(event.player_id)
        if event_team_id not in match_team_ids:
            continue
        extra_data = None
        if event.extra:
            try:
                extra_data = json.loads(event.extra)
            except json.JSONDecodeError:
                extra_data = None
        scorable_events.append({"event_type": event.event_type, "team_id": event_team_id, "extra": extra_data})

    scores = score_events(scorable_events, sport_type)
    return scores.get(match.home_team_id, 0), scores.get(match.away_team_id, 0)


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


@router.post("/{match_id}/schedule-on-sporty")
async def schedule_on_sporty(match_id: int, db=Depends(get_db)):
    """Register this fixture on the Sporty backend as a *scheduled* match (so it
    appears on the Sporty matches page as upcoming) and link it — WITHOUT
    simulating or registering players. Idempotent: schedule_match get-or-creates
    by fixture identity, so re-pushing returns the same Sporty match id."""
    match = db.query(Match).filter_by(id=match_id).first()
    if not match:
        _not_found("Match", match_id)

    sport = db.query(Sport).filter_by(id=match.sport_id).first()
    sport_type = resolve_sport_type(sport.name if sport else None)
    if sport_type is SportType.UNKNOWN:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Unknown sport '{sport.name if sport else None}'",
        )

    home = db.query(Team).filter_by(id=match.home_team_id).first()
    away = db.query(Team).filter_by(id=match.away_team_id).first()
    if not home or not away:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="Match teams not found")

    client = get_backend_client()
    schedule = await client.schedule_match(
        {
            "sport": sport_type.value,
            "home_team": home.name,
            "away_team": away.name,
            "match_date": match.match_date.isoformat() if match.match_date else None,
            "external_ref": feeder_match_external_ref(match.id),
        }
    )
    sporty_match_id = schedule["sporty_match_id"]
    upsert_link(db, "match", match.id, sporty_match_id)
    return {
        "feeder_match_id": match.id,
        "sporty_match_id": sporty_match_id,
        "created": schedule.get("created", True),
    }


@router.delete("/{match_id}/schedule-on-sporty")
async def unschedule_on_sporty(match_id: int, force: bool = False, db=Depends(get_db)):
    """Delete this fixture's scheduled match on the Sporty backend and drop the
    feeder→Sporty link. The inverse of POST /{match_id}/schedule-on-sporty.

    The backend refuses to delete a match that is currently LIVE (409) unless
    `force=true` (to clear a simulation orphaned in `live`); a match already
    absent on the backend (404) is treated as success and the stale link is
    cleaned up locally."""
    match = db.query(Match).filter_by(id=match_id).first()
    if not match:
        _not_found("Match", match_id)

    sporty_match_id = get_sporty_uuid(db, "match", match_id)
    if not sporty_match_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Match is not linked to a Sporty match",
        )

    client = get_backend_client()
    try:
        result = await client.delete_match(sporty_match_id, force=force)
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code
        if code == status.HTTP_404_NOT_FOUND:
            # Already gone on the backend — drop the stale link, report success.
            delete_link(db, "match", match_id)
            return {
                "feeder_match_id": match_id,
                "sporty_match_id": sporty_match_id,
                "deleted": True,
                "already_absent": True,
            }
        detail = "Backend refused to delete the match"
        try:
            detail = exc.response.json().get("detail", detail)
        except (ValueError, AttributeError):
            pass
        raise HTTPException(status_code=code, detail=detail) from exc

    delete_link(db, "match", match_id)
    return {
        "feeder_match_id": match_id,
        "sporty_match_id": sporty_match_id,
        "deleted": True,
        "backend": result,
    }


@router.delete("/{match_id}")
async def delete_match(match_id: int, force: bool = False, db=Depends(get_db)):
    """Delete a feeder fixture entirely — its local events/predictions/ratings
    and the feeder record, plus (if linked) the scheduled match on Sporty.

    Refuses (409) while a simulation is running for this match, or if the linked
    Sporty match is currently live — stop/finish it first. Pass `force=true` to
    override both guards and clean up a fixture whose simulation is orphaned/stuck
    (e.g. the feeder died mid-run). If the fixture is not linked (or already gone
    on Sporty), the local delete still proceeds."""
    match = db.query(Match).filter_by(id=match_id).first()
    if not match:
        _not_found("Match", match_id)

    if is_running(match_id) and not force:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A simulation is running for this match — stop it, or pass force=true to override",
        )

    # Remove the scheduled match on Sporty first so the fixture also disappears
    # from the Sporty matches page. A 404 (already gone / unlinked) is fine; a
    # 409 means it's live on Sporty — abort rather than half-delete (unless
    # force, in which case the backend override already prevented the 409).
    sporty_match_id = get_sporty_uuid(db, "match", match_id)
    sporty_result = None
    if sporty_match_id:
        try:
            sporty_result = await get_backend_client().delete_match(sporty_match_id, force=force)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != status.HTTP_404_NOT_FOUND:
                detail = "Backend refused to delete the Sporty match"
                try:
                    detail = exc.response.json().get("detail", detail)
                except (ValueError, AttributeError):
                    pass
                raise HTTPException(status_code=exc.response.status_code, detail=detail) from exc
            sporty_result = {"deleted": True, "already_absent": True}
        except httpx.HTTPError:
            # Backend unreachable — proceed with the local delete regardless.
            sporty_result = {"deleted": False, "error": "backend unreachable"}

    # No ON DELETE CASCADE on these FKs, so clear children before the match row.
    db.query(Event).filter_by(match_id=match_id).delete(synchronize_session=False)
    db.query(MatchPrediction).filter_by(match_id=match_id).delete(synchronize_session=False)
    db.query(PlayerMatchRating).filter_by(match_id=match_id).delete(synchronize_session=False)
    delete_link(db, "match", match_id, commit=False)
    db.delete(match)
    db.commit()

    return {
        "feeder_match_id": match_id,
        "deleted": True,
        "sporty_match_id": sporty_match_id,
        "sporty": sporty_result,
    }


def _match_lineups(db, match: Match) -> dict:
    """The players that play in this match's simulation, per team — the same
    deterministic selection (`_select_lineup`) the simulator uses."""
    sport = db.query(Sport).filter_by(id=match.sport_id).first()
    sport_type = resolve_sport_type(sport.name if sport else None)
    size = LINEUP_SIZE.get(sport_type, 11)

    def team_lineup(team_id: int) -> dict:
        team = db.query(Team).filter_by(id=team_id).first()
        roster = (
            db.query(Player).filter_by(team_id=team_id).order_by(Player.id.asc()).all()
        )
        chosen = _select_lineup(roster, size, None)
        return {
            "team_id": team_id,
            "team_name": team.name if team else None,
            "players": [
                {"id": p.id, "name": p.name, "position": p.position} for p in chosen
            ],
        }

    return {
        "match_id": match.id,
        "lineup_size": size,
        "home": team_lineup(match.home_team_id),
        "away": team_lineup(match.away_team_id),
    }


@router.get("/{match_id}/lineups")
def get_match_lineups(match_id: int, db=Depends(get_db)):
    """Players playing in this match's simulation, grouped by team."""
    match = db.query(Match).filter_by(id=match_id).first()
    if not match:
        _not_found("Match", match_id)
    return _match_lineups(db, match)


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
