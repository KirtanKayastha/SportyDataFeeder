# /home/sam069/projects/SportyDataFeeder/app/routers/demo.py
#
# One-call demo launcher. Given a feeder match it: (1) registers the fixture on
# the Sporty backend (/feed/schedule-match) and links it, (2) registers the
# simulated lineup as backend players (/feed/register-players) and links each so
# pushed events carry a real sporty_player_id, (3) optionally sets up a demo
# fantasy lineup so a user's total scores, and (4) starts the live simulation.
#
# This is the glue the integration was missing: nothing previously turned a
# feeder match + roster into a Sporty match + players for the live feed.

import logging
import unicodedata

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from app.database import Match, Player, Sport, Team, get_db
from app.services.backend_client import get_backend_client
from app.services.links import upsert_link
from app.services.ml_models import predict_outcome_v2
from app.services.simulation import LINEUP_SIZE, is_running, start_simulation
from app.services.sport_resolver import SportType, resolve_sport_type

logger = logging.getLogger(__name__)
router = APIRouter()

# Feeder position codes → Sporty football position codes.
_FOOTBALL_POSITION = {"G": "GKP", "GK": "GKP", "GKP": "GKP", "D": "DEF", "DEF": "DEF",
                      "M": "MID", "MID": "MID", "F": "FWD", "FWD": "FWD", "ST": "FWD"}


class DemoLaunchRequest(BaseModel):
    match_id: int
    # All-in-one throwaway demo defaults. For a REAL league (your own users draft
    # the registered players), call with fantasy_demo=false and simulate=false to
    # "prepare" (schedule + register players + link + prediction), let users draft,
    # then POST /simulate. See DEMO_RUNBOOK.md.
    fantasy_demo: bool = True    # auto-create a throwaway demo user fantasy lineup
    simulate: bool = True        # start the simulation now (false = prepare only)
    push_prediction: bool = True  # push the outcome prediction to the backend
    # Real-league flow: instead of CREATING feeder-owned players, map the
    # simulated lineup onto the players that ALREADY exist in the backend (the
    # ones your users drafted) by name, so the simulated match credits their
    # real fantasy teams. Unmatched simulated players just don't score.
    resolve_existing: bool = False
    # Names to force into the simulated lineup (matched by case/accent-folded
    # substring against each team's roster). Use this to guarantee a specific
    # drafted player is simulated — by default a team's lowest-id 11 play, which
    # may not be the player a user drafted. Featured players fill first, then the
    # rest of the lineup is filled by id up to the sport's lineup size.
    featured_players: list[str] = []
    # Simulation pace is controlled by the SIMULATION_SPEED env var.


def _position(sport_type: SportType, raw: str | None) -> str:
    if sport_type is SportType.BASKETBALL:
        return (raw or "G")[:20]
    return _FOOTBALL_POSITION.get((raw or "").upper(), "MID")


def _fold(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value or "")
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    lowered = stripped.lower()
    for src, dst in (("ø", "o"), ("å", "a"), ("æ", "ae"), ("ß", "ss"), ("ł", "l"), ("đ", "d")):
        lowered = lowered.replace(src, dst)
    return " ".join(lowered.split())


def _lineup(db, team_id: int, sport_type: SportType, featured: list[str] | None = None) -> list[Player]:
    size = LINEUP_SIZE[sport_type]
    roster = db.query(Player).filter_by(team_id=team_id).order_by(Player.id.asc()).all()
    if not featured:
        return roster[:size]
    wanted = [_fold(f) for f in featured if f.strip()]
    chosen, seen = [], set()
    for player in roster:  # featured first (substring match), preserving id order
        folded = _fold(player.name)
        if any(term and term in folded for term in wanted):
            chosen.append(player)
            seen.add(player.id)
    for player in roster:  # fill the rest by id
        if len(chosen) >= size:
            break
        if player.id not in seen:
            chosen.append(player)
            seen.add(player.id)
    return chosen[:size]


@router.post("/demo/launch")
async def demo_launch(payload: DemoLaunchRequest, request: Request, db=Depends(get_db)):
    match = db.query(Match).filter_by(id=payload.match_id).first()
    if not match:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Match {payload.match_id} not found")
    if is_running(match.id):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Match {match.id} already simulating")

    sport = db.query(Sport).filter_by(id=match.sport_id).first()
    sport_type = resolve_sport_type(sport.name if sport else None)
    if sport_type is SportType.UNKNOWN:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=f"Unknown sport '{sport.name if sport else None}'")
    sport_slug = sport_type.value

    home = db.query(Team).filter_by(id=match.home_team_id).first()
    away = db.query(Team).filter_by(id=match.away_team_id).first()
    if not home or not away:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="Match teams not found")

    client = get_backend_client()

    # 1) Register the fixture on the backend and link it.
    schedule = await client.schedule_match({
        "sport": sport_slug,
        "home_team": home.name,
        "away_team": away.name,
        "match_date": match.match_date.isoformat() if match.match_date else None,
    })
    sporty_match_id = schedule["sporty_match_id"]
    upsert_link(db, "match", match.id, sporty_match_id, commit=False)

    # 2) Map the simulated lineup to backend players and link each, so pushed
    #    events carry a real sporty_player_id. Two modes:
    #      - register (default): CREATE feeder-owned players (throwaway demos).
    #      - resolve_existing:   map onto EXISTING players your users drafted, by
    #        name (no rows created); only matched players will score.
    lineup = (
        _lineup(db, home.id, sport_type, payload.featured_players)
        + _lineup(db, away.id, sport_type, payload.featured_players)
    )
    team_name = {home.id: home.name, away.id: away.name}
    entries = [
        {
            "external_ref": f"feeder:player:{p.id}",
            "name": p.name,
            "position": _position(sport_type, p.position),
            "real_team": team_name.get(p.team_id, ""),
            "cost": 5.0,
        }
        for p in lineup
    ]
    if payload.resolve_existing:
        resolved = await client.resolve_players({"sport": sport_slug, "players": entries})
    else:
        resolved = await client.register_players({"sport": sport_slug, "players": entries})
    draftable = []  # what your users can draft (or already drafted), so they score
    player_uuids: list[str] = []
    for p in lineup:
        sporty_player_id = resolved["players"].get(f"feeder:player:{p.id}")
        if sporty_player_id:
            upsert_link(db, "player", p.id, sporty_player_id, commit=False)
            player_uuids.append(sporty_player_id)
            draftable.append({"name": p.name, "real_team": team_name.get(p.team_id, ""),
                              "sporty_player_id": sporty_player_id})
    db.commit()

    # 3) Push the outcome prediction so the frontend prediction card populates.
    prediction = None
    if payload.push_prediction:
        bundle = getattr(
            request.app.state,
            "outcome_v2" if sport_type is SportType.FOOTBALL else "outcome_v2_basketball",
            None,
        )
        result = predict_outcome_v2(bundle, home.name, away.name)
        if result is not None:
            try:
                await client.push_prediction({
                    "sporty_match_id": sporty_match_id,
                    "home_win_prob": result["home_win_prob"],
                    "draw_prob": result["draw_prob"],
                    "away_win_prob": result["away_win_prob"],
                    "model_version": result["model_version"],
                })
                prediction = result
            except Exception as exc:
                logger.warning("prediction push failed: %s", exc)

    # 4) Optional throwaway demo fantasy lineup (skip for a real league).
    fantasy = None
    if payload.fantasy_demo and player_uuids:
        try:
            fantasy = await client.demo_setup({
                "sport": sport_slug,
                "sporty_match_id": sporty_match_id,
                "player_uuids": player_uuids[:9],
                "captain_uuid": player_uuids[0],
            })
        except Exception as exc:  # best-effort; never block the live feed
            logger.warning("demo-setup failed (live feed still works): %s", exc)
            fantasy = {"error": str(exc)}

    # 5) Start the live simulation now, or leave it for a later POST /simulate
    #    (prepare mode — gives your users time to draft the registered players).
    if payload.simulate:
        event_rates = getattr(request.app.state, "event_rates", None)
        start_simulation(match.id, sport_type, event_rates, client, payload.featured_players)

    return {
        "status": "launched" if payload.simulate else "prepared",
        "mode": "resolve_existing" if payload.resolve_existing else "register",
        "feeder_match_id": match.id,
        "sporty_match_id": sporty_match_id,
        "players_linked": len(player_uuids),
        "players_in_lineup": len(lineup),
        "draftable_players": draftable,
        "prediction": prediction,
        "fantasy_demo": fantasy,
        "status_url": f"/simulate/{match.id}/status" if payload.simulate else None,
        "frontend_url": f"/match/{sporty_match_id}",
    }
