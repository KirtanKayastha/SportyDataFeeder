# /home/sam069/projects/SportyDataFeeder/app/services/simulation.py
#
# Stat-weighted live simulation (PRD R-4.1). Runs as an asyncio background
# task: POST /simulate returns 202 immediately and the loop Bernoulli-samples
# per-player per-minute event probabilities (event_rates.pkl, cold-start
# fallback per R-3.1). Every fired event gets a UUID event_id (idempotency
# key), is written to the local events table, and the minute batch is pushed
# to the Sporty backend in ONE HTTP call. Scores come from scoring_rules
# (R-2.4) — never computed here.

import asyncio
import json
import logging
import unicodedata
import uuid
from dataclasses import dataclass, field

import numpy

from app.config import get_settings
from app.database import EntityLink, Event, Match, Player, PlayerMatchRating, SessionFactory, Sport
from app.services.backend_client import BackendClient
from app.services.features import BASKETBALL_FALLBACK_RATES, FOOTBALL_FALLBACK_RATES
from app.services.rater import find_man_of_match, rate_players
from app.services.scoring_rules import BASKETBALL_POINT_VALUES, event_score_value
from app.services.sport_resolver import SportType, resolve_sport_type

logger = logging.getLogger(__name__)

TOTAL_MINUTES = {SportType.FOOTBALL: 90, SportType.BASKETBALL: 48}
# On-court players per side: football 11, basketball 5 (not the 10-man roster —
# running 10 players for the full 48 minutes doubled the real on-court minutes
# and inflated every basketball stat ~2x).
LINEUP_SIZE = {SportType.FOOTBALL: 11, SportType.BASKETBALL: 5}

# Demo affordance: a `featured` player's primary scoring event gets this
# per-minute probability floor so they reliably register a stat in a single
# simulated match (a defender/striker the dice would usually skip otherwise).
# Applied AFTER calibration so it isn't scaled back to the league average.
# Expected events over a match ≈ floor × total_minutes (≈2.7 goals for football),
# which makes a goalless run for the featured player vanishingly likely.
FEATURED_RATE_FLOOR = {
    SportType.FOOTBALL: {"goal": 0.03},
    SportType.BASKETBALL: {"point_2": 0.06, "point_3": 0.03},
}

# Real league HOME/AWAY scoring averages used to calibrate simulated scoring.
# Calibrating home and away separately bakes in home advantage, so simulated
# home-win rates approach reality (EPL ~45% home; NBA ~58% home).
# Sources: 13 EPL seasons, 18 NBA seasons (see reports/).
FOOTBALL_HOME_GOALS = 1.55
FOOTBALL_AWAY_GOALS = 1.25
BASKETBALL_HOME_POINTS = 104.9
BASKETBALL_AWAY_POINTS = 102.2

# Basketball has no draws: a regulation tie goes to 10-minute overtime periods,
# repeated until the tie is broken (capped for safety).
OVERTIME_MINUTES = 10
MAX_OVERTIME_PERIODS = 6


@dataclass
class SimulationState:
    match_id: int
    sport_type: SportType
    total_minutes: int
    status: str = "running"  # running | finished | stopped | error
    current_minute: int = 0
    home_score: int = 0
    away_score: int = 0
    events_inserted: int = 0
    push_failures: int = 0
    stop_requested: bool = False
    error: str | None = None
    task: asyncio.Task | None = field(default=None, repr=False)


# Match-scoped registry: concurrent simulations of different matches never
# share state. One running simulation per match (409 enforced at the router).
_simulations: dict[int, SimulationState] = {}


def get_simulation_state(match_id: int) -> SimulationState | None:
    return _simulations.get(match_id)


def is_running(match_id: int) -> bool:
    state = _simulations.get(match_id)
    return state is not None and state.status == "running"


def request_stop(match_id: int) -> bool:
    state = _simulations.get(match_id)
    if state is None or state.status != "running":
        return False
    state.stop_requested = True
    return True


def running_count() -> int:
    return sum(1 for state in _simulations.values() if state.status == "running")


def _fallback_rates(sport_type: SportType) -> dict[str, float]:
    if sport_type is SportType.BASKETBALL:
        return BASKETBALL_FALLBACK_RATES
    return FOOTBALL_FALLBACK_RATES


def calibrate_scoring_rates(rates_by_player, home_lineup, away_lineup, sport_type, total_minutes):
    """Scale only the SCORING-event rates so each side's expected score matches
    its real HOME / AWAY league average, regardless of how raw the per-player
    rates are (trained or league-average fallback). Home and away are scaled
    independently, which bakes in home advantage. Non-scoring events (cards,
    assists, rebounds) are left untouched.

    Returns (scaled_rates, (home_factor, away_factor)); a side's factor is 1.0
    when it has no scoring rate to scale. Within a side the factor is uniform,
    so each player's share of the scoring is preserved; only the level changes.
    Expected side score = total_minutes * sum(per-player scoring rate)."""
    if sport_type is SportType.FOOTBALL:
        scoring = ("goal",)
        home_target, away_target = FOOTBALL_HOME_GOALS, FOOTBALL_AWAY_GOALS

        def per_player(rates):
            return rates.get("goal", 0.0)
    elif sport_type is SportType.BASKETBALL:
        scoring = tuple(BASKETBALL_POINT_VALUES)
        home_target, away_target = BASKETBALL_HOME_POINTS, BASKETBALL_AWAY_POINTS

        def per_player(rates):
            return sum(BASKETBALL_POINT_VALUES[ev] * rates.get(ev, 0.0) for ev in BASKETBALL_POINT_VALUES)
    else:
        return rates_by_player, (1.0, 1.0)

    def _factor(lineup, target):
        expected = total_minutes * sum(per_player(rates_by_player[p.id]) for p in lineup)
        return target / expected if expected > 0 else 1.0

    home_factor = _factor(home_lineup, home_target)
    away_factor = _factor(away_lineup, away_target)
    factor_by_player = {p.id: home_factor for p in home_lineup}
    factor_by_player.update({p.id: away_factor for p in away_lineup})

    scaled = {}
    for player_id, rates in rates_by_player.items():
        factor = factor_by_player.get(player_id, 1.0)
        new_rates = dict(rates)
        for event_type in scoring:
            if event_type in new_rates:
                new_rates[event_type] *= factor
        scaled[player_id] = new_rates
    return scaled, (home_factor, away_factor)


def _load_entity_uuid_map(db, entity: str, feeder_ids: list[int]) -> dict[int, str]:
    if not feeder_ids:
        return {}
    links = (
        db.query(EntityLink)
        .filter(EntityLink.feeder_entity == entity, EntityLink.feeder_id.in_(feeder_ids))
        .all()
    )
    return {link.feeder_id: link.sporty_uuid for link in links}


def _fold_name(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value or "")
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    lowered = stripped.lower()
    for src, dst in (("ø", "o"), ("å", "a"), ("æ", "ae"), ("ß", "ss"), ("ł", "l"), ("đ", "d")):
        lowered = lowered.replace(src, dst)
    return " ".join(lowered.split())


def _select_lineup(roster: list[Player], size: int, featured: list[str] | None) -> list[Player]:
    """Lowest-id `size` players, except names in `featured` (case/accent-folded
    substring match) are pulled in first — so a specific drafted player is played
    rather than just the lowest-id 11. Mirrors the demo launcher's selection."""
    if not featured:
        return roster[:size]
    wanted = [_fold_name(f) for f in featured if f.strip()]
    chosen, seen = [], set()
    for player in roster:
        if any(term and term in _fold_name(player.name) for term in wanted):
            chosen.append(player)
            seen.add(player.id)
    for player in roster:
        if len(chosen) >= size:
            break
        if player.id not in seen:
            chosen.append(player)
            seen.add(player.id)
    return chosen[:size]


def _prepare(db, match_id: int, event_rates: dict | None, featured: list[str] | None = None) -> dict:
    """Synchronous setup: lineups, per-player rates, ID mappings."""
    match = db.query(Match).filter_by(id=match_id).first()
    if not match:
        raise ValueError(f"Match {match_id} not found")

    sport = db.query(Sport).filter_by(id=match.sport_id).first()
    sport_type = resolve_sport_type(sport.name if sport else None)
    if sport_type is SportType.UNKNOWN:
        raise ValueError(f"Unknown sport '{sport.name if sport else None}' for match {match_id}")

    lineup_size = LINEUP_SIZE[sport_type]
    team_lineups: dict[int, list[Player]] = {}
    for team_id in (match.home_team_id, match.away_team_id):
        roster = (
            db.query(Player)
            .filter_by(team_id=team_id)
            .order_by(Player.id.asc())
            .all()
        )
        if not roster:
            raise ValueError(f"Team {team_id} has no players; cannot simulate match {match_id}")
        team_lineups[team_id] = _select_lineup(roster, lineup_size, featured)
    home_lineup = team_lineups[match.home_team_id]
    away_lineup = team_lineups[match.away_team_id]
    lineups = home_lineup + away_lineup

    event_rates = event_rates or {}
    rates_by_player: dict[int, dict[str, float]] = {}
    cold_starts = 0
    for player in lineups:
        rates = event_rates.get(player.id)
        if not rates:
            rates = _fallback_rates(sport_type)
            cold_starts += 1
        rates_by_player[player.id] = rates
    if cold_starts:
        logger.warning(
            "Match %s: %s/%s lineup players have no trained rates; using league-average fallback",
            match_id, cold_starts, len(lineups),
        )

    if get_settings().SIMULATION_CALIBRATE:
        rates_by_player, (home_factor, away_factor) = calibrate_scoring_rates(
            rates_by_player, home_lineup, away_lineup, sport_type, TOTAL_MINUTES[sport_type]
        )
        if (home_factor, away_factor) != (1.0, 1.0):
            logger.info(
                "Match %s: scoring calibrated to home/away league averages (home x%.3f, away x%.3f)",
                match_id, home_factor, away_factor,
            )

    # Featured players get a scoring-event floor so they reliably register a
    # stat (applied last, after calibration). Matching mirrors _select_lineup.
    if featured:
        wanted = [_fold_name(f) for f in featured if f.strip()]
        floor = FEATURED_RATE_FLOOR.get(sport_type, {})
        for player in lineups:
            if any(term and term in _fold_name(player.name) for term in wanted):
                rates = dict(rates_by_player[player.id])
                for event_type, minimum in floor.items():
                    rates[event_type] = max(rates.get(event_type, 0.0), minimum)
                rates_by_player[player.id] = rates
                logger.info("Match %s: featured player %s (%s) scoring boosted", match_id, player.id, player.name)

    player_ids = [player.id for player in lineups]
    mappings = {
        "match": _load_entity_uuid_map(db, "match", [match_id]).get(match_id),
        "teams": _load_entity_uuid_map(db, "team", [match.home_team_id, match.away_team_id]),
        "players": _load_entity_uuid_map(db, "player", player_ids),
    }
    if mappings["match"] is None:
        logger.warning(
            "Match %s has no sporty_match_id link; pushes are skipped (events persist locally — "
            "map it via POST /links and use replay-push)",
            match_id,
        )

    return {
        "match": match,
        "sport_type": sport_type,
        "lineups": lineups,
        "player_team_map": {player.id: player.team_id for player in lineups},
        "rates_by_player": rates_by_player,
        "mappings": mappings,
    }


def _sample_minute_events(setup: dict, minute: int) -> list[dict]:
    """Bernoulli-sample each event type for each active player for one minute."""
    fired: list[dict] = []
    for player in setup["lineups"]:
        for event_type, probability in setup["rates_by_player"][player.id].items():
            p = min(max(float(probability), 0.0), 1.0)
            if p <= 0.0 or not numpy.random.binomial(1, p):
                continue
            extra = None
            if event_type in BASKETBALL_POINT_VALUES and setup["sport_type"] is SportType.BASKETBALL:
                extra = {"points": BASKETBALL_POINT_VALUES[event_type]}
            fired.append(
                {
                    "event_id": str(uuid.uuid4()),
                    "event_type": event_type,
                    "player_id": player.id,
                    "team_id": player.team_id,
                    "minute": minute,
                    "extra": extra,
                }
            )
    return fired


def build_match_result_payload(state: SimulationState, mappings: dict, status: str, events: list[dict]) -> dict:
    return {
        "sporty_match_id": mappings["match"],
        "sport": state.sport_type.value,
        "status": status,
        "home_score": state.home_score,
        "away_score": state.away_score,
        "current_minute": state.current_minute,
        "events": [
            {
                "event_id": event["event_id"],
                "event_type": event["event_type"],
                "sporty_player_id": mappings["players"].get(event["player_id"]),
                "sporty_team_id": mappings["teams"].get(event["team_id"]),
                "minute": event["minute"],
            }
            for event in events
        ],
    }


def build_player_ratings_payload(
    state: SimulationState,
    mappings: dict,
    ratings: dict[int, float],
    events_by_player: dict[int, list[str]],
    man_of_match: int | None,
) -> dict:
    return {
        "sporty_match_id": mappings["match"],
        "sport": state.sport_type.value,
        "man_of_match_sporty_player_id": mappings["players"].get(man_of_match),
        "ratings": [
            {
                "sporty_player_id": mappings["players"].get(player_id),
                "rating": rating,
                "goals": events_by_player.get(player_id, []).count("goal"),
                "assists": events_by_player.get(player_id, []).count("assist"),
                "minutes_played": state.current_minute,
                "events": events_by_player.get(player_id, []),
            }
            for player_id, rating in sorted(ratings.items())
        ],
    }


def _store_ratings(db, match_id: int, ratings: dict[int, float], man_of_match: int | None) -> None:
    db.query(PlayerMatchRating).filter_by(match_id=match_id).delete()
    for player_id, rating in ratings.items():
        db.add(
            PlayerMatchRating(
                match_id=match_id,
                player_id=player_id,
                rating=rating,
                is_man_of_match=player_id == man_of_match,
            )
        )
    db.commit()


async def run_simulation(
    state: SimulationState,
    event_rates: dict | None,
    client: BackendClient,
    featured: list[str] | None = None,
) -> SimulationState:
    """The background simulation loop. Never raises: failures mark the state
    (and match) as error; push failures are tolerated (R-4.2)."""
    db = SessionFactory()
    speed = get_settings().SIMULATION_SPEED
    events_by_player: dict[int, list[str]] = {}
    try:
        setup = _prepare(db, state.match_id, event_rates, featured)
        match = setup["match"]
        mappings = setup["mappings"]
        push_enabled = mappings["match"] is not None

        match.status = "live"
        db.commit()
        logger.info("Match %s: simulation started (%s, %s minutes)", state.match_id, state.sport_type.value, state.total_minutes)

        async def play_minute(minute: int) -> None:
            state.current_minute = minute
            minute_events = _sample_minute_events(setup, minute)

            for event in minute_events:
                db.add(
                    Event(
                        event_id=event["event_id"],
                        match_id=state.match_id,
                        event_type=event["event_type"],
                        player_id=event["player_id"],
                        minute=event["minute"],
                        extra=json.dumps(event["extra"]) if event["extra"] else None,
                    )
                )
                events_by_player.setdefault(event["player_id"], []).append(event["event_type"])
                delta = event_score_value(event["event_type"], event["extra"], state.sport_type)
                if delta:
                    if event["team_id"] == match.home_team_id:
                        state.home_score += delta
                    else:
                        state.away_score += delta
            db.commit()
            state.events_inserted += len(minute_events)
            logger.debug(
                "Match %s minute %s: %s events, score %s-%s",
                state.match_id, minute, len(minute_events), state.home_score, state.away_score,
            )

            # One HTTP call per minute tick — never per-event (R-4.1 step 5).
            if push_enabled and minute_events:
                payload = build_match_result_payload(state, mappings, "live", minute_events)
                if not await client.push_match_result(payload):
                    state.push_failures += 1

            await asyncio.sleep(speed)

        for minute in range(1, state.total_minutes + 1):
            if state.stop_requested:
                logger.info("Match %s: stop requested at minute %s", state.match_id, state.current_minute)
                break
            await play_minute(minute)

        # Basketball has no draws: a regulation tie goes to 10-minute overtime
        # periods until it is broken (capped for safety).
        if state.sport_type is SportType.BASKETBALL and not state.stop_requested:
            minute = state.total_minutes
            periods = 0
            while state.home_score == state.away_score and periods < MAX_OVERTIME_PERIODS:
                periods += 1
                logger.info(
                    "Match %s: tied %s-%s after %s min; overtime period %s",
                    state.match_id, state.home_score, state.away_score, minute, periods,
                )
                for _ in range(OVERTIME_MINUTES):
                    if state.stop_requested:
                        break
                    minute += 1
                    await play_minute(minute)
                if state.stop_requested:
                    break

        state.status = "stopped" if state.stop_requested else "finished"
        match.status = "finished"
        db.commit()

        ratings = rate_players(events_by_player, state.sport_type)
        man_of_match = find_man_of_match(ratings)
        if ratings:
            _store_ratings(db, state.match_id, ratings, man_of_match)

        if push_enabled:
            final_payload = build_match_result_payload(state, mappings, "finished", [])
            if not await client.push_match_result(final_payload):
                state.push_failures += 1
            if ratings:
                ratings_payload = build_player_ratings_payload(state, mappings, ratings, events_by_player, man_of_match)
                if not await client.push_player_ratings(ratings_payload):
                    state.push_failures += 1

        logger.info(
            "Match %s: simulation %s at minute %s, score %s-%s, %s events, %s push failures",
            state.match_id, state.status, state.current_minute,
            state.home_score, state.away_score, state.events_inserted, state.push_failures,
        )
    except Exception as exc:
        state.status = "error"
        state.error = str(exc)
        logger.exception("Match %s: simulation crashed; partial events retained locally", state.match_id)
        try:
            db.rollback()
            match = db.query(Match).filter_by(id=state.match_id).first()
            if match:
                match.status = "error"
                db.commit()
        except Exception:
            logger.exception("Match %s: failed to mark match as error", state.match_id)
    finally:
        db.close()
    return state


def start_simulation(
    match_id: int,
    sport_type: SportType,
    event_rates: dict | None,
    client: BackendClient,
    featured: list[str] | None = None,
) -> SimulationState:
    """Register state and schedule the background task on the running loop."""
    state = SimulationState(
        match_id=match_id,
        sport_type=sport_type,
        total_minutes=TOTAL_MINUTES[sport_type],
    )
    _simulations[match_id] = state
    state.task = asyncio.get_running_loop().create_task(
        run_simulation(state, event_rates, client, featured)
    )
    return state
