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

# ── Injuries, penalties, possession (football only) ──────────────────────────
# Per-match incident counts are drawn ONCE at kickoff from capped distributions
# — never independent per-minute rolls, which would make incidents routine just
# because there are 90 minutes to roll against. Assumed real-world rates (tune
# freely): forced-off injuries ~82% of matches none / 15% one / 3% two;
# minor knocks (player plays on) ~70/25/5; penalties awarded ~72/26/2
# (≈0.30/match ≈ one every 3.4 matches). Injury timing is biased late
# (fatigue): triangular with mode at ~70% of the match.
INJURY_FORCED_OFF_DIST = ((0, 0.82), (1, 0.15), (2, 0.03))
INJURY_MINOR_DIST = ((0, 0.70), (1, 0.25), (2, 0.05))
PENALTY_COUNT_DIST = ((0, 0.72), (1, 0.26), (2, 0.02))
# Penalty resolution: ~78% scored, ~14% saved, remainder off target. A save
# books BOTH penalty_saved (keeper) and penalty_missed (taker) — the FPL
# convention where penalties_missed means "not scored".
PENALTY_SCORED_PROB = 0.78
PENALTY_SAVED_PROB = 0.14
# Knockout ties: 30 minutes of extra time through the normal minute loop, then
# a shootout — alternating kicks, 5 rounds each, sudden death after that.
EXTRA_TIME_MINUTES = 30
SHOOTOUT_CONVERSION = 0.76  # ~real elite conversion in shootouts
MAX_SHOOTOUT_ROUNDS = 25  # ponytail: safety cap, coin-flip winner beyond it

# Possession: a single mean-reverting share of the ball for the home side,
# stepped once per minute (AR(1)) and nudged by events (conceding side pushes
# up; a red card shifts the anchor for the rest of the match). Goal probability
# is scaled by share/anchor, so expected goals still match the calibrated
# league averages while "who has the ball" modulates minute-to-minute chances.
POSSESSION_PULL = 0.15
POSSESSION_NOISE = 0.06
POSSESSION_GOAL_NUDGE = 0.04
POSSESSION_RED_CARD_SHIFT = 0.08

# Real league HOME/AWAY scoring averages used to calibrate simulated scoring.
# Calibrating home and away separately bakes in home advantage, so simulated
# home-win rates approach reality (EPL ~45% home; NBA ~58% home).
# Sources: 13 EPL seasons, 18 NBA seasons (see reports/).
# The penalty mechanism adds its goals ON TOP of open-play sampling, so the
# open-play targets are the league averages minus the expected penalty goals
# (split evenly between home and away) — total scoring stays at league level.
_EXPECTED_PENALTY_GOALS = sum(k * p for k, p in PENALTY_COUNT_DIST) * PENALTY_SCORED_PROB
FOOTBALL_HOME_GOALS = 1.55 - _EXPECTED_PENALTY_GOALS / 2
FOOTBALL_AWAY_GOALS = 1.25 - _EXPECTED_PENALTY_GOALS / 2
BASKETBALL_HOME_POINTS = 104.9
BASKETBALL_AWAY_POINTS = 102.2

# Basketball has no draws: a regulation tie goes to 10-minute overtime periods,
# repeated until the tie is broken (capped for safety).
OVERTIME_MINUTES = 10
MAX_OVERTIME_PERIODS = 6

# Assists are NOT independent events — they only happen because a teammate
# scored. So we never sample "assist" on its own; instead, when a scoring event
# fires we credit a different teammate an assist with this probability (and not
# every goal/basket is assisted — solo efforts exist). Rates from real data:
# ~75% of EPL goals are assisted; ~58% of NBA made field goals are assisted
# (free throws are never assisted, so they're excluded below).
ASSIST_PROBABILITY = {
    SportType.FOOTBALL: 0.75,
    SportType.BASKETBALL: 0.58,
}
ASSISTABLE_EVENTS = {
    SportType.FOOTBALL: {"goal"},
    SportType.BASKETBALL: {"point_2", "point_3"},
}

# ── Substitutions & discipline ────────────────────────────────────────────────
# Bench sizes mirror the real rules: EPL squads name 9 substitutes (of whom at
# most 5 may be used); NBA rosters dress ~13 but a realistic playing rotation is
# 9-10, i.e. the starting 5 plus ~5 bench players cycling in and out freely.
BENCH_SIZE = {SportType.FOOTBALL: 9, SportType.BASKETBALL: 5}
FOOTBALL_MAX_SUBS = 5

# Football sub timing (per real usage): a small share are 1st-half (injury /
# tactical emergency), a burst at half-time, and the bulk between ~55' and ~85'.
SUB_FIRST_HALF_PROB = 0.08
SUB_HALF_TIME_PROB = 0.25

# Basketball rotation model: unlike football, players RETURN after resting.
# NBA teams substitute at dead balls in clusters — starters play 4-8 minute
# stints, sit a few minutes, and come back (32-36 total minutes; the bench
# fills the rest). We approximate that with a rotation checkpoint every 4
# simulated minutes where each team swaps 1-2 players: longest current stint
# goes off, most-rested bench player comes on. Over 48 minutes this yields
# ~30-40 substitution events per game and starter minutes in the mid-30s —
# both close to real NBA box scores.
BASKETBALL_ROTATION_INTERVAL = 4


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
    # Football extras: running possession split, and the shootout tally for
    # knockout matches still level after extra time (regulation score is
    # home_score/away_score; the shootout is tracked separately).
    possession_home_pct: float | None = None
    possession_away_pct: float | None = None
    shootout_home: int | None = None
    shootout_away: int | None = None
    shootout_winner_team_id: int | None = None
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


def list_simulations() -> list[SimulationState]:
    """All known simulation states (running + recently finished/stopped), most
    recently active first. Powers the admin panel's live monitor."""
    return sorted(
        _simulations.values(),
        key=lambda s: (s.status != "running", -s.match_id),
    )


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
    bench_size = BENCH_SIZE[sport_type]
    team_lineups: dict[int, list[Player]] = {}
    bench_by_team: dict[int, list[Player]] = {}
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
        starter_ids = {p.id for p in team_lineups[team_id]}
        bench_by_team[team_id] = [p for p in roster if p.id not in starter_ids][:bench_size]
    home_lineup = team_lineups[match.home_team_id]
    away_lineup = team_lineups[match.away_team_id]
    lineups = home_lineup + away_lineup
    bench = bench_by_team[match.home_team_id] + bench_by_team[match.away_team_id]

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

    # Possession anchor from PRE-calibration goal rates (calibration forces
    # both sides to league targets, which would erase the quality difference):
    # the stronger squad tends to hold more of the ball. Small home tilt on
    # top, clamped so neither side starts camped in the other's half.
    possession_anchor = 0.5
    if sport_type is SportType.FOOTBALL:
        home_exp = sum(rates_by_player[p.id].get("goal", 0.0) for p in home_lineup)
        away_exp = sum(rates_by_player[p.id].get("goal", 0.0) for p in away_lineup)
        if home_exp + away_exp > 0:
            possession_anchor = home_exp / (home_exp + away_exp)
        possession_anchor = min(max(possession_anchor + 0.02, 0.35), 0.65)

    home_factor = away_factor = 1.0
    if get_settings().SIMULATION_CALIBRATE:
        rates_by_player, (home_factor, away_factor) = calibrate_scoring_rates(
            rates_by_player, home_lineup, away_lineup, sport_type, TOTAL_MINUTES[sport_type]
        )
        if (home_factor, away_factor) != (1.0, 1.0):
            logger.info(
                "Match %s: scoring calibrated to home/away league averages (home x%.3f, away x%.3f)",
                match_id, home_factor, away_factor,
            )

    # Bench players get rates too (they may come on) and inherit their team's
    # calibration factor so a substitute scores at the same calibrated level as
    # the starters they replace. Calibration itself is computed on the starting
    # lineup: the on-court/on-pitch player count never changes, so expected
    # scoring stays at the league level.
    scoring_events = ("goal",) if sport_type is SportType.FOOTBALL else tuple(BASKETBALL_POINT_VALUES)
    for player in bench:
        rates = event_rates.get(player.id) or _fallback_rates(sport_type)
        factor = home_factor if player.team_id == match.home_team_id else away_factor
        scaled = dict(rates)
        for event_type in scoring_events:
            if event_type in scaled:
                scaled[event_type] = scaled[event_type] * factor
        rates_by_player[player.id] = scaled

    # Featured players get a scoring-event floor so they reliably register a
    # stat (applied last, after calibration). Matching mirrors _select_lineup.
    featured_ids: set[int] = set()
    if featured:
        wanted = [_fold_name(f) for f in featured if f.strip()]
        floor = FEATURED_RATE_FLOOR.get(sport_type, {})
        for player in lineups:
            if any(term and term in _fold_name(player.name) for term in wanted):
                featured_ids.add(player.id)
                rates = dict(rates_by_player[player.id])
                for event_type, minimum in floor.items():
                    rates[event_type] = max(rates.get(event_type, 0.0), minimum)
                rates_by_player[player.id] = rates
                logger.info("Match %s: featured player %s (%s) scoring boosted", match_id, player.id, player.name)

    pool = lineups + bench
    player_ids = [player.id for player in pool]
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
        "bench_by_team": bench_by_team,
        "featured_ids": featured_ids,
        "player_team_map": {player.id: player.team_id for player in pool},
        "rates_by_player": rates_by_player,
        "possession_anchor": possession_anchor,
        "mappings": mappings,
    }


def _pick_assister(scorer, teammates: list, rates_by_player: dict):
    """Pick a teammate (never the scorer) to credit with the assist, weighted by
    their assist rate so playmakers assist more often. Returns None when the
    scorer has no teammates on the pitch/court."""
    candidates = [p for p in teammates if p.id != scorer.id]
    if not candidates:
        return None
    weights = numpy.array(
        [max(float(rates_by_player.get(p.id, {}).get("assist", 0.0)), 1e-4) for p in candidates],
        dtype=float,
    )
    probs = weights / weights.sum()
    return candidates[int(numpy.random.choice(len(candidates), p=probs))]


def _make_event(event_type: str, player, minute: int, extra=None, related_player_id=None) -> dict:
    event = {
        "event_id": str(uuid.uuid4()),
        "event_type": event_type,
        "player_id": player.id,
        "team_id": player.team_id,
        "minute": minute,
        "extra": extra,
    }
    if related_player_id is not None:
        event["related_player_id"] = related_player_id
    return event


def _sample_minute_events(setup: dict, minute: int) -> list[dict]:
    """Bernoulli-sample each event type for each active player for one minute.

    Assists are special: they are never sampled standalone (a real assist only
    exists because a teammate scored). Instead, when a scoring event fires we
    credit a different teammate an assist with ASSIST_PROBABILITY."""
    sport_type = setup["sport_type"]
    rates_by_player = setup["rates_by_player"]
    assistable = ASSISTABLE_EVENTS.get(sport_type, set())
    assist_prob = ASSIST_PROBABILITY.get(sport_type, 0.0)
    # This minute's possession share (set by the loop; None for basketball):
    # the side with the ball gets proportionally more goal chances. Dividing by
    # the anchor keeps the expected value at the calibrated league level.
    share = setup.get("minute_share")
    home_team_id = setup["match"].home_team_id if share is not None else None
    anchor = setup.get("possession_anchor", 0.5)

    teammates_by_team: dict = {}
    for player in setup["lineups"]:
        teammates_by_team.setdefault(player.team_id, []).append(player)

    def make_event(event_type: str, player, extra=None) -> dict:
        return _make_event(event_type, player, minute, extra)

    fired: list[dict] = []
    for player in setup["lineups"]:
        for event_type, probability in rates_by_player[player.id].items():
            # Assists are coupled to scoring events below — never standalone.
            if event_type == "assist":
                continue
            p = min(max(float(probability), 0.0), 1.0)
            if share is not None and event_type == "goal":
                mult = (share / anchor) if player.team_id == home_team_id else ((1.0 - share) / (1.0 - anchor))
                p = min(p * mult, 1.0)
            if p <= 0.0 or not numpy.random.binomial(1, p):
                continue
            extra = None
            if event_type in BASKETBALL_POINT_VALUES and sport_type is SportType.BASKETBALL:
                extra = {"points": BASKETBALL_POINT_VALUES[event_type]}
            fired.append(make_event(event_type, player, extra))

            # A scoring event may be assisted by a teammate (not every one is).
            if event_type in assistable and numpy.random.binomial(1, assist_prob):
                assister = _pick_assister(player, teammates_by_team.get(player.team_id, []), rates_by_player)
                if assister is not None:
                    fired.append(make_event("assist", assister))
    return fired


def _is_keeper(player) -> bool:
    return (player.position or "").strip().upper().startswith("G")


def _draw_count(dist) -> int:
    values, probs = zip(*dist)
    return int(numpy.random.choice(values, p=probs))


def _draw_injuries(total_minutes: int) -> list[tuple[int, str]]:
    """Pre-drawn (minute, severity) injury schedule for the whole match, from
    the capped per-match count distributions; timing biased late (fatigue)."""
    injuries = [
        (int(numpy.random.triangular(5, 0.7 * total_minutes, total_minutes)), severity)
        for severity, dist in (("forced_off", INJURY_FORCED_OFF_DIST), ("minor", INJURY_MINOR_DIST))
        for _ in range(_draw_count(dist))
    ]
    return sorted(injuries)


def _draw_penalty_minutes(total_minutes: int) -> list[int]:
    return sorted(int(numpy.random.randint(1, total_minutes + 1)) for _ in range(_draw_count(PENALTY_COUNT_DIST)))


def _draw_sub_minutes(n: int, total_minutes: int) -> list[int]:
    """Realistic football substitution minutes: a small 1st-half share
    (injuries/tactical emergencies), a burst at half-time, and the bulk in the
    55'-85' window."""
    half = total_minutes // 2
    minutes = []
    for _ in range(n):
        roll = numpy.random.random()
        if roll < SUB_FIRST_HALF_PROB:
            minutes.append(int(numpy.random.randint(20, half)))
        elif roll < SUB_FIRST_HALF_PROB + SUB_HALF_TIME_PROB:
            minutes.append(half + 1)
        else:
            minutes.append(int(numpy.random.randint(half + 10, total_minutes - 5)))
    return sorted(minutes)


def _setup_dynamics(setup: dict) -> dict:
    """Mutable in-match state for substitutions and discipline: per-team bench,
    football sub windows, basketball stint/rest clocks, yellow-card memory,
    sent-off players and per-player minutes."""
    sport_type = setup["sport_type"]
    teams: dict[int, dict] = {}
    for team_id, bench in setup["bench_by_team"].items():
        entry: dict = {"bench": list(bench)}
        if sport_type is SportType.FOOTBALL:
            planned = min(FOOTBALL_MAX_SUBS, len(bench))
            entry["sub_minutes"] = _draw_sub_minutes(planned, TOTAL_MINUTES[sport_type]) if planned else []
            # Shared budget between planned (tactical) and injury-forced subs.
            entry["subs_left"] = FOOTBALL_MAX_SUBS
        else:
            entry["stint"] = {}
            entry["rest"] = {}
        teams[team_id] = entry
    dynamics = {"teams": teams, "yellows": set(), "sent_off": set(), "minutes": {}}
    if sport_type is SportType.FOOTBALL:
        total = TOTAL_MINUTES[sport_type]
        dynamics["injuries"] = _draw_injuries(total)
        dynamics["penalties"] = _draw_penalty_minutes(total)
        anchor = setup.get("possession_anchor", 0.5)
        dynamics["possession"] = {"anchor": anchor, "share": anchor, "sum_home": 0.0, "minutes": 0}
    return dynamics


def _swap_players(setup: dict, player_off, player_on) -> None:
    setup["lineups"].remove(player_off)
    setup["lineups"].append(player_on)


def _football_substitutions(setup: dict, dynamics: dict, minute: int) -> list[dict]:
    """Execute this minute's planned subs (max 5/team). The player coming off is
    a random outfielder — never the keeper, never a featured (demo) player, and
    never someone already sent off; the replacement comes from the bench and
    does NOT return (football subs are permanent)."""
    events: list[dict] = []
    for team_id, team in dynamics["teams"].items():
        while team["sub_minutes"] and team["sub_minutes"][0] <= minute:
            team["sub_minutes"].pop(0)
            # subs_left is shared with injury-forced subs, so a team that
            # burned windows on injuries skips its remaining planned ones.
            if not team["bench"] or team["subs_left"] <= 0:
                break
            active = [p for p in setup["lineups"] if p.team_id == team_id]
            eligible = [
                p for p in active
                if p.id not in setup["featured_ids"] and not _is_keeper(p)
            ] or [p for p in active if p.id not in setup["featured_ids"]] or active
            if not eligible:
                continue
            player_off = eligible[int(numpy.random.randint(len(eligible)))]
            player_on = team["bench"].pop(int(numpy.random.randint(len(team["bench"]))))
            team["subs_left"] -= 1
            _swap_players(setup, player_off, player_on)
            events.append(_make_event(
                "substitution", player_on, minute,
                extra={"player_out": player_off.id, "player_out_name": player_off.name},
                related_player_id=player_off.id,
            ))
    return events


def _basketball_rotation(setup: dict, dynamics: dict, minute: int) -> list[dict]:
    """Rotation checkpoint (see BASKETBALL_ROTATION_INTERVAL note): each team
    swaps 1-2 players — longest current stint off, most-rested bench player on.
    Unlike football, players return: whoever comes off joins the bench pool."""
    if minute <= 1 or (minute - 1) % BASKETBALL_ROTATION_INTERVAL != 0:
        return []
    events: list[dict] = []
    for team_id, team in dynamics["teams"].items():
        if not team["bench"]:
            continue
        swaps = int(numpy.random.randint(1, min(2, len(team["bench"])) + 1))
        for _ in range(swaps):
            if not team["bench"]:
                break
            active = [p for p in setup["lineups"] if p.team_id == team_id]
            candidates = [p for p in active if p.id not in setup["featured_ids"]] or active
            player_off = max(candidates, key=lambda p: (team["stint"].get(p.id, 0), -p.id))
            player_on = max(team["bench"], key=lambda p: (team["rest"].get(p.id, 0), -p.id))
            team["bench"].remove(player_on)
            team["bench"].append(player_off)
            team["stint"][player_on.id] = 0
            team["rest"][player_off.id] = 0
            _swap_players(setup, player_off, player_on)
            events.append(_make_event(
                "substitution", player_on, minute,
                extra={"player_out": player_off.id, "player_out_name": player_off.name},
                related_player_id=player_off.id,
            ))
    return events


def _injury_events(setup: dict, dynamics: dict, minute: int) -> list[dict]:
    """Fire this minute's pre-drawn injuries. A minor knock is just an event
    (player plays on). A forced-off injury consumes a substitution if the team
    still has bench and sub windows (keeper injuries prefer a bench keeper);
    with no sub available the player goes off and the team plays short."""
    events: list[dict] = []
    while dynamics["injuries"] and dynamics["injuries"][0][0] <= minute:
        _, severity = dynamics["injuries"].pop(0)
        candidates = [p for p in setup["lineups"] if p.id not in setup["featured_ids"]]
        if not candidates:
            continue
        player = candidates[int(numpy.random.randint(len(candidates)))]
        events.append(_make_event("injury", player, minute, extra={"severity": severity}))
        if severity != "forced_off":
            continue
        team = dynamics["teams"][player.team_id]
        if team["bench"] and team["subs_left"] > 0:
            bench = team["bench"]
            keepers = [p for p in bench if _is_keeper(p)] if _is_keeper(player) else []
            player_on = keepers[0] if keepers else bench[int(numpy.random.randint(len(bench)))]
            bench.remove(player_on)
            team["subs_left"] -= 1
            _swap_players(setup, player, player_on)
            events.append(_make_event(
                "substitution", player_on, minute,
                extra={"player_out": player.id, "player_out_name": player.name, "reason": "injury"},
                related_player_id=player.id,
            ))
        else:
            setup["lineups"].remove(player)
    return events


def _penalty_events(setup: dict, dynamics: dict, minute: int) -> list[dict]:
    """Resolve this minute's pre-drawn penalty kicks. The attacking side is
    drawn from the current possession share (the team with the ball wins the
    penalty); the taker is the on-pitch player with the best goal rate. A
    scored penalty is a normal goal event (extra.penalty=true) so match score
    and fantasy points need no special casing; a save credits the opposing
    keeper penalty_saved AND books penalty_missed for the taker."""
    events: list[dict] = []
    match = setup["match"]
    while dynamics["penalties"] and dynamics["penalties"][0] <= minute:
        dynamics["penalties"].pop(0)
        share = dynamics["possession"]["share"]
        attacking = match.home_team_id if numpy.random.random() < share else match.away_team_id
        attackers = [p for p in setup["lineups"] if p.team_id == attacking and not _is_keeper(p)]
        if not attackers:
            continue
        rates = setup["rates_by_player"]
        taker = max(attackers, key=lambda p: (rates.get(p.id, {}).get("goal", 0.0), -p.id))
        roll = numpy.random.random()
        if roll < PENALTY_SCORED_PROB:
            events.append(_make_event("goal", taker, minute, extra={"penalty": True}))
        else:
            events.append(_make_event("penalty_missed", taker, minute))
            if roll < PENALTY_SCORED_PROB + PENALTY_SAVED_PROB:
                keepers = [p for p in setup["lineups"] if p.team_id != attacking and _is_keeper(p)]
                if keepers:
                    events.append(_make_event("penalty_saved", keepers[0], minute))
    return events


def _update_possession(dynamics: dict) -> float:
    """One AR(1) step of the home side's ball share; accumulates the running
    average that becomes the full-time possession_pct."""
    pos = dynamics["possession"]
    share = pos["share"] + POSSESSION_PULL * (pos["anchor"] - pos["share"]) \
        + float(numpy.random.normal(0.0, POSSESSION_NOISE))
    pos["share"] = min(max(share, 0.25), 0.75)
    pos["sum_home"] += pos["share"]
    pos["minutes"] += 1
    return pos["share"]


def _nudge_possession(dynamics: dict, minute_events: list[dict], home_team_id: int) -> None:
    """Event-driven possession shifts: the side that concedes pushes for a
    response (temporary share nudge); a red card shifts the anchor for the
    rest of the match (ten men sit deep)."""
    pos = dynamics["possession"]
    for event in minute_events:
        home_side = event["team_id"] == home_team_id
        if event["event_type"] == "goal":
            pos["share"] += -POSSESSION_GOAL_NUDGE if home_side else POSSESSION_GOAL_NUDGE
        elif event["event_type"] == "red_card":
            pos["anchor"] += -POSSESSION_RED_CARD_SHIFT if home_side else POSSESSION_RED_CARD_SHIFT
    pos["share"] = min(max(pos["share"], 0.25), 0.75)
    pos["anchor"] = min(max(pos["anchor"], 0.30), 0.70)


def _run_shootout(setup: dict, state: SimulationState, minute: int) -> list[dict]:
    """Penalty shootout: alternating kicks, 5 rounds each, sudden death until
    decided. Kicks are shootout_goal / shootout_miss events — scoring_rules
    ignores those types, so the regulation/ET score is untouched. Takers cycle
    best-goal-rate-first. ponytail: rounds always complete (no early stop when
    mathematically decided)."""
    match = setup["match"]
    rates = setup["rates_by_player"]
    takers = {
        team_id: sorted(
            [p for p in setup["lineups"] if p.team_id == team_id],
            key=lambda p: (-rates.get(p.id, {}).get("goal", 0.0), p.id),
        )
        for team_id in (match.home_team_id, match.away_team_id)
    }
    events: list[dict] = []
    scores = {match.home_team_id: 0, match.away_team_id: 0}
    for round_no in range(1, MAX_SHOOTOUT_ROUNDS + 1):
        for team_id in (match.home_team_id, match.away_team_id):
            pool = takers[team_id]
            if not pool:  # entire side sent off/injured out — automatic miss
                continue
            kicker = pool[(round_no - 1) % len(pool)]
            if numpy.random.random() < SHOOTOUT_CONVERSION:
                scores[team_id] += 1
                events.append(_make_event("shootout_goal", kicker, minute))
            else:
                events.append(_make_event("shootout_miss", kicker, minute))
        if round_no >= 5 and scores[match.home_team_id] != scores[match.away_team_id]:
            break
    state.shootout_home = scores[match.home_team_id]
    state.shootout_away = scores[match.away_team_id]
    if scores[match.home_team_id] != scores[match.away_team_id]:
        state.shootout_winner_team_id = max(scores, key=lambda t: scores[t])
    else:
        state.shootout_winner_team_id = (
            match.home_team_id if numpy.random.random() < 0.5 else match.away_team_id
        )
        logger.warning(
            "Match %s: shootout undecided after %s rounds; coin-flip winner",
            state.match_id, MAX_SHOOTOUT_ROUNDS,
        )
    return events


def _apply_discipline(setup: dict, dynamics: dict, minute_events: list[dict], minute: int) -> None:
    """Football card rules: a second yellow to the same player becomes a red
    card, and any red (straight or second-yellow) sends the player off — the
    team plays on a player short, with no replacement allowed. Appends the
    derived red-card events to minute_events and shrinks the active lineup."""
    if setup["sport_type"] is not SportType.FOOTBALL:
        return
    by_id = {p.id: p for p in setup["lineups"]}
    for event in list(minute_events):
        pid = event["player_id"]
        if pid in dynamics["sent_off"]:
            continue
        if event["event_type"] == "yellow_card":
            if pid in dynamics["yellows"]:
                dynamics["sent_off"].add(pid)
                player = by_id.get(pid)
                if player is not None:
                    minute_events.append(_make_event(
                        "red_card", player, minute, extra={"reason": "second_yellow"}
                    ))
            else:
                dynamics["yellows"].add(pid)
        elif event["event_type"] == "red_card":
            dynamics["sent_off"].add(pid)
    if dynamics["sent_off"]:
        setup["lineups"][:] = [p for p in setup["lineups"] if p.id not in dynamics["sent_off"]]


def _advance_clocks(setup: dict, dynamics: dict) -> None:
    """Per-minute bookkeeping: minutes played for everyone on the pitch/court,
    and basketball stint/rest counters that drive the rotation."""
    for player in setup["lineups"]:
        dynamics["minutes"][player.id] = dynamics["minutes"].get(player.id, 0) + 1
    if setup["sport_type"] is SportType.BASKETBALL:
        for team_id, team in dynamics["teams"].items():
            for player in setup["lineups"]:
                if player.team_id == team_id:
                    team["stint"][player.id] = team["stint"].get(player.id, 0) + 1
            for player in team["bench"]:
                team["rest"][player.id] = team["rest"].get(player.id, 0) + 1


def build_match_result_payload(state: SimulationState, mappings: dict, status: str, events: list[dict]) -> dict:
    payload = {
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
                # Substitutions carry the player coming OFF so the backend can
                # publish a LINEUP_CHANGE; None (and ignored) for other events.
                "related_sporty_player_id": mappings["players"].get(event.get("related_player_id")),
                # Event detail passthrough (penalty flag, injury severity,
                # substitution reason) so the frontend can render it.
                "extra": event.get("extra"),
            }
            for event in events
        ],
    }
    # Football extras (absent for basketball / pre-feature replays): running
    # possession split and, for knockout ties, the shootout tally + winner.
    if state.possession_home_pct is not None:
        payload["possession_home_pct"] = state.possession_home_pct
        payload["possession_away_pct"] = state.possession_away_pct
    if state.shootout_home is not None:
        payload["shootout"] = {
            "home": state.shootout_home,
            "away": state.shootout_away,
            "winner_sporty_team_id": mappings["teams"].get(state.shootout_winner_team_id),
        }
    return payload


def build_player_ratings_payload(
    state: SimulationState,
    mappings: dict,
    ratings: dict[int, float],
    events_by_player: dict[int, list[str]],
    man_of_match: int | None,
    minutes_by_player: dict[int, int] | None = None,
) -> dict:
    minutes_by_player = minutes_by_player or {}
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
                # Real minutes (subs play partial matches); fall back to the
                # full clock for pre-substitution replays.
                "minutes_played": minutes_by_player.get(player_id, state.current_minute),
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
        dynamics = _setup_dynamics(setup)

        match.status = "live"
        db.commit()
        logger.info("Match %s: simulation started (%s, %s minutes)", state.match_id, state.sport_type.value, state.total_minutes)

        # Push the kickoff status change immediately, before any events exist —
        # otherwise Sporty stays "scheduled" until whichever minute first
        # produces an event, since the per-minute push below is gated on
        # minute_events being non-empty.
        if push_enabled:
            kickoff_payload = build_match_result_payload(state, mappings, "live", [])
            if not await client.push_match_result(kickoff_payload):
                state.push_failures += 1

        async def record_and_push(minute_events: list[dict]) -> None:
            """Persist a batch of events, apply score deltas, push ONE HTTP
            call (R-4.1 step 5). Shared by the minute loop and the shootout."""
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

            if push_enabled and minute_events:
                payload = build_match_result_payload(state, mappings, "live", minute_events)
                if not await client.push_match_result(payload):
                    state.push_failures += 1

        async def play_minute(minute: int) -> None:
            state.current_minute = minute
            # Substitutions happen first so players entering at minute m play
            # minute m; then possession steps, pre-drawn injuries/penalties
            # resolve, events are sampled from the post-sub lineup, and
            # football discipline (2nd yellow -> red -> off) is applied last.
            if state.sport_type is SportType.FOOTBALL:
                sub_events = _football_substitutions(setup, dynamics, minute)
                setup["minute_share"] = _update_possession(dynamics)
                sub_events += _injury_events(setup, dynamics, minute)
                sub_events += _penalty_events(setup, dynamics, minute)
            else:
                sub_events = _basketball_rotation(setup, dynamics, minute)
            minute_events = sub_events + _sample_minute_events(setup, minute)
            _apply_discipline(setup, dynamics, minute_events, minute)
            if state.sport_type is SportType.FOOTBALL:
                _nudge_possession(dynamics, minute_events, match.home_team_id)
                pos = dynamics["possession"]
                state.possession_home_pct = round(100.0 * pos["sum_home"] / pos["minutes"], 1)
                state.possession_away_pct = round(100.0 - state.possession_home_pct, 1)
            _advance_clocks(setup, dynamics)

            await record_and_push(minute_events)
            logger.debug(
                "Match %s minute %s: %s events, score %s-%s",
                state.match_id, minute, len(minute_events), state.home_score, state.away_score,
            )
            await asyncio.sleep(speed)

        for minute in range(1, state.total_minutes + 1):
            if state.stop_requested:
                logger.info("Match %s: stop requested at minute %s", state.match_id, state.current_minute)
                break
            await play_minute(minute)

        # Knockout football tied after 90: 30 minutes of extra time through
        # the same minute loop, then a shootout if still level. The shootout
        # tally never touches home_score/away_score.
        if (
            state.sport_type is SportType.FOOTBALL
            and bool(getattr(match, "knockout", False))
            and not state.stop_requested
            and state.home_score == state.away_score
        ):
            minute = state.total_minutes
            logger.info("Match %s: knockout tie after %s min; extra time", state.match_id, minute)
            for _ in range(EXTRA_TIME_MINUTES):
                if state.stop_requested:
                    break
                minute += 1
                await play_minute(minute)
            if not state.stop_requested and state.home_score == state.away_score:
                shootout_events = _run_shootout(setup, state, minute + 1)
                await record_and_push(shootout_events)
                logger.info(
                    "Match %s: shootout %s-%s, winner team %s",
                    state.match_id, state.shootout_home, state.shootout_away,
                    state.shootout_winner_team_id,
                )

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

        # Persist the full-time possession split locally (team-level stat, no
        # player) so it survives the in-memory simulation registry.
        if state.sport_type is SportType.FOOTBALL and dynamics["possession"]["minutes"]:
            db.add(
                Event(
                    event_id=str(uuid.uuid4()),
                    match_id=state.match_id,
                    event_type="possession",
                    player_id=None,
                    minute=state.current_minute,
                    extra=json.dumps({
                        "home_pct": state.possession_home_pct,
                        "away_pct": state.possession_away_pct,
                    }),
                )
            )
            state.events_inserted += 1

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
                ratings_payload = build_player_ratings_payload(
                    state, mappings, ratings, events_by_player, man_of_match,
                    minutes_by_player=dynamics["minutes"],
                )
                if not await client.push_player_ratings(ratings_payload):
                    state.push_failures += 1
            # The just-finished match may settle stored predictions: refresh
            # the model-performance scorecard on the backend (best-effort —
            # a failure never affects the simulation outcome).
            try:
                from app.services.prediction_metrics import build_metrics_push_payload

                if not await client.push_model_metrics(build_metrics_push_payload(db)):
                    state.push_failures += 1
            except Exception:
                logger.exception("Match %s: model-metrics push failed", state.match_id)

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
