# /home/sam069/projects/SportyDataFeeder/app/services/simulation.py

import json
import logging
import random
from datetime import date, datetime, time

from app.database import Event, Match, Player, Sport, Team

logger = logging.getLogger(__name__)

FOOTBALL_EVENT_TYPES = [
    ("Goal", "goal"),
    ("Assist", "assist"),
    ("Yellow Card", "yellow_card"),
    ("Red Card", "red_card"),
]
BASKETBALL_EVENT_TYPES = [
    ("2PT", "point_2"),
    ("3PT", "point_3"),
    ("FT", "free_throw"),
    ("Assist", "assist"),
    ("Rebound", "rebound"),
    ("Block", "block"),
    ("Steal", "steal"),
    ("Turnover", "turnover"),
]


def clean_text(value: str) -> str:
    return (value or "").strip()


def get_team_name(db, team_id):
    team = db.query(Team).filter_by(id=team_id).first()
    return team.name if team else "Unknown"


def get_player_name(db, player_id):
    player = db.query(Player).filter_by(id=player_id).first()
    return player.name if player else "Unknown"


def get_sport_name(db, sport_id):
    sport = db.query(Sport).filter_by(id=sport_id).first()
    return sport.name if sport else "Unknown"


def humanize_event_type(event_type: str) -> str:
    mapping = {
        "goal": "Goal",
        "assist": "Assist",
        "yellow_card": "Yellow Card",
        "red_card": "Red Card",
        "point_2": "2PT",
        "point_3": "3PT",
        "free_throw": "FT",
        "rebound": "Rebound",
        "block": "Block",
        "steal": "Steal",
        "turnover": "Turnover",
    }
    return mapping.get(event_type, event_type.replace("_", " ").title())


def get_supported_event_types(sport_name: str):
    sport_name = (sport_name or "").lower()
    if "basket" in sport_name:
        return BASKETBALL_EVENT_TYPES
    return FOOTBALL_EVENT_TYPES


def build_random_match_simulation(db, sport_name, home_team_id=None, away_team_id=None):
    normalized_sport_name = clean_text(sport_name).lower()
    sport_aliases = {
        "football": ["football", "soccer", "premier league"],
        "basketball": ["basketball", "nba"],
    }

    sport = None
    for alias in sport_aliases.get(normalized_sport_name, [normalized_sport_name]):
        sport = db.query(Sport).filter(Sport.name.ilike(alias)).first()
        if sport:
            break
        sport = db.query(Sport).filter(Sport.name.ilike(f"%{alias}%")).first()
        if sport:
            break

    if not sport:
        raise ValueError(f"Sport '{sport_name}' was not found in the database.")

    teams = db.query(Team).filter_by(sport_id=sport.id).order_by(Team.name.asc()).all()
    if len(teams) < 2:
        raise ValueError(f"Need at least two teams for {sport.name} before simulating a match.")

    if home_team_id is not None and away_team_id is not None:
        team_lookup = {team.id: team for team in teams}
        home_team = team_lookup.get(home_team_id)
        away_team = team_lookup.get(away_team_id)
        if not home_team or not away_team:
            raise ValueError(f"Selected teams are not available for {sport.name}.")
        if home_team.id == away_team.id:
            raise ValueError("Home and away teams must be different.")
    else:
        home_team, away_team = random.sample(teams, 2)

    players_by_team = {}
    all_players = (
        db.query(Player)
        .filter(Player.sport_id == sport.id)
        .order_by(Player.name.asc())
        .all()
    )
    player_team_map = {player.id: player.team_id for player in all_players}
    for team in teams:
        team_players = [player for player in all_players if player.team_id == team.id]
        players_by_team[team.id] = team_players

    def pick_team_player(team_id):
        team_players = players_by_team.get(team_id) or all_players
        if not team_players:
            raise ValueError(f"Need players for {sport.name} before simulating a match.")
        return random.choice(team_players)

    events = []
    if "basket" in sport.name.lower():
        event_count = random.randint(40, 70)
        weighted_event_types = [
            "point_2",
            "point_2",
            "point_2",
            "point_3",
            "point_3",
            "free_throw",
            "assist",
            "assist",
            "rebound",
            "rebound",
            "block",
            "steal",
            "turnover",
        ]
        point_values = {"point_2": 2, "point_3": 3, "free_throw": 1}

        for _ in range(event_count):
            event_type = random.choice(weighted_event_types)
            team = random.choice([home_team, away_team])
            player = pick_team_player(team.id)
            extra_payload = {}

            if event_type in point_values:
                extra_payload["points"] = point_values[event_type]
                if random.random() < 0.35:
                    assist_candidates = [p for p in players_by_team.get(team.id, []) if p.id != player.id]
                    if assist_candidates:
                        extra_payload["assist_player_id"] = random.choice(assist_candidates).id

            events.append(
                {
                    "minute": random.randint(1, 48),
                    "event_type": event_type,
                    "player_id": player.id,
                    "extra": extra_payload,
                }
            )

        events.sort(key=lambda item: (item["minute"], item["event_type"], item["player_id"]))
    else:
        goal_count = random.randint(0, 5)
        card_count = random.randint(0, 3)
        goal_minutes = random.sample(range(1, 91), goal_count) if goal_count else []
        card_minutes = random.sample(range(1, 91), card_count) if card_count else []

        for minute in goal_minutes:
            scoring_team = random.choice([home_team, away_team])
            scorer = pick_team_player(scoring_team.id)
            extra_payload = {}
            assist_candidates = [p for p in players_by_team.get(scoring_team.id, []) if p.id != scorer.id]
            if assist_candidates and random.random() < 0.7:
                extra_payload["assist_player_id"] = random.choice(assist_candidates).id
            events.append(
                {
                    "minute": minute,
                    "event_type": "goal",
                    "player_id": scorer.id,
                    "extra": extra_payload,
                }
            )

        for minute in card_minutes:
            card_type = random.choice(["yellow_card", "red_card"])
            card_team = random.choice([home_team, away_team])
            card_player = pick_team_player(card_team.id)
            events.append(
                {
                    "minute": minute,
                    "event_type": card_type,
                    "player_id": card_player.id,
                    "extra": {},
                }
            )

        events.sort(key=lambda item: (item["minute"], item["event_type"], item["player_id"]))

    home_score = 0
    away_score = 0
    for event_data in events:
        event_team_id = player_team_map.get(event_data["player_id"])
        if event_team_id not in {home_team.id, away_team.id}:
            continue
        if "basket" in sport.name.lower():
            if event_data["event_type"] in {"point_2", "point_3", "free_throw"}:
                points_value = event_data["extra"].get("points", 0)
                if event_team_id == home_team.id:
                    home_score += points_value
                else:
                    away_score += points_value
        elif event_data["event_type"] == "goal":
            if event_team_id == home_team.id:
                home_score += 1
            else:
                away_score += 1

    return {
        "sport_id": sport.id,
        "sport_name": sport.name,
        "home_team_id": home_team.id,
        "home_team_name": home_team.name,
        "away_team_id": away_team.id,
        "away_team_name": away_team.name,
        "events": events,
        "home_score": home_score,
        "away_score": away_score,
        "goal_events": sum(1 for event in events if event["event_type"] == "goal"),
        "card_events": sum(1 for event in events if event["event_type"] in {"yellow_card", "red_card"}),
        "total_events": len(events),
    }


def simulate_match_live(db, match_id):
    match = db.query(Match).filter_by(id=match_id).first()
    if not match:
        raise ValueError(f"Match {match_id} not found")

    sport = db.query(Sport).filter_by(id=match.sport_id).first()
    if not sport:
        raise ValueError(f"Sport {match.sport_id} not found for match {match_id}")

    plan = build_random_match_simulation(
        db,
        sport.name,
        home_team_id=match.home_team_id,
        away_team_id=match.away_team_id,
    )

    match.status = "live"
    db.commit()

    running_home_score = 0
    running_away_score = 0
    inserted_events = 0
    player_team_map = {
        player.id: player.team_id
        for player in db.query(Player).filter(Player.sport_id == plan["sport_id"]).all()
    }

    for index, event_data in enumerate(plan["events"], start=1):
        db.add(
            Event(
                match_id=match.id,
                event_type=event_data["event_type"],
                player_id=event_data["player_id"],
                minute=event_data["minute"],
                extra=json.dumps(event_data["extra"]) if event_data["extra"] else None,
            )
        )
        db.commit()
        inserted_events += 1

        event_team_id = player_team_map.get(event_data["player_id"])
        score_delta = 0
        if "basket" in plan["sport_name"].lower():
            if event_data["event_type"] in {"point_2", "point_3", "free_throw"}:
                score_delta = int(event_data["extra"].get("points", 0) or 0)
        elif event_data["event_type"] == "goal":
            score_delta = 1

        if event_team_id == plan["home_team_id"]:
            running_home_score += score_delta
        elif event_team_id == plan["away_team_id"]:
            running_away_score += score_delta

        player_name = get_player_name(db, event_data["player_id"])
        if event_data["event_type"] == "goal" and event_data["extra"].get("assist_player_id"):
            assist_name = get_player_name(db, event_data["extra"]["assist_player_id"])
            latest_event = f"⚽ Goal by {player_name} ({event_data['minute']}') • Assist: {assist_name}"
        elif event_data["event_type"] in {"point_2", "point_3", "free_throw"}:
            latest_event = f"🏀 {humanize_event_type(event_data['event_type'])} by {player_name} ({event_data['minute']}')"
        elif event_data["event_type"] in {"yellow_card", "red_card"}:
            latest_event = f"🟨 {humanize_event_type(event_data['event_type'])} for {player_name} ({event_data['minute']}')"
        else:
            latest_event = f"{humanize_event_type(event_data['event_type'])} by {player_name} ({event_data['minute']}')"

        logger.info(
            "Match %s: %s -> %s-%s (evt %s/%s)",
            match.id,
            latest_event,
            running_home_score,
            running_away_score,
            index,
            max(len(plan["events"]), 1),
        )

    match.status = "finished"
    db.commit()
    return inserted_events
