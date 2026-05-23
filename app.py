import json
import random
import queue
import threading
import time as time_module
from contextlib import contextmanager
from datetime import date, datetime, time

import pandas as pd
import streamlit as st

from database import Event, Match, Player, Session, Sport, Team


SIMULATION_LOCK = threading.Lock()
SIMULATION_TOAST_QUEUE = queue.Queue()


def init_simulation_state():
    if "simulations_active" not in st.session_state:
        st.session_state.simulations_active = {}
    if "simulation_running" not in st.session_state:
        st.session_state.simulation_running = False
    if "stop_requested" not in st.session_state:
        st.session_state.stop_requested = threading.Event()
    if "simulation_threads" not in st.session_state:
        st.session_state.simulation_threads = []
    if "simulation_summary" not in st.session_state:
        st.session_state.simulation_summary = []


def update_simulation_entry(match_id, **fields):
    with SIMULATION_LOCK:
        try:
            active = dict(st.session_state.simulations_active)
            entry = dict(active.get(match_id, {}))
            entry.update(fields)
            active[match_id] = entry
            st.session_state.simulations_active = active
        except Exception:
            # Streamlit session state may be unavailable in background threads;
            # fallback to printing a compact log so terminal still shows activity.
            try:
                print(f"[SIM] update_simulation_entry skipped for match {match_id}: {fields}")
            except Exception:
                pass


def remove_simulation_entry(match_id):
    with SIMULATION_LOCK:
        try:
            active = dict(st.session_state.simulations_active)
            if match_id in active:
                active.pop(match_id, None)
                st.session_state.simulations_active = active
        except Exception:
            try:
                print(f"[SIM] remove_simulation_entry skipped for match {match_id}")
            except Exception:
                pass


def queue_simulation_toast(message: str, icon: str = "✅"):
    SIMULATION_TOAST_QUEUE.put((message, icon))


def flush_simulation_toasts():
    while True:
        try:
            message, icon = SIMULATION_TOAST_QUEUE.get_nowait()
        except queue.Empty:
            break
        toast(message, icon=icon)


st.set_page_config(page_title="Sporty Feeder", layout="wide")
st.title("📋 Sporty Manual Data Feeder")

STATUS_OPTIONS = ["scheduled", "live", "finished"]
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


@contextmanager
def session_scope():
    session = Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def confirmation_container(label: str):
    popover = getattr(st, "popover", None)
    if popover is not None:
        return popover(label)
    return st.expander(label)


def toast(message: str, icon: str = "✅"):
    st.toast(message, icon=icon)


def format_status_badge(status: str) -> str:
    normalized_status = (status or "scheduled").lower()
    if normalized_status == "live":
        return "🟢 Live"
    if normalized_status == "finished":
        return "🔴 Finished"
    return "⚪ Scheduled"


def clean_text(value: str) -> str:
    return (value or "").strip()


def format_dt(value) -> str:
    if value is None:
        return "-"
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value)


def format_match_datetime(value) -> str:
    if value is None:
        return "-"
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value)


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


def get_team_name(session, team_id):
    team = session.query(Team).filter_by(id=team_id).first()
    return team.name if team else "Unknown"


def get_player_name(session, player_id):
    player = session.query(Player).filter_by(id=player_id).first()
    return player.name if player else "Unknown"


def get_sport_name(session, sport_id):
    sport = session.query(Sport).filter_by(id=sport_id).first()
    return sport.name if sport else "Unknown"


def build_match_label(match_row: dict) -> str:
    return (
        f"Match ID {match_row['id']}: {match_row['home_name']} vs {match_row['away_name']}"
        f" ({format_match_datetime(match_row['match_date'])})"
    )


def load_sports():
    with session_scope() as session:
        sports = session.query(Sport).order_by(Sport.name.asc()).all()
        rows = []
        for sport in sports:
            team_count = session.query(Team).filter_by(sport_id=sport.id).count()
            player_count = session.query(Player).filter_by(sport_id=sport.id).count()
            match_count = session.query(Match).filter_by(sport_id=sport.id).count()
            rows.append(
                {
                    "id": sport.id,
                    "name": sport.name,
                    "teams": team_count,
                    "players": player_count,
                    "matches": match_count,
                    "deletable": team_count == 0 and player_count == 0 and match_count == 0,
                }
            )
        return rows


def load_teams(sport_id=None):
    with session_scope() as session:
        query = session.query(Team)
        if sport_id is not None:
            query = query.filter_by(sport_id=sport_id)
        teams = query.order_by(Team.name.asc()).all()
        rows = []
        for team in teams:
            player_count = session.query(Player).filter_by(team_id=team.id).count()
            match_count = (
                session.query(Match)
                .filter((Match.home_team_id == team.id) | (Match.away_team_id == team.id))
                .count()
            )
            rows.append(
                {
                    "id": team.id,
                    "name": team.name,
                    "sport": team.sport.name if team.sport else "Unknown",
                    "players": player_count,
                    "matches": match_count,
                    "deletable": player_count == 0 and match_count == 0,
                }
            )
        return rows


def load_players(sport_id=None, team_id=None, limit=25, offset=0):
    with session_scope() as session:
        query = session.query(Player)
        if sport_id is not None:
            query = query.filter_by(sport_id=sport_id)
        if team_id is not None:
            query = query.filter_by(team_id=team_id)
        total = query.count()
        players = query.order_by(Player.name.asc()).offset(int(offset)).limit(int(limit)).all()
        rows = []
        for player in players:
            event_count = session.query(Event).filter_by(player_id=player.id).count()
            rows.append(
                {
                    "id": player.id,
                    "name": player.name,
                    "position": player.position or "",
                    "team": player.team.name if player.team else "Unknown",
                    "sport": player.sport.name if player.sport else "Unknown",
                    "events": event_count,
                    "deletable": event_count == 0,
                }
            )
        return rows, total


def load_matches(sport_id=None, statuses=None):
    with session_scope() as session:
        query = session.query(Match)
        if sport_id is not None:
            query = query.filter_by(sport_id=sport_id)
        if statuses:
            query = query.filter(Match.status.in_(statuses))
        matches = query.order_by(Match.match_date.desc(), Match.id.desc()).all()
        rows = []
        for match in matches:
            event_count = session.query(Event).filter_by(match_id=match.id).count()
            rows.append(
                {
                    "id": match.id,
                    "home_id": match.home_team_id,
                    "away_id": match.away_team_id,
                    "home_name": get_team_name(session, match.home_team_id),
                    "away_name": get_team_name(session, match.away_team_id),
                    "status": match.status or "scheduled",
                    "match_date": match.match_date,
                    "sport_id": match.sport_id,
                    "sport_name": get_sport_name(session, match.sport_id),
                    "events": event_count,
                }
            )
        return rows


def load_match_by_id(match_id):
    with session_scope() as session:
        match = session.query(Match).filter_by(id=match_id).first()
        if not match:
            return None
        return {
            "id": match.id,
            "home_id": match.home_team_id,
            "away_id": match.away_team_id,
            "home_name": get_team_name(session, match.home_team_id),
            "away_name": get_team_name(session, match.away_team_id),
            "status": match.status or "scheduled",
            "match_date": match.match_date,
            "sport_id": match.sport_id,
            "sport_name": get_sport_name(session, match.sport_id),
        }


def load_match_events(match_id):
    with session_scope() as session:
        events = (
            session.query(Event)
            .filter_by(match_id=match_id)
            .order_by(Event.minute.asc(), Event.created_at.asc(), Event.id.asc())
            .all()
        )
        rows = []
        for event in events:
            extra_data = {}
            if event.extra:
                try:
                    extra_data = json.loads(event.extra)
                except json.JSONDecodeError:
                    extra_data = {}
            extra_parts = []
            if "assist_player_id" in extra_data:
                extra_parts.append(f"Assist: {get_player_name(session, extra_data['assist_player_id'])}")
            if "points" in extra_data:
                extra_parts.append(f"Points: {extra_data['points']}")
            rows.append(
                {
                    "id": event.id,
                    "minute": event.minute,
                    "event_type": event.event_type,
                    "event_label": humanize_event_type(event.event_type),
                    "player_id": event.player_id,
                    "player_name": get_player_name(session, event.player_id),
                    "extra": extra_data,
                    "extra_text": " | ".join(extra_parts) if extra_parts else "",
                }
            )
        return rows


def load_all_match_groups(limit=10, offset=0):
    with session_scope() as session:
        matches = session.query(Match).order_by(Match.match_date.desc(), Match.id.desc()).all()
        total = len(matches)
        slice_matches = matches[int(offset) : int(offset) + int(limit)]
        groups = []
        for match in slice_matches:
            sport_name = get_sport_name(session, match.sport_id)
            events = (
                session.query(Event)
                .filter_by(match_id=match.id)
                .order_by(Event.minute.asc(), Event.created_at.asc(), Event.id.asc())
                .all()
            )
            players = session.query(Player).filter_by(sport_id=match.sport_id).all()
            player_team_map = {player.id: player.team_id for player in players}
            home_score = 0
            away_score = 0

            event_rows = []
            for event in events:
                extra_data = {}
                if event.extra:
                    try:
                        extra_data = json.loads(event.extra)
                    except json.JSONDecodeError:
                        extra_data = {}
                extra_parts = []
                if "assist_player_id" in extra_data:
                    extra_parts.append(f"Assist: {get_player_name(session, extra_data['assist_player_id'])}")
                if "points" in extra_data:
                    extra_parts.append(f"Points: {extra_data['points']}")

                event_team_id = player_team_map.get(event.player_id)
                if event_team_id in {match.home_team_id, match.away_team_id}:
                    if "basket" in sport_name.lower() and event.event_type in {"point_2", "point_3", "free_throw"}:
                        points_value = extra_data.get("points")
                        if points_value is None:
                            points_value = {"point_2": 2, "point_3": 3, "free_throw": 1}.get(event.event_type, 0)
                        if event_team_id == match.home_team_id:
                            home_score += int(points_value or 0)
                        else:
                            away_score += int(points_value or 0)
                    elif "basket" not in sport_name.lower() and event.event_type == "goal":
                        if event_team_id == match.home_team_id:
                            home_score += 1
                        else:
                            away_score += 1

                event_rows.append(
                    {
                        "id": event.id,
                        "minute": event.minute,
                        "event_type": humanize_event_type(event.event_type),
                        "player": get_player_name(session, event.player_id),
                        "extra": " | ".join(extra_parts) if extra_parts else "",
                    }
                )
            groups.append(
                {
                    "id": match.id,
                    "home_name": get_team_name(session, match.home_team_id),
                    "away_name": get_team_name(session, match.away_team_id),
                    "home_score": home_score,
                    "away_score": away_score,
                    "match_date": match.match_date,
                    "status": match.status or "scheduled",
                    "events": event_rows,
                }
            )
        return groups, total


def get_supported_event_types(sport_name: str):
    sport_name = (sport_name or "").lower()
    if "basket" in sport_name:
        return BASKETBALL_EVENT_TYPES
    return FOOTBALL_EVENT_TYPES


def build_random_match_simulation(sport_name, home_team_id=None, away_team_id=None):
    normalized_sport_name = clean_text(sport_name).lower()
    sport_aliases = {
        "football": ["football", "soccer", "premier league"],
        "basketball": ["basketball", "nba"],
    }

    with session_scope() as session:
        sport = None
        for alias in sport_aliases.get(normalized_sport_name, [normalized_sport_name]):
            sport = session.query(Sport).filter(Sport.name.ilike(alias)).first()
            if sport:
                break
            sport = session.query(Sport).filter(Sport.name.ilike(f"%{alias}%")).first()
            if sport:
                break

        if not sport:
            raise ValueError(f"Sport '{sport_name}' was not found in the database.")

        teams = session.query(Team).filter_by(sport_id=sport.id).order_by(Team.name.asc()).all()
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
            session.query(Player)
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


def simulate_match_live(sport_name, home_team_id=None, away_team_id=None, live_ui=None, stop_flag_key="stop_simulation"):
    plan = build_random_match_simulation(sport_name, home_team_id=home_team_id, away_team_id=away_team_id)

    with Session() as session:
        try:
            sport = session.query(Sport).filter_by(id=plan["sport_id"]).first()
            if not sport:
                raise ValueError(f"Sport '{sport_name}' was not found in the database.")

            match = Match(
                home_team_id=plan["home_team_id"],
                away_team_id=plan["away_team_id"],
                match_date=datetime.combine(date.today(), time.min),
                status="live",
                sport_id=sport.id,
            )
            session.add(match)
            session.commit()
            session.refresh(match)

            # Print to terminal so the server/terminal shows live match activity
            print(f"[SIM] Match started: ID={match.id} - {plan['home_team_name']} vs {plan['away_team_name']} (live)")
            if live_ui:
                live_ui["status"].markdown(
                    f"**Status:** <span style='color:#16a34a;font-weight:700;'>● Live</span>",
                    unsafe_allow_html=True,
                )
                live_ui["score"].markdown(
                    f"### {plan['home_team_name']} 0 - 0 {plan['away_team_name']}"
                )
                live_ui["latest"].info("Latest event will appear here.")
                live_ui["progress"].progress(0)

            running_home_score = 0
            running_away_score = 0
            inserted_events = 0

            for index, event_data in enumerate(plan["events"], start=1):
                if st.session_state.get(stop_flag_key, False):
                    break

                session.add(
                    Event(
                        match_id=match.id,
                        event_type=event_data["event_type"],
                        player_id=event_data["player_id"],
                        minute=event_data["minute"],
                        extra=json.dumps(event_data["extra"]) if event_data["extra"] else None,
                    )
                )
                session.commit()
                inserted_events += 1

                player_name = get_player_name(session, event_data["player_id"])
                event_team_id = session.query(Player.team_id).filter(Player.id == event_data["player_id"]).scalar()
                event_text = humanize_event_type(event_data["event_type"])
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

                extra_text = ""
                if event_data["event_type"] == "goal" and event_data["extra"].get("assist_player_id"):
                    assist_name = get_player_name(session, event_data["extra"]["assist_player_id"])
                    extra_text = f" • Assist: {assist_name}"
                elif event_data["event_type"] in {"point_2", "point_3", "free_throw"}:
                    extra_text = f" • Points: {event_data['extra'].get('points', 0)}"
                elif event_data["event_type"] in {"yellow_card", "red_card"}:
                    extra_text = f" • {humanize_event_type(event_data['event_type'])}"

                if live_ui:
                    live_ui["status"].markdown(
                        f"**Status:** <span style='color:#16a34a;font-weight:700;'>● Live</span>",
                        unsafe_allow_html=True,
                    )
                    live_ui["score"].markdown(
                        f"### {plan['home_team_name']} {running_home_score} - {running_away_score} {plan['away_team_name']}"
                    )
                    live_ui["latest"].info(
                        f"{event_text} by {player_name} ({event_data['minute']}'){extra_text}"
                    )
                    live_ui["progress"].progress(int((index / max(len(plan['events']), 1)) * 100))
                    toast(
                        f"{event_text} by {player_name} ({event_data['minute']}')",
                        icon="⚽" if event_data["event_type"] == "goal" else "🏀" if "point" in event_data["event_type"] else "🟨",
                    )
                    # also print a compact update to the server terminal
                    print(f"[SIM] Match ID={match.id}: {event_text} by {player_name} ({event_data['minute']}') -> {running_home_score}-{running_away_score}")
                    time_module.sleep(random.uniform(2, 5))

            session.query(Match).filter_by(id=match.id).update({"status": "finished"})
            session.commit()

            if live_ui:
                live_ui["status"].markdown(
                    f"**Status:** <span style='color:#dc2626;font-weight:700;'>● Finished</span>",
                    unsafe_allow_html=True,
                )
                live_ui["progress"].progress(100)

            # Final summary in terminal
            print(
                f"[SIM] Match finished: ID={match.id} - {plan['home_team_name']} {running_home_score} - {running_away_score} {plan['away_team_name']} (events_inserted={inserted_events})"
            )
            return {
                "sport_name": plan["sport_name"],
                "home_name": plan["home_team_name"],
                "away_name": plan["away_team_name"],
                "match_id": match.id,
                "goal_events": plan["goal_events"],
                "card_events": plan["card_events"],
                "total_events": inserted_events,
                "home_score": running_home_score,
                "away_score": running_away_score,
            }
        except Exception:
            session.rollback()
            raise


def simulate_random_match(sport_name, home_team_id=None, away_team_id=None):
    plan = build_random_match_simulation(sport_name, home_team_id=home_team_id, away_team_id=away_team_id)

    with session_scope() as session:
        sport = session.query(Sport).filter_by(id=plan["sport_id"]).first()
        if not sport:
            raise ValueError(f"Sport '{sport_name}' was not found in the database.")

        match = Match(
            home_team_id=plan["home_team_id"],
            away_team_id=plan["away_team_id"],
            match_date=datetime.combine(date.today(), time.min),
            status="finished",
            sport_id=sport.id,
        )
        session.add(match)
        session.flush()

        for event_data in plan["events"]:
            session.add(
                Event(
                    match_id=match.id,
                    event_type=event_data["event_type"],
                    player_id=event_data["player_id"],
                    minute=event_data["minute"],
                    extra=json.dumps(event_data["extra"]) if event_data["extra"] else None,
                )
            )

        return {
            "sport_name": plan["sport_name"],
            "home_name": plan["home_team_name"],
            "away_name": plan["away_team_name"],
            "match_id": match.id,
            "goal_events": plan["goal_events"],
            "card_events": plan["card_events"],
            "total_events": plan["total_events"],
            "home_score": plan["home_score"],
            "away_score": plan["away_score"],
        }


def sleep_with_stop(stop_event, duration_seconds):
    deadline = time_module.time() + max(0, duration_seconds)
    while time_module.time() < deadline:
        if stop_event.is_set():
            return True
        time_module.sleep(min(0.5, deadline - time_module.time()))
    return stop_event.is_set()


def build_unique_team_pairs_for_sport(sport_name: str, max_matches: int = 5):
    with session_scope() as session:
        sport = session.query(Sport).filter(Sport.name.ilike(sport_name)).first()
        if not sport:
            if sport_name == "football":
                sport = session.query(Sport).filter(Sport.name.ilike("%football%")).first()
            else:
                sport = session.query(Sport).filter(Sport.name.ilike("%basket%")).first()
        if not sport:
            raise ValueError(f"Sport '{sport_name}' was not found in the database.")

        teams = session.query(Team).filter_by(sport_id=sport.id).order_by(Team.name.asc()).all()
        if len(teams) < 2:
            raise ValueError(f"Need at least two teams for {sport.name} before simulating a match.")

        shuffled_teams = teams[:]
        random.shuffle(shuffled_teams)
        pair_count = min(max_matches, len(shuffled_teams) // 2)
        return sport.id, sport.name, [
            (
                {
                    "id": shuffled_teams[index].id,
                    "name": shuffled_teams[index].name,
                },
                {
                    "id": shuffled_teams[index + 1].id,
                    "name": shuffled_teams[index + 1].name,
                },
            )
            for index in range(0, pair_count * 2, 2)
        ]


def run_threaded_match_simulation(sport_name, home_team, away_team, stop_event):
    plan = build_random_match_simulation(sport_name, home_team_id=home_team["id"], away_team_id=away_team["id"])
    session = Session()
    try:
        match = Match(
            home_team_id=plan["home_team_id"],
            away_team_id=plan["away_team_id"],
            match_date=datetime.combine(date.today(), time.min),
            status="live",
            sport_id=plan["sport_id"],
        )
        session.add(match)
        session.commit()
        session.refresh(match)
        # Terminal output for started threaded simulation
        print(f"[SIM] Threaded match started: ID={match.id} - {plan['home_team_name']} vs {plan['away_team_name']}")
        update_simulation_entry(
            match.id,
            sport_name=plan["sport_name"],
            home_team=plan["home_team_name"],
            away_team=plan["away_team_name"],
            home_score=0,
            away_score=0,
            last_event="Match started",
            events_done=0,
            events_total=max(len(plan["events"]), 1),
            status="live",
        )

        player_team_map = {player.id: player.team_id for player in session.query(Player).filter(Player.sport_id == plan["sport_id"]).all()}
        running_home_score = 0
        running_away_score = 0
        total_events = max(len(plan["events"]), 1)
        target_duration = random.uniform(300, 420)
        per_event_delay = target_duration / total_events

        for index, event_data in enumerate(plan["events"], start=1):
            if stop_event.is_set():
                break

            event = Event(
                match_id=match.id,
                event_type=event_data["event_type"],
                player_id=event_data["player_id"],
                minute=event_data["minute"],
                extra=json.dumps(event_data["extra"]) if event_data["extra"] else None,
            )
            session.add(event)
            session.commit()

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

            player_name = get_player_name(session, event_data["player_id"])
            if event_data["event_type"] == "goal" and event_data["extra"].get("assist_player_id"):
                assist_name = get_player_name(session, event_data["extra"]["assist_player_id"])
                latest_event = f"⚽ Goal by {player_name} ({event_data['minute']}') • Assist: {assist_name}"
            elif event_data["event_type"] in {"point_2", "point_3", "free_throw"}:
                latest_event = f"🏀 {humanize_event_type(event_data['event_type'])} by {player_name} ({event_data['minute']}')"
            elif event_data["event_type"] in {"yellow_card", "red_card"}:
                latest_event = f"🟨 {humanize_event_type(event_data['event_type'])} for {player_name} ({event_data['minute']}')"
            else:
                latest_event = f"{humanize_event_type(event_data['event_type'])} by {player_name} ({event_data['minute']}')"

            if event_data["event_type"] == "goal":
                queue_simulation_toast(latest_event, icon="⚽")
            elif event_data["event_type"] in {"point_2", "point_3", "free_throw"}:
                queue_simulation_toast(latest_event, icon="🏀")
            else:
                queue_simulation_toast(latest_event, icon="🟨")
            # Print compact update for terminal visibility
            try:
                print(f"[SIM] Match ID={match.id}: {latest_event} -> {running_home_score}-{running_away_score} (evt {index}/{total_events})")
            except Exception:
                pass
            update_simulation_entry(
                match.id,
                home_score=running_home_score,
                away_score=running_away_score,
                last_event=latest_event,
                events_done=index,
                events_total=total_events,
                status="live",
            )

            if sleep_with_stop(stop_event, random.uniform(max(1.0, per_event_delay * 0.5), per_event_delay * 1.5)):
                break

        session.query(Match).filter_by(id=match.id).update({"status": "finished"})
        session.commit()

        final_entry = {
            "match_id": match.id,
            "sport": "football" if "basket" not in plan["sport_name"].lower() else "basketball",
            "sport_name": plan["sport_name"],
            "home_name": plan["home_team_name"],
            "away_name": plan["away_team_name"],
            "home_score": running_home_score,
            "away_score": running_away_score,
            "status": "finished",
            "last_event": "Match finished",
            "events_done": min(len(plan["events"]), total_events),
            "events_total": total_events,
        }
        update_simulation_entry(match.id, **{k: v for k, v in final_entry.items() if k != "match_id"})

        # Print final score summary to terminal
        try:
            print(f"[SIM] Threaded match finished: ID={match.id} - {plan['home_team_name']} {running_home_score} - {running_away_score} {plan['away_team_name']}")
        except Exception:
            pass

        with SIMULATION_LOCK:
            try:
                completed = list(st.session_state.get("simulation_summary", []))
                completed = [item for item in completed if item.get("match_id") != match.id]
                completed.append(final_entry)
                st.session_state.simulation_summary = completed
                st.session_state.auto_simulation_results = completed
            except Exception:
                # session_state may be unavailable in background threads; print fallback
                try:
                    print(f"[SIM] Could not update session_state.simulation_summary for match {match.id}; final_entry={final_entry}")
                except Exception:
                    pass

    except Exception as exc:
        session.rollback()
        update_simulation_entry(
            match.id if "match" in locals() and getattr(match, "id", None) else -1,
            status="finished",
            last_event=f"Simulation error: {exc}",
        )
        raise
    finally:
        session.close()
        Session.remove()


def render_auto_simulate_day_section():
    init_simulation_state()
    flush_simulation_toasts()

    st.caption("Generate 5 football matches and 5 basketball matches for testing.")

    results = st.session_state.get("auto_simulation_results", [])
    if results and not st.session_state.simulation_running:
        with st.expander("Last simulation summary", expanded=True):
            football_results = [result for result in results if result.get("sport") == "football"]
            basketball_results = [result for result in results if result.get("sport") == "basketball"]

            if football_results:
                st.markdown("**Football**")
                for result in football_results:
                    st.write(f"{result['home_name']} {result['home_score']} - {result['away_score']} {result['away_name']}")

            if basketball_results:
                st.markdown("**Basketball**")
                for result in basketball_results:
                    st.write(f"{result['home_name']} {result['home_score']} - {result['away_score']} {result['away_name']}")

    if st.session_state.simulation_running:
        active_matches = dict(st.session_state.simulations_active)
        total_matches = max(len(active_matches), 1)
        finished_matches = len([entry for entry in active_matches.values() if entry.get("status") == "finished"])

        st.info("Live simulation running. The page refreshes every second while matches are active.")
        st.progress(int((finished_matches / total_matches) * 100))

        match_items = sorted(active_matches.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else str(item[0]))
        for index in range(0, len(match_items), 2):
            columns = st.columns(2)
            for column, (match_id, match_state) in zip(columns, match_items[index : index + 2]):
                with column:
                    st.container()
                    st.markdown(f"#### {match_state.get('home_team', 'Home')} vs {match_state.get('away_team', 'Away')}")
                    status = match_state.get("status", "scheduled")
                    if status == "live":
                        st.markdown("<span style='color:#16a34a;font-weight:700;'>● Live</span>", unsafe_allow_html=True)
                    elif status == "finished":
                        st.markdown("<span style='color:#dc2626;font-weight:700;'>● Finished</span>", unsafe_allow_html=True)
                    else:
                        st.markdown("<span style='color:#6b7280;font-weight:700;'>● Scheduled</span>", unsafe_allow_html=True)
                    st.write(f"**Score:** {match_state.get('home_score', 0)} - {match_state.get('away_score', 0)}")
                    st.write(f"**Latest event:** {match_state.get('last_event', 'Waiting for first event...')}")
                    st.progress(int((match_state.get('events_done', 0) / max(match_state.get('events_total', 1), 1)) * 100))

        if st.session_state.get("stop_requested") and st.session_state.stop_requested.is_set():
            for thread in list(st.session_state.simulation_threads):
                thread.join(timeout=0.1)

        alive_threads = [thread for thread in st.session_state.simulation_threads if thread.is_alive()]
        if not alive_threads:
            st.session_state.simulation_running = False
            st.session_state.simulation_threads = []
            st.session_state.stop_requested.clear()
            st.session_state.auto_simulation_results = list(st.session_state.simulation_summary)
            flush_simulation_toasts()
            if st.session_state.auto_simulation_results:
                football_count = len([item for item in st.session_state.auto_simulation_results if item.get("sport") == "football"])
                basketball_count = len([item for item in st.session_state.auto_simulation_results if item.get("sport") == "basketball"])
                if football_count or basketball_count:
                    toast(f"✅ Simulated {football_count} football matches and {basketball_count} basketball matches")
            st.rerun()

        flush_simulation_toasts()
        time_module.sleep(1)
        st.rerun()

    start_col, stop_col = st.columns(2)
    with start_col:
        start_clicked = st.button("🎲 Auto-Simulate Day", key="auto_simulate_day_run", disabled=st.session_state.simulation_running)
    with stop_col:
        stop_clicked = st.button("⏹ Stop Simulation", key="stop_simulation_button", disabled=not st.session_state.simulation_running)

    if stop_clicked and st.session_state.simulation_running:
        st.session_state.stop_requested.set()
        for thread in list(st.session_state.simulation_threads):
            thread.join(timeout=0.1)
        toast("Stop requested.", icon="⚠️")
        st.rerun()

    if start_clicked and not st.session_state.simulation_running:
        st.session_state.stop_requested = threading.Event()
        st.session_state.simulation_running = True
        st.session_state.simulations_active = {}
        st.session_state.simulation_summary = []
        st.session_state.auto_simulation_results = []

        try:
            football_sport_id, football_sport_name, football_pairs = build_unique_team_pairs_for_sport("football")
            basketball_sport_id, basketball_sport_name, basketball_pairs = build_unique_team_pairs_for_sport("basketball")
        except Exception as exc:
            st.session_state.simulation_running = False
            toast(f"Failed to prepare simulations: {exc}", icon="⚠️")
            return

        threads = []
        for sport_name, pairs in [
            (football_sport_name, football_pairs),
            (basketball_sport_name, basketball_pairs),
        ]:
            for home_team, away_team in pairs:
                thread = threading.Thread(
                    target=run_threaded_match_simulation,
                    args=(sport_name, home_team, away_team, st.session_state.stop_requested),
                    daemon=True,
                )
                thread.start()
                threads.append(thread)

        st.session_state.simulation_threads = threads
        toast(f"Started {len(threads)} live match simulations.", icon="✅")
        # Mirror the start to the terminal so the user sees server-side activity
        try:
            print(f"[SIM] Started {len(threads)} live match simulations.")
        except Exception:
            pass
        st.rerun()


def render_delete_button_for_sport(sport_row):
    if not sport_row["deletable"]:
        st.button(
            "Delete",
            key=f"delete_sport_disabled_{sport_row['id']}",
            disabled=True,
            help="Delete is disabled while teams, players, or matches still exist.",
        )
        return

    with confirmation_container(f"❌ Delete {sport_row['name']}"):
        st.warning("This will permanently remove the sport.")
        if st.button("Confirm delete", key=f"confirm_delete_sport_{sport_row['id']}"):
            with session_scope() as session:
                sport = session.query(Sport).filter_by(id=sport_row["id"]).first()
                if sport:
                    session.delete(sport)
            toast(f"Deleted sport: {sport_row['name']}")
            st.rerun()


def render_team_editor(sport_id, sport_name):
    st.subheader(f"Teams in {sport_name}")
    with st.form("add_team_form"):
        team_name = st.text_input("Team name")
        add_team = st.form_submit_button("Add Team")
        if add_team:
            team_name = clean_text(team_name)
            if not team_name:
                toast("Team name is required.", icon="⚠️")
            else:
                with session_scope() as session:
                    existing = (
                        session.query(Team)
                        .filter(Team.sport_id == sport_id)
                        .filter(Team.name == team_name)
                        .first()
                    )
                    if existing:
                        toast("That team already exists for this sport.", icon="⚠️")
                    else:
                        session.add(Team(name=team_name, sport_id=sport_id))
                        toast(f"Added team: {team_name}")
                        st.rerun()

    teams = load_teams(sport_id)
    if not teams:
        st.info("No teams yet for this sport.")
        return

    team_df = pd.DataFrame(teams).set_index("id")
    team_df = team_df[["name", "players", "matches", "deletable"]]
    team_df.rename(columns={"deletable": "Delete"}, inplace=True)
    edited_df = st.data_editor(
        team_df,
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "name": st.column_config.TextColumn("Team Name"),
            "players": st.column_config.NumberColumn("Players", disabled=True),
            "matches": st.column_config.NumberColumn("Matches", disabled=True),
            "Delete": st.column_config.CheckboxColumn("Delete"),
        },
        disabled=["players", "matches"],
        key=f"team_editor_{sport_id}",
    )

    with confirmation_container("Confirm team deletions"):
        st.warning("Rows marked for deletion will be removed when you apply changes.")
        team_delete_confirmed = st.checkbox(
            "I understand selected teams will be deleted only if they have no players or matches.",
            key=f"team_delete_confirm_{sport_id}",
        )

    if st.button("Apply Team Changes", key=f"apply_team_changes_{sport_id}"):
        if edited_df is None or edited_df.empty:
            toast("No teams to update.", icon="⚠️")
            return

        deleted_count = 0
        updated_count = 0
        blocked_names = []
        with session_scope() as session:
            for team_id, row in edited_df.iterrows():
                team = session.query(Team).filter_by(id=int(team_id)).first()
                if not team:
                    continue
                new_name = clean_text(str(row.get("name", "")))
                mark_delete = bool(row.get("Delete", False))

                if mark_delete:
                    if not team_delete_confirmed:
                        blocked_names.append(team.name)
                        continue
                    player_count = session.query(Player).filter_by(team_id=team.id).count()
                    match_count = (
                        session.query(Match)
                        .filter((Match.home_team_id == team.id) | (Match.away_team_id == team.id))
                        .count()
                    )
                    if player_count or match_count:
                        blocked_names.append(team.name)
                        continue
                    session.delete(team)
                    deleted_count += 1
                    continue

                if new_name and new_name != team.name:
                    duplicate = (
                        session.query(Team)
                        .filter(Team.sport_id == sport_id)
                        .filter(Team.name == new_name)
                        .filter(Team.id != team.id)
                        .first()
                    )
                    if duplicate:
                        blocked_names.append(team.name)
                        continue
                    team.name = new_name
                    updated_count += 1

        if deleted_count:
            toast(f"Deleted {deleted_count} team(s).")
        if updated_count:
            toast(f"Updated {updated_count} team(s).")
        if blocked_names:
            toast(f"Skipped: {', '.join(blocked_names)}", icon="⚠️")
        if deleted_count or updated_count:
            st.rerun()


def render_player_editor(sport_id, sport_name):
    st.subheader(f"Players in {sport_name}")
    teams = load_teams(sport_id)
    if not teams:
        st.info("Add a team first, then add players.")
        return

    team_options = {team["name"]: team["id"] for team in teams}
    team_name = st.selectbox("Filter by Team", ["All teams"] + list(team_options.keys()), key=f"player_team_filter_{sport_id}")
    selected_team_id = None if team_name == "All teams" else team_options[team_name]

    with st.form("add_player_form"):
        columns = st.columns(2)
        with columns[0]:
            player_name = st.text_input("Player name", key=f"add_player_name_{sport_id}")
            add_team_name = st.selectbox("Team", list(team_options.keys()), key=f"add_player_team_{sport_id}")
        with columns[1]:
            position = st.text_input("Position", key=f"add_player_position_{sport_id}")
        add_player = st.form_submit_button("Add Player")
        if add_player:
            player_name = clean_text(player_name)
            position = clean_text(position)
            add_team_id = team_options.get(add_team_name)
            if not player_name:
                toast("Player name is required.", icon="⚠️")
            elif add_team_id is None:
                toast("Select a valid team.", icon="⚠️")
            else:
                try:
                    with session_scope() as session:
                        duplicate = (
                            session.query(Player)
                            .filter(Player.team_id == add_team_id)
                            .filter(Player.name == player_name)
                            .first()
                        )
                        if duplicate:
                            toast("That player already exists on this team.", icon="⚠️")
                        else:
                            session.add(
                                Player(
                                    name=player_name,
                                    position=position,
                                    team_id=add_team_id,
                                    sport_id=sport_id,
                                )
                            )
                            toast(f"Added player: {player_name}")
                            st.rerun()
                except Exception as exc:
                    toast(f"Failed to add player: {exc}", icon="⚠️")

    limit_col, offset_col = st.columns(2)
    with limit_col:
        player_limit = int(st.number_input("Limit", min_value=1, max_value=200, value=10, step=5, key=f"player_limit_{sport_id}"))
    with offset_col:
        player_offset = int(st.number_input("Offset", min_value=0, value=0, step=max(1, player_limit), key=f"player_offset_{sport_id}"))

    players, total = load_players(sport_id=sport_id, team_id=selected_team_id, limit=player_limit, offset=player_offset)
    if not players:
        st.info("No players found for the selected filter.")
        return

    player_df = pd.DataFrame(players).set_index("id")
    player_df = player_df[["name", "position", "team", "events", "deletable"]]
    player_df.rename(columns={"deletable": "Delete"}, inplace=True)
    edited_df = st.data_editor(
        player_df,
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "name": st.column_config.TextColumn("Player Name"),
            "position": st.column_config.TextColumn("Position"),
            "team": st.column_config.TextColumn("Team", disabled=True),
            "events": st.column_config.NumberColumn("Events", disabled=True),
            "Delete": st.column_config.CheckboxColumn("Delete"),
        },
        disabled=["team", "events"],
        key=f"player_editor_{sport_id}_{selected_team_id if selected_team_id else 'all'}",
    )

    with confirmation_container("Confirm player deletions"):
        st.warning("Rows marked for deletion will be removed when you apply changes.")
        player_delete_confirmed = st.checkbox(
            "I understand selected players will be deleted only if they have no events.",
            key=f"player_delete_confirm_{sport_id}_{selected_team_id if selected_team_id else 'all'}",
        )

    if st.button("Apply Player Changes", key=f"apply_player_changes_{sport_id}_{selected_team_id if selected_team_id else 'all'}"):
        deleted_count = 0
        updated_count = 0
        blocked_names = []
        with session_scope() as session:
            for player_id, row in edited_df.iterrows():
                player = session.query(Player).filter_by(id=int(player_id)).first()
                if not player:
                    continue
                new_name = clean_text(str(row.get("name", "")))
                new_position = clean_text(str(row.get("position", "")))
                mark_delete = bool(row.get("Delete", False))

                if mark_delete:
                    if not player_delete_confirmed:
                        blocked_names.append(player.name)
                        continue
                    event_count = session.query(Event).filter_by(player_id=player.id).count()
                    if event_count:
                        blocked_names.append(player.name)
                        continue
                    session.delete(player)
                    deleted_count += 1
                    continue

                changed = False
                if new_name and new_name != player.name:
                    duplicate = (
                        session.query(Player)
                        .filter_by(team_id=player.team_id)
                        .filter(Player.name == new_name)
                        .filter(Player.id != player.id)
                        .first()
                    )
                    if duplicate:
                        blocked_names.append(player.name)
                    else:
                        player.name = new_name
                        changed = True
                if new_position != (player.position or ""):
                    player.position = new_position
                    changed = True
                if changed:
                    updated_count += 1

        if deleted_count:
            toast(f"Deleted {deleted_count} player(s).")
        if updated_count:
            toast(f"Updated {updated_count} player(s).")
        if blocked_names:
            toast(f"Skipped: {', '.join(blocked_names)}", icon="⚠️")
        if deleted_count or updated_count:
            st.rerun()

    total_label = f"Showing {len(players)} of {total} player(s)"
    st.caption(total_label)


def render_sports_section():
    st.subheader("Sports")
    with st.form("add_sport_form"):
        sport_name = st.text_input("Sport name")
        add_sport = st.form_submit_button("Add Sport")
        if add_sport:
            sport_name = clean_text(sport_name)
            if not sport_name:
                toast("Sport name is required.", icon="⚠️")
            else:
                with session_scope() as session:
                    duplicate = session.query(Sport).filter(Sport.name == sport_name).first()
                    if duplicate:
                        toast("That sport already exists.", icon="⚠️")
                    else:
                        session.add(Sport(name=sport_name))
                        toast(f"Added sport: {sport_name}")
                        st.rerun()

    sports = load_sports()
    if not sports:
        st.info("No sports yet.")
        return

    for sport_row in sports:
        left, middle, right = st.columns([3, 1, 1])
        with left:
            st.write(
                f"**{sport_row['name']}**  • Teams: {sport_row['teams']}  • Players: {sport_row['players']}  • Matches: {sport_row['matches']}"
            )
        with middle:
            st.write("")
        with right:
            render_delete_button_for_sport(sport_row)


def render_team_section():
    sports = load_sports()
    if not sports:
        st.info("Add a sport first.")
        return

    sport_map = {sport["name"]: sport["id"] for sport in sports}
    sport_name = st.selectbox("Sport", list(sport_map.keys()), key="team_sport_select")
    render_team_editor(sport_map[sport_name], sport_name)


def render_player_section():
    sports = load_sports()
    if not sports:
        st.info("Add a sport first.")
        return

    sport_map = {sport["name"]: sport["id"] for sport in sports}
    sport_name = st.selectbox("Sport", list(sport_map.keys()), key="player_sport_select")
    render_player_editor(sport_map[sport_name], sport_name)


def render_schedule_matches_section():
    st.subheader("Schedule a Match")
    sports = load_sports()
    if not sports:
        st.info("Add a sport first.")
        return

    sport_map = {sport["name"]: sport["id"] for sport in sports}
    sport_name = st.selectbox("Sport", list(sport_map.keys()), key="match_sport_select")
    sport_id = sport_map[sport_name]
    teams = load_teams(sport_id)
    if len(teams) < 2:
        st.info("Add at least two teams for this sport before scheduling matches.")
    else:
        team_map = {team["name"]: team["id"] for team in teams}
        with st.form("schedule_match_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                home_name = st.selectbox("Home Team", list(team_map.keys()), key="home_team_select")
            with col2:
                away_name = st.selectbox("Away Team", list(team_map.keys()), key="away_team_select")
            with col3:
                match_day = st.date_input("Match Date", value=date.today(), key="match_date_select")
            schedule = st.form_submit_button("Schedule Match")
            if schedule:
                if home_name == away_name:
                    toast("Home and away teams must be different.", icon="⚠️")
                else:
                    session = Session()
                    try:
                        match_exists = (
                            session.query(Match)
                            .filter(Match.sport_id == sport_id)
                            .filter(Match.home_team_id == team_map[home_name])
                            .filter(Match.away_team_id == team_map[away_name])
                            .filter(Match.match_date == datetime.combine(match_day, time.min))
                            .first()
                        )
                        if match_exists:
                            toast("That match already exists for this date.", icon="⚠️")
                        else:
                            session.add(
                                Match(
                                    home_team_id=team_map[home_name],
                                    away_team_id=team_map[away_name],
                                    match_date=datetime.combine(match_day, time.min),
                                    sport_id=sport_id,
                                    status="scheduled",
                                )
                            )
                            session.commit()
                            toast("Match scheduled!")
                            st.rerun()
                    except Exception as exc:
                        session.rollback()
                        toast(f"Failed to schedule match: {exc}", icon="⚠️")
                    finally:
                        session.close()

    st.divider()
    st.subheader("Existing Matches")
    matches = load_matches()
    if not matches:
        st.info("No matches scheduled yet.")
    else:
        header = st.columns([3, 3, 2, 2, 3])
        header[0].markdown("**Home**")
        header[1].markdown("**Away**")
        header[2].markdown("**Status**")
        header[3].markdown("**Date**")
        header[4].markdown("**Actions**")

        for match_row in matches:
            row_cols = st.columns([3, 3, 2, 2, 3])
            row_cols[0].write(match_row["home_name"])
            row_cols[1].write(match_row["away_name"])
            status_key = f"match_status_{match_row['id']}"
            if status_key not in st.session_state:
                st.session_state[status_key] = match_row["status"] if match_row["status"] in STATUS_OPTIONS else STATUS_OPTIONS[0]
            row_cols[2].markdown(format_status_badge(st.session_state[status_key]))
            row_cols[2].selectbox(
                "Status",
                STATUS_OPTIONS,
                index=STATUS_OPTIONS.index(st.session_state[status_key]),
                key=status_key,
                label_visibility="collapsed",
            )
            row_cols[3].write(format_dt(match_row["match_date"]))
            with row_cols[4]:
                action_left, action_right = st.columns(2)
                with action_left:
                    if st.button("Update", key=f"update_match_{match_row['id']}"):
                        with session_scope() as session:
                            match = session.query(Match).filter_by(id=match_row["id"]).first()
                            if match:
                                match.status = st.session_state[status_key]
                        toast("Match status updated.")
                        st.rerun()
                with action_right:
                    with confirmation_container(f"End Match {match_row['id']}"):
                        st.warning("This marks the match as finished.")
                        if st.button("Confirm end", key=f"confirm_end_match_{match_row['id']}"):
                            with session_scope() as session:
                                match = session.query(Match).filter_by(id=match_row["id"]).first()
                                if match:
                                    match.status = "finished"
                            toast("Match ended!", icon="✅")
                            st.rerun()

                    with confirmation_container(f"❌ Delete {match_row['id']}"):
                        st.warning("Deleting a match also deletes its events.")
                        if st.button("Confirm delete", key=f"confirm_delete_match_{match_row['id']}"):
                            with session_scope() as session:
                                session.query(Event).filter_by(match_id=match_row["id"]).delete()
                                match = session.query(Match).filter_by(id=match_row["id"]).first()
                                if match:
                                    session.delete(match)
                            toast("Match deleted.")
                            st.rerun()

        with confirmation_container("⚠️ Delete All Matches"):
            st.warning("This removes every match and all related events.")
            if st.button("Delete All Matches", key="delete_all_matches_existing_button"):
                with session_scope() as session:
                    session.query(Event).delete()
                    session.query(Match).delete()
                st.session_state["auto_simulation_results"] = []
                toast("Deleted all matches and events.")
                st.rerun()

    st.divider()
    show_dev_tools = st.checkbox("Show development tools", key="show_dev_tools")
    if show_dev_tools:
        with confirmation_container("⚠️ Delete All Matches"):
            st.warning("Development only. This removes all matches and all related events.")
            confirm_all = st.checkbox("I understand this will delete all matches.", key="confirm_delete_all_matches")
            if st.button("Delete All Matches", key="delete_all_matches_button"):
                if not confirm_all:
                    toast("Confirm the deletion checkbox first.", icon="⚠️")
                else:
                    with session_scope() as session:
                        session.query(Event).delete()
                        session.query(Match).delete()
                    toast("Deleted all matches and events.")
                    st.rerun()


def render_event_type_buttons(sport_name: str, current_type: str, state_key: str):
    event_types = get_supported_event_types(sport_name)
    columns = st.columns(len(event_types))
    for column, (label, code) in zip(columns, event_types):
        with column:
            button_label = f"✓ {label}" if current_type == code else label
            if st.button(
                button_label,
                key=f"{state_key}_{code}",
                use_container_width=True,
                type="primary" if current_type == code else "secondary",
            ):
                st.session_state[state_key] = code
                st.rerun()
    if current_type not in {code for _, code in event_types}:
        current_type = event_types[0][1]
        st.session_state[state_key] = current_type
    return current_type


def render_live_event_entry_section():
    st.subheader("Live Event Entry")
    matches = load_matches(statuses=["scheduled", "live"])
    if not matches:
        st.info("No matches scheduled yet.")
        return

    match_labels = [build_match_label(match_row) for match_row in matches]
    match_lookup = {build_match_label(match_row): match_row for match_row in matches}
    selected_label = st.selectbox("Match", match_labels, key="live_match_selector")
    match_row = match_lookup[selected_label]
    match_info = load_match_by_id(match_row["id"])
    if not match_info:
        st.info("Selected match is no longer available.")
        return

    sport_name = match_info["sport_name"]
    match_type_key = f"live_event_type_{match_info['id']}"
    default_event_type = get_supported_event_types(sport_name)[0][1]
    if match_type_key not in st.session_state:
        st.session_state[match_type_key] = default_event_type

    top_cols = st.columns(4)
    top_cols[0].metric("Match ID", match_info["id"])
    top_cols[1].metric("Status", match_info["status"])
    top_cols[2].metric("Date", format_match_datetime(match_info["match_date"]))
    top_cols[3].metric("Sport", sport_name.title())
    st.write(f"**{match_info['home_name']}** vs **{match_info['away_name']}**")

    events = load_match_events(match_info["id"])
    st.caption("Existing events")
    if events:
        event_table = pd.DataFrame(
            [
                {
                    "minute": event["minute"],
                    "event": event["event_label"],
                    "player": event["player_name"],
                    "extra": event["extra_text"],
                }
                for event in events
            ]
        )
        st.dataframe(event_table, use_container_width=True, hide_index=True)
        for event in events:
            row_cols = st.columns([1, 2, 2, 3, 1])
            row_cols[0].write(f"{event['minute']}'")
            row_cols[1].write(event["event_label"])
            row_cols[2].write(event["player_name"])
            row_cols[3].write(event["extra_text"] or "")
            with row_cols[4]:
                with confirmation_container(f"❌ Delete event {event['id']}"):
                    st.warning("This will remove the event immediately.")
                    if st.button("Confirm delete", key=f"delete_event_{event['id']}"):
                        with session_scope() as session:
                            target = session.query(Event).filter_by(id=event["id"]).first()
                            if target:
                                session.delete(target)
                        toast("Event deleted.")
                        st.rerun()
    else:
        st.info("No events yet.")

    st.divider()
    st.subheader("Add Event")
    current_event_type = st.session_state[match_type_key]
    current_event_type = render_event_type_buttons(sport_name, current_event_type, match_type_key)

    team_choice = st.selectbox(
        "Team",
        [f"Home Team ({match_info['home_name']})", f"Away Team ({match_info['away_name']})"],
        key=f"live_team_choice_{match_info['id']}",
    )
    selected_team_id = match_info["home_id"] if team_choice.startswith("Home Team") else match_info["away_id"]

    with session_scope() as session:
        players = (
            session.query(Player)
            .filter(Player.team_id == selected_team_id)
            .order_by(Player.name.asc())
            .all()
        )
        player_options = [
            {
                "id": player.id,
                "label": player.name,
            }
            for player in players
        ]

    if not player_options:
        st.info("Add players to the two teams before logging events.")
        return

    player_map = {player["label"]: player["id"] for player in player_options}
    player_label = st.selectbox("Player", list(player_map.keys()), key=f"live_player_{match_info['id']}")
    minute = int(st.number_input("Minute / Quarter", min_value=0, max_value=999, value=0, step=1, key=f"live_minute_{match_info['id']}"))

    extra_payload = {}
    if current_event_type == "goal":
        assist_choices = ["None"] + list(player_map.keys())
        assist_label = st.selectbox("Assist player", assist_choices, key=f"live_goal_assist_{match_info['id']}")
        if assist_label != "None":
            extra_payload["assist_player_id"] = player_map[assist_label]
    elif current_event_type in {"point_2", "point_3", "free_throw"}:
        points_map = {"point_2": 2, "point_3": 3, "free_throw": 1}
        extra_payload["points"] = points_map[current_event_type]
        assist_choices = ["None"] + list(player_map.keys())
        assist_label = st.selectbox("Assist player", assist_choices, key=f"live_basket_assist_{match_info['id']}")
        if assist_label != "None":
            extra_payload["assist_player_id"] = player_map[assist_label]

    if st.button("Add Event", key=f"add_event_{match_info['id']}"):
        with session_scope() as session:
            session.add(
                Event(
                    match_id=match_info["id"],
                    event_type=current_event_type,
                    player_id=player_map[player_label],
                    minute=minute,
                    extra=json.dumps(extra_payload) if extra_payload else None,
                )
            )
        toast("Event added", icon="✅")
        st.rerun()


def render_view_events_section():
    st.subheader("View Events")
    limit_col, offset_col = st.columns(2)
    with limit_col:
        match_limit = int(st.number_input("Limit", min_value=1, max_value=100, value=5, step=1, key="view_events_limit"))
    with offset_col:
        match_offset = int(st.number_input("Offset", min_value=0, value=0, step=max(1, match_limit), key="view_events_offset"))

    groups, total = load_all_match_groups(limit=match_limit, offset=match_offset)
    st.caption(f"Showing {len(groups)} of {total} match group(s)")
    if not groups:
        st.info("No events yet.")
        return

    for match_group in groups:
        with st.expander(
            f"{format_status_badge(match_group['status'])} • 🏆 {match_group['home_name']} {match_group['home_score']} - {match_group['away_score']} {match_group['away_name']} ({format_match_datetime(match_group['match_date'])})",
            expanded=False,
        ):
            if not match_group["events"]:
                st.write("No events yet")
            else:
                with session_scope() as session:
                    match_row = session.query(Match).filter_by(id=match_group["id"]).first()
                    events = (
                        session.query(Event)
                        .filter_by(match_id=match_group["id"])
                        .order_by(Event.minute.asc(), Event.created_at.asc(), Event.id.asc())
                        .all()
                    )

                    event_rows = []
                    for event in events:
                        extra_data = {}
                        if event.extra:
                            try:
                                extra_data = json.loads(event.extra)
                            except json.JSONDecodeError:
                                extra_data = {}

                        extra_parts = []
                        if "assist_player_id" in extra_data:
                            extra_parts.append(f"Assist: {get_player_name(session, extra_data['assist_player_id'])}")
                        if "points" in extra_data:
                            extra_parts.append(f"Points: {extra_data['points']}")

                        player = session.query(Player).filter_by(id=event.player_id).first()
                        player_name = player.name if player else "Unknown"
                        team_name = player.team.name if player and player.team else "Unknown"

                        home_away = "-"
                        if match_row and player:
                            if player.team_id == match_row.home_team_id:
                                home_away = "H"
                            elif player.team_id == match_row.away_team_id:
                                home_away = "A"

                        event_rows.append(
                            {
                                "minute": event.minute,
                                "event_type": humanize_event_type(event.event_type),
                                "player": player_name,
                                "team": team_name,
                                "h_a": home_away,
                                "extra": " | ".join(extra_parts) if extra_parts else "",
                            }
                        )

                event_df = pd.DataFrame(event_rows)
                event_df = event_df[["minute", "event_type", "player", "team", "h_a", "extra"]]
                st.dataframe(event_df, use_container_width=True, hide_index=True)

            with confirmation_container(f"❌ Delete all events for match {match_group['id']}"):
                st.warning("This removes every event linked to the match.")
                if st.button("Confirm delete all events", key=f"delete_all_events_{match_group['id']}"):
                    with session_scope() as session:
                        session.query(Event).filter_by(match_id=match_group["id"]).delete()
                    toast("Deleted all events for the match.")
                    st.rerun()


manage_tab, match_tab, view_tab = st.tabs(["⚙️ Manage Data", "🏟️ Matches & Events", "📋 View Events"])

with manage_tab:
    sports_tab, teams_tab, players_tab = st.tabs(["Sports", "Teams", "Players"])
    with sports_tab:
        render_sports_section()
    with teams_tab:
        render_team_section()
    with players_tab:
        render_player_section()

with match_tab:
    render_auto_simulate_day_section()
    st.divider()
    schedule_tab, live_tab = st.tabs(["Schedule Matches", "Live Event Entry"])
    with schedule_tab:
        render_schedule_matches_section()
    with live_tab:
        render_live_event_entry_section()

with view_tab:
    render_view_events_section()