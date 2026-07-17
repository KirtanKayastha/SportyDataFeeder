# /home/sam069/projects/SportyDataFeeder/tests/test_simulation_extras.py
#
# Injuries, penalties, shootouts and possession in the live simulation:
#   - injuries are pre-drawn per match (not per-minute rolls): a forced-off
#     injury consumes a substitution (reason=injury) while subs remain, and a
#     minor knock changes nothing;
#   - a penalty resolves to exactly one outcome: a goal (extra.penalty=true)
#     or a miss (a save additionally credits the opposing keeper);
#   - knockout matches tied after 90'+ET run a shootout whose tally never
#     touches the regulation score;
#   - possession is tracked per minute and exposed as possession_pct plus a
#     persisted team-level "possession" event.

import json

import pytest

import app.services.simulation as simulation_service
from app.database import Event, Session


@pytest.fixture
def football_world(client):
    """Football teams with a full XI plus 3 bench players each."""
    sport_id = client.post("/sports", json={"name": "football"}).json()["id"]
    home = client.post("/teams", json={"name": "Inj FC", "sport_id": sport_id}).json()["id"]
    away = client.post("/teams", json={"name": "Pen FC", "sport_id": sport_id}).json()["id"]
    players = {home: [], away: []}
    for team_id, prefix in ((home, "H"), (away, "A")):
        for index in range(14):
            position = "G" if index == 0 else "M"
            player = client.post(
                "/players",
                json={"name": f"{prefix} P{index}", "team_id": team_id,
                      "position": position, "sport_id": sport_id},
            ).json()
            players[team_id].append(player["id"])
    return {"sport_id": sport_id, "home": home, "away": away, "players": players}


def make_match(client, world, knockout=False) -> int:
    return client.post(
        "/matches",
        json={"home_team_id": world["home"], "away_team_id": world["away"],
              "match_date": "2026-06-10T18:00:00", "sport_id": world["sport_id"],
              "knockout": knockout},
    ).json()["id"]


def all_events(match_id: int) -> list[Event]:
    db = Session()
    try:
        return db.query(Event).filter_by(match_id=match_id).order_by(Event.minute.asc(), Event.id.asc()).all()
    finally:
        db.close()


def quiet_rates(world) -> dict:
    """Near-zero rates: no organic goals, so outcomes are driven by the
    mechanism under test."""
    return {pid: {"goal": 0.0} for pids in world["players"].values() for pid in pids}


def test_injury_severities_and_forced_sub(client, football_world, wait_for_simulation, monkeypatch):
    """A forced-off injury at 10' produces injury + substitution(reason=injury)
    and the injured player never acts again; a minor knock at 20' produces only
    the injury event and the player plays on."""
    match_id = make_match(client, football_world)
    # Everyone scores every minute -> activity traces who is on the pitch.
    client.app.state.event_rates = {
        pid: {"goal": 1.0} for pids in football_world["players"].values() for pid in pids
    }
    monkeypatch.setattr(simulation_service, "_draw_sub_minutes", lambda n, total: [])
    monkeypatch.setattr(simulation_service, "_draw_penalty_minutes", lambda total: [])
    monkeypatch.setattr(
        simulation_service, "_draw_injuries",
        lambda total: [(10, "forced_off"), (20, "minor")],
    )
    monkeypatch.setattr(simulation_service, "TOTAL_MINUTES",
                        {**simulation_service.TOTAL_MINUTES, simulation_service.SportType.FOOTBALL: 30})

    client.post("/simulate", json={"match_id": match_id})
    final = wait_for_simulation(client, match_id)
    assert final["status"] == "finished"

    events = all_events(match_id)
    injuries = [e for e in events if e.event_type == "injury"]
    assert [(e.minute, json.loads(e.extra)["severity"]) for e in injuries] == \
        [(10, "forced_off"), (20, "minor")]

    subs = [e for e in events if e.event_type == "substitution"]
    assert len(subs) == 1
    sub = subs[0]
    sub_extra = json.loads(sub.extra)
    assert sub.minute == 10
    assert sub_extra["reason"] == "injury"
    assert sub_extra["player_out"] == injuries[0].player_id
    # Forced-off player is gone; the replacement plays from minute 10 on.
    assert not [e for e in events if e.minute > 10 and e.player_id == injuries[0].player_id]
    assert [e for e in events if e.minute >= 10 and e.player_id == sub.player_id and e.event_type == "goal"]
    # Minor knock: the player keeps playing (scores after minute 20).
    assert [e for e in events if e.minute > 20 and e.player_id == injuries[1].player_id]


def test_forced_off_without_subs_means_man_down(client, football_world, wait_for_simulation, monkeypatch):
    """With no subs left, a forced-off injury removes the player with no
    replacement — no substitution event is emitted."""
    match_id = make_match(client, football_world)
    client.app.state.event_rates = {
        pid: {"goal": 1.0} for pids in football_world["players"].values() for pid in pids
    }
    monkeypatch.setattr(simulation_service, "_draw_sub_minutes", lambda n, total: [])
    monkeypatch.setattr(simulation_service, "_draw_penalty_minutes", lambda total: [])
    monkeypatch.setattr(simulation_service, "FOOTBALL_MAX_SUBS", 0)
    monkeypatch.setattr(simulation_service, "_draw_injuries", lambda total: [(5, "forced_off")])
    monkeypatch.setattr(simulation_service, "TOTAL_MINUTES",
                        {**simulation_service.TOTAL_MINUTES, simulation_service.SportType.FOOTBALL: 10})

    client.post("/simulate", json={"match_id": match_id})
    final = wait_for_simulation(client, match_id)
    assert final["status"] == "finished"

    events = all_events(match_id)
    assert not [e for e in events if e.event_type == "substitution"]
    injured = [e for e in events if e.event_type == "injury"][0]
    assert not [e for e in events if e.minute > 5 and e.player_id == injured.player_id]


def test_knocked_player_comes_off_at_next_sub_window(client, wait_for_simulation, monkeypatch):
    """A minor knock doesn't sub the player immediately, but the next tactical
    window takes THEM off rather than a random teammate. All-outfielder roster
    so the knock can't land on a keeper (keepers are never tactically subbed)."""
    sport_id = client.post("/sports", json={"name": "football"}).json()["id"]
    home = client.post("/teams", json={"name": "Knock FC", "sport_id": sport_id}).json()["id"]
    away = client.post("/teams", json={"name": "Fresh FC", "sport_id": sport_id}).json()["id"]
    players = {home: [], away: []}
    for team_id, prefix in ((home, "H"), (away, "A")):
        for index in range(14):
            player = client.post(
                "/players",
                json={"name": f"{prefix} P{index}", "team_id": team_id,
                      "position": "M", "sport_id": sport_id},
            ).json()
            players[team_id].append(player["id"])
    match_id = client.post(
        "/matches",
        json={"home_team_id": home, "away_team_id": away,
              "match_date": "2026-06-10T18:00:00", "sport_id": sport_id},
    ).json()["id"]
    client.app.state.event_rates = {
        pid: {"goal": 0.0} for pids in players.values() for pid in pids
    }
    monkeypatch.setattr(simulation_service, "_draw_injuries", lambda total: [(10, "minor")])
    monkeypatch.setattr(simulation_service, "_draw_penalty_minutes", lambda total: [])
    monkeypatch.setattr(simulation_service, "_draw_sub_minutes", lambda n, total: [20][:n])
    monkeypatch.setattr(simulation_service, "TOTAL_MINUTES",
                        {**simulation_service.TOTAL_MINUTES, simulation_service.SportType.FOOTBALL: 30})

    client.post("/simulate", json={"match_id": match_id})
    final = wait_for_simulation(client, match_id)
    assert final["status"] == "finished"

    events = all_events(match_id)
    knock = [e for e in events if e.event_type == "injury"][0]
    assert json.loads(knock.extra)["severity"] == "minor"
    # No sub at the knock itself; at the 20' window, the knocked player's team
    # takes exactly the knocked player off.
    subs = [e for e in events if e.event_type == "substitution"]
    assert all(e.minute == 20 for e in subs)
    knocked_team_subs = [
        e for e in subs if json.loads(e.extra)["player_out"] == knock.player_id
    ]
    assert len(knocked_team_subs) == 1


def test_penalty_resolves_to_goal_or_miss(client, football_world, wait_for_simulation, monkeypatch):
    """One scheduled penalty yields exactly one attempt: a goal flagged
    extra.penalty, or a penalty_missed (a save adds penalty_saved for the
    opposing keeper). No organic goals, so the score reflects the outcome."""
    match_id = make_match(client, football_world)
    client.app.state.event_rates = quiet_rates(football_world)
    monkeypatch.setattr(simulation_service, "_draw_sub_minutes", lambda n, total: [])
    monkeypatch.setattr(simulation_service, "_draw_injuries", lambda total: [])
    monkeypatch.setattr(simulation_service, "_draw_penalty_minutes", lambda total: [30])
    monkeypatch.setattr(simulation_service, "TOTAL_MINUTES",
                        {**simulation_service.TOTAL_MINUTES, simulation_service.SportType.FOOTBALL: 40})

    client.post("/simulate", json={"match_id": match_id})
    final = wait_for_simulation(client, match_id)
    assert final["status"] == "finished"

    events = all_events(match_id)
    goals = [e for e in events if e.event_type == "goal"]
    misses = [e for e in events if e.event_type == "penalty_missed"]
    saves = [e for e in events if e.event_type == "penalty_saved"]
    assert len(goals) + len(misses) == 1  # exactly one attempt, one outcome
    if goals:
        assert json.loads(goals[0].extra)["penalty"] is True
        assert final["home_score"] + final["away_score"] == 1
        assert not saves
    else:
        assert final["home_score"] + final["away_score"] == 0
        if saves:  # a save is credited to the OTHER team's keeper
            db = Session()
            try:
                from app.database import Player
                taker = db.query(Player).filter_by(id=misses[0].player_id).first()
                keeper = db.query(Player).filter_by(id=saves[0].player_id).first()
                assert keeper.team_id != taker.team_id
                assert keeper.position == "G"
            finally:
                db.close()


def test_knockout_tie_goes_to_extra_time_and_shootout(client, football_world, wait_for_simulation, monkeypatch):
    """A knockout match with no goals stays level, plays extra time, then runs
    a shootout: kicks are recorded as shootout events, the regulation score
    stays tied, and the status endpoint reports the tally + winner."""
    match_id = make_match(client, football_world, knockout=True)
    client.app.state.event_rates = quiet_rates(football_world)
    monkeypatch.setattr(simulation_service, "_draw_sub_minutes", lambda n, total: [])
    monkeypatch.setattr(simulation_service, "_draw_injuries", lambda total: [])
    monkeypatch.setattr(simulation_service, "_draw_penalty_minutes", lambda total: [])
    monkeypatch.setattr(simulation_service, "TOTAL_MINUTES",
                        {**simulation_service.TOTAL_MINUTES, simulation_service.SportType.FOOTBALL: 6})
    monkeypatch.setattr(simulation_service, "EXTRA_TIME_MINUTES", 2)

    client.post("/simulate", json={"match_id": match_id})
    final = wait_for_simulation(client, match_id)
    assert final["status"] == "finished"

    assert final["home_score"] == final["away_score"] == 0
    assert final["current_minute"] == 8  # 6 regulation + 2 extra time
    assert final["shootout_home"] is not None and final["shootout_away"] is not None
    assert final["shootout_home"] != final["shootout_away"]
    winner = final["shootout_winner_team_id"]
    expected = football_world["home"] if final["shootout_home"] > final["shootout_away"] else football_world["away"]
    assert winner == expected

    kicks = [e for e in all_events(match_id) if e.event_type in ("shootout_goal", "shootout_miss")]
    assert len(kicks) >= 10  # at least 5 rounds per side
    assert all(e.minute == 9 for e in kicks)


def test_league_tie_does_not_shootout(client, football_world, wait_for_simulation, monkeypatch):
    """Non-knockout football matches may end level — no ET, no shootout."""
    match_id = make_match(client, football_world, knockout=False)
    client.app.state.event_rates = quiet_rates(football_world)
    monkeypatch.setattr(simulation_service, "_draw_sub_minutes", lambda n, total: [])
    monkeypatch.setattr(simulation_service, "_draw_injuries", lambda total: [])
    monkeypatch.setattr(simulation_service, "_draw_penalty_minutes", lambda total: [])
    monkeypatch.setattr(simulation_service, "TOTAL_MINUTES",
                        {**simulation_service.TOTAL_MINUTES, simulation_service.SportType.FOOTBALL: 6})

    client.post("/simulate", json={"match_id": match_id})
    final = wait_for_simulation(client, match_id)
    assert final["status"] == "finished"
    assert final["current_minute"] == 6
    assert final["shootout_home"] is None
    assert not [e for e in all_events(match_id) if e.event_type.startswith("shootout")]


def test_possession_tracked_and_persisted(client, football_world, wait_for_simulation, monkeypatch):
    """Possession is exposed on the status endpoint (sums to 100) and persisted
    as a team-level event with the final split."""
    match_id = make_match(client, football_world)
    client.app.state.event_rates = quiet_rates(football_world)
    monkeypatch.setattr(simulation_service, "_draw_sub_minutes", lambda n, total: [])
    monkeypatch.setattr(simulation_service, "_draw_injuries", lambda total: [])
    monkeypatch.setattr(simulation_service, "_draw_penalty_minutes", lambda total: [])
    monkeypatch.setattr(simulation_service, "TOTAL_MINUTES",
                        {**simulation_service.TOTAL_MINUTES, simulation_service.SportType.FOOTBALL: 20})

    client.post("/simulate", json={"match_id": match_id})
    final = wait_for_simulation(client, match_id)
    assert final["status"] == "finished"

    home_pct, away_pct = final["possession_home_pct"], final["possession_away_pct"]
    assert home_pct is not None and away_pct is not None
    assert round(home_pct + away_pct, 1) == 100.0
    assert 25.0 <= home_pct <= 75.0  # per-minute share is clamped to this band

    possession_events = [e for e in all_events(match_id) if e.event_type == "possession"]
    assert len(possession_events) == 1
    extra = json.loads(possession_events[0].extra)
    assert extra == {"home_pct": home_pct, "away_pct": away_pct}
    assert possession_events[0].player_id is None
