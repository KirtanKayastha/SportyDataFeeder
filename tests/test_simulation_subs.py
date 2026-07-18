# /home/sam069/projects/SportyDataFeeder/tests/test_simulation_subs.py
#
# Substitutions and discipline in the live simulation:
#   - football: a second yellow to the same player becomes a red card and the
#     player is sent off (no replacement, team plays short);
#   - football: up to 5 permanent substitutions from the bench, weighted to
#     the 2nd half; keepers and featured players are never subbed off;
#   - basketball: rotation checkpoints swap players both ways (players return).

import pytest

import app.services.simulation as simulation_service
from app.database import Event, Session


@pytest.fixture
def football_world(client):
    """Football teams with a full XI plus 3 bench players each."""
    sport_id = client.post("/sports", json={"name": "football"}).json()["id"]
    home = client.post("/teams", json={"name": "Sub FC", "sport_id": sport_id}).json()["id"]
    away = client.post("/teams", json={"name": "Card FC", "sport_id": sport_id}).json()["id"]
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


def make_match(client, world) -> int:
    return client.post(
        "/matches",
        json={"home_team_id": world["home"], "away_team_id": world["away"],
              "match_date": "2026-06-10T18:00:00", "sport_id": world["sport_id"]},
    ).json()["id"]


def all_events(match_id: int) -> list[Event]:
    db = Session()
    try:
        return db.query(Event).filter_by(match_id=match_id).order_by(Event.minute.asc(), Event.id.asc()).all()
    finally:
        db.close()


def test_second_yellow_becomes_red_and_sends_off(client, football_world, wait_for_simulation):
    """Every player yellow-carded every minute: minute 1 books everyone, minute
    2 produces second yellows -> reds, and nobody plays a third minute."""
    match_id = make_match(client, football_world)
    client.app.state.event_rates = {
        pid: {"yellow_card": 1.0}
        for pids in football_world["players"].values() for pid in pids
    }

    client.post("/simulate", json={"match_id": match_id})
    final = wait_for_simulation(client, match_id)
    assert final["status"] == "finished"

    events = all_events(match_id)
    reds = [e for e in events if e.event_type == "red_card"]
    assert reds, "second yellows must produce red cards"
    assert all(e.minute == 2 for e in reds)
    assert all('"second_yellow"' in (e.extra or "") for e in reds)
    # One red per starter, and every starter has exactly 2 yellows.
    yellows_by_player: dict[int, int] = {}
    for e in events:
        if e.event_type == "yellow_card":
            yellows_by_player[e.player_id] = yellows_by_player.get(e.player_id, 0) + 1
    assert set(yellows_by_player.values()) == {2}
    assert {e.player_id for e in reds} == set(yellows_by_player)
    # Sent-off players generate nothing after minute 2 (subs excluded: benches
    # may legally come on later and get booked themselves).
    starters = set(yellows_by_player)
    assert not [e for e in events if e.minute > 2 and e.player_id in starters]


def test_football_substitutions_swap_active_players(client, football_world, wait_for_simulation, monkeypatch):
    """Subs at fixed minutes: the player coming off stops producing events, the
    replacement starts, and no team exceeds its planned windows."""
    match_id = make_match(client, football_world)
    # Everyone scores every minute -> activity is a perfect on-pitch tracer.
    client.app.state.event_rates = {
        pid: {"goal": 1.0}
        for pids in football_world["players"].values() for pid in pids
    }
    monkeypatch.setattr(simulation_service, "_draw_sub_minutes", lambda n, total: [46, 60, 75][:n])
    monkeypatch.setattr(simulation_service, "TOTAL_MINUTES",
                        {**simulation_service.TOTAL_MINUTES, simulation_service.SportType.FOOTBALL: 80})
    # Random injuries/penalties would add substitutions and events at other
    # minutes; disable them so the planned-window assertions stay exact.
    monkeypatch.setattr(simulation_service, "_draw_injuries", lambda total: [])
    monkeypatch.setattr(simulation_service, "_draw_penalty_minutes", lambda total: [])

    client.post("/simulate", json={"match_id": match_id})
    final = wait_for_simulation(client, match_id)
    assert final["status"] == "finished"

    events = all_events(match_id)
    subs = [e for e in events if e.event_type == "substitution"]
    assert len(subs) == 6  # 3 planned windows per team, bench of 3 each
    assert sorted({e.minute for e in subs}) == [46, 60, 75]
    for sub in subs:
        import json as _json
        player_out = _json.loads(sub.extra)["player_out"]
        # Permanent: the replaced player never acts again...
        assert not [e for e in events if e.minute > sub.minute and e.player_id == player_out]
        # ...and the substitute scores from their entry minute on.
        assert [e for e in events
                if e.minute >= sub.minute and e.player_id == sub.player_id and e.event_type == "goal"]


def test_pick_replacement_position_rules():
    """Keepers only replace keepers; outfielders only outfielders — except a
    keeper going off with no bench keeper still gets a body on the pitch."""
    from types import SimpleNamespace

    gk = SimpleNamespace(position="G")
    mid = SimpleNamespace(position="M")
    assert simulation_service._pick_replacement([gk], off_is_keeper=False) is None
    assert simulation_service._pick_replacement([gk, mid], off_is_keeper=False) is mid
    assert simulation_service._pick_replacement([gk, mid], off_is_keeper=True) is gk
    assert simulation_service._pick_replacement([mid], off_is_keeper=True) is mid


def test_bench_keeper_never_subbed_on_for_outfielder(client, football_world, wait_for_simulation, monkeypatch):
    """With a spare keeper on every bench, tactical windows only ever bring on
    outfielders; the keeper-only window goes unused (2 subs/team, not 3)."""
    # Re-position the last rostered player of each team (bench, highest id)
    # as a second keeper.
    db = Session()
    try:
        from app.database import Player
        bench_keepers = set()
        for pids in football_world["players"].values():
            keeper = db.query(Player).filter_by(id=pids[-1]).first()
            keeper.position = "G"
            bench_keepers.add(keeper.id)
        db.commit()
    finally:
        db.close()

    match_id = make_match(client, football_world)
    client.app.state.event_rates = {
        pid: {"goal": 1.0}
        for pids in football_world["players"].values() for pid in pids
    }
    monkeypatch.setattr(simulation_service, "_draw_sub_minutes", lambda n, total: [46, 60, 75][:n])
    monkeypatch.setattr(simulation_service, "TOTAL_MINUTES",
                        {**simulation_service.TOTAL_MINUTES, simulation_service.SportType.FOOTBALL: 80})
    monkeypatch.setattr(simulation_service, "_draw_injuries", lambda total: [])
    monkeypatch.setattr(simulation_service, "_draw_penalty_minutes", lambda total: [])

    client.post("/simulate", json={"match_id": match_id})
    final = wait_for_simulation(client, match_id)
    assert final["status"] == "finished"

    subs = [e for e in all_events(match_id) if e.event_type == "substitution"]
    assert not [s for s in subs if s.player_id in bench_keepers], \
        "a spare keeper must never come on for an outfielder"
    # 3 windows but only 2 outfielders per bench: the last window is skipped.
    assert len(subs) == 4


def test_basketball_rotation_gives_bench_court_time(client, wait_for_simulation):
    """Rotation checkpoints cycle the 5+3 pool: bench players get on court (and
    score), substitution events exist in both directions."""
    sport_id = client.post("/sports", json={"name": "basketball"}).json()["id"]
    home = client.post("/teams", json={"name": "RTH", "sport_id": sport_id}).json()["id"]
    away = client.post("/teams", json={"name": "RTA", "sport_id": sport_id}).json()["id"]
    players = {home: [], away: []}
    for team_id, prefix in ((home, "H"), (away, "A")):
        for index in range(8):
            player = client.post(
                "/players",
                json={"name": f"{prefix} {index}", "team_id": team_id,
                      "position": "G", "sport_id": sport_id},
            ).json()
            players[team_id].append(player["id"])
    match_id = client.post(
        "/matches",
        json={"home_team_id": home, "away_team_id": away,
              "match_date": "2026-06-10T18:00:00", "sport_id": sport_id},
    ).json()["id"]
    client.app.state.event_rates = {
        pid: {"point_2": 1.0} for pids in players.values() for pid in pids
    }

    client.post("/simulate", json={"match_id": match_id})
    final = wait_for_simulation(client, match_id)
    assert final["status"] == "finished"

    events = all_events(match_id)
    subs = [e for e in events if e.event_type == "substitution"]
    assert len(subs) >= 10  # checkpoints every 4 minutes, 1-2 swaps per team
    # Every rostered player saw the court (8-man rotation): all of them score.
    scorers = {e.player_id for e in events if e.event_type == "point_2"}
    assert scorers == {pid for pids in players.values() for pid in pids}
    # Players RETURN in basketball: someone comes on more than once.
    on_counts: dict[int, int] = {}
    for sub in subs:
        on_counts[sub.player_id] = on_counts.get(sub.player_id, 0) + 1
    assert max(on_counts.values()) >= 2