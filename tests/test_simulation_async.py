# /home/sam069/projects/SportyDataFeeder/tests/test_simulation_async.py

import json
import logging
import time
from types import SimpleNamespace

import httpx
import pytest

import app.routers.predict as predict_router
import app.routers.simulation as simulation_router
import app.services.simulation as simulation_service
from app.services.backend_client import BackendClient

MATCH_UUID = "aaaaaaaa-0000-0000-0000-000000000001"
HOME_TEAM_UUID = "bbbbbbbb-0000-0000-0000-000000000001"
AWAY_TEAM_UUID = "bbbbbbbb-0000-0000-0000-000000000002"


@pytest.fixture
def football_world(client):
    sport_id = client.post("/sports", json={"name": "football"}).json()["id"]
    home = client.post("/teams", json={"name": "Strong FC", "sport_id": sport_id}).json()["id"]
    away = client.post("/teams", json={"name": "Weak FC", "sport_id": sport_id}).json()["id"]
    players = {home: [], away: []}
    for team_id, prefix in ((home, "H"), (away, "A")):
        for index in range(3):
            player = client.post(
                "/players",
                json={"name": f"{prefix} Player {index}", "team_id": team_id, "position": "M", "sport_id": sport_id},
            ).json()
            players[team_id].append(player["id"])
    return {"sport_id": sport_id, "home": home, "away": away, "players": players}


def make_match(client, world) -> int:
    return client.post(
        "/matches",
        json={
            "home_team_id": world["home"],
            "away_team_id": world["away"],
            "match_date": "2026-06-10T18:00:00",
            "sport_id": world["sport_id"],
        },
    ).json()["id"]


def inject_high_goal_rates(client, world, rate=0.5):
    """Guarantee events fire every match without relying on fallback luck."""
    rates = {pid: {"goal": rate} for pids in world["players"].values() for pid in pids}
    client.app.state.event_rates = rates
    return rates


@pytest.fixture
def basketball_world(client):
    sport_id = client.post("/sports", json={"name": "basketball"}).json()["id"]
    home = client.post("/teams", json={"name": "BKH", "sport_id": sport_id}).json()["id"]
    away = client.post("/teams", json={"name": "BKA", "sport_id": sport_id}).json()["id"]
    players = {home: [], away: []}
    for team_id, prefix in ((home, "H"), (away, "A")):
        for index in range(5):
            player = client.post(
                "/players",
                json={"name": f"{prefix} {index}", "team_id": team_id, "position": "G", "sport_id": sport_id},
            ).json()
            players[team_id].append(player["id"])
    return {"sport_id": sport_id, "home": home, "away": away, "players": players}


@pytest.fixture
def mock_backend(monkeypatch):
    """Capture pushes in-memory; swap `handler` to simulate outages."""
    captured = {"match_result": [], "prediction": [], "player_ratings": []}
    behaviour = {"fail": False}

    def handler(request: httpx.Request) -> httpx.Response:
        if behaviour["fail"]:
            raise httpx.ConnectError("backend down")
        kind = request.url.path.rsplit("/", 1)[-1].replace("-", "_")
        captured[kind].append(json.loads(request.content))
        return httpx.Response(200)

    def factory():
        return BackendClient(
            base_url="http://sporty.test",
            secret="test-secret",
            backoff_base=0.0,
            transport=httpx.MockTransport(handler),
        )

    monkeypatch.setattr(simulation_router, "get_backend_client", factory)
    monkeypatch.setattr(predict_router, "get_backend_client", factory)
    return SimpleNamespace(captured=captured, behaviour=behaviour)


class TestBackgroundSimulation:
    def test_basketball_tie_goes_to_overtime_no_draw(
        self, client, basketball_world, wait_for_simulation, monkeypatch
    ):
        """A regulation tie triggers overtime until a winner emerges (no draws)."""
        import uuid

        def fake_sample(setup, minute):
            home_tid = setup["match"].home_team_id
            away_tid = setup["match"].away_team_id
            home_p = next(p for p in setup["lineups"] if p.team_id == home_tid)
            away_p = next(p for p in setup["lineups"] if p.team_id == away_tid)

            def ev(player, tid):
                return {
                    "event_id": str(uuid.uuid4()), "event_type": "point_2",
                    "player_id": player.id, "team_id": tid, "minute": minute,
                    "extra": {"points": 2},
                }

            if minute == 1:  # tied 2-2 through regulation
                return [ev(home_p, home_tid), ev(away_p, away_tid)]
            if minute == 49:  # first overtime minute: home breaks the tie
                return [ev(home_p, home_tid)]
            return []

        monkeypatch.setattr(simulation_service, "_sample_minute_events", fake_sample)

        match_id = make_match(client, basketball_world)
        client.post("/simulate", json={"match_id": match_id})
        final = wait_for_simulation(client, match_id)

        assert final["status"] == "finished"
        assert final["home_score"] != final["away_score"]   # no draw
        assert final["current_minute"] > 48                  # overtime was played
        assert (final["home_score"], final["away_score"]) == (4, 2)

    def test_returns_202_quickly_and_finishes(self, client, football_world, wait_for_simulation):
        match_id = make_match(client, football_world)
        started = time.monotonic()
        response = client.post("/simulate", json={"match_id": match_id})
        elapsed = time.monotonic() - started
        assert response.status_code == 202
        # PRD target is <200ms; allow slack for shared-suite CPU contention.
        assert elapsed < 1.0
        assert response.json()["status_url"] == f"/simulate/{match_id}/status"

        final = wait_for_simulation(client, match_id)
        assert final["status"] == "finished"
        assert final["current_minute"] == 90
        assert final["total_minutes"] == 90

    def test_score_matches_derived_match_score(self, client, football_world, wait_for_simulation):
        inject_high_goal_rates(client, football_world)
        match_id = make_match(client, football_world)
        client.post("/simulate", json={"match_id": match_id})
        final = wait_for_simulation(client, match_id)

        detail = client.get(f"/matches/{match_id}").json()
        assert detail["status"] == "finished"
        assert (detail["home_score"], detail["away_score"]) == (final["home_score"], final["away_score"])
        assert final["events_inserted"] > 0

    def test_two_concurrent_simulations_do_not_interfere(self, client, football_world, wait_for_simulation):
        inject_high_goal_rates(client, football_world)
        match_a = make_match(client, football_world)
        match_b = make_match(client, football_world)

        assert client.post("/simulate", json={"match_id": match_a}).status_code == 202
        assert client.post("/simulate", json={"match_id": match_b}).status_code == 202

        final_a = wait_for_simulation(client, match_a)
        final_b = wait_for_simulation(client, match_b)
        assert final_a["status"] == "finished"
        assert final_b["status"] == "finished"

        events_a = client.get(f"/matches/{match_a}/events").json()
        events_b = client.get(f"/matches/{match_b}/events").json()
        assert len(events_a) == final_a["events_inserted"]
        assert len(events_b) == final_b["events_inserted"]
        ids_a = {event["event_id"] for event in events_a}
        ids_b = {event["event_id"] for event in events_b}
        assert not ids_a & ids_b

    def test_duplicate_simulate_returns_409_and_stop_works(
        self, client, football_world, wait_for_simulation, monkeypatch
    ):
        monkeypatch.setattr(
            simulation_service, "get_settings", lambda: SimpleNamespace(SIMULATION_SPEED=0.05, SIMULATION_CALIBRATE=False)
        )
        match_id = make_match(client, football_world)
        assert client.post("/simulate", json={"match_id": match_id}).status_code == 202

        duplicate = client.post("/simulate", json={"match_id": match_id})
        assert duplicate.status_code == 409

        assert client.post(f"/simulate/{match_id}/stop").status_code == 200
        final = wait_for_simulation(client, match_id)
        assert final["status"] == "stopped"
        assert final["current_minute"] < 90
        # Stopping is graceful: the match is closed out, not left dangling.
        assert client.get(f"/matches/{match_id}").json()["status"] == "finished"

    def test_simulate_from_team_ids_creates_match_and_links(
        self, client, football_world, wait_for_simulation, mock_backend
    ):
        response = client.post(
            "/simulate",
            json={
                "home_team_id": football_world["home"],
                "away_team_id": football_world["away"],
                "sport_id": football_world["sport_id"],
                "sporty_match_id": MATCH_UUID,
                "sporty_home_team_id": HOME_TEAM_UUID,
                "sporty_away_team_id": AWAY_TEAM_UUID,
            },
        )
        assert response.status_code == 202
        match_id = response.json()["match_id"]
        wait_for_simulation(client, match_id)

        links = {(l["feeder_entity"], l["feeder_id"]): l["sporty_uuid"] for l in client.get("/links").json()}
        assert links[("match", match_id)] == MATCH_UUID
        assert links[("team", football_world["home"])] == HOME_TEAM_UUID

    def test_unknown_sport_is_422(self, client):
        sport_id = client.post("/sports", json={"name": "quidditch"}).json()["id"]
        home = client.post("/teams", json={"name": "Gryffindor", "sport_id": sport_id}).json()["id"]
        away = client.post("/teams", json={"name": "Slytherin", "sport_id": sport_id}).json()["id"]
        response = client.post(
            "/simulate",
            json={"home_team_id": home, "away_team_id": away, "sport_id": sport_id},
        )
        assert response.status_code == 422

    def test_match_without_players_errors_without_crashing(self, client, wait_for_simulation):
        sport_id = client.post("/sports", json={"name": "football"}).json()["id"]
        home = client.post("/teams", json={"name": "Empty A", "sport_id": sport_id}).json()["id"]
        away = client.post("/teams", json={"name": "Empty B", "sport_id": sport_id}).json()["id"]
        response = client.post(
            "/simulate",
            json={"home_team_id": home, "away_team_id": away, "sport_id": sport_id},
        )
        assert response.status_code == 202
        final = wait_for_simulation(client, response.json()["match_id"])
        assert final["status"] == "error"
        assert "no players" in final["error"]

    def test_status_for_unknown_simulation_is_404(self, client):
        assert client.get("/simulate/999/status").status_code == 404
        assert client.post("/simulate/999/stop").status_code == 409


class TestPushContract:
    def test_minute_batches_and_final_payloads(self, client, football_world, wait_for_simulation, mock_backend):
        inject_high_goal_rates(client, football_world)
        response = client.post(
            "/simulate",
            json={
                "home_team_id": football_world["home"],
                "away_team_id": football_world["away"],
                "sport_id": football_world["sport_id"],
                "sporty_match_id": MATCH_UUID,
                "sporty_home_team_id": HOME_TEAM_UUID,
                "sporty_away_team_id": AWAY_TEAM_UUID,
            },
        )
        match_id = response.json()["match_id"]
        final = wait_for_simulation(client, match_id)
        assert final["status"] == "finished"
        assert final["push_failures"] == 0

        match_pushes = mock_backend.captured["match_result"]
        assert match_pushes, "expected at least one minute-batch push"
        batched_events = [event for push in match_pushes for event in push["events"]]
        assert len(batched_events) == final["events_inserted"]
        for push in match_pushes:
            assert push["sporty_match_id"] == MATCH_UUID
            assert push["sport"] == "football"
        for event in batched_events:
            assert event["event_id"]
            assert event["sporty_team_id"] in {HOME_TEAM_UUID, AWAY_TEAM_UUID}

        assert match_pushes[-1]["status"] == "finished"
        assert match_pushes[-1]["home_score"] == final["home_score"]

        ratings_pushes = mock_backend.captured["player_ratings"]
        assert len(ratings_pushes) == 1
        ratings = ratings_pushes[0]["ratings"]
        assert ratings and all(1.0 <= entry["rating"] <= 10.0 for entry in ratings)

    def test_backend_down_simulation_survives_and_replay_recovers(
        self, client, football_world, wait_for_simulation, mock_backend, caplog
    ):
        inject_high_goal_rates(client, football_world)
        mock_backend.behaviour["fail"] = True

        with caplog.at_level(logging.ERROR):
            response = client.post(
                "/simulate",
                json={
                    "home_team_id": football_world["home"],
                    "away_team_id": football_world["away"],
                    "sport_id": football_world["sport_id"],
                    "sporty_match_id": MATCH_UUID,
                },
            )
            match_id = response.json()["match_id"]
            final = wait_for_simulation(client, match_id)

        assert final["status"] == "finished"  # no crash: simulation completed anyway
        assert final["push_failures"] > 0
        assert "failed after 3 attempts" in caplog.text
        assert mock_backend.captured["match_result"] == []

        stored_events = client.get(f"/matches/{match_id}/events").json()
        assert len(stored_events) == final["events_inserted"]

        # Backend comes back: replay delivers every stored event in one batch.
        mock_backend.behaviour["fail"] = False
        replay = client.post(f"/matches/{match_id}/replay-push")
        assert replay.status_code == 200
        assert replay.json() == {"match_id": match_id, "delivered": True, "events_sent": len(stored_events)}

        replay_push = mock_backend.captured["match_result"][-1]
        assert replay_push["sporty_match_id"] == MATCH_UUID
        assert len(replay_push["events"]) == len(stored_events)
        assert all(event["event_id"] for event in replay_push["events"])

    def test_replay_without_match_link_is_422(self, client, football_world):
        match_id = make_match(client, football_world)
        assert client.post(f"/matches/{match_id}/replay-push").status_code == 422
