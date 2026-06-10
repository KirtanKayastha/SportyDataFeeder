# /home/sam069/projects/SportyDataFeeder/tests/test_crud.py

import pytest


@pytest.fixture
def sport_id(client) -> int:
    return client.post("/sports", json={"name": "football"}).json()["id"]


@pytest.fixture
def team_ids(client, sport_id) -> tuple[int, int]:
    home = client.post("/teams", json={"name": "Arsenal", "sport_id": sport_id}).json()["id"]
    away = client.post("/teams", json={"name": "Chelsea", "sport_id": sport_id}).json()["id"]
    return home, away


@pytest.fixture
def player_id(client, sport_id, team_ids) -> int:
    response = client.post(
        "/players",
        json={"name": "Declan Rice", "team_id": team_ids[0], "position": "M", "sport_id": sport_id},
    )
    return response.json()["id"]


@pytest.fixture
def match_id(client, sport_id, team_ids) -> int:
    home, away = team_ids
    response = client.post(
        "/matches",
        json={
            "home_team_id": home,
            "away_team_id": away,
            "match_date": "2026-06-10T18:00:00",
            "sport_id": sport_id,
        },
    )
    return response.json()["id"]


class TestSports:
    def test_create_and_list(self, client):
        created = client.post("/sports", json={"name": "football"})
        assert created.status_code == 201
        names = [sport["name"] for sport in client.get("/sports").json()]
        assert names == ["football"]

    def test_duplicate_rejected(self, client, sport_id):
        response = client.post("/sports", json={"name": "football"})
        assert response.status_code == 400

    def test_delete(self, client, sport_id):
        assert client.delete(f"/sports/{sport_id}").status_code == 200
        assert client.get("/sports").json() == []

    def test_delete_blocked_by_dependents(self, client, sport_id, team_ids):
        assert client.delete(f"/sports/{sport_id}").status_code == 400


class TestTeams:
    def test_create_and_filter_by_sport(self, client, sport_id, team_ids):
        teams = client.get("/teams", params={"sport_id": sport_id}).json()
        assert {team["name"] for team in teams} == {"Arsenal", "Chelsea"}

    def test_create_requires_existing_sport(self, client):
        response = client.post("/teams", json={"name": "Ghosts", "sport_id": 999})
        assert response.status_code == 404

    def test_delete(self, client, team_ids):
        assert client.delete(f"/teams/{team_ids[1]}").status_code == 200


class TestPlayers:
    def test_create_and_list_by_team(self, client, team_ids, player_id):
        players = client.get("/players", params={"team_id": team_ids[0]}).json()
        assert [player["name"] for player in players] == ["Declan Rice"]

    def test_sport_must_match_team(self, client, sport_id, team_ids):
        other_sport = client.post("/sports", json={"name": "basketball"}).json()["id"]
        response = client.post(
            "/players",
            json={"name": "Wrong Sport", "team_id": team_ids[0], "sport_id": other_sport},
        )
        assert response.status_code == 400

    def test_delete(self, client, player_id):
        assert client.delete(f"/players/{player_id}").status_code == 200


class TestMatches:
    def test_create_and_get_with_derived_score(self, client, match_id, player_id):
        detail = client.get(f"/matches/{match_id}").json()
        assert (detail["home_score"], detail["away_score"]) == (0, 0)

        client.post(f"/matches/{match_id}/events", json={"event_type": "goal", "player_id": player_id, "minute": 12})
        client.post(f"/matches/{match_id}/events", json={"event_type": "yellow_card", "player_id": player_id, "minute": 30})

        detail = client.get(f"/matches/{match_id}").json()
        assert (detail["home_score"], detail["away_score"]) == (1, 0)

    def test_same_team_rejected(self, client, sport_id, team_ids):
        response = client.post(
            "/matches",
            json={
                "home_team_id": team_ids[0],
                "away_team_id": team_ids[0],
                "match_date": "2026-06-10T18:00:00",
                "sport_id": sport_id,
            },
        )
        assert response.status_code == 400

    def test_get_missing_match_404(self, client):
        assert client.get("/matches/999").status_code == 404


class TestEvents:
    def test_create_and_list(self, client, match_id, player_id):
        created = client.post(
            f"/matches/{match_id}/events",
            json={"event_type": "goal", "player_id": player_id, "minute": 55, "extra": {"assist_player_id": 7}},
        )
        assert created.status_code == 201

        events = client.get(f"/matches/{match_id}/events").json()
        assert len(events) == 1
        assert events[0]["event_type"] == "goal"
        assert events[0]["extra"] == {"assist_player_id": 7}

    def test_event_for_missing_match_404(self, client):
        response = client.post("/matches/999/events", json={"event_type": "goal"})
        assert response.status_code == 404


class TestSimulation:
    def test_simulate_match_runs_in_background_and_finishes(
        self, client, sport_id, team_ids, match_id, wait_for_simulation
    ):
        home, away = team_ids
        for index in range(3):
            client.post(
                "/players",
                json={"name": f"Home Player {index}", "team_id": home, "position": "M", "sport_id": sport_id},
            )
            client.post(
                "/players",
                json={"name": f"Away Player {index}", "team_id": away, "position": "M", "sport_id": sport_id},
            )

        response = client.post("/simulate", json={"match_id": match_id})
        assert response.status_code == 202

        final = wait_for_simulation(client, match_id)
        assert final["status"] == "finished"
        assert client.get(f"/matches/{match_id}").json()["status"] == "finished"

        events = client.get(f"/matches/{match_id}/events").json()
        assert len(events) == final["events_inserted"]
        assert all(event["event_id"] for event in events)
