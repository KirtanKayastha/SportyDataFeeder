# /home/sam069/projects/SportyDataFeeder/tests/test_prediction_metrics.py
#
# /predict/metrics (step 4 of reports/MODEL_IMPROVEMENT_PLAN.md): stored
# predictions are scored against finished matches, per model_version, with the
# latest prediction per (match, version) winning. Also covers POST
# /models/reload (hot-swap of rebuilt bundles).

import pytest

from app.database import Match, Session


@pytest.fixture
def football_match(client):
    sport_id = client.post("/sports", json={"name": "football"}).json()["id"]
    home = client.post("/teams", json={"name": "Home FC", "sport_id": sport_id}).json()["id"]
    away = client.post("/teams", json={"name": "Away FC", "sport_id": sport_id}).json()["id"]
    match_id = client.post(
        "/matches",
        json={"home_team_id": home, "away_team_id": away,
              "match_date": "2026-06-10T18:00:00", "sport_id": sport_id},
    ).json()["id"]
    striker = client.post(
        "/players",
        json={"name": "Home Striker", "team_id": home, "position": "F", "sport_id": sport_id},
    ).json()["id"]
    return {"match_id": match_id, "home_striker": striker}


def _finish(match_id: int) -> None:
    db = Session()
    try:
        db.query(Match).filter_by(id=match_id).update({"status": "finished"})
        db.commit()
    finally:
        db.close()


def test_metrics_empty_when_nothing_finished(client, football_match):
    client.post("/predict", json={"match_id": football_match["match_id"]})

    body = client.get("/predict/metrics").json()
    assert body["finished_matches_scored"] == 0
    assert body["predictions_scored"] == 0
    assert body["by_model_version"] == {}


def test_metrics_scores_prediction_against_result(client, football_match):
    match_id = football_match["match_id"]
    client.post("/predict", json={"match_id": match_id})
    # Home scores; the heuristic's top class is Home -> prediction is correct.
    client.post(f"/matches/{match_id}/events",
                json={"event_type": "goal", "player_id": football_match["home_striker"], "minute": 12})
    _finish(match_id)

    body = client.get("/predict/metrics").json()
    assert body["finished_matches_scored"] == 1
    assert body["predictions_scored"] == 1
    m = body["by_model_version"]["heuristic_v1"]
    assert m["n"] == 1
    assert m["accuracy"] == 1.0
    assert m["log_loss"] > 0
    assert 0 <= m["brier"] <= 2
    assert len(m["calibration"]) == 1
    assert m["calibration"][0]["n"] == 1


def test_metrics_use_latest_prediction_per_version(client, football_match):
    match_id = football_match["match_id"]
    client.post("/predict", json={"match_id": match_id})
    client.post("/predict", json={"match_id": match_id})  # supersedes the first
    client.post(f"/matches/{match_id}/events",
                json={"event_type": "goal", "player_id": football_match["home_striker"], "minute": 40})
    _finish(match_id)

    body = client.get("/predict/metrics").json()
    assert body["predictions_scored"] == 1  # not 2
    assert body["by_model_version"]["heuristic_v1"]["n"] == 1


def test_metrics_push_sends_scorecard_to_backend(client, football_match, monkeypatch):
    import json

    import httpx

    from app.routers import predict as predict_router
    from app.services.backend_client import BackendClient

    match_id = football_match["match_id"]
    client.post("/predict", json={"match_id": match_id})
    client.post(f"/matches/{match_id}/events",
                json={"event_type": "goal", "player_id": football_match["home_striker"], "minute": 5})
    _finish(match_id)

    captured = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200)

    monkeypatch.setattr(
        predict_router,
        "get_backend_client",
        lambda: BackendClient(
            base_url="http://sporty.test", secret="s", backoff_base=0.0,
            transport=httpx.MockTransport(handler),
        ),
    )

    body = client.post("/predict/metrics/push").json()
    assert body["pushed"] is True
    path, payload = captured[0]
    assert path == "/api/v1/feed/model-metrics"
    assert payload["predictions_scored"] == 1
    assert "generated_at" in payload
    assert payload["by_model_version"]["heuristic_v1"]["n"] == 1


def test_models_reload_endpoint(client):
    response = client.post("/models/reload")
    assert response.status_code == 200
    body = response.json()
    assert body["reloaded"] is True
    assert set(body["models"]) >= {"outcome_model", "outcome_v2",
                                   "outcome_v2_basketball", "event_rates"}


def test_models_reload_requires_secret(client):
    response = client.post("/models/reload", headers={"X-Feeder-Secret": "wrong"})
    assert response.status_code == 401
