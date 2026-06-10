# /home/sam069/projects/SportyDataFeeder/tests/test_predict.py

import json
from types import SimpleNamespace

import httpx
import pytest

import app.routers.predict as predict_router
from app.database import MatchPrediction, Session
from app.services.backend_client import BackendClient

MATCH_UUID = "aaaaaaaa-0000-0000-0000-000000000099"


@pytest.fixture
def match_setup(client):
    sport_id = client.post("/sports", json={"name": "football"}).json()["id"]
    home = client.post("/teams", json={"name": "Strong FC", "sport_id": sport_id}).json()["id"]
    away = client.post("/teams", json={"name": "Weak FC", "sport_id": sport_id}).json()["id"]
    match_id = client.post(
        "/matches",
        json={"home_team_id": home, "away_team_id": away, "match_date": "2026-06-10T18:00:00", "sport_id": sport_id},
    ).json()["id"]
    return {"match_id": match_id}


def test_predict_uses_heuristic_without_model_and_stores_row(client, match_setup):
    response = client.post("/predict", json={"match_id": match_setup["match_id"]})
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "heuristic_v1"
    total = body["home_win_prob"] + body["draw_prob"] + body["away_win_prob"]
    assert total == pytest.approx(1.0)
    assert body["pushed"] is False  # no sporty_match_id link

    db = Session()
    try:
        row = db.query(MatchPrediction).filter_by(match_id=match_setup["match_id"]).one()
        assert row.model_version == "heuristic_v1"
        assert row.home_win_prob == pytest.approx(body["home_win_prob"])
    finally:
        db.close()


def test_predict_pushes_when_match_is_linked(client, match_setup, monkeypatch):
    captured = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200)

    monkeypatch.setattr(
        predict_router,
        "get_backend_client",
        lambda: BackendClient(
            base_url="http://sporty.test", secret="s", backoff_base=0.0, transport=httpx.MockTransport(handler)
        ),
    )
    client.post(
        "/links",
        json={"feeder_entity": "match", "feeder_id": match_setup["match_id"], "sporty_uuid": MATCH_UUID},
    )

    body = client.post("/predict", json={"match_id": match_setup["match_id"]}).json()
    assert body["pushed"] is True
    path, payload = captured[0]
    assert path == "/api/v1/feed/prediction"
    assert payload["sporty_match_id"] == MATCH_UUID
    assert payload["model_version"] == body["model_version"]


def test_predict_unknown_match_is_404(client):
    assert client.post("/predict", json={"match_id": 12345}).status_code == 404


def test_predict_uses_loaded_model_when_present(client, match_setup):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler

    X = [[0.9, 0.1, 1.0], [0.1, 0.9, 1.0], [0.5, 0.5, 1.0]] * 7
    y = [2, 0, 1] * 7
    model = Pipeline([("scaler", MinMaxScaler()), ("clf", LogisticRegression(max_iter=500))])
    model.fit(X, y)
    client.app.state.outcome_model = model

    body = client.post("/predict", json={"match_id": match_setup["match_id"]}).json()
    assert body["model_version"] == "outcome_v1_logistic"
    total = body["home_win_prob"] + body["draw_prob"] + body["away_win_prob"]
    assert total == pytest.approx(1.0)
