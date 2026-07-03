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


def test_predict_prefers_outcome_v2_for_football(client, match_setup):
    """When an outcome_v2 bundle is loaded, football predictions use it (Elo)
    in preference to the v1 strength model."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # Minimal elo_logistic bundle: 1 feature (elo_diff), classes A/D/H.
    elo_diff = [[-300], [-100], [0], [100], [300]] * 6
    labels = ["A", "A", "D", "H", "H"] * 6
    model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500))])
    model.fit(elo_diff, labels)
    client.app.state.outcome_v2 = {
        "kind": "elo_logistic",
        "model": model,
        "elo_ratings": {"Strong FC": 1800.0, "Weak FC": 1300.0},
        "home_advantage": 65.0,
        "base": 1500.0,
    }
    # Also set a v1 model to prove v2 takes precedence.
    client.app.state.outcome_model = model

    body = client.post("/predict", json={"match_id": match_setup["match_id"]}).json()
    assert body["model_version"] == "outcome_v2_elo"
    assert body["home_win_prob"] + body["draw_prob"] + body["away_win_prob"] == pytest.approx(1.0)
    # Strong home (1800+65) vs weak away (1300): home win should dominate.
    assert body["home_win_prob"] > body["away_win_prob"]


def test_predict_v4_bundle_uses_sot_feature(client, match_setup):
    """A v4 football bundle (features=[elo_diff, sot_net_diff] + sot_form)
    assembles both features at predict time and reports its own version."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X = [[-300, -2.0], [-100, -1.0], [0, 0.0], [100, 1.0], [300, 2.0]] * 6
    labels = ["A", "A", "D", "H", "H"] * 6
    model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500))])
    model.fit(X, labels)
    client.app.state.outcome_v2 = {
        "kind": "elo_logistic",
        "model": model,
        "features": ["elo_diff", "sot_net_diff"],
        "elo_ratings": {"Strong FC": 1800.0, "Weak FC": 1300.0},
        "sot_form": {"Strong FC": 2.5, "Weak FC": -1.5},
        "home_advantage": 65.0,
        "base": 1500.0,
        "model_version": "outcome_v4_elo_sot",
    }

    body = client.post("/predict", json={"match_id": match_setup["match_id"]}).json()
    assert body["model_version"] == "outcome_v4_elo_sot"
    assert body["home_win_prob"] + body["draw_prob"] + body["away_win_prob"] == pytest.approx(1.0)
    assert body["home_win_prob"] > body["away_win_prob"]


def test_predict_unknown_bundle_feature_falls_back_to_v1(client, match_setup):
    """A bundle declaring a feature this build can't compute is skipped in
    favour of the v1 path instead of guessing."""
    client.app.state.outcome_v2 = {
        "kind": "elo_logistic",
        "model": object(),
        "features": ["elo_diff", "some_future_feature"],
        "elo_ratings": {"Strong FC": 1800.0},
        "home_advantage": 65.0,
        "base": 1500.0,
    }
    body = client.post("/predict", json={"match_id": match_setup["match_id"]}).json()
    assert body["model_version"] == "heuristic_v1"  # no v1 model loaded in tests


def test_predict_uses_basketball_bundle_with_no_draw(client):
    """A basketball match uses the basketball outcome_v2 bundle (2-class, draw=0)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    sport_id = client.post("/sports", json={"name": "basketball"}).json()["id"]
    home = client.post("/teams", json={"name": "MIL", "sport_id": sport_id}).json()["id"]
    away = client.post("/teams", json={"name": "DET", "sport_id": sport_id}).json()["id"]
    match_id = client.post(
        "/matches",
        json={"home_team_id": home, "away_team_id": away, "match_date": "2026-06-10T18:00:00", "sport_id": sport_id},
    ).json()["id"]

    # 2-class elo_logistic bundle (classes A/H, no draw), keyed by abbreviation.
    elo_diff = [[-400], [-150], [150], [400]] * 8
    labels = ["A", "A", "H", "H"] * 8
    model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500))])
    model.fit(elo_diff, labels)
    client.app.state.outcome_v2_basketball = {
        "kind": "elo_logistic",
        "sport": "basketball",
        "model": model,
        "elo_ratings": {"MIL": 1700.0, "DET": 1350.0},
        "aliases": {},
        "home_advantage": 100.0,
        "base": 1500.0,
    }

    body = client.post("/predict", json={"match_id": match_id}).json()
    assert body["model_version"] == "outcome_v2_elo"
    assert body["draw_prob"] == pytest.approx(0.0)  # basketball has no draws
    assert body["home_win_prob"] + body["away_win_prob"] == pytest.approx(1.0)
    assert body["home_win_prob"] > body["away_win_prob"]  # strong home vs weak away
