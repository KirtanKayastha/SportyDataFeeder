# /home/sam069/projects/SportyDataFeeder/tests/test_ml_models.py

import logging

import pytest

from app.services.ml_models import (
    EVENT_RATES_FILE,
    MODEL_VERSION,
    heuristic_outcome,
    load_all_models,
    load_model,
    predict_outcome,
    save_model,
)


def test_save_and_load_round_trip(tmp_path):
    rates = {1: {"goal": 0.01}, 2: {"goal": 0.02}}
    save_model(rates, EVENT_RATES_FILE, models_dir=tmp_path)
    assert load_model(EVENT_RATES_FILE, models_dir=tmp_path) == rates


def test_missing_model_warns_and_returns_none(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        assert load_model("outcome_model.pkl", models_dir=tmp_path) is None
    assert "not found" in caplog.text


def test_load_all_models_with_empty_dir(tmp_path):
    loaded = load_all_models(models_dir=tmp_path)
    assert loaded == {"outcome_model": None, "event_rates": None}


def test_app_boots_without_pkls_and_health_reports_models(client):
    body = client.get("/health").json()
    assert body["models"]["outcome_model"] is False
    assert body["models"]["event_rates"] is False


def test_predict_outcome_maps_through_classes(tmp_path):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler

    # Train on only two classes (home=2, away=0) so classes_ is [0, 2]:
    # a positional [away, draw, home] assumption would misreport home as draw.
    X = [[0.9, 0.1, 1.0], [0.8, 0.2, 1.0], [0.1, 0.9, 1.0], [0.2, 0.8, 1.0]] * 5
    y = [2, 2, 0, 0] * 5
    model = Pipeline([("scaler", MinMaxScaler()), ("clf", LogisticRegression(max_iter=500))])
    model.fit(X, y)

    result = predict_outcome(model, home_strength=0.9, away_strength=0.1)
    assert result["model_version"] == MODEL_VERSION
    assert result["draw_prob"] == 0.0  # class 1 absent from training
    assert result["home_win_prob"] > result["away_win_prob"]
    total = result["home_win_prob"] + result["draw_prob"] + result["away_win_prob"]
    assert total == pytest.approx(1.0)


def test_predict_outcome_without_model_uses_heuristic():
    result = predict_outcome(None, home_strength=0.8, away_strength=0.2)
    assert result["model_version"] == "heuristic_v1"
    assert result["home_win_prob"] > result["away_win_prob"]
    total = result["home_win_prob"] + result["draw_prob"] + result["away_win_prob"]
    assert total == pytest.approx(1.0)


def test_heuristic_has_home_advantage():
    # Equal strengths: the home side is still favoured (home_bias).
    result = heuristic_outcome(0.5, 0.5)
    assert result["home_win_prob"] > result["away_win_prob"]
