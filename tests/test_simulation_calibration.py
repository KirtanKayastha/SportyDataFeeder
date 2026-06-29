# Unit tests for the scoring-rate calibration (app/services/simulation.py).
# These exercise the pure helper directly, so they are independent of the
# SIMULATION_CALIBRATE flag (which conftest disables for the API sim tests).

from types import SimpleNamespace

from app.services.scoring_rules import BASKETBALL_POINT_VALUES
from app.services.simulation import (
    BASKETBALL_AWAY_POINTS,
    BASKETBALL_HOME_POINTS,
    FOOTBALL_AWAY_GOALS,
    FOOTBALL_HOME_GOALS,
    calibrate_scoring_rates,
)
from app.services.sport_resolver import SportType


def _goals(rates, lineup, minutes):
    return minutes * sum(rates[p.id].get("goal", 0.0) for p in lineup)


def _points(rates, lineup, minutes):
    return minutes * sum(
        sum(BASKETBALL_POINT_VALUES[ev] * rates[p.id].get(ev, 0.0) for ev in BASKETBALL_POINT_VALUES)
        for p in lineup
    )


def test_football_calibration_applies_home_advantage_and_keeps_cards():
    home = [SimpleNamespace(id=i) for i in range(11)]
    away = [SimpleNamespace(id=100 + i) for i in range(11)]
    rates = {p.id: {"goal": 0.01, "yellow_card": 0.002} for p in home + away}

    scaled, (hf, af) = calibrate_scoring_rates(rates, home, away, SportType.FOOTBALL, 90)

    assert round(_goals(scaled, home, 90), 6) == round(FOOTBALL_HOME_GOALS, 6)
    assert round(_goals(scaled, away, 90), 6) == round(FOOTBALL_AWAY_GOALS, 6)
    assert hf > af  # home advantage: home scaled to a higher target
    assert all(scaled[p.id]["yellow_card"] == 0.002 for p in home + away)  # cards untouched


def test_basketball_calibration_home_advantage_and_preserves_mix():
    home = [SimpleNamespace(id=i) for i in range(5)]
    away = [SimpleNamespace(id=100 + i) for i in range(5)]
    rates = {p.id: {"point_2": 0.05, "point_3": 0.02, "free_throw": 0.03, "rebound": 0.1} for p in home + away}

    scaled, (hf, af) = calibrate_scoring_rates(rates, home, away, SportType.BASKETBALL, 48)

    assert round(_points(scaled, home, 48), 4) == round(BASKETBALL_HOME_POINTS, 4)
    assert round(_points(scaled, away, 48), 4) == round(BASKETBALL_AWAY_POINTS, 4)
    assert hf > af
    # all scoring events scaled by the SAME (home) factor; rebound untouched
    assert scaled[0]["point_3"] / 0.02 == hf
    assert scaled[0]["point_2"] / 0.05 == hf
    assert scaled[0]["rebound"] == 0.1


def test_calibration_noop_when_no_scoring_rate():
    home = [SimpleNamespace(id=i) for i in range(11)]
    away = [SimpleNamespace(id=100 + i) for i in range(11)]
    rates = {p.id: {"goal": 0.0, "yellow_card": 0.002} for p in home + away}
    scaled, factors = calibrate_scoring_rates(rates, home, away, SportType.FOOTBALL, 90)
    assert factors == (1.0, 1.0)
    assert scaled == rates


def test_calibration_noop_for_unknown_sport():
    home = [SimpleNamespace(id=0)]
    away = [SimpleNamespace(id=1)]
    rates = {0: {"goal": 0.5}, 1: {"goal": 0.5}}
    scaled, factors = calibrate_scoring_rates(rates, home, away, SportType.UNKNOWN, 90)
    assert factors == (1.0, 1.0)
    assert scaled == rates
