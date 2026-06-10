# /home/sam069/projects/SportyDataFeeder/tests/test_scoring_rules.py

from app.services.scoring_rules import event_score_value, score_events
from app.services.sport_resolver import SportType

HOME = 1
AWAY = 2


def test_football_scoring_counts_only_goals():
    events = [
        {"event_type": "goal", "team_id": HOME, "extra": {}},
        {"event_type": "goal", "team_id": HOME, "extra": {"assist_player_id": 5}},
        {"event_type": "goal", "team_id": AWAY, "extra": None},
        {"event_type": "yellow_card", "team_id": AWAY, "extra": {}},
        {"event_type": "red_card", "team_id": HOME, "extra": {}},
        {"event_type": "assist", "team_id": AWAY, "extra": {}},
    ]
    scores = score_events(events, SportType.FOOTBALL)
    assert scores.get(HOME, 0) == 2
    assert scores.get(AWAY, 0) == 1


def test_basketball_scoring_sums_points():
    events = [
        {"event_type": "point_2", "team_id": HOME, "extra": {"points": 2}},
        {"event_type": "point_3", "team_id": HOME, "extra": {"points": 3}},
        {"event_type": "free_throw", "team_id": AWAY, "extra": {"points": 1}},
        {"event_type": "rebound", "team_id": AWAY, "extra": {}},
        {"event_type": "turnover", "team_id": HOME, "extra": {}},
    ]
    scores = score_events(events, SportType.BASKETBALL)
    assert scores.get(HOME, 0) == 5
    assert scores.get(AWAY, 0) == 1


def test_basketball_falls_back_to_default_point_values():
    # Events without an explicit "points" in extra use the canonical values.
    assert event_score_value("point_2", None, SportType.BASKETBALL) == 2
    assert event_score_value("point_3", {}, SportType.BASKETBALL) == 3
    assert event_score_value("free_throw", {"assist_player_id": 9}, SportType.BASKETBALL) == 1


def test_goal_does_not_score_in_basketball_and_vice_versa():
    assert event_score_value("goal", {}, SportType.BASKETBALL) == 0
    assert event_score_value("point_2", {"points": 2}, SportType.FOOTBALL) == 0


def test_events_without_team_are_ignored():
    events = [{"event_type": "goal", "team_id": None, "extra": {}}]
    assert score_events(events, SportType.FOOTBALL) == {}


def test_unknown_sport_scores_nothing():
    events = [{"event_type": "goal", "team_id": HOME, "extra": {}}]
    assert score_events(events, SportType.UNKNOWN) == {}
