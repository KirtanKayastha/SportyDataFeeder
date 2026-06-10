# /home/sam069/projects/SportyDataFeeder/tests/test_rater.py

import pytest

from app.services.rater import BASE_RATING, find_man_of_match, rate_player, rate_players
from app.services.sport_resolver import SportType


class TestFootballWeights:
    def test_base_rating_with_no_events(self):
        assert rate_player([], SportType.FOOTBALL) == BASE_RATING

    def test_goal_and_assist(self):
        assert rate_player(["goal", "assist"], SportType.FOOTBALL) == pytest.approx(6.0 + 2.0 + 1.2)

    def test_cards_subtract(self):
        assert rate_player(["yellow_card", "red_card"], SportType.FOOTBALL) == pytest.approx(6.0 - 0.5 - 2.5)

    def test_unknown_event_types_are_neutral(self):
        assert rate_player(["throw_in"], SportType.FOOTBALL) == BASE_RATING


class TestBasketballWeights:
    def test_scoring_events(self):
        rating = rate_player(["point_2", "point_3", "free_throw"], SportType.BASKETBALL)
        assert rating == pytest.approx(6.0 + 0.8 + 1.2 + 0.3)

    def test_defensive_events(self):
        rating = rate_player(["steal", "block", "rebound", "assist"], SportType.BASKETBALL)
        assert rating == pytest.approx(6.0 + 0.6 + 0.5 + 0.4 + 0.7)


class TestClamping:
    def test_clamped_at_10(self):
        assert rate_player(["goal"] * 5, SportType.FOOTBALL) == 10.0

    def test_clamped_at_1(self):
        assert rate_player(["red_card"] * 3, SportType.FOOTBALL) == 1.0


class TestManOfMatch:
    def test_argmax(self):
        events = {1: ["goal"], 2: ["goal", "goal", "assist"], 3: ["yellow_card"]}
        ratings = rate_players(events, SportType.FOOTBALL)
        assert find_man_of_match(ratings) == 2

    def test_tie_resolves_to_lowest_player_id(self):
        ratings = {7: 8.4, 3: 8.4, 9: 6.0}
        assert find_man_of_match(ratings) == 3

    def test_empty_ratings(self):
        assert find_man_of_match({}) is None
