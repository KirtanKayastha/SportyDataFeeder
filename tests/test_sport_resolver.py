# /home/sam069/projects/SportyDataFeeder/tests/test_sport_resolver.py

import pytest

from app.services.sport_resolver import SportType, resolve_sport_type


@pytest.mark.parametrize(
    "name",
    ["football", "Football", "soccer", "SOCCER", "premier league", "epl", "English Premier League"],
)
def test_football_aliases(name):
    assert resolve_sport_type(name) is SportType.FOOTBALL


@pytest.mark.parametrize(
    "name",
    ["basketball", "Basketball", "NBA", "nba", "NBA Basketball", "basket"],
)
def test_basketball_aliases(name):
    assert resolve_sport_type(name) is SportType.BASKETBALL


@pytest.mark.parametrize("name", ["cricket", "Cricket", "IPL"])
def test_cricket_aliases(name):
    assert resolve_sport_type(name) is SportType.CRICKET


@pytest.mark.parametrize("name", ["handball", "", None, "  ", "quidditch"])
def test_unknown_sports(name):
    assert resolve_sport_type(name) is SportType.UNKNOWN
