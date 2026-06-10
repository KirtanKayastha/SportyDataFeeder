# /home/sam069/projects/SportyDataFeeder/tests/test_features.py

import pytest

from app.database import Player, PlayerStat, Session, Sport, Team
from app.services.features import (
    BASKETBALL_FALLBACK_RATES,
    FOOTBALL_FALLBACK_RATES,
    NEUTRAL_FORM_INDEX,
    compute_player_features,
    compute_team_strength,
)
from app.services.sport_resolver import SportType


@pytest.fixture
def db(client):
    session = Session()
    try:
        yield session
    finally:
        session.close()


def make_player(db, sport_name="football", team_name="Arsenal", player_name="Player A"):
    sport = db.query(Sport).filter_by(name=sport_name).first()
    if not sport:
        sport = Sport(name=sport_name)
        db.add(sport)
        db.flush()
    team = db.query(Team).filter_by(name=team_name, sport_id=sport.id).first()
    if not team:
        team = Team(name=team_name, sport_id=sport.id)
        db.add(team)
        db.flush()
    player = Player(name=player_name, team_id=team.id, position="M", sport_id=sport.id)
    db.add(player)
    db.commit()
    return player


def add_stat(db, player, gameweek, *, minutes=90, goals=0, assists=0, yellows=0, reds=0, points=None, pts=None, ast=None, reb=None):
    db.add(
        PlayerStat(
            player_id=player.id,
            gameweek=gameweek,
            season="2025-26",
            minutes=minutes,
            goals=goals,
            assists=assists,
            yellows=yellows,
            reds=reds,
            points=points,
            pts=pts,
            ast=ast,
            reb=reb,
        )
    )
    db.commit()


class TestColdStart:
    def test_player_with_no_stats_gets_football_fallback(self, db):
        player = make_player(db)
        features = compute_player_features(player.id, db)
        assert features["cold_start"] is True
        assert features["event_rates"] == FOOTBALL_FALLBACK_RATES
        assert features["form_index"] == NEUTRAL_FORM_INDEX

    def test_basketball_player_gets_basketball_fallback(self, db):
        player = make_player(db, sport_name="basketball", team_name="LAL")
        features = compute_player_features(player.id, db)
        assert features["event_rates"] == BASKETBALL_FALLBACK_RATES

    def test_missing_player_never_raises(self, db):
        features = compute_player_features(99999, db)
        assert features["cold_start"] is True

    def test_zero_minute_rows_fall_back(self, db):
        player = make_player(db)
        add_stat(db, player, 1, minutes=0, goals=2)
        assert compute_player_features(player.id, db)["cold_start"] is True


class TestDerivedRates:
    def test_football_rates_from_stats(self, db):
        player = make_player(db)
        # 3 goals, 2 assists, 1 yellow in 180 minutes
        add_stat(db, player, 1, minutes=90, goals=2, assists=1, yellows=1, points=8.0)
        add_stat(db, player, 2, minutes=90, goals=1, assists=1, points=7.0)
        features = compute_player_features(player.id, db)
        assert features["cold_start"] is False
        assert features["event_rates"]["goal"] == pytest.approx(3 / 180)
        assert features["goals_per90"] == pytest.approx(1.5)
        assert features["assists_per90"] == pytest.approx(1.0)
        assert features["cards_per90"] == pytest.approx(0.5)
        assert features["minutes_ratio"] == 1.0

    def test_form_index_weights_newest_row_highest(self, db):
        player = make_player(db)
        add_stat(db, player, 1, points=4.0)
        add_stat(db, player, 2, points=10.0)  # newest
        form = compute_player_features(player.id, db)["form_index"]
        # EWMA newest-first: (0.4*10 + 0.24*4) / 0.64 = 7.75
        assert form == pytest.approx(7.75)
        assert form > (10.0 + 4.0) / 2  # newer high score dominates the plain mean

    def test_basketball_rates_decompose_points(self, db):
        player = make_player(db, sport_name="basketball", team_name="LAL")
        add_stat(db, player, 1, minutes=100, pts=50, ast=10, reb=20)
        features = compute_player_features(player.id, db)
        rates = features["event_rates"]
        assert rates["assist"] == pytest.approx(0.1)
        assert rates["rebound"] == pytest.approx(0.2)
        # 60% of 50 pts via 2s -> 15 made twos in 100 min
        assert rates["point_2"] == pytest.approx(0.15)
        assert rates["point_3"] == pytest.approx(50 * 0.25 / 3 / 100)
        assert rates["free_throw"] == pytest.approx(50 * 0.15 / 1 / 100)


class TestTeamStrength:
    def test_empty_team_is_neutral(self, db):
        sport = Sport(name="football")
        db.add(sport)
        db.flush()
        team = Team(name="Ghosts", sport_id=sport.id)
        db.add(team)
        db.commit()
        assert compute_team_strength(team.id, db) == 0.5

    def test_strength_is_normalised_mean_form(self, db):
        player = make_player(db)
        add_stat(db, player, 1, points=9.0)
        strength = compute_team_strength(player.team_id, db)
        assert strength == pytest.approx(9.0 / 15.0)

    def test_strength_is_clamped_to_1(self, db):
        player = make_player(db, sport_name="basketball", team_name="LAL")
        add_stat(db, player, 1, minutes=100, pts=200)  # absurd 72 pts/36min form
        assert compute_team_strength(player.team_id, db) == 1.0
