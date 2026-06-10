# /home/sam069/projects/SportyDataFeeder/tests/test_train_models.py

import logging

import pytest

from app.database import Event, Match, Player, PlayerStat, Session, Sport, Team
from app.services.ml_models import EVENT_RATES_FILE, OUTCOME_MODEL_FILE, load_model
from scripts.train_models import main as train_main


@pytest.fixture
def db(client):
    session = Session()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def football_world(db):
    """Two teams with one player each, both with stat history."""
    sport = Sport(name="football")
    db.add(sport)
    db.flush()
    home_team = Team(name="Strong FC", sport_id=sport.id)
    away_team = Team(name="Weak FC", sport_id=sport.id)
    db.add_all([home_team, away_team])
    db.flush()
    striker = Player(name="Star Striker", team_id=home_team.id, position="F", sport_id=sport.id)
    defender = Player(name="Quiet Defender", team_id=away_team.id, position="D", sport_id=sport.id)
    db.add_all([striker, defender])
    db.flush()
    db.add_all(
        [
            PlayerStat(player_id=striker.id, gameweek=1, season="2025-26", minutes=90, goals=2, points=8.5),
            PlayerStat(player_id=defender.id, gameweek=1, season="2025-26", minutes=90, goals=0, points=6.2),
        ]
    )
    db.commit()
    return {"sport": sport, "home": home_team, "away": away_team, "striker": striker, "defender": defender}


def make_finished_match(db, world, home_goals, away_goals):
    match = Match(
        home_team_id=world["home"].id,
        away_team_id=world["away"].id,
        sport_id=world["sport"].id,
        status="finished",
    )
    db.add(match)
    db.flush()
    for minute in range(home_goals):
        db.add(Event(match_id=match.id, event_type="goal", player_id=world["striker"].id, minute=minute + 1))
    for minute in range(away_goals):
        db.add(Event(match_id=match.id, event_type="goal", player_id=world["defender"].id, minute=minute + 1))
    db.commit()
    return match


def test_few_matches_skips_outcome_training_with_warning(db, football_world, tmp_path, caplog):
    make_finished_match(db, football_world, 2, 0)  # one finished match < 5
    with caplog.at_level(logging.WARNING):
        summary = train_main(models_dir=tmp_path)

    assert summary["outcome_model_trained"] is False
    assert "skipping outcome model" in caplog.text
    assert (tmp_path / EVENT_RATES_FILE).exists()
    assert not (tmp_path / OUTCOME_MODEL_FILE).exists()


def test_event_rates_keyed_by_player_id(db, football_world, tmp_path):
    train_main(models_dir=tmp_path)
    rates = load_model(EVENT_RATES_FILE, models_dir=tmp_path)
    striker_rates = rates[football_world["striker"].id]
    assert striker_rates["goal"] == pytest.approx(2 / 90)


def test_twenty_plus_matches_trains_with_split_and_report(db, football_world, tmp_path, caplog):
    for _ in range(8):
        make_finished_match(db, football_world, 2, 0)  # home win
    for _ in range(8):
        make_finished_match(db, football_world, 0, 1)  # away win
    for _ in range(8):
        make_finished_match(db, football_world, 1, 1)  # draw

    with caplog.at_level(logging.INFO):
        summary = train_main(models_dir=tmp_path)

    assert summary["finished_matches"] == 24
    assert summary["outcome_model_trained"] is True
    assert "held-out report" in caplog.text

    model = load_model(OUTCOME_MODEL_FILE, models_dir=tmp_path)
    assert model is not None
    # The scaler travels inside the pipeline — never a separate scaler.pkl.
    assert [name for name, _ in model.steps] == ["scaler", "clf"]
    assert not (tmp_path / "scaler.pkl").exists()
    probabilities = model.predict_proba([[0.6, 0.4, 1.0]])[0]
    assert probabilities.sum() == pytest.approx(1.0)
