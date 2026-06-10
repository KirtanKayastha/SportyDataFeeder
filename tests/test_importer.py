# /home/sam069/projects/SportyDataFeeder/tests/test_importer.py

from pathlib import Path

import pytest

from app.database import Player, PlayerStat, Session
from app.services.importer import import_nba, import_premier_league, infer_gameweek_and_season
from scripts.seed_sports import seed_sports

EPL_CSV_NAME = "premier_league_complete_stats_until31thGameDayOnSeason2025-26.csv"
NBA_CSV_NAME = "nba_player_stats_2026.csv"

EPL_CSV = """player_name,team_name,position,appearances,assists,goals,minutesPlayed,rating,redCards,yellowCards
Declan Rice,Arsenal,M,30,5,4,2585,7.49,0,2
Bukayo Saka,Arsenal,F,28,9,12,2410,7.81,0,1
Erling Haaland,Manchester City,F,29,3,21,2500,7.55,0,3
"""

NBA_CSV = """PLAYER_ID,RANK,PLAYER,TEAM_ID,TEAM,GP,MIN,AST,STL,BLK,REB,PTS
1629029,1,Luka Doncic,1610612747,LAL,64,2289,530,105,34,495,2143
203999,2,Nikola Jokic,1610612743,DEN,70,2400,680,120,60,890,2050
"""


@pytest.fixture
def db():
    session = Session()
    try:
        seed_sports(session)
        yield session
    finally:
        session.close()


@pytest.fixture
def epl_csv(tmp_path) -> Path:
    path = tmp_path / EPL_CSV_NAME
    path.write_text(EPL_CSV)
    return path


@pytest.fixture
def nba_csv(tmp_path) -> Path:
    path = tmp_path / NBA_CSV_NAME
    path.write_text(NBA_CSV)
    return path


def test_infer_gameweek_and_season_from_filenames():
    assert infer_gameweek_and_season(Path(EPL_CSV_NAME)) == (31, "2025-26")
    assert infer_gameweek_and_season(Path(NBA_CSV_NAME)) == (0, "2026")


def test_epl_import_creates_rosters_and_stats(db, epl_csv):
    result = import_premier_league(db, [epl_csv])
    totals = result["totals"]
    assert totals["teams_added"] == 2
    assert totals["players_added"] == 3
    assert totals["stats_added"] == 3

    rice = db.query(Player).filter_by(name="Declan Rice").one()
    stat = db.query(PlayerStat).filter_by(player_id=rice.id).one()
    assert (stat.gameweek, stat.season) == (31, "2025-26")
    assert stat.minutes == 2585
    assert stat.goals == 4
    assert stat.assists == 5
    assert stat.yellows == 2
    assert stat.reds == 0
    assert stat.points == pytest.approx(7.49)


def test_epl_import_is_idempotent(db, epl_csv):
    import_premier_league(db, [epl_csv])
    second = import_premier_league(db, [epl_csv])
    totals = second["totals"]
    assert totals["players_added"] == 0
    assert totals["stats_added"] == 0
    assert totals["stats_skipped"] == 3
    assert db.query(PlayerStat).count() == 3


def test_epl_import_same_players_new_gameweek_adds_stat_rows(db, epl_csv, tmp_path):
    import_premier_league(db, [epl_csv])
    later_csv = tmp_path / "premier_league_complete_stats_until35thGameDayOnSeason2025-26.csv"
    later_csv.write_text(EPL_CSV)
    result = import_premier_league(db, [later_csv])
    assert result["totals"]["players_added"] == 0
    assert result["totals"]["stats_added"] == 3
    assert db.query(PlayerStat).count() == 6


def test_nba_import_creates_rosters_and_stats(db, nba_csv):
    result = import_nba(db, nba_csv)
    assert result["teams_added"] == 2
    assert result["players_added"] == 2
    assert result["stats_added"] == 2

    luka = db.query(Player).filter_by(name="Luka Doncic").one()
    stat = db.query(PlayerStat).filter_by(player_id=luka.id).one()
    assert (stat.gameweek, stat.season) == (0, "2026")
    assert stat.minutes == 2289
    assert stat.pts == 2143
    assert stat.ast == 530
    assert stat.reb == 495
    assert stat.stl == 105
    assert stat.blk == 34


def test_nba_import_is_idempotent(db, nba_csv):
    import_nba(db, nba_csv)
    second = import_nba(db, nba_csv)
    assert second["players_added"] == 0
    assert second["stats_added"] == 0
    assert second["stats_skipped"] == 2
    assert db.query(PlayerStat).count() == 2
