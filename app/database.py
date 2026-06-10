# /home/sam069/projects/SportyDataFeeder/app/database.py

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, scoped_session, sessionmaker

from app.config import get_settings


def utcnow() -> datetime:
    """Naive UTC timestamp (matches the historical datetime.utcnow column defaults)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


Base = declarative_base()


class Sport(Base):
    __tablename__ = 'sports'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)


class Team(Base):
    __tablename__ = 'teams'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    sport_id = Column(Integer, ForeignKey('sports.id'))
    sport = relationship("Sport")


class Player(Base):
    __tablename__ = 'players'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    team_id = Column(Integer, ForeignKey('teams.id'))
    position = Column(String)
    sport_id = Column(Integer, ForeignKey('sports.id'))
    team = relationship("Team")
    sport = relationship("Sport")


class Match(Base):
    __tablename__ = 'matches'
    id = Column(Integer, primary_key=True)
    home_team_id = Column(Integer, ForeignKey('teams.id'))
    away_team_id = Column(Integer, ForeignKey('teams.id'))
    match_date = Column(DateTime, default=utcnow)
    status = Column(String, default='scheduled')
    sport_id = Column(Integer, ForeignKey('sports.id'))


class Event(Base):
    __tablename__ = 'events'
    id = Column(Integer, primary_key=True)
    # Idempotency key for pushes/replays (PRD R-4.1); nullable for legacy rows.
    event_id = Column(String, unique=True, nullable=True)
    match_id = Column(Integer, ForeignKey('matches.id'))
    event_type = Column(String)
    player_id = Column(Integer, ForeignKey('players.id'))
    minute = Column(Integer)
    extra = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utcnow)


class EntityLink(Base):
    __tablename__ = 'entity_links'
    id = Column(Integer, primary_key=True)
    feeder_entity = Column(Text, nullable=False)
    feeder_id = Column(Integer, nullable=False)
    sporty_uuid = Column(Text, nullable=False)
    __table_args__ = (
        UniqueConstraint('feeder_entity', 'feeder_id', name='uq_entity_links_feeder_entity_feeder_id'),
    )


class PlayerStat(Base):
    __tablename__ = 'player_stats'
    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey('players.id'), nullable=False)
    gameweek = Column(Integer, nullable=False)
    season = Column(String, nullable=False)
    minutes = Column(Integer, nullable=True)
    # Football columns
    goals = Column(Integer, nullable=True)
    assists = Column(Integer, nullable=True)
    yellows = Column(Integer, nullable=True)
    reds = Column(Integer, nullable=True)
    points = Column(Float, nullable=True)
    # Basketball columns
    pts = Column(Integer, nullable=True)
    ast = Column(Integer, nullable=True)
    reb = Column(Integer, nullable=True)
    stl = Column(Integer, nullable=True)
    blk = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=utcnow)
    __table_args__ = (
        UniqueConstraint('player_id', 'gameweek', 'season', name='uq_player_stats_player_gameweek_season'),
    )


class MatchPrediction(Base):
    __tablename__ = 'match_predictions'
    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey('matches.id'), nullable=False)
    home_win_prob = Column(Float, nullable=False)
    draw_prob = Column(Float, nullable=False)
    away_win_prob = Column(Float, nullable=False)
    model_version = Column(String, nullable=False)
    created_at = Column(DateTime, default=utcnow)


class PlayerMatchRating(Base):
    __tablename__ = 'player_match_ratings'
    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey('matches.id'), nullable=False)
    player_id = Column(Integer, ForeignKey('players.id'), nullable=False)
    rating = Column(Float, nullable=False)
    is_man_of_match = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=utcnow)
    __table_args__ = (
        UniqueConstraint('match_id', 'player_id', name='uq_player_match_ratings_match_player'),
    )


DATABASE_URL = get_settings().DATABASE_URL

# SQLite is only used by the test suite; FastAPI's TestClient runs requests in
# worker threads, which SQLite rejects unless check_same_thread is disabled.
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=_connect_args)
# SessionFactory for code that needs sessions independent of the thread-local
# registry (e.g. concurrent asyncio simulation tasks share one thread).
SessionFactory = sessionmaker(bind=engine)
Session = scoped_session(SessionFactory)


def get_db():
    db = Session()
    try:
        yield db
    finally:
        db.close()
