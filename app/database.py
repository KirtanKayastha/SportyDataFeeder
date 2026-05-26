# /home/sam069/projects/SportyDataFeeder/app/database.py

import os
from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, scoped_session, sessionmaker

load_dotenv()

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
    match_date = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default='scheduled')
    sport_id = Column(Integer, ForeignKey('sports.id'))


class Event(Base):
    __tablename__ = 'events'
    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey('matches.id'))
    event_type = Column(String)
    player_id = Column(Integer, ForeignKey('players.id'))
    minute = Column(Integer)
    extra = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://neondb_owner:npg_fO5nNzYcZm3y@ep-crimson-paper-akak32va.c-3.us-west-2.aws.neon.tech/neondb?sslmode=require",
)

engine = create_engine(DATABASE_URL)
Session = scoped_session(sessionmaker(bind=engine))


def get_db():
    db = Session()
    try:
        yield db
    finally:
        db.close()
