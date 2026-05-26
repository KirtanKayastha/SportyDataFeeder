# /home/sam069/projects/SportyDataFeeder/app/schemas.py

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class SportCreate(BaseModel):
    name: str


class SportRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str


class TeamCreate(BaseModel):
    name: str
    sport_id: int


class TeamRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    sport_id: int


class PlayerCreate(BaseModel):
    name: str
    team_id: int
    position: str | None = None
    sport_id: int


class PlayerRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    team_id: int
    position: str | None = None
    sport_id: int


class MatchCreate(BaseModel):
    home_team_id: int
    away_team_id: int
    match_date: datetime
    sport_id: int


class MatchRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    home_team_id: int
    away_team_id: int
    match_date: datetime
    status: str
    sport_id: int


class MatchDetailRead(MatchRead):
    model_config = ConfigDict(from_attributes=True)

    home_score: int
    away_score: int


class EventCreate(BaseModel):
    event_type: str
    player_id: int | None = None
    minute: int | None = None
    extra: dict[str, Any] | None = None


class EventRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    match_id: int
    event_type: str
    player_id: int | None = None
    minute: int | None = None
    extra: dict[str, Any] | None = None
    created_at: datetime


class SimulationRequest(BaseModel):
    match_id: int


class SimulationResult(BaseModel):
    match_id: int
    events_inserted: int
    status: str


class ImportNBARequest(BaseModel):
    csv_path: str

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "csv_path": "nba_player_stats_2026.csv",
                }
            ]
        }
    )


class ImportPremierLeagueRequest(BaseModel):
    csv_paths: list[str]

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "csv_paths": [
                        "premier_league_complete_stats_until31thGameDayOnSeason2025-26.csv",
                        "premier_league_complete_stats_until35thGameDayOnSeason2025-26.csv",
                    ]
                }
            ]
        }
    )
