# /home/sam069/projects/SportyDataFeeder/app/schemas.py

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

FeederEntity = Literal["sport", "team", "player", "match"]


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
    knockout: bool = False


class MatchRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    home_team_id: int
    away_team_id: int
    match_date: datetime
    status: str
    sport_id: int
    knockout: bool = False


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
    event_id: str | None = None
    match_id: int
    event_type: str
    player_id: int | None = None
    minute: int | None = None
    extra: dict[str, Any] | None = None
    created_at: datetime


class EntityLinkCreate(BaseModel):
    feeder_entity: FeederEntity
    feeder_id: int
    sporty_uuid: str


class EntityLinkRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    feeder_entity: str
    feeder_id: int
    sporty_uuid: str


class SimulateStartRequest(BaseModel):
    """Start a simulation for an existing match (match_id) OR create one from
    team ids + sport. Optional sporty UUIDs are upserted into entity_links."""

    match_id: int | None = None
    home_team_id: int | None = None
    away_team_id: int | None = None
    sport_id: int | None = None
    # Only used when creating the match here; an existing match_id keeps its
    # stored knockout flag.
    knockout: bool = False
    sporty_match_id: str | None = None
    sporty_home_team_id: str | None = None
    sporty_away_team_id: str | None = None


class SimulateStartResponse(BaseModel):
    match_id: int
    status: str
    status_url: str


class SimulationStatusRead(BaseModel):
    match_id: int
    status: str
    current_minute: int
    total_minutes: int
    home_score: int
    away_score: int
    events_inserted: int
    push_failures: int
    # Football only (None for basketball): running possession split and, for
    # knockout ties, the shootout tally + winner.
    possession_home_pct: float | None = None
    possession_away_pct: float | None = None
    shootout_home: int | None = None
    shootout_away: int | None = None
    shootout_winner_team_id: int | None = None
    error: str | None = None


class ReplayPushResult(BaseModel):
    match_id: int
    delivered: bool
    events_sent: int


class PredictRequest(BaseModel):
    match_id: int


class PredictResponse(BaseModel):
    match_id: int
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    model_version: str
    home_strength: float
    away_strength: float
    pushed: bool


class ImportNBARequest(BaseModel):
    csv_path: str
    gameweek: int | None = None
    season: str | None = None

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
    gameweek: int | None = None
    season: str | None = None

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
