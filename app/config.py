# /home/sam069/projects/SportyDataFeeder/app/config.py

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Single source of configuration. Values come from the environment or .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    DATABASE_URL: str
    SPORTY_BACKEND_URL: str = "http://localhost:8001"
    FEEDER_SECRET: str = "change-me"
    SIMULATION_SPEED: float = 0.5


@lru_cache
def get_settings() -> Settings:
    return Settings()
