# Sporty Data Feeder

Sporty Data Feeder is now an API-first sports data management service built with FastAPI and SQLAlchemy. The project is configured for Postgres via `DATABASE_URL`, and the legacy Streamlit UI has been removed; the business logic was migrated into the `app/` package.

## What it does

- Manages sports, teams, players, matches, and match events.
- Computes match scores from stored events.
- Supports manual event logging through API endpoints.
- Provides match simulation as a backend service without any Streamlit dependency.
- Includes CSV importer service functions for NBA and Premier League player data.
- Exposes API endpoints for running the CSV import services.

## Architecture

- [app/main.py](app/main.py) - FastAPI entry point and router registration.
- [app/database.py](app/database.py) - SQLAlchemy ORM models and environment-based database connection.
- [app/schemas.py](app/schemas.py) - Pydantic v2 request and response schemas.
- [app/routers/](app/routers) - CRUD and match/event/simulation endpoints.
- [app/services/simulation.py](app/services/simulation.py) - Random match generation and simulation logic.
- [app/services/importer.py](app/services/importer.py) - CSV import logic extracted from the old scripts.
- [app/routers/imports.py](app/routers/imports.py) - API endpoints for running the importer services.

## Database schema

- `sports`: `id`, `name`
- `teams`: `id`, `name`, `sport_id`
- `players`: `id`, `name`, `team_id`, `position`, `sport_id`
- `matches`: `id`, `home_team_id`, `away_team_id`, `match_date`, `status`, `sport_id`
- `events`: `id`, `match_id`, `event_type`, `player_id`, `minute`, `extra`, `created_at`

## Run it

1. Create and activate a virtual environment if needed.

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and confirm `DATABASE_URL` points at your Postgres instance.

4. Start the API.

```bash
uvicorn app.main:app --reload
```

5. Open the docs at `http://localhost:8000/docs`.

## Importing data

The old standalone import scripts were replaced by service functions in [app/services/importer.py](app/services/importer.py). The API layer can call them directly when you wire in an endpoint or a job runner.

## Notes

- The FastAPI app currently creates tables on startup via `Base.metadata.create_all(...)`; that is a temporary Phase 1 setup.
- `alembic/` is present as a placeholder for Phase 2 migrations.
- `streamlit` has been removed from the dependency list because the UI is no longer part of the project.
- `alembic/` contains the Phase 2 migration scaffold and can be expanded when migrations are enabled.

## Code locations

- API: [app/main.py](app/main.py)
- Models: [app/database.py](app/database.py)
- Schemas: [app/schemas.py](app/schemas.py)
- Routers: [app/routers](app/routers)
- Services: [app/services](app/services)
