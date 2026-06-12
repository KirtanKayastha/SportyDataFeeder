# Sporty Data Feeder

Sporty Data Feeder is an API-first sports data management service built with FastAPI and SQLAlchemy. It is configured for Postgres via `DATABASE_URL`; the legacy Streamlit UI has been removed and the business logic lives in the `app/` package.

## What it does

- Manages sports, teams, players, matches, and match events.
- Computes match scores from stored events via a single scoring-rules module.
- Supports manual event logging through API endpoints.
- Runs **stat-weighted live simulations** as asyncio background tasks: per-player per-minute event probabilities from `event_rates.pkl`, one push batch per simulated minute, ratings on finish.
- Pushes live data to the Sporty backend (`/api/v1/feed/*`, `X-Feeder-Secret`, retries with backoff, replay recovery after outages).
- Predicts match outcomes (`POST /predict`) via the trained model or a heuristic fallback.
- Imports NBA and Premier League rosters **and per-gameweek player stats** from CSV.
- Maps feeder integer IDs to Sporty backend UUIDs through the `entity_links` table.

## Architecture

- [app/main.py](app/main.py) - FastAPI entry point and router registration.
- [app/config.py](app/config.py) - Single `Settings` class (pydantic-settings); all configuration comes from `.env`.
- [app/database.py](app/database.py) - SQLAlchemy ORM models, engine, and session.
- [app/schemas.py](app/schemas.py) - Pydantic v2 request and response schemas.
- [app/routers/](app/routers) - CRUD, match/event/simulation, import, and entity-link endpoints.
- [app/services/scoring_rules.py](app/services/scoring_rules.py) - Single source of truth for event → score rules (used by both the match endpoint and the simulator).
- [app/services/sport_resolver.py](app/services/sport_resolver.py) - Maps free-form sport names to a `SportType` enum; the only place sport aliases live.
- [app/services/simulation.py](app/services/simulation.py) - Stat-weighted background simulation loop, match-scoped state registry, push payload builders.
- [app/services/backend_client.py](app/services/backend_client.py) - Async httpx client for Sporty feed pushes (3 attempts, 1.5^n backoff, never fatal).
- [app/services/importer.py](app/services/importer.py) - CSV roster + player-stats import logic.
- [app/services/links.py](app/services/links.py) - Entity-link upsert and `get_sporty_uuid` lookup helper.
- [app/services/features.py](app/services/features.py) - Per-player rates/form (EWMA α=0.4) from `player_stats`, with league-average cold-start fallbacks, and `compute_team_strength`.
- [app/services/ml_models.py](app/services/ml_models.py) - pkl load/save helpers, `predict_outcome` (maps probabilities through `model.classes_`), heuristic fallback when no model is present.
- [app/services/rater.py](app/services/rater.py) - Rule-based post-match ratings (base 6.0, clamp 1–10) and `find_man_of_match`.
- [scripts/seed_sports.py](scripts/seed_sports.py) - Idempotent seeding of the `sports` table.
- [scripts/train_models.py](scripts/train_models.py) - Builds `models_pkl/event_rates.pkl` and trains `models_pkl/outcome_model.pkl`.

## Database schema

Managed by **Alembic** (the app no longer creates tables on startup):

- `sports`: `id`, `name`
- `teams`: `id`, `name`, `sport_id`
- `players`: `id`, `name`, `team_id`, `position`, `sport_id`
- `matches`: `id`, `home_team_id`, `away_team_id`, `match_date`, `status`, `sport_id`
- `events`: `id`, `match_id`, `event_type`, `player_id`, `minute`, `extra`, `created_at`
- `entity_links`: `id`, `feeder_entity`, `feeder_id`, `sporty_uuid` — unique on `(feeder_entity, feeder_id)`
- `player_stats`: `id`, `player_id`, `gameweek`, `season`, `minutes`, football (`goals`, `assists`, `yellows`, `reds`, `points`) and basketball (`pts`, `ast`, `reb`, `stl`, `blk`) columns — unique on `(player_id, gameweek, season)`
- `match_predictions`, `player_match_ratings` — Phase 3 tables (schema in place, populated later)

## Run it

1. Create and activate a virtual environment, then install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and set `DATABASE_URL` to your Postgres instance (plus `SPORTY_BACKEND_URL`, `FEEDER_SECRET`, `SIMULATION_SPEED`).

3. Apply migrations and seed the sports table.

```bash
alembic upgrade head
python -m scripts.seed_sports
```

> **Existing databases:** if your tables were originally created by the old `create_all` startup hook, register the baseline once with `alembic stamp 0001_initial_schema`, then run `alembic upgrade head`.

4. Start the API.

```bash
uvicorn app.main:app --reload
```

5. Open the docs at `http://localhost:8000/docs`. Health check at `/health` (also reports whether the ML models are loaded).

## API authentication

Every endpoint except `/health`, `/docs`, and `/openapi.json` requires the shared secret header:

```
X-Feeder-Secret: <FEEDER_SECRET from .env>
```

Requests without it (or with a wrong value) get `401`. The same secret authenticates the feeder's outbound pushes to the Sporty backend, so the value must be identical in this repo's `.env` and the Sporty backend's `.env`. In the Swagger UI at `/docs`, pass the header per-request via the endpoint's parameters, or use `curl`:

```bash
curl -H "X-Feeder-Secret: $FEEDER_SECRET" http://localhost:8000/sports/
```

## Docker

```bash
docker build -t sporty-feeder .
docker run --env-file .env -p 8000:8000 sporty-feeder
```

The container runs `alembic upgrade head` and seeds the sports table on start, then serves the API on `:8000`. `models_pkl/` is not baked into the image (the pkls are keyed by your database's player ids) — either mount one in (`-v $(pwd)/models_pkl:/srv/feeder/models_pkl`) or train inside the container:

```bash
docker exec <container> python -m scripts.train_models   # then restart the container
```

Works with podman too (`podman build` / `podman run`); if rootless podman's `-p` port mapping doesn't respond on your machine, run with `--network=host` instead.

## Training the models

After importing stats (and ideally finishing some matches):

```bash
python -m scripts.train_models
```

This writes `models_pkl/event_rates.pkl` (per-player per-minute event probabilities, keyed by player id) and — given at least 5 finished matches — `models_pkl/outcome_model.pkl` (a sklearn `Pipeline(MinMaxScaler, LogisticRegression)`; the scaler is serialised inside the pipeline, never separately). With 20+ finished matches it does an 80/20 stratified split and logs a `classification_report`. Restart the API afterwards: models are loaded into `app.state` at startup, and a missing pkl only logs a warning (prediction falls back to a heuristic).

## Simulation and push

```
POST /simulate                      # 202; body: {"match_id": N} or team ids + sport_id,
                                    # optionally sporty_match_id / sporty_*_team_id UUIDs
GET  /simulate/{match_id}/status    # live minute, score, status, push failures
POST /simulate/{match_id}/stop      # graceful stop after the current minute
POST /matches/{match_id}/replay-push  # re-send all stored events (recovery after outage)
POST /predict                       # outcome probabilities; stores match_predictions row; pushes if linked
```

The simulation Bernoulli-samples each player's per-minute event probabilities (cold-start players use league averages), writes every event with a UUID `event_id` idempotency key, and pushes one batch per simulated minute to `{SPORTY_BACKEND_URL}/api/v1/feed/match-result` with the `X-Feeder-Secret` header. `SIMULATION_SPEED` is real seconds per simulated minute (0.5 ≈ 45 s per football match; 0 = max speed). Pushes are only sent when the match has a `sporty_match_id` entity link (`POST /links`); a backend outage never crashes a simulation — events persist locally and `replay-push` re-delivers them (the backend dedupes on `event_id`).

## Importing data

`POST /imports/premier-league` and `POST /imports/nba` import rosters and per-gameweek stat rows in one pass. Gameweek and season are inferred from the CSV filename (e.g. `...until31thGameDayOnSeason2025-26.csv` → gameweek 31, season `2025-26`; `nba_player_stats_2026.csv` → gameweek 0 = season-to-date, season `2026`) and can be overridden with the optional `gameweek` / `season` request fields. Re-running an import is idempotent: players dedupe on `(player_name, team_id)`, stats on `(player_id, gameweek, season)`.

The `sports` table must be seeded first (`python -m scripts.seed_sports`).

## End-to-end with Sporty (backend + frontend)

The full live-match pipeline is: **feeder simulation → `POST {SPORTY_BACKEND_URL}/api/v1/feed/*` → Sporty backend upserts events + publishes to Redis → frontend WebSocket (`/api/ws/match/{id}`) renders live score, prediction, and ratings** on the match page (`/match/{matchId}`).

To run all three locally:

1. **Sporty backend** (`~/projects/Sporty/Sporty_Backend`) on port **8000** — the frontend dev proxy assumes this. Set `FEEDER_SECRET` in its `.env` to the same value as this repo's `.env` (empty disables the feed endpoints with a 503).
2. **Sporty frontend** (`~/projects/Sporty/sporty-frontend`): `yarn dev` on port 3000.
3. **Feeder** on a different port, pointing at the backend:
   ```bash
   SPORTY_BACKEND_URL=http://localhost:8000 uvicorn app.main:app --reload --port 8010
   ```
4. **Link the entities.** Pushes are skipped (with a WARNING) until the feeder match is mapped to a Sporty match UUID. For each feeder match/team/player involved:
   ```bash
   curl -X POST http://localhost:8010/links \
     -H "X-Feeder-Secret: $FEEDER_SECRET" -H "Content-Type: application/json" \
     -d '{"feeder_entity": "match", "feeder_id": 1, "sporty_uuid": "<sporty match UUID>"}'
   ```
   (Player links let the backend credit fantasy points; an unlinked player's events still update the score.)
5. **Simulate:**
   ```bash
   curl -X POST http://localhost:8010/simulate \
     -H "X-Feeder-Secret: $FEEDER_SECRET" -H "Content-Type: application/json" \
     -d '{"match_id": 1}'
   ```
6. Open `http://localhost:3000/match/<sporty match UUID>` — the score ticker, events, prediction card, and (after the final whistle) player ratings update live.

If the backend was down during a simulation, re-deliver everything with `POST /matches/{id}/replay-push` — the backend dedupes on `event_id`, so replays are always safe.

## Tests

```bash
pytest
```

The suite (in [tests/](tests)) runs against a throwaway SQLite database — no Postgres needed. It covers app boot/health, the auth middleware, CRUD for every resource, scoring rules for football and basketball, sport-resolver aliases, entity links, importer idempotency, sport seeding, and verifies that `alembic upgrade head` builds the full schema on an empty database.

## Code locations

- API: [app/main.py](app/main.py)
- Settings: [app/config.py](app/config.py)
- Models: [app/database.py](app/database.py)
- Schemas: [app/schemas.py](app/schemas.py)
- Routers: [app/routers](app/routers)
- Services: [app/services](app/services)
- Migrations: [alembic/versions](alembic/versions)
- Scripts: [scripts](scripts)
- Tests: [tests](tests)
