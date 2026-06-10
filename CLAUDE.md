# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Sporty Data Feeder is a FastAPI + SQLAlchemy service for managing sports data (sports, teams, players, matches, events), running stat-weighted live match simulations, pushing live data to the Sporty backend, and importing player rosters and per-gameweek stats from CSV. It was migrated from a Streamlit app to an API-first design; the Streamlit UI is gone. Phase 2 hardening (Stage B, R-2.1–R-2.9), Phase 3 features + ML (Stage C, R-3.1–R-3.5), and Phase 4 simulation + push (Stage D, R-4.1–R-4.5) are complete.

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # set DATABASE_URL (Postgres, e.g. Neon)

# Schema (Alembic is the source of truth; the app does NOT create tables)
alembic upgrade head
python -m scripts.seed_sports   # idempotent: inserts football + basketball

# Train ML models (after importing stats / finishing matches)
python -m scripts.train_models   # writes models_pkl/{event_rates,outcome_model}.pkl

# Run the API (docs at http://localhost:8000/docs, health at /health)
uvicorn app.main:app --reload

# Tests (run against throwaway SQLite; no Postgres needed)
pytest
```

There is no lint/format config in the repo.

## Architecture

The live application is the `app/` package. Request flow: `app/main.py` registers routers → routers in `app/routers/` handle HTTP and validation → business logic lives in `app/services/` → `app/database.py` defines ORM models and the session.

- `app/main.py` — FastAPI entry point, router registration, `/health`.
- `app/config.py` — single pydantic-settings `Settings` class reading `.env` (`DATABASE_URL`, `SPORTY_BACKEND_URL`, `FEEDER_SECRET`, `SIMULATION_SPEED`). `get_settings()` is cached — set env vars before importing app modules (tests do this in `tests/conftest.py`).
- `app/database.py` — SQLAlchemy 2.0 models (`Sport`, `Team`, `Player`, `Match`, `Event`, `EntityLink`, `PlayerStat`, `MatchPrediction`, `PlayerMatchRating`), engine, `scoped_session`, `get_db()`.
- `app/schemas.py` — Pydantic v2 schemas (`*Create` / `*Read`); reads use `ConfigDict(from_attributes=True)`.
- `app/routers/` — `sports`/`teams`/`players`/`matches` are mounted under a prefix; `events`/`simulation`/`imports`/`links`/`predict` declare full paths in their own routes (`/matches/{id}/events`, `/simulate`, `/simulate/{id}/status`, `/simulate/{id}/stop`, `/matches/{id}/replay-push`, `/predict`, `/imports/*`, `/links`).
- `app/services/scoring_rules.py` — THE single source of scoring truth: `score_events(events, sport_type)` and `event_score_value(...)`. Used by both `matches.py:_score_match` and `simulation.py`. Never duplicate point values at a call site.
- `app/services/sport_resolver.py` — `resolve_sport_type(name) -> SportType` enum (`FOOTBALL`, `BASKETBALL`, `CRICKET`, `UNKNOWN`) and the alias map. The ONLY place sport-name matching lives; adding cricket touches only this file.
- `app/services/simulation.py` — stat-weighted live simulation as an asyncio background task: `POST /simulate` returns 202, the loop Bernoulli-samples per-player per-minute probabilities from `app.state.event_rates` (league-average fallback for cold starts), writes events with UUID `event_id` idempotency keys, pushes ONE HTTP batch per minute, sleeps `SIMULATION_SPEED` per minute (0 = max speed). Match-scoped state registry (`_simulations`) keeps concurrent simulations independent; duplicate `/simulate` for a running match → 409. On finish: ratings (R-3.5) stored in `player_match_ratings` and pushed. Pushes are skipped (with a WARNING) when the match has no `entity_links` `sporty_match_id` mapping — use `/links` then `/matches/{id}/replay-push`.
- `app/services/backend_client.py` — httpx async pushes to `{SPORTY_BACKEND_URL}/api/v1/feed/*` with `X-Feeder-Secret`; 3 attempts, backoff 1.5^n; exhausted retries log ERROR and return False (simulation continues; replay recovers).
- `app/services/importer.py` — CSV import for NBA and Premier League: rosters + `player_stats` rows in one pass.
- `app/services/links.py` — `upsert_link` / `get_sporty_uuid(entity, id)` for feeder-ID ↔ Sporty-UUID mapping.
- `app/services/features.py` — `compute_player_features(player_id, db)` (per-90 rates, EWMA form_index α=0.4, per-minute event rates, cold-start fallbacks that never raise) and `compute_team_strength(team_id, db)` (mean form /15, clamped 0..1, 0.5 if empty). All feature reads go through here, never raw stat rows.
- `app/services/ml_models.py` — pkl save/load (`models_pkl/`), `MODEL_VERSION`, `predict_outcome` (maps probabilities via `model.classes_`, labels 0=away/1=draw/2=home), heuristic fallback when the model is missing.
- `app/services/rater.py` — rule-based ratings (base 6.0, clamp 1–10), `find_man_of_match` (ties → lowest player id).
- `scripts/seed_sports.py` — idempotent sports seeding; exposes `seed_sports(db)` for tests.
- `scripts/train_models.py` — builds `event_rates.pkl` and trains `outcome_model.pkl` (Pipeline with MinMaxScaler inside — never a separate scaler.pkl; <5 finished matches skips with a warning, ≥20 stratified 80/20 split + classification_report). Note: `multi_class=` was removed from sklearn 1.7+; lbfgs is multinomial by default.

## Conventions and gotchas specific to this codebase

- **Configuration only via `app/config.py`.** No `os.getenv` or hardcoded URLs anywhere else; `alembic/env.py` falls back to `Settings` when no URL is set programmatically. Root `legacy_database.py.bak` is a dead artifact — never import it.

- **Scores are never stored; they are derived from events** by replaying them through `scoring_rules.score_events`. Football counts `goal` events; basketball sums `points` from `extra` (falling back to canonical values for `point_2`/`point_3`/`free_throw`). Change scoring in `app/services/scoring_rules.py` only.

- **Sport type comes from `resolve_sport_type`, never substring checks.** `SportType.UNKNOWN` is scored as zero by `scoring_rules`; `matches.py:_score_match` deliberately falls back UNKNOWN→FOOTBALL to preserve pre-resolver behaviour.

- **`Event.extra` is a TEXT column holding JSON.** It is `json.dumps`'d on write and `json.loads`'d on read (with silent degrade to `None`/`{}`). Do not treat it as JSONB. Migration to JSONB is Phase 4+.

- **Importers expect a `Sport` row to already exist** (`ilike("football")` / exact `"basketball"`). Run `python -m scripts.seed_sports` first. NBA CSVs need `TEAM`/`PLAYER` columns; Premier League CSVs need `team_name`/`player_name`/`position`. Imports batch at `BATCH_SIZE = 500`; players dedupe on `(player_name, team_id)`, stats on `(player_id, gameweek, season)`. Gameweek/season are inferred from the CSV filename (`until31thGameDayOnSeason2025-26` → 31, `2025-26`; no gameweek → 0 = season-to-date) with optional request-body overrides.

- **Alembic is the source of truth for schema.** `main.py` does NOT call `create_all`. Migrations: `0001_initial_schema` (original 5 tables), `0002_entity_links_and_stats` (`entity_links`, `player_stats`, `match_predictions`, `player_match_ratings`). For a DB that predates Alembic, run `alembic stamp 0001_initial_schema` once, then `alembic upgrade head`.

- **Tests configure the environment before importing app modules.** `tests/conftest.py` sets `DATABASE_URL` to a temp SQLite file at import time because `Settings` is cached and the engine is created at module import. Follow the same pattern in new test files (just depend on the fixtures).

- **Models load into `app.state` at startup** (lifespan hook in `main.py`); `/health` reports `models.outcome_model` / `models.event_rates` plus `simulations_running`. A missing pkl is a WARNING, never a boot failure. `models_pkl/` is gitignored — pkls are environment-specific because `event_rates` is keyed by this DB's player ids.

- **Every event carries a UUID `event_id`** (migration `0003`; nullable for legacy rows — replay-push backfills them). It is the idempotency key for all pushes/replays; the Sporty backend upserts `ON CONFLICT (event_id) DO NOTHING`.

- **Simulation tests need care:** concurrent asyncio tasks share one thread, so simulations use `SessionFactory()` (not the thread-scoped `Session`). Tests that pass `sporty_match_id` MUST use the `mock_backend` fixture or pushes hit an unreachable URL and burn ~3.75s of retry backoff per minute-batch. `tests/conftest.py` clears the simulation registry between tests.

The build plan is PRD.md — follow requirement IDs and stage gates. Stages B, C, and D are complete; next is Stage E (Phase 5: Sporty backend integration, R-5.1–R-5.5 — built in the Sporty_Backend repo, not here). R-2.10 (auth middleware) was deliberately deferred out of the Stage B run.
