# Sporty Data Feeder — Engineering Documentation

This is a from-source technical audit of the Sporty Data Feeder codebase, written for a
senior engineer who has never seen the project. Every claim below was verified by reading
the actual file it references (not the docstrings, not `CLAUDE.md`, not the PRD) as of
the commit checked out when this was written (`main` / branch `Sam`, HEAD `ad1bd55`).
Where the code and the pre-existing prose docs (`CLAUDE.md`, `PRD.md`, `README.md`)
disagreed, the code wins and the discrepancy is called out explicitly.

**Could not determine from the code:** the exact production deployment target (only a
`Dockerfile` and manual-run instructions exist — no CI/CD pipeline, Kubernetes manifest,
or hosting config is present in this repository). Anything below about "production" means
"the Dockerfile's runtime contract," not an observed live environment.

## Document map

| File | Contents |
|---|---|
| [`00_INDEX.md`](00_INDEX.md) | This file — executive summary, workflow, doc map |
| [`GLOSSARY.md`](GLOSSARY.md) | Every abbreviation and domain term used across these docs |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | System diagrams, folder-by-folder walkthrough, request lifecycle, design/resilience patterns, algorithms |
| [`DATABASE.md`](DATABASE.md) | Schema, relationships, migrations, constraints, transaction behavior |
| [`API.md`](API.md) | Every HTTP endpoint: method, request/response shape, validation, auth, error codes |
| [`MODELS.md`](MODELS.md) | Every ML/statistical model: Elo, Dixon-Coles, event-rate tables, rule-based raters, evaluation metrics |
| [`SIMULATION.md`](SIMULATION.md) | Full simulation-engine deep dive: minute loop, event sampling, substitutions, discipline, overtime |
| [`HOW_A_RESULT_IS_PRODUCED.md`](HOW_A_RESULT_IS_PRODUCED.md) | Teach-ready explainer with worked numbers: historical data → per-player rates → calibration → Bernoulli minute loop → result |
| [`CONFIGURATION_DEPLOYMENT.md`](CONFIGURATION_DEPLOYMENT.md) | Every environment variable, Docker, health checks, logging, security posture |
| [`IMPROVEMENTS.md`](IMPROVEMENTS.md) | Concrete, code-referenced improvement opportunities |

## 1. Executive summary

### What this project is

The **Sporty Data Feeder** ("the feeder") is a standalone Python/FastAPI backend service
that owns all sports-data concerns for a fantasy-sports platform called **Sporty**. It is
a *separate* codebase and a *separate* Postgres database from the main Sporty backend
(`~/projects/Sporty/Sporty_Backend`, referred to throughout as "the Sporty backend" or
just "the backend") and the Sporty frontend (`~/projects/Sporty/sporty-frontend`). The
feeder never talks to the frontend and never touches the Sporty backend's database — the
only integration point is a set of authenticated outbound HTTP `POST`/`DELETE` calls the
feeder makes to `{SPORTY_BACKEND_URL}/api/v1/feed/*` (`app/services/backend_client.py`).

Concretely, the feeder:

1. Stores sports, teams, players, matches, and events in its own schema
   (`app/database.py`).
2. Imports real-world player rosters and per-gameweek statistics from CSV files
   (`app/services/importer.py`), for Premier League football and NBA basketball.
3. Trains and serves machine-learning artifacts — Elo-rating-based outcome predictors and
   per-player event-rate tables — from that historical data (`scripts/train_*.py`,
   `scripts/finalize_*.py`, `app/services/ml_models.py`).
4. Runs a **live match simulation engine** as an `asyncio` background task
   (`app/services/simulation.py`) that generates realistic minute-by-minute events (goals,
   assists, cards, substitutions, rebounds, etc.) using those trained per-player
   probabilities, rather than pure randomness.
5. Pushes each simulated minute's events to the Sporty backend over HTTP so the backend
   can fan them out to end users over WebSocket, and computes post-match player ratings.
6. Serves pre-match win/draw/loss probability predictions (`POST /predict`).

### What problem it solves

The system exists to feed a fantasy-sports product with **realistic-looking live match
data** without needing a real, licensed, real-time sports-data feed. Instead of pure
random-number-generator events (the system's own commit history and `PRD.md` §2.1
describe this as the "Phase 1" starting point — "a bench player and a star striker have
identical goal probabilities"), the feeder trains statistical models from real historical
football and basketball data (13 seasons of English Premier League results from
`football-data.co.uk`, ~19 seasons of NBA games from a Kaggle SQLite dump plus a
`basketball-reference.com` scrape) and uses those models to drive both pre-match
predictions and the live simulation.

### Main goals (per `PRD.md` §3.1, cross-checked against the shipped code)

| Goal | Status found in code |
|---|---|
| Stat-weighted simulation instead of uniform-random events | Done — `app/services/simulation.py` samples per-player, per-event-type Bernoulli probabilities |
| Pre-match outcome prediction from team/Elo strength | Done — `POST /predict` in `app/routers/predict.py`, backed by `app/services/ml_models.py` |
| Rule-based post-match player ratings | Done — `app/services/rater.py` |
| Push live match data to the Sporty backend | Done — `app/services/backend_client.py`, called from the simulation loop |
| Harden the codebase (config, migrations, tests, auth) | Done — single `Settings` class, Alembic-only schema, `pytest` suite, shared-secret middleware |

### Users

There is no end-user-facing UI in this repository. The "users" of this service are:

- **The Sporty backend**, which receives pushed match/prediction/rating data over HTTP.
- **A human operator** (developer/demo runner), who drives the feeder through
  `http://localhost:8000/docs` (Swagger UI) or the bundled static admin console at
  `GET /admin` (`app/static/admin.html`), or `curl`/`scripts/*.py` on the command line.
- **CI/local test runs**, via `pytest` against a throwaway SQLite database.

### High-level workflow

```
1. Seed reference data:      alembic upgrade head && python -m scripts.seed_sports
2. Import real rosters/stats: POST /imports/premier-league, POST /imports/nba
3. Import real HISTORICAL match results (separate, file-based, NOT via the API):
   scripts/load_historical.py (EPL) / scripts/load_nba.py (NBA) read CSV/SQLite
   files directly — they never touch the feeder's own `matches` table.
4. Train/refresh ML bundles:  scripts/finalize_outcome_v2.py (football),
   scripts/finalize_outcome_basketball.py (basketball), scripts/train_models.py
   (per-player event_rates.pkl + legacy outcome_model.pkl)
5. Create a scheduled match:  POST /matches
6. Predict its outcome:       POST /predict
7. Link it to the Sporty backend (or use the all-in-one POST /demo/launch):
   POST /links  (or /demo/launch does schedule + link + register players)
8. Simulate it live:          POST /simulate  -> asyncio background task
   -> per simulated minute: sample events -> write locally -> POST to Sporty backend
9. On finish: compute + store + push player ratings, refresh prediction metrics
10. Recovery: POST /matches/{id}/replay-push re-sends everything (backend dedupes
    on event_id)
```

### Technologies used

| Concern | Technology | Where |
|---|---|---|
| Web framework | FastAPI (ASGI) | `app/main.py`, `app/routers/*.py` |
| ASGI server | Uvicorn | `Dockerfile`, README run instructions |
| ORM | SQLAlchemy 2.0 (declarative, `scoped_session`) | `app/database.py` |
| Schema migrations | Alembic | `alembic/` |
| Validation/serialization | Pydantic v2 (`pydantic-settings` for config) | `app/schemas.py`, `app/config.py` |
| Production database | PostgreSQL (via `psycopg2-binary`), e.g. Neon | `.env` `DATABASE_URL` |
| Test database | SQLite (file-based, throwaway) | `tests/conftest.py` |
| ML / stats | scikit-learn (`LogisticRegression`, `Pipeline`, scalers), NumPy, pandas, SciPy (`scipy.optimize.minimize` for Dixon-Coles) | `app/services/ml_models.py`, `app/services/dixon_coles.py`, `scripts/train_*.py` |
| Outbound HTTP | httpx (async) | `app/services/backend_client.py` |
| Background concurrency | Python `asyncio` (in-process tasks, not Celery/RQ) | `app/services/simulation.py` |
| Data source scraping | `urllib.request` + `pandas.read_html` (basketball-reference), `football-data.co.uk` CSVs | `scripts/fetch_*.py` |

### What this system deliberately does NOT have

Documented explicitly so later sections don't need repeated caveats:

- **No Redis, no message broker, no Kafka.** `PRD.md` §1 states this as a design decision
  ("No Redis in the feeder... The feeder only pushes HTTP"), and no import of `redis`,
  `kafka`, `celery`, or similar appears anywhere in `app/` or `scripts/`. Background work
  is plain `asyncio.Task` objects held in an in-memory Python dict
  (`app/services/simulation.py:_simulations`). This means **all background simulation
  state is lost on process restart** — there is no persistence or recovery of an
  in-flight simulation across a crash (see `IMPROVEMENTS.md`).
- **No authentication/authorization for individual end users.** There is exactly one
  shared secret (`FEEDER_SECRET`), checked with `secrets.compare_digest` in
  `app/main.py`. There are no user accounts, roles, JWTs, sessions, or cookies in this
  codebase (see `CONFIGURATION_DEPLOYMENT.md` §Security for the full picture).
- **No caching layer.** No `functools.lru_cache` on hot paths besides
  `app.config.get_settings` (which caches the single `Settings` object, not query
  results).
- **No CI/CD configuration file** (no `.github/workflows`, no `Jenkinsfile`, etc.) found
  in this repository.
