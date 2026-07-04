# Configuration, Deployment, Security, Logging, Background Jobs

## 1. Configuration

**File:** `app/config.py`. There is exactly **one** configuration mechanism in this
codebase: a single `pydantic_settings.BaseSettings` subclass, read from environment
variables or a `.env` file (`model_config = SettingsConfigDict(env_file=".env",
env_file_encoding="utf-8", extra="ignore")`). `CLAUDE.md` states, and code inspection
confirms, there is no `os.getenv` call anywhere else in `app/` or `scripts/` — every
config value flows through `get_settings()`.

### Every environment variable

| Variable | Type | Default | Required? | What breaks if missing/wrong |
|---|---|---|---|---|
| `DATABASE_URL` | `str` | **none — no default** | **Yes, always** | App fails to boot: `app/database.py:138` calls `get_settings().DATABASE_URL` at *module import time* (not lazily), so a missing value raises a Pydantic validation error before FastAPI even starts. Must be a SQLAlchemy-compatible URL (Postgres in real use — `postgresql://user:pass@host/db?sslmode=require`; SQLite `sqlite:///path.db` in tests). |
| `SPORTY_BACKEND_URL` | `str` | `"http://localhost:8001"` | No | If wrong/unreachable, every outbound push (`app/services/backend_client.py`) fails after 3 retries and logs an ERROR — the simulation and `/predict` endpoints keep working locally, only the live-feed integration silently stops functioning. `POST /demo/launch`'s orchestration calls (`schedule_match`, `register_players`) will raise (they use `_post_json`, which does `response.raise_for_status()` with no retry) and surface as a 5xx/exception from that endpoint. |
| `FEEDER_SECRET` | `str` | `"change-me"` | No default is safe | This is both (a) the value every inbound request must present in `X-Feeder-Secret` and (b) the value the feeder sends *outbound* to the Sporty backend. Leaving the placeholder `"change-me"` in any environment reachable by anyone but the operator is a real security hole — see §Security below. |
| `SIMULATION_SPEED` | `float` | `0.5` | No | Real seconds slept per simulated minute (`asyncio.sleep(speed)` in the minute loop). `0.5` ⇒ a 90-minute football match takes ~45 real seconds. `0` ⇒ no sleep at all — used by the test suite (`tests/conftest.py` sets it to `"0"`) for near-instant simulation runs; also usable in production for "fast-forward" demos. |
| `SIMULATION_CALIBRATE` | `bool` | `True` | No | When `True`, simulated scoring event rates are rescaled per-match so expected team totals match real league averages (see `SIMULATION.md` §3.3). Tests set this to `"false"` explicitly so they can inject fixed, known event rates and assert on raw un-rescaled counts — **if you write a new test that injects rates and expects them un-rescaled, you must also disable this**, or the test will silently observe scaled numbers. |

No other environment variables are read anywhere in the codebase (confirmed by grepping
`app/` and `scripts/` for `Settings(` / `os.environ` / `os.getenv`). `CLAUDE.md` notes
`REDIS_URL` and a `feeder.db` path were dead leftovers from before Phase 2 and were
deliberately deleted, not merely undocumented.

### `.env.example` (the shipped template)
```bash
DATABASE_URL=postgresql://USER:PASSWORD@HOST/DBNAME?sslmode=require
SPORTY_BACKEND_URL=https://your-sporty-backend.example.com
FEEDER_SECRET=32_char_random_string
SIMULATION_SPEED=0.5
```
Note `SIMULATION_CALIBRATE` is **not** in `.env.example` at all — it only ever needs
overriding by the test suite; an operator relying on the example file gets the safe
default (`True`).

### Caching behavior operators must know about
`get_settings()` is `@lru_cache`d — **the settings object is built exactly once per
process** and never re-read from the environment again, even if the underlying env vars
change. `tests/conftest.py`'s own comment states this explicitly: *"app.config.Settings
is cached and app.database creates the engine at import time, so DATABASE_URL has to
point at the throwaway SQLite file first"* — meaning any code (tests or otherwise) that
needs a different `DATABASE_URL` must set the environment variable **before the first
import** of any `app.*` module in that process, or it has no effect.

## 2. Deployment

### Docker (`Dockerfile`)
```dockerfile
FROM python:3.12-slim
WORKDIR /srv/feeder
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY alembic.ini . ; COPY alembic/ alembic/ ; COPY app/ app/ ; COPY scripts/ scripts/
EXPOSE 8000
CMD ["sh", "-c", "alembic upgrade head && python -m scripts.seed_sports && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
```
Key observations:
- **`models_pkl/` is deliberately excluded from the image.** The comment explains why:
  `event_rates.pkl` is keyed by *this specific database's* player integer ids, so baking
  a pkl trained against one database into a generic image would silently serve wrong
  predictions/rates against any other database. Operators must either mount a volume
  (`-v $(pwd)/models_pkl:/srv/feeder/models_pkl`) or train inside the running container
  (`docker exec <container> python -m scripts.train_models`, then restart so
  `app.state` picks it up — or use `POST /models/reload` to avoid the restart).
- **Migrations and seeding run on every container start**, not as a separate deploy
  step — `alembic upgrade head` is a documented no-op when the schema is already current,
  so this is safe to run on every restart, including in a multi-replica deployment
  *as long as migrations themselves are safe to run concurrently* (Alembic does not
  provide distributed locking here — a genuine risk if this image is ever scaled to >1
  replica starting simultaneously against a fresh database; see `IMPROVEMENTS.md`).
- **No multi-stage build, no non-root user, no `HEALTHCHECK` instruction** in the
  Dockerfile itself (health is exposed via `GET /health` but nothing in the image wires
  it to Docker's own healthcheck mechanism).
- Confirmed compatible with rootless Podman per the README, with a documented workaround
  (`--network=host`) for a known rootless port-mapping quirk.

### Running without Docker
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env               # edit DATABASE_URL etc.
alembic upgrade head
python -m scripts.seed_sports
uvicorn app.main:app --reload      # dev; drop --reload for anything resembling prod
```

### No CI/CD pipeline is present in this repository
No `.github/workflows/`, no `Jenkinsfile`, no `.gitlab-ci.yml` was found. `pytest` and any
build/push steps are manual, operator-run commands as documented in `README.md`.

### Scaling
The codebase is architected as a **single process**. The two structural blockers to
horizontal scaling, both already noted elsewhere in these docs and repeated here because
they are deployment-critical:
1. `app/services/simulation._simulations` is an in-memory Python dict — a second replica
   has no visibility into simulations running on the first, so `GET /simulate/{id}/status`
   would 404 if routed to the "wrong" replica, and the `409`-on-duplicate-start guard
   would not prevent two replicas from independently starting the same match twice.
2. `models_pkl/*.pkl` are loaded into each process's own `app.state` at boot / on
   `POST /models/reload` — with multiple replicas, a `/models/reload` call only refreshes
   whichever single replica received that specific HTTP request, leaving the others
   stale until independently triggered.

## 3. Health checks

`GET /health` (`app/main.py:106-122`, no auth required) is the only health endpoint.
It performs a real `SELECT 1` against the configured database (not just a "the process is
alive" check), and reports:
```json
{"status": "ok", "database": "up", "simulations_running": 2,
 "models": {"outcome_model": true, "outcome_v2": true,
            "outcome_v2_basketball": false, "event_rates": true,
            "event_rates_players": 842}}
```
A database failure returns `503` with the raw exception string in `detail` — useful for
debugging but note this **leaks connection/driver error text** to any caller, including
unauthenticated ones (the health endpoint is auth-exempt), which could reveal internal
network topology or credentials-adjacent detail depending on the driver's error
formatting. Missing model files never cause a non-`200` response — they're reported as
`false` flags, by design (graceful degradation, see `ARCHITECTURE.md`).

## 4. Logging

Every module uses the Python standard library's `logging` module
(`logger = logging.getLogger(__name__)`), never `print()` in application code (scripts'
own CLI output, e.g. `scripts/train_models.py`'s `main()`, does use `print`/`logger.info`
mixed, since those are meant to be read interactively at the terminal, not shipped
structured logs).

- **No structured logging (JSON logs) and no log-aggregation/tracing integration** (no
  OpenTelemetry, no Sentry, no `structlog`) — everything is plain-text `logging` module
  output, configured ad hoc per entry point:
  - `scripts/*.py` call `logging.basicConfig(level=logging.INFO, format="%(levelname)s
    %(message)s")` in their own `if __name__ == "__main__":` blocks.
  - `app/main.py` sets **no** `logging.basicConfig` at all — under plain `uvicorn
    app.main:app`, Uvicorn's own default logging configuration governs formatting
    unless the operator configures Python logging separately.
- **Log levels used meaningfully, not decoratively:**
  - `WARNING` — recoverable degraded states: missing pkl file, cold-start player/team,
    unlinked match (push skipped), a retried-but-not-yet-exhausted push failure.
  - `ERROR` — a push exhausted all 3 retry attempts (still non-fatal to the caller).
  - `INFO` — normal lifecycle events: simulation started/finished, model saved/loaded,
    calibration factor applied.
  - `DEBUG` — per-minute simulation detail (`Match %s minute %s: %s events, score
    %s-%s`) — verbose enough that it's explicitly `DEBUG`, not `INFO`, to avoid flooding
    production logs during a live simulation.
  - `logger.exception(...)` (captures full traceback) is used specifically at the two
    simulation-crash recovery points (`run_simulation`'s outer `except`).
- **No metrics/tracing system** (no Prometheus client, no `/metrics` endpoint, no APM
  agent) is present anywhere in `app/` or `requirements.txt`.

## 5. Background jobs

There is **no task-queue system** (no Celery, RQ, Dramatiq, or APScheduler) anywhere in
this codebase. "Background work" means exactly one thing: `asyncio.Task` objects created
directly on the running event loop (`app/services/simulation.py:start_simulation`), living
only as long as the process does.

**Scheduled/periodic work** is handled entirely outside the application process, via an
**operator-run or cron-invoked script**, not application code:
```bash
# suggested in scripts/refresh_bundles.py's own header comment
0 4 * * 1  cd /path/to/SportyDataFeeder && .venv/bin/python -m scripts.refresh_bundles \
           --fetch-nba --reload-url http://localhost:8000 >> refresh.log 2>&1
```
`scripts/refresh_bundles.py` is the only script explicitly designed to be cron-driven; it
(1) optionally re-scrapes recent NBA seasons, (2) rebuilds both production Elo bundles
from whatever CSV/SQLite data is currently on disk, and (3) optionally calls
`POST {url}/models/reload` on a live process to hot-swap the new bundles in without a
restart. No cron entry is actually installed by anything in this repository — it is
documentation of a *recommended* schedule, not a shipped cron job.

## 6. Security

### Authentication model
A **single shared secret** (`FEEDER_SECRET`), checked via `secrets.compare_digest` in
`app/main.py:require_feeder_secret` on every request except the small exempt-path set.
There is:
- **No JWT.** No `jwt`/`python-jose`/`pyjwt` dependency, no token issuance/verification
  code anywhere.
- **No sessions or cookies.** No `SessionMiddleware`, no cookie-setting code.
- **No refresh tokens, no OAuth, no per-user identity at all.** Every caller with the
  secret has full, undifferentiated access to every endpoint — there is exactly one
  privilege level.
- **No roles or permissions system.** Not applicable — see above.

This is an appropriate model for a single-operator internal tool talking to a single
trusted backend, and the code visibly treats it that way (the secret doubles as both
inbound-auth and outbound-auth credential, per `README.md`: *"the value must be identical
in this repo's `.env` and the Sporty backend's `.env`"*). It would **not** be appropriate
to expose this API directly to end users or the public internet without adding a real
auth layer.

### Input validation
Every request body is validated by Pydantic v2 schemas (`app/schemas.py`) before a router
function body ever runs — type coercion, required-field checks, and `Literal` enums
(e.g. `FeederEntity = Literal["sport", "team", "player", "match"]`) are enforced
automatically by FastAPI, with no custom validation middleware needed for the common case.
Business-rule validation (name non-blank, teams differ, sport IDs match) is hand-coded per
router (see `API.md`'s per-endpoint notes) — there is some duplication of the `_not_found`
helper across router files rather than a shared import, a minor consistency gap (see
`IMPROVEMENTS.md`).

### SQL injection
The codebase uses SQLAlchemy's ORM query API (`db.query(Model).filter_by(...)`,
`.filter(Column == value)`) throughout — **no raw string-interpolated SQL** was found in
`app/` or `scripts/`. The one raw-SQL usage, `text("SELECT 1")` in `GET /health`
(`app/main.py:110`), is a static string literal with no user input concatenated into it,
so it carries no injection risk.

### Secrets handling
- `FEEDER_SECRET` lives in `.env` (gitignored — confirmed via `.gitignore`) and is read
  once into the process via `pydantic-settings`. It is never logged.
- The Dockerfile does not embed any secret — it's supplied at `docker run --env-file .env`
  time.
- `X-Feeder-Secret` is compared with `secrets.compare_digest`, not `==`, specifically to
  avoid a timing side-channel that could let an attacker guess the secret one character at
  a time via response-time measurement — a genuinely good practice, correctly applied.

### Rate limiting
**None.** No `slowapi`, no custom middleware, no per-IP or per-key throttling anywhere.
Any caller with a valid (or even a repeatedly-guessed, since there's no lockout either)
secret can call any endpoint at any rate, including `POST /simulate`, which spins up a
background `asyncio.Task` — a malicious or buggy caller could exhaust server resources by
starting many simulations in a tight loop (bounded only by how many distinct `match_id`s
exist, since one match can't have two concurrent simulations).

### CORS
**No CORS middleware is configured** (`app/main.py` never imports or adds
`CORSMiddleware`). By default, a browser-based frontend on a different origin cannot call
this API directly — which lines up with the architecture: `PRD.md` states *"Frontend...
Never talks to the feeder. Reads everything via the Sporty backend."* The static admin
panel (`GET /admin`) works around this by being served *same-origin* from the feeder
itself (explicitly noted in `app/main.py`'s docstring: *"same-origin, so no CORS and the
secret is entered in the UI, never baked into a build"*).

### CSRF
Not applicable in the traditional sense — there is no cookie/session-based auth for CSRF
to exploit; the header-based shared-secret model is inherently not vulnerable to the
classic CSRF attack (a malicious page cannot induce a browser to attach a custom header
it doesn't know).

### XSS
The only HTML surface is the static `app/static/admin.html` console. Not independently
line-by-line audited in this pass, but it is a same-origin, operator-only page rendering
data the operator's own feeder database returns — the practical XSS blast radius is
limited to an operator attacking themselves via data they control, not an external
attacker (unless CSV imports or `POST` payloads containing malicious strings are later
reflected unescaped in the admin console — **could not fully verify from this audit
without a targeted line-by-line review of `admin.html`'s rendering code**).

## 7. Performance notes (see also `IMPROVEMENTS.md`)

- **Batched writes:** the importer commits every `BATCH_SIZE = 500` rows
  (`app/services/importer.py`), not once per row — a deliberate throughput optimization
  for large CSV imports.
- **One HTTP push per simulated minute**, never per event — the single biggest
  request-volume optimization in the whole system (explicitly commented in
  `simulation.py`).
- **Vectorised NumPy computation** in the Dixon-Coles negative-log-likelihood
  (`app/services/dixon_coles.py`) and the Poisson PMF — the Python-level loop is
  restricted only to the small subset of low-score match rows needing the `tau`
  correction.
- **No caching layer** beyond `get_settings()`'s `lru_cache` — every request that needs
  team strength, entity links, or player features re-queries the database fresh; for the
  demo-scale data volumes this project targets, this is a non-issue, but it would not
  scale to a high-traffic public API without changes.

## 8. Explain Like I'm New

Configuration is a single settings object read from one `.env` file — no scattered
magic strings. Security is intentionally simple: one shared password (not per-user
logins) protects the whole API, like a shared Wi-Fi password rather than individual
accounts, which is a reasonable choice for a backend service that only ever talks to one
other trusted backend service, not directly to the public. There's no separate task-queue
system for background work — "background jobs" here just means Python's built-in async
tasks running inside the same web-server process, so if that process restarts, anything
mid-flight is simply gone (no persistence, no resume).
