# Sporty Data Feeder — Product Requirements Document

| Field | Value |
|---|---|
| **Version** | 2.0 — supersedes PRD v1.1 |
| **Date** | June 2026 |
| **Status** | Phase 1 complete (FastAPI CRUD + random simulation). This document covers Phases 2–5. |
| **System** | Sporty Data Feeder — ML-enhanced simulation service for the Sporty fantasy platform |
| **Audience** | Claude Code / engineering — written to be directly executable as a build plan |
| **Codebase** | Existing FastAPI + SQLAlchemy app (migrated from Streamlit). See Section 2 for current state. |

> **Purpose.** This PRD reflects the ACTUAL current codebase (per CLAUDE.md) and defines the remaining work: replacing random simulation with stat-weighted ML simulation, pushing live data to the Sporty backend, and hardening the existing architecture. Every requirement is written against the code that exists today — not a greenfield design.

---

## 1. Executive summary

The Sporty Data Feeder is a standalone FastAPI service that owns everything sports-specific for the Sporty fantasy platform: rosters, matches, events, simulation, and (after this phase) ML-driven predictions and ratings. The main Sporty backend stays domain-agnostic — it receives clean, enriched data from the feeder and applies fantasy logic (leagues, squads, scoring, standings) on top.

Phase 1 is complete: the Streamlit prototype has been replaced by an API-first FastAPI app with SQLAlchemy 2.0 models, CRUD routers, CSV importers (NBA + Premier League), and a random match simulator. The core problem remains: **the simulation is random**. A bench player and a star striker have identical goal probabilities, so the data is structurally valid but statistically meaningless. Phases 2–5 fix this and connect the feeder to the Sporty backend in real time.

| Dimension | Decision |
|---|---|
| Primary output | Enriched match data: stat-weighted live events, outcome predictions, player ratings |
| Database | Feeder keeps its OWN Postgres DB (already exists — Neon). It does NOT share the Sporty DB. **No new DB needed.** |
| Redis | **No Redis in the feeder.** The Sporty backend owns Redis pub/sub for WebSocket fanout. The feeder only pushes HTTP. |
| Integration | Outbound push: 3 authenticated `POST /feed/*` endpoints on the Sporty backend |
| ML approach | scikit-learn — pkl models loaded into `app.state` at startup. No model server, no GPU. |
| Frontend | Never talks to the feeder. Reads everything via the Sporty backend (REST + WebSocket). |

---

## 2. Current state of the codebase

This section documents what exists today so the build plan can be incremental. Source: CLAUDE.md and the live `app/` package.

### 2.1 What is built and working

| Component | Location | State |
|---|---|---|
| FastAPI entry point | `app/main.py` | Registers routers, `/health`, calls `Base.metadata.create_all()` on startup |
| ORM models | `app/database.py` | `Sport`, `Team`, `Player`, `Match`, `Event` — SQLAlchemy 2.0, scoped_session, `get_db()` |
| Pydantic schemas | `app/schemas.py` | `*Create` / `*Read` pairs, `ConfigDict(from_attributes=True)` |
| CRUD routers | `app/routers/` | sports, teams, players, matches (prefixed); events, simulation, imports (full paths) |
| Random simulator | `app/services/simulation.py` | Generates random events; live simulation loop exists |
| CSV importers | `app/services/importer.py` | NBA (`TEAM`/`PLAYER` cols) + Premier League (`team_name`/`player_name`/`position`). Batch 500, dedupe on `(player_name, team_id)` |
| Alembic | `alembic/` | Scaffolded, one migration exists, NOT yet source of truth (`create_all` still active) |

### 2.2 Known debt and gotchas (must be respected by all new code)

| # | Gotcha | Implication for new work |
|---|---|---|
| G1 | Two `database.py` files — root one is legacy | Only ever edit `app/database.py`. Never import the root module. |
| G2 | Scores are derived, never stored — `_score_match` replays events; same rules duplicated in `simulation.py` | Phase 2 must centralise scoring rules in ONE module both callers import (see R-2.4) |
| G3 | Sport detection by substring: `"basket" in name.lower()` | Phase 2 introduces a sport_type resolver function; all new code uses it (see R-2.5) |
| G4 | `Event.extra` is TEXT holding JSON (`json.dumps`/`loads`, silent degrade) | Keep as-is for now; do not assume JSONB. Migration to JSONB is Phase 4+. |
| G5 | Importers require `Sport` row to pre-exist (exact `"basketball"` / `ilike("football")`) | Add a seed script (R-2.6) so fresh environments work |
| G6 | `create_all` on startup, Alembic not authoritative | Phase 2 flips this: remove `create_all`, make Alembic source of truth (R-2.7) |
| G7 | Hardcoded Neon URL fallback in several files; `feeder.db` + `REDIS_URL` are dead leftovers | Phase 2 cleanup: single Settings class, delete dead config (R-2.8 / R-2.2) |
| G8 | No tests, no lint config | Phase 2 adds pytest + a minimal smoke suite (R-2.9) |

---

## 3. Goals and non-goals

### 3.1 Goals

| ID | Goal | Success metric |
|---|---|---|
| G1 | Replace random simulation with stat-weighted event generation | Simulated goal rates within 15% of historical per-player averages over 100 simulated matches |
| G2 | Pre-match outcome predictions from team strength | Outcome model accuracy > 50% on held-out matches (3-class baseline is 33%) |
| G3 | Post-match player ratings (1–10), rule-based | Ratings correlate with goal+assist counts (r > 0.7) |
| G4 | Push live match data to Sporty backend during simulation | Event batch received by backend < 1s after simulation tick |
| G5 | Frontend shows live score via Sporty WebSocket | Score update visible < 2s after simulated event |
| G6 | Harden existing codebase (Alembic, config, tests, scoring single-source) | All G1–G8 gotchas from Section 2.2 resolved or explicitly deferred |

### 3.2 Non-goals

> Out of scope: user authentication for end-users (Sporty backend owns it) · frontend implementation · real live sports API ingestion · deep learning / GPU models · Kafka or event streaming infra · multi-tenancy · public-facing API · **cricket simulation** (Sporty supports cricket, but the feeder explicitly defers it — the sport resolver must be built so cricket can be added without touching every branch) · **sharing the Sporty database** (feeder keeps its own DB) · **running Redis inside the feeder**.

---

## 4. Target architecture

The feeder keeps its existing own database. It does not read or write the Sporty database directly — all integration is over HTTP push. This preserves the separation that already exists in the codebase and avoids coupling two schemas.

| Layer | Component | Responsibility |
|---|---|---|
| 1 — Data | Feeder Postgres (existing Neon DB) | sports, teams, players, matches, events + new: `player_stats`, `entity_links`, `match_predictions`, `player_match_ratings` |
| 2 — Import | `app/services/importer.py` (existing) | Kaggle CSV → players/teams. Already built; gains a stats-row importer in Phase 2. |
| 3 — Features | `app/services/features.py` (new) | goals_per90, assists_per90, cards_per90, form_index (EWMA α=0.4), team_strength |
| 4 — ML | `models_pkl/*.pkl` + `app/services/ml_models.py` (new) | `outcome_model.pkl` (Pipeline: MinMaxScaler + LogisticRegression), `event_rates.pkl` (per-player per-minute probabilities) |
| 5 — Simulation | `app/services/simulation.py` (rewrite) | Bernoulli sampling per player per minute from event_rates; async background task; never blocks a request |
| 6 — Push | `app/services/backend_client.py` (new) | httpx async POST to Sporty `/api/v1/feed/*` with `X-Feeder-Secret` header; 3 retries, exponential backoff |
| 7 — Sporty side | `Sporty_Backend/app/api/v1/feed.py` (new, in Sporty repo) | Validates secret, upserts events (idempotent on event_id), updates score, publishes to Redis `live:match:{id}`, triggers fantasy scoring on finish |

> **ID contract.** The feeder uses its own integer IDs internally (existing schema). Every push payload carries `sporty_match_id`, `sporty_player_id`, and `sporty_team_id` as string UUIDs supplied when the match is scheduled. A new mapping table `entity_links` (`feeder_entity`, `feeder_id`, `sporty_uuid`) stores the correspondence. The Sporty backend never sees feeder integer IDs.

---

## 5. Phase 2 — Hardening and features (build first)

Phase 2 resolves the codebase debt and builds the feature layer. No ML yet. Each requirement has an ID for tracking; acceptance criteria are testable.

| ID | Requirement | Acceptance criteria |
|---|---|---|
| R-2.1 | Single `Settings` class (pydantic-settings) reading `.env`: `DATABASE_URL`, `SPORTY_BACKEND_URL`, `FEEDER_SECRET`, `SIMULATION_SPEED` | No hardcoded Neon URL remains anywhere; grep for `neon` returns only `.env.example` |
| R-2.2 | Delete dead config: `feeder.db`, `REDIS_URL`, root `database.py` legacy module | Root `database.py` removed or renamed `legacy_database.py.bak`; app boots |
| R-2.3 | `entity_links` table + CRUD: `(feeder_entity TEXT, feeder_id INT, sporty_uuid TEXT, UNIQUE(feeder_entity, feeder_id))` | `POST /links` upserts a mapping; lookup helper `get_sporty_uuid(entity, id)` exists |
| R-2.4 | Centralise scoring rules: new `app/services/scoring_rules.py` exposing `score_events(events, sport_type)` — imported by BOTH `matches.py:_score_match` and `simulation.py` | The duplicated scoring logic is deleted from both call sites; one source of truth; unit test proves football and basketball scoring |
| R-2.5 | Sport resolver: `app/services/sport_resolver.py` with `resolve_sport_type(name) -> SportType` enum (`FOOTBALL`, `BASKETBALL`, `CRICKET`, `UNKNOWN`) using the existing alias maps | All `"basket" in name.lower()` substring checks replaced by the resolver; adding cricket later touches only this file |
| R-2.6 | Seed script `scripts/seed_sports.py` inserting football + basketball Sport rows idempotently | Fresh DB + seed + import works end to end |
| R-2.7 | Alembic becomes source of truth: remove `create_all` from `main.py` startup; autogenerate migration for `entity_links` + Phase 3 tables | `alembic upgrade head` on empty DB produces full schema; app starts without `create_all` |
| R-2.8 | Stats importer: extend `importer.py` to ingest per-gameweek stat rows from Kaggle CSVs into a new `player_stats` table (`player_id`, `gameweek`, `season`, `minutes`, `goals`, `assists`, `yellows`, `reds`, `points` — football; `minutes`, `pts`, `ast`, `reb`, `stl`, `blk` — basketball, nullable columns) | Importing the EPL CSV produces stat rows; importing NBA CSV produces stat rows; re-running is idempotent (dedupe on `player_id`+`gameweek`+`season`) |
| R-2.9 | pytest smoke suite: test app boots, test CRUD on each resource, test `scoring_rules` for both sports, test `sport_resolver` aliases | `pytest` passes locally; documented in README |
| R-2.10 | Auth middleware: validate `X-Feeder-Secret` header on every route except `/health`, `/docs`, `/openapi.json` | Requests without the header get 401; `/health` stays open |

**Stage B gate:** pytest green; `alembic upgrade head` builds full schema on empty DB; no hardcoded URLs.

---

## 6. Phase 3 — Features and ML (offline)

| ID | Requirement | Acceptance criteria |
|---|---|---|
| R-3.1 | `app/services/features.py`: `compute_player_features(player_id, db)` returning `goals_per90`, `assists_per90`, `cards_per90`, `form_index` (EWMA α=0.4 over last 5–20 stat rows, newest first), `minutes_ratio`, and per-minute event probabilities. Cold-start fallback to league averages — football: goal 0.003/min, assist 0.003, yellow 0.002, red 0.0002; basketball: point_2 0.04, point_3 0.015, free_throw 0.02, assist 0.05, rebound 0.07 | A player with zero stat rows returns fallback dict, never raises; a player with stats returns rates derived from `player_stats` table |
| R-3.2 | `compute_team_strength(team_id, db) -> float` (0..1): mean form_index of team players, normalised /15 and clamped; 0.5 if team empty | Unit tested with synthetic data |
| R-3.3 | `scripts/train_models.py` — builds `event_rates.pkl` (dict keyed by player_id) from `player_stats`; trains outcome model on finished matches (label 0=away, 1=draw, 2=home) with features `[home_strength, away_strength, home_bias]`; `Pipeline(MinMaxScaler, LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=500))` so the scaler is serialised WITH the model — **never a separate scaler.pkl** | Running the script with <5 finished matches skips outcome training with a warning; ≥20 matches does 80/20 stratified split and prints `classification_report`; pkl files land in `models_pkl/` |
| R-3.4 | `app/services/ml_models.py`: load/save helpers; `predict_outcome` uses `model.classes_` to map probabilities (never assume class order); `MODEL_VERSION` constant in every prediction payload | Missing pkl at startup logs a warning and the app still boots (heuristic fallback for `/predict`) |
| R-3.5 | `app/services/rater.py`: rule-based rating. Base 6.0. Football weights: goal +2.0, assist +1.2, yellow_card −0.5, red_card −2.5. Basketball: point_2 +0.8, point_3 +1.2, free_throw +0.3, assist +0.7, rebound +0.4, steal +0.6, block +0.5. Clamp 1.0–10.0. `find_man_of_match` = argmax of ratings | Unit tests cover clamping and MOTM tie behaviour |

**Stage C gate:** `train_models.py` produces both pkl files; `/health` reports models loaded.

---

## 7. Phase 4 — Stat-weighted simulation + push

### 7.1 Simulation rewrite (R-4.1)

Rewrite `app/services/simulation.py`. The simulation runs as an **asyncio background task** — `POST /simulate` returns 202 immediately with a job handle. The loop:

1. Load lineups (existing Player rows for both teams; max 11 football, 10 basketball).
2. Load per-player rates from `event_rates.pkl` in `app.state`; cold-start fallback per R-3.1.
3. For each minute 1..N (90 football, 48 basketball): for each active player, Bernoulli-sample each event type (`numpy.random.binomial(1, p)`). Fired events get a UUID `event_id` (idempotency key), are written to the local `events` table, and appended to the minute batch.
4. Score is updated via the centralised `scoring_rules` module (R-2.4) — football: goal = +1; basketball: point_2 = +2, point_3 = +3, free_throw = +1.
5. The minute batch is pushed to the Sporty backend in ONE HTTP call (never per-event).
6. `await asyncio.sleep(SIMULATION_SPEED)` — 0.5 default (≈45s full football match), 0 = max speed for tests.
7. On finish: final push with `status=finished`, then ratings computed (R-3.5) and pushed.

| ID | Requirement | Acceptance criteria |
|---|---|---|
| R-4.1 | Background stat-weighted simulation as described above; stop endpoint sets a flag checked each minute; concurrent simulations of different matches must not interfere (match-scoped state dict) | `POST /simulate` returns 202 in <200ms; two parallel simulations complete with independent scores; GET status reflects live minute |
| R-4.2 | `backend_client.py`: httpx async; `X-Feeder-Secret` header; 3 retries, backoff 1.5^n seconds; on final failure log ERROR and continue simulation (events persist locally for replay) | Backend down → simulation completes anyway; WARN/ERROR logged; no crash |
| R-4.3 | Replay endpoint `POST /matches/{id}/replay-push` re-sends all stored events for a match to the backend (recovery after outage) | After simulated outage, replay delivers all events; backend dedupes on `event_id` |
| R-4.4 | Prediction endpoint `POST /predict`: computes strengths, calls model, stores `match_predictions` row, pushes to backend `/feed/prediction` | Returns 3 probabilities summing to ~1.0; heuristic fallback when model missing |

### 7.2 Push contract — payloads (R-4.5)

**`POST {SPORTY_BACKEND_URL}/api/v1/feed/match-result`** — once per minute tick:

```json
{
  "sporty_match_id": "uuid",
  "sport": "football",
  "status": "live",
  "home_score": 2,
  "away_score": 1,
  "current_minute": 67,
  "events": [
    {
      "event_id": "uuid",
      "event_type": "goal",
      "sporty_player_id": "uuid",
      "sporty_team_id": "uuid",
      "minute": 67
    }
  ]
}
```

**`POST /api/v1/feed/prediction`** — after a match is scheduled:

```json
{
  "sporty_match_id": "uuid",
  "home_win_prob": 0.62,
  "draw_prob": 0.21,
  "away_win_prob": 0.17,
  "model_version": "outcome_v1_logistic"
}
```

**`POST /api/v1/feed/player-ratings`** — once when the match finishes:

```json
{
  "sporty_match_id": "uuid",
  "sport": "football",
  "man_of_match_sporty_player_id": "uuid",
  "ratings": [
    {
      "sporty_player_id": "uuid",
      "rating": 8.4,
      "goals": 2,
      "assists": 1,
      "minutes_played": 90,
      "events": ["goal", "goal", "assist"]
    }
  ]
}
```

> **Idempotency rule.** Every event carries `event_id` (UUID, generated once at sample time, stored in the feeder `events` table). The Sporty backend MUST upsert `ON CONFLICT (event_id) DO NOTHING`. Retries and replays are therefore always safe.

**Stage D gate:** two concurrent simulations complete; backend-down test passes (no crash, replay works).

---

## 8. Phase 5 — Sporty backend integration (other repo)

Built in the **Sporty_Backend repository**, not the feeder. Listed here because the feeder is useless without it. One router file: `app/api/v1/feed.py`.

| ID | Requirement | Notes |
|---|---|---|
| R-5.1 | `X-Feeder-Secret` header validation dependency; `FEEDER_SECRET` added to Sporty Settings + `.env` | 401 on mismatch; secret never logged |
| R-5.2 | `POST /api/v1/feed/match-result`: upsert events idempotently, update match score/status, publish JSON to Redis channel `live:match:{sporty_match_id}` | Sporty's existing WebSocket server already subscribes to `live:match:*` — no WS changes needed |
| R-5.3 | `POST /api/v1/feed/prediction`: `SETEX prediction:match:{id} 86400` | Frontend reads via existing REST |
| R-5.4 | `POST /api/v1/feed/player-ratings`: `SETEX ratings:match:{id} 86400`; log MOTM | Optionally map ratings into PlayerGameweekStat later |
| R-5.5 | On `status=finished`: trigger Sporty's scoring service for the affected gameweek immediately (do not wait for the daily cron) | Call the existing scoring engine service directly |

**Stage E gate:** end-to-end — simulate → backend receives → Redis publish → frontend WebSocket shows live score.

---

## 9. Feeder API surface (target, after Phase 4)

| Method | Route | Auth | Description |
|---|---|---|---|
| GET | `/health` | none | DB + models loaded + running sim count |
| GET/POST/DELETE | `/sports`, `/teams`, `/players` | secret | Existing CRUD (unchanged) |
| GET/POST | `/matches`, `/matches/{id}` | secret | Existing; score derived via `scoring_rules` |
| GET/POST | `/matches/{id}/events` | secret | Existing event log |
| POST | `/simulate` | secret | 202; body: feeder match id OR team ids + sporty UUIDs; starts background task |
| GET | `/simulate/{id}/status` | secret | Live minute, score, status |
| POST | `/simulate/{id}/stop` | secret | Graceful stop after current minute |
| POST | `/matches/{id}/replay-push` | secret | Re-push stored events (recovery) |
| POST | `/predict` | secret | Outcome probabilities; pushes to backend |
| POST | `/import/nba`, `/import/premier-league` | secret | Existing importers + new stats ingestion |

---

## 10. Configuration (.env)

```env
# Feeder's OWN database (existing Neon DB) — NOT the Sporty DB
DATABASE_URL=postgresql://...

# Where to push live data
SPORTY_BACKEND_URL=https://your-sporty-backend.example.com

# Shared secret — same value in Sporty backend .env
FEEDER_SECRET=32_char_random_string

# Real seconds per simulated minute. 0.5 = ~45s full match. 0 = max speed (tests).
SIMULATION_SPEED=0.5
```

> **Explicit decision: no new database, no new Redis.** The feeder already has its own Postgres (Neon) — keep it. Redis is NOT used by the feeder at all; the Sporty backend owns Redis and publishes to `live:match:{id}` when it receives pushes. The `REDIS_URL` leftover in the feeder's `.env.example` must be deleted (R-2.2).

---

## 11. Build order and gates

| Stage | Work | Gate to proceed |
|---|---|---|
| A (done) | FastAPI scaffold, CRUD, importers, random simulation | Complete |
| B | Phase 2 hardening: R-2.1 → R-2.10 | pytest green; `alembic upgrade head` builds full schema on empty DB; no hardcoded URLs |
| C | Phase 3 features + ML: R-3.1 → R-3.5 | `train_models.py` produces both pkl files; `/health` reports models loaded |
| D | Phase 4 simulation + push: R-4.1 → R-4.5 | Two concurrent simulations complete; backend-down test passes (no crash, replay works) |
| E | Phase 5 Sporty integration: R-5.1 → R-5.5 | End-to-end: simulate → backend receives → Redis publish → frontend WebSocket shows live score |
| F | Polish: model retraining docs, README, Docker deployment | Fresh clone to running simulation in under 15 minutes following README |

---

## 12. Error handling matrix

| Scenario | Handling | Recovery |
|---|---|---|
| Sporty backend unreachable on push | 3 retries with backoff; then log ERROR, continue simulation | `POST /matches/{id}/replay-push` after outage; backend dedupes on `event_id` |
| pkl model missing at startup | WARN; app boots; `/predict` uses heuristic; simulator uses fallback rates | Run `train_models.py`, restart |
| Player has no stats (cold start) | League-average per-minute rates (R-3.1) | Rates improve automatically after stats import + retrain |
| Simulation crashes mid-match | Status set to error; partial events retained locally | Re-trigger `/simulate`; backend upsert prevents duplicates |
| Duplicate `/simulate` for running match | 409 Conflict | Wait or stop the running simulation first |
| Unknown sport name | `sport_resolver` returns UNKNOWN; 422 with clear message | Seed sports table; use known aliases |

---

## 13. Observability

Python `logging` module only (never `print`).

- **DEBUG**: per-minute simulation progress
- **INFO**: simulation start/finish, push confirmations, import summaries
- **WARNING**: retries, cold-start fallbacks, model missing
- **ERROR**: exhausted retries, DB exceptions

Key metrics to watch manually until a metrics endpoint exists: simulation wall-time per match (< 60s at speed 0.5), push success rate (> 99%), unmatched player rate on CSV import (< 20%).

---

## 14. Glossary

| Term | Definition |
|---|---|
| `event_rates.pkl` | Dict `{player_id: {event_type_prob_per_min}}` computed from `player_stats` history |
| `outcome_model.pkl` | sklearn Pipeline (MinMaxScaler + LogisticRegression) for home/draw/away probabilities |
| `form_index` | EWMA (α=0.4) of a player's recent fantasy/match points, newest weighted highest |
| `entity_links` | Feeder table mapping feeder integer IDs to Sporty UUIDs |
| `event_id` | UUID generated at sample time; the idempotency key for all pushes and replays |
| push | Outbound authenticated HTTP POST from feeder to Sporty `/api/v1/feed/*` |
| `SIMULATION_SPEED` | Real seconds per simulated minute (0.5 default; 0 for tests) |
| Cold start | Player with no stat history — simulator uses league-average fallback rates |

---

*Sporty Data Feeder — PRD v2.0 — June 2026 — Internal engineering document. Written to be used as the authoritative build plan for Phases 2–5 in Claude Code.*