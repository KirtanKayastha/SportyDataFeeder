# Local Demo Runbook — Feeder → Backend → Frontend (live + fantasy)

End-to-end pipeline for a local demo of the Sporty multisport fantasy platform:

```
Feeder /demo/launch
  ├─ POST backend /feed/schedule-match     → creates Sporty match, returns UUID  (linked)
  ├─ POST backend /feed/register-players   → creates Sporty players, returns UUIDs (linked)
  ├─ POST backend /feed/demo-setup         → demo user + league + open window + fantasy lineup
  └─ start simulation
        └─ per sim-minute: POST /feed/match-result  (X-Feeder-Secret)
              backend: upsert events → publish "match:{id}" to Redis           → WS ticker
                       apply_live_points → publish FANTASY_POINTS_DELTA          → live per-player points
              on finish: persist_match_stats → player_gameweek_stats → scoring  → user fantasy total
```

One feeder call (`POST /demo/launch {"match_id": N}`) drives the whole thing.

## Prerequisites (run once)
- **Postgres** + **Redis** running locally.
- A **shared secret** value `S` used by both the feeder and backend (`FEEDER_SECRET`). Empty on the backend → feed endpoints return 503.

## 1. Sporty backend (`~/projects/Sporty/Sporty_Backend`)
Set `.env`: `DATABASE_URL` (local Postgres), `REDIS_URL` (local Redis), `FEEDER_SECRET=S`. Then:
```bash
alembic upgrade head
python scripts/create_sports.py            # seeds the football/basketball sport slugs
uvicorn app.main:app --port 8000           # frontend/runbook assume port 8000
```
> The live feed needs only Postgres + Redis. Kafka/Influx are off by default. Celery enqueues on finish to the Redis broker (no worker required for the ticker).

## 2. Sporty frontend (`~/projects/Sporty/sporty-frontend`)
Set `.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```
```bash
yarn dev        # http://localhost:3000
```

## 3. Feeder (this repo)
`.env`: `SPORTY_BACKEND_URL=http://localhost:8000`, `FEEDER_SECRET=S`, `SIMULATION_SPEED=1.5` (a watchable pace), `DATABASE_URL` (feeder's own DB). Make sure models + sports are loaded (`alembic upgrade head`, `python -m scripts.seed_sports`, import rosters, `python -m scripts.train_models`). Then:
```bash
uvicorn app.main:app --port 8010
```

## 4. Launch a demo match (one call)
Pick a scheduled feeder football or basketball match id, then:
```bash
curl -X POST http://localhost:8010/demo/launch \
  -H "X-Feeder-Secret: $FEEDER_SECRET" -H "Content-Type: application/json" \
  -d '{"match_id": 16, "fantasy_demo": true}'
```
Response includes `sporty_match_id` and `frontend_url` (`/match/<UUID>`), plus the demo login:
`demo@sporty.local` / `demo1234!`.

## 5. Watch it live
1. Log into the frontend as **demo@sporty.local / demo1234!**.
2. Open `http://localhost:3000/match/<sporty_match_id>`.
3. The score ticker and events stream live (Redis → WebSocket). Per-player fantasy points tick up (`FANTASY_POINTS_DELTA`). When the match finishes, the demo user's **fantasy total** updates (gameweek scoring runs on the booked stats).

## Real-league flow (your own users draft and score)

Use this when a real user has created a league and you want **their** fantasy
teams to score from simulated matches (not the throwaway demo team). The golden
rule: **a team scores only for the exact players it drafted**, so the feeder's
players must be in the backend's draftable pool *before* the draft.

1. **Open transfer window** covering *now* and the match date must exist
   (lineups can't be set without one; stats book into the window covering the
   match date). Seed one with `scripts/seed_season_and_windows.py` if needed.
2. **Prepare** (register the feeder's players + link + push prediction, but do
   NOT simulate or create a throwaway team):
   ```bash
   curl -X POST http://localhost:8010/demo/launch \
     -H "X-Feeder-Secret: $FEEDER_SECRET" -H "Content-Type: application/json" \
     -d '{"match_id": 16, "simulate": false, "fantasy_demo": false}'
   ```
   The response's `draftable_players` lists the players (name, team, UUID) now
   available to draft, and the prediction is pushed.
3. **Draft + lineup** in the frontend: your ≥2 league members draft those
   players and set lineups for the open window. Finishing the draft flips the
   league `DRAFTING → ACTIVE` (≥2 members required to start; auto-ACTIVE on the
   last pick).
4. **Simulate** the match:
   ```bash
   curl -X POST http://localhost:8010/simulate \
     -H "X-Feeder-Secret: $FEEDER_SECRET" -d '{"match_id": 16}'
   ```
5. Watch live, and at full-time the **drafted players' points** roll up into
   each fantasy team's `team_weekly_scores` (gameweek scoring).

> A team only scores for players that (a) it drafted, (b) are in its lineup for
> the window, and (c) play in a simulated match. So simulate the match(es) whose
> teams contain the drafted players.

## How fantasy points work (two layers)
- **Live per-player** — `feed_scoring.apply_live_points` increments `fantasy:match:{id}:player:{uuid}` Redis hashes and publishes `FANTASY_POINTS_DELTA`. Needs the player links the launcher creates.
- **User total** — on finish, `persist_match_stats` folds the match's `live_events` into `player_gameweek_stats`; the gameweek engine computes per-player `fantasy_points` and the demo team's `team_weekly_scores`. Needs an open transfer window (the launcher's `demo-setup` creates one) and a fantasy lineup (ditto).

## Notes / gotchas
- Re-running `/demo/launch` on the same feeder match re-simulates; use a fresh **scheduled** match for a clean run, or reset events first.
- `demo-setup` is idempotent (get-or-create) and is best-effort: if it fails, the live ticker + per-player points still work.
- Player/team alignment is guaranteed because the feeder **registers its own players** in the backend (no fragile name matching).
- `models_pkl/` is gitignored; build the Elo bundles on deploy with `scripts.finalize_outcome_v2` + `scripts.finalize_outcome_basketball`.
