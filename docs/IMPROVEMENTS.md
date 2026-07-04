# Improvement Suggestions

Every item below is anchored to a specific observation from reading the code (cross-
referenced to the relevant doc/file), not a generic best-practice checklist. Items the
codebase's own `reports/MODEL_IMPROVEMENT_PLAN.md` already identifies are marked
**(known, in `reports/`)** so this doesn't duplicate work already tracked in-repo.

## Architecture

1. **Simulation state has no persistence.** `app/services/simulation._simulations` is a
   plain in-memory dict (`ARCHITECTURE.md` §"What this system deliberately does NOT
   have," `SIMULATION.md` §1/§11). A process restart mid-simulation silently loses all
   progress and status — the match is left `status="live"` in the database forever with
   no running task to finish it. **Fix direction:** persist minimal simulation state
   (current minute, scores, status) to the database on each minute tick, and add a
   startup reconciliation pass that marks any match still `"live"` from a previous
   process lifetime as `"error"` (or resumes it, if that's ever desired).
2. **`_not_found(entity, id)` is duplicated verbatim across six router files**
   (`sports.py`, `teams.py`, `players.py`, `matches.py`, `events.py`, `simulation.py`) —
   confirmed identical implementations, not imported from a shared module. Low risk, but
   a one-line fix that would prevent future drift (e.g. if the 404 message format ever
   needs to change, six places need editing in lockstep).
3. **Multi-replica deployment is unsafe as currently built** (`CONFIGURATION_DEPLOYMENT.md`
   §Scaling): the in-memory simulation registry and per-process model state mean a
   second replica has no visibility into the first's running simulations or freshly
   reloaded models. If horizontal scaling is ever needed, the registry would need to move
   to a shared store (even something as simple as a `simulation_state` DB table would
   work, given there's already no Redis in this system).

## Performance

4. **No index on `events.match_id`, `events.event_type`, or `events.minute`** beyond
   whatever the FK creates implicitly (`DATABASE.md` §5). Every score computation
   (`_score_match`), event listing, and replay-push does a full ordered scan. At current
   demo-scale data volumes this is irrelevant; if `events` ever grows to real production
   volume, `CREATE INDEX ix_events_match_id ON events(match_id)` (and possibly a
   composite `(match_id, minute)`) would materially speed up every score read.
5. **`features.py:compute_player_features`'s `minutes_ratio` is computed but never
   consumed** anywhere downstream (`MODELS.md` §5 — verified: `simulation.py:_prepare`
   only reads `event_rates` off the features dict). Either wire it into calibration (a
   player who typically plays 60 minutes shouldn't necessarily fire events at a rate
   scaled for 90) or remove the dead computation.
6. **`compute_team_strength` re-runs `compute_player_features` per player, per call, with
   no caching** (`app/services/features.py:155-163`) — for a team of N players this is N
   separate multi-row `player_stats` queries every time `/predict`'s v1 fallback path or
   `scripts/train_models.py`'s training loop runs. Since the v2 Elo path is the primary
   `/predict` path now and doesn't touch this function, the impact is currently limited
   to the fallback case and offline training — but if the fallback path ever became
   primary again, a per-request in-memory cache (or bulk query) would help.

## Security

7. **`FEEDER_SECRET` defaults to the literal string `"change-me"`** (`app/config.py:15`)
   if unset. There's no startup check that refuses to boot (or at least loudly warns) if
   this default is still in effect — a misconfigured deployment could run indefinitely
   with a publicly-guessable secret. **Fix direction:** log a `WARNING` (or refuse to
   start outside a debug flag) at boot when `FEEDER_SECRET == "change-me"`.
8. **No rate limiting anywhere** (`CONFIGURATION_DEPLOYMENT.md` §Security). A leaked or
   brute-forced secret allows unlimited-rate calls to `POST /simulate`, which spawns
   background `asyncio.Task`s — a straightforward resource-exhaustion vector if this API
   is ever reachable beyond a fully trusted network.
9. **`GET /health`'s 503 response leaks the raw database exception string** to an
   unauthenticated caller (`CONFIGURATION_DEPLOYMENT.md` §Health checks). Consider
   returning a generic message publicly and logging the detailed exception server-side
   only.
10. **App-level uniqueness checks (team name per sport, player name per team) are
    check-then-insert, not enforced by a database constraint** (`DATABASE.md` §5). Two
    concurrent identical `POST` requests could race past the `SELECT` check and both
    insert, producing duplicate rows the `IntegrityError` handler wouldn't catch (there's
    no DB-level unique constraint to violate). Adding the matching `UniqueConstraint`s
    would close this race outright.

## Models

11. **(known, in `reports/`)** Basketball has no shots/rest-equivalent second feature in
    the shipped bundle — `elo_rest`/`elo_mov_rest` beat the incumbent in backtest but
    weren't shipped because "they need a real schedule feed at predict time." A concrete
    path to actually shipping this: since `matches.match_date` already exists in the
    feeder's own schema, a live `rest_diff` feature could be computed from the feeder's
    own match history at predict time rather than needing an external schedule feed —
    worth revisiting now that the infrastructure question ("where does rest data come
    from live") has an obvious in-repo answer.
12. **Elo ratings and SoT/form are frozen at bundle-build time** with no online update
    between `scripts/refresh_bundles.py` runs (`MODELS.md` §1 Weaknesses). A match
    simulated and finished by this very feeder never feeds back into the next
    `/predict` call for a rematch, even though the feeder already has that result in its
    own database. An incremental "nudge the bundle's Elo ratings after each finished
    feeder match" step (using `EloModel.update` directly against the loaded bundle) would
    close this gap cheaply, without waiting for the next full offline retrain.
13. **`BASKETBALL_POINT_MIX` (60/25/15% two/three/free-throw split) is a single global
    constant applied uniformly to every player** (`app/services/features.py:41`,
    `MODELS.md` §5) — a 3-point specialist and a paint-only center get the identical
    scoring-event *mix*, differing only in total volume. If `player_stats` ever grows a
    genuine 2pt/3pt/FT split column (many public NBA datasets have this), replacing the
    fixed mix with a per-player one would materially improve simulation realism for
    perimeter-heavy vs. interior-heavy players.
14. **No confidence interval / uncertainty quantification anywhere** — every prediction
    (`PredictResponse`) is a bare point-probability with no indication of how much data
    backed it (`home_known`/`away_known` flags exist in `predict_outcome_v2`'s return dict
    but are **not surfaced in the actual `PredictResponse` schema** — verified by reading
    `app/schemas.py:PredictResponse`, which has no such field). Surfacing "this team's Elo
    rating is a cold-start default" to the caller would materially improve trust in edge
    cases (newly promoted teams, first games of a season).

## Simulation

15. **No RNG seeding** (`SIMULATION.md` §12) means simulation runs cannot be replayed
    deterministically for debugging a specific reported bug ("match 42 produced an
    impossible event sequence" is currently unreproducible). A `random_seed` parameter on
    `SimulateStartRequest`, threaded down to a per-simulation `numpy.random.Generator`
    instance (rather than the shared global RNG state), would make this debuggable without
    sacrificing the "no two runs look identical" property in normal operation.
16. **Overtime periods are capped at 6 "for safety"** (`MAX_OVERTIME_PERIODS`,
    `SIMULATION.md` §8) but the code never surfaces to the caller that this cap was hit —
    a match tied after 6 OT periods simply ends `"finished"` still tied (basketball
    scores are always integers, so an exact tie after the cap is possible, if rare), which
    silently violates the domain assumption "basketball has no draws" that other code
    (e.g. `CLASSES = ["H","A"]` throughout the model layer) relies on.

## Code quality

17. **Two parallel `_fold`/`_fold_name` implementations** exist
    (`app/services/simulation.py:_fold_name`, `app/routers/demo.py:_fold`) — byte-for-byte
    identical logic (NFKD decomposition + the same 6-character accent map), duplicated
    rather than shared from one module. Same risk pattern as the `_not_found` duplication
    above: a future accent-handling fix (e.g. adding `ñ→n`) would need to be applied
    twice and could easily be missed in one location.
18. **`import_nba`'s per-100-rows progress hook is a dead `pass` statement**
    (`app/services/importer.py:268-269`: `if idx % PROGRESS_EVERY == 0: pass`) — either
    wire it to an actual log line (matching the intent implied by the `PROGRESS_EVERY`
    constant's name) or remove the dead branch.
19. **`app/routers/matches.py` imports `Event`, `Player`, `Sport`, `Team` at module scope
    but the router's own docstring-less functions re-import nothing** — not a bug, but
    the file mixes CRUD, Sporty-backend orchestration (`schedule_on_sporty`,
    `unschedule_on_sporty`), and lineup-preview concerns in one 327-line file; splitting
    the Sporty-orchestration endpoints into their own router (mirroring how `demo.py`
    already exists as a separate orchestration router) would keep each file focused on
    one concern as the endpoint count grows.

## Scalability

20. **File-path-based CSV import (`csv_path`/`csv_paths` fields, not a multipart upload)**
    (`API.md` §`/imports/*`) only works when the operator has filesystem access to the
    same machine/container running the API. Fine for the current operator-driven,
    single-host demo use case; would need a real upload endpoint (`UploadFile` in
    FastAPI) before this could be used by a remote/multi-tenant caller.
21. **Migrations run automatically on every container start with no distributed lock**
    (`CONFIGURATION_DEPLOYMENT.md` §Deployment) — safe today because the deployment model
    is single-replica, but would need `alembic`'s advisory-lock support (or an explicit
    migration-then-app-start split in the deploy pipeline) before scaling to multiple
    replicas starting concurrently against a fresh database.

## Testing (observed gaps, not a criticism of what exists — the suite is genuinely solid)

22. The `scripts/train_outcome_v*.py` / `scripts/backtest_*.py` research pipeline has
    **no automated tests at all** — correctness is currently verified only by eyeballing
    the generated `reports/*.md` tables. A small smoke test (e.g. "does
    `annotate_pre_match_elo` ever let a later match's result leak into an earlier row's
    features") would formalize the causal-correctness property these scripts' docstrings
    repeatedly assert by hand.
23. `tests/test_prediction_metrics.py:test_models_reload_endpoint` covers `POST
    /models/reload` only shallowly — it asserts a `200` and that the response's `models`
    dict has the right keys, but does not actually verify a real end-to-end hot-swap
    (e.g. load bundle A into `app.state`, write bundle B to `models_pkl/`, call reload,
    assert `app.state` now reflects B's ratings). Worth strengthening given this is the
    only way to update live predictions without a process restart.
