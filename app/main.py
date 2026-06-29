# /home/sam069/projects/SportyDataFeeder/app/main.py

import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.config import get_settings
from app.database import engine
from app.routers import demo, events, imports, links, matches, players, predict, simulation, sports, teams
from app.services.ml_models import load_all_models, load_outcome_v2, load_outcome_v2_basketball
from app.services.simulation import running_count

# Schema is managed by Alembic: run `alembic upgrade head` before starting the app.


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Missing pkl files only WARN (inside load_all_models); the app boots
    # regardless and /predict falls back to heuristics (PRD R-3.4).
    loaded = load_all_models()
    app.state.outcome_model = loaded["outcome_model"]
    app.state.event_rates = loaded["event_rates"]
    # outcome_v2: real-data Elo models used by /predict, per sport (football +
    # basketball). Fall back to outcome_model/heuristic when a sport's bundle is
    # absent.
    app.state.outcome_v2 = load_outcome_v2()
    app.state.outcome_v2_basketball = load_outcome_v2_basketball()
    yield


app = FastAPI(title="Sporty Data Feeder API", version="1.0.0", lifespan=lifespan)

# R-2.10: every route requires the shared X-Feeder-Secret header except the
# open endpoints below (health probes and interactive docs stay reachable).
_AUTH_EXEMPT_PATHS = {"/health", "/docs", "/openapi.json"}


@app.middleware("http")
async def require_feeder_secret(request: Request, call_next):
    if request.url.path not in _AUTH_EXEMPT_PATHS:
        provided = request.headers.get("X-Feeder-Secret", "")
        if not secrets.compare_digest(provided, get_settings().FEEDER_SECRET):
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Missing or invalid X-Feeder-Secret header"},
            )
    return await call_next(request)


app.include_router(sports.router, prefix="/sports", tags=["Sports"])
app.include_router(teams.router, prefix="/teams", tags=["Teams"])
app.include_router(players.router, prefix="/players", tags=["Players"])
app.include_router(matches.router, prefix="/matches", tags=["Matches"])
app.include_router(events.router, tags=["Events"])
app.include_router(simulation.router, tags=["Simulation"])
app.include_router(imports.router, tags=["Imports"])
app.include_router(links.router, tags=["Entity Links"])
app.include_router(predict.router, tags=["Prediction"])
app.include_router(demo.router, tags=["Demo"])


@app.get("/health")
def health(request: Request):
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database connection failed: {exc}",
        ) from exc

    event_rates = getattr(request.app.state, "event_rates", None)
    return {
        "status": "ok",
        "database": "up",
        "simulations_running": running_count(),
        "models": {
            "outcome_model": getattr(request.app.state, "outcome_model", None) is not None,
            "outcome_v2": getattr(request.app.state, "outcome_v2", None) is not None,
            "outcome_v2_basketball": getattr(request.app.state, "outcome_v2_basketball", None) is not None,
            "event_rates": event_rates is not None,
            "event_rates_players": len(event_rates) if event_rates else 0,
        },
    }
