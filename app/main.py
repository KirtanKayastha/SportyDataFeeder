# /home/sam069/projects/SportyDataFeeder/app/main.py

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from sqlalchemy import text

from app.database import engine
from app.routers import events, imports, links, matches, players, predict, simulation, sports, teams
from app.services.ml_models import load_all_models
from app.services.simulation import running_count

# Schema is managed by Alembic: run `alembic upgrade head` before starting the app.


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Missing pkl files only WARN (inside load_all_models); the app boots
    # regardless and /predict falls back to heuristics (PRD R-3.4).
    loaded = load_all_models()
    app.state.outcome_model = loaded["outcome_model"]
    app.state.event_rates = loaded["event_rates"]
    yield


app = FastAPI(title="Sporty Data Feeder API", version="1.0.0", lifespan=lifespan)

app.include_router(sports.router, prefix="/sports", tags=["Sports"])
app.include_router(teams.router, prefix="/teams", tags=["Teams"])
app.include_router(players.router, prefix="/players", tags=["Players"])
app.include_router(matches.router, prefix="/matches", tags=["Matches"])
app.include_router(events.router, tags=["Events"])
app.include_router(simulation.router, tags=["Simulation"])
app.include_router(imports.router, tags=["Imports"])
app.include_router(links.router, tags=["Entity Links"])
app.include_router(predict.router, tags=["Prediction"])


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
            "event_rates": event_rates is not None,
            "event_rates_players": len(event_rates) if event_rates else 0,
        },
    }
