# /home/sam069/projects/SportyDataFeeder/app/main.py

import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import text

from app.config import get_settings
from app.database import engine
from app.routers import demo, events, imports, links, matches, players, predict, simulation, sports, teams
from app.services.backend_client import get_backend_client
from app.services.ml_models import load_all_models, load_outcome_v2, load_outcome_v2_basketball
from app.services.simulation import running_count

# Schema is managed by Alembic: run `alembic upgrade head` before starting the app.


def _load_models_into_state(app: FastAPI) -> None:
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models_into_state(app)
    yield
    await get_backend_client().aclose()


app = FastAPI(title="Sporty Data Feeder API", version="1.0.0", lifespan=lifespan)

# R-2.10: every route requires the shared X-Feeder-Secret header except the
# open endpoints below (health probes, interactive docs, and the admin control
# panel HTML — the panel's JS still sends the secret on every API call, so only
# the static page itself is exempt, not the data routes).
_AUTH_EXEMPT_PATHS = {"/health", "/docs", "/openapi.json", "/admin", "/admin/"}

_ADMIN_HTML = Path(__file__).resolve().parent / "static" / "admin.html"


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


@app.get("/admin", include_in_schema=False)
def admin_panel():
    """Serve the static feeder control panel (same-origin, so no CORS and the
    secret is entered in the UI, never baked into a build)."""
    if not _ADMIN_HTML.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Admin panel not built")
    return FileResponse(_ADMIN_HTML, media_type="text/html")


@app.post("/models/reload")
def reload_models(request: Request):
    """Re-read every pkl from models_pkl/ into app.state without a restart, so
    scripts/refresh_bundles.py (cron) can rebuild bundles and hot-swap them
    (step 4 of reports/MODEL_IMPROVEMENT_PLAN.md). Auth middleware applies."""
    _load_models_into_state(request.app)
    return {
        "reloaded": True,
        "models": _models_status(request),
    }


def _models_status(request: Request) -> dict:
    event_rates = getattr(request.app.state, "event_rates", None)
    return {
        "outcome_model": getattr(request.app.state, "outcome_model", None) is not None,
        "outcome_v2": getattr(request.app.state, "outcome_v2", None) is not None,
        "outcome_v2_basketball": getattr(request.app.state, "outcome_v2_basketball", None) is not None,
        "event_rates": event_rates is not None,
        "event_rates_players": len(event_rates) if event_rates else 0,
    }


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

    return {
        "status": "ok",
        "database": "up",
        "simulations_running": running_count(),
        "models": _models_status(request),
    }
