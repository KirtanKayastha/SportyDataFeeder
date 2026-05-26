# /home/sam069/projects/SportyDataFeeder/app/main.py

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, status
from sqlalchemy import text

from app.database import Base, engine
from app.routers import events, imports, matches, players, simulation, sports, teams

Base.metadata.create_all(bind=engine)  # remove in Phase 2 when Alembic takes over

app = FastAPI(title="Sporty Data Feeder API", version="1.0.0")

app.include_router(sports.router, prefix="/sports", tags=["Sports"])
app.include_router(teams.router, prefix="/teams", tags=["Teams"])
app.include_router(players.router, prefix="/players", tags=["Players"])
app.include_router(matches.router, prefix="/matches", tags=["Matches"])
app.include_router(events.router, tags=["Events"])
app.include_router(simulation.router, tags=["Simulation"])
app.include_router(imports.router, tags=["Imports"])


@app.get("/health")
def health():
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return {"status": "ok", "database": "up"}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database connection failed: {exc}",
        ) from exc
