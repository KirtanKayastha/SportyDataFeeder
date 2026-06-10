# /home/sam069/projects/SportyDataFeeder/app/routers/imports.py

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status

from app.database import get_db
from app.schemas import ImportNBARequest, ImportPremierLeagueRequest
from app.services.importer import import_nba, import_premier_league

router = APIRouter()


@router.post("/imports/nba")
def run_nba_import(payload: ImportNBARequest, db=Depends(get_db)):
    try:
        result = import_nba(db, Path(payload.csv_path), gameweek=payload.gameweek, season=payload.season)
        return {"status": "ok", "result": result}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/imports/premier-league")
def run_premier_league_import(payload: ImportPremierLeagueRequest, db=Depends(get_db)):
    try:
        result = import_premier_league(
            db,
            [Path(csv_path) for csv_path in payload.csv_paths],
            gameweek=payload.gameweek,
            season=payload.season,
        )
        return {"status": "ok", "result": result}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc