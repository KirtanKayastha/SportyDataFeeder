# /home/sam069/projects/SportyDataFeeder/scripts/load_nba.py
#
# Loader for the real Kaggle NBA database (nba.sqlite). Read-only: pulls regular
# season games into ONE tidy, causally sorted DataFrame for the basketball
# outcome backtest/training harness — the basketball analogue of
# scripts/load_historical.py. Does NOT touch the application database.
#
# Elo identity is the stable franchise `team_id` (survives relocations); the
# current abbreviation (what the app uses) is attached for prediction-time
# lookup. Basketball has no draws, so labels are 'H' | 'A' only.
#
# Usage:  python -m scripts.load_nba           # prints a summary
#         from scripts.load_nba import load_nba_games

import sqlite3
import sys
from pathlib import Path

import pandas as pd

NBA_DB = Path(__file__).resolve().parents[1] / "nba.sqlite"
# Start after the messy 2002-2004 relocations (Charlotte/New Orleans team_id
# reuse) so franchise identity is clean; still ~19 seasons of data.
SINCE = "2004-10-01"


def _season_label(season_id: str) -> str:
    year = int(str(season_id)[1:])
    return f"{year}-{str(year + 1)[-2:]}"


def load_nba_games(db_path: Path | None = None, since: str = SINCE) -> pd.DataFrame:
    """Return real NBA regular-season games as one causally sorted DataFrame.

    Columns: date, season, home/away (str team_id, the Elo key), home_abbr,
    away_abbr, fthg/ftag (home/away points), ftr ('H'|'A').
    """
    db_path = Path(db_path) if db_path else NBA_DB
    con = sqlite3.connect(db_path)
    g = pd.read_sql(
        "SELECT season_id, game_date, team_id_home, team_id_away, "
        "team_abbreviation_home, team_abbreviation_away, wl_home, pts_home, pts_away "
        "FROM game",
        con,
    )
    teams = pd.read_sql("SELECT id, abbreviation FROM team", con)
    con.close()

    current_ids = set(teams["id"])
    abbr_by_id = dict(zip(teams["id"], teams["abbreviation"]))

    g["date"] = pd.to_datetime(g["game_date"], errors="coerce")
    g = g[g["season_id"].astype(str).str.startswith("2")]          # regular season
    g = g.dropna(subset=["date", "wl_home", "pts_home", "pts_away"])
    g = g[g["date"] >= pd.Timestamp(since)]
    g = g[g["team_id_home"].isin(current_ids) & g["team_id_away"].isin(current_ids)]

    out = pd.DataFrame({
        "date": g["date"],
        "season": g["season_id"].map(_season_label),
        "home": g["team_id_home"].astype(str),
        "away": g["team_id_away"].astype(str),
        # current abbreviation for the franchise (prediction-time key)
        "home_abbr": g["team_id_home"].map(abbr_by_id),
        "away_abbr": g["team_id_away"].map(abbr_by_id),
        "fthg": g["pts_home"].astype(int),
        "ftag": g["pts_away"].astype(int),
        "ftr": g["wl_home"].map({"W": "H", "L": "A"}),
    })
    out = out.sort_values(["date", "home"], kind="mergesort").reset_index(drop=True)
    return out


def main() -> None:
    df = load_nba_games()
    seasons = list(dict.fromkeys(df["season"]))
    print(f"Loaded {len(df)} real NBA regular-season games across {len(seasons)} seasons")
    print(f"Seasons: {seasons}")
    print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"Home win rate: {(df['ftr'] == 'H').mean():.3f}")
    print(f"Avg points  home={df['fthg'].mean():.1f}  away={df['ftag'].mean():.1f}  total={(df['fthg']+df['ftag']).mean():.1f}")
    print(f"Distinct franchises: {df['home_abbr'].nunique()}")
    print(f"Games/season (head):\n{df.groupby('season').size().head().to_string()}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
