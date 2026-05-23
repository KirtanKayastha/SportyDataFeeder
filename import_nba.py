import os
import sys

import pandas as pd
from sqlalchemy.exc import SQLAlchemyError

from database import Session, Sport, Team, Player

CSV_FILE = "nba_player_stats_2026.csv"
SPORT_NAME = "basketball"
BATCH_SIZE = 500
PROGRESS_EVERY = 100


def clean_value(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def main():
    if not os.path.exists(CSV_FILE):
        print(f"ERROR: CSV file not found: {CSV_FILE}")
        print(f"Current directory: {os.getcwd()}")
        return 1

    try:
        df = pd.read_csv(CSV_FILE)
    except Exception as exc:
        print(f"ERROR: Failed to read CSV: {exc}")
        return 1

    required_columns = {"TEAM", "PLAYER"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        print(f"ERROR: CSV is missing required columns: {sorted(missing_columns)}")
        return 1

    session = Session()
    teams_added = 0
    players_added = 0
    skipped_rows = 0
    duplicate_players = 0
    pending_changes = 0

    try:
        sport = session.query(Sport).filter_by(name=SPORT_NAME).first()
        if not sport:
            print(f"ERROR: Sport '{SPORT_NAME}' not found in sports table.")
            return 1

        team_cache = {
            name: team_id
            for name, team_id in session.query(Team.name, Team.id).filter_by(sport_id=sport.id).all()
        }

        existing_players = set(
            session.query(Player.name, Player.team_id).filter_by(sport_id=sport.id).all()
        )

        total_rows = len(df)
        print(f"Starting NBA import from {CSV_FILE}")
        print(f"Total rows in CSV: {total_rows}")
        print(f"Existing teams: {len(team_cache)}, existing players: {len(existing_players)}")

        for idx, row in enumerate(df.itertuples(index=False), start=1):
            team_name = clean_value(getattr(row, "TEAM", None))
            player_name = clean_value(getattr(row, "PLAYER", None))

            if not team_name or not player_name:
                skipped_rows += 1
                continue

            team_id = team_cache.get(team_name)
            if team_id is None:
                team = Team(name=team_name, sport_id=sport.id)
                session.add(team)
                session.flush()
                team_id = team.id
                team_cache[team_name] = team_id
                teams_added += 1
                pending_changes += 1

            player_key = (player_name, team_id)
            if player_key in existing_players:
                duplicate_players += 1
            else:
                player = Player(
                    name=player_name,
                    team_id=team_id,
                    position="Unknown",
                    sport_id=sport.id,
                )
                session.add(player)
                existing_players.add(player_key)
                players_added += 1
                pending_changes += 1

            if pending_changes >= BATCH_SIZE:
                session.commit()
                pending_changes = 0

            if idx % PROGRESS_EVERY == 0:
                print(
                    f"Processed {idx}/{total_rows} rows | "
                    f"Teams added: {teams_added} | Players added: {players_added}"
                )

        if pending_changes > 0:
            session.commit()

        print("Import complete.")
        print(f"Teams added: {teams_added}")
        print(f"Players added: {players_added}")
        print(f"Duplicate players skipped: {duplicate_players}")
        print(f"Rows skipped (missing TEAM/PLAYER): {skipped_rows}")
        return 0

    except SQLAlchemyError as exc:
        session.rollback()
        print(f"Database error during import: {exc}")
        return 1
    except Exception as exc:
        session.rollback()
        print(f"Unexpected error during import: {exc}")
        return 1
    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())
