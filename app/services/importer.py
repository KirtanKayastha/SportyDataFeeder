# /home/sam069/projects/SportyDataFeeder/app/services/importer.py

import csv
import io
from pathlib import Path
from typing import Any, Iterable

from app.database import Player, Sport, Team

BATCH_SIZE = 500
PROGRESS_EVERY = 100


def normalize_text(value: Any, default: str | None = None) -> str | None:
    if value is None:
        return default
    cleaned = " ".join(str(value).split()).strip()
    return cleaned if cleaned else default


def get_row_value(row: dict, key: str, default: str | None = None) -> str | None:
    lower_map = {str(k).strip().lower(): v for k, v in row.items()}
    return lower_map.get(key.lower(), default)


def load_csv_rows(csv_path: Path) -> Iterable[dict]:
    raw_bytes = csv_path.read_bytes()
    decoded_text = None

    for encoding in ("utf-8", "latin1"):
        try:
            decoded_text = raw_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue

    if decoded_text is None:
        raise UnicodeDecodeError(
            "utf-8",
            raw_bytes,
            0,
            1,
            f"Unable to decode {csv_path.name} with utf-8 or latin1",
        )

    reader = csv.DictReader(io.StringIO(decoded_text))
    for row in reader:
        yield row


def get_football_sport_id(db) -> int:
    sport = db.query(Sport).filter(Sport.name.ilike("football")).first()
    if not sport:
        raise ValueError("Sport 'football' was not found in the database.")
    return sport.id


def get_basketball_sport_id(db) -> int:
    sport = db.query(Sport).filter(Sport.name.ilike("basketball")).first()
    if not sport:
        raise ValueError("Sport 'basketball' was not found in the database.")
    return sport.id


def load_team_cache(db, sport_id: int):
    teams = db.query(Team).filter(Team.sport_id == sport_id).all()
    return {normalize_text(team.name, ""): team for team in teams if normalize_text(team.name, "")}


def load_player_key_cache(db, sport_id: int):
    existing_players = db.query(Player.name, Player.team_id).filter(Player.sport_id == sport_id).all()
    return {
        (normalize_text(name, ""), team_id)
        for name, team_id in existing_players
        if normalize_text(name, "")
    }


def get_or_create_team(db, team_cache, sport_id: int, team_name: str):
    normalized_team_name = normalize_text(team_name, None)
    if not normalized_team_name:
        raise ValueError("Team name is empty")

    cached_team = team_cache.get(normalized_team_name)
    if cached_team is not None:
        return cached_team, False

    existing_team = (
        db.query(Team)
        .filter(Team.sport_id == sport_id)
        .filter(Team.name == normalized_team_name)
        .first()
    )
    if existing_team:
        team_cache[normalized_team_name] = existing_team
        return existing_team, False

    new_team = Team(name=normalized_team_name, sport_id=sport_id)
    db.add(new_team)
    db.flush()
    team_cache[normalized_team_name] = new_team
    return new_team, True


def import_nba(db, csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    rows = list(load_csv_rows(csv_path))
    if not rows:
        return {
            "file": str(csv_path),
            "teams_added": 0,
            "players_added": 0,
            "duplicate_players": 0,
            "skipped_rows": 0,
            "rows_processed": 0,
        }

    required_columns = {"team", "player"}
    present_columns = {str(column).strip().lower() for column in rows[0].keys()}
    missing_columns = required_columns - present_columns
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {sorted(missing_columns)}")

    sport = db.query(Sport).filter_by(name="basketball").first()
    if not sport:
        raise ValueError("Sport 'basketball' not found in sports table.")

    team_cache = {
        name: team_id
        for name, team_id in db.query(Team.name, Team.id).filter_by(sport_id=sport.id).all()
    }
    existing_players = set(db.query(Player.name, Player.team_id).filter_by(sport_id=sport.id).all())

    teams_added = 0
    players_added = 0
    skipped_rows = 0
    duplicate_players = 0
    pending_changes = 0

    total_rows = len(rows)
    for idx, row in enumerate(rows, start=1):
        team_name = normalize_text(get_row_value(row, "TEAM"), None)
        player_name = normalize_text(get_row_value(row, "PLAYER"), None)

        if not team_name or not player_name:
            skipped_rows += 1
            continue

        team_id = team_cache.get(team_name)
        if team_id is None:
            team = Team(name=team_name, sport_id=sport.id)
            db.add(team)
            db.flush()
            team_id = team.id
            team_cache[team_name] = team_id
            teams_added += 1
            pending_changes += 1

        player_key = (player_name, team_id)
        if player_key in existing_players:
            duplicate_players += 1
        else:
            db.add(
                Player(
                    name=player_name,
                    team_id=team_id,
                    position="Unknown",
                    sport_id=sport.id,
                )
            )
            existing_players.add(player_key)
            players_added += 1
            pending_changes += 1

        if pending_changes >= BATCH_SIZE:
            db.commit()
            pending_changes = 0

        if idx % PROGRESS_EVERY == 0:
            pass

    if pending_changes > 0:
        db.commit()

    return {
        "file": str(csv_path),
        "teams_added": teams_added,
        "players_added": players_added,
        "duplicate_players": duplicate_players,
        "skipped_rows": skipped_rows,
        "rows_processed": total_rows,
    }


def import_premier_league(db, csv_paths):
    sport_id = get_football_sport_id(db)
    team_cache = load_team_cache(db, sport_id)
    player_key_cache = load_player_key_cache(db, sport_id)

    results = []
    totals = {
        "teams_added": 0,
        "players_added": 0,
        "duplicates_skipped": 0,
        "invalid_rows_skipped": 0,
        "rows_processed": 0,
    }

    for csv_path_like in csv_paths:
        csv_path = Path(csv_path_like)
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing file: {csv_path}")

        teams_added = 0
        players_added = 0
        duplicates_skipped = 0
        invalid_rows_skipped = 0
        processed_rows = 0
        pending_rows = 0

        for row in load_csv_rows(csv_path):
            processed_rows += 1
            team_name = normalize_text(get_row_value(row, "team_name"), None)
            player_name = normalize_text(get_row_value(row, "player_name"), None)
            position = normalize_text(get_row_value(row, "position"), "Unknown")

            if not team_name or not player_name:
                invalid_rows_skipped += 1
                continue

            team, created_team = get_or_create_team(db, team_cache, sport_id, team_name)
            if created_team:
                teams_added += 1

            player_key = (player_name, team.id)
            if player_key in player_key_cache:
                duplicates_skipped += 1
                continue

            db.add(
                Player(
                    name=player_name,
                    team_id=team.id,
                    position=position or "Unknown",
                    sport_id=sport_id,
                )
            )
            player_key_cache.add(player_key)
            players_added += 1
            pending_rows += 1

            if pending_rows >= BATCH_SIZE:
                db.commit()
                pending_rows = 0

        if pending_rows > 0:
            db.commit()

        file_result = {
            "file": str(csv_path),
            "teams_added": teams_added,
            "players_added": players_added,
            "duplicates_skipped": duplicates_skipped,
            "invalid_rows_skipped": invalid_rows_skipped,
            "rows_processed": processed_rows,
        }
        results.append(file_result)
        totals["teams_added"] += teams_added
        totals["players_added"] += players_added
        totals["duplicates_skipped"] += duplicates_skipped
        totals["invalid_rows_skipped"] += invalid_rows_skipped
        totals["rows_processed"] += processed_rows

    return {"files": results, "totals": totals}
