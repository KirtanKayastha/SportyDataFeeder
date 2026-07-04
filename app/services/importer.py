# /home/sam069/projects/SportyDataFeeder/app/services/importer.py

import csv
import io
import re
from pathlib import Path
from typing import Any, Iterable

from app.database import Player, PlayerStat, Sport, Team
from app.services.team_ratings import APP_TEAM_ALIASES
from app.services.team_ratings import normalize_team_name as _elo_normalize_team_name

BATCH_SIZE = 500
PROGRESS_EVERY = 100


def _canonicalize_team_name(team_cache: dict, name: str) -> str:
    """Some seed CSVs use a short/alt form of a team name (e.g. "Liverpool"
    instead of "Liverpool FC"), which previously created a duplicate Team
    row per name variant. If `name` is the short alias (per APP_TEAM_ALIASES)
    of a team already present in this sport's cache, resolve to that team's
    actual stored name instead.

    Deliberately data-driven rather than a static reverse-alias dict: two
    long names can share one short alias (e.g. both "Wolverhampton" and
    "Wolverhampton Wanderers" -> "Wolves"), so a static reverse map could
    resolve to a name that doesn't match what's actually stored — recreating
    the very duplicate-row problem this is meant to prevent.
    """
    for existing_name in team_cache:
        if _elo_normalize_team_name(existing_name, APP_TEAM_ALIASES) == name:
            return existing_name
    return name

GAMEWEEK_PATTERN = re.compile(r"(\d+)\s*(?:st|nd|rd|th)?\s*gameday", re.IGNORECASE)
SEASON_PATTERN = re.compile(r"season\s*(\d{4}(?:-\d{2,4})?)", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"(\d{4})")


def normalize_text(value: Any, default: str | None = None) -> str | None:
    if value is None:
        return default
    cleaned = " ".join(str(value).split()).strip()
    return cleaned if cleaned else default


def get_row_value(row: dict, key: str, default: str | None = None) -> str | None:
    lower_map = {str(k).strip().lower(): v for k, v in row.items()}
    return lower_map.get(key.lower(), default)


def parse_int(value: Any) -> int | None:
    text = normalize_text(value, None)
    if text is None:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_float(value: Any) -> float | None:
    text = normalize_text(value, None)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def infer_gameweek_and_season(csv_path: Path) -> tuple[int, str]:
    """Best-effort gameweek/season from a stats CSV filename.

    Files like ``...until31thGameDayOnSeason2025-26.csv`` yield (31, "2025-26").
    Season-total files without a gameweek (e.g. ``nba_player_stats_2026.csv``)
    yield gameweek 0 — "season to date".
    """
    stem = csv_path.stem
    gameweek_match = GAMEWEEK_PATTERN.search(stem)
    gameweek = int(gameweek_match.group(1)) if gameweek_match else 0

    season_match = SEASON_PATTERN.search(stem)
    if season_match:
        season = season_match.group(1)
    else:
        year_match = YEAR_PATTERN.search(stem)
        season = year_match.group(1) if year_match else "unknown"
    return gameweek, season


def load_stat_key_cache(db) -> set[tuple[int, int, str]]:
    return set(db.query(PlayerStat.player_id, PlayerStat.gameweek, PlayerStat.season).all())


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


def load_player_key_cache(db, sport_id: int) -> dict[tuple[str, int], int]:
    existing_players = db.query(Player.name, Player.team_id, Player.id).filter(Player.sport_id == sport_id).all()
    return {
        (normalize_text(name, ""), team_id): player_id
        for name, team_id, player_id in existing_players
        if normalize_text(name, "")
    }


def get_or_create_team(db, team_cache, sport_id: int, team_name: str):
    normalized_team_name = normalize_text(team_name, None)
    if not normalized_team_name:
        raise ValueError("Team name is empty")
    normalized_team_name = _canonicalize_team_name(team_cache, normalized_team_name)

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


def import_nba(db, csv_path, gameweek: int | None = None, season: str | None = None):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    inferred_gameweek, inferred_season = infer_gameweek_and_season(csv_path)
    gameweek = gameweek if gameweek is not None else inferred_gameweek
    season = season if season is not None else inferred_season

    rows = list(load_csv_rows(csv_path))
    if not rows:
        return {
            "file": str(csv_path),
            "teams_added": 0,
            "players_added": 0,
            "duplicate_players": 0,
            "skipped_rows": 0,
            "rows_processed": 0,
            "stats_added": 0,
            "stats_skipped": 0,
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
    existing_players = {
        (name, team_id): player_id
        for name, team_id, player_id in db.query(Player.name, Player.team_id, Player.id)
        .filter_by(sport_id=sport.id)
        .all()
    }
    stat_key_cache = load_stat_key_cache(db)

    teams_added = 0
    players_added = 0
    skipped_rows = 0
    duplicate_players = 0
    stats_added = 0
    stats_skipped = 0
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
        player_id = existing_players.get(player_key)
        if player_id is not None:
            duplicate_players += 1
        else:
            player = Player(
                name=player_name,
                team_id=team_id,
                position="Unknown",
                sport_id=sport.id,
            )
            db.add(player)
            db.flush()
            player_id = player.id
            existing_players[player_key] = player_id
            players_added += 1
            pending_changes += 1

        stat_key = (player_id, gameweek, season)
        if stat_key in stat_key_cache:
            stats_skipped += 1
        else:
            db.add(
                PlayerStat(
                    player_id=player_id,
                    gameweek=gameweek,
                    season=season,
                    minutes=parse_int(get_row_value(row, "MIN")),
                    pts=parse_int(get_row_value(row, "PTS")),
                    ast=parse_int(get_row_value(row, "AST")),
                    reb=parse_int(get_row_value(row, "REB")),
                    stl=parse_int(get_row_value(row, "STL")),
                    blk=parse_int(get_row_value(row, "BLK")),
                )
            )
            stat_key_cache.add(stat_key)
            stats_added += 1
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
        "stats_added": stats_added,
        "stats_skipped": stats_skipped,
    }


def import_premier_league(db, csv_paths, gameweek: int | None = None, season: str | None = None):
    sport_id = get_football_sport_id(db)
    team_cache = load_team_cache(db, sport_id)
    player_key_cache = load_player_key_cache(db, sport_id)
    stat_key_cache = load_stat_key_cache(db)

    results = []
    totals = {
        "teams_added": 0,
        "players_added": 0,
        "duplicates_skipped": 0,
        "invalid_rows_skipped": 0,
        "rows_processed": 0,
        "stats_added": 0,
        "stats_skipped": 0,
    }

    for csv_path_like in csv_paths:
        csv_path = Path(csv_path_like)
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing file: {csv_path}")

        inferred_gameweek, inferred_season = infer_gameweek_and_season(csv_path)
        file_gameweek = gameweek if gameweek is not None else inferred_gameweek
        file_season = season if season is not None else inferred_season

        teams_added = 0
        players_added = 0
        duplicates_skipped = 0
        invalid_rows_skipped = 0
        stats_added = 0
        stats_skipped = 0
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
            player_id = player_key_cache.get(player_key)
            if player_id is not None:
                duplicates_skipped += 1
            else:
                player = Player(
                    name=player_name,
                    team_id=team.id,
                    position=position or "Unknown",
                    sport_id=sport_id,
                )
                db.add(player)
                db.flush()
                player_id = player.id
                player_key_cache[player_key] = player_id
                players_added += 1
                pending_rows += 1

            stat_key = (player_id, file_gameweek, file_season)
            if stat_key in stat_key_cache:
                stats_skipped += 1
            else:
                db.add(
                    PlayerStat(
                        player_id=player_id,
                        gameweek=file_gameweek,
                        season=file_season,
                        minutes=parse_int(get_row_value(row, "minutesPlayed")),
                        goals=parse_int(get_row_value(row, "goals")),
                        assists=parse_int(get_row_value(row, "assists")),
                        yellows=parse_int(get_row_value(row, "yellowCards")),
                        reds=parse_int(get_row_value(row, "redCards")),
                        points=parse_float(get_row_value(row, "rating")),
                    )
                )
                stat_key_cache.add(stat_key)
                stats_added += 1
                pending_rows += 1

            if pending_rows >= BATCH_SIZE:
                db.commit()
                pending_rows = 0

        if pending_rows > 0:
            db.commit()

        file_result = {
            "file": str(csv_path),
            "gameweek": file_gameweek,
            "season": file_season,
            "teams_added": teams_added,
            "players_added": players_added,
            "duplicates_skipped": duplicates_skipped,
            "invalid_rows_skipped": invalid_rows_skipped,
            "rows_processed": processed_rows,
            "stats_added": stats_added,
            "stats_skipped": stats_skipped,
        }
        results.append(file_result)
        totals["teams_added"] += teams_added
        totals["players_added"] += players_added
        totals["duplicates_skipped"] += duplicates_skipped
        totals["invalid_rows_skipped"] += invalid_rows_skipped
        totals["rows_processed"] += processed_rows
        totals["stats_added"] += stats_added
        totals["stats_skipped"] += stats_skipped

    return {"files": results, "totals": totals}
