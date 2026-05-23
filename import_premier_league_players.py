from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from database import Player, Session, Sport, Team


CSV_FILES = [
    "premier_league_complete_stats_until31thGameDayOnSeason2025-26.csv",
    "premier_league_complete_stats_until35thGameDayOnSeason2025-26.csv",
]

BATCH_SIZE = 500
PROGRESS_EVERY = 100


def normalize_text(value: Optional[str], default: Optional[str] = None) -> Optional[str]:
    if value is None:
        return default
    cleaned = " ".join(str(value).split()).strip()
    return cleaned if cleaned else default


def get_row_value(row: dict, key: str, default: Optional[str] = None) -> Optional[str]:
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


def get_football_sport_id(session) -> int:
    sport = session.query(Sport).filter(Sport.name.ilike("football")).first()
    if not sport:
        raise RuntimeError("Sport 'football' was not found in the database.")
    return sport.id


def load_team_cache(session, sport_id: int) -> Dict[str, Team]:
    teams = session.query(Team).filter(Team.sport_id == sport_id).all()
    return {normalize_text(team.name, ""): team for team in teams if normalize_text(team.name, "")}


def load_player_key_cache(session, sport_id: int) -> set[Tuple[str, int]]:
    existing_players = session.query(Player.name, Player.team_id).filter(Player.sport_id == sport_id).all()
    return {
        (normalize_text(name, ""), team_id)
        for name, team_id in existing_players
        if normalize_text(name, "")
    }


def get_or_create_team(session, team_cache: Dict[str, Team], sport_id: int, team_name: str) -> Tuple[Team, bool]:
    normalized_team_name = normalize_text(team_name, None)
    if not normalized_team_name:
        raise ValueError("Team name is empty")

    cached_team = team_cache.get(normalized_team_name)
    if cached_team is not None:
        return cached_team, False

    existing_team = (
        session.query(Team)
        .filter(Team.sport_id == sport_id)
        .filter(Team.name == normalized_team_name)
        .first()
    )
    if existing_team:
        team_cache[normalized_team_name] = existing_team
        return existing_team, False

    new_team = Team(name=normalized_team_name, sport_id=sport_id)
    session.add(new_team)
    session.flush()
    team_cache[normalized_team_name] = new_team
    return new_team, True


def process_file(csv_path: Path, sport_id: int) -> None:
    session = Session()

    try:
        team_cache = load_team_cache(session, sport_id)
        player_key_cache = load_player_key_cache(session, sport_id)

        teams_added = 0
        players_added = 0
        duplicates_skipped = 0
        invalid_rows_skipped = 0
        processed_rows = 0
        pending_rows = 0

        print(f"\nImporting {csv_path.name} ...")

        for row in load_csv_rows(csv_path):
            processed_rows += 1

            team_name = normalize_text(get_row_value(row, "team_name"), None)
            player_name = normalize_text(get_row_value(row, "player_name"), None)
            position = normalize_text(get_row_value(row, "position"), "Unknown")

            if not team_name or not player_name:
                invalid_rows_skipped += 1
                if processed_rows % PROGRESS_EVERY == 0:
                    print(
                        f"{csv_path.name}: processed {processed_rows} rows "
                        f"(teams added={teams_added}, players added={players_added}, "
                        f"duplicates skipped={duplicates_skipped}, invalid skipped={invalid_rows_skipped})"
                    )
                continue

            team, created_team = get_or_create_team(session, team_cache, sport_id, team_name)
            if created_team:
                teams_added += 1

            player_key = (player_name, team.id)
            if player_key in player_key_cache:
                duplicates_skipped += 1
                if processed_rows % PROGRESS_EVERY == 0:
                    print(
                        f"{csv_path.name}: processed {processed_rows} rows "
                        f"(teams added={teams_added}, players added={players_added}, "
                        f"duplicates skipped={duplicates_skipped}, invalid skipped={invalid_rows_skipped})"
                    )
                continue

            session.add(
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
                session.commit()
                pending_rows = 0

            if processed_rows % PROGRESS_EVERY == 0:
                print(
                    f"{csv_path.name}: processed {processed_rows} rows "
                    f"(teams added={teams_added}, players added={players_added}, "
                    f"duplicates skipped={duplicates_skipped}, invalid skipped={invalid_rows_skipped})"
                )

        if pending_rows > 0:
            session.commit()

        print(
            f"Finished {csv_path.name}: teams added={teams_added}, "
            f"players added={players_added}, duplicates skipped={duplicates_skipped}, "
            f"invalid rows skipped={invalid_rows_skipped}, rows processed={processed_rows}"
        )

    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def main() -> int:
    base_dir = Path(__file__).resolve().parent

    session = Session()
    try:
        sport_id = get_football_sport_id(session)
    finally:
        session.close()

    for file_name in CSV_FILES:
        csv_path = base_dir / file_name
        if not csv_path.exists():
            print(f"Missing file: {csv_path}")
            continue
        process_file(csv_path, sport_id)

    print("\nAll imports completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
