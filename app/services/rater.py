# /home/sam069/projects/SportyDataFeeder/app/services/rater.py
#
# Rule-based post-match player ratings (PRD R-3.5). Base 6.0, per-event
# weights per sport, clamped to 1.0–10.0.

from typing import Iterable, Mapping

from app.services.sport_resolver import SportType

BASE_RATING = 6.0
MIN_RATING = 1.0
MAX_RATING = 10.0

FOOTBALL_WEIGHTS = {
    "goal": 2.0,
    "assist": 1.2,
    "yellow_card": -0.5,
    "red_card": -2.5,
}
BASKETBALL_WEIGHTS = {
    "point_2": 0.8,
    "point_3": 1.2,
    "free_throw": 0.3,
    "assist": 0.7,
    "rebound": 0.4,
    "steal": 0.6,
    "block": 0.5,
}


def _weights_for(sport_type: SportType) -> dict[str, float]:
    if sport_type is SportType.BASKETBALL:
        return BASKETBALL_WEIGHTS
    return FOOTBALL_WEIGHTS


def rate_player(event_types: Iterable[str], sport_type: SportType) -> float:
    """Rating for one player's match events, clamped to 1.0–10.0.
    Event types without a weight (e.g. turnover) contribute nothing."""
    weights = _weights_for(sport_type)
    rating = BASE_RATING + sum(weights.get(event_type, 0.0) for event_type in event_types)
    return max(MIN_RATING, min(MAX_RATING, rating))


def rate_players(events_by_player: Mapping[int, Iterable[str]], sport_type: SportType) -> dict[int, float]:
    return {
        player_id: rate_player(event_types, sport_type)
        for player_id, event_types in events_by_player.items()
    }


def find_man_of_match(ratings: Mapping[int, float]) -> int | None:
    """Player id with the highest rating; ties resolve to the lowest player id
    (deterministic). None for an empty mapping."""
    if not ratings:
        return None
    return min(ratings, key=lambda player_id: (-ratings[player_id], player_id))
