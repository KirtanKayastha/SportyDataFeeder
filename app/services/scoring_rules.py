# /home/sam069/projects/SportyDataFeeder/app/services/scoring_rules.py
#
# Single source of truth for how events translate into match scores.
# Both app/routers/matches.py (score replay on read) and
# app/services/simulation.py (live score updates) import from here —
# never duplicate these rules at a call site.

from typing import Any, Iterable, Mapping

from app.services.sport_resolver import SportType

BASKETBALL_POINT_VALUES = {"point_2": 2, "point_3": 3, "free_throw": 1}


def event_score_value(event_type: str, extra: Mapping[str, Any] | None, sport_type: SportType) -> int:
    """Points a single event adds to its team's score. 0 for non-scoring events."""
    if sport_type is SportType.BASKETBALL:
        if event_type in BASKETBALL_POINT_VALUES:
            points = (extra or {}).get("points")
            if points is None:
                points = BASKETBALL_POINT_VALUES[event_type]
            return int(points or 0)
        return 0
    if sport_type is SportType.FOOTBALL:
        return 1 if event_type == "goal" else 0
    return 0


def score_events(events: Iterable[Mapping[str, Any]], sport_type: SportType) -> dict[int, int]:
    """Aggregate scores per team.

    Each event is a mapping with keys ``event_type``, ``team_id`` and
    optionally ``extra`` (a dict). Events whose ``team_id`` is None are
    ignored. Returns ``{team_id: score}``.
    """
    scores: dict[int, int] = {}
    for event in events:
        team_id = event.get("team_id")
        if team_id is None:
            continue
        value = event_score_value(event["event_type"], event.get("extra"), sport_type)
        if value:
            scores[team_id] = scores.get(team_id, 0) + value
    return scores
