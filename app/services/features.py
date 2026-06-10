# /home/sam069/projects/SportyDataFeeder/app/services/features.py
#
# Feature layer (PRD R-3.1 / R-3.2): per-player rates and form derived from
# player_stats rows, with league-average cold-start fallbacks, plus team
# strength. All downstream consumers (training, simulation, prediction) read
# player features through compute_player_features — never raw stat rows.

import logging

from app.database import Player, PlayerStat, Sport
from app.services.sport_resolver import SportType, resolve_sport_type

logger = logging.getLogger(__name__)

EWMA_ALPHA = 0.4
MIN_FORM_ROWS = 5
MAX_FORM_ROWS = 20

# Neutral form for cold-start players: /15 normalisation puts it at 0.5 team strength.
NEUTRAL_FORM_INDEX = 7.5

FULL_MATCH_MINUTES = {SportType.FOOTBALL: 90, SportType.BASKETBALL: 48}

# League-average per-minute event probabilities (PRD R-3.1).
FOOTBALL_FALLBACK_RATES = {
    "goal": 0.003,
    "assist": 0.003,
    "yellow_card": 0.002,
    "red_card": 0.0002,
}
BASKETBALL_FALLBACK_RATES = {
    "point_2": 0.04,
    "point_3": 0.015,
    "free_throw": 0.02,
    "assist": 0.05,
    "rebound": 0.07,
}

# player_stats only stores total basketball points, not the 2pt/3pt/FT split.
# Approximate NBA scoring mix used to decompose totals into event rates.
BASKETBALL_POINT_MIX = {"point_2": 0.60, "point_3": 0.25, "free_throw": 0.15}
BASKETBALL_POINT_VALUE = {"point_2": 2, "point_3": 3, "free_throw": 1}


def _fallback_features(player_id: int, sport_type: SportType) -> dict:
    rates = BASKETBALL_FALLBACK_RATES if sport_type is SportType.BASKETBALL else FOOTBALL_FALLBACK_RATES
    return {
        "player_id": player_id,
        "sport_type": sport_type,
        "goals_per90": rates.get("goal", 0.0) * 90,
        "assists_per90": rates.get("assist", 0.0) * 90,
        "cards_per90": (rates.get("yellow_card", 0.0) + rates.get("red_card", 0.0)) * 90,
        "form_index": NEUTRAL_FORM_INDEX,
        "minutes_ratio": 1.0,
        "event_rates": dict(rates),
        "cold_start": True,
    }


def _ewma(values: list[float], alpha: float = EWMA_ALPHA) -> float:
    """Exponentially weighted average; values must be ordered newest first."""
    if not values:
        return 0.0
    weighted_sum = 0.0
    weight_total = 0.0
    weight = alpha
    for value in values:
        weighted_sum += weight * value
        weight_total += weight
        weight *= 1 - alpha
    return weighted_sum / weight_total


def _resolve_player_sport_type(db, player: Player) -> SportType:
    sport = db.query(Sport).filter_by(id=player.sport_id).first()
    return resolve_sport_type(sport.name if sport else None)


def compute_player_features(player_id: int, db) -> dict:
    """Per-player rates and form. Never raises for missing players or stats —
    cold starts get league-average fallbacks (PRD R-3.1)."""
    player = db.query(Player).filter_by(id=player_id).first()
    if not player:
        logger.warning("compute_player_features: player %s not found; using football fallback", player_id)
        return _fallback_features(player_id, SportType.FOOTBALL)

    sport_type = _resolve_player_sport_type(db, player)
    if sport_type is SportType.UNKNOWN:
        sport_type = SportType.FOOTBALL

    stat_rows = (
        db.query(PlayerStat)
        .filter_by(player_id=player_id)
        .order_by(PlayerStat.season.desc(), PlayerStat.gameweek.desc())
        .limit(MAX_FORM_ROWS)
        .all()
    )
    total_minutes = sum(row.minutes or 0 for row in stat_rows)
    if not stat_rows or total_minutes <= 0:
        logger.warning("Cold start for player %s (%s stat rows)", player_id, len(stat_rows))
        return _fallback_features(player_id, sport_type)

    full_match = FULL_MATCH_MINUTES.get(sport_type, 90)
    minutes_ratio = min(1.0, (total_minutes / len(stat_rows)) / full_match)

    if sport_type is SportType.BASKETBALL:
        total_pts = sum(row.pts or 0 for row in stat_rows)
        total_ast = sum(row.ast or 0 for row in stat_rows)
        total_reb = sum(row.reb or 0 for row in stat_rows)
        event_rates = {
            event_type: (total_pts * share / BASKETBALL_POINT_VALUE[event_type]) / total_minutes
            for event_type, share in BASKETBALL_POINT_MIX.items()
        }
        event_rates["assist"] = total_ast / total_minutes
        event_rates["rebound"] = total_reb / total_minutes
        goals_per90 = 0.0
        assists_per90 = (total_ast / total_minutes) * 90
        cards_per90 = 0.0
        # Form basis: points per 36 minutes (comparable scale to football ratings).
        form_values = [
            ((row.pts or 0) / row.minutes) * 36 for row in stat_rows if (row.minutes or 0) > 0
        ]
    else:
        total_goals = sum(row.goals or 0 for row in stat_rows)
        total_assists = sum(row.assists or 0 for row in stat_rows)
        total_yellows = sum(row.yellows or 0 for row in stat_rows)
        total_reds = sum(row.reds or 0 for row in stat_rows)
        event_rates = {
            "goal": total_goals / total_minutes,
            "assist": total_assists / total_minutes,
            "yellow_card": total_yellows / total_minutes,
            "red_card": total_reds / total_minutes,
        }
        goals_per90 = event_rates["goal"] * 90
        assists_per90 = event_rates["assist"] * 90
        cards_per90 = (event_rates["yellow_card"] + event_rates["red_card"]) * 90
        # Form basis: per-row match points/rating, newest first.
        form_values = [row.points for row in stat_rows if row.points is not None]

    form_index = _ewma(form_values[:MAX_FORM_ROWS]) if form_values else NEUTRAL_FORM_INDEX

    return {
        "player_id": player_id,
        "sport_type": sport_type,
        "goals_per90": goals_per90,
        "assists_per90": assists_per90,
        "cards_per90": cards_per90,
        "form_index": form_index,
        "minutes_ratio": minutes_ratio,
        "event_rates": event_rates,
        "cold_start": False,
    }


def compute_team_strength(team_id: int, db) -> float:
    """Mean form_index of the team's players, normalised /15 and clamped to 0..1.
    0.5 for a team with no players (PRD R-3.2)."""
    players = db.query(Player).filter_by(team_id=team_id).all()
    if not players:
        return 0.5
    form_indexes = [compute_player_features(player.id, db)["form_index"] for player in players]
    strength = (sum(form_indexes) / len(form_indexes)) / 15.0
    return max(0.0, min(1.0, strength))
