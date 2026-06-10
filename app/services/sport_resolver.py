# /home/sam069/projects/SportyDataFeeder/app/services/sport_resolver.py
#
# Single place that maps free-form sport names to a SportType. Adding a new
# sport (e.g. cricket simulation) means extending the alias map here — no
# other module may do substring checks on sport names.

from enum import Enum


class SportType(str, Enum):
    FOOTBALL = "football"
    BASKETBALL = "basketball"
    CRICKET = "cricket"
    UNKNOWN = "unknown"


SPORT_ALIASES: dict[SportType, list[str]] = {
    SportType.FOOTBALL: ["football", "soccer", "premier league", "epl"],
    SportType.BASKETBALL: ["basketball", "basket", "nba"],
    SportType.CRICKET: ["cricket", "ipl"],
}


def resolve_sport_type(name: str | None) -> SportType:
    normalized = " ".join((name or "").split()).lower()
    if not normalized:
        return SportType.UNKNOWN

    for sport_type, aliases in SPORT_ALIASES.items():
        if normalized in aliases:
            return sport_type

    # Tolerate decorated names like "NBA Basketball" or "English Premier League".
    for sport_type, aliases in SPORT_ALIASES.items():
        if any(alias in normalized for alias in aliases):
            return sport_type

    return SportType.UNKNOWN


def get_aliases(sport_type: SportType) -> list[str]:
    return SPORT_ALIASES.get(sport_type, [])
