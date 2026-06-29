# /home/sam069/projects/SportyDataFeeder/app/services/team_ratings.py
#
# Causal Elo ratings for team-level outcome modelling (see
# reports/OUTCOME_MODEL_TRAINING_PLAN.md). Ratings are updated strictly forward
# in time: a match's pre-match ratings depend only on earlier matches, so no
# future information leaks into any feature. New teams (promotions) start at the
# default rating; ratings mean-revert toward the default at each season boundary.
#
# This is a pure rating engine — no DB, no sklearn — so it is trivially testable
# and reusable by both the backtest harness and (later) live prediction.

import html
from dataclasses import dataclass, field

# The application stores football teams under long/official names (imported from
# the Premier League roster CSVs), while football-data.co.uk uses short names.
# This maps app names -> the canonical name used in the Elo ratings table so
# live /predict can look up a team's rating. Names not listed pass through
# unchanged (after HTML-unescaping & trimming).
APP_TEAM_ALIASES = {
    "Brighton & Hove Albion": "Brighton",
    "Leeds United": "Leeds",
    "Liverpool FC": "Liverpool",
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton": "Wolves",
    "Wolverhampton Wanderers": "Wolves",
    "Sheffield United": "Sheffield Utd",
    "Nott'm Forest": "Nottingham Forest",
}


def normalize_team_name(name: str | None, aliases: dict | None = None) -> str | None:
    """HTML-unescape + trim a team name, then map it through `aliases` (a
    bundle-specific app-name -> Elo-key map). Names not in the map pass through.
    Football uses APP_TEAM_ALIASES; basketball uses abbreviations (identity)."""
    if name is None:
        return None
    n = html.unescape(name).strip()
    return (aliases or {}).get(n, n)


def canonical_team_name(name: str | None) -> str | None:
    """Football convenience: normalize with the built-in APP_TEAM_ALIASES."""
    return normalize_team_name(name, APP_TEAM_ALIASES)


@dataclass
class EloModel:
    k: float = 20.0           # update step size
    home_advantage: float = 65.0  # Elo points added to the home side
    base: float = 1500.0      # rating for an unseen team
    season_regression: float = 0.25  # fraction pulled back to `base` each new season
    ratings: dict[str, float] = field(default_factory=dict)
    _last_season: str | None = field(default=None, repr=False)

    def rating(self, team: str) -> float:
        return self.ratings.get(team, self.base)

    def expected_home(self, home: str, away: str) -> float:
        """Elo expected score for the home team in [0, 1] (win=1, draw=0.5)."""
        diff = (self.rating(home) + self.home_advantage) - self.rating(away)
        return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

    def _maybe_regress(self, season: str) -> None:
        """At a new season, pull every rating partway back to the mean."""
        if self._last_season is not None and season != self._last_season:
            for team, r in self.ratings.items():
                self.ratings[team] = r + self.season_regression * (self.base - r)
        self._last_season = season

    def update(self, home: str, away: str, fthg: int, ftag: int, season: str) -> None:
        """Apply one finished match. Call in chronological order."""
        self._maybe_regress(season)
        exp_home = self.expected_home(home, away)
        if fthg > ftag:
            score_home = 1.0
        elif fthg < ftag:
            score_home = 0.0
        else:
            score_home = 0.5
        delta = self.k * (score_home - exp_home)
        self.ratings[home] = self.rating(home) + delta
        self.ratings[away] = self.rating(away) - delta


def fit_elo(df, k: float = 20.0, home_advantage: float = 65.0) -> EloModel:
    """Process every match in chronological order and return the EloModel whose
    `ratings` reflect the state AFTER the last match — i.e. each team's current
    rating, for live prediction. Caller must pass a causally sorted frame."""
    elo = EloModel(k=k, home_advantage=home_advantage)
    for row in df.itertuples(index=False):
        elo.update(row.home, row.away, int(row.fthg), int(row.ftag), row.season)
    return elo


def annotate_pre_match_elo(df, k: float = 20.0, home_advantage: float = 65.0):
    """Add causal pre-match Elo columns to a chronologically sorted match frame.

    Returns the frame with new columns:
      elo_home, elo_away  — ratings BEFORE the match (no leakage)
      elo_diff            — (elo_home + home_advantage) - elo_away
      elo_exp_home        — Elo expected home score in [0, 1]
    The model is updated with each match's result only AFTER its features are
    recorded, so row t never sees its own or any later result.
    """
    elo = EloModel(k=k, home_advantage=home_advantage)
    eh, ea, ediff, eexp = [], [], [], []
    for row in df.itertuples(index=False):
        rh, ra = elo.rating(row.home), elo.rating(row.away)
        eh.append(rh)
        ea.append(ra)
        ediff.append((rh + elo.home_advantage) - ra)
        eexp.append(elo.expected_home(row.home, row.away))
        elo.update(row.home, row.away, int(row.fthg), int(row.ftag), row.season)
    out = df.copy()
    out["elo_home"] = eh
    out["elo_away"] = ea
    out["elo_diff"] = ediff
    out["elo_exp_home"] = eexp
    return out
