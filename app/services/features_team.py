# /home/sam069/projects/SportyDataFeeder/app/services/features_team.py
#
# Causal, leakage-free rolling team form features for the logistic outcome
# candidate (Phase C, reports/OUTCOME_MODEL_TRAINING_PLAN.md). For each match
# we record each side's recent form computed ONLY from its earlier matches,
# then update histories after the row is emitted. Home form uses the home
# team's home matches; away form uses the away team's away matches.

from collections import defaultdict, deque


def annotate_rolling_form(df, window: int = 5):
    """Add causal rolling-form columns to a chronologically sorted match frame.

    New columns (all pre-match):
      home_ppg, away_ppg          — points/game over last `window` matches (any venue)
      home_gf, home_ga            — home team goals for/against per game (home venue)
      away_gf, away_ga            — away team goals for/against per game (away venue)
    Cold start (no history) defaults to league-ish priors: 1.4 ppg, 1.4 GF, 1.4 GA.
    """
    overall = defaultdict(lambda: deque(maxlen=window))   # team -> recent points
    home_for = defaultdict(lambda: deque(maxlen=window))
    home_against = defaultdict(lambda: deque(maxlen=window))
    away_for = defaultdict(lambda: deque(maxlen=window))
    away_against = defaultdict(lambda: deque(maxlen=window))

    def _avg(dq, prior):
        return sum(dq) / len(dq) if dq else prior

    rows = {k: [] for k in ["home_ppg", "away_ppg", "home_gf", "home_ga", "away_gf", "away_ga"]}
    for r in df.itertuples(index=False):
        h, a, hg, ag = r.home, r.away, int(r.fthg), int(r.ftag)
        rows["home_ppg"].append(_avg(overall[h], 1.4))
        rows["away_ppg"].append(_avg(overall[a], 1.4))
        rows["home_gf"].append(_avg(home_for[h], 1.4))
        rows["home_ga"].append(_avg(home_against[h], 1.4))
        rows["away_gf"].append(_avg(away_for[a], 1.4))
        rows["away_ga"].append(_avg(away_against[a], 1.4))

        # update AFTER recording (causal)
        hp = 3 if hg > ag else (1 if hg == ag else 0)
        ap = 3 if ag > hg else (1 if hg == ag else 0)
        overall[h].append(hp); overall[a].append(ap)
        home_for[h].append(hg); home_against[h].append(ag)
        away_for[a].append(ag); away_against[a].append(hg)

    out = df.copy()
    for k, v in rows.items():
        out[k] = v
    return out
