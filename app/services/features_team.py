# /home/sam069/projects/SportyDataFeeder/app/services/features_team.py
#
# Causal, leakage-free rolling team form features for the logistic outcome
# candidate (Phase C, reports/OUTCOME_MODEL_TRAINING_PLAN.md). For each match
# we record each side's recent form computed ONLY from its earlier matches,
# then update histories after the row is emitted. Home form uses the home
# team's home matches; away form uses the away team's away matches.

from collections import defaultdict, deque

REST_CAP_DAYS = 5  # a week+ off tells you no more than 5 days off


def annotate_rest_days(df):
    """Add causal schedule-rest columns to a chronologically sorted match frame
    (basketball step 2 of reports/MODEL_IMPROVEMENT_PLAN.md — back-to-backs and
    rest differential are well-documented NBA effects).

    New columns (all pre-match, from each team's PREVIOUS game date only):
      home_rest, away_rest  — full days off before this game, capped at
                              REST_CAP_DAYS; season openers get the cap
      rest_diff             — home_rest - away_rest
      home_b2b, away_b2b    — 1.0 when playing the second night of a back-to-back
    """
    last_game = {}
    cols = {k: [] for k in ["home_rest", "away_rest", "rest_diff", "home_b2b", "away_b2b"]}
    for r in df.itertuples(index=False):
        rests = []
        for team in (r.home, r.away):
            prev = last_game.get(team)
            if prev is None:
                rests.append(float(REST_CAP_DAYS))
            else:
                rests.append(float(min(max((r.date - prev).days - 1, 0), REST_CAP_DAYS)))
        hr, ar = rests
        cols["home_rest"].append(hr)
        cols["away_rest"].append(ar)
        cols["rest_diff"].append(hr - ar)
        cols["home_b2b"].append(1.0 if hr == 0 else 0.0)
        cols["away_b2b"].append(1.0 if ar == 0 else 0.0)
        last_game[r.home] = r.date
        last_game[r.away] = r.date

    out = df.copy()
    for k, v in cols.items():
        out[k] = v
    return out


# League-average shots on target per game (football-data EPL, ~2013-2026).
SOT_PRIOR = 4.3


def annotate_shot_form(df, window: int = 10):
    """Add causal rolling shots-on-target form to a chronologically sorted
    football match frame (step 3 of reports/MODEL_IMPROVEMENT_PLAN.md — SoT is
    a much less noisy team-quality signal than goals). Requires HST/AST columns
    (football-data schema).

    New columns (all pre-match, any venue, last `window` matches):
      home_sot_for, home_sot_against, away_sot_for, away_sot_against
      sot_net_diff — (home for-against) - (away for-against), the compact form
    Cold start (no history) defaults to the league prior.
    """
    sot_for = defaultdict(lambda: deque(maxlen=window))
    sot_against = defaultdict(lambda: deque(maxlen=window))

    def _avg(dq):
        return sum(dq) / len(dq) if dq else SOT_PRIOR

    cols = _run_shot_form(df, sot_for, sot_against, _avg)

    out = df.copy()
    for k, v in cols.items():
        out[k] = v
    return out


def shot_form_state(df, window: int = 10) -> dict[str, float]:
    """Each team's CURRENT net SoT form ((for - against) mean over its last
    `window` matches) after processing every row — the prediction-time
    counterpart of annotate_shot_form, baked into the outcome_v2 bundle the way
    elo_ratings are. Keys are team names as they appear in df."""
    sot_for = defaultdict(lambda: deque(maxlen=window))
    sot_against = defaultdict(lambda: deque(maxlen=window))

    def _avg(dq):
        return sum(dq) / len(dq) if dq else SOT_PRIOR

    _run_shot_form(df, sot_for, sot_against, _avg)
    return {team: _avg(sot_for[team]) - _avg(sot_against[team]) for team in sot_for}


def _run_shot_form(df, sot_for, sot_against, _avg) -> dict[str, list]:
    """Shared causal SoT loop: emits per-row pre-match columns while mutating
    the caller's deques to the end-of-data state."""
    cols = {k: [] for k in ["home_sot_for", "home_sot_against",
                            "away_sot_for", "away_sot_against", "sot_net_diff"]}
    for r in df.itertuples(index=False):
        hf, ha = _avg(sot_for[r.home]), _avg(sot_against[r.home])
        af, aa = _avg(sot_for[r.away]), _avg(sot_against[r.away])
        cols["home_sot_for"].append(hf)
        cols["home_sot_against"].append(ha)
        cols["away_sot_for"].append(af)
        cols["away_sot_against"].append(aa)
        cols["sot_net_diff"].append((hf - ha) - (af - aa))

        # update AFTER recording (causal); rows with missing shot stats leave
        # the histories untouched rather than poisoning them with NaN
        hst, ast = float(r.HST), float(r.AST)
        if hst == hst and ast == ast:  # not NaN
            sot_for[r.home].append(hst)
            sot_against[r.home].append(ast)
            sot_for[r.away].append(ast)
            sot_against[r.away].append(hst)

    return cols


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
