# /home/sam069/projects/SportyDataFeeder/app/services/dixon_coles.py
#
# Dixon-Coles bivariate-Poisson goal model for football (Phase C of
# reports/OUTCOME_MODEL_TRAINING_PLAN.md). Each team has an attack and a
# defence parameter; goals are Poisson with the Dixon-Coles low-score
# correction (which fixes the well-known under-count of 0-0/1-0/0-1/1-1).
# Match outcome probabilities (H/D/A) are obtained by summing the score-line
# probability matrix.
#
#   log(lambda_home) = home_adv + attack[home] + defence[away]
#   log(mu_away)     =           attack[away] + defence[home]
#
# Training uses exponential time-decay weighting (recent matches count more)
# and light L2 regularisation (stabilises low-data / newly promoted teams).
# Fitting is causal — callers pass only matches strictly before the target.

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

MAX_GOALS = 10  # score matrix truncation (P(>10 goals) is negligible)


def _poisson_pmf(k: np.ndarray, lam: float) -> np.ndarray:
    from math import lgamma
    # vectorised Poisson pmf for k = 0..MAX_GOALS
    logp = -lam + k * np.log(lam) - np.array([lgamma(int(i) + 1) for i in k])
    return np.exp(logp)


def _tau(x, y, lam, mu, rho):
    """Dixon-Coles low-score dependency correction."""
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    if x == 0 and y == 1:
        return 1.0 + lam * rho
    if x == 1 and y == 0:
        return 1.0 + mu * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


@dataclass
class DixonColes:
    decay_half_life_days: float = 180.0
    l2: float = 0.01
    teams: list[str] = field(default_factory=list)
    index: dict[str, int] = field(default_factory=dict)
    attack: np.ndarray | None = None
    defence: np.ndarray | None = None
    home_adv: float = 0.25
    rho: float = -0.05

    # ---------------------------------------------------------------- fitting
    def fit(self, matches: list[dict], as_of) -> "DixonColes":
        """matches: list of {home, away, fthg, ftag, date}. `as_of` is the
        reference date for time-decay weighting (the test match/season start)."""
        teams = sorted({m["home"] for m in matches} | {m["away"] for m in matches})
        self.teams = teams
        self.index = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        from math import lgamma
        hi = np.array([self.index[m["home"]] for m in matches])
        ai = np.array([self.index[m["away"]] for m in matches])
        hg = np.array([m["fthg"] for m in matches], float)
        ag = np.array([m["ftag"] for m in matches], float)
        xi = np.log(2) / self.decay_half_life_days
        age = np.array([(as_of - m["date"]).days for m in matches], float)
        w = np.exp(-xi * np.clip(age, 0, None))

        # Precompute constants that don't depend on params.
        lg_hg = np.array([lgamma(g + 1) for g in hg])
        lg_ag = np.array([lgamma(g + 1) for g in ag])
        low = np.where((hg <= 1) & (ag <= 1))[0]  # only these need tau correction
        low_x = hg[low].astype(int)
        low_y = ag[low].astype(int)

        # params: [attack(n), defence(n), home_adv, rho]
        def unpack(p):
            return p[:n], p[n:2 * n], p[2 * n], p[2 * n + 1]

        def nll(p):
            att, dfn, hadv, rho = unpack(p)
            lam = np.exp(hadv + att[hi] + dfn[ai])
            mu = np.exp(att[ai] + dfn[hi])
            log_ph = -lam + hg * np.log(lam) - lg_hg
            log_pa = -mu + ag * np.log(mu) - lg_ag
            log_tau = np.zeros(len(matches))
            for k, idx in enumerate(low):
                t = _tau(int(low_x[k]), int(low_y[k]), lam[idx], mu[idx], rho)
                log_tau[idx] = np.log(max(t, 1e-10))
            ll = w * (log_tau + log_ph + log_pa)
            penalty = self.l2 * (np.sum(att ** 2) + np.sum(dfn ** 2))
            return -np.sum(ll) + penalty

        x0 = np.zeros(2 * n + 2)
        x0[2 * n] = 0.25     # home_adv
        x0[2 * n + 1] = -0.05  # rho
        bounds = [(-3, 3)] * (2 * n) + [(-1, 1), (-0.2, 0.2)]
        res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 500})
        att, dfn, hadv, rho = unpack(res.x)
        self.attack, self.defence, self.home_adv, self.rho = att, dfn, hadv, rho
        return self

    # ------------------------------------------------------------- prediction
    def _rates(self, home: str, away: str) -> tuple[float, float]:
        ah = self.attack[self.index[home]] if home in self.index else 0.0
        dh = self.defence[self.index[home]] if home in self.index else 0.0
        aa = self.attack[self.index[away]] if away in self.index else 0.0
        da = self.defence[self.index[away]] if away in self.index else 0.0
        lam = np.exp(self.home_adv + ah + da)
        mu = np.exp(aa + dh)
        return float(lam), float(mu)

    def predict_proba(self, home: str, away: str) -> dict:
        """Return {'H':p,'D':p,'A':p} and expected goals (lam, mu)."""
        lam, mu = self._rates(home, away)
        ks = np.arange(MAX_GOALS + 1)
        ph = _poisson_pmf(ks, lam)
        pa = _poisson_pmf(ks, mu)
        mat = np.outer(ph, pa)
        # apply tau to the 2x2 low-score block
        for x in (0, 1):
            for y in (0, 1):
                mat[x, y] *= _tau(x, y, lam, mu, self.rho)
        mat = np.clip(mat, 0, None)
        mat /= mat.sum()
        home_win = np.tril(mat, -1).sum()   # x > y
        away_win = np.triu(mat, 1).sum()    # y > x
        draw = np.trace(mat)
        return {"H": float(home_win), "D": float(draw), "A": float(away_win),
                "lambda_home": lam, "mu_away": mu}
