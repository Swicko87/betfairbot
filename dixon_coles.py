"""
dixon_coles.py
==============
A pre-match Dixon-Coles goals model + value-betting backtest, built to answer ONE
question before you sink weeks into this: is there any edge in a basic model
against the market's own odds?

You do NOT need your recorder or any live data for this. Free historical results
*with* bookmaker odds are downloadable today from football-data.co.uk (one CSV per
league per season, e.g. E0 = English Premier League). Point this at one or more of
those CSVs and it will:
  1. Fit a Dixon-Coles model  - team attack/defence strengths + home advantage,
     with the low-score correction and exponential time-decay weighting.
  2. Turn it into 1X2 probabilities per match (home / draw / away).
  3. Backtest flat-stake value betting against the odds in the file, after
     commission, out-of-sample, with a rolling refit.

HONEST EXPECTATION: against closing odds - especially a sharp book like Pinnacle -
a basic model usually does NOT show a durable edge. The market is very good. A flat
or slightly negative ROI here is a valuable, money-saving finding, not a failure.
Edge, if it exists, tends to hide in early prices, lower/less-liquid leagues, and
narrow niches - things worth chasing only if this baseline looks promising.

Data columns expected (football-data.co.uk format, comma-separated):
  Date, HomeTeam, AwayTeam, FTHG, FTAG          (results - required)
  Benchmark 1X2 odds, tried in order:
    PSH/PSD/PSA (Pinnacle) -> B365H/D/A (Bet365) -> AvgH/AvgD/AvgA (market average)

Usage:
  pip install pandas numpy scipy
  python dixon_coles.py E0_2223.csv E0_2324.csv E0_2425.csv
"""

import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

# ---------------------------- knobs ----------------------------
MAX_GOALS = 10           # scoreline grid 0..MAX_GOALS for each team
XI = 0.0018              # time-decay per day (~1-year half-life). 0 disables decay
TRAIN_FRAC = 0.5         # fit on earliest 50% of matches, bet on the rest
REFIT_EVERY_DAYS = 30    # rolling refit cadence during the betting period
COMMISSION = 0.02        # Betfair commission on net winnings
EDGE_THRESHOLD = 0.05    # only back when model says >5% value at the offered odds
STAKE = 1.0              # flat stake unit

ODDS_SETS = [("PSH", "PSD", "PSA"), ("B365H", "B365D", "B365A"), ("AvgH", "AvgD", "AvgA")]


def load(paths):
    frames = [pd.read_csv(p, encoding="latin-1") for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    df[["FTHG", "FTAG"]] = df[["FTHG", "FTAG"]].astype(int)
    # pick the first odds set fully present in the file
    for h, d, a in ODDS_SETS:
        if all(c in df.columns for c in (h, d, a)):
            df = df.dropna(subset=[h, d, a])
            df = df.rename(columns={h: "oH", d: "oD", a: "oA"})
            print(f"Using benchmark odds: {h}/{d}/{a}")
            break
    else:
        raise SystemExit("No usable 1X2 odds columns found in the CSV(s).")
    return df.sort_values("Date").reset_index(drop=True)


def _tau(hg, ag, lam, mu, rho):
    """Dixon-Coles low-score dependency correction (vectorised)."""
    t = np.ones_like(lam, dtype=float)
    t = np.where((hg == 0) & (ag == 0), 1.0 - lam * mu * rho, t)
    t = np.where((hg == 0) & (ag == 1), 1.0 + lam * rho, t)
    t = np.where((hg == 1) & (ag == 0), 1.0 + mu * rho, t)
    t = np.where((hg == 1) & (ag == 1), 1.0 - rho, t)
    return t


class DixonColes:
    def __init__(self, teams):
        self.teams = list(teams)
        self.idx = {t: i for i, t in enumerate(self.teams)}
        self.n = len(self.teams)
        self.params = None

    def _unpack(self, p):
        # attack/defence use n-1 free params each; last team = -sum (sum-to-zero)
        att = np.empty(self.n)
        dfc = np.empty(self.n)
        att[:-1] = p[: self.n - 1]
        att[-1] = -att[:-1].sum()
        dfc[:-1] = p[self.n - 1 : 2 * (self.n - 1)]
        dfc[-1] = -dfc[:-1].sum()
        home = p[-2]
        rho = p[-1]
        return att, dfc, home, rho

    def _nll(self, p, hi, ai, hg, ag, w):
        att, dfc, home, rho = self._unpack(p)
        lam = np.exp(home + att[hi] + dfc[ai])   # home goals rate
        mu = np.exp(att[ai] + dfc[hi])           # away goals rate
        lam = np.clip(lam, 1e-6, 25)
        mu = np.clip(mu, 1e-6, 25)
        tau = _tau(hg, ag, lam, mu, rho)
        tau = np.clip(tau, 1e-9, None)           # keep log finite
        ll = np.log(tau) + poisson.logpmf(hg, lam) + poisson.logpmf(ag, mu)
        return -np.sum(w * ll)

    def fit(self, df, ref_date=None):
        hi = df["HomeTeam"].map(self.idx).to_numpy()
        ai = df["AwayTeam"].map(self.idx).to_numpy()
        hg = df["FTHG"].to_numpy()
        ag = df["FTAG"].to_numpy()
        if ref_date is not None and XI > 0:
            age = (ref_date - df["Date"]).dt.days.to_numpy()
            w = np.exp(-XI * np.clip(age, 0, None))
        else:
            w = np.ones(len(df))
        x0 = np.concatenate([np.zeros(2 * (self.n - 1)), [0.25, -0.05]])
        res = minimize(self._nll, x0, args=(hi, ai, hg, ag, w),
                       method="L-BFGS-B", options={"maxiter": 200})
        self.params = res.x
        return self

    def match_probs(self, home, away):
        """Return (P_home, P_draw, P_away) for one fixture."""
        att, dfc, hadv, rho = self._unpack(self.params)
        i, j = self.idx[home], self.idx[away]
        lam = np.clip(np.exp(hadv + att[i] + dfc[j]), 1e-6, 25)
        mu = np.clip(np.exp(att[j] + dfc[i]), 1e-6, 25)
        gh = poisson.pmf(np.arange(MAX_GOALS + 1), lam)
        ga = poisson.pmf(np.arange(MAX_GOALS + 1), mu)
        M = np.outer(gh, ga)
        # apply DC correction to the four low-score cells
        M[0, 0] *= 1.0 - lam * mu * rho
        M[0, 1] *= 1.0 + lam * rho
        M[1, 0] *= 1.0 + mu * rho
        M[1, 1] *= 1.0 - rho
        M = np.clip(M, 0, None)
        M /= M.sum()
        ph = np.tril(M, -1).sum()   # home goals > away goals
        pa = np.triu(M, 1).sum()    # away goals > home goals
        pd_ = np.trace(M)           # draw
        return ph, pd_, pa


def backtest(df):
    cut = df["Date"].quantile(TRAIN_FRAC)
    bet_df = df[df["Date"] > cut].reset_index(drop=True)
    teams = pd.unique(pd.concat([df["HomeTeam"], df["AwayTeam"]]))
    model = DixonColes(teams)

    last_fit = None
    staked = pnl = bets = wins = 0.0
    for _, m in bet_df.iterrows():
        # rolling refit on everything strictly before this match
        if last_fit is None or (m["Date"] - last_fit).days >= REFIT_EVERY_DAYS:
            train = df[df["Date"] < m["Date"]]
            model.fit(train, ref_date=m["Date"])
            last_fit = m["Date"]

        if m["HomeTeam"] not in model.idx or m["AwayTeam"] not in model.idx:
            continue
        probs = model.match_probs(m["HomeTeam"], m["AwayTeam"])
        odds = (m["oH"], m["oD"], m["oA"])
        result = 0 if m["FTHG"] > m["FTAG"] else (2 if m["FTHG"] < m["FTAG"] else 1)

        for k in range(3):  # 0=home 1=draw 2=away
            value = probs[k] * odds[k] - 1.0          # model EV per unit at offered price
            if value > EDGE_THRESHOLD:
                staked += STAKE
                bets += 1
                if result == k:
                    pnl += (odds[k] - 1.0) * STAKE * (1 - COMMISSION)
                    wins += 1
                else:
                    pnl -= STAKE

    print("\n" + "=" * 50)
    print(" DIXON-COLES VALUE BACKTEST (out-of-sample)")
    print("=" * 50)
    print(f" matches in bet period : {len(bet_df)}")
    print(f" value bets placed     : {int(bets)}")
    if bets:
        print(f" strike rate           : {wins/bets:.1%}")
        print(f" total staked          : {staked:.0f} units")
        print(f" net P&L (after comm)  : {pnl:+.1f} units")
        print(f" ROI                   : {pnl/staked:+.1%}")
    else:
        print(" no bets cleared the edge threshold")
    print("=" * 50)
    print(" Positive, stable ROI across many bets = worth pursuing.")
    print(" Flat/negative = the market's odds already beat this model.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python dixon_coles.py <league_season.csv> [more.csv ...]")
    backtest(load(sys.argv[1:]))
