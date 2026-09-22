"""
Phase 3 — Outcome estimator.

Converts a 26-man roster's stat line into an estimated win total:

  batters : linear-weights runs scored
            runs = events + outs * OUT_RUNS   (OUT_RUNS calibrated 2015: -0.08)
  pitchers: basic-FIP runs allowed
            FIP = (13*HR + 3*BB - 2*SO)/IP + C; runs = FIP * IP/9
            (C = 3.15 puts FIP on the ERA scale)
  wins    : Pythagenpat  win% = RS^x / (RS^x + RA^x), x = ((RS+RA)/9)^0.287

Validation gate (plan Phase 3): applied to ACTUAL 2015 rosters (top-13 PA batters,
top-13 IP pitchers, staff scaled to full-season 1458 IP), predicted wins must
correlate with real W-L at r >= 0.85 across all 30 teams.

Pure functions; no I/O in the estimators.
"""
import numpy as np
import pandas as pd

# Linear-weights event coefficients (wOBA-style fixed run values)
LW = {
    "1B": 0.47, "2B": 0.75, "3B": 1.03, "HR": 1.40,
    "BB": 0.33, "HBP": 0.38, "SB": 0.20, "CS": -0.40,
}
OUT_RUNS = -0.08           # per batting out, calibrated on 2015 team lines
FIP_C = 3.15               # FIP constant (ERA-scale alignment)
FULL_SEASON_IPOUTS = 4374.0  # 162 games × 9 IP × 3 outs

# Environment-alignment: put estimator output on the same league scale as real
# runs for the season being estimated. Defaults calibrated on 2015; for other
# seasons use calibrate_season() to recompute (league-total alignment).
RS_CAL = 0.920
RA_CAL = 1.097


def calibrate_season(bat_all, pit_all, real_R_mean, real_RA_mean, games=162.0, n_teams=30.0):
    """Return (rs_cal, ra_cal) aligning estimator means to real league means.

    bat_all: all hitter rows for the season (league-wide), pit_all: all pitchers.
    This uses only aggregate stat data (descriptive, not outcome-fitting).
    n_teams: teams in that season's league (28 pre-1993, 30 from 1998).
    """
    n_teams = float(n_teams)
    rs_mean = float(batter_runs(bat_all).sum()) / n_teams
    ip_outs_total = float(pit_all["IPouts"].sum())
    if ip_outs_total <= 0:
        return RS_CAL, RA_CAL
    # per-team full-season outs, then FIP runs at that scale
    team_outs = ip_outs_total / n_teams            # one team's staff outs (all its pitchers)
    fip_total = float(pitcher_fip_runs(pit_all).sum())
    fip_per_team_full = fip_total / n_teams * (27.0 * games / team_outs)
    ra_mean = fip_per_team_full
    rs_cal = real_R_mean / rs_mean if rs_mean > 0 else RS_CAL
    ra_cal = real_RA_mean / ra_mean if ra_mean > 0 else RA_CAL
    return rs_cal, ra_cal


def batter_runs(df):
    """Linear-weights runs created. df needs: 1B 2B 3B HR BB HBP SB CS AB H SF SH."""
    events = (df["1B"] * LW["1B"] + df["2B"] * LW["2B"] + df["3B"] * LW["3B"]
              + df["HR"] * LW["HR"] + df["BB"].fillna(0) * LW["BB"]
              + df["HBP"].fillna(0) * LW["HBP"]
              + df["SB"].fillna(0) * LW["SB"]
              + df["CS"].fillna(0) * LW["CS"])
    outs = (df["AB"] - df["H"]).fillna(0) + df["SF"].fillna(0) + df["SH"].fillna(0)
    return events + outs * OUT_RUNS


def pitcher_fip_runs(df):
    """FIP runs allowed. df needs: SO BB HR IPouts."""
    ip = (df["IPouts"] / 3.0).replace(0, np.nan)
    fip = ((13 * df["HR"] + 3 * df["BB"].fillna(0) - 2 * df["SO"].fillna(0))
           / ip + FIP_C)
    return (fip * ip / 9.0).fillna(0.0)


def pythagenpat(rs, ra, games=162):
    """Pythagenpat expected wins."""
    rs = np.asarray(rs, dtype=float)
    ra = np.asarray(ra, dtype=float)
    x = np.power((rs + ra) / np.maximum(games, 1), 0.287)
    win_pct = np.power(rs, x) / (np.power(rs, x) + np.power(ra, x))
    return win_pct * games


def roster_runs(roster_bat):
    """Team runs scored from a rebuilt batting roster (calibrated)."""
    return float(batter_runs(roster_bat).sum()) * RS_CAL


def roster_runs_allowed(roster_pit, ip_outs_col="IPouts"):
    """Team runs allowed, scaled to a full 1458-IP season."""
    raw = float(pitcher_fip_runs(roster_pit).sum())
    ip_outs = float(roster_pit[ip_outs_col].sum())
    if ip_outs <= 0:
        return raw * RA_CAL
    return raw * (FULL_SEASON_IPOUTS / ip_outs) * RA_CAL


def team_wins_from_stats(rs, ra):
    """Pythagenpat wins from team RS/RA."""
    return float(pythagenpat(rs, ra, games=162))


def team_wins(roster_bat, roster_pit):
    rs = roster_runs(roster_bat)
    ra = roster_runs_allowed(roster_pit)
    return team_wins_from_stats(rs, ra)
