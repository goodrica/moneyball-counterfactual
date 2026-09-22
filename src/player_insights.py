"""
Phase 6 — Player-level insights.

Who does the OLD (1985-2002 market) model overpay vs. the modern market, and
who does it underpay? For each season in the panel we have, per player:
  - actual salary (modern market price)
  - OLD-model predicted price (rescaled to the season's league payroll)
  - bWAR (observed production)

Definitions:
  overpaid_by_old  = price_old - salary   (old GM would spend MORE than modern market)
  underpaid_by_old = salary - price_old   (old GM would spend LESS; the Moneyball discount)

"Biggest Moneyball losers"  = players the OLD model undervalues most (high salary
or high WAR per old-price dollar: walk men, defensive aces, cheap controllable WAR).
"Biggest old-school marks"  = players the OLD model overvalues most (RBI/HR guys,
save-total closers, high-AVG no-walk hitters the old market chased).

Also computes a value-efficiency view: WAR per $1M of OLD price (what an old GM
would think is a bargain) vs WAR per $1M actual.

Outputs results/player_insights.csv (season-player rows for top/bottom N) and
results/player_insights_summary.csv (career-aggregated winners/losers).

Usage: python src/player_insights.py [--min-pa 200 --min-ip 60 --top 50]
"""
import os, sys, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roster_builder as rb

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
RESULTS = os.path.join(BASE, "results")


def season_prices(year, scale_cache={}):
    bat, pit = rb.build_pool(year)
    scale = rb.scale_old_prices(bat, pit, year)   # in-place multiplies price_old
    bat = bat[["player_ID", "name", "primary_pos", "PA", "WAR", "salary", "price_old"]].copy()
    bat["group"] = "bat"
    pit = pit[["player_ID", "name", "WAR", "salary", "price_old"]].copy()
    pit["primary_pos"] = "P"
    pit["PA"] = np.nan
    pit["group"] = "pit"
    return pd.concat([bat, pit], ignore_index=True), scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2010)
    ap.add_argument("--end", type=int, default=2024)
    ap.add_argument("--min-pa", type=int, default=200)
    ap.add_argument("--min-ip", type=float, default=60.0)
    ap.add_argument("--top", type=int, default=50)
    args = ap.parse_args()

    ipouts_min = args.min_ip * 3
    frames = []
    for year in range(args.start, args.end + 1):
        df, scale = season_prices(year)
        df["year"] = year
        keep = ((df.group == "bat") & (df.PA >= args.min_pa)) | \
               ((df.group == "pit") & (df.WAR.notna()))
        df = df[keep].copy()
        if args.min_ip > 0:
            piti = pd.read_csv(os.path.join(DATA, "war", "bwar_pitch.csv"))
            ip = piti[piti.year_ID == year].groupby("player_ID")["IPouts"].sum()
            df = df[(df.group == "bat") | (df.player_ID.map(ip).fillna(0) >= ipouts_min)]
        df["delta"] = df.price_old - df.salary
        df["pct_old_of_salary"] = df.price_old / df.salary
        frames.append(df)
        print(f"{year}: {len(df)} player-seasons (scale {scale:.2f})", flush=True)

    allp = pd.concat(frames, ignore_index=True)
    allp["war_per_old_M"] = allp.WAR / (allp.price_old / 1e6).clip(lower=0.1)
    allp["war_per_real_M"] = allp.WAR / (allp.salary / 1e6).clip(lower=0.1)

    qual = allp[allp.salary > 0]
    losers = qual.nlargest(args.top, "delta")     # old market would OVERPAY these
    winners = qual.nsmallest(args.top, "delta")   # old market UNDERPAYS (discount)
    losers.assign(kind="old_overpays").to_csv(
        os.path.join(RESULTS, "player_insights_overpay.csv"), index=False)
    winners.assign(kind="old_underpays").to_csv(
        os.path.join(RESULTS, "player_insights_underpay.csv"), index=False)
    qual.to_csv(os.path.join(RESULTS, "player_insights.csv"), index=False)

    print("\n=== Players the OLD market would OVERPAY most (modern GM's marks?) ===")
    print(losers.groupby(["name", "primary_pos"]).delta.sum().nlargest(15)
          .apply(lambda v: f"${v/1e6:+.0f}M").to_string())
    print("\n=== Players the OLD market UNDERPAYS most (the Moneyball discount) ===")
    print(winners.groupby(["name", "primary_pos"]).delta.sum().nsmallest(15)
          .apply(lambda v: f"${v/1e6:+.0f}M").to_string())

    # H1 check: what stats drive the overpayment? correlate delta with
    # RBI-heavy vs BB-heavy profile for batters
    b = qual[(qual.group == "bat")].copy()
    la = pd.read_csv(os.path.join(DATA, "lahman", "Batting.csv"))
    key = la.groupby(["playerID", "yearID"])[["RBI", "BB", "HR"]].sum().reset_index()
    b = b.merge(key, left_on=["player_ID", "year"], right_on=["playerID", "yearID"])
    b = b.dropna(subset=["delta"])
    print("\nH1: corr(OLD overpayment $, stat) across player-seasons:")
    for c in ["RBI", "HR", "BB"]:
        print(f"  {c:4s}: {np.corrcoef(b.delta, b[c])[0,1]:+.3f}")
    print("  RBI-BB gap should be POSITIVE: old model chases RBI, ignores walks")

    # Moneyball test: walk-heavy vs RBI-heavy players with similar WAR —
    # within WAR decile, does BB raise or lower the OLD-model price?
    b["war_bin"] = pd.qcut(b.WAR, 8, duplicates="drop")
    grp = (b.groupby("war_bin", observed=True)
             .apply(lambda g: np.corrcoef(g.BB, g.delta)[0, 1] if len(g) > 20 else np.nan)
             .rename("corr(BB, old-price premium)"))
    print("\nWithin WAR bins, corr(walks, OLD overpayment) — Moneyball says NEGATIVE:")
    print(grp.to_string())
    print(f"\naverage: {grp.mean():+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
