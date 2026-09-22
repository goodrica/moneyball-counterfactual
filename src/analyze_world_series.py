"""
Phase 7 — World Series winners vs. what each GM philosophy would have built.

Question: for WS winners 1990-1999, were they actually the teams a MODERN GM
would have assembled under their budget — or did "old-school" valuations
happen to endorse them? A winner that a modern rebuild would NOT have picked
(poor NEW-arm rank) but an old GM WOULD have (good OLD-arm rank) is direct
evidence that old-school stats can win while modern "on paper" teams fail.

Method:
  - Uses the control panel (1988-1999) per-season mean estimated wins from
    the exclusive-draft rosters (results/control_year_YYYY.csv) plus the
    actual-roster bWAR and estimated wins already in the panel.
  - For each season: rank all teams by (a) actual bWAR, (b) NEW-arm est wins,
    (c) OLD-arm est wins. Compare the WS winner's rank under each lens.
  - Note: in the control era both arms price the same market the teams
    actually played in, so arms ≈ re-drafts, not philosophy tests. The
    interesting column is rank(actual bWAR) — did the "best on paper" team
    win? — and how far the rebuild arms' rankings diverge from the trophy.

Output: results/world_series_1990s.csv + printed table.

Usage: python src/analyze_world_series.py [--start 1990 --end 1999]
"""
import os, sys, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
RESULTS = os.path.join(BASE, "results")

LAH_TO_BWAR = {"CHA": "CHW", "CHN": "CHC", "FLA": "MIA", "KCA": "KCR",
               "LAN": "LAD", "NYA": "NYY", "NYN": "NYM", "SDN": "SDP",
               "SFN": "SFG", "SLN": "STL", "TBA": "TBR", "WAS": "WSN"}


def ws_winners(start, end):
    sp = pd.read_csv(os.path.join(DATA, "lahman", "SeriesPost.csv"))
    ws = sp[(sp["round"] == "WS") & sp["yearID"].between(start, end)].copy()
    ws["teamIDwinner"] = ws["teamIDwinner"].replace(LAH_TO_BWAR)
    ws["teamIDloser"] = ws["teamIDloser"].replace(LAH_TO_BWAR)
    # one row per WS: winner + loser columns (no mirrored row)
    out = ws[["yearID", "teamIDwinner", "teamIDloser"]].rename(
        columns={"teamIDwinner": "champ", "teamIDloser": "loser"})
    return out


def season_table(year):
    for pat in (f"control_year_{year}.csv", f"panel_year_{year}.csv"):
        p = os.path.join(RESULTS, pat)
        if os.path.exists(p):
            return pd.read_csv(p)
    raise FileNotFoundError(f"no panel file for {year}; run run_experiment first")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1990)
    ap.add_argument("--end", type=int, default=1999)
    args = ap.parse_args()

    winners = ws_winners(args.start, args.end)
    rows = []
    for _, w in winners.iterrows():
        year, champ, runner = int(w["yearID"]), w["champ"], w["loser"]
        df = season_table(year)
        df = df.sort_values("actual_war", ascending=False).reset_index(drop=True)
        df["rank_war"] = np.arange(1, len(df) + 1)
        df["rank_new"] = df["new_wins_mean"].rank(ascending=False)
        df["rank_old"] = df["old_wins_mean"].rank(ascending=False)

        def get(team, col):
            m = df[df["team"] == team]
            return float(m[col].iloc[0]) if len(m) else np.nan

        row = {"year": year, "champ": champ, "runner_up": runner,
               "champ_wins": get(champ, "real_wins"),
               "champ_actual_war": get(champ, "actual_war"),
               "champ_rank_actual_war": get(champ, "rank_war"),
               "champ_rank_new_arm": get(champ, "rank_new"),
               "champ_rank_old_arm": get(champ, "rank_old"),
               "best_on_paper_team": df.iloc[0]["team"],
               "best_on_paper_war": df.iloc[0]["actual_war"],
               "new_arm_best_team": df.loc[df["new_wins_mean"].idxmax(), "team"],
               "old_arm_best_team": df.loc[df["old_wins_mean"].idxmax(), "team"]}
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "world_series_1990s.csv"), index=False)

    print(out[["year", "champ", "champ_wins", "champ_rank_actual_war",
               "champ_rank_new_arm", "champ_rank_old_arm",
               "best_on_paper_team"]].to_string(index=False,
               formatters={"champ_rank_actual_war": "{:.0f}".format,
                           "champ_rank_new_arm": "{:.0f}".format,
                           "champ_rank_old_arm": "{:.0f}".format}))
    n = len(out)
    agree_paper = (out["champ_rank_actual_war"] <= 3).sum()
    print(f"\nChamps ranked top-3 by actual bWAR: {agree_paper}/{n}")
    print(f"Champs the NEW-arm would have built top-3: "
          f"{(out['champ_rank_new_arm'] <= 3).sum()}/{n}")
    print(f"Champs the OLD-arm would have built top-3: "
          f"{(out['champ_rank_old_arm'] <= 3).sum()}/{n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
