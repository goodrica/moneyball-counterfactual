"""
Phase 4 — Experiment panel runner (exclusive-draft design).

Phase 2's pilot ran each team's optimizer independently over the full league
pool — every team drafted the same stars (the 2015 pilot's ~40% overlap and
140-win averages were artifacts). Phase 4 fixes this: each seed is a
LEAGUE-WIDE allocation. Teams draft in randomized order from a shared,
depleting pool, each maximizing observed bWAR subject to its real payroll
and the arm's price schedule (OLD prices vs NEW=actual salaries).

Per-seed rosters are scored with the Phase-3 estimator (per-season
calibration; identical machinery for OLD, NEW, and Actual arms), and results
are reported as MEAN ± STD across seeds — never a single seed.

Outputs (results/):
  panel_year_YYYY.csv        per-team-season summary (wins mean/std, WAR,
                             overlap, real W)
  panel_year_YYYY_rosters.csv  one full 26-man roster per team-arm
                             (the median-wins seed) for Phase-5 diffs
  panel_team_season.csv      concat of the panel
  control_team_season.csv    era-backtest control (1988-1995)

Usage:
    python src/run_experiment.py --panel 2010-2024 --seeds 100
    python src/run_experiment.py --panel 1988-1995 --seeds 100 --control
"""
import os, sys, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roster_builder as rb
import outcomes as oc

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
RESULTS = os.path.join(BASE, "results")

# Lahman teamIDs differ from bWAR/MLB.com IDs for many franchises, and some
# aliases are era-dependent (Marlins: FLO in Lahman 1993-2011, but bWAR says
# FLA through 2011 and MIA from 2012; Brewers ML4 1998 -> MIL). We resolve
# per-year by matching against the payroll (bWAR) team-ID set.
TEAM_ALIASES = {
    "CHA": {"CHW"}, "CHN": {"CHC"}, "KCA": {"KCR"}, "LAN": {"LAD"},
    "NYA": {"NYY"}, "NYN": {"NYM"}, "SDN": {"SDP"}, "SFN": {"SFG"},
    "SLN": {"STL"}, "WAS": {"WSN"},
    "FLO": {"MIA", "FLA"}, "ML4": {"MIL"},
    # era pairs: lahman TBA/TBD/ANA/CAL vs bWAR TBD/TBR/ANA/LAA
    "TBA": {"TBR", "TBD"}, "TBD": {"TBD", "TBR"},
    "ANA": {"ANA", "LAA"}, "CAL": {"ANA", "LAA"}, "ARI": {"ARI"},
}


def resolve_team_ids(lah_ids, target_ids):
    """Map each Lahman teamID to the ID used in target (bWAR) set for that year."""
    tset = set(target_ids)
    mapping = {}
    for lid in set(lah_ids):
        if lid in tset:
            mapping[lid] = lid
            continue
        for alt in TEAM_ALIASES.get(lid, ()):
            if alt in tset:
                mapping[lid] = alt
                break
        else:
            mapping[lid] = lid  # unmatched; will surface as NaN
    return mapping


def _teams_bwar_ids(year):
    import pandas as _pd
    t = _pd.read_csv(os.path.join(DATA, "lahman", "Teams.csv"))
    t = t[t["yearID"] == year].copy()
    p = _pd.read_csv(os.path.join(DATA, "payroll.csv"))
    p = p[p["yearID"] == year]["teamID"].unique()
    t["teamID"] = t["teamID"].map(resolve_team_ids(t["teamID"], p))
    return t.set_index("teamID")


def _player_stat_vectors(year, bat, pit):
    """Precompute per-player estimator contributions (vectorized scoring)."""
    bat_lah = pd.read_csv(os.path.join(DATA, "lahman", "Batting.csv"))
    pit_lah = pd.read_csv(os.path.join(DATA, "lahman", "Pitching.csv"))
    by = bat_lah[bat_lah["yearID"] == year]
    agg = (by.groupby("playerID")[["AB", "H", "2B", "3B", "HR", "BB",
                                   "HBP", "SF", "SH", "SB", "CS"]]
           .sum(min_count=1).fillna(0.0).reset_index())
    agg["1B"] = agg["H"] - agg["2B"] - agg["3B"] - agg["HR"]
    runs = oc.batter_runs(agg)
    bmap = dict(zip(agg["playerID"], runs))
    bat_runs = np.array([bmap.get(p, 0.0) for p in bat["player_ID"]])

    py = pit_lah[pit_lah["yearID"] == year]
    pagg = (py.groupby("playerID")[["SO", "BB", "HR", "IPouts"]]
            .sum(min_count=1).fillna(0.0).reset_index())
    for c in ["SO", "BB", "HR", "IPouts"]:
        pagg[c] = pd.to_numeric(pagg[c], errors="coerce").fillna(0.0)
    fra = oc.pitcher_fip_runs(pagg)
    pmap_r = dict(zip(pagg["playerID"], fra))
    pmap_i = dict(zip(pagg["playerID"], pagg["IPouts"]))
    pit_runs = np.array([pmap_r.get(p, 0.0) for p in pit["player_ID"]])
    pit_ipouts = np.array([pmap_i.get(p, 0.0) for p in pit["player_ID"]])
    return bat_runs, pit_runs, pit_ipouts


def run_season(year, seeds=100, pay=None, force=False):
    out_path = os.path.join(RESULTS, f"panel_year_{year}.csv")
    ros_path = os.path.join(RESULTS, f"panel_year_{year}_rosters.csv")
    if os.path.exists(out_path) and not force:
        print(f"[skip] {year}: cached")
        return pd.read_csv(out_path)

    print(f"[panel] {year}: building pool...", flush=True)
    bat, pit = rb.build_pool(year)
    rb.scale_old_prices(bat, pit, year)

    if pay is None:
        p = pd.read_csv(os.path.join(DATA, "payroll.csv"))
        pay = p[p["yearID"] == year].set_index("teamID")["payroll"]
    teams = sorted(pay.index)
    act_b, act_p = rb.actual_rosters(year)

    teams_y = _teams_bwar_ids(year)
    games = float(teams_y["G"].mean())
    full_outs = games * 27.0

    # per-season calibration
    cal_bat = pd.read_csv(os.path.join(DATA, "lahman", "Batting.csv"))
    cal_pit = pd.read_csv(os.path.join(DATA, "lahman", "Pitching.csv"))
    byc = cal_bat[cal_bat["yearID"] == year]
    byc = (byc.groupby("playerID")[["AB", "H", "2B", "3B", "HR", "BB",
                                    "HBP", "SF", "SH", "SB", "CS"]]
           .sum(min_count=1).fillna(0.0).reset_index())
    byc["1B"] = byc["H"] - byc["2B"] - byc["3B"] - byc["HR"]
    byc["PA"] = byc[["AB", "BB", "HBP", "SF"]].sum(axis=1)
    pyc = cal_pit[cal_pit["yearID"] == year]
    pyc = (pyc.groupby("playerID")[["SO", "BB", "HR", "IPouts"]]
           .sum(min_count=1).fillna(0.0))
    for c in ["SO", "BB", "HR", "IPouts"]:
        pyc[c] = pd.to_numeric(pyc[c], errors="coerce").fillna(0.0)
    rs_cal, ra_cal = oc.calibrate_season(
        byc[byc["PA"] >= 1], pyc[pyc["IPouts"] >= 1],
        teams_y["R"].mean(), teams_y["RA"].mean(),
        games=teams_y["G"].mean(), n_teams=float(len(teams_y)))

    bat_runs, pit_runs, pit_ipouts = _player_stat_vectors(year, bat, pit)

    def wins_for(cb, cp):
        rs = bat_runs[list(cb)].sum() * rs_cal
        ip_outs = pit_ipouts[list(cp)].sum()
        raw_ra = pit_runs[list(cp)].sum()
        ra = (raw_ra * (full_outs / ip_outs) if ip_outs > 0 else raw_ra) * ra_cal
        w = oc.team_wins_from_stats(rs, ra)
        return w * games / 162.0 if games < 162 else w

    # actual-roster estimates through the same machinery
    id2bi = {p: i for i, p in enumerate(bat["player_ID"])}
    id2pi = {p: i for i, p in enumerate(pit["player_ID"])}
    actual_wins, actual_war = {}, {}
    for team in teams:
        cb = [id2bi[p] for p in act_b.loc[act_b["team_ID"] == team, "player_ID"] if p in id2bi]
        cp = [id2pi[p] for p in act_p.loc[act_p["team_ID"] == team, "player_ID"] if p in id2pi]
        actual_wins[team] = wins_for(cb, cp) if cb and cp else np.nan
        bwar = act_b.loc[act_b["team_ID"] == team, "WAR"].sum()
        pwar = act_p.loc[act_p["team_ID"] == team, "WAR"].sum()
        actual_war[team] = float(bwar + pwar)

    n_team, n_seed = len(teams), seeds
    acc = {arm: {t: {"wins": [], "war": []} for t in teams} for arm in rb.ARMS}
    chosen_store = {arm: {} for arm in rb.ARMS}   # median-wins seed per team

    for arm in rb.ARMS:
        price_col = f"price_{arm}"
        for s in range(n_seed):
            rng = np.random.default_rng(rb._stable_hash(year, arm, "draft", s))
            # draft order weighted by payroll: richest teams pick first on
            # average, but each seed reshuffles with noise
            noise = rng.normal(0, 0.35, len(teams))
            pw = np.array([float(pay[t]) for t in teams])
            rank = np.argsort(-(pw / pw.mean() + noise))
            order = [teams[i] for i in rank]
            free_b = np.ones(len(bat), dtype=bool)
            free_p = np.ones(len(pit), dtype=bool)
            for team in order:
                budget = float(pay[team])
                cb, cp, tw = rb.optimize(bat, pit, budget,
                                         np.random.default_rng(rb._stable_hash(team, arm, year, s)),
                                         price_col, free_b=free_b, free_p=free_p)
                free_b[list(cb)] = False
                free_p[list(cp)] = False
                w = wins_for(cb, cp)
                acc[arm][team]["wins"].append(w)
                acc[arm][team]["war"].append(tw)
                chosen_store[arm].setdefault(team, []).append((w, s, cb, cp))

    rows = []
    for team in teams:
        rec = {"year": year, "team": team, "payroll": float(pay[team]),
               "real_wins": float(teams_y.loc[team, "W"]) if team in teams_y.index else np.nan,
               "actual_wins_est": float(actual_wins[team]),
               "actual_war": actual_war[team]}
        for arm in rb.ARMS:
            wins = np.array(acc[arm][team]["wins"])
            wars = np.array(acc[arm][team]["war"])
            rec[f"{arm}_wins_mean"] = float(wins.mean())
            rec[f"{arm}_wins_std"] = float(wins.std())
            rec[f"{arm}_war_mean"] = float(wars.mean())
            med = min(chosen_store[arm][team], key=lambda z: abs(z[0] - wins.mean()))
            _, s, cb, cp = med
            rb_ids = (set(bat.iloc[list(cb)]["player_ID"]) |
                      set(pit.iloc[list(cp)]["player_ID"]))
            act_ids = (set(act_b.loc[act_b["team_ID"] == team, "player_ID"]) |
                       set(act_p.loc[act_p["team_ID"] == team, "player_ID"]))
            rec[f"{arm}_overlap"] = len(rb_ids & act_ids) / 26.0
        rec["gap_new_minus_old"] = rec["new_wins_mean"] - rec["old_wins_mean"]
        rows.append(rec)
        print(f"  {team}: real {rec['real_wins']:.0f} | est(act) {rec['actual_wins_est']:.1f} | "
              f"OLD {rec['old_wins_mean']:.1f}±{rec['old_wins_std']:.1f} | "
              f"NEW {rec['new_wins_mean']:.1f}±{rec['new_wins_std']:.1f} | "
              f"gap {rec['gap_new_minus_old']:+.1f}", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)

    # median-seed rosters for Phase-5 diffs
    ros_rows = []
    for arm in rb.ARMS:
        for team in teams:
            wins = np.array(acc[arm][team]["wins"])
            med = min(chosen_store[arm][team], key=lambda z: abs(z[0] - wins.mean()))
            _, s, cb, cp = med
            for i in cb:
                r = bat.iloc[i]
                ros_rows.append({"year": year, "arm": arm, "team": team, "seed": s,
                                 "player_ID": r["player_ID"], "name": r["name"],
                                 "pos": r["primary_pos"], "WAR": r["WAR"],
                                 "price": r[f"price_{arm}"], "actual_salary": r["salary"]})
            for i in cp:
                r = pit.iloc[i]
                ros_rows.append({"year": year, "arm": arm, "team": team, "seed": s,
                                 "player_ID": r["player_ID"], "name": r["name"],
                                 "pos": "P", "WAR": r["WAR"],
                                 "price": r[f"price_{arm}"], "actual_salary": r["salary"]})
    pd.DataFrame(ros_rows).to_csv(ros_path, index=False)
    df.to_csv(out_path, index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="2010-2024")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-run even if per-year cache exists")
    args = ap.parse_args()

    y0, y1 = map(int, args.panel.split("-"))
    for year in range(y0, y1 + 1):
        try:
            run_season(year, args.seeds, force=args.force)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[ERROR] {year}: {e}", flush=True)
    # merge cached per-year files, filtered to this run's range, so re-running
    # a subset never truncates the panel (control and panel ranges are disjoint)
    import glob as _glob
    tag = "control" if args.control else "panel"
    files = sorted(f for f in _glob.glob(os.path.join(RESULTS, "panel_year_[0-9]*.csv"))
                   if "_rosters" not in f)
    if not files:
        return 1
    all_y = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    panel = all_y[all_y.year.between(y0, y1)]
    if panel.empty:
        return 1
    panel.to_csv(os.path.join(RESULTS, f"{tag}_team_season.csv"), index=False)

    print(f"\n=== {tag} summary ({y0}-{y1}, {len(panel)} team-seasons) ===")
    print(f"mean wins gap NEW-OLD: {panel['gap_new_minus_old'].mean():+.2f} "
          f"(across-team std {panel['gap_new_minus_old'].std():.2f})")
    ok = panel.dropna(subset=["actual_wins_est", "real_wins"])
    if len(ok) > 5:
        r_ = np.corrcoef(ok["actual_wins_est"], ok["real_wins"])[0, 1]
        print(f"estimator r(actual_est, real W) = {r_:.3f}")
    print(f"mean NEW-arm roster overlap with actual: {panel['new_overlap'].mean():.1%}")
    print(f"mean OLD-arm roster overlap with actual: {panel['old_overlap'].mean():.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
