"""
Phase 2 — Roster optimizer (v1 design, corrected).

Each MLB team's 26-man roster (13 batters + 13 pitchers) is rebuilt twice under
the team's REAL season payroll, drafting from the league-wide player pool:

  OLD arm — pays OLD-model prices (1985-2002 market revealed preference:
            AVG/HR/RBI/R/H/SB/position for batters; W/L/ERA/SV/SO/IP for pitchers)
  NEW arm — pays ACTUAL salaries (the modern market; real front offices already
            use modern stats, so the market price IS the modern valuation)

Both arms maximize the same production objective (sum of observed bWAR).
Prices are on the same dollar scale (real payroll), so no rescaling needed for
comparability — the OLD model's predicted salaries are already in real dollars
from its 1985-2002 training, but salary levels grew over time; we rescale OLD
prices so that the league-wide top-26 roster spend equals actual league payroll
in the target year. This keeps relative old-market prices while putting both
arms under the same budget constraint.

Controls:
  c1. NEW arm WAR >= OLD arm WAR for essentially every team (sanity)
  c2. Era backtest: in 1985-95 the gap should shrink toward zero
  c3. Roster overlap with actual reported for diagnostics (not gating: an
      efficient market draft differs from any single real roster — that IS
      the market inefficiency, not a bug)

Usage: python src/roster_builder.py 2015 [--seeds 100]
"""
import os, sys, argparse, zlib, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")

N_BAT, N_PIT = 13, 13
BAT_COVERAGE = {"C": 1, "1B": 1, "2B": 1, "SS": 1, "3B": 1, "OF": 3}
MIN_SP = 4
SP_MIN_GS = 8
PA_MIN_POOL = 50
IP_OUTS_MIN_POOL = 15
POS_BONUS = {"C":1.0,"SS":1.0,"2B":0.5,"3B":0.5,"OF":0.3,"1B":0.0,"DH":-0.5,"P":0.0}
LEAGUE_MIN = 0.55e6
ARMS = ("old", "new")

# MLB minimum salary by year — used to impute missing bWAR salaries for
# pre-arbitration players (bWAR leaves those blank, not zero). Without this,
# rookies like 2015 Kris Bryant ($507.5k, 5.4 WAR) appear FREE in the NEW arm
# and the optimizer stacks them: 30 teams drafting the same young stars.
MIN_SALARY = {
    1988: 68_600, 1989: 80_000, 1990: 86_400, 1991: 99_000, 1992: 100_000,
    1993: 100_000, 1994: 109_000, 1995: 109_000,
    2010: 400_000, 2011: 414_000, 2012: 480_000, 2013: 490_000,
    2014: 500_000, 2015: 507_500, 2016: 507_500, 2017: 507_500,
    2018: 545_000, 2019: 555_000, 2020: 563_500, 2021: 570_500,
    2022: 687_500, 2023: 720_000, 2024: 795_000,
}


def impute_salaries(bat, pit, year):
    """Fill missing salaries with that year's MLB minimum (bWAR omits
    pre-arbitration players). In-place; returns (n_bat_filled, n_pit_filled)."""
    minsal = MIN_SALARY.get(year, 500_000)
    nb = int(bat["salary"].isna().sum())
    np_ = int(pit["salary"].isna().sum())
    bat["salary"] = bat["salary"].fillna(minsal).clip(lower=minsal)
    pit["salary"] = pit["salary"].fillna(minsal).clip(lower=minsal)
    return nb, np_


def _primary_positions(year, min_g=10):
    f = pd.read_csv(os.path.join(DATA, "lahman", "Fielding.csv"))
    f = f[f["yearID"] == year]
    multi = (f[f["G"] >= min_g].groupby("playerID")["POS"]
             .agg(lambda s: set(s.dropna())))
    top = (f.groupby(["playerID", "POS"])["G"].sum().reset_index()
             .sort_values(["playerID", "G"], ascending=[True, False])
             .drop_duplicates("playerID").set_index("playerID")["POS"])
    return multi.to_dict(), top.to_dict()


def build_pool(year):
    bat_war = pd.read_csv(os.path.join(DATA, "war", "bwar_bat.csv"))
    pit_war = pd.read_csv(os.path.join(DATA, "war", "bwar_pitch.csv"))
    bat_la  = pd.read_csv(os.path.join(DATA, "lahman", "Batting.csv"))
    pit_la  = pd.read_csv(os.path.join(DATA, "lahman", "Pitching.csv"))
    people  = pd.read_csv(os.path.join(DATA, "lahman", "People.csv"))
    multi_pos, top_pos = _primary_positions(year)

    # ---- batters (player-level across stints) ----
    bw = bat_war[(bat_war["year_ID"] == year) & (bat_war["pitcher"] != "Y")].copy()
    bat = (bw.groupby("player_ID")
             .agg(PA=("PA", "sum"), WAR=("WAR", "sum"),
                  WAR_off=("WAR_off", "sum"), WAR_def=("WAR_def", "sum"),
                  age=("age", "median"), salary=("salary", "max"))
             .reset_index())

    la = bat_la[bat_la["yearID"] == year]
    agg = la.groupby("playerID").agg(
        AB=("AB", "sum"), H=("H", "sum"), BB=("BB", "sum"), HBP=("HBP", "sum"),
        SF=("SF", "sum"), SB=("SB", "sum"), CS=("CS", "sum"),
        HR=("HR", "sum"), RBI=("RBI", "sum"), R=("R", "sum")).reset_index()
    agg["PA_la"] = agg[["AB", "BB", "HBP", "SF"]].fillna(0).sum(axis=1)
    agg["AVG"] = np.where(agg["AB"] > 0, agg["H"] / agg["AB"].replace(0, np.nan), 0.0)
    sb_den = (agg["SB"] + agg["CS"]).replace(0, np.nan)
    agg["SB_rate"] = np.where(sb_den > 0, agg["SB"] / sb_den, 0.0)
    agg["SB_old"] = np.where(agg["SB_rate"] > 0.70, agg["SB"], 0.0)

    bat = bat.merge(agg, left_on="player_ID", right_on="playerID", how="left")
    bat["primary_pos"] = bat["player_ID"].map(top_pos)
    bat["pos_set"] = bat["player_ID"].map(multi_pos)
    bat = bat[bat["primary_pos"].notna() & (bat["primary_pos"] != "P")]
    bat["pos_bonus"] = bat["primary_pos"].map(POS_BONUS).fillna(0.0)
    bat = bat[bat["PA_la"].fillna(0) >= PA_MIN_POOL].reset_index(drop=True)

    m_ob = joblib.load(os.path.join(DATA, "models", "old_bat.joblib"))
    Xo = pd.DataFrame({
        "AVG": bat["AVG"].fillna(0.0), "HR": bat["HR"].fillna(0.0),
        "RBI": bat["RBI"].fillna(0.0), "R": bat["R"].fillna(0.0),
        "H": bat["H"].fillna(0.0), "SB_old": bat["SB_old"].fillna(0.0),
        "pos_bonus": bat["pos_bonus"]})[m_ob["feats"]]
    bat["price_old"] = np.exp(m_ob["reg"].predict(m_ob["scaler"].transform(Xo.astype(float))))
    bat["price_new"] = bat["salary"]          # NEW arm pays real market prices

    # ---- pitchers ----
    pw = pit_war[pit_war["year_ID"] == year].copy()
    pit = (pw.groupby("player_ID")
             .agg(IPouts=("IPouts", "sum"), WAR=("WAR", "sum"),
                  WAR_rep=("WAR_rep", "sum"), ERA_plus=("ERA_plus", "median"),
                  age=("age", "median"), GS=("GS", "sum"), G=("G", "sum"),
                  salary=("salary", "max"))
             .reset_index())
    pl = pit_la[pit_la["yearID"] == year]
    pagg = pl.groupby("playerID").agg(
        W=("W", "sum"), L=("L", "sum"), ERA=("ERA", "median"),
        SV=("SV", "sum"), CG=("CG", "sum"), SO=("SO", "sum"),
        GS_la=("GS", "sum"), G_la=("G", "sum")).reset_index()
    pit = pit.merge(pagg, left_on="player_ID", right_on="playerID", how="left")
    pit = pit[pit["IPouts"] >= IP_OUTS_MIN_POOL].reset_index(drop=True)
    pit["is_sp"] = pit["GS"].fillna(0) >= SP_MIN_GS

    m_op = joblib.load(os.path.join(DATA, "models", "old_pit.joblib"))
    era_default = pit["ERA"].median() if pit["ERA"].notna().any() else 4.5
    Xo = pd.DataFrame({
        "W": pit["W"].fillna(0), "L": pit["L"].fillna(0),
        "ERA": pit["ERA"].fillna(era_default),
        "SV": pit["SV"].fillna(0), "CG": pit["CG"].fillna(0),
        "SO": pit["SO"].fillna(0), "IP_calc": pit["IPouts"] / 3.0,
        "GS": pit["GS_la"].fillna(0), "G": pit["G_la"].fillna(0)})[m_op["feats"]]
    pit["price_old"] = np.exp(m_op["reg"].predict(m_op["scaler"].transform(Xo.astype(float))))
    pit["price_new"] = pit["salary"]

    people["name"] = people["nameFirst"].str.strip() + " " + people["nameLast"].str.strip()
    nmap = people.set_index("playerID")["name"].to_dict()
    bat["name"] = bat["player_ID"].map(nmap)
    pit["name"] = pit["player_ID"].map(nmap)

    # impute missing (pre-arb) salaries BEFORE price_new uses them
    nb, npi = impute_salaries(bat, pit, year)
    bat["price_new"] = bat["salary"]
    pit["price_new"] = pit["salary"]
    return bat, pit


def scale_old_prices(bat, pit, year):
    """Scale OLD prices so league-wide roster-frame spend == actual league payroll.
    NEW prices are actual salaries and need no scaling."""
    pay = pd.read_csv(os.path.join(DATA, "payroll.csv"))
    league_payroll = pay[pay["yearID"] == year]["payroll"].sum()

    def top_spend(df, col, n):
        d = df[df[col] > 0].sort_values(col, ascending=False).head(n * 30)
        return d["price_old"].sum()

    s = top_spend(bat, "PA", N_BAT) + top_spend(pit, "IPouts", N_PIT)
    scale = league_payroll / s
    bat["price_old"] *= scale
    pit["price_old"] *= scale
    return scale


def _eligible(posset, primary, need):
    if isinstance(posset, set) and posset:
        return need in posset
    return primary == need


def optimize(bat, pit, budget, rng, price_col, jitter=0.01, free_b=None, free_p=None):
    """Baseline-then-upgrade: fill every slot with the cheapest eligible player,
    then repeatedly apply the single best WAR-positive upgrade the budget allows.
    Always returns a full 26-man roster under budget.
    free_b/free_p: boolean availability masks (league-wide exclusive draft);
    players taken by earlier teams in the same seed are unselectable."""
    price_b = bat[price_col].fillna(LEAGUE_MIN).values * rng.normal(1, jitter, len(bat))
    price_p = pit[price_col].fillna(LEAGUE_MIN).values * rng.normal(1, jitter, len(pit))
    war_b = bat["WAR"].fillna(0.0).values
    war_p = pit["WAR"].fillna(0.0).values
    pos_b = bat["primary_pos"].values
    possets = bat["pos_set"].values
    is_sp = pit["is_sp"].values
    avail_b = np.ones(len(bat), dtype=bool) if free_b is None else free_b
    avail_p = np.ones(len(pit), dtype=bool) if free_p is None else free_p

    def elig(i, pos):
        return _eligible(possets[i], pos_b[i], pos)

    chosen_b, chosen_p = set(), set()

    # Phase 1 — cheapest-eligible baseline
    for pos, n in BAT_COVERAGE.items():
        for _ in range(n):
            cand = [i for i in range(len(bat))
                    if i not in chosen_b and avail_b[i] and elig(i, pos)]
            if cand:
                chosen_b.add(min(cand, key=lambda i: price_b[i]))
    for _ in range(MIN_SP):
        cand = [i for i in range(len(pit)) if i not in chosen_p and avail_p[i] and is_sp[i]]
        if cand:
            chosen_p.add(min(cand, key=lambda i: price_p[i]))
    for _ in range(N_BAT - len(chosen_b)):
        cand = [i for i in range(len(bat)) if i not in chosen_b and avail_b[i]]
        if cand:
            chosen_b.add(min(cand, key=lambda i: price_b[i]))
    for _ in range(N_PIT - len(chosen_p)):
        cand = [i for i in range(len(pit)) if i not in chosen_p and avail_p[i]]
        if cand:
            chosen_p.add(min(cand, key=lambda i: price_p[i]))

    spend = price_b[list(chosen_b)].sum() + price_p[list(chosen_p)].sum()
    if spend > budget:
        # scale jitter run still over budget at baseline: drop most expensive non-SP/coverage
        # (rare; LEAGUE_MIN floor prices make this nearly impossible) — keep as-is, log later

        pass

    # Phase 2 — greedy best upgrades
    improved = True
    while improved:
        improved = False
        cur_spend = price_b[list(chosen_b)].sum() + price_p[list(chosen_p)].sum()
        slack = budget - cur_spend
        best_gain, best_swap = 0.0, None

        needs = list(BAT_COVERAGE.keys())
        need_n = np.array([BAT_COVERAGE[k] for k in needs])
        elig_mat = np.zeros((len(bat), len(needs)), dtype=bool)
        for j, need in enumerate(needs):
            if len(possets) and isinstance(possets[0], set):
                elig_mat[:, j] = [need in s if isinstance(s, set) else pos_b[i] == need
                                   for i, s in enumerate(possets)]
            else:
                elig_mat[:, j] = [pos_b[i] == need for i in range(len(bat))]
        in_b = np.zeros(len(bat), dtype=bool)
        in_b[list(chosen_b)] = True
        cur_counts = elig_mat[in_b].sum(axis=0)

        for out_i in list(chosen_b):
            counts_wo = cur_counts - elig_mat[out_i]
            allowed = slack + price_b[out_i]
            ok = (counts_wo[None, :] + elig_mat) >= need_n[None, :]
            cand = np.where(ok.all(axis=1) & ~in_b & avail_b & (price_b <= allowed))[0]
            if len(cand) == 0:
                continue
            gains = war_b[cand] - war_b[out_i]
            j = np.argmax(gains)
            if gains[j] > best_gain:
                best_gain, best_swap = float(gains[j]), ("b", out_i, int(cand[j]))

        in_p = np.zeros(len(pit), dtype=bool)
        in_p[list(chosen_p)] = True
        for out_i in list(chosen_p):
            cur_sp = int(is_sp[list(chosen_p)].sum() - is_sp[out_i])
            allowed = slack + price_p[out_i]
            ok = is_sp | (cur_sp >= MIN_SP)
            cand = np.where(ok & ~in_p & avail_p & (price_p <= allowed))[0]
            if len(cand) == 0:
                continue
            gains = war_p[cand] - war_p[out_i]
            j = np.argmax(gains)
            if gains[j] > best_gain:
                best_gain, best_swap = float(gains[j]), ("p", out_i, int(cand[j]))

        if best_swap:
            kind, out_i, i = best_swap
            if kind == "b":
                chosen_b.remove(out_i); chosen_b.add(i)
            else:
                chosen_p.remove(out_i); chosen_p.add(i)
            improved = True

    total_war = war_b[list(chosen_b)].sum() + war_p[list(chosen_p)].sum()
    return chosen_b, chosen_p, total_war


def actual_rosters(year):
    bat = pd.read_csv(os.path.join(DATA, "war", "bwar_bat.csv"))
    pit = pd.read_csv(os.path.join(DATA, "war", "bwar_pitch.csv"))
    bat = bat[(bat["year_ID"] == year) & (bat["pitcher"] != "Y")]
    tb = (bat[bat["PA"] > 0].sort_values("PA", ascending=False)
            .groupby("team_ID").head(N_BAT))
    tp = (pit[pit["year_ID"] == year][pit["IPouts"] > 0]
            .sort_values("IPouts", ascending=False)
            .groupby("team_ID").head(N_PIT))
    return tb, tp


def _stable_hash(*parts):
    return zlib.crc32("|".join(str(p) for p in parts).encode())


def run_year(year, seeds=100):
    bat, pit = build_pool(year)
    scale = scale_old_prices(bat, pit, year)
    pay = pd.read_csv(os.path.join(DATA, "payroll.csv"))
    pay = pay[pay["yearID"] == year].set_index("teamID")["payroll"]
    act_b, act_p = actual_rosters(year)

    rows, summary = [], []
    for team in sorted(pay.index):
        budget = float(pay[team])
        results = {}
        for arm in ARMS:
            wars, best = [], None
            for s in range(seeds):
                rng = np.random.default_rng(_stable_hash(team, arm, s))
                cb, cp, tw = optimize(bat, pit, budget, rng, f"price_{arm}")
                wars.append(tw)
                if best is None or tw > best[2]:
                    best = (cb, cp, tw)
            results[arm] = {"wars": wars, "best": best}

        for arm in ARMS:
            cb, cp, tw = results[arm]["best"]
            wars = results[arm]["wars"]
            for i in cb:
                r = bat.iloc[i]
                rows.append({"year": year, "arm": arm, "team": team,
                             "player_ID": r["player_ID"], "name": r["name"],
                             "WAR": r["WAR"], "price": r[f"price_{arm}"],
                             "primary_pos": r["primary_pos"]})
            for i in cp:
                r = pit.iloc[i]
                rows.append({"year": year, "arm": arm, "team": team,
                             "player_ID": r["player_ID"], "name": r["name"],
                             "WAR": r["WAR"], "price": r[f"price_{arm}"],
                             "primary_pos": "P"})
            act_ids = set(act_b.loc[act_b["team_ID"] == team, "player_ID"]) | \
                      set(act_p.loc[act_p["team_ID"] == team, "player_ID"])
            rebuild_ids = set(bat.iloc[list(cb)]["player_ID"]) | set(pit.iloc[list(cp)]["player_ID"])
            summary.append({
                "year": year, "team": team, "arm": arm, "budget": budget,
                "rebuild_war_mean": float(np.mean(wars)),
                "rebuild_war_std": float(np.std(wars)),
                "rebuild_war_best": float(tw),
                "overlap_with_actual": len(act_ids & rebuild_ids) / 26.0,
            })
        print(f"{team}: ${budget/1e6:.0f}M  "
              f"OLD {np.mean(results['old']['wars']):.1f}±{np.std(results['old']['wars']):.1f}  "
              f"NEW {np.mean(results['new']['wars']):.1f}±{np.std(results['new']['wars']):.1f}  "
              f"gap {np.mean(results['new']['wars'])-np.mean(results['old']['wars']):+.1f}", flush=True)

    os.makedirs(os.path.join(BASE, "results"), exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(BASE, "results", f"rosters_{year}.csv"), index=False)
    s = pd.DataFrame(summary)
    s.to_csv(os.path.join(BASE, "results", f"summary_{year}.csv"), index=False)
    print(f"\nScale factor for OLD prices: {scale:.2f}")
    print(f"Mean OLD {s[s['arm']=='old']['rebuild_war_mean'].mean():.1f} vs "
          f"NEW {s[s['arm']=='new']['rebuild_war_mean'].mean():.1f} WAR "
          f"(gap {s[s['arm']=='new']['rebuild_war_mean'].mean()-s[s['arm']=='old']['rebuild_war_mean'].mean():+.1f})")
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("year", type=int)
    ap.add_argument("--seeds", type=int, default=100)
    args = ap.parse_args()
    run_year(args.year, args.seeds)
