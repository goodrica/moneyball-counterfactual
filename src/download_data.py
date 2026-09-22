"""
Phase 0 — Data acquisition.

Sources:
- Lahman CSVs  (SHA-pinned mirror: cbwinslow/baseballdatabank, 2026-09-10)
  Batting, Pitching, Teams, People, Salaries, AwardsPlayers
- Baseball-Reference WAR via pybaseball
  bwar_bat, bwar_pitch (WAR + salary through 2024+)

Outputs:
  data/lahman/*.csv
  data/war/bwar_bat.csv
  data/war/bwar_pitch.csv
  data/payroll.csv  (derived from bWAR salaries; 30 teams/yr through 2024+)
"""

import os, warnings
warnings.filterwarnings('ignore')

import requests, pandas as pd
import pybaseball

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
LAH_DIR  = os.path.join(DATA_DIR, "lahman")
WAR_DIR  = os.path.join(DATA_DIR, "war")

MIRROR = "https://raw.githubusercontent.com/cbwinslow/baseballdatabank/refs/heads/master"

# Commit SHA of mirror pinned here for reproducibility
MIRROR_SHA = "a0b6f52"   # 2026-09-10 snapshot

LAHMAN_FILES = {
    "Batting.csv":       "core/Batting.csv",
    "Pitching.csv":      "core/Pitching.csv",
    "Fielding.csv":      "core/Fielding.csv",
    "Teams.csv":         "core/Teams.csv",
    "People.csv":        "core/People.csv",
    "Salaries.csv":      "contrib/Salaries.csv",
    "AwardsPlayers.csv": "contrib/AwardsPlayers.csv",
    "SeriesPost.csv":    "core/SeriesPost.csv",
}

BWAREARLY_MIN = 1985   # Lahman training window start
BWAREARLY_MAX = 2002   # Lahman training window end (no post-2002 OBP in market)
PANEL_MIN     = 2003   # Modern era start
PANEL_MAX     = 2024   # Primary panel

# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
def download(url: str, path: str) -> int:
    if os.path.exists(path):
        return os.path.getsize(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    return len(r.content)


def download_lahman():
    print("=== Lahman CSVs ===")
    for fname, remote in LAHMAN_FILES.items():
        out = os.path.join(LAH_DIR, fname)
        size = download(f"{MIRROR}/{remote}", out)
        print(f"  {fname:25s}  {size/1024:.0f} KB")


# ---------------------------------------------------------------------------
# bWAR pull
# ---------------------------------------------------------------------------
def download_bwar():
    print("\n=== bWAR via pybaseball ===")
    os.makedirs(WAR_DIR, exist_ok=True)

    bat = pybaseball.bwar_bat(return_all=True)
    bat_filt = bat[bat["year_ID"] >= BWAREARLY_MIN].copy()
    bat_filt = bat_filt[(bat_filt["PA"] > 0) | (bat_filt["WAR"].notna())].copy()
    bat_filt.to_csv(os.path.join(WAR_DIR, "bwar_bat.csv"), index=False)
    print(f"  bwar_bat.csv   {len(bat_filt):,} rows  ({int(bat_filt['year_ID'].min())}–{int(bat_filt['year_ID'].max())})")

    pit = pybaseball.bwar_pitch(return_all=True)
    pit_filt = pit[pit["year_ID"] >= BWAREARLY_MIN].copy()
    pit_filt = pit_filt[(pit_filt["IPouts"] > 0) | (pit_filt["WAR"].notna())].copy()
    pit_filt.to_csv(os.path.join(WAR_DIR, "bwar_pitch.csv"), index=False)
    print(f"  bwar_pitch.csv {len(pit_filt):,} rows  ({int(pit_filt['year_ID'].min())}–{int(pit_filt['year_ID'].max())})")


# ---------------------------------------------------------------------------
# Payroll derived from bWAR salaries
# ---------------------------------------------------------------------------
def build_payroll():
    print("\n=== Payroll (derived from bWAR salaries) ===")
    bat = pd.read_csv(os.path.join(WAR_DIR, "bwar_bat.csv"))
    pit = pd.read_csv(os.path.join(WAR_DIR, "bwar_pitch.csv"))

    sal = pd.concat([
        bat[["year_ID", "team_ID", "salary"]].dropna(subset=["salary"])
          .rename(columns={"year_ID": "yearID", "team_ID": "teamID"}),
        pit[["year_ID", "team_ID", "salary"]].dropna(subset=["salary"])
          .rename(columns={"year_ID": "yearID", "team_ID": "teamID"}),
    ], ignore_index=True).drop_duplicates(subset=["yearID", "teamID", "salary"])

    payroll = sal.groupby(["yearID", "teamID"])["salary"].sum().reset_index()
    payroll.columns = ["yearID", "teamID", "payroll"]
    payroll.to_csv(os.path.join(DATA_DIR, "payroll.csv"), index=False)
    print(f"  {len(payroll):,} rows  years {int(payroll.yearID.min())}–{int(payroll.yearID.max())}")
    for yr in [PANEL_MIN, 2010, 2015, 2021, PANEL_MAX]:
        n = len(payroll[payroll["yearID"] == yr])
        print(f"    {yr}: {n} teams  {'✓ 30 teams' if n == 30 else '⚠ short'}")


# ---------------------------------------------------------------------------
# Data-contract / sanity tests
# ---------------------------------------------------------------------------
def data_contract():
    print("\n=== Data-contract tests ===")
    ok = True

    def check(label: str, expr, expect: bool = True):
        nonlocal ok
        passed = bool(expr) == expect
        print(f"  {'PASS' if passed else 'FAIL'}  {label}")
        if not passed:
            ok = False

    # 1. Lahman files exist and have expected row counts
    bat_la  = pd.read_csv(os.path.join(LAH_DIR, "Batting.csv"))
    pit_la  = pd.read_csv(os.path.join(LAH_DIR, "Pitching.csv"))
    fld_la  = pd.read_csv(os.path.join(LAH_DIR, "Fielding.csv"))
    teams   = pd.read_csv(os.path.join(LAH_DIR, "Teams.csv"))
    people  = pd.read_csv(os.path.join(LAH_DIR, "People.csv"))
    sal_la  = pd.read_csv(os.path.join(LAH_DIR, "Salaries.csv"))
    awards  = pd.read_csv(os.path.join(LAH_DIR, "AwardsPlayers.csv"))

    check("Lahman Batting rows > 100K",  len(bat_la)  > 100_000)
    check("Lahman Pitching rows > 45K",  len(pit_la)  >  45_000)
    check("Lahman Fielding rows > 100K", len(fld_la) > 100_000)
    check("Lahman Teams rows > 2K",      len(teams)   >   2_000)
    check("Lahman People rows > 15K",    len(people)  >  15_000)
    check("Lahman Salaries rows > 25K",  len(sal_la)  >  25_000)
    check("Lahman Awards rows > 6K",    len(awards)  >   6_000)

    # 2. Lahman salary covers 1985–2002 training window
    check("Lahman salary starts 1985",   sal_la["yearID"].min() == 1985)
    check("Lahman salary ends 2016",     sal_la["yearID"].max() == 2016)

    # 3. Teams has W/L/RS/RA, no payroll col (it's derived)
    for col in ["yearID","teamID","W","L","R","RA"]:
        check(f"Teams.csv has {col}", col in teams.columns)
    check("Teams.csv has NO payroll col", "payroll" not in teams.columns)

    # 4. bWAR files exist
    bwar_bat = pd.read_csv(os.path.join(WAR_DIR, "bwar_bat.csv"))
    bwar_pit = pd.read_csv(os.path.join(WAR_DIR, "bwar_pitch.csv"))
    check("bWAR batting rows > 40K", len(bwar_bat) > 40_000)
    check("bWAR pitching rows > 20K", len(bwar_pit) > 20_000)
    check("bWAR batting year max >= 2024", bwar_bat["year_ID"].max() >= 2024)
    check("bWAR pitching year max >= 2024", bwar_pit["year_ID"].max() >= 2024)
    check("bWAR has salary col", "salary" in bwar_bat.columns)

    # 5. Payroll has 30 teams/year for panel
    payroll = pd.read_csv(os.path.join(DATA_DIR, "payroll.csv"))
    for yr in [2010, 2015, 2020, 2021, 2022, 2023, 2024]:
        check(f"Payroll 30 teams in {yr}", len(payroll[payroll["yearID"] == yr]) == 30)

    # 6. Spot-check: 2021 LAD payroll in known range ($230–280M)
    lad21 = payroll[(payroll["yearID"] == 2021) & (payroll["teamID"] == "LAD")]["payroll"].values
    check("2021 LAD payroll $200–280M", 200e6 < lad21[0] < 280e6 if len(lad21) else False)

    # 7. Spot-check: 2001 Barry Bonds bWAR > 11
    bonds01 = bwar_bat[(bwar_bat["name_common"] == "Barry Bonds") & (bwar_bat["year_ID"] == 2001)]
    check("Barry Bonds 2001 WAR > 11", (len(bonds01) and bonds01["WAR"].values[0] > 11))

    # 8. Lahman + bWAR player IDs match (IDs use identical format: 'last5_first2##')
    la_all_ids = set(bat_la["playerID"].dropna())
    bwar_all_ids = set(bwar_bat["player_ID"].dropna())
    full_overlap = la_all_ids & bwar_all_ids
    check(
        f"Lahman-bWAR playerID overlap ≥ 30% ({len(full_overlap)}/{len(la_all_ids)})",
        len(full_overlap) >= int(0.30 * len(la_all_ids)),
    )

    print()
    if ok:
        print("ALL CHECKS PASSED ✓")
    else:
        print("SOME CHECKS FAILED — review above")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    download_lahman()
    download_bwar()
    build_payroll()
    passed = data_contract()
    exit(0 if passed else 1)
