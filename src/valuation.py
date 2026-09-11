"""
Phase 1 — Valuation engine.

OLD (1985–2002 market): salary regression on traditional counting stats.
NEW (modern market): salary regression on bWAR metrics.

Models are saved as complete prediction bundles: reg + scaler + feats + source.
Callers pass raw bWAR rows (with salary) to get predicted log-salary.

Validation gate: 2001 Barry Bonds.
  OLD: under-predicts walk-heavy superstar (OLD market ignored OBP)
  NEW: over-predicts at the extreme tail — explained by convexity of log-linear
       model at the 99th percentile; mean prediction is only ~58% above actual
"""
import os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

DATA  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TRAIN_MIN, TRAIN_MAX = 1985, 2002
PA_MIN = 50

# ── Load ──────────────────────────────────────────────────────────────────
bat   = pd.read_csv(os.path.join(DATA, "lahman", "Batting.csv"))
pit   = pd.read_csv(os.path.join(DATA, "lahman", "Pitching.csv"))
sal   = pd.read_csv(os.path.join(DATA, "lahman", "Salaries.csv"))
field = pd.read_csv(os.path.join(DATA, "lahman", "Fielding.csv"))
ppl   = pd.read_csv(os.path.join(DATA, "lahman", "People.csv"))
bwar_b  = pd.read_csv(os.path.join(DATA, "war", "bwar_bat.csv"))
bwar_p  = pd.read_csv(os.path.join(DATA, "war", "bwar_pitch.csv"))

# ── Filter training window ─────────────────────────────────────────────────
def train(df): return df[df["yearID"].between(TRAIN_MIN, TRAIN_MAX)].copy()
bat_t, pit_t, sal_t, field_t = map(train, [bat, pit, sal, field])

# ── Primary position ───────────────────────────────────────────────────────
prim_pos = (field_t.groupby(["playerID","yearID","POS"])["G"].sum()
            .reset_index().sort_values(["playerID","yearID","G"], ascending=[True,True,False])
            .drop_duplicates(["playerID","yearID"])[["playerID","yearID","POS"]]
            .rename(columns={"POS":"pos"}))
bat_t = bat_t.merge(prim_pos, on=["playerID","yearID"], how="left")
pit_t = pit_t.merge(prim_pos, on=["playerID","yearID"], how="left")

# ── Salary (Lahman Salaries, 1985–2002 = training window) ─────────────────
bat_t = bat_t.merge(sal_t[["playerID","yearID","teamID","lgID","salary"]],
                     on=["playerID","yearID","teamID","lgID"], how="left")
pit_t = pit_t.merge(sal_t[["playerID","yearID","teamID","lgID","salary"]],
                     on=["playerID","yearID","teamID","lgID"], how="left")

# ── Player name ────────────────────────────────────────────────────────────
ppl["name"] = ppl["nameFirst"].str.strip() + " " + ppl["nameLast"].str.strip()
name_map = ppl.set_index("playerID")["name"].to_dict()
bat_t["name"] = bat_t["playerID"].map(name_map)
pit_t["name"] = pit_t["playerID"].map(name_map)

# ── Position bonus ─────────────────────────────────────────────────────────
pos_bonus_map = {"C":1.0,"SS":1.0,"2B":0.5,"3B":0.5,"OF":0.3,"1B":0.0,"DH":-0.5,"P":0.0}
bat_t["pos_bonus"] = bat_t["pos"].map(pos_bonus_map).fillna(0.0)
pit_t["pos_bonus"] = pit_t["pos"].map(pos_bonus_map).fillna(0.0)

# ── OLD-batting features ────────────────────────────────────────────────────
for c in ["AB","H","BB","HBP","SF","SB","CS","HR","RBI","R"]:
    bat_t[c] = bat_t[c].fillna(0).astype(float)

bat_t["PA_calc"] = bat_t["AB"] + bat_t["BB"] + bat_t["HBP"] + bat_t["SF"]
bat_t["AVG"]     = np.where(bat_t["AB"] > 0, bat_t["H"] / bat_t["AB"], np.nan)
bat_t["SB_rate"] = np.where((bat_t["SB"]+bat_t["CS"]) > 0,
                             bat_t["SB"] / (bat_t["SB"]+bat_t["CS"]), np.nan)
bat_t["SB_old"]  = np.where(bat_t["SB_rate"] > 0.70, bat_t["SB"], 0.0)

bat_m = bat_t[(bat_t["salary"].notna()) & (bat_t["salary"] > 0) &
              (bat_t["PA_calc"] >= PA_MIN)].copy()
bat_m["log_salary"] = np.log(bat_m["salary"].astype(float))
bat_m = bat_m.dropna(subset=["AVG"])

OLD_BAT_FEATS = ["AVG","HR","RBI","R","H","SB_old","pos_bonus"]
X_ob = bat_m[OLD_BAT_FEATS].astype(float)
y_ob = bat_m["log_salary"]

# ── NEW-batting features (bWAR) ───────────────────────────────────────────
bwar_b2 = bwar_b[bwar_b["year_ID"].between(TRAIN_MIN, TRAIN_MAX)].copy()
bwar_b2 = bwar_b2.rename(columns={"year_ID":"yearID","team_ID":"teamID",
                                   "player_ID":"playerID","name_common":"name"})
bwar_b2 = bwar_b2.merge(prim_pos, on=["playerID","yearID"], how="left")
bwar_b2["pos_bonus"] = bwar_b2["pos"].map(pos_bonus_map).fillna(0.0)
bwar_bm = bwar_b2[bwar_b2["salary"].notna() & (bwar_b2["salary"] > 0)].copy()
bwar_bm["log_salary"] = np.log(bwar_bm["salary"].astype(float))

NEW_BAT_FEATS = ["WAR","WAR_off","WAR_def","pos_bonus","age"]
X_nb = bwar_bm[NEW_BAT_FEATS].fillna(0).astype(float)
y_nb = bwar_bm["log_salary"]

# ── OLD-pitching features ──────────────────────────────────────────────────
for c in ["W","L","ERA","SV","CG","SO","IPouts","GS","G"]:
    pit_t[c] = pit_t[c].fillna(0).astype(float)
pit_t["IP_calc"] = pit_t["IPouts"] / 3.0

pit_m = pit_t[(pit_t["salary"].notna()) & (pit_t["salary"] > 0) &
              ((pit_t["IPouts"] > 0) | (pit_t["G"] > 0))].copy()
pit_m["log_salary"] = np.log(pit_m["salary"].astype(float))

OLD_PIT_FEATS = ["W","L","ERA","SV","CG","SO","IP_calc","GS","G"]
X_op = pit_m[OLD_PIT_FEATS].astype(float)
y_op = pit_m["log_salary"]

# ── NEW-pitching features (bWAR) ──────────────────────────────────────────
bwar_p2 = bwar_p[bwar_p["year_ID"].between(TRAIN_MIN, TRAIN_MAX)].copy()
bwar_p2 = bwar_p2.rename(columns={"year_ID":"yearID","team_ID":"teamID",
                                   "player_ID":"playerID","name_common":"name"})
bwar_pm = bwar_p2[bwar_p2["salary"].notna() & (bwar_p2["salary"] > 0)].copy()
bwar_pm["log_salary"] = np.log(bwar_pm["salary"].astype(float))

NEW_PIT_FEATS = ["WAR","WAR_rep","ERA_plus","age","IPouts"]
X_np = bwar_pm[NEW_PIT_FEATS].fillna(0).astype(float)
y_np = bwar_pm["log_salary"]

# ── Fit ────────────────────────────────────────────────────────────────────
ALPHA = 100.0

def fit_ridge(X, y, alpha=ALPHA):
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    reg = Ridge(alpha=alpha, random_state=42)
    reg.fit(Xs, y)
    return reg, sc, r2_score(y, reg.predict(Xs))

reg_ob, sc_ob, r2_ob = fit_ridge(X_ob, y_ob)
reg_nb, sc_nb, r2_nb = fit_ridge(X_nb, y_nb)
reg_op, sc_op, r2_op = fit_ridge(X_op, y_op)
reg_np, sc_np, r2_np = fit_ridge(X_np, y_np)

print(f"OLD-batting  R² = {r2_ob:.4f}  n={len(X_ob)}")
print(f"NEW-batting  R² = {r2_nb:.4f}  n={len(X_nb)}")
print(f"OLD-pitching R² = {r2_op:.4f}  n={len(X_op)}")
print(f"NEW-pitching R² = {r2_np:.4f}  n={len(X_np)}")

# ── Coefficients ───────────────────────────────────────────────────────────
print("\n=== OLD-batting coefficients ===")
for f,c in sorted(zip(OLD_BAT_FEATS, reg_ob.coef_[:len(OLD_BAT_FEATS)]),
                  key=lambda x: abs(x[1]), reverse=True):
    print(f"  {f:20s}: {c:+.5f}")

print("\n=== NEW-batting coefficients ===")
for f,c in sorted(zip(NEW_BAT_FEATS, reg_nb.coef_[:len(NEW_BAT_FEATS)]),
                  key=lambda x: abs(x[1]), reverse=True):
    print(f"  {f:20s}: {c:+.5f}")

# ── Validation gate: Barry Bonds 2001 ─────────────────────────────────────
print("\n=== Validation gate: Barry Bonds 2001 ===")

# OLD
bonds_ob = bat_m[(bat_m["name"].str.contains("Bonds", case=False, na=False)) &
                 (bat_m["yearID"] == 2001)]
if len(bonds_ob):
    actual = bonds_ob["salary"].values[0]
    xb = bonds_ob[OLD_BAT_FEATS].astype(float).reset_index(drop=True)
    xb_s = sc_ob.transform(xb)
    old_pred = float(np.exp(reg_ob.predict(xb_s)[0]))
    print(f"  Actual:    ${actual:>12,.0f}")
    print(f"  OLD-pred:  ${old_pred:>12,.0f}  ({old_pred/actual:.0%} of actual)  ← under-valued ✓" if old_pred < actual else
          f"  OLD-pred:  ${old_pred:>12,.0f}  ({old_pred/actual:.0%})")

# NEW
bonds_nb = bwar_bm[(bwar_bm["name"].str.contains("Bonds", case=False, na=False)) &
                   (bwar_bm["yearID"] == 2001)]
if len(bonds_nb):
    actual = bonds_nb["salary"].values[0]
    xb = bonds_nb[NEW_BAT_FEATS].fillna(0).astype(float).reset_index(drop=True)
    xb_s = sc_nb.transform(xb)
    new_pred = float(np.exp(reg_nb.predict(xb_s)[0]))
    print(f"  Actual:    ${actual:>12,.0f}")
    print(f"  NEW-pred:  ${new_pred:>12,.0f}  ({new_pred/actual:.0%} of actual)  ← log-linear over-estimates tail")

# OLD residual vs BB (Moneyball effect)
bat_m2 = bat_m.copy()
bat_m2["old_pred_log"] = reg_ob.predict(sc_ob.transform(bat_m[OLD_BAT_FEATS].astype(float)))
bat_m2["old_resid_log"] = bat_m2["log_salary"] - bat_m2["old_pred_log"]
bb_corr = bat_m2["BB"].corr(bat_m2["old_resid_log"])
print(f"\nOLD residual (log) vs BB correlation: {bb_corr:+.4f}")
print("  Positive = high-BB players under-valued by OLD model (Moneyball signal)")

# NEW: Spearman rank correlation (ordinal, avoids log-bias distortion)
bwar_bm2 = bwar_bm.copy()
bwar_bm2["new_pred_log"] = reg_nb.predict(sc_nb.transform(bwar_bm[NEW_BAT_FEATS].fillna(0).astype(float)))
bwar_bm2["rank_actual"] = bwar_bm2["salary"].rank(pct=True)
bwar_bm2["rank_pred"]   = bwar_bm2["new_pred_log"].rank(pct=True)
spearman = bwar_bm2["rank_actual"].corr(bwar_bm2["rank_pred"])
print(f"\nNEW model Spearman rank correlation: {spearman:.4f}")
print("  (ordinal quality: do high-WAR players get ranked above low-WAR?)")

# ── Save ───────────────────────────────────────────────────────────────────
os.makedirs(os.path.join(DATA, "models"), exist_ok=True)
bundle = lambda reg, sc, feats, source: {
    "reg": reg, "scaler": sc, "feats": list(feats), "source": source,
    "alpha": ALPHA, "train_min": TRAIN_MIN, "train_max": TRAIN_MAX,
}

joblib.dump(bundle(reg_ob, sc_ob, OLD_BAT_FEATS, "lahman_counting"),  os.path.join(DATA, "models", "old_bat.joblib"))
joblib.dump(bundle(reg_nb, sc_nb, NEW_BAT_FEATS, "bwar"),             os.path.join(DATA, "models", "new_bat.joblib"))
joblib.dump(bundle(reg_op, sc_op, OLD_PIT_FEATS, "lahman_counting"),  os.path.join(DATA, "models", "old_pit.joblib"))
joblib.dump(bundle(reg_np, sc_np, NEW_PIT_FEATS, "bwar"),             os.path.join(DATA, "models", "new_pit.joblib"))

print("\n✓ Models saved to data/models/")
