import glob, os
import pandas as pd, numpy as np
for p in sorted(glob.glob('results/panel_year_*.csv')):
    if 'rosters' in p: continue
    df = pd.read_csv(p)
    y = int(os.path.basename(p).split('_')[-1].split('.')[0])
    ok = df.dropna(subset=['actual_wins_est','real_wins'])
    r = np.corrcoef(ok.actual_wins_est, ok.real_wins)[0,1] if len(ok)>5 else np.nan
    print(f"{y}: n={len(df):2d} gap NEW-OLD {df.gap_new_minus_old.mean():+.2f}  "
          f"OLD {df.old_wins_mean.mean():.1f} NEW {df.new_wins_mean.mean():.1f} "
          f"ACTest {df.actual_wins_est.mean():.1f} REAL {df.real_wins.mean():.1f} r(est,real) {r:.2f}")
