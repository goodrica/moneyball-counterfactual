import pandas as pd, numpy as np
p = pd.read_csv('results/panel_team_season.csv')
c = pd.read_csv('results/control_team_season.csv')
for tag, df in [('PANEL 2010-2024', p), ('CONTROL 1988-1999', c)]:
    print(f"=== {tag}: {len(df)} team-seasons ===")
    print(f"  mean gap NEW-OLD wins: {df.gap_new_minus_old.mean():+.2f}")
    print(f"  gap by year:")
    g = df.groupby('year').gap_new_minus_old.mean()
    print('   ', ' '.join(f"{y}:{v:+.1f}" for y, v in g.items()))
    print(f"  estimator r(act_est, real W) = "
          f"{np.corrcoef(df.actual_wins_est, df.real_wins)[0,1]:.3f}")
    print(f"  mean actual WAR OLD {df.old_war_mean.mean():.1f} | NEW {df.new_war_mean.mean():.1f} | "
          f"actual {df.actual_war.mean():.1f}")
    print()
# H2 check: gap vs payroll (big spenders worse off under OLD?)
p['payroll_m'] = p.payroll / 1e6
p = p.sort_values('payroll_m')
lo = p.nsmallest(60, 'payroll_m'); hi = p.nlargest(60, 'payroll_m')
print("H2 (panel): gap by payroll tercile")
t = p.payroll_m.quantile([1/3, 2/3]).values
print(f"  low  (<${t[0]:.0f}M): {p[p.payroll_m < t[0]].gap_new_minus_old.mean():+.2f}")
print(f"  mid : {p[(p.payroll_m >= t[0]) & (p.payroll_m < t[1])].gap_new_minus_old.mean():+.2f}")
print(f"  high (>${t[1]:.0f}M): {p[p.payroll_m >= t[1]].gap_new_minus_old.mean():+.2f}")
# H4: effect over time
p['half'] = np.where(p.year < 2017, '2010-2016', '2017-2024')
print("\nH4 (panel): gap by half-decade")
print(p.groupby('half').gap_new_minus_old.mean().round(2))
