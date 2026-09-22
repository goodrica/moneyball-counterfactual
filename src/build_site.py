"""
Phase 5 — Static site builder.

Reads:
  results/panel_team_season.csv     (2010-2024 arms comparison)
  results/control_team_season.csv   (1988-1999 era backtest)
  results/panel_year_YYYY_rosters.csv (median-seed rosters per arm)
  results/world_series_1990s.csv    (Phase 7, if present)

Writes site/index.html (single file, JSON inlined — no server, no fetch,
works from file:// and GitHub Pages) + site/data/*.json for reuse.

Charts are dependency-free SVG generated in JS from the inlined data.

Usage: python src/build_site.py
"""
import os, sys, json, glob, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "results")
SITE = os.path.join(BASE, "site")


def load_frames():
    d = {}
    for key, f in [("panel", "panel_team_season.csv"),
                   ("control", "control_team_season.csv"),
                   ("ws", "world_series_1990s.csv")]:
        p = os.path.join(RESULTS, f)
        d[key] = pd.read_csv(p) if os.path.exists(p) else None
    return d


def build_payload(fr):
    panel, control, ws = fr["panel"], fr["control"], fr["ws"]
    os.makedirs(SITE, exist_ok=True)
    os.makedirs(os.path.join(SITE, "data"), exist_ok=True)

    def tidy(df):
        out = {}
        for _, r in df.iterrows():
            y = int(r.year)
            out.setdefault(y, []).append({
                "team": r.team,
                "payroll_m": round(r.payroll / 1e6, 1),
                "real": None if pd.isna(r.real_wins) else int(r.real_wins),
                "act_est": round(r.actual_wins_est, 1),
                "old": round(r.old_wins_mean, 1), "old_sd": round(r.old_wins_std, 1),
                "new": round(r.new_wins_mean, 1), "new_sd": round(r.new_wins_std, 1),
                "gap": round(r.gap_new_minus_old, 1),
            })
        return out

    payload = {
        "panel": tidy(panel) if panel is not None else {},
        "control": tidy(control) if control is not None else {},
        "ws": ws.to_dict(orient="records") if ws is not None else [],
        "summary": {},
    }
    if panel is not None:
        ok = panel.dropna(subset=["actual_wins_est", "real_wins"])
        payload["summary"] = {
            "seasons": sorted(int(y) for y in panel.year.unique()),
            "mean_gap": round(float(panel.gap_new_minus_old.mean()), 2),
            "r_est_real": round(float(np.corrcoef(ok.actual_wins_est, ok.real_wins)[0, 1]), 3) if len(ok) > 5 else None,
            "old_war_mean": round(float(panel.old_war_mean.mean()), 1),
            "new_war_mean": round(float(panel.new_war_mean.mean()), 1),
            "actual_war_mean": round(float(panel.actual_war.mean()), 1),
        }
    json.dump(payload, open(os.path.join(SITE, "data", "payload.json"), "w"))
    return payload


HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Moneyball Counterfactual — What if every MLB team hired a 1995 GM?</title>
<style>
:root{--bg:#0e1116;--card:#171c24;--tx:#dbe2ea;--dim:#8a94a3;--acc:#5aa7ff;--old:#ffb454;--new:#5aa7ff;--act:#7ee08a}
*{box-sizing:border-box}body{margin:0;font:15px/1.55 system-ui,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--tx)}
.wrap{max-width:1080px;margin:0 auto;padding:28px 18px 80px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:36px 0 10px}
.sub{color:var(--dim);margin-bottom:24px}
.hero{background:linear-gradient(135deg,#1b2330,#141a23);border:1px solid #232d3b;border-radius:12px;padding:22px;margin-bottom:8px}
.big{font-size:44px;font-weight:700;color:var(--acc)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-top:14px}
.kpi{background:var(--card);border:1px solid #222c39;border-radius:10px;padding:12px 14px}
.kpi b{font-size:20px}.kpi span{display:block;color:var(--dim);font-size:12.5px}
select{background:var(--card);color:var(--tx);border:1px solid #2c3a4c;border-radius:8px;padding:6px 10px;font-size:15px}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{padding:6px 8px;border-bottom:1px solid #1f2733;text-align:right}th{color:var(--dim);font-weight:600}
td:first-child,th:first-child{text-align:left}
.pos{color:#7ee08a}.neg{color:#ff7b72}
.legend{font-size:12.5px;color:var(--dim);margin:8px 0}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin:0 4px 0 12px}
footer{margin-top:44px;color:var(--dim);font-size:12.5px}
a{color:var(--acc)}
</style></head><body><div class="wrap">
<div class="hero">
<h1>Moneyball Counterfactual</h1>
<div class="sub">What if every MLB team hired a 1995 GM? Each season, all 30 rosters are re-drafted twice under one shared league-wide pool — once paying 1985–2002 old-school market prices, once paying modern market salaries — then scored with the same runs-estimator + Pythagenpat machinery.</div>
<div>Headline finding: <span class="big" id="hl">…</span> wins average difference per team between a modern-rebuild and an old-school rebuild</div>
<div class="grid" id="kpis"></div>
</div>

<h2>League view — rebuilt wins vs reality</h2>
<div class="legend"><span class="dot" style="background:var(--act)"></span>actual est. <span class="dot" style="background:var(--new)"></span>NEW-GM rebuild <span class="dot" style="background:var(--old)"></span>OLD-GM rebuild (bars = ±1 seed std)</div>
<select id="ysel"></select>
<div id="chart"></div>

<h2>Season explorer</h2>
<select id="ysel2"></select>
<div id="tbl"></div>

<h2>Era backtest (control)</h2>
<div class="sub" id="ctl-note"></div>
<div id="ctl"></div>

<h2>World Series winners 1990–1999 — were they the "best" team?</h2>
<div id="ws"></div>

<h2>Methodology</h2>
<div class="sub" style="max-width:860px">
OLD prices: ridge regression on log salary using traditional stats (AVG, HR, RBI, R, H, SB≥70%, position bonus) fit on 1985–2002 player-seasons — the pre-Moneyball market's revealed preferences.
NEW arm pays actual salaries (the modern market). Prices are in real dollars; OLD prices are rescaled per season so the league-wide top-26 spend equals actual league payroll.
Each seed: teams draft in payroll-weighted random order from an exclusive league-wide pool (13 batters + 13 pitchers, position coverage, min 4 SP), maximizing observed bWAR under budget; 100 seeds; means reported, error bars ±1 seed-std.
Outcomes: linear-weights runs scored + basic-FIP runs allowed (per-season aggregate calibration), Pythagenpat wins. All three arms run through identical machinery so estimator bias cancels.
Controls: backtest on 1988–1999 (both arms price the same era's market — gap ≈ 0 expected); H1–H4 pre-registered hypotheses.
Known caveats: the NEW arm is a market-repricing of the same player pool, not a counterfactual draft class (no trades/development); salaries come from one source (Baseball-Reference) and double as both price target and budget.
</div>
<footer>Data: Lahman 2025 (CRAN) + Baseball-Reference WAR via pybaseball · code: <a href="https://github.com/goodrica/moneyball-counterfactual">moneyball-counterfactual</a> · regenerated by build_site.py</footer>
</div>
<script>
const P = __PAYLOAD__;
const fmt=n=>n==null?"—":n.toFixed(1);
function kpis(){
  const s=P.summary; document.getElementById('hl').textContent=(s.mean_gap>0?"+":"")+s.mean_gap;
  const items=[["seasons",s.seasons.length+" team-seasons: "+s.seasons[0]+"–"+s.seasons[s.seasons.length-1]],
   ["estimator r", "predicted vs real wins: "+s.r_est_real],
   ["rebuild WAR","OLD "+s.old_war_mean+" · NEW "+s.new_war_mean+" · actual "+s.actual_war_mean],
   ["seeds","100 per team-arm"]];
  document.getElementById('kpis').innerHTML=items.map(i=>`<div class="kpi"><b>${i[1]}</b><span>${i[0]}</span></div>`).join('');
}
function bars(el, rows){
  // rows: [{team, act, old, oldsd, new, newsd}] grouped bars per team
  const W=Math.max(720, rows.length*22), H=340, bw=7;
  const maxW=Math.max(...rows.map(r=>Math.max(r.new+r.newsd,r.old+r.oldsd,r.act_est)))*1.02;
  const y=v=>H-40-((v-30)/(maxW-30))*(H-70);
  let s=`<svg viewBox="0 0 ${rows.length*26+40} ${H}" style="width:100%">`;
  [40,60,80,100,120].forEach(g=>{if(g<maxW)s+=`<line x1="30" x2="${rows.length*26+36}" y1="${y(g)}" y2="${y(g)}" stroke="#2230" class="gl"/><text x="26" y="${y(g)+4}" font-size="10" fill="#8a94a3" text-anchor="end">${g}</text>`});
  rows.forEach((r,i)=>{
    const x=38+i*26;
    const b=(v,sd,c,off)=>`<rect x="${x+off}" y="${Math.min(y(v),y(30))}" width="${bw}" height="${Math.abs(y(v)-y(30))}" fill="${c}"/><line x1="${x+off+bw/2}" x2="${x+off+bw/2}" y1="${y(v+sd)}" y2="${y(Math.max(v-sd,30))}" stroke="${c}" stroke-opacity=".6"/>`;
    s+=b(r.act_est,0,"#7ee08a",0)+b(r.new,r.new_sd,"#5aa7ff",9)+b(r.old,r.old_sd,"#ffb454",18);
    s+=`<text x="${x+12}" y="${H-24}" font-size="9" fill="#8a94a3" text-anchor="middle" transform="rotate(60 ${x+12} ${H-24})">${r.team}</text>`;
  });
  s+="</svg>"; el.innerHTML=s;
}
function chart(year){
  const src=P.panel[year]||P.control[year]||[];
  const rows=src.map(r=>({team:r.team,act_est:r.act_est,new:r.new,newsd:r.new_sd,old:r.old,old_sd:r.old_sd}))
                .sort((a,b)=>a.team.localeCompare(b.team));
  bars(document.getElementById('chart'),rows);
}
function table(year){
  const rows=(P.panel[year]||[]).slice().sort((a,b)=>b.gap-a.gap);
  document.getElementById('tbl').innerHTML=`<table><tr><th>Team</th><th>Payroll $M</th><th>Real W</th><th>Actual est</th><th>NEW-GM</th><th>OLD-GM</th><th>NEW−OLD</th></tr>`+
   rows.map(r=>`<tr><td>${r.team}</td><td>${r.payroll_m}</td><td>${r.real}</td><td>${fmt(r.act_est)}</td><td>${fmt(r.new)} ±${r.new_sd}</td><td>${fmt(r.old)} ±${r.old_sd}</td><td class="${r.gap>=0?'pos':'neg'}">${r.gap>0?'+':''}${r.gap}</td></tr>`).join('')+`</table>`;
}
function ctl(){
  if(!P.control||!Object.keys(P.control).length){document.getElementById('ctl').textContent='control panel not computed yet';return;}
  const yrs=Object.keys(P.control).sort();
  const per=y=>{const rs=P.control[y];return rs.reduce((a,r)=>a+r.gap,0)/rs.length;};
  document.getElementById('ctl-note').textContent=`${yrs[0]}–${yrs[yrs.length-1]}: both arms price the same pre-Moneyball market, so the gap should hover at zero — it averages ${Math.round(yrs.reduce((a,y)=>a+per(y),0)/yrs.length*100)/100} wins, vs the modern panel's ${P.summary.mean_gap}. (Pre-1993 leagues had 26–28 teams.)`;
  const rows=yrs.map(y=>`<tr><td>${y}</td><td>${Math.round(per(y)*100)/100}</td></tr>`).join('');
  document.getElementById('ctl').innerHTML=`<table><tr><th>Season</th><th>mean NEW−OLD wins</th></tr>${rows}</table>`;
}
function ws(){
  const w=P.ws||[];
  if(!w.length){document.getElementById('ws').textContent='pending Phase 7 output';return;}
  document.getElementById('ws').innerHTML=`<table><tr><th>Year</th><th>Champion</th><th>Wins</th><th>Rank by actual bWAR (n teams)</th><th>"Best on paper" team</th></tr>`+
   w.map(r=>`<tr><td>${r.year}</td><td>${r.champ}</td><td>${r.champ_wins}</td><td>${r.champ_rank_actual_war==null?'—':Math.round(r.champ_rank_actual_war)}</td><td>${r.best_on_paper_team}</td></tr>`).join('')+`</table>`+
   `<div class="sub">Top-3 by bWAR among champs: ${w.filter(r=>r.champ_rank_actual_war<=3).length}/${w.length}</div>`;
}
function populate(sel,data){sel.innerHTML=Object.keys(data).sort().reverse().map(y=>`<option>${y}</option>`).join('');}
const P2=P.panel&&Object.keys(P.panel).length?P.panel:(P.control||{});
kpis();
const y1=document.getElementById('ysel'),y2=document.getElementById('ysel2');
populate(y1,P.panel||{});populate(y2,P.panel||{});
y1.onchange=e=>chart(+e.target.value);y2.onchange=e=>table(+e.target.value);
chart(+y1.value);table(+y2.value);ctl();ws();
</script></body></html>
"""


def main():
    fr = load_frames()
    payload = build_payload(fr)
    html = HTML.replace("__PAYLOAD__", json.dumps(payload))
    os.makedirs(SITE, exist_ok=True)
    with open(os.path.join(SITE, "index.html"), "w") as f:
        f.write(html)
    print(f"wrote site/index.html ({len(html)/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
