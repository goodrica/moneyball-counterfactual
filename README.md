# Moneyball Counterfactual

**What if every MLB team hired a 1995 GM?**

This project tests whether modern "baseball nerd" stats (WAR, wRC+, FIP) actually produce better teams than traditional stats (AVG, HR, RBI, Wins, ERA) — by rebuilding every MLB roster the way a pre-Moneyball GM would have, and comparing the results to what really happened.

## The idea

- Learn what "old stats" actually meant **from the historical market itself**: fit a salary regression on 1985–2002 player seasons (before front offices adopted OBP), using only old-school stats. That regression prices today's players the way a 1990s GM would.
- Do the same with modern stats (WAR/wRC+/FIP) as the "new GM" model.
- A constrained optimizer rebuilds each team's roster under each philosophy, holding the team's **real payroll** constant.
- Score every roster (actual, old-GM, new-GM) with the same runs estimator + Pythagenpat win estimator so estimator bias cancels out.

## Controls (keeping it honest)

- The **new-GM rebuild should approximate each team's actual roster** — if it doesn't, the pipeline is broken.
- A **backtest on 1985–1995** should show old-GM ≈ new-GM (the inefficiency hadn't opened yet).
- 100 optimizer seeds per run; results reported as mean ± std, never a single cherry-picked number.
- All 30 teams, all seasons — no picking favorites.

## The site

Static single-page report (no server, no build step):
- League-wide chart: old-GM wins vs. actual wins per team per season
- Season explorer with roster diffs ("Out: X, Y. In: Z")
- Biggest winners/losers of old-school valuation
- Full methodology for auditability

Deployed via GitHub Pages (this repo).

## Setup

```bash
pip install -r requirements.txt
python src/download_data.py   # pulls Lahman + FanGraphs WAR, caches to data/
python src/run_experiment.py  # runs the panel
python src/build_site.py      # renders site/index.html + site/data/*.json
```

## Status

🚧 Planning complete — see [docs/PLAN.md](docs/PLAN.md) for the full research design.

- [ ] Phase 0: Data acquisition (Lahman, FanGraphs WAR)
- [ ] Phase 1: Valuation engine (old/new salary regressions)
- [ ] Phase 2: Roster optimizer
- [ ] Phase 3: Outcome estimator
- [ ] Phase 4: Experiment panel
- [ ] Phase 5: Static site
- [ ] Phase 6: Player-level insights

## Data sources

- [Lahman Database](https://www.seanlahman.com/baseball-archive-statistics) — batting/pitching/salaries 1985–2024
- [FanGraphs](https://www.fangraphs.com) — fWAR, wRC+, FIP (via `pybaseball`)
- [Baseball-Reference](https://www.baseball-reference.com) — bWAR fallback
