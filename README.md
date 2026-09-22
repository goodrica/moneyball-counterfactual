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
python src/download_data.py   # pulls Lahman + bWAR, derives payroll, runs data-contract tests
python src/run_experiment.py  # runs the panel
python src/build_site.py      # renders site/index.html + site/data/*.json
```

## Status

✅ Pipeline complete through Phase 6 + the Phase 7 World Series check. Live site: https://goodrica.github.io/moneyball-counterfactual/

- [x] Phase 0: Data acquisition (Lahman 2025 release via CRAN + bWAR; payroll derived; data-contract tests pass)
- [x] Phase 1: Valuation engine (old/new salary regressions)
- [x] Phase 2: Roster optimizer (+ 2015 pilot panel)
- [x] Phase 3: Outcome estimator (+ 2010–2021 validation panel; r=0.91 vs real wins)
- [x] Phase 4: Experiment panel — **league-wide exclusive draft** per seed (the pilot's
      independent-team design let every team draft the same stars); 2010–2024 panel +
      1988–1999 era backtest, 100 seeds, mean±std
- [x] Phase 5: Static site (single-file HTML, dependency-free SVG charts, GitHub Pages)
- [x] Phase 6: Player-level insights (results/player_insights*.csv)
- [x] Phase 7: World Series winners 1990–1999 vs "best on paper" (results/world_series_1990s.csv)

**Headline (honest):** with a shared player pool, the NEW−OLD win gap averages **+0.22
wins** across 2010–24 (t≈+0.6 — statistically zero), and **−0.55** in the 1988–99
control (t≈−2.0 — old prices slightly better in their own era, as expected when
re-drafting from a pool the old market itself priced). Rebuilding teams under old-school
valuation does NOT lose 5–15 wins; the philosophy gap in *roster construction* is near
zero once prices are market-learned. The tercile story (H2, inverted): low-payroll teams
do **worse** under old prices (−7.1 wins) — old-market prices are compressed, so poor
teams can't exploit bargains — while high-payroll teams do marginally **better** with
old prices (+2.8): flat pricing makes stars look affordable. The Moneyball core survives
where it always lived — in *player pricing*: within equal-WAR bins the old model
systematically discounts walk-heavy players (r≈−0.10, negative in 7/8 bins).

## Data sources

- [Lahman Database](https://github.com/cbwinslow/baseballdatabank) (SHA: `a0b6f52`) — batting/pitching counting stats, Teams W-L/RS/RA, Salaries for 1985–2002 training window
- [Baseball-Reference WAR](https://www.baseball-reference.com) via `pybaseball.bwar_bat` / `bwar_pitch` — WAR, salary, position runs through 2024+
- Derived payroll (`data/payroll.csv`) — 30 teams/year, 2003–2024
