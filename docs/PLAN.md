# Moneyball Counterfactual — "Old Stats vs. New Stats" Roster Construction Study

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Test whether front offices using only traditional/basic stats (AVG, HR, RBI, W, ERA) would build materially different (and plausibly worse) teams than actual post-Moneyball front offices using modern stats (WAR, wRC+, FIP, etc.) — and showcase the results on a static website.

**Architecture:** Free raw MLB data (Lahman + Retrosheet + FanGraphs/Baseball-Reference for WAR) → a Python "valuation engine" that prices every player-season twice (once with an OLD-ERA stat model, once with the modern one) → a constrained roster-construction simulator that rebuilds each team's roster under each philosophy → head-to-head comparison with real win totals as ground truth. Website is a static single-file HTML report (no server), per user preference.

**Tech Stack:** Python 3.11, pandas, pybaseball (FanGraphs leaderboards via HTML tables — no API key needed), Lahman CSVs, pytsxtr-style Monte Carlo (plain numpy), single-file HTML/JS report with cached JSON. Static first.

---

## Research Design (the scientific part)

### The core idea: "re-draft every team every year, twice"

For a target season (e.g. 2015, or 2010–2024 as a panel):

1. **Ground truth:** each team's real roster, real salary, real wins.
2. **OLD-GM simulation:** rebuild the 26-man roster using ONLY pre-sabermetric stats, with an old-school GM's spending habits.
3. **NEW-GM baseline = the team's actual roster** (real front offices already use modern stats post-2002; that's the whole premise). No need to simulate the "new" side separately — reality *is* the new-stats run.

This gives a clean comparison: **"what the team would look like if run by a 1990s GM" vs. "what the team actually is"** — and the win-difference between simulated-old and actual is the effect size.

### OLD vs NEW stat definitions (the exact parameters)

**OLD-ERA batting model** (what a 1990s GM pays for; weights approximating old salary arbitration/markets):
- AVG (batting average)
- HR (home runs)
- RBI (runs batted in)
- R (runs scored)
- H (hits) — only as a sanity cap
- SB (stolen bases, only if success rate > 70% — the old "don't run into outs" rule)
- Position scarcity: catchers/shortstops get a bonus; DH/DH-only players get a penalty

**Explicitly EXCLUDED from OLD model:** walks (BB), OBP, SLG, OPS, ISO, pitch framing, defense beyond position bonus, park adjustments, age curves, plate-discipline metrics. This is the point — the old model systematically undervalues walks/OBP (the Moneyball insight) and overvalues RBI (a teammate-quality stat, not a skill).

**OLD-ERA pitching model:**
- W (wins), L, W%
- ERA, saves (SV), complete games if you want era flavor
- K/9 as a minor factor (old scouts liked "stuff" but didn't price it)

**NEW (modern) model — the baseline being tested against:**
- fWAR/bWAR (wins above replacement, park- and league-adjusted)
- wRC+ for hitters, FIP/SIERA for pitchers
- These are the stats actual post-2003 front offices use, so "actual team performance" proxies for them.

### How the OLD model prices a player

Instead of inventing weights, **learn them from the pre-Moneyball market (1985–2002)**:

```
salary_oldschool ~ regression on [AVG, HR, RBI, R, SB, position, PA-qualifier]
salary_modern    ~ regression on [WAR, wRC+, FIP, position, PA-qualifier]
```

Fit both on 1985–2002 player-seasons (before front offices adopted OBP). The OLD regression, applied to post-2003 players, literally reproduces "what a 1990s GM would pay this player." This is the scientifically defensible version of "old stats" — it's not my opinion of old stats, it's the actual historical market's revealed preference. Validate: the OLD model should have visibly lower R² than the modern one, and its residuals should show that walks/BB-heavy players were systematically underpaid (this IS Moneyball, rediscovered empirically).

### Roster construction simulation

Each simulated "old-school GM" rebuilds one team for one season:

1. **Budget:** use the team's *actual* opening-day payroll (constraints stay fair; only the philosophy changes).
2. **Universe:** all players who actually played that season (no counterfactual free agents who didn't exist).
3. **Objective:** maximize total OLD-model value (predicted salary surplus = performance − OLD-market price) subject to:
   - exactly 13 position players, 13 pitchers
   - position coverage (2 C, 1 SS, etc. — approximate with min/max per position)
   - per-player price comes from the OLD regression
4. **Output:** the 26-man roster the old GM would have built.
5. **Same procedure with the NEW model** as a second arm (so we have three teams: Actual, Old-GM, New-GM — New-GM should ≈ Actual, which doubles as a sanity check on the whole pipeline).

### How to score simulated teams without simulating 162 games

Simulating full games (Retrosheet event-level) is a v2 luxury. V1 uses the standard sabermetric identity:

```
team_runs_estimate = linear-weights-style conversion of the roster's stat line
```

- For **batters**: use the roster's *actual* (already-played) stats from that season, sum weighted by projected PA. Convert to runs via a fixed formula (e.g. BaseRuns or simple wRC-style: `runs ≈ 0.47·1B + 0.75·2B + 1.03·3B + 1.44·HR + 0.33·BB + …`). These are real, observed stats — no simulation noise.
- For **pitchers**: sum FIP-based runs allowed with IP weighting.
- **Wins = (runs_scored − runs_allowed) via Pythagenpat** (`win% = RS^x / (RS^x + RA^x)`, `x = ((RS+RA)/9)^0.287`), × 162. This is the industry-standard estimator with ~±4-win error bands — good enough for a 5-15 win effect size.
- **Crucial:** apply the *same* estimator to the actual roster too, so estimator bias cancels. All three arms (Actual, Old-GM, New-GM) go through identical machinery.

### Control / placebo checks (keeping some teams the same)

- **New-GM ≈ Actual** is the built-in control: if the "modern" rebuild doesn't approximate the real roster, the pipeline is broken.
- **Backtest on 1985–1995:** run the OLD-GM on pre-Moneyball seasons. There the OLD and NEW models should find similar rosters (the market hadn't diverged yet). If OLD-GM looks "just as good" in 1990 as NEW-GM, but clearly worse in 2015+, the effect is a *market inefficiency that closed*, not a coding artifact.
- **Placebo seasons / random seeds:** rerun the optimizer 100× with random tie-breaks and report mean ± std of win deltas, not a single number.
- **No cherry-picking:** report all 30 teams, all seasons, ranked — including teams where the old GM would have done *fine* (small-market teams were often efficient by necessity; e.g. the A's were already Moneyball).

### Expected findings to test (pre-registering the hypotheses)

- H1: OLD-GM rosters over-invest in high-RBI/HR sluggers and undervalue walk-heavy OBP guys.
- H2: OLD-GM win deltas are negative on average, larger for big-payroll teams (they could afford to be wrong; Moneyball's original edge was a *poor-team* inefficiency).
- H3: Specific "winners" of the old era: players like high-AVG/no-walk contact guys, RBI machines on good teams, closers with big save totals. Specific "losers": OBP-first guys, low-strikeout pitch-to-contact starters, elite defensive catchers/shortstops.
- H4: The effect shrinks toward zero over time as markets got efficient.

---

## Data needed (all free)

| Dataset | Source | What it gives |
|---|---|---|
| Lahman Database | seanlahman.com / GitHub `chadwickbureau` mirrors | Batting, Pitching, Fielding, Salaries 1985–2024, master name/ID table |
| FanGraphs leaderboards | `pybaseball.fangraphs_*` | fWAR, wRC+, FIP, SIERA per player-season (free HTML tables) |
| Baseball-Reference WAR | `pybaseball.bwar_bat` / `bwar_pitch` | bWAR fallback (different flavor of WAR, good cross-check) |
| Team seasons | Lahman `Teams.csv` | Real W-L, RS, RA, payroll per team-season |
| (v2 only) Retrosheet event files | retrosheet.org | Event-level game simulation if we ever go past Pythagenpat |

All reachable without API keys. `pybaseball` pulls FanGraphs via HTML scraping; rate-limit politely, cache to CSV.

---

## Website plan (static, per preference)

Single-page static site, one HTML file + cached JSON, no server:

- **Hero:** "What if every MLB team hired a 1995 GM?" with the headline number (avg win delta across 2015–2024).
- **Big league-wide chart:** per-team OLD-GM vs Actual wins (bar chart), sortable by season.
- **Season explorer:** pick a year → table of 30 teams, each row shows Actual wins vs. Old-GM wins, delta, and the biggest roster changes ("Out: X, Y. In: Z").
- **Player pages (v2):** biggest winners/losers of the old valuation — "Players the old GM overpays / underpays."
- **Methodology tab:** everything in this plan's Research Design section, so the numbers are auditable.
- **Tech:** one `index.html` + a few JSON files the Python pipeline regenerates. Can be dropped on GitHub Pages as-is. Static first; no auto-refresh.

---

## Phased plan

### Phase 0 — Setup & data acquisition (first)
- Create project dir `D:\Hermes\Scituate\moneyball-counterfactual\` (or wherever user prefers), git init.
- `pip install pandas numpy pybaseball` (check requirements.txt for existing deps first).
- Download Lahman CSVs (one zip), cache to `data/lahman/`.
- Pull FanGraphs + BR WAR leaderboards 1985–2024 via pybaseball, cache to `data/war/`.
- **Validation:** row counts sanity-checked against known values (e.g. 30 teams/season post-1998, Mike Trout's 2012–2019 fWAR values spot-check).

### Phase 1 — Valuation engine
- Fit OLD and NEW salary regressions on 1985–2002 (`src/valuation.py`).
- Report R² for both; residual analysis showing walks underpriced by OLD model.
- **Validation:** OLD model must rank, e.g., 2002 Barry Bonds (walk machine) well below his actual value; NEW model must rank him #1.

### Phase 2 — Roster optimizer
- Constrained knapsack-style optimizer (`src/roster_builder.py`): maximize surplus subject to roster/position/budget constraints. Integer greedy + local search is fine (no need for exact MILP in v1; scipy or pulp if needed).
- **Validation:** New-GM rebuilds of 2015 teams must have ≥80% player overlap with actual rosters (otherwise the NEW model/pipeline is mis-specified).

### Phase 3 — Outcome estimation
- Linear-weights runs estimator + Pythagenpat (`src/outcomes.py`).
- **Validation:** apply to *actual* 2015 rosters; predicted wins must correlate with real wins at r ≥ 0.85 across all 30 teams.

### Phase 4 — Experiment runs
- Panel: 2010–2024 (or 2003–2024 if cheap), all 30 teams, 100 optimizer seeds.
- Output: tidy CSV of team-season × {actual, old-gm, new-gm} wins + roster diffs.
- Backtest 1988–1995 as control.

### Phase 5 — Static website
- Python renders `site/index.html` + `site/data/*.json` from the results CSV.
- Deploy: GitHub Pages (free) or just open the file locally.

### Phase 6 — Player-level insights (v2, only if wanted)
- "Biggest Moneyball losers": players whose OLD-model price >> NEW-model value (overpaid by old GMs) and vice versa.
- Count of players whose *careers* would differ (v2: simple playing-time counterfactual).

---

## Files likely to be created

```
moneyball-counterfactual/
├── requirements.txt
├── src/
│   ├── download_data.py
│   ├── valuation.py        # OLD/NEW salary models
│   ├── roster_builder.py   # constrained optimizer
│   ├── outcomes.py         # runs estimator + Pythagenpat
│   ├── run_experiment.py   # panel runner
│   └── build_site.py       # renders static site
├── data/                   # cached CSVs (git-ignored)
├── results/                # tidy CSVs of experiment output
└── site/
    ├── index.html
    └── data/*.json
```

## Tests / validation

- pytest on pure functions: linear weights formula, Pythagenpat, roster constraint checks, valuation-model fit on synthetic data.
- Data sanity checks in `download_data.py` (row counts, spot-check known player values).
- End-to-end checks (Phase 1–3 validations above) — these ARE the science.

## Risks & tradeoffs

- **The premise cuts both ways:** WAR is not "stupid" — it's the current market price. If OLD-GM rosters lose by only 1-2 wins, the honest headline is "modern stats are worth ~2 wins" which is still a real finding.
- **Replacement level & position adjustments:** OLD model has no defense except a position bonus; that's historically accurate but adds noise. Mitigate with the New-GM≈Actual control.
- **FanGraphs scraping rate limits:** cache aggressively, retry politely, fall back to Baseball-Reference WAR.
- **Salary data pre-1985 missing:** that's why the valuation fits on 1985–2002 only.
- **Scope creep:** game-by-game simulation is explicitly v2+; Pythagenpat carries v1.
- **Player name matching** between Lahman/FG/BR is fiddly (accents, Jr.). Use key_player IDs from Lahman as canonical; build a lookup during Phase 0 validation.

## Open questions (need user input, not blocking start)

1. Season range: 2010–2024 (my default) vs. full 2003–2024 panel?
2. Website: GitHub Pages public, or local-only static file for now?
3. Name for the project/site? (matters for repo/GH Pages)
