"""
Phase 3 tests: outcome estimator.
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import outcomes as oc


def test_batter_runs_known_values():
    """Hand-computed linear weights: 10 1B, 5 2B, 3 HR, 10 BB, 60 AB, 25 H."""
    df = pd.DataFrame({"1B": [10.0], "2B": [5.0], "3B": [0.0], "HR": [3.0],
                        "BB": [10.0], "HBP": [1.0], "SB": [2.0], "CS": [1.0],
                        "AB": [60.0], "H": [18.0], "SF": [1.0], "SH": [0.0]})
    events = 10*.47 + 5*.75 + 3*1.4 + 10*.33 + 1*.38 + 2*.2 - 1*.4
    outs = (60-18) + 1
    expected = events + outs * (-0.08)
    got = float(oc.batter_runs(df).iloc[0])
    assert abs(got - expected) < 1e-9, f"{got} != {expected}"
    print(f"PASS: batter_runs hand-check ({got:.3f})")


def test_fip_reference_value():
    """League-average line: 180 IP, 25 HR, 55 BB, 160 SO → FIP ≈ (13*25+3*55-2*160)/180+3.15 = 4.00."""
    df = pd.DataFrame({"SO": [160.0], "BB": [55.0], "HR": [25.0], "IPouts": [540.0]})
    runs = float(oc.pitcher_fip_runs(df).iloc[0])
    expected_fip = (13*25 + 3*55 - 2*160)/180 + 3.15
    expected_runs = expected_fip * 180/9
    assert abs(runs - expected_runs) < 1e-9
    print(f"PASS: FIP runs ({runs:.1f} ≈ FIP {expected_fip:.2f} × 20 IP)")


def test_pythagenpat_equal_runs_half():
    w = oc.pythagenpat([700], [700])
    assert abs(w[0] - 81.0) < 0.01, "equal RS/RA must give .500 → 81 wins"
    print("PASS: pythagenpat .500 identity")


def test_more_runs_more_wins():
    w1 = oc.pythagenpat([750], [700])
    w2 = oc.pythagenpat([700], [700])
    assert w1[0] > w2[0]
    print("PASS: run edge converts to wins")


def test_team_wins_full_pipeline():
    """A league-average 26-man stat line should land near 81 wins."""
    rng = np.random.default_rng(5)
    bat = pd.DataFrame({
        "1B": rng.uniform(40, 90, 13), "2B": rng.uniform(15, 35, 13),
        "3B": rng.uniform(0, 6, 13), "HR": rng.uniform(5, 30, 13),
        "BB": rng.uniform(20, 70, 13), "HBP": rng.uniform(1, 8, 13),
        "SB": rng.uniform(0, 15, 13), "CS": rng.uniform(0, 5, 13),
        "AB": rng.uniform(250, 550, 13), "H": [0]*13,
        "SF": rng.uniform(2, 8, 13), "SH": rng.uniform(0, 3, 13),
    })
    bat["H"] = bat["1B"] + bat["2B"] + bat["3B"] + bat["HR"]
    pit = pd.DataFrame({
        "SO": rng.uniform(40, 160, 13), "BB": rng.uniform(15, 50, 13),
        "HR": rng.uniform(5, 25, 13), "IPouts": rng.uniform(150, 550, 13),
    })
    pit["IPouts"] = pit["IPouts"] * (4374.0 / pit["IPouts"].sum())
    w = oc.team_wins(bat, pit)
    assert 40 <= w <= 130, f"implausible win total {w:.0f}"
    print(f"PASS: pipeline sanity ({w:.0f} wins from random roster)")


if __name__ == "__main__":
    test_batter_runs_known_values()
    test_fip_reference_value()
    test_pythagenpat_equal_runs_half()
    test_more_runs_more_wins()
    test_team_wins_full_pipeline()
    print("\nAll Phase 3 tests passed.")
