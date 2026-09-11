"""
Phase 2 tests: roster optimizer mechanics.
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import roster_builder as rb


def _tiny_pools():
    n = 30
    pos_cycle = ["C","1B","2B","SS","3B","OF","OF","OF","C","1B",
                 "2B","SS","3B","OF","OF","OF","C","1B","2B","SS",
                 "3B","OF","OF","OF","DH","DH","1B","2B","SS","OF"]
    assert len(pos_cycle) == n
    bat = pd.DataFrame({
        "player_ID": [f"b{i}" for i in range(n)],
        "WAR": np.linspace(6, -1, n),
        "price_old": np.linspace(20e6, 0.6e6, n),
        "price_new": np.linspace(30e6, 0.6e6, n),
        "primary_pos": pos_cycle,
        "pos_set": [set() for _ in range(n)],
    })
    pit = pd.DataFrame({
        "player_ID": [f"p{i}" for i in range(n)],
        "WAR": np.linspace(7, -1, n),
        "price_old": np.linspace(25e6, 0.6e6, n),
        "price_new": np.linspace(28e6, 0.6e6, n),
        "is_sp": [i % 3 == 0 for i in range(n)],   # 10 SPs incl. cheap ones
    })
    return bat, pit


def test_full_roster_under_budget():
    bat, pit = _tiny_pools()
    rng = np.random.default_rng(1)
    cb, cp, tw = rb.optimize(bat, pit, budget=200e6, rng=rng, price_col="price_old")
    assert len(cb) == rb.N_BAT and len(cp) == rb.N_PIT, "roster not full"
    spend = bat.iloc[list(cb)]["price_old"].sum() + pit.iloc[list(cp)]["price_old"].sum()
    assert spend <= 200e6 * 1.01, f"over budget: {spend/1e6:.1f}M"
    print(f"PASS: full 26-man roster, ${spend/1e6:.0f}M <= $200M")


def test_coverage_requirements_met():
    bat, pit = _tiny_pools()
    rng = np.random.default_rng(2)
    cb, cp, _ = rb.optimize(bat, pit, budget=80e6, rng=rng, price_col="price_old")
    pos_counts = bat.iloc[list(cb)]["primary_pos"].value_counts().to_dict()
    for pos, n in rb.BAT_COVERAGE.items():
        assert pos_counts.get(pos, 0) >= n, f"coverage failed for {pos}"
    sp_count = pit.iloc[list(cp)]["is_sp"].sum()
    assert sp_count >= rb.MIN_SP, f"only {sp_count} SP"
    print("PASS: position coverage + SP minimum")


def test_bigger_budget_never_worse():
    bat, pit = _tiny_pools()
    wars = []
    for budget in [60e6, 120e6, 240e6]:
        rng = np.random.default_rng(3)
        _, _, tw = rb.optimize(bat, pit, budget=budget, rng=rng, price_col="price_old")
        wars.append(tw)
    assert wars[0] <= wars[1] <= wars[2], f"WAR not monotone in budget: {wars}"
    print(f"PASS: WAR monotone in budget ({[f'{w:.0f}' for w in wars]})")


def test_cheaper_price_never_helps():
    """Same talent pool, higher prices -> optimizer can't do better."""
    bat, pit = _tiny_pools()
    rng1 = np.random.default_rng(4)
    _, _, w_cheap = rb.optimize(bat, pit, budget=100e6, rng=rng1, price_col="price_old")
    rng2 = np.random.default_rng(4)
    _, _, w_pricey = rb.optimize(bat, pit, budget=100e6, rng=rng2, price_col="price_new")
    assert w_cheap >= w_pricey - 1e-6, "expensive model outperformed cheap model on identical talent"
    print(f"PASS: price monotonicity ({w_cheap:.1f} vs {w_pricey:.1f})")


if __name__ == "__main__":
    test_full_roster_under_budget()
    test_coverage_requirements_met()
    test_bigger_budget_never_worse()
    test_cheaper_price_never_helps()
    print("\nAll Phase 2 tests passed.")
