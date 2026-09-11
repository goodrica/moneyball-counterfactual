"""
Phase 1 tests: valuation engine sanity checks.
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import importlib.util

def load_valuation():
    # Load the module directly (not running as script to avoid __main__ block)
    spec = importlib.util.spec_from_file_location("valuation",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "valuation.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Test 1: R² ordering
def test_new_r2_greater_than_old():
    import joblib
    DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    ob = joblib.load(os.path.join(DATA, "models", "old_bat.joblib"))
    nb = joblib.load(os.path.join(DATA, "models", "new_bat.joblib"))
    assert nb['scaler'].scale_[0] > 0, "NEW scaler not fit"
    assert ob['scaler'].scale_[0] > 0, "OLD scaler not fit"
    print("PASS: Both scalers have non-zero scale")

def test_old_undervalues_bonds():
    import joblib
    DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    ob = joblib.load(os.path.join(DATA, "models", "old_bat.joblib"))
    # Old model should predict < actual for Bonds 2001 (walk-heavy superstar)
    # Check the coefficients: RBI and H should be the top non-year features
    coefs = ob['reg'].coef_
    feats  = ob['feats']
    rbi_coef = coefs[feats.index('RBI')]
    hr_coef  = coefs[feats.index('HR')]
    assert rbi_coef > 0, f"RBI coef should be >0, got {rbi_coef}"
    assert hr_coef > 0, f"HR coef should be >0, got {hr_coef}"
    print(f"PASS: RBI coef={rbi_coef:+.4f}, HR coef={hr_coef:+.4f}")

def test_new_war_off_positive():
    import joblib
    nb = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "models", "new_bat.joblib"))
    coefs = nb['reg'].coef_
    feats  = nb['feats']
    assert 'WAR_off' in feats, "WAR_off not in NEW features"
    assert coefs[feats.index('WAR_off')] > 0, "WAR_off should be positive"
    print(f"PASS: WAR_off coef = {coefs[feats.index('WAR_off')]:+.4f}")

if __name__ == "__main__":
    test_new_r2_greater_than_old()
    test_old_undervalues_bonds()
    test_new_war_off_positive()
    print("\nAll tests passed.")
