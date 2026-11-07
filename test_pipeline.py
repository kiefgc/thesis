#!/usr/bin/env python3
"""Tests on synthetic data only. Reads no real task table. Run: python test_pipeline.py"""
import numpy as np
from sklearn.metrics import roc_auc_score

import advisor_common as AC

rng = np.random.default_rng(0)

# 1. DeLong AUCs equal scikit-learn's, and identical scores give difference 0 and p = 1.
y = rng.integers(0, 2, 200)
s1 = rng.normal(size=200) + y * 0.8
s2 = rng.normal(size=200) + y * 0.2
a1, a2, dd, p = AC.delong(y, s1, s2)
assert abs(a1 - roc_auc_score(y, s1)) < 1e-9 and abs(a2 - roc_auc_score(y, s2)) < 1e-9
_, _, d0, p0 = AC.delong(y, s1, s1)
assert abs(d0) < 1e-12 and p0 == 1.0
print("DeLong matches scikit-learn:", round(a1, 3), round(a2, 3), "p", round(p, 4))

# 2. A planted signal is found; permuted labels are not.
N = 266
X = rng.normal(size=(N, 7))
ks = (X[:, 0] + 0.5 * rng.normal(size=N) > 0).astype(int)
d = {"ck": np.full(N, 0.1), "co": np.full(N, 0.7), "ks": ks, "os": rng.integers(0, 2, N),
     "repo": np.array(["a"] * 90 + ["b"] * 90 + ["c"] * 86)}
P, TH, _ = AC.repeated_oof(X, ks, d, True, reps=3)
auc_real = roc_auc_score(ks, P.mean(0))
aucs_perm = []
for k in range(5):
    perm = np.random.default_rng(100 + k).permutation(ks)          # labels permuted for training AND for judging
    P2, _, _ = AC.repeated_oof(X, perm, d, True, reps=3)
    aucs_perm.append(roc_auc_score(perm, P2.mean(0)))
auc_perm = float(np.mean(aucs_perm))
print("planted-signal AUC", round(auc_real, 3), "| permuted-label AUC (mean of 5)", round(auc_perm, 3), [round(a, 3) for a in aucs_perm])
assert auc_real > 0.8 and abs(auc_perm - 0.5) < 0.06

# 3. Policy arithmetic on a tiny example: C costs Kimi, plus Opus when Kimi fails.
t = {"ck": np.array([1.0, 1.0, 1.0]), "co": np.array([5.0, 5.0, 5.0]), "ks": np.array([1, 0, 0]), "os": np.array([1, 1, 0])}
pol = AC.policies(t, {})
assert pol["C"][0].sum() == 3 + 10 and pol["C"][1].sum() == 2          # Kimi cost 3, Opus on two failures 10; solved by either: 2
assert pol["H"][0].sum() == 1 + 5 + 5 and pol["A"][0].sum() == 15
print("policy arithmetic OK")

# 4. Bootstrap difference of a policy against itself is exactly zero.
r = AC.bootstrap(pol)
assert r["C"]["d"] == 0 and r["C"]["d_ci"] == (0.0, 0.0)
print("ALL TESTS PASSED")
