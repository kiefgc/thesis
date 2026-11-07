#!/usr/bin/env python3
"""Shared code for the primary and secondary analyses (protocol v1.0).

Not run directly. Used by analysis_primary.py and analysis_secondary.py.
Requires numpy, scipy-free; scikit-learn for logistic regression. features.py must sit in the same folder.

Choices that the protocol leaves open are marked CLARIFICATION below. They must be written into
protocol section 12 (dated, before the first real run).
"""
import csv, hashlib, json, re, statistics, sys
from math import erf, sqrt
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED, REPS, B = 42, 20, 10000
CS = (0.01, 0.1, 1.0, 10.0)            # protocol section 6
NONINF_PP = 2.0                         # protocol section 9
EXPECTED_SHA = {
    "tasks_pro266.csv": "9D4E96BA3375FA08A5BCD3E6187E1F63411BED459AF9F87FD716469B57D654D5",
    "tasks_label100.csv": "62990EDEB7EFC45A267110B24B902566CE07677D1972790F4BD31E1EE404AD4C",
}
# (feature name in features.py, enters as ln(1+x)?)   protocol section 5, F1..F8
FEATURE_SPEC = [("n_chars", True), ("n_code_blocks", True), ("has_trace_or_error", False), ("n_code_refs", True),
                ("n_headings", True), ("n_urls", True), ("mentions_tests", False), ("n_list_items", True)]
EXPECTED_KEPT = {"n_chars", "n_code_blocks", "has_trace_or_error", "n_code_refs", "n_headings", "mentions_tests", "n_list_items"}
KW = re.compile(r"\b(refactor|redesign|architecture|migrate|migration|security|concurrency|race condition|rewrite)\b", re.I)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def load_table(path):
    """Reads task_id, pre_* and out_* cost/solved columns only. Never reads post_* columns."""
    path = Path(path)
    if path.name in EXPECTED_SHA and sha(path) != EXPECTED_SHA[path.name]:
        sys.exit(f"{path.name}: hash differs from the frozen table. Not running.")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    return {
        "ids": [r["task_id"] for r in rows],
        "repo": np.array([r["pre_repo"] for r in rows]),
        "text": [r["pre_issue_text"] for r in rows],
        "ck": np.array([float(r["out_kimi_cost_usd"]) for r in rows]),
        "co": np.array([float(r["out_opus_cost_usd"]) for r in rows]),
        "ks": np.array([int(r["out_kimi_solved"]) for r in rows]),
        "os": np.array([int(r["out_opus_solved"]) for r in rows]),
    }


def feature_matrix(texts, keep=None):
    """Handcrafted features from the issue text only (features.py). Returns X, kept names, share nonzero."""
    import features as F
    raw = [F.features(t) for t in texts]
    cols, share = {}, {}
    for name, log in FEATURE_SPEC:
        v = np.array([r[name] for r in raw], float)
        share[name] = float((v > 0).mean())
        cols[name] = np.log1p(v) if log else v
    if keep is None:                    # outcome-blind screening rule, protocol section 5: drop if nonzero in < 10%
        keep = [n for n, _ in FEATURE_SPEC if share[n] >= 0.10]
        if set(keep) != EXPECTED_KEPT:
            sys.exit(f"Screening kept {sorted(keep)} but the protocol says {sorted(EXPECTED_KEPT)}. Not running.")
    return np.column_stack([cols[n] for n in keep]), keep, share


def repo_matrix(repo, cats):
    return np.column_stack([(repo == c).astype(float) for c in cats])


# ---------------- model ----------------
def make_model(C, scale):
    steps = [StandardScaler()] if scale else []
    steps.append(LogisticRegression(C=C, max_iter=2000))
    return make_pipeline(*steps)


def fit_predict(Xtr, ytr, Xte, scale):
    """CLARIFICATION 1: the protocol says C is chosen by inner 5-fold cross-validation but not by which metric.
    This code uses mean log loss (the metric the model itself optimises, and the one that matters because the
    probabilities are compared with a cost threshold). Ties go to the smaller C."""
    inner = StratifiedKFold(5, shuffle=True, random_state=SEED)
    best = (np.inf, None)
    for C in CS:
        L = []
        for a, b in inner.split(Xtr, ytr):
            m = make_model(C, scale).fit(Xtr[a], ytr[a])
            L.append(log_loss(ytr[b], m.predict_proba(Xtr[b])[:, 1], labels=[0, 1]))
        if np.mean(L) < best[0]:
            best = (float(np.mean(L)), C)
    m = make_model(best[1], scale).fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1], best[1]


def repeated_oof(X, y, d, scale, reps=REPS):
    """Out-of-fold probabilities and cost-ratio thresholds, repeated stratified 5-fold (protocol section 6)."""
    N = len(y)
    P, TH, chosen = np.zeros((reps, N)), np.zeros((reps, N)), []
    for r in range(reps):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED + r).split(X, y):
            p, C = fit_predict(X[tr], y[tr], X[te], scale)
            P[r, te] = p
            TH[r, te] = d["ck"][tr].sum() / d["co"][tr].sum()   # theta = mean Kimi cost / mean Opus cost, training folds
            chosen.append(C)
    return P, TH, chosen


def loro(X, y, d, scale):
    """Leave-one-repository-out. X=None means the repository-only rule, which falls back to the training base rate."""
    N = len(y)
    p, th = np.zeros(N), np.zeros(N)
    for r in np.unique(d["repo"]):
        te = d["repo"] == r
        tr = ~te
        p[te] = y[tr].mean() if X is None else fit_predict(X[tr], y[tr], X[te], scale)[0]
        th[te] = d["ck"][tr].sum() / d["co"][tr].sum()
    return p, th


def rule_share(y, N, rule, reps=REPS):
    """Average over repeated folds of a rule's 'send straight to frontier' decision. rule(train_idx, test_idx) -> bool array."""
    share = np.zeros(N)
    for r in range(reps):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED + r).split(np.zeros(N), y):
            share[te] += np.asarray(rule(tr, te), float) / reps
    return share


# ---------------- policies and statistics ----------------
def policies(d, shares):
    ck, co, ks, os_ = d["ck"], d["co"], d["ks"], d["os"]
    Ccost = ck + (1 - ks) * co
    Csolved = ((ks + os_) > 0).astype(float)
    pol = {"A": (co.copy(), os_.astype(float)), "B": (ck.copy(), ks.astype(float)), "C": (Ccost, Csolved),
           "H": (np.where(ks == 1, ck, co), Csolved)}
    for k, s in shares.items():
        pol[k] = (s * co + (1 - s) * Ccost, s * os_ + (1 - s) * Csolved)
    return pol


def _ci(v):
    v = np.sort(v[~np.isnan(v)])
    return float(v[int(0.025 * len(v))]), float(v[int(0.975 * len(v)) - 1])


def bootstrap(pols, ref="C"):
    N = len(next(iter(pols.values()))[0])
    idx = np.random.default_rng(SEED).integers(0, N, size=(B, N))
    cps, sr = {}, {}
    for k, (c, s) in pols.items():
        ss = s[idx].sum(1)
        cps[k] = c[idx].sum(1) / np.where(ss > 0, ss, np.nan)
        sr[k] = ss / N
    out = {}
    for k, (c, s) in pols.items():
        out[k] = dict(spend=float(c.sum()), solved=float(s.sum()), cps=float(c.sum() / s.sum()), cps_ci=_ci(cps[k]),
                      d=float(c.sum() / s.sum() - pols[ref][0].sum() / pols[ref][1].sum()), d_ci=_ci(cps[k] - cps[ref]),
                      dsolve_pp=float((s.sum() - pols[ref][1].sum()) / N * 100), dsolve_ci=_ci((sr[k] - sr[ref]) * 100))
    return out


def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    out = np.empty(N)
    out[J] = T + 1
    return out


def delong(y, s1, s2):
    """Fast DeLong test for two correlated AUCs on the same tasks. Returns auc1, auc2, difference, two-sided p."""
    y = np.asarray(y)
    order = np.argsort(-y, kind="stable")
    y = y[order]
    m = int(y.sum())
    n = len(y) - m
    sc = np.vstack([np.asarray(s1)[order], np.asarray(s2)[order]])
    tx = np.array([_midrank(r[:m]) for r in sc])
    ty = np.array([_midrank(r[m:]) for r in sc])
    tz = np.array([_midrank(r) for r in sc])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (aucs[0] - aucs[1]) / sqrt(var) if var > 0 else 0.0
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return float(aucs[0]), float(aucs[1]), float(aucs[0] - aucs[1]), float(p)


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def verdict(row):
    """Protocol section 9: supported only if the cost-per-solved interval vs C lies wholly below zero AND the
    solve-rate difference is non-inferior. CLARIFICATION 2: non-inferior = lower 95% bound of the solve-rate
    difference is above -2 percentage points."""
    cheaper = row["d_ci"][1] < 0
    noninf = row["dsolve_ci"][0] > -NONINF_PP
    return cheaper, noninf, ("supported" if cheaper and noninf else "not supported")


# ---------------- the analysis ----------------
def md_table(pols_res, names):
    L = ["| Policy | Spend | Solved | Cost per solved | 95% interval | Difference vs C | 95% interval | Solve-rate diff vs C (pp) | 95% interval |",
         "|---|---|---|---|---|---|---|---|---|"]
    for k, nm in names.items():
        o = pols_res[k]
        L.append(f"| {nm} | ${o['spend']:.2f} | {o['solved']:.1f} | ${o['cps']:.3f} | {o['cps_ci'][0]:.3f} to {o['cps_ci'][1]:.3f} | "
                 f"{o['d']:+.3f} | {o['d_ci'][0]:+.3f} to {o['d_ci'][1]:+.3f} | {o['dsolve_pp']:+.2f} | {o['dsolve_ci'][0]:+.2f} to {o['dsolve_ci'][1]:+.2f} |")
    return L


NAMES = {"A": "A Frontier for all", "B": "B Cheaper for all", "C": "C Cheaper first", "D": "D Advisor",
         "E": "E Repository-only", "F": "F Existing-practice rule", "G": "G Scrouting text head", "H": "H Oracle (ceiling)"}


def run_analysis(d, d100, Xp, Xl, scale, tag, repo_root, y_train=None, reps=REPS, secondary_of=None):
    """Returns a list of markdown lines. y_train overrides the training label (used only by the shuffled-label check)."""
    y = d["ks"] if y_train is None else y_train
    N = len(y)
    L = []
    w = L.append
    w(f"# {tag}\n")
    # --- Advisor and repository-only predictors ---
    P, TH, chosen = repeated_oof(Xp, y, d, scale, reps)
    cats = sorted(set(d["repo"]))
    Xr = repo_matrix(d["repo"], cats)
    PE, THE, _ = repeated_oof(Xr, y, d, False, reps)
    shares = {"D": (P < TH).mean(0), "E": (PE < THE).mean(0)}
    # --- F: existing-practice rule, G: Scrouting text-only head (released probabilities) ---
    lens = np.array([len(t) for t in d["text"]])
    kw = np.array([bool(KW.search(t)) for t in d["text"]])
    shares["F"] = rule_share(y, N, lambda tr, te: (lens[te] > np.median(lens[tr])) | kw[te], reps)
    pv = json.load(open(Path(repo_root) / "data/receipts/routing_scenarios.json", encoding="utf-8"))["probability_vectors"]
    pt = np.array([pv[t]["kimi-k2-5"]["p_text"] for t in d["ids"]])
    shares["G"] = rule_share(y, N, lambda tr, te: pt[te] < d["ck"][tr].sum() / d["co"][tr].sum(), reps)
    res = bootstrap(policies(d, shares))
    w(f"{N} tasks, {reps} repetitions of stratified 5-fold, seed {SEED}, {B} bootstrap resamples. f = 0 (no false-solve correction). My arithmetic.\n")
    w("## 1. Policies (primary analysis)\n")
    L += md_table(res, NAMES)
    ch, ni, v = verdict(res["D"])
    w(f"\n**Primary claim, Advisor against cheaper-first:** cost per solved task difference {res['D']['d']:+.3f} "
      f"(95% interval {res['D']['d_ci'][0]:+.3f} to {res['D']['d_ci'][1]:+.3f}); interval wholly below zero: {ch}. "
      f"Solve-rate difference {res['D']['dsolve_pp']:+.2f} pp (lower 95% bound {res['D']['dsolve_ci'][0]:+.2f}); non-inferior (bound above -{NONINF_PP:.0f} pp): {ni}. "
      f"Verdict under the protocol rule: **{v}**.\n")
    w(f"Share of tasks sent straight to the frontier model: Advisor {shares['D'].mean():.1%}, repository-only {shares['E'].mean():.1%}, "
      f"rule F {shares['F'].mean():.1%}, Scrouting head {shares['G'].mean():.1%}. Chosen C values (counts): "
      + ", ".join(f"{c}: {chosen.count(c)}" for c in CS) + ".\n")
    # --- threshold sensitivity ---
    w("## 2. Threshold sensitivity (Advisor)\n")
    w("| θ | Spend | Solved | Cost per solved | Difference vs C | 95% interval | Solve-rate diff (pp) | lower bound |")
    w("|---|---|---|---|---|---|---|---|")
    for th in (0.30, 0.50, 0.70):
        sh = (P < th).mean(0)
        o = bootstrap(policies(d, {"D": sh}))["D"]
        w(f"| {th} | ${o['spend']:.2f} | {o['solved']:.1f} | ${o['cps']:.3f} | {o['d']:+.3f} | {o['d_ci'][0]:+.3f} to {o['d_ci'][1]:+.3f} | {o['dsolve_pp']:+.2f} | {o['dsolve_ci'][0]:+.2f} |")
    w("\nThe primary θ is the training-fold cost ratio (about 0.14).\n")
    # --- prediction ---
    w("## 3. Prediction of Kimi success (out-of-fold, scores averaged over repetitions)\n")
    sD, sE = P.mean(0), PE.mean(0)
    yy = d["ks"]
    a1, a2, dd, p1 = delong(yy, sD, sE)
    b1, b2, dd2, p2 = delong(yy, sD, pt)
    w("| Comparison | AUC (first) | AUC (second) | Difference | DeLong p |")
    w("|---|---|---|---|---|")
    w(f"| Advisor vs repository-only | {a1:.3f} | {a2:.3f} | {dd:+.3f} | {p1:.3f} |")
    w(f"| Advisor vs Scrouting text head | {b1:.3f} | {b2:.3f} | {dd2:+.3f} | {p2:.3f} |")
    brier = lambda PP: float(np.mean([brier_score_loss(yy, PP[r]) for r in range(PP.shape[0])]))
    w(f"\nBrier score (lower is better): Advisor {brier(P):.4f}, repository-only {brier(PE):.4f}, base rate {yy.mean() * (1 - yy.mean()):.4f}.")
    w("With 266 tasks only a substantial gain in AUC is detectable (protocol section 9); a non-significant difference is not evidence of no effect.\n")
    # --- leave-one-repository-out ---
    w("## 4. Leave-one-repository-out (secondary scheme)\n")
    pl, thl = loro(Xp, y, d, scale)
    pe, the = loro(None, y, d, False)
    resl = bootstrap(policies(d, {"D": (pl < thl).astype(float), "E": (pe < the).astype(float)}))
    L += md_table(resl, {k: NAMES[k] for k in ("A", "B", "C", "D", "E", "H")})
    w(f"\nAUC pooled over held-out repositories: Advisor {auc(yy, pl):.3f}. Repository-only is undefined for an unseen repository and falls back to the training base rate.\n")
    # --- Label 100 (separate check) ---
    w("## 5. Label 100 (a separate check, never pooled; text form differs, so the shift is large)\n")
    pL, _ = fit_predict(Xp, y, Xl, scale)
    theta = d["ck"].sum() / d["co"].sum()
    resL = bootstrap(policies(d100, {"D": (pL < theta).astype(float)}))
    L += md_table(resL, {k: NAMES[k] for k in ("A", "B", "C", "D", "H")})
    w(f"\nAUC against Kimi success on the 100 tasks: {auc(d100['ks'], pL):.3f}.")
    return L
